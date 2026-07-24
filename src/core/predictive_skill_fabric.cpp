#include "rlf/core/predictive_skill_fabric.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rlf::core {
namespace {

constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;
constexpr double tie_tolerance = 1.0e-12;

void hash_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (unsigned int byte = 0U; byte < 8U; ++byte) {
        hash ^= (value >> (byte * 8U)) & 0xFFULL;
        hash *= fnv_prime;
    }
}

[[nodiscard]] double normalized_angular_error(
    const PhaseVector& left,
    const PhaseVector& right
) {
    return left.mean_angular_error(right) / std::numbers::pi_v<double>;
}

[[nodiscard]] double clamp_finite(
    const double value,
    const double lower,
    const double upper
) noexcept {
    if (!std::isfinite(value)) {
        return lower;
    }
    return std::clamp(value, lower, upper);
}

struct BridgeNode final {
    PhaseVector state{std::vector<float>{0.0F}};
    std::vector<std::uint64_t> route;
    std::uint64_t hash{};
};

struct BridgeLayer final {
    std::vector<BridgeNode> nodes;
    std::unordered_map<std::uint64_t, std::size_t> by_hash;
};

[[nodiscard]] bool routes_equal(
    const std::span<const std::uint64_t> left,
    const std::span<const std::uint64_t> right
) noexcept {
    return left.size() == right.size() &&
        std::equal(left.begin(), left.end(), right.begin());
}

}  // namespace

std::string_view to_string(const Rlf2StopReason reason) noexcept {
    switch (reason) {
    case Rlf2StopReason::successful_halt:
        return "successful_halt";
    case Rlf2StopReason::learned_halt:
        return "learned_halt";
    case Rlf2StopReason::planner_failure:
        return "planner_failure";
    case Rlf2StopReason::cycle_limit:
        return "cycle_limit";
    case Rlf2StopReason::loop_detected:
        return "loop_detected";
    case Rlf2StopReason::abstained:
        return "abstained";
    case Rlf2StopReason::no_candidate:
        return "no_candidate";
    }
    return "unknown";
}

PredictiveSkillFabric::PredictiveSkillFabric(
    PredictiveSkillConfig config,
    const std::uint64_t seed
)
    : config_(std::move(config)),
      seed_(seed),
      rng_(seed) {
    if (config_.dimension == 0U || config_.maximum_cycles == 0U ||
        config_.maximum_route_depth == 0U ||
        config_.planner_node_budget == 0U || config_.maximum_skills == 0U ||
        config_.maximum_subgoal_prototypes == 0U ||
        config_.maximum_skill_length == 0U ||
        config_.minimum_skill_support == 0U ||
        config_.nearest_prototypes == 0U ||
        !std::isfinite(config_.goal_similarity_threshold) ||
        config_.goal_similarity_threshold <= 0.0 ||
        config_.goal_similarity_threshold > 1.0 ||
        !std::isfinite(config_.prototype_merge_distance) ||
        config_.prototype_merge_distance < 0.0 ||
        !std::isfinite(config_.prototype_distance_scale) ||
        config_.prototype_distance_scale <= 0.0 ||
        !std::isfinite(config_.action_temperature) ||
        config_.action_temperature <= 0.0 ||
        !std::isfinite(config_.abstention_uncertainty_threshold) ||
        config_.abstention_uncertainty_threshold < 0.0 ||
        config_.abstention_uncertainty_threshold > 1.0) {
        throw std::invalid_argument("invalid predictive skill configuration");
    }
}

const PredictiveSkillConfig& PredictiveSkillFabric::config() const noexcept {
    return config_;
}

std::uint64_t PredictiveSkillFabric::seed() const noexcept {
    return seed_;
}

std::uint64_t PredictiveSkillFabric::training_step() const noexcept {
    return training_step_;
}

std::span<const PredictiveOperator>
PredictiveSkillFabric::operators() const noexcept {
    return operators_;
}

std::span<const CausalSkill> PredictiveSkillFabric::skills() const noexcept {
    return skills_;
}

std::span<const SubgoalPrototype>
PredictiveSkillFabric::prototypes() const noexcept {
    return prototypes_;
}

const Rlf2TrainingStats&
PredictiveSkillFabric::training_stats() const noexcept {
    return training_stats_;
}

std::uint64_t PredictiveSkillFabric::register_operator(
    std::string name,
    TransformationOperator transformation,
    const double cost
) {
    if (name.empty() || transformation.dimension() != config_.dimension ||
        !std::isfinite(cost) || cost <= 0.0) {
        throw std::invalid_argument("invalid predictive operator registration");
    }
    if (std::any_of(
            operators_.begin(),
            operators_.end(),
            [&name](const PredictiveOperator& value) {
                return value.name == name;
            })) {
        throw std::invalid_argument("predictive operator names must be unique");
    }
    if (skills_.size() >= config_.maximum_skills) {
        throw std::runtime_error("predictive skill capacity exhausted");
    }
    const std::uint64_t operator_id = next_operator_id_++;
    TransformationOperator inverse = transformation.inverse();
    operators_.push_back({
        .id = operator_id,
        .name = std::move(name),
        .forward = transformation,
        .inverse = inverse,
        .cost = cost,
    });
    skills_.push_back({
        .id = next_skill_id_++,
        .name = operators_.back().name,
        .primitive_route = {operator_id},
        .forward = std::move(transformation),
        .inverse = std::move(inverse),
        .primitive_length = 1U,
        .support = 1ULL,
        .success_count = 0ULL,
        .failure_count = 0ULL,
        .utility = 0.0,
        .mean_causal_advantage = 0.0,
        .accepted = true,
    });
    return operator_id;
}

const PredictiveOperator& PredictiveSkillFabric::operator_by_id(
    const std::uint64_t operator_id
) const {
    const auto iterator = std::find_if(
        operators_.begin(),
        operators_.end(),
        [operator_id](const PredictiveOperator& value) {
            return value.id == operator_id;
        }
    );
    if (iterator == operators_.end()) {
        throw std::out_of_range("unknown predictive operator ID");
    }
    return *iterator;
}

const CausalSkill& PredictiveSkillFabric::skill_by_id(
    const std::uint64_t skill_id
) const {
    const auto iterator = std::find_if(
        skills_.begin(),
        skills_.end(),
        [skill_id](const CausalSkill& value) {
            return value.id == skill_id;
        }
    );
    if (iterator == skills_.end()) {
        throw std::out_of_range("unknown predictive skill ID");
    }
    return *iterator;
}

std::uint64_t PredictiveSkillFabric::route_hash(
    const std::span<const std::uint64_t> route
) noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, static_cast<std::uint64_t>(route.size()));
    for (const std::uint64_t value : route) {
        hash_u64(hash, value);
    }
    return hash;
}

double PredictiveSkillFabric::profile_distance(
    const std::span<const float> left,
    const std::span<const float> right
) {
    if (left.size() != right.size() || left.empty()) {
        throw std::invalid_argument("profile distance dimension mismatch");
    }
    double total = 0.0;
    for (std::size_t index = 0U; index < left.size(); ++index) {
        const double difference =
            static_cast<double>(left[index]) - static_cast<double>(right[index]);
        total += difference * difference;
    }
    return total / static_cast<double>(left.size());
}

bool PredictiveSkillFabric::is_goal(
    const PhaseVector& state,
    const PhaseVector& goal
) const {
    return state.similarity(goal) >= config_.goal_similarity_threshold;
}

std::vector<float> PredictiveSkillFabric::response_profile(
    const PhaseVector& current,
    const PhaseVector& goal
) const {
    if (current.size() != config_.dimension ||
        goal.size() != config_.dimension) {
        throw std::invalid_argument("predictive response profile mismatch");
    }
    std::vector<float> profile;
    const std::size_t operator_count = operators_.size();
    profile.reserve(
        2U + 8U + (2U * operator_count) +
        (operator_count * operator_count) + (3U * operator_count)
    );
    profile.push_back(static_cast<float>(current.similarity(goal)));
    profile.push_back(static_cast<float>(normalized_angular_error(current, goal)));

    double cosine[4U]{0.0, 0.0, 0.0, 0.0};
    double sine[4U]{0.0, 0.0, 0.0, 0.0};
    for (std::size_t coordinate = 0U;
         coordinate < config_.dimension;
         ++coordinate) {
        const double difference = std::remainder(
            static_cast<double>(goal[coordinate]) -
                static_cast<double>(current[coordinate]),
            2.0 * std::numbers::pi_v<double>
        );
        for (std::size_t harmonic = 0U; harmonic < 4U; ++harmonic) {
            const double value = static_cast<double>(harmonic + 1U) * difference;
            cosine[harmonic] += std::cos(value);
            sine[harmonic] += std::sin(value);
        }
    }
    for (std::size_t harmonic = 0U; harmonic < 4U; ++harmonic) {
        profile.push_back(static_cast<float>(
            cosine[harmonic] / static_cast<double>(config_.dimension)
        ));
        profile.push_back(static_cast<float>(
            sine[harmonic] / static_cast<double>(config_.dimension)
        ));
    }

    std::vector<PhaseVector> one_step;
    one_step.reserve(operator_count);
    for (const PredictiveOperator& operator_value : operators_) {
        PhaseVector successor = operator_value.forward.apply(current);
        profile.push_back(static_cast<float>(successor.similarity(goal)));
        profile.push_back(static_cast<float>(
            normalized_angular_error(successor, goal)
        ));
        one_step.push_back(std::move(successor));
    }

    for (const PhaseVector& first_successor : one_step) {
        double maximum = 0.0;
        double total = 0.0;
        double squared_total = 0.0;
        for (const PredictiveOperator& second : operators_) {
            const PhaseVector second_successor =
                second.forward.apply(first_successor);
            const double value = second_successor.similarity(goal);
            profile.push_back(static_cast<float>(value));
            maximum = std::max(maximum, value);
            total += value;
            squared_total += value * value;
        }
        const double denominator = operator_count == 0U
            ? 1.0
            : static_cast<double>(operator_count);
        const double mean = total / denominator;
        const double variance = std::max(
            0.0,
            (squared_total / denominator) - (mean * mean)
        );
        profile.push_back(static_cast<float>(maximum));
        profile.push_back(static_cast<float>(mean));
        profile.push_back(static_cast<float>(std::sqrt(variance)));
    }
    return profile;
}

PredictiveSkillFabric::PrototypeEstimate
PredictiveSkillFabric::estimate_for_skill(
    const std::span<const float> profile,
    const std::uint64_t skill_id
) const {
    struct Neighbor final {
        std::size_t index{};
        double distance{};
    };
    std::vector<Neighbor> neighbors;
    for (std::size_t index = 0U; index < prototypes_.size(); ++index) {
        if (prototypes_[index].skill_id != skill_id) {
            continue;
        }
        neighbors.push_back({
            .index = index,
            .distance = profile_distance(
                profile,
                prototypes_[index].response_profile
            ),
        });
    }
    if (neighbors.empty()) {
        return {};
    }
    std::stable_sort(
        neighbors.begin(),
        neighbors.end(),
        [this](const Neighbor& left, const Neighbor& right) {
            if (std::abs(left.distance - right.distance) > tie_tolerance) {
                return left.distance < right.distance;
            }
            return prototypes_[left.index].id < prototypes_[right.index].id;
        }
    );
    const std::size_t count = std::min(
        config_.nearest_prototypes,
        neighbors.size()
    );
    double weight_total = 0.0;
    double value_total = 0.0;
    double remaining_total = 0.0;
    double advantage_total = 0.0;
    double confidence_total = 0.0;
    for (std::size_t rank = 0U; rank < count; ++rank) {
        const Neighbor& neighbor = neighbors[rank];
        const SubgoalPrototype& prototype = prototypes_[neighbor.index];
        const double weight =
            static_cast<double>(prototype.support) * prototype.confidence /
            (1.0 + config_.prototype_distance_scale * neighbor.distance);
        weight_total += weight;
        value_total += weight * prototype.terminal_value;
        remaining_total += weight * prototype.remaining_steps;
        advantage_total += weight * prototype.causal_advantage;
        confidence_total += weight * prototype.confidence;
    }
    if (weight_total <= 0.0) {
        return {};
    }
    return {
        .value = value_total / weight_total,
        .remaining_steps = remaining_total / weight_total,
        .causal_advantage = advantage_total / weight_total,
        .distance = neighbors.front().distance,
        .confidence = confidence_total / weight_total,
        .found = true,
    };
}

PredictiveSkillFabric::PrototypeEstimate
PredictiveSkillFabric::estimate_state_value(
    const std::span<const float> profile
) const {
    PrototypeEstimate best;
    std::uint64_t best_skill_id = std::numeric_limits<std::uint64_t>::max();
    for (const CausalSkill& skill : skills_) {
        if (!skill.accepted) {
            continue;
        }
        PrototypeEstimate candidate = estimate_for_skill(profile, skill.id);
        if (!candidate.found) {
            continue;
        }
        const double candidate_score = candidate.confidence * candidate.value /
            (1.0 + candidate.remaining_steps) - candidate.distance;
        const double best_score = best.found
            ? best.confidence * best.value / (1.0 + best.remaining_steps) -
                best.distance
            : -std::numeric_limits<double>::infinity();
        if (!best.found || candidate_score > best_score + tie_tolerance ||
            (std::abs(candidate_score - best_score) <= tie_tolerance &&
             skill.id < best_skill_id)) {
            best = candidate;
            best_skill_id = skill.id;
        }
    }
    return best;
}

void PredictiveSkillFabric::update_prototype(
    const std::span<const float> profile,
    const std::uint64_t skill_id,
    const double remaining_steps,
    const double terminal_value,
    const double causal_advantage
) {
    std::size_t best_index = 0U;
    double best_distance = std::numeric_limits<double>::infinity();
    bool found = false;
    for (std::size_t index = 0U; index < prototypes_.size(); ++index) {
        if (prototypes_[index].skill_id != skill_id) {
            continue;
        }
        const double distance = profile_distance(
            profile,
            prototypes_[index].response_profile
        );
        if (!found || distance < best_distance - tie_tolerance ||
            (std::abs(distance - best_distance) <= tie_tolerance &&
             prototypes_[index].id < prototypes_[best_index].id)) {
            best_index = index;
            best_distance = distance;
            found = true;
        }
    }
    if (found && best_distance <= config_.prototype_merge_distance) {
        SubgoalPrototype& prototype = prototypes_[best_index];
        const double old_support = static_cast<double>(prototype.support);
        const double new_support = old_support + 1.0;
        for (std::size_t index = 0U; index < profile.size(); ++index) {
            prototype.response_profile[index] = static_cast<float>(
                (static_cast<double>(prototype.response_profile[index]) *
                     old_support + static_cast<double>(profile[index])) /
                new_support
            );
        }
        prototype.remaining_steps =
            (prototype.remaining_steps * old_support + remaining_steps) /
            new_support;
        prototype.terminal_value =
            (prototype.terminal_value * old_support + terminal_value) /
            new_support;
        prototype.causal_advantage =
            (prototype.causal_advantage * old_support + causal_advantage) /
            new_support;
        prototype.confidence = clamp_finite(
            prototype.confidence + (0.08 * (terminal_value - 0.5)),
            0.05,
            1.0
        );
        ++prototype.support;
        prototype.last_used_step = training_step_;
        ++training_stats_.prototypes_merged;
        return;
    }

    if (prototypes_.size() >= config_.maximum_subgoal_prototypes) {
        const auto worst = std::min_element(
            prototypes_.begin(),
            prototypes_.end(),
            [](const SubgoalPrototype& left, const SubgoalPrototype& right) {
                const double left_score =
                    left.confidence * static_cast<double>(left.support);
                const double right_score =
                    right.confidence * static_cast<double>(right.support);
                if (std::abs(left_score - right_score) > tie_tolerance) {
                    return left_score < right_score;
                }
                return left.id < right.id;
            }
        );
        prototypes_.erase(worst);
    }
    prototypes_.push_back({
        .id = next_prototype_id_++,
        .response_profile = std::vector<float>(profile.begin(), profile.end()),
        .skill_id = skill_id,
        .remaining_steps = remaining_steps,
        .terminal_value = terminal_value,
        .causal_advantage = causal_advantage,
        .confidence = terminal_value > 0.0 ? 0.65 : 0.35,
        .support = 1ULL,
        .creation_step = training_step_,
        .last_used_step = training_step_,
    });
    ++training_stats_.prototypes_created;
}

TransformationOperator PredictiveSkillFabric::compose_route(
    const std::span<const std::uint64_t> primitive_route
) const {
    if (primitive_route.empty()) {
        return TransformationOperator::identity(config_.dimension);
    }
    TransformationOperator combined =
        operator_by_id(primitive_route.front()).forward;
    for (std::size_t index = 1U; index < primitive_route.size(); ++index) {
        combined = combined.then(operator_by_id(primitive_route[index]).forward);
    }
    return combined;
}

std::vector<std::uint64_t> PredictiveSkillFabric::ordered_operator_ids(
    const PhaseVector& state,
    const PhaseVector& goal,
    const bool learned_ordering
) const {
    std::vector<std::uint64_t> ids;
    ids.reserve(operators_.size());
    for (const PredictiveOperator& value : operators_) {
        ids.push_back(value.id);
    }
    if (!learned_ordering || prototypes_.empty()) {
        return ids;
    }
    const std::vector<float> profile = response_profile(state, goal);
    std::stable_sort(
        ids.begin(),
        ids.end(),
        [this, &profile](const std::uint64_t left,
                         const std::uint64_t right) {
            const auto left_skill = std::find_if(
                skills_.begin(),
                skills_.end(),
                [left](const CausalSkill& value) {
                    return value.primitive_route.size() == 1U &&
                        value.primitive_route.front() == left;
                }
            );
            const auto right_skill = std::find_if(
                skills_.begin(),
                skills_.end(),
                [right](const CausalSkill& value) {
                    return value.primitive_route.size() == 1U &&
                        value.primitive_route.front() == right;
                }
            );
            const PrototypeEstimate left_estimate =
                left_skill == skills_.end()
                    ? PrototypeEstimate{}
                    : estimate_for_skill(profile, left_skill->id);
            const PrototypeEstimate right_estimate =
                right_skill == skills_.end()
                    ? PrototypeEstimate{}
                    : estimate_for_skill(profile, right_skill->id);
            const double left_score = left_estimate.found
                ? left_estimate.value + left_estimate.causal_advantage -
                    left_estimate.distance
                : -1.0;
            const double right_score = right_estimate.found
                ? right_estimate.value + right_estimate.causal_advantage -
                    right_estimate.distance
                : -1.0;
            if (std::abs(left_score - right_score) > tie_tolerance) {
                return left_score > right_score;
            }
            return left < right;
        }
    );
    return ids;
}

std::optional<std::vector<std::uint64_t>>
PredictiveSkillFabric::plan_primitive_bridge(
    const PhaseVector& start,
    const PhaseVector& goal,
    std::size_t maximum_primitive_depth,
    std::size_t* const explored_nodes,
    std::size_t* const forward_nodes,
    std::size_t* const backward_nodes,
    const bool use_learned_ordering
) const {
    if (start.size() != config_.dimension || goal.size() != config_.dimension) {
        throw std::invalid_argument("predictive bridge dimension mismatch");
    }
    if (maximum_primitive_depth == 0U) {
        maximum_primitive_depth = config_.maximum_route_depth;
    }
    maximum_primitive_depth = std::min(
        maximum_primitive_depth,
        config_.maximum_route_depth
    );
    std::size_t total_nodes = 0U;
    std::size_t forward_total = 0U;
    std::size_t backward_total = 0U;
    const auto set_counts = [&]() {
        if (explored_nodes != nullptr) {
            *explored_nodes = total_nodes;
        }
        if (forward_nodes != nullptr) {
            *forward_nodes = forward_total;
        }
        if (backward_nodes != nullptr) {
            *backward_nodes = backward_total;
        }
    };
    if (is_goal(start, goal)) {
        total_nodes = 1U;
        forward_total = 1U;
        set_counts();
        return std::vector<std::uint64_t>{};
    }
    if (operators_.empty()) {
        set_counts();
        return std::nullopt;
    }

    const std::size_t forward_depth = (maximum_primitive_depth + 1U) / 2U;
    const std::size_t backward_depth = maximum_primitive_depth / 2U;
    std::vector<BridgeLayer> forward_layers(forward_depth + 1U);
    std::vector<BridgeLayer> backward_layers(backward_depth + 1U);
    const std::uint64_t start_hash = LatentRouter::phase_state_hash(start);
    const std::uint64_t goal_hash = LatentRouter::phase_state_hash(goal);
    forward_layers[0U].nodes.push_back({start, {}, start_hash});
    forward_layers[0U].by_hash.emplace(start_hash, 0U);
    backward_layers[0U].nodes.push_back({goal, {}, goal_hash});
    backward_layers[0U].by_hash.emplace(goal_hash, 0U);
    total_nodes = 2U;
    forward_total = 1U;
    backward_total = 1U;

    for (std::size_t depth = 1U; depth <= backward_depth; ++depth) {
        BridgeLayer& layer = backward_layers[depth];
        const BridgeLayer& previous = backward_layers[depth - 1U];
        for (const BridgeNode& node : previous.nodes) {
            const std::vector<std::uint64_t> ids = ordered_operator_ids(
                node.state,
                start,
                false
            );
            for (const std::uint64_t operator_id : ids) {
                const PredictiveOperator& operator_value =
                    operator_by_id(operator_id);
                PhaseVector predecessor = operator_value.inverse.apply(node.state);
                const std::uint64_t hash =
                    LatentRouter::phase_state_hash(predecessor);
                ++total_nodes;
                ++backward_total;
                if (total_nodes > config_.planner_node_budget) {
                    set_counts();
                    return std::nullopt;
                }
                if (layer.by_hash.contains(hash)) {
                    continue;
                }
                std::vector<std::uint64_t> route;
                route.reserve(node.route.size() + 1U);
                route.push_back(operator_id);
                route.insert(route.end(), node.route.begin(), node.route.end());
                const std::size_t index = layer.nodes.size();
                layer.nodes.push_back({
                    .state = std::move(predecessor),
                    .route = std::move(route),
                    .hash = hash,
                });
                layer.by_hash.emplace(hash, index);
            }
        }
    }

    const auto find_match = [&](const std::size_t forward_layer_index,
                                const std::size_t backward_layer_index)
        -> std::optional<std::vector<std::uint64_t>> {
        const BridgeLayer& left = forward_layers[forward_layer_index];
        const BridgeLayer& right = backward_layers[backward_layer_index];
        for (const BridgeNode& left_node : left.nodes) {
            const auto iterator = right.by_hash.find(left_node.hash);
            if (iterator == right.by_hash.end()) {
                continue;
            }
            const BridgeNode& right_node = right.nodes[iterator->second];
            if (left_node.state.similarity(right_node.state) < 0.99999) {
                continue;
            }
            std::vector<std::uint64_t> route = left_node.route;
            route.insert(
                route.end(),
                right_node.route.begin(),
                right_node.route.end()
            );
            return route;
        }
        return std::nullopt;
    };

    for (std::size_t total_depth = 0U;
         total_depth <= maximum_primitive_depth;
         ++total_depth) {
        const std::size_t minimum_forward = total_depth > backward_depth
            ? total_depth - backward_depth
            : 0U;
        const std::size_t maximum_forward = std::min(
            forward_depth,
            total_depth
        );
        for (std::size_t left_depth = minimum_forward;
             left_depth <= maximum_forward;
             ++left_depth) {
            const std::size_t right_depth = total_depth - left_depth;
            if (right_depth > backward_depth) {
                continue;
            }
            if (left_depth > 0U &&
                forward_layers[left_depth].nodes.empty()) {
                continue;
            }
            if (const auto route = find_match(left_depth, right_depth);
                route.has_value()) {
                set_counts();
                return route;
            }
        }

        const std::size_t next_forward_depth = total_depth + 1U;
        if (next_forward_depth > forward_depth ||
            !forward_layers[next_forward_depth].nodes.empty()) {
            continue;
        }
        BridgeLayer& layer = forward_layers[next_forward_depth];
        const BridgeLayer& previous = forward_layers[next_forward_depth - 1U];
        for (const BridgeNode& node : previous.nodes) {
            const std::vector<std::uint64_t> ids = ordered_operator_ids(
                node.state,
                goal,
                use_learned_ordering
            );
            for (const std::uint64_t operator_id : ids) {
                PhaseVector successor =
                    operator_by_id(operator_id).forward.apply(node.state);
                const std::uint64_t hash =
                    LatentRouter::phase_state_hash(successor);
                ++total_nodes;
                ++forward_total;
                if (total_nodes > config_.planner_node_budget) {
                    set_counts();
                    return std::nullopt;
                }
                if (layer.by_hash.contains(hash)) {
                    continue;
                }
                std::vector<std::uint64_t> route = node.route;
                route.push_back(operator_id);
                const std::size_t index = layer.nodes.size();
                layer.nodes.push_back({
                    .state = std::move(successor),
                    .route = std::move(route),
                    .hash = hash,
                });
                layer.by_hash.emplace(hash, index);
            }
        }
    }

    for (std::size_t total_depth = 0U;
         total_depth <= maximum_primitive_depth;
         ++total_depth) {
        const std::size_t minimum_forward = total_depth > backward_depth
            ? total_depth - backward_depth
            : 0U;
        const std::size_t maximum_forward = std::min(
            forward_depth,
            total_depth
        );
        for (std::size_t left_depth = minimum_forward;
             left_depth <= maximum_forward;
             ++left_depth) {
            const std::size_t right_depth = total_depth - left_depth;
            if (right_depth > backward_depth) {
                continue;
            }
            if (const auto route = find_match(left_depth, right_depth);
                route.has_value()) {
                set_counts();
                return route;
            }
        }
    }
    set_counts();
    return std::nullopt;
}

double PredictiveSkillFabric::intervention_advantage(
    const PhaseVector& state,
    const PhaseVector& goal,
    const std::uint64_t chosen_operator,
    const std::size_t remaining_steps
) {
    if (!config_.enable_intervention_credit || operators_.size() <= 1U ||
        remaining_steps == 0U || remaining_steps > 4U) {
        return 0.5;
    }
    std::size_t alternatives = 0U;
    std::size_t reachable_alternatives = 0U;
    for (const PredictiveOperator& candidate : operators_) {
        if (candidate.id == chosen_operator) {
            continue;
        }
        ++alternatives;
        ++training_stats_.intervention_tests;
        const PhaseVector successor = candidate.forward.apply(state);
        std::size_t nodes = 0U;
        const auto route = plan_primitive_bridge(
            successor,
            goal,
            remaining_steps - 1U,
            &nodes,
            nullptr,
            nullptr,
            false
        );
        if (route.has_value()) {
            ++reachable_alternatives;
            ++training_stats_.intervention_alternative_successes;
        }
    }
    if (alternatives == 0U) {
        return 1.0;
    }
    return 1.0 - static_cast<double>(reachable_alternatives) /
        static_cast<double>(alternatives);
}

void PredictiveSkillFabric::observe_successful_route(
    const PhaseVector& start,
    const PhaseVector& goal,
    const std::span<const std::uint64_t> primitive_route
) {
    if (start.size() != config_.dimension || goal.size() != config_.dimension ||
        primitive_route.empty() ||
        primitive_route.size() > config_.maximum_route_depth) {
        throw std::invalid_argument("invalid predictive training route");
    }
    PhaseVector verification = start;
    for (const std::uint64_t operator_id : primitive_route) {
        verification = operator_by_id(operator_id).forward.apply(verification);
    }
    if (!is_goal(verification, goal)) {
        throw std::invalid_argument("predictive training route misses its goal");
    }

    ++training_step_;
    ++training_stats_.observed_routes;
    PhaseVector state = start;
    for (std::size_t step = 0U; step < primitive_route.size(); ++step) {
        const std::uint64_t operator_id = primitive_route[step];
        const auto skill_iterator = std::find_if(
            skills_.begin(),
            skills_.end(),
            [operator_id](const CausalSkill& value) {
                return value.primitive_route.size() == 1U &&
                    value.primitive_route.front() == operator_id;
            }
        );
        if (skill_iterator == skills_.end()) {
            throw std::logic_error("primitive skill is missing");
        }
        const std::size_t remaining = primitive_route.size() - step;
        const std::vector<float> profile = response_profile(state, goal);
        const double advantage = intervention_advantage(
            state,
            goal,
            operator_id,
            remaining
        );
        update_prototype(
            profile,
            skill_iterator->id,
            static_cast<double>(remaining),
            1.0,
            advantage
        );
        CausalSkill& primitive_skill = *std::find_if(
            skills_.begin(),
            skills_.end(),
            [id = skill_iterator->id](const CausalSkill& value) {
                return value.id == id;
            }
        );
        const double old_support = static_cast<double>(primitive_skill.support);
        primitive_skill.mean_causal_advantage =
            (primitive_skill.mean_causal_advantage * old_support + advantage) /
            (old_support + 1.0);
        primitive_skill.utility = clamp_finite(
            primitive_skill.utility + 0.05 * (advantage - 0.5),
            -1.0,
            1.0
        );
        ++primitive_skill.support;
        ++primitive_skill.success_count;
        ++training_stats_.observed_transitions;
        state = operator_by_id(operator_id).forward.apply(state);
    }

    if (config_.enable_skill_consolidation) {
        for (std::size_t length = 2U;
             length <= std::min(
                 config_.maximum_skill_length,
                 primitive_route.size()
             );
             ++length) {
            for (std::size_t begin = 0U;
                 begin + length <= primitive_route.size();
                 ++begin) {
                const std::span<const std::uint64_t> fragment =
                    primitive_route.subspan(begin, length);
                const std::uint64_t hash = route_hash(fragment);
                ++fragment_counts_[hash];
                fragment_routes_[hash] = std::vector<std::uint64_t>(
                    fragment.begin(),
                    fragment.end()
                );
            }
        }
        static_cast<void>(consolidate_skills());
    }
    ++training_stats_.routes_segmented;
}

std::size_t PredictiveSkillFabric::consolidate_skills() {
    if (!config_.enable_skill_consolidation) {
        return 0U;
    }
    std::vector<std::uint64_t> hashes;
    hashes.reserve(fragment_counts_.size());
    for (const auto& [hash, count] : fragment_counts_) {
        if (count >= config_.minimum_skill_support) {
            hashes.push_back(hash);
        }
    }
    std::sort(hashes.begin(), hashes.end());
    std::size_t created = 0U;
    for (const std::uint64_t hash : hashes) {
        if (skills_.size() >= config_.maximum_skills) {
            break;
        }
        const std::vector<std::uint64_t>& route = fragment_routes_.at(hash);
        const bool exists = std::any_of(
            skills_.begin(),
            skills_.end(),
            [&route](const CausalSkill& value) {
                return routes_equal(value.primitive_route, route);
            }
        );
        if (exists) {
            continue;
        }
        ++training_stats_.skills_proposed;
        TransformationOperator forward = compose_route(route);
        TransformationOperator inverse = forward.inverse();
        bool valid = true;
        for (std::size_t sample = 0U; sample < 4U; ++sample) {
            const PhaseVector input = PhaseVector::random(config_.dimension, rng_);
            const PhaseVector reconstructed =
                inverse.apply(forward.apply(input));
            if (reconstructed.similarity(input) < 0.99999) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            continue;
        }
        const std::size_t support = fragment_counts_.at(hash);
        skills_.push_back({
            .id = next_skill_id_++,
            .name = "causal_skill_" + std::to_string(hash),
            .primitive_route = route,
            .forward = std::move(forward),
            .inverse = std::move(inverse),
            .primitive_length = route.size(),
            .support = static_cast<std::uint64_t>(support),
            .success_count = static_cast<std::uint64_t>(support),
            .failure_count = 0ULL,
            .utility = std::min(1.0, 0.1 * static_cast<double>(support)),
            .mean_causal_advantage = 0.5,
            .accepted = true,
        });
        ++created;
        ++training_stats_.skills_accepted;
    }
    return created;
}

std::vector<std::uint64_t> PredictiveSkillFabric::segment_route(
    const std::span<const std::uint64_t> primitive_route
) const {
    const std::size_t length = primitive_route.size();
    const std::size_t unreachable = std::numeric_limits<std::size_t>::max() / 4U;
    std::vector<std::size_t> best(length + 1U, unreachable);
    std::vector<std::uint64_t> chosen(length, 0ULL);
    best[length] = 0U;
    for (std::size_t reverse = 0U; reverse < length; ++reverse) {
        const std::size_t position = length - reverse - 1U;
        for (const CausalSkill& skill : skills_) {
            if (!skill.accepted || skill.primitive_route.empty() ||
                position + skill.primitive_route.size() > length) {
                continue;
            }
            const std::span<const std::uint64_t> fragment =
                primitive_route.subspan(position, skill.primitive_route.size());
            if (!routes_equal(fragment, skill.primitive_route)) {
                continue;
            }
            const std::size_t tail = position + skill.primitive_route.size();
            if (best[tail] == unreachable) {
                continue;
            }
            const std::size_t candidate = 1U + best[tail];
            if (candidate < best[position] ||
                (candidate == best[position] &&
                 (chosen[position] == 0ULL || skill.id < chosen[position]))) {
                best[position] = candidate;
                chosen[position] = skill.id;
            }
        }
    }
    if (best[0U] == unreachable) {
        throw std::logic_error("primitive skills must make routes segmentable");
    }
    std::vector<std::uint64_t> result;
    std::size_t position = 0U;
    while (position < length) {
        const std::uint64_t skill_id = chosen[position];
        if (skill_id == 0ULL) {
            throw std::logic_error("invalid route segmentation state");
        }
        result.push_back(skill_id);
        position += skill_by_id(skill_id).primitive_route.size();
    }
    return result;
}

std::vector<Rlf2ActionCandidate> PredictiveSkillFabric::score_actions(
    const PhaseVector& current,
    const PhaseVector& goal,
    const std::span<const std::uint64_t> recent_skills
) const {
    const double before_similarity = current.similarity(goal);
    const std::vector<float> profile = response_profile(current, goal);
    std::vector<Rlf2ActionCandidate> candidates;
    candidates.reserve(skills_.size());
    for (const CausalSkill& skill : skills_) {
        if (!skill.accepted) {
            continue;
        }
        const PhaseVector successor = skill.forward.apply(current);
        const double after_similarity = successor.similarity(goal);
        const double progress = after_similarity - before_similarity;
        const PrototypeEstimate action_estimate =
            estimate_for_skill(profile, skill.id);
        const std::vector<float> successor_profile =
            response_profile(successor, goal);
        const PrototypeEstimate successor_estimate =
            estimate_state_value(successor_profile);
        const bool repeated = std::find(
            recent_skills.begin(),
            recent_skills.end(),
            skill.id
        ) != recent_skills.end();
        const double learned_value = action_estimate.found
            ? action_estimate.confidence * action_estimate.value /
                (1.0 + action_estimate.remaining_steps)
            : 0.0;
        const double successor_value = successor_estimate.found
            ? successor_estimate.confidence * successor_estimate.value /
                (1.0 + successor_estimate.remaining_steps)
            : 0.0;
        const double causal = action_estimate.found
            ? action_estimate.causal_advantage
            : skill.mean_causal_advantage;
        const double cost = static_cast<double>(skill.primitive_length);
        const double score =
            config_.learned_value_weight * learned_value +
            config_.successor_value_weight * successor_value +
            config_.direct_progress_weight * progress +
            config_.causal_advantage_weight * causal +
            0.20 * skill.utility -
            config_.skill_cost_weight * cost -
            (repeated ? config_.repetition_penalty : 0.0) -
            (action_estimate.found ? action_estimate.distance : 0.15);
        candidates.push_back({
            .skill_id = skill.id,
            .skill_name = skill.name,
            .primitive_length = skill.primitive_length,
            .direct_progress = progress,
            .learned_value = learned_value,
            .successor_value = successor_value,
            .causal_advantage = causal,
            .prototype_distance = action_estimate.found
                ? action_estimate.distance
                : 1.0,
            .score = score,
            .normalized_weight = 0.0,
        });
    }
    std::stable_sort(
        candidates.begin(),
        candidates.end(),
        [](const Rlf2ActionCandidate& left,
           const Rlf2ActionCandidate& right) {
            if (std::abs(left.score - right.score) > tie_tolerance) {
                return left.score > right.score;
            }
            return left.skill_id < right.skill_id;
        }
    );
    return candidates;
}

double PredictiveSkillFabric::candidate_uncertainty(
    std::vector<Rlf2ActionCandidate>& candidates
) const {
    if (candidates.empty()) {
        return 1.0;
    }
    const double maximum = candidates.front().score;
    double total = 0.0;
    for (Rlf2ActionCandidate& candidate : candidates) {
        candidate.normalized_weight = std::exp(
            (candidate.score - maximum) / config_.action_temperature
        );
        total += candidate.normalized_weight;
    }
    if (total <= 0.0 || !std::isfinite(total)) {
        return 1.0;
    }
    double entropy = 0.0;
    for (Rlf2ActionCandidate& candidate : candidates) {
        candidate.normalized_weight /= total;
        if (candidate.normalized_weight > 0.0) {
            entropy -= candidate.normalized_weight *
                std::log(candidate.normalized_weight);
        }
    }
    const double maximum_entropy = candidates.size() <= 1U
        ? 1.0
        : std::log(static_cast<double>(candidates.size()));
    return clamp_finite(entropy / maximum_entropy, 0.0, 1.0);
}

Rlf2ExecutionResult PredictiveSkillFabric::execute_autonomous(
    const PhaseVector& start,
    const PhaseVector& goal
) {
    if (start.size() != config_.dimension || goal.size() != config_.dimension) {
        throw std::invalid_argument("predictive execution dimension mismatch");
    }
    Rlf2ExecutionResult result;
    result.final_state = start;
    std::unordered_set<std::uint64_t> visited;
    visited.insert(LatentRouter::phase_state_hash(start));
    double uncertainty_total = 0.0;
    std::vector<std::uint64_t> recent_skills;

    for (std::size_t cycle = 0U; cycle < config_.maximum_cycles; ++cycle) {
        if (is_goal(result.final_state, goal)) {
            result.success = true;
            result.stop_reason = Rlf2StopReason::successful_halt;
            break;
        }
        std::vector<Rlf2ActionCandidate> candidates = score_actions(
            result.final_state,
            goal,
            recent_skills
        );
        if (candidates.empty()) {
            result.stop_reason = Rlf2StopReason::no_candidate;
            break;
        }
        const double uncertainty = candidate_uncertainty(candidates);
        uncertainty_total += uncertainty;
        const Rlf2ActionCandidate& selected = candidates.front();
        if (uncertainty >= config_.abstention_uncertainty_threshold &&
            selected.score < config_.abstention_value_threshold) {
            result.abstained = true;
            result.stop_reason = Rlf2StopReason::abstained;
            break;
        }
        const CausalSkill& skill = skill_by_id(selected.skill_id);
        const double before = result.final_state.similarity(goal);
        PhaseVector successor = skill.forward.apply(result.final_state);
        const double after = successor.similarity(goal);
        const PrototypeEstimate estimate = estimate_state_value(
            response_profile(successor, goal)
        );
        result.trace.push_back({
            .cycle = cycle,
            .state_hash = LatentRouter::phase_state_hash(result.final_state),
            .goal_hash = LatentRouter::phase_state_hash(goal),
            .selected_skill_id = skill.id,
            .selected_skill_name = skill.name,
            .goal_similarity_before = before,
            .goal_similarity_after = after,
            .uncertainty = uncertainty,
            .predicted_remaining_steps = estimate.found
                ? estimate.remaining_steps
                : -1.0,
            .bridge_subgoal = false,
            .candidates = std::move(candidates),
        });
        result.skill_route.push_back(skill.id);
        result.primitive_route.insert(
            result.primitive_route.end(),
            skill.primitive_route.begin(),
            skill.primitive_route.end()
        );
        result.primitive_steps += skill.primitive_length;
        result.final_state = std::move(successor);
        recent_skills.push_back(skill.id);
        if (recent_skills.size() > 3U) {
            recent_skills.erase(recent_skills.begin());
        }
        const std::uint64_t hash =
            LatentRouter::phase_state_hash(result.final_state);
        if (!visited.insert(hash).second) {
            result.stop_reason = Rlf2StopReason::loop_detected;
            break;
        }
        result.cycles = cycle + 1U;
    }
    if (is_goal(result.final_state, goal)) {
        result.success = true;
        result.stop_reason = Rlf2StopReason::successful_halt;
    } else if (result.stop_reason == Rlf2StopReason::cycle_limit) {
        result.cycles = config_.maximum_cycles;
    }
    result.final_goal_similarity = result.final_state.similarity(goal);
    result.mean_uncertainty = result.trace.empty()
        ? 1.0
        : uncertainty_total / static_cast<double>(result.trace.size());
    return result;
}

Rlf2ExecutionResult PredictiveSkillFabric::execute_subgoal_bridge(
    const PhaseVector& start,
    const PhaseVector& goal,
    std::size_t maximum_primitive_depth,
    const bool use_learned_ordering
) {
    if (maximum_primitive_depth == 0U) {
        maximum_primitive_depth = config_.maximum_route_depth;
    }
    Rlf2ExecutionResult result;
    result.final_state = start;
    std::size_t explored = 0U;
    std::size_t forward = 0U;
    std::size_t backward = 0U;
    const auto route = plan_primitive_bridge(
        start,
        goal,
        maximum_primitive_depth,
        &explored,
        &forward,
        &backward,
        use_learned_ordering
    );
    result.planner_nodes = explored;
    result.forward_nodes = forward;
    result.backward_nodes = backward;
    result.subgoals_considered = backward;
    if (!route.has_value()) {
        result.stop_reason = Rlf2StopReason::planner_failure;
        result.final_goal_similarity = start.similarity(goal);
        return result;
    }
    result.primitive_route = *route;
    result.primitive_steps = route->size();
    result.skill_route = segment_route(*route);
    double uncertainty_total = 0.0;
    for (std::size_t cycle = 0U; cycle < result.skill_route.size(); ++cycle) {
        const CausalSkill& skill = skill_by_id(result.skill_route[cycle]);
        const double before = result.final_state.similarity(goal);
        const std::vector<Rlf2ActionCandidate> scored = score_actions(
            result.final_state,
            goal,
            std::span<const std::uint64_t>{}
        );
        std::vector<Rlf2ActionCandidate> candidates = scored;
        const double uncertainty = candidate_uncertainty(candidates);
        uncertainty_total += uncertainty;
        result.final_state = skill.forward.apply(result.final_state);
        const double after = result.final_state.similarity(goal);
        result.trace.push_back({
            .cycle = cycle,
            .state_hash = LatentRouter::phase_state_hash(result.final_state),
            .goal_hash = LatentRouter::phase_state_hash(goal),
            .selected_skill_id = skill.id,
            .selected_skill_name = skill.name,
            .goal_similarity_before = before,
            .goal_similarity_after = after,
            .uncertainty = uncertainty,
            .predicted_remaining_steps = static_cast<double>(
                result.skill_route.size() - cycle - 1U
            ),
            .bridge_subgoal = true,
            .candidates = std::move(candidates),
        });
    }
    result.cycles = result.skill_route.size();
    result.success = is_goal(result.final_state, goal);
    result.stop_reason = result.success
        ? Rlf2StopReason::successful_halt
        : Rlf2StopReason::planner_failure;
    result.final_goal_similarity = result.final_state.similarity(goal);
    result.mean_uncertainty = result.trace.empty()
        ? 0.0
        : uncertainty_total / static_cast<double>(result.trace.size());
    return result;
}

PredictiveSkillSnapshot PredictiveSkillFabric::snapshot() const {
    return {
        .config = config_,
        .seed = seed_,
        .training_step = training_step_,
        .next_operator_id = next_operator_id_,
        .next_skill_id = next_skill_id_,
        .next_prototype_id = next_prototype_id_,
        .operators = operators_,
        .skills = skills_,
        .prototypes = prototypes_,
        .training_stats = training_stats_,
    };
}

PredictiveSkillFabric PredictiveSkillFabric::from_snapshot(
    PredictiveSkillSnapshot snapshot
) {
    PredictiveSkillFabric fabric(snapshot.config, snapshot.seed);
    fabric.training_step_ = snapshot.training_step;
    fabric.next_operator_id_ = snapshot.next_operator_id;
    fabric.next_skill_id_ = snapshot.next_skill_id;
    fabric.next_prototype_id_ = snapshot.next_prototype_id;
    fabric.operators_ = std::move(snapshot.operators);
    fabric.skills_ = std::move(snapshot.skills);
    fabric.prototypes_ = std::move(snapshot.prototypes);
    fabric.training_stats_ = snapshot.training_stats;
    return fabric;
}

}  // namespace rlf::core
