#include "rlf/core/sparse_world_model.hpp"

#include "rlf/core/deterministic_rng.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <queue>
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
constexpr double probability_floor = 1.0e-12;

void hash_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (unsigned int byte = 0U; byte < 8U; ++byte) {
        hash ^= (value >> (byte * 8U)) & 0xFFULL;
        hash *= fnv_prime;
    }
}

[[nodiscard]] std::uint64_t combine_key(
    const std::uint64_t first,
    const std::uint64_t second,
    const std::uint64_t third = 0ULL
) noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, first);
    hash_u64(hash, second);
    hash_u64(hash, third);
    return hash;
}

[[nodiscard]] double normalized_error(
    const PhaseVector& left,
    const PhaseVector& right
) {
    return left.mean_angular_error(right) / std::numbers::pi_v<double>;
}

[[nodiscard]] bool finite_probability(const double value) noexcept {
    return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

struct PlannerNode final {
    std::uint64_t state_id{};
    std::uint64_t context_id{};
    double cost{};
    double probability{1.0};
    double heuristic{};
    std::size_t depth{};
    std::size_t parent{std::numeric_limits<std::size_t>::max()};
    std::uint64_t action_id{};
    std::uint64_t transition_next_state{};
    std::uint64_t transition_next_context{};
    double outcome_probability{1.0};
};

struct QueueEntry final {
    double priority{};
    std::size_t node_index{};
    std::uint64_t stable_order{};
};

struct QueueCompare final {
    [[nodiscard]] bool operator()(
        const QueueEntry& left,
        const QueueEntry& right
    ) const noexcept {
        if (left.priority != right.priority) {
            return left.priority > right.priority;
        }
        return left.stable_order > right.stable_order;
    }
};

}  // namespace

std::string_view to_string(const Rlf3PlanStopReason reason) noexcept {
    switch (reason) {
    case Rlf3PlanStopReason::goal_reached:
        return "goal_reached";
    case Rlf3PlanStopReason::no_state_match:
        return "no_state_match";
    case Rlf3PlanStopReason::no_goal_match:
        return "no_goal_match";
    case Rlf3PlanStopReason::no_transition:
        return "no_transition";
    case Rlf3PlanStopReason::node_budget:
        return "node_budget";
    case Rlf3PlanStopReason::depth_limit:
        return "depth_limit";
    case Rlf3PlanStopReason::exhausted:
        return "exhausted";
    }
    return "unknown";
}

SparseWorldModel::SparseWorldModel(
    SparseWorldModelConfig config,
    const std::uint64_t seed
) : config_(std::move(config)), seed_(seed) {
    if (config_.dimension == 0U || config_.maximum_states == 0U ||
        config_.maximum_contexts == 0U ||
        config_.maximum_transitions == 0U ||
        config_.maximum_outcomes_per_transition == 0U ||
        config_.maximum_subgoals == 0U || config_.hash_dimensions == 0U ||
        config_.hash_dimensions > config_.dimension ||
        config_.phase_bins < 2U || config_.phase_bins > 256U ||
        config_.maximum_bucket_candidates == 0U ||
        config_.nearest_subgoals == 0U || config_.planner_node_budget == 0U ||
        config_.maximum_plan_depth == 0U ||
        !std::isfinite(config_.state_merge_distance) ||
        config_.state_merge_distance <= 0.0 ||
        config_.state_merge_distance > std::numbers::pi_v<double> ||
        !std::isfinite(config_.context_merge_distance) ||
        config_.context_merge_distance <= 0.0 ||
        config_.context_merge_distance > std::numbers::pi_v<double> ||
        !finite_probability(config_.minimum_outcome_probability) ||
        !std::isfinite(config_.risk_penalty) || config_.risk_penalty < 0.0 ||
        !std::isfinite(config_.uncertainty_penalty) ||
        config_.uncertainty_penalty < 0.0 ||
        !std::isfinite(config_.heuristic_scale) ||
        config_.heuristic_scale < 0.0 ||
        config_.environment_layers < 5U || config_.environment_lanes < 3U ||
        !finite_probability(config_.environment_stochastic_probability) ||
        config_.environment_stochastic_probability <= 0.5 ||
        config_.environment_stochastic_probability >= 1.0 ||
        !std::isfinite(config_.environment_observation_noise) ||
        config_.environment_observation_noise < 0.0) {
        throw std::invalid_argument("invalid sparse world model configuration");
    }

    DeterministicRng rng(seed_ ^ 0x524C4633574F524CULL);
    std::unordered_set<std::size_t> selected;
    while (hash_coordinates_.size() < config_.hash_dimensions) {
        const std::size_t coordinate = rng.uniform_index(config_.dimension);
        if (selected.insert(coordinate).second) {
            hash_coordinates_.push_back(coordinate);
        }
    }
    std::sort(hash_coordinates_.begin(), hash_coordinates_.end());
}

const SparseWorldModelConfig& SparseWorldModel::config() const noexcept {
    return config_;
}

std::uint64_t SparseWorldModel::seed() const noexcept { return seed_; }
std::uint64_t SparseWorldModel::training_step() const noexcept {
    return training_step_;
}
std::span<const WorldAction> SparseWorldModel::actions() const noexcept {
    return actions_;
}
std::span<const WorldStatePrototype> SparseWorldModel::states() const noexcept {
    return states_;
}
std::span<const WorldContextPrototype> SparseWorldModel::contexts() const noexcept {
    return contexts_;
}
std::span<const WorldTransition> SparseWorldModel::transitions() const noexcept {
    return transitions_;
}
std::span<const SparseSubgoal> SparseWorldModel::subgoals() const noexcept {
    return subgoals_;
}
const SparseWorldModelStats& SparseWorldModel::stats() const noexcept {
    return stats_;
}

std::uint64_t SparseWorldModel::register_action(
    std::string name,
    const double cost
) {
    if (name.empty() || !std::isfinite(cost) || cost <= 0.0) {
        throw std::invalid_argument("invalid world action");
    }
    const auto duplicate = std::find_if(
        actions_.begin(), actions_.end(),
        [&name](const WorldAction& action) { return action.name == name; }
    );
    if (duplicate != actions_.end()) {
        throw std::invalid_argument("duplicate world action name");
    }
    const std::uint64_t id = next_action_id_++;
    actions_.push_back({id, std::move(name), cost});
    return id;
}

const WorldAction& SparseWorldModel::action_by_id(
    const std::uint64_t action_id
) const {
    const auto iterator = std::find_if(
        actions_.begin(), actions_.end(),
        [action_id](const WorldAction& action) { return action.id == action_id; }
    );
    if (iterator == actions_.end()) {
        throw std::out_of_range("unknown world action ID");
    }
    return *iterator;
}

const WorldStatePrototype& SparseWorldModel::state_by_id(
    const std::uint64_t state_id
) const {
    if (state_id == 0ULL || state_id > states_.size() ||
        states_[state_id - 1U].id != state_id) {
        throw std::out_of_range("unknown world state ID");
    }
    return states_[state_id - 1U];
}

const WorldContextPrototype& SparseWorldModel::context_by_id(
    const std::uint64_t context_id
) const {
    if (context_id == 0ULL || context_id > contexts_.size() ||
        contexts_[context_id - 1U].id != context_id) {
        throw std::out_of_range("unknown world context ID");
    }
    return contexts_[context_id - 1U];
}

std::uint64_t SparseWorldModel::phase_bucket_hash(
    const PhaseVector& vector
) const noexcept {
    std::uint64_t hash = fnv_offset_basis;
    for (const std::size_t coordinate : hash_coordinates_) {
        const double scaled = static_cast<double>(vector[coordinate]) /
            (2.0 * std::numbers::pi_v<double>);
        const auto bin = static_cast<std::uint64_t>(std::min<std::size_t>(
            config_.phase_bins - 1U,
            static_cast<std::size_t>(scaled *
                static_cast<double>(config_.phase_bins))
        ));
        hash_u64(hash, bin);
    }
    return hash;
}

std::vector<std::uint64_t> SparseWorldModel::nearby_bucket_hashes(
    const PhaseVector& vector
) const {
    std::vector<std::size_t> bins;
    bins.reserve(hash_coordinates_.size());
    for (const std::size_t coordinate : hash_coordinates_) {
        const double scaled = static_cast<double>(vector[coordinate]) /
            (2.0 * std::numbers::pi_v<double>);
        bins.push_back(std::min<std::size_t>(
            config_.phase_bins - 1U,
            static_cast<std::size_t>(scaled *
                static_cast<double>(config_.phase_bins))
        ));
    }
    const auto hash_bins = [](const std::span<const std::size_t> values) {
        std::uint64_t hash = fnv_offset_basis;
        for (const std::size_t value : values) {
            hash_u64(hash, static_cast<std::uint64_t>(value));
        }
        return hash;
    };
    std::vector<std::uint64_t> hashes;
    hashes.reserve(1U + bins.size() * 2U);
    hashes.push_back(hash_bins(bins));
    for (std::size_t index = 0U; index < bins.size(); ++index) {
        const std::size_t original = bins[index];
        bins[index] = (original + config_.phase_bins - 1U) %
            config_.phase_bins;
        hashes.push_back(hash_bins(bins));
        bins[index] = (original + 1U) % config_.phase_bins;
        hashes.push_back(hash_bins(bins));
        bins[index] = original;
    }
    std::sort(hashes.begin(), hashes.end());
    hashes.erase(std::unique(hashes.begin(), hashes.end()), hashes.end());
    return hashes;
}

SparseWorldModel::StateMatch SparseWorldModel::find_state(
    const PhaseVector& observation,
    const bool allow_fallback
) const {
    if (observation.size() != config_.dimension) {
        throw std::invalid_argument("world observation dimension mismatch");
    }
    StateMatch result;
    std::vector<std::uint64_t> candidates;
    for (const std::uint64_t hash : nearby_bucket_hashes(observation)) {
        const auto iterator = state_buckets_.find(hash);
        if (iterator == state_buckets_.end()) {
            continue;
        }
        candidates.insert(
            candidates.end(), iterator->second.begin(), iterator->second.end()
        );
        if (candidates.size() >= config_.maximum_bucket_candidates) {
            break;
        }
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    if (candidates.size() > config_.maximum_bucket_candidates) {
        candidates.resize(config_.maximum_bucket_candidates);
    }
    if (candidates.empty() && allow_fallback) {
        candidates.reserve(states_.size());
        for (const WorldStatePrototype& state : states_) {
            candidates.push_back(state.id);
        }
    }
    for (const std::uint64_t id : candidates) {
        const double distance = state_by_id(id).key.mean_angular_error(observation);
        ++result.comparisons;
        if (!result.found || distance < result.distance ||
            (distance == result.distance && id < result.id)) {
            result = {id, distance, result.comparisons, true};
        }
    }
    return result;
}

SparseWorldModel::StateMatch SparseWorldModel::find_context(
    const PhaseVector& context
) const {
    if (context.size() != config_.dimension) {
        throw std::invalid_argument("world context dimension mismatch");
    }
    StateMatch result;
    std::vector<std::uint64_t> candidates;
    for (const std::uint64_t hash : nearby_bucket_hashes(context)) {
        const auto iterator = context_buckets_.find(hash);
        if (iterator != context_buckets_.end()) {
            candidates.insert(
                candidates.end(), iterator->second.begin(), iterator->second.end()
            );
        }
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    if (candidates.empty()) {
        candidates.reserve(contexts_.size());
        for (const WorldContextPrototype& prototype : contexts_) {
            candidates.push_back(prototype.id);
        }
    }
    for (const std::uint64_t id : candidates) {
        const double distance = context_by_id(id).key.mean_angular_error(context);
        ++result.comparisons;
        if (!result.found || distance < result.distance ||
            (distance == result.distance && id < result.id)) {
            result = {id, distance, result.comparisons, true};
        }
    }
    return result;
}

std::uint64_t SparseWorldModel::match_or_create_state(
    const PhaseVector& observation
) {
    const StateMatch match = find_state(observation, true);
    ++stats_.state_index_queries;
    stats_.state_index_comparisons += match.comparisons;
    if (match.found && match.distance <= config_.state_merge_distance) {
        WorldStatePrototype& state = states_[match.id - 1U];
        ++state.support;
        state.last_used_step = training_step_;
        ++stats_.states_merged;
        return state.id;
    }
    if (states_.size() >= config_.maximum_states) {
        if (!match.found) {
            throw std::runtime_error("world state capacity exhausted");
        }
        ++stats_.state_index_fallbacks;
        return match.id;
    }
    const std::uint64_t id = next_state_id_++;
    states_.push_back({id, observation, 1ULL, training_step_, training_step_});
    state_buckets_[phase_bucket_hash(observation)].push_back(id);
    ++stats_.states_created;
    return id;
}

std::uint64_t SparseWorldModel::match_or_create_context(
    const PhaseVector& context
) {
    const StateMatch match = find_context(context);
    if (match.found && match.distance <= config_.context_merge_distance) {
        WorldContextPrototype& prototype = contexts_[match.id - 1U];
        ++prototype.support;
        prototype.last_used_step = training_step_;
        ++stats_.contexts_merged;
        return prototype.id;
    }
    if (contexts_.size() >= config_.maximum_contexts) {
        if (!match.found) {
            throw std::runtime_error("world context capacity exhausted");
        }
        return match.id;
    }
    const std::uint64_t id = next_context_id_++;
    contexts_.push_back({id, context, 1ULL, training_step_, training_step_});
    context_buckets_[phase_bucket_hash(context)].push_back(id);
    ++stats_.contexts_created;
    return id;
}

void SparseWorldModel::observe_transition(
    const WorldTransitionExperience& experience
) {
    static_cast<void>(action_by_id(experience.action_id));
    if (experience.observation.visible.size() != config_.dimension ||
        experience.observation.memory.size() != config_.dimension ||
        experience.next_observation.visible.size() != config_.dimension ||
        experience.next_observation.memory.size() != config_.dimension ||
        !std::isfinite(experience.reward)) {
        throw std::invalid_argument("invalid transition experience");
    }
    ++training_step_;
    ++stats_.experiences_observed;
    const std::uint64_t state_id = match_or_create_state(
        experience.observation.visible
    );
    const std::uint64_t context_id = match_or_create_context(
        experience.observation.memory
    );
    const std::uint64_t next_state_id = match_or_create_state(
        experience.next_observation.visible
    );
    const std::uint64_t next_context_id = match_or_create_context(
        experience.next_observation.memory
    );
    const std::uint64_t key = combine_key(
        state_id, context_id, experience.action_id
    );
    WorldTransition* transition = nullptr;
    const auto index_iterator = transition_index_.find(key);
    if (index_iterator != transition_index_.end()) {
        for (const std::size_t index : index_iterator->second) {
            WorldTransition& candidate = transitions_[index];
            if (candidate.state_id == state_id &&
                candidate.context_id == context_id &&
                candidate.action_id == experience.action_id) {
                transition = &candidate;
                break;
            }
        }
    }
    if (transition == nullptr) {
        if (transitions_.size() >= config_.maximum_transitions) {
            throw std::runtime_error("world transition capacity exhausted");
        }
        transitions_.push_back({
            next_transition_id_++, state_id, context_id,
            experience.action_id, 0ULL, 0.0, 0.0, {}
        });
        transition_index_[key].push_back(transitions_.size() - 1U);
        transition = &transitions_.back();
        ++stats_.transitions_created;
    }
    ++transition->support;
    transition->mean_reward +=
        (experience.reward - transition->mean_reward) /
        static_cast<double>(transition->support);
    transition->mean_prediction_error +=
        (state_by_id(next_state_id).key.mean_angular_error(
            experience.next_observation.visible
        ) - transition->mean_prediction_error) /
        static_cast<double>(transition->support);

    WorldTransitionOutcome* outcome = nullptr;
    for (WorldTransitionOutcome& candidate : transition->outcomes) {
        if (candidate.next_state_id == next_state_id &&
            candidate.next_context_id == next_context_id) {
            outcome = &candidate;
            break;
        }
    }
    if (outcome == nullptr) {
        if (transition->outcomes.size() >=
            config_.maximum_outcomes_per_transition) {
            auto least = std::min_element(
                transition->outcomes.begin(), transition->outcomes.end(),
                [](const WorldTransitionOutcome& left,
                   const WorldTransitionOutcome& right) {
                    return left.count < right.count;
                }
            );
            outcome = &*least;
        } else {
            transition->outcomes.push_back({
                next_state_id, next_context_id, 0ULL, 0.0, 0.0
            });
            outcome = &transition->outcomes.back();
            ++stats_.outcomes_created;
        }
    }
    ++outcome->count;
    outcome->mean_reward +=
        (experience.reward - outcome->mean_reward) /
        static_cast<double>(outcome->count);
    const double terminal_sample = experience.terminal ? 1.0 : 0.0;
    outcome->terminal_probability +=
        (terminal_sample - outcome->terminal_probability) /
        static_cast<double>(outcome->count);
}

void SparseWorldModel::observe_successful_route(
    const WorldRouteExperience& route
) {
    if (!route.success || route.observations.size() < 2U ||
        route.actions.size() + 1U != route.observations.size()) {
        throw std::invalid_argument("invalid successful world route");
    }
    std::vector<std::uint64_t> state_ids;
    state_ids.reserve(route.observations.size());
    for (const WorldObservation& observation : route.observations) {
        state_ids.push_back(match_or_create_state(observation.visible));
        static_cast<void>(match_or_create_context(observation.memory));
    }
    const std::uint64_t goal_state_id = state_ids.back();
    for (std::size_t index = 0U; index < route.actions.size(); ++index) {
        const std::uint64_t state_id = state_ids[index];
        const std::uint64_t action_id = route.actions[index];
        static_cast<void>(action_by_id(action_id));
        SparseSubgoal* record = nullptr;
        for (SparseSubgoal& candidate : subgoals_) {
            if (candidate.state_id == state_id &&
                candidate.goal_state_id == goal_state_id &&
                candidate.preferred_action_id == action_id) {
                record = &candidate;
                break;
            }
        }
        const double remaining = static_cast<double>(
            route.actions.size() - index
        );
        if (record == nullptr) {
            if (subgoals_.size() >= config_.maximum_subgoals) {
                continue;
            }
            subgoals_.push_back({
                next_subgoal_id_++, state_id, goal_state_id, action_id,
                remaining, 1.0, 1ULL, training_step_, training_step_
            });
            ++stats_.subgoals_created;
        } else {
            ++record->support;
            record->remaining_steps +=
                (remaining - record->remaining_steps) /
                static_cast<double>(record->support);
            record->success_probability +=
                (1.0 - record->success_probability) /
                static_cast<double>(record->support);
            record->last_used_step = training_step_;
            ++stats_.subgoals_merged;
        }
    }
    rebuild_indices();
}

std::optional<std::uint64_t> SparseWorldModel::match_state(
    const PhaseVector& observation,
    std::size_t* const comparisons
) const {
    const StateMatch match = find_state(observation, true);
    if (comparisons != nullptr) {
        *comparisons = match.comparisons;
    }
    if (!match.found || match.distance > config_.state_merge_distance * 2.0) {
        return std::nullopt;
    }
    return match.id;
}

std::optional<std::uint64_t> SparseWorldModel::match_context(
    const PhaseVector& context
) const {
    const StateMatch match = find_context(context);
    if (!match.found || match.distance > config_.context_merge_distance * 2.0) {
        return std::nullopt;
    }
    return match.id;
}

std::optional<WorldPrediction> SparseWorldModel::predict_by_id(
    const std::uint64_t state_id,
    const std::uint64_t context_id,
    const std::uint64_t action_id
) const {
    static_cast<void>(state_by_id(state_id));
    static_cast<void>(context_by_id(context_id));
    static_cast<void>(action_by_id(action_id));
    const std::uint64_t key = combine_key(state_id, context_id, action_id);
    const auto iterator = transition_index_.find(key);
    if (iterator == transition_index_.end()) {
        return std::nullopt;
    }
    const WorldTransition* selected = nullptr;
    for (const std::size_t index : iterator->second) {
        const WorldTransition& candidate = transitions_[index];
        if (candidate.state_id == state_id &&
            candidate.context_id == context_id &&
            candidate.action_id == action_id &&
            candidate.support >= config_.minimum_transition_support) {
            selected = &candidate;
            break;
        }
    }
    if (selected == nullptr || selected->support == 0ULL) {
        return std::nullopt;
    }
    WorldPrediction prediction;
    prediction.state_id = state_id;
    prediction.context_id = context_id;
    prediction.transition_id = selected->id;
    prediction.action_id = action_id;
    prediction.context_distance = 0.0;
    double maximum_probability = 0.0;
    for (const WorldTransitionOutcome& outcome : selected->outcomes) {
        const double probability = static_cast<double>(outcome.count) /
            static_cast<double>(selected->support);
        if (probability + probability_floor <
            config_.minimum_outcome_probability) {
            continue;
        }
        prediction.outcomes.push_back({
            outcome.next_state_id,
            outcome.next_context_id,
            probability,
            outcome.mean_reward,
            outcome.terminal_probability,
        });
        maximum_probability = std::max(maximum_probability, probability);
    }
    std::sort(
        prediction.outcomes.begin(), prediction.outcomes.end(),
        [](const WorldPredictionOutcome& left,
           const WorldPredictionOutcome& right) {
            if (left.probability != right.probability) {
                return left.probability > right.probability;
            }
            if (left.next_state_id != right.next_state_id) {
                return left.next_state_id < right.next_state_id;
            }
            return left.next_context_id < right.next_context_id;
        }
    );
    if (prediction.outcomes.empty()) {
        return std::nullopt;
    }
    prediction.uncertainty = std::clamp(1.0 - maximum_probability, 0.0, 1.0);
    return prediction;
}

std::optional<WorldPrediction> SparseWorldModel::predict(
    const WorldObservation& observation,
    const std::uint64_t action_id
) const {
    const auto state_id = match_state(observation.visible);
    const auto context_id = match_context(observation.memory);
    if (!state_id.has_value() || !context_id.has_value()) {
        return std::nullopt;
    }
    return predict_by_id(*state_id, *context_id, action_id);
}

SparseWorldModel::SubgoalEstimate SparseWorldModel::estimate_subgoal(
    const std::uint64_t state_id,
    const std::uint64_t goal_state_id
) const {
    SubgoalEstimate result;
    const PhaseVector& key = state_by_id(state_id).key;
    std::vector<std::size_t> candidates;
    for (const std::uint64_t bucket : nearby_bucket_hashes(key)) {
        const std::uint64_t index_key = combine_key(goal_state_id, bucket);
        const auto iterator = subgoal_index_.find(index_key);
        if (iterator != subgoal_index_.end()) {
            candidates.insert(
                candidates.end(), iterator->second.begin(), iterator->second.end()
            );
        }
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    struct Ranked final { double distance{}; std::size_t index{}; };
    std::vector<Ranked> ranked;
    ranked.reserve(candidates.size());
    for (const std::size_t index : candidates) {
        const SparseSubgoal& candidate = subgoals_[index];
        if (candidate.goal_state_id != goal_state_id ||
            candidate.support < config_.minimum_subgoal_support) {
            continue;
        }
        const double distance = state_by_id(candidate.state_id)
            .key.mean_angular_error(key);
        ranked.push_back({distance, index});
        ++result.comparisons;
    }
    std::sort(
        ranked.begin(), ranked.end(),
        [](const Ranked& left, const Ranked& right) {
            if (left.distance != right.distance) {
                return left.distance < right.distance;
            }
            return left.index < right.index;
        }
    );
    if (ranked.size() > config_.nearest_subgoals) {
        ranked.resize(config_.nearest_subgoals);
    }
    if (ranked.empty()) {
        return result;
    }
    double weight_sum = 0.0;
    double remaining_sum = 0.0;
    double success_sum = 0.0;
    std::unordered_map<std::uint64_t, double> action_weights;
    for (const Ranked& item : ranked) {
        const SparseSubgoal& candidate = subgoals_[item.index];
        const double weight = static_cast<double>(candidate.support) /
            (1.0 + item.distance * 8.0);
        weight_sum += weight;
        remaining_sum += weight * candidate.remaining_steps;
        success_sum += weight * candidate.success_probability;
        action_weights[candidate.preferred_action_id] += weight;
    }
    if (weight_sum <= 0.0) {
        return result;
    }
    result.remaining_steps = remaining_sum / weight_sum;
    result.success_probability = success_sum / weight_sum;
    result.found = true;
    double best_weight = -1.0;
    for (const auto& [action_id, weight] : action_weights) {
        if (weight > best_weight ||
            (weight == best_weight && action_id < result.preferred_action_id)) {
            best_weight = weight;
            result.preferred_action_id = action_id;
        }
    }
    return result;
}

Rlf3PlanResult SparseWorldModel::plan(
    const WorldObservation& start,
    const PhaseVector& goal,
    const bool use_subgoal_index,
    const std::size_t maximum_depth,
    const std::size_t node_budget
) const {
    Rlf3PlanResult result;
    const auto start_state = match_state(start.visible);
    const auto start_context = match_context(start.memory);
    if (!start_state.has_value() || !start_context.has_value()) {
        result.stop_reason = Rlf3PlanStopReason::no_state_match;
        return result;
    }
    const auto goal_state = match_state(goal);
    if (!goal_state.has_value()) {
        result.stop_reason = Rlf3PlanStopReason::no_goal_match;
        return result;
    }
    if (*start_state == *goal_state) {
        result.success = true;
        result.stop_reason = Rlf3PlanStopReason::goal_reached;
        result.predicted_success_probability = 1.0;
        return result;
    }
    const std::size_t depth_limit = maximum_depth == 0U
        ? config_.maximum_plan_depth
        : maximum_depth;
    const std::size_t budget = node_budget == 0U
        ? config_.planner_node_budget
        : node_budget;

    std::vector<PlannerNode> nodes;
    nodes.reserve(std::min<std::size_t>(budget, 65'536U));
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, QueueCompare> open;
    std::unordered_map<std::uint64_t, double> best_cost;
    auto heuristic_for = [&](const std::uint64_t state_id) {
        double heuristic = normalized_error(
            state_by_id(state_id).key,
            state_by_id(*goal_state).key
        ) * static_cast<double>(depth_limit) * config_.heuristic_scale;
        if (use_subgoal_index) {
            const SubgoalEstimate estimate = estimate_subgoal(
                state_id, *goal_state
            );
            ++result.subgoal_queries;
            result.subgoal_comparisons += estimate.comparisons;
            if (estimate.found) {
                heuristic = std::max(0.0, estimate.remaining_steps - 1.0);
            }
        }
        return heuristic;
    };
    nodes.push_back({
        *start_state, *start_context, 0.0, 1.0,
        heuristic_for(*start_state), 0U,
        std::numeric_limits<std::size_t>::max(), 0ULL,
        *start_state, *start_context, 1.0
    });
    open.push({nodes.front().heuristic, 0U, 0ULL});
    best_cost[combine_key(*start_state, *start_context)] = 0.0;
    std::uint64_t stable_order = 1ULL;
    std::size_t goal_node = std::numeric_limits<std::size_t>::max();
    bool saw_transition = false;
    bool hit_depth = false;

    while (!open.empty() && result.nodes_expanded < budget) {
        const QueueEntry entry = open.top();
        open.pop();
        const PlannerNode current = nodes[entry.node_index];
        if (current.state_id == *goal_state) {
            goal_node = entry.node_index;
            break;
        }
        ++result.nodes_expanded;
        if (current.depth >= depth_limit) {
            hit_depth = true;
            continue;
        }
        for (const WorldAction& action : actions_) {
            ++result.transitions_evaluated;
            const auto prediction = predict_by_id(
                current.state_id, current.context_id, action.id
            );
            if (!prediction.has_value()) {
                continue;
            }
            saw_transition = true;
            for (const WorldPredictionOutcome& outcome : prediction->outcomes) {
                ++result.outcome_branches;
                const double probability = std::max(
                    probability_floor,
                    current.probability * outcome.probability
                );
                const double risk = config_.risk_penalty * -std::log(probability);
                const double uncertainty_cost = config_.uncertainty_penalty *
                    prediction->uncertainty;
                const double next_cost = current.cost + action.cost +
                    risk + uncertainty_cost -
                    std::clamp(outcome.expected_reward, -2.0, 2.0) * 0.50;
                const std::uint64_t key = combine_key(
                    outcome.next_state_id, outcome.next_context_id
                );
                const auto existing = best_cost.find(key);
                if (existing != best_cost.end() &&
                    existing->second <= next_cost + 1.0e-12) {
                    continue;
                }
                best_cost[key] = next_cost;
                const double heuristic = heuristic_for(outcome.next_state_id);
                const std::size_t index = nodes.size();
                nodes.push_back({
                    outcome.next_state_id,
                    outcome.next_context_id,
                    next_cost,
                    probability,
                    heuristic,
                    current.depth + 1U,
                    entry.node_index,
                    action.id,
                    outcome.next_state_id,
                    outcome.next_context_id,
                    outcome.probability,
                });
                open.push({next_cost + heuristic, index, stable_order++});
            }
        }
    }

    if (goal_node == std::numeric_limits<std::size_t>::max()) {
        if (result.nodes_expanded >= budget) {
            result.stop_reason = Rlf3PlanStopReason::node_budget;
        } else if (!saw_transition) {
            result.stop_reason = Rlf3PlanStopReason::no_transition;
        } else if (hit_depth) {
            result.stop_reason = Rlf3PlanStopReason::depth_limit;
        } else {
            result.stop_reason = Rlf3PlanStopReason::exhausted;
        }
        return result;
    }

    std::vector<std::size_t> reverse_indices;
    for (std::size_t index = goal_node;
         nodes[index].parent != std::numeric_limits<std::size_t>::max();
         index = nodes[index].parent) {
        reverse_indices.push_back(index);
    }
    std::reverse(reverse_indices.begin(), reverse_indices.end());
    result.actions.reserve(reverse_indices.size());
    result.steps.reserve(reverse_indices.size());
    for (const std::size_t index : reverse_indices) {
        const PlannerNode& node = nodes[index];
        const PlannerNode& parent = nodes[node.parent];
        result.actions.push_back(node.action_id);
        result.steps.push_back({
            parent.state_id,
            parent.context_id,
            node.action_id,
            node.transition_next_state,
            node.transition_next_context,
            node.outcome_probability,
            node.cost,
            node.heuristic,
        });
    }
    result.success = true;
    result.stop_reason = Rlf3PlanStopReason::goal_reached;
    result.predicted_success_probability = nodes[goal_node].probability;
    result.total_cost = nodes[goal_node].cost;
    return result;
}

double SparseWorldModel::transition_top1_accuracy(
    const std::span<const WorldTransitionExperience> experiences
) const {
    if (experiences.empty()) {
        return 0.0;
    }
    std::size_t correct = 0U;
    std::size_t evaluated = 0U;
    for (const WorldTransitionExperience& experience : experiences) {
        const auto prediction = predict(experience.observation, experience.action_id);
        const auto actual_state = match_state(experience.next_observation.visible);
        const auto actual_context = match_context(experience.next_observation.memory);
        if (!prediction.has_value() || !actual_state.has_value() ||
            !actual_context.has_value()) {
            continue;
        }
        ++evaluated;
        if (!prediction->outcomes.empty() &&
            prediction->outcomes.front().next_state_id == *actual_state &&
            prediction->outcomes.front().next_context_id == *actual_context) {
            ++correct;
        }
    }
    return evaluated == 0U ? 0.0 :
        static_cast<double>(correct) / static_cast<double>(evaluated);
}

double SparseWorldModel::transition_negative_log_likelihood(
    const std::span<const WorldTransitionExperience> experiences
) const {
    if (experiences.empty()) {
        return 0.0;
    }
    double total = 0.0;
    std::size_t evaluated = 0U;
    for (const WorldTransitionExperience& experience : experiences) {
        const auto prediction = predict(experience.observation, experience.action_id);
        const auto actual_state = match_state(experience.next_observation.visible);
        const auto actual_context = match_context(experience.next_observation.memory);
        if (!prediction.has_value() || !actual_state.has_value() ||
            !actual_context.has_value()) {
            continue;
        }
        double probability = probability_floor;
        for (const WorldPredictionOutcome& outcome : prediction->outcomes) {
            if (outcome.next_state_id == *actual_state &&
                outcome.next_context_id == *actual_context) {
                probability = std::max(probability_floor, outcome.probability);
                break;
            }
        }
        total += -std::log(probability);
        ++evaluated;
    }
    return evaluated == 0U ? std::numeric_limits<double>::infinity() :
        total / static_cast<double>(evaluated);
}

double SparseWorldModel::transition_brier_score(
    const std::span<const WorldTransitionExperience> experiences
) const {
    if (experiences.empty()) {
        return 0.0;
    }
    double total = 0.0;
    std::size_t evaluated = 0U;
    for (const WorldTransitionExperience& experience : experiences) {
        const auto prediction = predict(experience.observation, experience.action_id);
        const auto actual_state = match_state(experience.next_observation.visible);
        const auto actual_context = match_context(experience.next_observation.memory);
        if (!prediction.has_value() || !actual_state.has_value() ||
            !actual_context.has_value()) {
            continue;
        }
        double sample = 0.0;
        bool actual_seen = false;
        for (const WorldPredictionOutcome& outcome : prediction->outcomes) {
            const bool actual = outcome.next_state_id == *actual_state &&
                outcome.next_context_id == *actual_context;
            const double target = actual ? 1.0 : 0.0;
            const double difference = outcome.probability - target;
            sample += difference * difference;
            actual_seen = actual_seen || actual;
        }
        if (!actual_seen) {
            sample += 1.0;
        }
        total += sample;
        ++evaluated;
    }
    return evaluated == 0U ? 1.0 : total / static_cast<double>(evaluated);
}

std::size_t SparseWorldModel::estimated_bytes() const noexcept {
    std::size_t bytes = sizeof(*this);
    bytes += actions_.capacity() * sizeof(WorldAction);
    for (const WorldAction& action : actions_) {
        bytes += action.name.capacity();
    }
    bytes += states_.capacity() * sizeof(WorldStatePrototype);
    for (const WorldStatePrototype& state : states_) {
        bytes += state.key.size() * sizeof(float);
    }
    bytes += contexts_.capacity() * sizeof(WorldContextPrototype);
    for (const WorldContextPrototype& context : contexts_) {
        bytes += context.key.size() * sizeof(float);
    }
    bytes += transitions_.capacity() * sizeof(WorldTransition);
    for (const WorldTransition& transition : transitions_) {
        bytes += transition.outcomes.capacity() * sizeof(WorldTransitionOutcome);
    }
    bytes += subgoals_.capacity() * sizeof(SparseSubgoal);
    return bytes;
}

SparseWorldModelSnapshot SparseWorldModel::snapshot() const {
    return {
        config_, seed_, training_step_, next_action_id_, next_state_id_,
        next_context_id_, next_transition_id_, next_subgoal_id_, actions_,
        states_, contexts_, transitions_, subgoals_, stats_
    };
}

SparseWorldModel SparseWorldModel::from_snapshot(
    SparseWorldModelSnapshot snapshot
) {
    SparseWorldModel model(snapshot.config, snapshot.seed);
    model.training_step_ = snapshot.training_step;
    model.next_action_id_ = snapshot.next_action_id;
    model.next_state_id_ = snapshot.next_state_id;
    model.next_context_id_ = snapshot.next_context_id;
    model.next_transition_id_ = snapshot.next_transition_id;
    model.next_subgoal_id_ = snapshot.next_subgoal_id;
    model.actions_ = std::move(snapshot.actions);
    model.states_ = std::move(snapshot.states);
    model.contexts_ = std::move(snapshot.contexts);
    model.transitions_ = std::move(snapshot.transitions);
    model.subgoals_ = std::move(snapshot.subgoals);
    model.stats_ = snapshot.stats;
    model.validate_snapshot();
    model.rebuild_indices();
    return model;
}

void SparseWorldModel::rebuild_indices() {
    state_buckets_.clear();
    context_buckets_.clear();
    transition_index_.clear();
    subgoal_index_.clear();
    for (const WorldStatePrototype& state : states_) {
        state_buckets_[phase_bucket_hash(state.key)].push_back(state.id);
    }
    for (const WorldContextPrototype& context : contexts_) {
        context_buckets_[phase_bucket_hash(context.key)].push_back(context.id);
    }
    for (std::size_t index = 0U; index < transitions_.size(); ++index) {
        const WorldTransition& transition = transitions_[index];
        transition_index_[combine_key(
            transition.state_id, transition.context_id, transition.action_id
        )].push_back(index);
    }
    for (std::size_t index = 0U; index < subgoals_.size(); ++index) {
        const SparseSubgoal& subgoal = subgoals_[index];
        const std::uint64_t bucket = phase_bucket_hash(
            state_by_id(subgoal.state_id).key
        );
        subgoal_index_[combine_key(subgoal.goal_state_id, bucket)].push_back(index);
    }
}

void SparseWorldModel::validate_snapshot() const {
    if (actions_.size() >= next_action_id_ || states_.size() >= next_state_id_ ||
        contexts_.size() >= next_context_id_ ||
        transitions_.size() >= next_transition_id_ ||
        subgoals_.size() >= next_subgoal_id_ ||
        states_.size() > config_.maximum_states ||
        contexts_.size() > config_.maximum_contexts ||
        transitions_.size() > config_.maximum_transitions ||
        subgoals_.size() > config_.maximum_subgoals) {
        throw std::runtime_error("invalid sparse world model snapshot counters");
    }
    for (std::size_t index = 0U; index < actions_.size(); ++index) {
        if (actions_[index].id != index + 1ULL || actions_[index].name.empty() ||
            !std::isfinite(actions_[index].cost) || actions_[index].cost <= 0.0) {
            throw std::runtime_error("invalid sparse world model action");
        }
    }
    for (std::size_t index = 0U; index < states_.size(); ++index) {
        if (states_[index].id != index + 1ULL ||
            states_[index].key.size() != config_.dimension ||
            states_[index].support == 0ULL) {
            throw std::runtime_error("invalid sparse world model state");
        }
    }
    for (std::size_t index = 0U; index < contexts_.size(); ++index) {
        if (contexts_[index].id != index + 1ULL ||
            contexts_[index].key.size() != config_.dimension ||
            contexts_[index].support == 0ULL) {
            throw std::runtime_error("invalid sparse world model context");
        }
    }
    for (std::size_t index = 0U; index < transitions_.size(); ++index) {
        const WorldTransition& transition = transitions_[index];
        if (transition.id != index + 1ULL || transition.support == 0ULL ||
            transition.state_id == 0ULL || transition.state_id > states_.size() ||
            transition.context_id == 0ULL ||
            transition.context_id > contexts_.size() ||
            transition.action_id == 0ULL ||
            transition.action_id > actions_.size() ||
            transition.outcomes.empty() ||
            transition.outcomes.size() > config_.maximum_outcomes_per_transition) {
            throw std::runtime_error("invalid sparse world model transition");
        }
        std::uint64_t total = 0ULL;
        for (const WorldTransitionOutcome& outcome : transition.outcomes) {
            if (outcome.count == 0ULL || outcome.next_state_id == 0ULL ||
                outcome.next_state_id > states_.size() ||
                outcome.next_context_id == 0ULL ||
                outcome.next_context_id > contexts_.size() ||
                !std::isfinite(outcome.mean_reward) ||
                !finite_probability(outcome.terminal_probability)) {
                throw std::runtime_error("invalid sparse world model outcome");
            }
            total += outcome.count;
        }
        if (total != transition.support) {
            throw std::runtime_error("world transition support mismatch");
        }
    }
    for (std::size_t index = 0U; index < subgoals_.size(); ++index) {
        const SparseSubgoal& subgoal = subgoals_[index];
        if (subgoal.id != index + 1ULL || subgoal.state_id == 0ULL ||
            subgoal.state_id > states_.size() || subgoal.goal_state_id == 0ULL ||
            subgoal.goal_state_id > states_.size() ||
            subgoal.preferred_action_id == 0ULL ||
            subgoal.preferred_action_id > actions_.size() ||
            !std::isfinite(subgoal.remaining_steps) ||
            subgoal.remaining_steps <= 0.0 ||
            !finite_probability(subgoal.success_probability) ||
            subgoal.support == 0ULL) {
            throw std::runtime_error("invalid sparse world model subgoal");
        }
    }
}

}  // namespace rlf::core
