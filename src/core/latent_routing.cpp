#include "rlf/core/latent_routing.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rlf::core {
namespace {

constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;

void hash_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (unsigned int byte = 0U; byte < 8U; ++byte) {
        hash ^= (value >> (byte * 8U)) & 0xFFULL;
        hash *= fnv_prime;
    }
}

[[nodiscard]] PhaseVector circular_interpolate(
    const PhaseVector& current,
    const PhaseVector& target,
    const double amount
) {
    const std::vector<PhaseVector> values{current, target};
    const std::vector<float> weights{
        static_cast<float>(1.0 - amount),
        static_cast<float>(amount),
    };
    return PhaseVector::weighted_circular_average(values, weights);
}

[[nodiscard]] double immediate_progress(
    const PhaseVector& before,
    const PhaseVector& after,
    const PhaseVector& goal,
    const double cost
) {
    return after.similarity(goal) - before.similarity(goal) -
        (0.001 * cost);
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

}  // namespace

std::string_view to_string(const LatentCreditStrategy strategy) noexcept {
    switch (strategy) {
    case LatentCreditStrategy::uniform_route:
        return "uniform_route";
    case LatentCreditStrategy::discounted_eligibility:
        return "discounted_eligibility";
    case LatentCreditStrategy::progress_weighted:
        return "progress_weighted";
    case LatentCreditStrategy::counterfactual_local:
        return "counterfactual_local";
    }
    return "unknown";
}

std::string_view to_string(const LatentHaltPolicy policy) noexcept {
    switch (policy) {
    case LatentHaltPolicy::goal_threshold:
        return "goal_threshold";
    case LatentHaltPolicy::learned_resonant:
        return "learned_resonant";
    case LatentHaltPolicy::combined_safe:
        return "combined_safe";
    }
    return "unknown";
}

std::string_view to_string(const LatentStopReason reason) noexcept {
    switch (reason) {
    case LatentStopReason::successful_halt:
        return "successful_halt";
    case LatentStopReason::learned_halt:
        return "learned_halt";
    case LatentStopReason::premature_halt:
        return "premature_halt";
    case LatentStopReason::cycle_limit:
        return "cycle_limit";
    case LatentStopReason::loop_detected:
        return "loop_detected";
    case LatentStopReason::abstained:
        return "abstained";
    case LatentStopReason::no_candidate:
        return "no_candidate";
    }
    return "unknown";
}

LatentRouter::LatentRouter(
    LatentRouterConfig config,
    const std::uint64_t seed
)
    : config_(std::move(config)),
      seed_(seed),
      rng_(seed) {
    if (config_.dimension == 0U ||
        config_.maximum_cycles == 0U ||
        config_.search_node_budget == 0U ||
        config_.route_memory_capacity == 0U ||
        config_.maximum_modes == 0U ||
        config_.macro_maximum_length == 0U ||
        !std::isfinite(config_.goal_similarity_threshold) ||
        config_.goal_similarity_threshold <= 0.0 ||
        config_.goal_similarity_threshold > 1.0 ||
        !std::isfinite(config_.mode_creation_similarity) ||
        config_.mode_creation_similarity < 0.0 ||
        config_.mode_creation_similarity > 1.0 ||
        !std::isfinite(config_.mode_learning_rate) ||
        config_.mode_learning_rate < 0.0 ||
        config_.mode_learning_rate > 1.0 ||
        !std::isfinite(config_.utility_learning_rate) ||
        config_.utility_learning_rate < 0.0 ||
        config_.utility_learning_rate > 1.0 ||
        !std::isfinite(config_.eligibility_decay) ||
        config_.eligibility_decay < 0.0 ||
        config_.eligibility_decay > 1.0 ||
        !std::isfinite(config_.action_temperature) ||
        config_.action_temperature <= 0.0 ||
        !std::isfinite(config_.successor_familiarity_weight) ||
        config_.successor_familiarity_weight < 0.0 ||
        !std::isfinite(config_.route_repetition_penalty) ||
        config_.route_repetition_penalty < 0.0 ||
        !std::isfinite(config_.abstention_resonance_threshold) ||
        config_.abstention_resonance_threshold < 0.0 ||
        config_.abstention_resonance_threshold > 1.0 ||
        config_.search_beam_width == 0U ||
        config_.search_lookahead_depth == 0U) {
        throw std::invalid_argument("invalid latent router configuration");
    }
}

const LatentRouterConfig& LatentRouter::config() const noexcept {
    return config_;
}

std::uint64_t LatentRouter::seed() const noexcept {
    return seed_;
}

std::uint64_t LatentRouter::training_step() const noexcept {
    return training_step_;
}

std::span<const RegisteredOperator> LatentRouter::operators() const noexcept {
    return operators_;
}

std::span<const LatentRoutingMode> LatentRouter::modes() const noexcept {
    return modes_;
}

std::span<const LatentHaltMode> LatentRouter::halt_modes() const noexcept {
    return halt_modes_;
}

std::span<const RouteMemoryRecord> LatentRouter::route_memory() const noexcept {
    return route_memory_;
}

std::uint64_t LatentRouter::register_operator(
    std::string name,
    TransformationOperator transformation,
    const double cost,
    const bool macro,
    std::vector<std::uint64_t> primitive_route
) {
    if (name.empty() || transformation.dimension() != config_.dimension ||
        !std::isfinite(cost) || cost <= 0.0) {
        throw std::invalid_argument("invalid latent operator registration");
    }
    for (const RegisteredOperator& existing : operators_) {
        if (existing.name == name) {
            throw std::invalid_argument("latent operator names must be unique");
        }
    }
    const std::uint64_t id = next_operator_id_++;
    operators_.push_back({
        .id = id,
        .name = std::move(name),
        .transformation = std::move(transformation),
        .cost = cost,
        .macro = macro,
        .primitive_route = std::move(primitive_route),
    });
    return id;
}

const RegisteredOperator& LatentRouter::operator_by_id(
    const std::uint64_t operator_id
) const {
    const auto iterator = std::find_if(
        operators_.begin(),
        operators_.end(),
        [operator_id](const RegisteredOperator& value) {
            return value.id == operator_id;
        }
    );
    if (iterator == operators_.end()) {
        throw std::out_of_range("unknown latent operator ID");
    }
    return *iterator;
}

PhaseVector LatentRouter::state_goal_signature(
    const PhaseVector& current,
    const PhaseVector& goal
) {
    return PhaseVector::phase_difference(current, goal);
}

std::uint64_t LatentRouter::phase_state_hash(
    const PhaseVector& state
) noexcept {
    std::uint64_t hash = fnv_offset_basis;
    constexpr double bins = 65'536.0;
    constexpr double tau = 2.0 * std::numbers::pi_v<double>;
    for (const float angle : state.angles()) {
        const auto quantized = static_cast<std::uint64_t>(
            std::llround((static_cast<double>(angle) / tau) * bins)
        ) & 0xFFFFULL;
        hash_u64(hash, quantized);
    }
    return hash;
}

std::uint64_t LatentRouter::episode_state_hash(
    const LatentEpisodeState& state
) noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, phase_state_hash(state.current_state));
    hash_u64(hash, phase_state_hash(state.goal_state));
    hash_u64(hash, phase_state_hash(state.working_state));
    hash_u64(hash, phase_state_hash(state.memory_summary));
    hash_u64(hash, phase_state_hash(state.route_summary));
    hash_u64(hash, std::bit_cast<std::uint64_t>(state.uncertainty_state));
    hash_u64(hash, std::bit_cast<std::uint64_t>(state.progress_state));
    hash_u64(hash, static_cast<std::uint64_t>(state.step_index));
    return hash;
}

std::uint64_t LatentRouter::route_hash(
    const std::span<const std::uint64_t> route
) noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, static_cast<std::uint64_t>(route.size()));
    for (const std::uint64_t value : route) {
        hash_u64(hash, value);
    }
    return hash;
}

PhaseVector LatentRouter::candidate_memory_summary(
    const std::span<const LatentActionCandidate> candidates
) const {
    std::vector<PhaseVector> values;
    std::vector<float> weights;
    values.reserve(candidates.size());
    weights.reserve(candidates.size());
    for (const LatentActionCandidate& candidate : candidates) {
        if (candidate.routing_mode_id == 0ULL ||
            candidate.normalized_weight <= 0.0) {
            continue;
        }
        const auto mode = std::find_if(
            modes_.begin(),
            modes_.end(),
            [&candidate](const LatentRoutingMode& value) {
                return value.id == candidate.routing_mode_id;
            }
        );
        if (mode == modes_.end()) {
            continue;
        }
        values.push_back(mode->key);
        weights.push_back(static_cast<float>(candidate.normalized_weight));
    }
    if (values.empty()) {
        return PhaseVector::zeros(config_.dimension);
    }
    return PhaseVector::weighted_circular_average(values, weights);
}

PhaseVector LatentRouter::operator_route_code(
    const std::size_t dimension,
    const std::uint64_t operator_id
) {
    std::vector<float> angles;
    angles.reserve(dimension);
    std::uint64_t state = operator_id ^ 0x9E3779B97F4A7C15ULL;
    constexpr double tau = 2.0 * std::numbers::pi_v<double>;
    for (std::size_t index = 0U; index < dimension; ++index) {
        state += 0x9E3779B97F4A7C15ULL +
            static_cast<std::uint64_t>(index);
        std::uint64_t value = state;
        value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
        value ^= value >> 31U;
        const double unit = static_cast<double>(value >> 11U) /
            static_cast<double>(1ULL << 53U);
        angles.push_back(static_cast<float>(unit * tau));
    }
    return PhaseVector(std::move(angles));
}

bool LatentRouter::is_goal(
    const PhaseVector& state,
    const PhaseVector& goal
) const {
    return state.similarity(goal) >= config_.goal_similarity_threshold;
}

LatentRouter::ModeSelection LatentRouter::best_mode_for_operator(
    const PhaseVector& signature,
    const std::uint64_t operator_id,
    std::size_t* const similarity_evaluations
) const {
    ModeSelection result;
    for (std::size_t index = 0U; index < modes_.size(); ++index) {
        const LatentRoutingMode& mode = modes_[index];
        if (mode.operator_id != operator_id) {
            continue;
        }
        const double resonance = signature.similarity(mode.key);
        if (similarity_evaluations != nullptr) {
            ++(*similarity_evaluations);
        }
        if (!result.found || resonance > result.resonance + 1.0e-12 ||
            (std::abs(resonance - result.resonance) <= 1.0e-12 &&
             mode.id < modes_[result.mode_index].id)) {
            result = {
                .mode_index = index,
                .resonance = resonance,
                .found = true,
            };
        }
    }
    return result;
}

std::optional<std::uint64_t> LatentRouter::route_memory_suggestion(
    const PhaseVector& signature,
    double* const similarity,
    std::size_t* const similarity_evaluations,
    const bool record_usage
) {
    if (!config_.enable_route_memory || route_memory_.empty()) {
        if (similarity != nullptr) {
            *similarity = 0.0;
        }
        return std::nullopt;
    }
    std::size_t best_index = 0U;
    double best_similarity = -1.0;
    for (std::size_t index = 0U; index < route_memory_.size(); ++index) {
        const double value = signature.similarity(
            route_memory_[index].start_goal_signature
        );
        if (similarity_evaluations != nullptr) {
            ++(*similarity_evaluations);
        }
        if (value > best_similarity + 1.0e-12 ||
            (std::abs(value - best_similarity) <= 1.0e-12 &&
             route_memory_[index].id < route_memory_[best_index].id)) {
            best_index = index;
            best_similarity = value;
        }
    }
    if (similarity != nullptr) {
        *similarity = std::max(0.0, best_similarity);
    }
    RouteMemoryRecord& record = route_memory_[best_index];
    if (record.route.empty()) {
        return std::nullopt;
    }
    if (record_usage) {
        ++record.usage_count;
        record.last_used_step = training_step_;
    }
    return record.route.front();
}

std::vector<LatentActionCandidate> LatentRouter::score_actions(
    const PhaseVector& current,
    const PhaseVector& goal,
    const bool allow_route_memory,
    const bool record_memory_usage,
    std::size_t* const similarity_evaluations
) {
    const PhaseVector signature = state_goal_signature(current, goal);
    std::optional<std::uint64_t> memory_action;
    double memory_similarity = 0.0;
    if (allow_route_memory) {
        memory_action = route_memory_suggestion(
            signature,
            &memory_similarity,
            similarity_evaluations,
            record_memory_usage
        );
    }

    std::vector<LatentActionCandidate> candidates;
    candidates.reserve(operators_.size());
    for (const RegisteredOperator& operator_value : operators_) {
        const ModeSelection mode = best_mode_for_operator(
            signature,
            operator_value.id,
            similarity_evaluations
        );
        const PhaseVector proposal =
            operator_value.transformation.apply(current);
        const double progress = immediate_progress(
            current,
            proposal,
            goal,
            operator_value.cost
        );
        double utility = 0.0;
        double confidence = 0.0;
        double eligibility = 0.0;
        std::uint64_t mode_id = 0ULL;
        if (mode.found) {
            const LatentRoutingMode& selected = modes_[mode.mode_index];
            utility = selected.utility;
            confidence = selected.confidence;
            eligibility = selected.eligibility;
            mode_id = selected.id;
        }
        const double memory_bonus =
            memory_action.has_value() &&
            *memory_action == operator_value.id
            ? config_.route_memory_weight * memory_similarity
            : 0.0;
        const double learned_score = mode.found
            ? (mode.resonance * confidence) + utility
            : -0.05;
        const double successor_familiarity = state_familiarity(
            proposal,
            goal,
            similarity_evaluations
        );
        candidates.push_back({
            .operator_id = operator_value.id,
            .routing_mode_id = mode_id,
            .resonance = mode.found ? mode.resonance : 0.0,
            .utility = utility,
            .eligibility = eligibility,
            .immediate_progress = progress,
            .route_memory_bonus = memory_bonus,
            .successor_familiarity = successor_familiarity,
            .score = learned_score +
                (config_.goal_progress_weight * progress) + memory_bonus +
                (config_.successor_familiarity_weight *
                 successor_familiarity),
            .normalized_weight = 0.0,
        });
    }
    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const LatentActionCandidate& left,
           const LatentActionCandidate& right) {
            if (std::abs(left.score - right.score) > 1.0e-12) {
                return left.score > right.score;
            }
            return left.operator_id < right.operator_id;
        }
    );
    return candidates;
}

double LatentRouter::state_familiarity(
    const PhaseVector& state,
    const PhaseVector& goal,
    std::size_t* const similarity_evaluations
) const {
    if (is_goal(state, goal)) {
        return 1.0;
    }
    const PhaseVector signature = state_goal_signature(state, goal);
    double best = 0.0;
    for (const LatentRoutingMode& mode : modes_) {
        const double resonance = signature.similarity(mode.key);
        if (similarity_evaluations != nullptr) {
            ++(*similarity_evaluations);
        }
        const double value = resonance * mode.confidence +
            (0.10 * std::max(0.0, mode.utility));
        best = std::max(best, value);
    }
    for (const RouteMemoryRecord& record : route_memory_) {
        const double resonance = signature.similarity(
            record.start_goal_signature
        );
        if (similarity_evaluations != nullptr) {
            ++(*similarity_evaluations);
        }
        const double value = resonance * record.confidence +
            (0.05 * std::max(0.0, record.utility));
        best = std::max(best, value);
    }
    return clamp_finite(best, 0.0, 1.0);
}

std::optional<std::uint64_t> LatentRouter::lookahead_action(
    const PhaseVector& current,
    const PhaseVector& goal,
    const std::size_t depth,
    const std::size_t beam_width,
    std::size_t* const search_nodes,
    std::size_t* const similarity_evaluations
) {
    if (depth == 0U || beam_width == 0U || operators_.empty()) {
        return std::nullopt;
    }
    struct SearchNode final {
        PhaseVector state;
        std::vector<std::uint64_t> route;
        double score{};
        double cumulative_action_score{};
    };

    std::vector<SearchNode> beam;
    beam.push_back({
        .state = current,
        .route = {},
        .score = state_familiarity(
            current,
            goal,
            similarity_evaluations
        ),
        .cumulative_action_score = 0.0,
    });
    std::optional<SearchNode> best_goal;

    for (std::size_t level = 0U; level < depth; ++level) {
        std::vector<SearchNode> expanded;
        expanded.reserve(beam.size() * operators_.size());
        for (const SearchNode& node : beam) {
            std::vector<LatentActionCandidate> candidates = score_actions(
                node.state,
                goal,
                false,
                false,
                similarity_evaluations
            );
            const std::size_t branch_count = std::min(
                candidates.size(),
                std::max<std::size_t>(2U, beam_width)
            );
            for (std::size_t index = 0U; index < branch_count; ++index) {
                const LatentActionCandidate& candidate = candidates[index];
                const RegisteredOperator& operation = operator_by_id(
                    candidate.operator_id
                );
                PhaseVector next = operation.transformation.apply(node.state);
                std::vector<std::uint64_t> route = node.route;
                route.push_back(candidate.operator_id);
                if (search_nodes != nullptr) {
                    ++(*search_nodes);
                }
                const double goal_similarity = next.similarity(goal);
                const double familiarity = state_familiarity(
                    next,
                    goal,
                    similarity_evaluations
                );
                const double cumulative = node.cumulative_action_score +
                    candidate.score;
                const double normalized_cumulative = cumulative /
                    static_cast<double>(route.size());
                const double score =
                    (1.40 * familiarity) +
                    (0.45 * goal_similarity) +
                    (0.15 * normalized_cumulative) -
                    (0.002 * static_cast<double>(route.size()));
                SearchNode child{
                    .state = std::move(next),
                    .route = std::move(route),
                    .score = score,
                    .cumulative_action_score = cumulative,
                };
                if (is_goal(child.state, goal)) {
                    if (!best_goal.has_value() ||
                        child.route.size() < best_goal->route.size() ||
                        (child.route.size() == best_goal->route.size() &&
                         child.route < best_goal->route)) {
                        best_goal = child;
                    }
                }
                expanded.push_back(std::move(child));
            }
        }
        if (best_goal.has_value()) {
            return best_goal->route.front();
        }
        std::sort(
            expanded.begin(),
            expanded.end(),
            [](const SearchNode& left, const SearchNode& right) {
                if (std::abs(left.score - right.score) > 1.0e-12) {
                    return left.score > right.score;
                }
                return left.route < right.route;
            }
        );
        if (expanded.size() > beam_width) {
            expanded.erase(
                expanded.begin() + static_cast<std::ptrdiff_t>(beam_width),
                expanded.end()
            );
        }
        beam = std::move(expanded);
        if (beam.empty()) {
            break;
        }
    }
    if (beam.empty() || beam.front().route.empty()) {
        return std::nullopt;
    }
    return beam.front().route.front();
}

double LatentRouter::candidate_uncertainty(
    std::vector<LatentActionCandidate>& candidates
) const {
    if (candidates.empty()) {
        return 1.0;
    }
    const double maximum = candidates.front().score;
    double total = 0.0;
    for (LatentActionCandidate& candidate : candidates) {
        candidate.normalized_weight = std::exp(
            (candidate.score - maximum) / config_.action_temperature
        );
        total += candidate.normalized_weight;
    }
    double entropy = 0.0;
    for (LatentActionCandidate& candidate : candidates) {
        candidate.normalized_weight /= total;
        if (candidate.normalized_weight > 0.0) {
            entropy -= candidate.normalized_weight *
                std::log(candidate.normalized_weight);
        }
    }
    const double maximum_entropy = candidates.size() <= 1U
        ? 1.0
        : std::log(static_cast<double>(candidates.size()));
    return candidates.size() <= 1U
        ? 0.0
        : clamp_finite(entropy / maximum_entropy, 0.0, 1.0);
}

double LatentRouter::halt_score(
    const PhaseVector& signature,
    std::size_t* const similarity_evaluations
) const {
    double best = 0.0;
    std::uint64_t best_id = std::numeric_limits<std::uint64_t>::max();
    for (const LatentHaltMode& mode : halt_modes_) {
        const double resonance = signature.similarity(mode.key);
        if (similarity_evaluations != nullptr) {
            ++(*similarity_evaluations);
        }
        const double score = resonance * mode.confidence + mode.utility;
        if (score > best + 1.0e-12 ||
            (std::abs(score - best) <= 1.0e-12 && mode.id < best_id)) {
            best = score;
            best_id = mode.id;
        }
    }
    return clamp_finite(best, 0.0, 1.0);
}

LatentExecutionResult LatentRouter::execute(
    const PhaseVector& start,
    const PhaseVector& goal,
    const std::optional<std::uint64_t> forced_first_operator,
    const bool allow_route_memory
) {
    return execute_internal(
        start,
        goal,
        forced_first_operator,
        allow_route_memory,
        false,
        1U,
        1U
    );
}

LatentExecutionResult LatentRouter::execute_with_bounded_lookahead(
    const PhaseVector& start,
    const PhaseVector& goal,
    const std::size_t lookahead_depth,
    const std::size_t beam_width,
    const bool allow_route_memory
) {
    if (lookahead_depth == 0U || beam_width == 0U) {
        throw std::invalid_argument(
            "latent lookahead depth and beam width must be positive"
        );
    }
    return execute_internal(
        start,
        goal,
        std::nullopt,
        allow_route_memory,
        true,
        lookahead_depth,
        beam_width
    );
}

LatentExecutionResult LatentRouter::execute_internal(
    const PhaseVector& start,
    const PhaseVector& goal,
    const std::optional<std::uint64_t> forced_first_operator,
    const bool allow_route_memory,
    const bool use_lookahead,
    const std::size_t lookahead_depth,
    const std::size_t beam_width
) {
    if (start.size() != config_.dimension || goal.size() != config_.dimension) {
        throw std::invalid_argument("latent execution dimension mismatch");
    }
    LatentExecutionResult result{};
    result.final_state = start;
    PhaseVector state = start;
    PhaseVector memory_summary = PhaseVector::zeros(config_.dimension);
    PhaseVector route_summary = PhaseVector::zeros(config_.dimension);
    std::unordered_set<std::uint64_t> visited;
    visited.insert(phase_state_hash(state));
    double uncertainty_total = 0.0;

    for (std::size_t cycle = 0U; cycle < config_.maximum_cycles; ++cycle) {
        const double similarity_before = state.similarity(goal);
        const PhaseVector signature = state_goal_signature(state, goal);
        if (is_goal(state, goal)) {
            result.stop_reason = LatentStopReason::successful_halt;
            result.success = true;
            result.cycles = cycle;
            result.final_state = state;
            result.final_goal_similarity = similarity_before;
            result.mean_uncertainty = cycle == 0U
                ? 0.0
                : uncertainty_total / static_cast<double>(cycle);
            return result;
        }

        const double learned_halt = halt_score(
            signature,
            &result.exact_similarity_evaluations
        );
        const bool learned_requests_halt =
            learned_halt >= config_.learned_halt_threshold;
        if (config_.halt_policy == LatentHaltPolicy::learned_resonant &&
            learned_requests_halt) {
            result.stop_reason = LatentStopReason::premature_halt;
            result.success = false;
            result.cycles = cycle;
            result.final_state = state;
            result.final_goal_similarity = similarity_before;
            result.mean_uncertainty = cycle == 0U
                ? 0.0
                : uncertainty_total / static_cast<double>(cycle);
            return result;
        }
        if (config_.halt_policy == LatentHaltPolicy::combined_safe &&
            learned_requests_halt &&
            similarity_before >= config_.learned_halt_goal_floor) {
            result.stop_reason = LatentStopReason::learned_halt;
            result.success = is_goal(state, goal);
            result.cycles = cycle;
            result.final_state = state;
            result.final_goal_similarity = similarity_before;
            result.mean_uncertainty = cycle == 0U
                ? 0.0
                : uncertainty_total / static_cast<double>(cycle);
            return result;
        }

        std::vector<LatentActionCandidate> candidates = score_actions(
            state,
            goal,
            allow_route_memory,
            true,
            &result.exact_similarity_evaluations
        );
        if (candidates.empty()) {
            result.stop_reason = LatentStopReason::no_candidate;
            result.cycles = cycle;
            result.final_state = state;
            result.final_goal_similarity = similarity_before;
            return result;
        }
        for (LatentActionCandidate& candidate : candidates) {
            const PhaseVector proposal = operator_by_id(candidate.operator_id)
                .transformation.apply(state);
            if (visited.contains(phase_state_hash(proposal))) {
                candidate.score -= config_.route_repetition_penalty;
            }
        }
        std::sort(
            candidates.begin(),
            candidates.end(),
            [](const LatentActionCandidate& left,
               const LatentActionCandidate& right) {
                if (std::abs(left.score - right.score) > 1.0e-12) {
                    return left.score > right.score;
                }
                return left.operator_id < right.operator_id;
            }
        );
        const double uncertainty = candidate_uncertainty(candidates);
        uncertainty_total += uncertainty;
        memory_summary = candidate_memory_summary(candidates);

        std::uint64_t selected_operator_id = candidates.front().operator_id;
        if (cycle == 0U && forced_first_operator.has_value()) {
            static_cast<void>(operator_by_id(*forced_first_operator));
            selected_operator_id = *forced_first_operator;
        } else if (use_lookahead) {
            const std::optional<std::uint64_t> searched = lookahead_action(
                state,
                goal,
                lookahead_depth,
                beam_width,
                &result.search_nodes,
                &result.exact_similarity_evaluations
            );
            if (searched.has_value()) {
                selected_operator_id = *searched;
            }
        }
        const auto selected_candidate = std::find_if(
            candidates.begin(),
            candidates.end(),
            [selected_operator_id](const LatentActionCandidate& value) {
                return value.operator_id == selected_operator_id;
            }
        );
        const std::uint64_t selected_mode_id =
            selected_candidate == candidates.end()
            ? 0ULL
            : selected_candidate->routing_mode_id;
        const double maximum_resonance = std::max(
            candidates.front().resonance,
            candidates.front().successor_familiarity
        );
        if (!use_lookahead &&
            uncertainty >= config_.abstention_entropy_threshold &&
            candidates.front().score < config_.minimum_action_score &&
            maximum_resonance < config_.abstention_resonance_threshold) {
            result.stop_reason = LatentStopReason::abstained;
            result.abstained = true;
            result.cycles = cycle;
            result.final_state = state;
            result.final_goal_similarity = similarity_before;
            result.mean_uncertainty = uncertainty_total /
                static_cast<double>(cycle + 1U);
            return result;
        }

        const RegisteredOperator& selected =
            operator_by_id(selected_operator_id);
        PhaseVector next_state = selected.transformation.apply(state);
        const double similarity_after = next_state.similarity(goal);
        const double progress = similarity_after - similarity_before -
            (0.001 * selected.cost);

        const LatentEpisodeState episode_state{
            .current_state = start,
            .goal_state = goal,
            .working_state = state,
            .memory_summary = memory_summary,
            .route_summary = route_summary,
            .uncertainty_state = uncertainty,
            .progress_state = progress,
            .step_index = cycle,
        };
        result.trace.push_back({
            .cycle = cycle,
            .state_hash = episode_state_hash(episode_state),
            .working_state_hash = phase_state_hash(state),
            .goal_state_hash = phase_state_hash(goal),
            .memory_summary_hash = phase_state_hash(memory_summary),
            .route_summary_hash = phase_state_hash(route_summary),
            .goal_similarity_before = similarity_before,
            .goal_similarity_after = similarity_after,
            .progress = progress,
            .uncertainty = uncertainty,
            .halt_score = learned_halt,
            .selected_operator_id = selected_operator_id,
            .selected_operator_name = selected.name,
            .selected_mode_id = selected_mode_id,
            .candidates = candidates,
        });
        result.route.push_back(selected_operator_id);
        route_summary = route_summary.composed(
            operator_route_code(config_.dimension, selected_operator_id)
        );
        ++result.active_mode_evaluations;
        if (selected_mode_id != 0ULL) {
            const auto mode = std::find_if(
                modes_.begin(),
                modes_.end(),
                [selected_mode_id](const LatentRoutingMode& value) {
                    return value.id == selected_mode_id;
                }
            );
            if (mode != modes_.end()) {
                ++mode->activation_count;
                mode->last_used_step = training_step_;
            }
        }

        const std::uint64_t next_hash = phase_state_hash(next_state);
        if (visited.contains(next_hash)) {
            result.stop_reason = LatentStopReason::loop_detected;
            result.cycles = cycle + 1U;
            result.final_state = std::move(next_state);
            result.final_goal_similarity = similarity_after;
            result.mean_uncertainty = uncertainty_total /
                static_cast<double>(cycle + 1U);
            return result;
        }
        visited.insert(next_hash);
        state = std::move(next_state);
    }

    result.stop_reason = LatentStopReason::cycle_limit;
    result.cycles = config_.maximum_cycles;
    result.final_state = std::move(state);
    result.final_goal_similarity = result.final_state.similarity(goal);
    result.success = is_goal(result.final_state, goal);
    result.mean_uncertainty = uncertainty_total /
        static_cast<double>(config_.maximum_cycles);
    return result;
}

std::optional<std::vector<std::uint64_t>> LatentRouter::discover_route(
    const PhaseVector& start,
    const PhaseVector& goal,
    const std::size_t maximum_depth,
    std::size_t* const explored_nodes
) const {
    if (start.size() != config_.dimension || goal.size() != config_.dimension) {
        throw std::invalid_argument("latent search dimension mismatch");
    }
    if (operators_.empty()) {
        return std::nullopt;
    }
    struct Node final {
        PhaseVector state;
        std::vector<std::uint64_t> route;
    };
    std::deque<Node> frontier;
    frontier.push_back({start, {}});
    std::unordered_set<std::uint64_t> visited;
    visited.insert(phase_state_hash(start));
    std::size_t explored = 0U;

    while (!frontier.empty() && explored < config_.search_node_budget) {
        Node node = std::move(frontier.front());
        frontier.pop_front();
        ++explored;
        if (node.state.similarity(goal) >=
            config_.goal_similarity_threshold) {
            if (explored_nodes != nullptr) {
                *explored_nodes = explored;
            }
            return node.route;
        }
        if (node.route.size() >= maximum_depth) {
            continue;
        }
        for (const RegisteredOperator& operator_value : operators_) {
            if (operator_value.macro) {
                continue;
            }
            PhaseVector next = operator_value.transformation.apply(node.state);
            const std::uint64_t hash = phase_state_hash(next);
            if (visited.contains(hash)) {
                continue;
            }
            visited.insert(hash);
            std::vector<std::uint64_t> route = node.route;
            route.push_back(operator_value.id);
            frontier.push_back({std::move(next), std::move(route)});
        }
    }
    if (explored_nodes != nullptr) {
        *explored_nodes = explored;
    }
    return std::nullopt;
}

std::size_t LatentRouter::create_or_update_mode(
    const PhaseVector& signature,
    const std::uint64_t operator_id,
    const double local_progress,
    const double terminal_credit,
    std::size_t* const created_count
) {
    ModeSelection selected = best_mode_for_operator(
        signature,
        operator_id,
        nullptr
    );
    if (!selected.found ||
        selected.resonance < config_.mode_creation_similarity) {
        if (modes_.size() >= config_.maximum_modes) {
            if (!selected.found) {
                throw std::runtime_error(
                    "latent routing mode budget exhausted"
                );
            }
        } else {
            modes_.push_back({
                .id = next_mode_id_++,
                .key = signature,
                .operator_id = operator_id,
                .utility = 0.0,
                .confidence = 0.25,
                .eligibility = 0.0,
                .activation_count = 0ULL,
                .success_count = 0ULL,
                .failure_count = 0ULL,
                .creation_step = training_step_,
                .last_used_step = training_step_,
            });
            selected = {
                .mode_index = modes_.size() - 1U,
                .resonance = 1.0,
                .found = true,
            };
            if (created_count != nullptr) {
                ++(*created_count);
            }
        }
    }
    LatentRoutingMode& mode = modes_[selected.mode_index];
    mode.key = circular_interpolate(
        mode.key,
        signature,
        config_.mode_learning_rate
    );
    const double evidence = clamp_finite(
        0.5 + (0.25 * local_progress) + (0.25 * terminal_credit),
        0.0,
        1.0
    );
    mode.confidence += config_.mode_learning_rate *
        (evidence - mode.confidence);
    mode.last_used_step = training_step_;
    if (terminal_credit > 0.0) {
        ++mode.success_count;
    } else if (terminal_credit < 0.0) {
        ++mode.failure_count;
    }
    return selected.mode_index;
}

void LatentRouter::create_or_update_halt_mode(
    const PhaseVector& signature
) {
    std::size_t best_index = 0U;
    double best_similarity = -1.0;
    for (std::size_t index = 0U; index < halt_modes_.size(); ++index) {
        const double similarity = signature.similarity(halt_modes_[index].key);
        if (similarity > best_similarity) {
            best_similarity = similarity;
            best_index = index;
        }
    }
    if (halt_modes_.empty() || best_similarity < config_.mode_creation_similarity) {
        halt_modes_.push_back({
            .id = next_halt_mode_id_++,
            .key = signature,
            .confidence = 0.75,
            .utility = 0.0,
            .activation_count = 0ULL,
        });
        return;
    }
    LatentHaltMode& mode = halt_modes_[best_index];
    mode.key = circular_interpolate(
        mode.key,
        signature,
        config_.mode_learning_rate
    );
    mode.confidence += config_.mode_learning_rate *
        (1.0 - mode.confidence);
}

void LatentRouter::remember_route(
    const PhaseVector& start,
    const PhaseVector& goal,
    const std::span<const std::uint64_t> route,
    const double reward
) {
    if (!config_.enable_route_memory || route.empty()) {
        return;
    }
    const PhaseVector signature = state_goal_signature(start, goal);
    const std::uint64_t hash = route_hash(route);
    for (RouteMemoryRecord& record : route_memory_) {
        if (record.route_hash == hash && record.route.size() == route.size() &&
            std::equal(record.route.begin(), record.route.end(), route.begin())) {
            record.confidence += config_.mode_learning_rate *
                ((reward > 0.0 ? 1.0 : 0.0) - record.confidence);
            record.utility += config_.utility_learning_rate * reward;
            record.last_used_step = training_step_;
            ++record.observation_count;
            ++route_occurrences_[hash];
            return;
        }
    }
    if (route_memory_.size() >= config_.route_memory_capacity) {
        const auto worst = std::min_element(
            route_memory_.begin(),
            route_memory_.end(),
            [](const RouteMemoryRecord& left,
               const RouteMemoryRecord& right) {
                if (left.utility != right.utility) {
                    return left.utility < right.utility;
                }
                if (left.usage_count != right.usage_count) {
                    return left.usage_count < right.usage_count;
                }
                return left.id < right.id;
            }
        );
        route_memory_.erase(worst);
    }
    route_memory_.push_back({
        .id = next_route_record_id_++,
        .start_goal_signature = signature,
        .route = std::vector<std::uint64_t>(route.begin(), route.end()),
        .confidence = reward > 0.0 ? 0.75 : 0.25,
        .utility = reward,
        .usage_count = 0ULL,
        .observation_count = 1ULL,
        .creation_step = training_step_,
        .last_used_step = training_step_,
        .route_hash = hash,
    });
    ++route_occurrences_[hash];
}

void LatentRouter::reinforce_route(
    const PhaseVector& start,
    const PhaseVector& goal,
    const std::span<const std::uint64_t> route,
    const double terminal_reward
) {
    if (start.size() != config_.dimension || goal.size() != config_.dimension) {
        throw std::invalid_argument("latent reinforcement dimension mismatch");
    }
    ++training_step_;
    PhaseVector state = start;
    std::vector<std::size_t> mode_indices;
    std::vector<double> progress_values;
    mode_indices.reserve(route.size());
    progress_values.reserve(route.size());
    std::size_t created = 0U;

    for (const std::uint64_t operator_id : route) {
        const RegisteredOperator& operator_value = operator_by_id(operator_id);
        const PhaseVector signature = state_goal_signature(state, goal);
        const PhaseVector next = operator_value.transformation.apply(state);
        const double progress = immediate_progress(
            state,
            next,
            goal,
            operator_value.cost
        );
        const std::size_t mode_index = create_or_update_mode(
            signature,
            operator_id,
            progress,
            terminal_reward,
            &created
        );
        mode_indices.push_back(mode_index);
        progress_values.push_back(progress);
        state = next;
    }

    for (LatentRoutingMode& mode : modes_) {
        mode.eligibility = 0.0;
    }
    for (std::size_t step = 0U; step < mode_indices.size(); ++step) {
        for (LatentRoutingMode& mode : modes_) {
            mode.eligibility *= config_.eligibility_decay;
        }
        modes_[mode_indices[step]].eligibility += 1.0;
    }

    const double route_length = route.empty()
        ? 1.0
        : static_cast<double>(route.size());
    for (std::size_t step = 0U; step < mode_indices.size(); ++step) {
        LatentRoutingMode& mode = modes_[mode_indices[step]];
        double credit = terminal_reward;
        switch (config_.credit_strategy) {
        case LatentCreditStrategy::uniform_route:
            credit = terminal_reward / route_length;
            break;
        case LatentCreditStrategy::discounted_eligibility:
            credit = terminal_reward * mode.eligibility;
            break;
        case LatentCreditStrategy::progress_weighted:
            credit = terminal_reward *
                (0.5 + std::clamp(progress_values[step], -0.5, 0.5));
            break;
        case LatentCreditStrategy::counterfactual_local: {
            const double best_progress = *std::max_element(
                progress_values.begin(),
                progress_values.end()
            );
            credit = terminal_reward *
                (0.5 + 0.5 * (progress_values[step] >= best_progress - 1.0e-12
                    ? 1.0
                    : -0.25));
            break;
        }
        }
        mode.utility = clamp_finite(
            mode.utility + config_.utility_learning_rate * credit,
            -1.0,
            1.0
        );
    }

    if (terminal_reward > 0.0 && is_goal(state, goal)) {
        create_or_update_halt_mode(state_goal_signature(state, goal));
        remember_route(start, goal, route, terminal_reward);
    }
}

LatentTrainingResult LatentRouter::train_episode(
    const PhaseVector& start,
    const PhaseVector& goal,
    const std::size_t maximum_search_depth
) {
    std::size_t explored = 0U;
    const std::optional<std::vector<std::uint64_t>> route = discover_route(
        start,
        goal,
        maximum_search_depth,
        &explored
    );
    if (!route.has_value()) {
        ++training_step_;
        return {
            .route_found = false,
            .success = false,
            .discovered_route = {},
            .search_nodes = explored,
            .terminal_reward = -1.0,
            .modes_created = 0U,
            .modes_updated = 0U,
        };
    }
    const std::size_t before = modes_.size();
    reinforce_route(start, goal, *route, 1.0);
    if (config_.enable_macro_operators) {
        static_cast<void>(consolidate_macros());
    }
    return {
        .route_found = true,
        .success = true,
        .discovered_route = *route,
        .search_nodes = explored,
        .terminal_reward = 1.0,
        .modes_created = modes_.size() - before,
        .modes_updated = route->size(),
    };
}

std::size_t LatentRouter::consolidate_macros() {
    if (!config_.enable_macro_operators) {
        return 0U;
    }
    std::size_t created = 0U;
    for (const RouteMemoryRecord& record : route_memory_) {
        if (record.route.size() < 2U ||
            record.route.size() > config_.macro_maximum_length) {
            continue;
        }
        const auto count = route_occurrences_.find(record.route_hash);
        const std::size_t observations = count == route_occurrences_.end()
            ? static_cast<std::size_t>(record.observation_count)
            : std::max(
                  count->second,
                  static_cast<std::size_t>(record.observation_count)
              );
        if (observations < config_.macro_minimum_occurrences) {
            continue;
        }
        const bool already_exists = std::any_of(
            operators_.begin(),
            operators_.end(),
            [&record](const RegisteredOperator& value) {
                return value.macro && value.primitive_route == record.route;
            }
        );
        if (already_exists) {
            continue;
        }
        TransformationOperator combined =
            operator_by_id(record.route.front()).transformation;
        double primitive_cost = operator_by_id(record.route.front()).cost;
        for (std::size_t index = 1U; index < record.route.size(); ++index) {
            const RegisteredOperator& next = operator_by_id(record.route[index]);
            combined = combined.then(next.transformation);
            primitive_cost += next.cost;
        }
        const std::string name = "macro_" + std::to_string(record.route_hash);
        static_cast<void>(register_operator(
            name,
            std::move(combined),
            std::max(1.0, primitive_cost * 0.35),
            true,
            record.route
        ));
        ++created;
    }
    return created;
}

LatentRouterSnapshot LatentRouter::snapshot() const {
    return {
        .config = config_,
        .seed = seed_,
        .training_step = training_step_,
        .next_operator_id = next_operator_id_,
        .next_mode_id = next_mode_id_,
        .next_halt_mode_id = next_halt_mode_id_,
        .next_route_record_id = next_route_record_id_,
        .operators = operators_,
        .modes = modes_,
        .halt_modes = halt_modes_,
        .route_memory = route_memory_,
    };
}

LatentRouter LatentRouter::from_snapshot(LatentRouterSnapshot snapshot) {
    LatentRouter router(snapshot.config, snapshot.seed);
    router.training_step_ = snapshot.training_step;
    router.next_operator_id_ = snapshot.next_operator_id;
    router.next_mode_id_ = snapshot.next_mode_id;
    router.next_halt_mode_id_ = snapshot.next_halt_mode_id;
    router.next_route_record_id_ = snapshot.next_route_record_id;
    router.operators_ = std::move(snapshot.operators);
    router.modes_ = std::move(snapshot.modes);
    router.halt_modes_ = std::move(snapshot.halt_modes);
    router.route_memory_ = std::move(snapshot.route_memory);
    for (const RouteMemoryRecord& record : router.route_memory_) {
        router.route_occurrences_[record.route_hash] +=
            static_cast<std::size_t>(record.observation_count);
    }
    return router;
}

}  // namespace rlf::core
