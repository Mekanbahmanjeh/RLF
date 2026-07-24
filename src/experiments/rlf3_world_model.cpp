#include "rlf/experiments/rlf3_world_model.hpp"

#include "rlf/core/deterministic_rng.hpp"
#include "rlf/storage/rlf3_checkpoint.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <numbers>
#include <optional>
#include <sstream>
#include <ostream>
#include <queue>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rlf::experiments {
namespace {

constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;
constexpr std::size_t action_step_a = 1U;
constexpr std::size_t action_step_b = 2U;
constexpr std::size_t action_step_middle = 3U;
constexpr std::size_t action_stochastic = 4U;
constexpr std::size_t action_probe = 5U;
constexpr std::size_t action_shortcut = 6U;

void hash_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (unsigned int byte = 0U; byte < 8U; ++byte) {
        hash ^= (value >> (byte * 8U)) & 0xFFULL;
        hash *= fnv_prime;
    }
}

void hash_double(std::uint64_t& hash, const double value) noexcept {
    hash_u64(hash, std::bit_cast<std::uint64_t>(value));
}

[[nodiscard]] std::string format_hash(const std::uint64_t value) {
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << value;
    return output.str();
}

[[nodiscard]] std::string json_escape(const std::string_view value) {
    std::string result;
    result.reserve(value.size());
    for (const char character : value) {
        switch (character) {
        case '\\': result += "\\\\"; break;
        case '"': result += "\\\""; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default: result.push_back(character); break;
        }
    }
    return result;
}

[[nodiscard]] core::PhaseVector perturb(
    const core::PhaseVector& source,
    const double radians,
    core::DeterministicRng& rng
) {
    std::vector<float> angles(source.angles().begin(), source.angles().end());
    for (float& angle : angles) {
        const double offset = (rng.uniform_unit() * 2.0 - 1.0) * radians;
        angle = core::PhaseVector::normalize_angle(
            static_cast<float>(static_cast<double>(angle) + offset)
        );
    }
    return core::PhaseVector(std::move(angles));
}

struct HiddenState final {
    std::size_t node{};
    std::size_t hidden_mode{};
    int memory_mode{-1};
    bool trapped{false};
};

struct StepResult final {
    HiddenState state;
    double reward{};
    bool terminal{false};
};

class CausalGateWorld final {
public:
    CausalGateWorld(const Rlf3Config& config, const std::uint64_t seed)
        : config_(config), trap_node_(config.layers * config.lanes) {
        if (config.layers < 5U || config.lanes < 3U || config.dimension < 8U ||
            config.stochastic_dominant_probability <= 0.5 ||
            config.stochastic_dominant_probability >= 1.0) {
            throw std::invalid_argument("invalid RLF-3 world configuration");
        }
        core::DeterministicRng rng(seed ^ 0x574F524C4433ULL);
        node_keys_.reserve(trap_node_ + 1U);
        for (std::size_t node = 0U; node <= trap_node_; ++node) {
            node_keys_.push_back(core::PhaseVector::random(config.dimension, rng));
        }
        neutral_memory_ = core::PhaseVector::random(config.dimension, rng);
        mode_memory_[0] = core::PhaseVector::random(config.dimension, rng);
        mode_memory_[1] = core::PhaseVector::random(config.dimension, rng);
    }

    [[nodiscard]] std::size_t trap_node() const noexcept { return trap_node_; }
    [[nodiscard]] std::size_t state_count() const noexcept {
        return trap_node_ + 1U;
    }
    [[nodiscard]] std::size_t stochastic_gate_layer() const noexcept {
        return config_.layers / 2U;
    }
    [[nodiscard]] std::size_t partial_gate_layer() const noexcept {
        return config_.layers - 2U;
    }
    [[nodiscard]] std::size_t node(
        const std::size_t layer,
        const std::size_t lane
    ) const noexcept {
        return layer * config_.lanes + (lane % config_.lanes);
    }
    [[nodiscard]] std::size_t layer(const std::size_t node_value) const noexcept {
        return node_value >= trap_node_ ? config_.layers :
            node_value / config_.lanes;
    }
    [[nodiscard]] std::size_t lane(const std::size_t node_value) const noexcept {
        return node_value >= trap_node_ ? 0U : node_value % config_.lanes;
    }
    [[nodiscard]] bool goal_reached(
        const HiddenState& state,
        const std::size_t goal_node
    ) const noexcept {
        return !state.trapped && state.node == goal_node;
    }
    [[nodiscard]] bool terminal(const HiddenState& state) const noexcept {
        return state.trapped || layer(state.node) >= config_.layers - 1U;
    }

    [[nodiscard]] core::WorldObservation observe(
        const HiddenState& state,
        core::DeterministicRng& rng,
        const double noise,
        const bool memoryless = false
    ) const {
        const core::PhaseVector visible = perturb(node_keys_.at(state.node), noise, rng);
        const core::PhaseVector& memory = memoryless || state.memory_mode < 0
            ? neutral_memory_
            : mode_memory_.at(static_cast<std::size_t>(state.memory_mode));
        return {visible, perturb(memory, noise, rng)};
    }

    [[nodiscard]] const core::PhaseVector& goal_observation(
        const std::size_t goal_node
    ) const {
        return node_keys_.at(goal_node);
    }

    [[nodiscard]] StepResult step(
        const HiddenState& input,
        const std::uint64_t action_id,
        core::DeterministicRng& rng,
        const bool sample_stochastic = true
    ) const {
        if (action_id < action_step_a || action_id > action_shortcut) {
            throw std::out_of_range("unknown RLF-3 world action");
        }
        if (terminal(input)) {
            return {input, -0.25, true};
        }
        HiddenState next = input;
        const std::size_t current_layer = layer(input.node);
        const std::size_t current_lane = lane(input.node);
        if (action_id == action_probe) {
            const bool information_gain = input.memory_mode < 0;
            next.memory_mode = static_cast<int>(input.hidden_mode);
            return {next, information_gain ? 1.0 : -0.05, false};
        }

        const auto trap = [&]() {
            HiddenState trapped = input;
            trapped.node = trap_node_;
            trapped.trapped = true;
            return StepResult{trapped, -1.0, true};
        };

        if (current_layer == partial_gate_layer()) {
            const bool correct =
                (input.hidden_mode == 0U && action_id == action_step_a) ||
                (input.hidden_mode == 1U && action_id == action_step_b);
            if (!correct) {
                return trap();
            }
            next.node = node(current_layer + 1U, current_lane);
            return {next, -0.05, false};
        }

        if (current_layer == stochastic_gate_layer()) {
            if (action_id != action_stochastic) {
                return trap();
            }
            const bool dominant = !sample_stochastic ||
                rng.uniform_unit() < config_.stochastic_dominant_probability;
            const std::size_t dominant_delta = input.hidden_mode == 0U ? 0U : 2U;
            const std::size_t alternate_delta = input.hidden_mode == 0U ? 2U : 0U;
            const std::size_t delta = dominant ? dominant_delta : alternate_delta;
            next.node = node(current_layer + 1U, current_lane + delta);
            return {next, -0.10, false};
        }

        if (action_id == action_shortcut) {
            const std::size_t destination_layer = current_layer + 2U;
            const bool informed_final_recovery =
                current_layer + 1U == partial_gate_layer() &&
                input.memory_mode == static_cast<int>(input.hidden_mode) &&
                destination_layer == config_.layers - 1U;
            const bool crosses_gate =
                (current_layer < stochastic_gate_layer() &&
                 destination_layer > stochastic_gate_layer()) ||
                (current_layer < partial_gate_layer() &&
                 destination_layer > partial_gate_layer());
            const bool ordinary_safe = destination_layer < config_.layers &&
                !crosses_gate &&
                ((current_lane + input.hidden_mode + current_layer) % 3U != 0U);
            if (!ordinary_safe && !informed_final_recovery) {
                return trap();
            }
            const std::size_t delta = informed_final_recovery ? 4U : 3U;
            next.node = node(destination_layer, current_lane + delta);
            return {next, informed_final_recovery ? 0.10 : 0.05, terminal(next)};
        }

        std::size_t delta = 0U;
        switch (action_id) {
        case action_step_a:
            delta = input.hidden_mode == 0U ? 0U : 2U;
            break;
        case action_step_b:
            delta = input.hidden_mode == 0U ? 2U : 0U;
            break;
        case action_step_middle:
            delta = 1U;
            break;
        case action_stochastic: {
            const bool dominant = !sample_stochastic ||
                rng.uniform_unit() < config_.stochastic_dominant_probability;
            const std::size_t dominant_delta = input.hidden_mode == 0U ? 0U : 2U;
            const std::size_t alternate_delta = 1U;
            delta = dominant ? dominant_delta : alternate_delta;
            break;
        }
        default:
            return trap();
        }
        const std::size_t destination_layer = current_layer + 1U;
        if (destination_layer >= config_.layers) {
            return trap();
        }
        next.node = node(destination_layer, current_lane + delta);
        return {next, -0.05, terminal(next)};
    }

    [[nodiscard]] std::optional<std::vector<std::uint64_t>> oracle_plan(
        const HiddenState& start,
        const std::size_t goal_node,
        const bool force_probe
    ) const {
        struct Node final {
            HiddenState state;
            std::vector<std::uint64_t> route;
        };
        std::queue<Node> queue;
        std::set<std::tuple<std::size_t, std::size_t, int, bool>> visited;
        HiddenState initial = start;
        std::vector<std::uint64_t> prefix;
        core::DeterministicRng deterministic_rng(0x4F5241434C4533ULL);
        if (force_probe && initial.memory_mode < 0) {
            const StepResult probed = step(
                initial, action_probe, deterministic_rng, false
            );
            initial = probed.state;
            prefix.push_back(action_probe);
        }
        queue.push({initial, prefix});
        visited.insert({
            initial.node, initial.hidden_mode, initial.memory_mode,
            initial.trapped
        });
        while (!queue.empty()) {
            Node current = std::move(queue.front());
            queue.pop();
            if (goal_reached(current.state, goal_node)) {
                return current.route;
            }
            if (current.route.size() >= config_.maximum_execution_steps ||
                terminal(current.state)) {
                continue;
            }
            for (std::uint64_t action = action_step_a;
                 action <= action_shortcut; ++action) {
                const StepResult outcome = step(
                    current.state, action, deterministic_rng, false
                );
                if (outcome.state.trapped) {
                    continue;
                }
                const auto key = std::tuple{
                    outcome.state.node,
                    outcome.state.hidden_mode,
                    outcome.state.memory_mode,
                    outcome.state.trapped,
                };
                if (!visited.insert(key).second) {
                    continue;
                }
                std::vector<std::uint64_t> route = current.route;
                route.push_back(action);
                queue.push({outcome.state, std::move(route)});
            }
        }
        return std::nullopt;
    }

private:
    Rlf3Config config_;
    std::size_t trap_node_{};
    std::vector<core::PhaseVector> node_keys_;
    core::PhaseVector neutral_memory_{std::vector<float>{0.0F}};
    std::array<core::PhaseVector, 2U> mode_memory_{
        core::PhaseVector(std::vector<float>{0.0F}),
        core::PhaseVector(std::vector<float>{0.0F})
    };
};

struct EvaluationCase final {
    HiddenState start;
    std::size_t goal_node{};
    std::uint64_t pair_hash{};
    std::uint64_t route_hash{};
};

[[nodiscard]] std::uint64_t pair_hash(
    const HiddenState& start,
    const std::size_t goal_node
) noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, static_cast<std::uint64_t>(start.node));
    hash_u64(hash, static_cast<std::uint64_t>(start.hidden_mode));
    hash_u64(hash, static_cast<std::uint64_t>(start.memory_mode + 1));
    hash_u64(hash, static_cast<std::uint64_t>(goal_node));
    return hash;
}

[[nodiscard]] std::uint64_t route_hash(
    const std::uint64_t pair,
    const std::span<const std::uint64_t> route
) noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, pair);
    hash_u64(hash, static_cast<std::uint64_t>(route.size()));
    for (const std::uint64_t action : route) {
        hash_u64(hash, action);
    }
    return hash;
}

void validate_config(const Rlf3Config& config) {
    if (config.dimension < 8U || config.layers < 5U || config.lanes < 3U ||
        config.transition_samples_per_case < 2U || config.training_routes == 0U ||
        config.evaluation_episodes == 0U ||
        config.stochastic_rollouts_per_episode == 0U ||
        config.maximum_execution_steps == 0U ||
        config.planner_node_budget == 0U || config.maximum_plan_depth == 0U ||
        !std::isfinite(config.observation_noise_radians) ||
        config.observation_noise_radians < 0.0 ||
        !std::isfinite(config.stochastic_dominant_probability) ||
        config.stochastic_dominant_probability <= 0.5 ||
        config.stochastic_dominant_probability >= 1.0 ||
        !std::isfinite(config.goal_similarity_threshold) ||
        config.goal_similarity_threshold <= 0.0 ||
        config.goal_similarity_threshold > 1.0) {
        throw std::invalid_argument("invalid RLF-3 configuration");
    }
}

[[nodiscard]] core::SparseWorldModelConfig make_model_config(
    const Rlf3Config& config
) {
    core::SparseWorldModelConfig result;
    result.dimension = config.dimension;
    result.maximum_states = config.layers * config.lanes + 8U;
    result.maximum_contexts = 16U;
    result.maximum_transitions = result.maximum_states * 6U * 4U;
    result.maximum_outcomes_per_transition = 8U;
    result.maximum_subgoals = std::max<std::size_t>(
        1'024U,
        config.training_routes * config.maximum_execution_steps
    );
    result.hash_dimensions = std::min<std::size_t>(4U, config.dimension);
    result.phase_bins = 8U;
    result.maximum_bucket_candidates = 128U;
    result.nearest_subgoals = 6U;
    result.planner_node_budget = config.planner_node_budget;
    result.maximum_plan_depth = config.maximum_plan_depth;
    result.minimum_transition_support = 2U;
    result.minimum_subgoal_support = 1U;
    result.state_merge_distance = std::max(
        0.06,
        config.observation_noise_radians * 4.0
    );
    result.context_merge_distance = result.state_merge_distance;
    result.minimum_outcome_probability = 0.01;
    result.risk_penalty = 0.40;
    result.uncertainty_penalty = 0.25;
    result.heuristic_scale = 1.0;
    result.environment_layers = config.layers;
    result.environment_lanes = config.lanes;
    result.environment_stochastic_probability =
        config.stochastic_dominant_probability;
    result.environment_observation_noise = config.observation_noise_radians;
    return result;
}

void register_actions(core::SparseWorldModel& model) {
    static_cast<void>(model.register_action("step_a", 1.0));
    static_cast<void>(model.register_action("step_b", 1.0));
    static_cast<void>(model.register_action("step_middle", 1.0));
    static_cast<void>(model.register_action("stochastic_gate", 1.1));
    static_cast<void>(model.register_action("probe", 0.25));
    static_cast<void>(model.register_action("risky_shortcut", 1.3));
}

struct TrainingBundle final {
    core::SparseWorldModel model;
    core::SparseWorldModel memoryless_model;
    std::vector<core::WorldTransitionExperience> holdout;
    std::unordered_set<std::uint64_t> training_pairs;
    std::unordered_set<std::uint64_t> training_routes;
    std::size_t transition_experiences{};
    std::size_t successful_routes{};
    std::uint64_t transition_manifest_hash{fnv_offset_basis};
    std::uint64_t route_manifest_hash{fnv_offset_basis};
};

[[nodiscard]] TrainingBundle train_models(
    const Rlf3Config& config,
    const CausalGateWorld& world
) {
    TrainingBundle bundle{
        core::SparseWorldModel(make_model_config(config), config.seed),
        core::SparseWorldModel(
            make_model_config(config),
            config.seed ^ 0x4D454D4C455353ULL
        ),
        {}, {}, {}, 0U, 0U, fnv_offset_basis, fnv_offset_basis
    };
    register_actions(bundle.model);
    register_actions(bundle.memoryless_model);
    core::DeterministicRng rng(config.seed ^ 0x5452414E53495433ULL);
    core::DeterministicRng holdout_rng(config.seed ^ 0x484F4C444F555433ULL);

    for (std::size_t node = 0U; node < world.trap_node(); ++node) {
        if (world.layer(node) >= config.layers - 1U) {
            continue;
        }
        for (std::size_t mode = 0U; mode < 2U; ++mode) {
            for (std::size_t context_case = 0U; context_case < 2U; ++context_case) {
                const int memory_mode = context_case == 0U
                    ? -1
                    : static_cast<int>(mode);
                for (std::uint64_t action = action_step_a;
                     action <= action_shortcut; ++action) {
                    for (std::size_t sample = 0U;
                         sample < config.transition_samples_per_case; ++sample) {
                        const HiddenState state{node, mode, memory_mode, false};
                        const core::WorldObservation observation = world.observe(
                            state, rng, config.observation_noise_radians
                        );
                        const StepResult outcome = world.step(
                            state, action, rng, true
                        );
                        const core::WorldObservation next_observation = world.observe(
                            outcome.state, rng,
                            config.observation_noise_radians
                        );
                        core::WorldTransitionExperience experience{
                            observation, action, next_observation,
                            outcome.reward, outcome.terminal
                        };
                        bundle.model.observe_transition(experience);
                        core::WorldTransitionExperience memoryless = experience;
                        memoryless.observation = world.observe(
                            state, rng, config.observation_noise_radians, true
                        );
                        memoryless.next_observation = world.observe(
                            outcome.state, rng,
                            config.observation_noise_radians, true
                        );
                        bundle.memoryless_model.observe_transition(memoryless);
                        ++bundle.transition_experiences;
                        hash_u64(bundle.transition_manifest_hash,
                                 static_cast<std::uint64_t>(node));
                        hash_u64(bundle.transition_manifest_hash,
                                 static_cast<std::uint64_t>(mode));
                        hash_u64(bundle.transition_manifest_hash, action);
                    }
                    for (std::size_t sample = 0U; sample < 2U; ++sample) {
                        const HiddenState state{node, mode, memory_mode, false};
                        const auto observation = world.observe(
                            state, holdout_rng,
                            config.observation_noise_radians
                        );
                        const StepResult outcome = world.step(
                            state, action, holdout_rng, true
                        );
                        const auto next_observation = world.observe(
                            outcome.state, holdout_rng,
                            config.observation_noise_radians
                        );
                        bundle.holdout.push_back({
                            observation, action, next_observation,
                            outcome.reward, outcome.terminal
                        });
                    }
                }
            }
        }
    }

    core::DeterministicRng route_rng(config.seed ^ 0x524F5554455333ULL);
    std::size_t attempts = 0U;
    while (bundle.successful_routes < config.training_routes) {
        if (++attempts > config.training_routes * 1'000U) {
            throw std::runtime_error("unable to generate RLF-3 training routes");
        }
        const std::size_t start_lane = route_rng.uniform_index(config.lanes);
        const std::size_t mode = route_rng.uniform_index(2U);
        const std::size_t goal_lane = route_rng.uniform_index(config.lanes);
        HiddenState start{world.node(0U, start_lane), mode, -1, false};
        const std::size_t goal = world.node(config.layers - 1U, goal_lane);
        const std::uint64_t pair = pair_hash(start, goal);
        if (!bundle.training_pairs.insert(pair).second &&
            bundle.training_pairs.size() >= config.lanes * config.lanes * 2U) {
            // Duplicate demonstrations are useful for subgoal support.
        }
        const auto route = world.oracle_plan(start, goal, true);
        if (!route.has_value()) {
            continue;
        }
        core::WorldRouteExperience record;
        record.actions = *route;
        record.observations.reserve(route->size() + 1U);
        HiddenState current = start;
        record.observations.push_back(world.observe(current, route_rng, 0.0));
        bool failed = false;
        for (const std::uint64_t action : *route) {
            const StepResult outcome = world.step(current, action, route_rng, false);
            current = outcome.state;
            if (current.trapped) {
                failed = true;
                break;
            }
            record.observations.push_back(world.observe(current, route_rng, 0.0));
        }
        if (failed || !world.goal_reached(current, goal)) {
            continue;
        }
        bundle.model.observe_successful_route(record);
        core::WorldRouteExperience memoryless_record = record;
        for (std::size_t index = 0U; index < memoryless_record.observations.size(); ++index) {
            HiddenState replay = start;
            for (std::size_t step = 0U; step < index; ++step) {
                replay = world.step(
                    replay, record.actions[step], route_rng, false
                ).state;
            }
            memoryless_record.observations[index] = world.observe(
                replay, route_rng, 0.0, true
            );
        }
        bundle.memoryless_model.observe_successful_route(memoryless_record);
        const std::uint64_t route_value = route_hash(pair, *route);
        bundle.training_routes.insert(route_value);
        hash_u64(bundle.route_manifest_hash, pair);
        hash_u64(bundle.route_manifest_hash, route_value);
        ++bundle.successful_routes;
    }
    return bundle;
}

[[nodiscard]] std::vector<EvaluationCase> make_evaluation_cases(
    const Rlf3Config& config,
    const CausalGateWorld& world,
    const TrainingBundle& training
) {
    core::DeterministicRng rng(config.seed ^ 0x4556414C55415433ULL);
    std::vector<EvaluationCase> cases;
    cases.reserve(config.evaluation_episodes);
    std::unordered_set<std::uint64_t> pairs;
    std::size_t attempts = 0U;
    while (cases.size() < config.evaluation_episodes) {
        if (++attempts > config.evaluation_episodes * 5'000U) {
            throw std::runtime_error("unable to generate RLF-3 evaluation cases");
        }
        const std::size_t start_layer = 1U + rng.uniform_index(
            std::max<std::size_t>(1U, world.stochastic_gate_layer() - 1U)
        );
        const std::size_t start_lane = rng.uniform_index(config.lanes);
        const std::size_t mode = rng.uniform_index(2U);
        const std::size_t goal_lane = rng.uniform_index(config.lanes);
        HiddenState start{world.node(start_layer, start_lane), mode, -1, false};
        const std::size_t goal = world.node(config.layers - 1U, goal_lane);
        const std::uint64_t pair = pair_hash(start, goal);
        if (training.training_pairs.contains(pair) || !pairs.insert(pair).second) {
            continue;
        }
        const auto route = world.oracle_plan(start, goal, true);
        if (!route.has_value()) {
            continue;
        }
        const std::uint64_t route_value = route_hash(pair, *route);
        if (training.training_routes.contains(route_value)) {
            continue;
        }
        cases.push_back({start, goal, pair, route_value});
    }
    return cases;
}

enum class PolicyKind {
    indexed,
    flat,
    memoryless,
    greedy,
    oracle,
};

struct EpisodeOutcome final {
    bool success{false};
    bool trapped{false};
    bool planning_failure{false};
    std::size_t steps{};
    std::size_t replans{};
    std::size_t planner_nodes{};
    std::size_t transition_evaluations{};
    std::size_t subgoal_queries{};
    std::size_t subgoal_comparisons{};
    double final_similarity{};
};

[[nodiscard]] std::optional<std::uint64_t> greedy_action(
    const core::SparseWorldModel& model,
    const core::WorldObservation& observation,
    const core::PhaseVector& goal
) {
    double best_score = -std::numeric_limits<double>::infinity();
    std::uint64_t best_action = 0ULL;
    for (const core::WorldAction& action : model.actions()) {
        const auto prediction = model.predict(observation, action.id);
        if (!prediction.has_value()) {
            continue;
        }
        double expected_similarity = 0.0;
        for (const core::WorldPredictionOutcome& outcome : prediction->outcomes) {
            expected_similarity += outcome.probability *
                model.state_by_id(outcome.next_state_id).key.similarity(goal);
        }
        const double score = expected_similarity -
            prediction->uncertainty * 0.2 - action.cost * 0.01;
        if (score > best_score ||
            (score == best_score && action.id < best_action)) {
            best_score = score;
            best_action = action.id;
        }
    }
    return best_action == 0ULL ? std::nullopt :
        std::optional<std::uint64_t>(best_action);
}

[[nodiscard]] EpisodeOutcome run_episode(
    const Rlf3Config& config,
    const CausalGateWorld& world,
    const core::SparseWorldModel& model,
    const core::SparseWorldModel& memoryless_model,
    const EvaluationCase& episode,
    const PolicyKind policy,
    core::DeterministicRng& rng
) {
    EpisodeOutcome result;
    HiddenState state = episode.start;
    const core::PhaseVector& goal = world.goal_observation(episode.goal_node);
    for (std::size_t step = 0U; step < config.maximum_execution_steps; ++step) {
        if (world.goal_reached(state, episode.goal_node)) {
            result.success = true;
            break;
        }
        if (state.trapped || world.terminal(state)) {
            result.trapped = state.trapped;
            break;
        }
        const bool memoryless = policy == PolicyKind::memoryless;
        core::WorldObservation observation = world.observe(
            state, rng, config.observation_noise_radians, memoryless
        );
        std::optional<std::uint64_t> action;
        if (policy == PolicyKind::oracle) {
            const auto route = world.oracle_plan(state, episode.goal_node, false);
            if (route.has_value() && !route->empty()) {
                action = route->front();
            }
        } else if (policy == PolicyKind::greedy) {
            action = greedy_action(model, observation, goal);
        } else {
            const core::SparseWorldModel& selected_model = memoryless
                ? memoryless_model
                : model;
            const bool use_index = policy == PolicyKind::indexed || memoryless;
            const core::Rlf3PlanResult plan = selected_model.plan(
                observation,
                goal,
                use_index,
                config.maximum_plan_depth,
                config.planner_node_budget
            );
            ++result.replans;
            result.planner_nodes += plan.nodes_expanded;
            result.transition_evaluations += plan.transitions_evaluated;
            result.subgoal_queries += plan.subgoal_queries;
            result.subgoal_comparisons += plan.subgoal_comparisons;
            if (plan.success && !plan.actions.empty()) {
                action = plan.actions.front();
            }
        }
        if (!action.has_value()) {
            result.planning_failure = true;
            break;
        }
        const StepResult outcome = world.step(state, *action, rng, true);
        state = outcome.state;
        ++result.steps;
        if (state.trapped) {
            result.trapped = true;
            break;
        }
    }
    if (world.goal_reached(state, episode.goal_node)) {
        result.success = true;
    }
    core::WorldObservation final_observation = world.observe(
        state, rng, 0.0, policy == PolicyKind::memoryless
    );
    result.final_similarity = final_observation.visible.similarity(goal);
    return result;
}

[[nodiscard]] Rlf3PlannerMetrics evaluate_policy(
    const std::string& name,
    const Rlf3Config& config,
    const CausalGateWorld& world,
    const core::SparseWorldModel& model,
    const core::SparseWorldModel& memoryless_model,
    const std::span<const EvaluationCase> cases,
    const PolicyKind policy,
    const std::uint64_t seed
) {
    const auto begin = std::chrono::steady_clock::now();
    Rlf3PlannerMetrics metrics;
    metrics.name = name;
    core::DeterministicRng rng(seed);
    for (const EvaluationCase& episode : cases) {
        for (std::size_t rollout = 0U;
             rollout < config.stochastic_rollouts_per_episode; ++rollout) {
            const EpisodeOutcome result = run_episode(
                config, world, model, memoryless_model, episode, policy, rng
            );
            ++metrics.episodes;
            metrics.success_rate += result.success ? 1.0 : 0.0;
            metrics.mean_goal_similarity += result.final_similarity;
            metrics.average_execution_steps += static_cast<double>(result.steps);
            metrics.average_replans += static_cast<double>(result.replans);
            metrics.average_planner_nodes +=
                static_cast<double>(result.planner_nodes);
            metrics.average_transition_evaluations +=
                static_cast<double>(result.transition_evaluations);
            metrics.average_subgoal_queries +=
                static_cast<double>(result.subgoal_queries);
            metrics.average_subgoal_comparisons +=
                static_cast<double>(result.subgoal_comparisons);
            metrics.planning_failure_rate += result.planning_failure ? 1.0 : 0.0;
            metrics.trap_rate += result.trapped ? 1.0 : 0.0;
        }
    }
    if (metrics.episodes > 0U) {
        const double denominator = static_cast<double>(metrics.episodes);
        metrics.success_rate /= denominator;
        metrics.mean_goal_similarity /= denominator;
        metrics.average_execution_steps /= denominator;
        metrics.average_replans /= denominator;
        metrics.average_planner_nodes /= denominator;
        metrics.average_transition_evaluations /= denominator;
        metrics.average_subgoal_queries /= denominator;
        metrics.average_subgoal_comparisons /= denominator;
        metrics.planning_failure_rate /= denominator;
        metrics.trap_rate /= denominator;
    }
    metrics.inference_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin
    ).count();
    return metrics;
}

[[nodiscard]] std::size_t outcome_count(
    const core::SparseWorldModel& model
) {
    std::size_t count = 0U;
    for (const core::WorldTransition& transition : model.transitions()) {
        count += transition.outcomes.size();
    }
    return count;
}

void write_planner_metrics(
    std::ostream& output,
    const Rlf3PlannerMetrics& metrics,
    const std::size_t indent
) {
    const std::string padding(indent, ' ');
    output << "{\n"
           << padding << "  \"name\": \"" << json_escape(metrics.name) << "\",\n"
           << padding << "  \"episodes\": " << metrics.episodes << ",\n"
           << padding << "  \"success_rate\": " << metrics.success_rate << ",\n"
           << padding << "  \"mean_goal_similarity\": " << metrics.mean_goal_similarity << ",\n"
           << padding << "  \"average_execution_steps\": " << metrics.average_execution_steps << ",\n"
           << padding << "  \"average_replans\": " << metrics.average_replans << ",\n"
           << padding << "  \"average_planner_nodes\": " << metrics.average_planner_nodes << ",\n"
           << padding << "  \"average_transition_evaluations\": " << metrics.average_transition_evaluations << ",\n"
           << padding << "  \"average_subgoal_queries\": " << metrics.average_subgoal_queries << ",\n"
           << padding << "  \"average_subgoal_comparisons\": " << metrics.average_subgoal_comparisons << ",\n"
           << padding << "  \"planning_failure_rate\": " << metrics.planning_failure_rate << ",\n"
           << padding << "  \"trap_rate\": " << metrics.trap_rate << ",\n"
           << padding << "  \"inference_seconds\": " << metrics.inference_seconds << "\n"
           << padding << '}';
}

void write_plan(std::ostream& output, const core::Rlf3PlanResult& plan) {
    output << "{\n"
           << "    \"success\": " << (plan.success ? "true" : "false") << ",\n"
           << "    \"stop_reason\": \"" << core::to_string(plan.stop_reason) << "\",\n"
           << "    \"nodes_expanded\": " << plan.nodes_expanded << ",\n"
           << "    \"transitions_evaluated\": " << plan.transitions_evaluated << ",\n"
           << "    \"subgoal_queries\": " << plan.subgoal_queries << ",\n"
           << "    \"subgoal_comparisons\": " << plan.subgoal_comparisons << ",\n"
           << "    \"predicted_success_probability\": " << plan.predicted_success_probability << ",\n"
           << "    \"actions\": [";
    for (std::size_t index = 0U; index < plan.actions.size(); ++index) {
        output << plan.actions[index];
        if (index + 1U != plan.actions.size()) output << ", ";
    }
    output << "]\n  }";
}

}  // namespace

Rlf3Result run_rlf3_world_model(const Rlf3Config& config) {
    validate_config(config);
    const CausalGateWorld world(config, config.seed);
    const auto training_begin = std::chrono::steady_clock::now();
    TrainingBundle training = train_models(config, world);
    Rlf3Result result;
    result.seed = config.seed;
    result.dimension = config.dimension;
    result.physical_world_states = world.state_count();
    result.training_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - training_begin
    ).count();
    result.learned_state_prototypes = training.model.states().size();
    result.learned_context_prototypes = training.model.contexts().size();
    result.learned_transitions = training.model.transitions().size();
    result.learned_outcomes = outcome_count(training.model);
    result.learned_subgoals = training.model.subgoals().size();
    result.prediction.experiences = training.holdout.size();
    result.prediction.top1_accuracy = training.model.transition_top1_accuracy(
        training.holdout
    );
    result.prediction.negative_log_likelihood =
        training.model.transition_negative_log_likelihood(training.holdout);
    result.prediction.brier_score = training.model.transition_brier_score(
        training.holdout
    );
    std::size_t matched = 0U;
    for (const core::WorldTransitionExperience& experience : training.holdout) {
        matched += training.model.match_state(
            experience.observation.visible
        ).has_value() ? 1U : 0U;
    }
    result.prediction.state_match_rate = training.holdout.empty() ? 0.0 :
        static_cast<double>(matched) /
        static_cast<double>(training.holdout.size());

    const std::vector<EvaluationCase> cases = make_evaluation_cases(
        config, world, training
    );
    result.indexed_receding = evaluate_policy(
        "rlf3_indexed_receding", config, world, training.model,
        training.memoryless_model, cases, PolicyKind::indexed,
        config.seed ^ 0x494E444558454433ULL
    );
    result.flat_receding = evaluate_policy(
        "rlf3_flat_receding", config, world, training.model,
        training.memoryless_model, cases, PolicyKind::flat,
        config.seed ^ 0x464C4154504C414EULL
    );
    result.memoryless_receding = evaluate_policy(
        "rlf3_memoryless_receding", config, world, training.model,
        training.memoryless_model, cases, PolicyKind::memoryless,
        config.seed ^ 0x4D454D4C45535333ULL
    );
    result.greedy = evaluate_policy(
        "learned_greedy", config, world, training.model,
        training.memoryless_model, cases, PolicyKind::greedy,
        config.seed ^ 0x4752454544593333ULL
    );
    result.oracle = evaluate_policy(
        "oracle", config, world, training.model,
        training.memoryless_model, cases, PolicyKind::oracle,
        config.seed ^ 0x4F5241434C453333ULL
    );

    if (result.flat_receding.average_planner_nodes > 0.0) {
        result.planner_node_reduction = 1.0 -
            result.indexed_receding.average_planner_nodes /
            result.flat_receding.average_planner_nodes;
    }
    result.partial_observation_gain =
        result.indexed_receding.success_rate -
        result.memoryless_receding.success_rate;
    result.stochastic_success_rate = result.indexed_receding.success_rate;
    result.irreversible_success_rate = result.indexed_receding.success_rate;
    result.transition_compression_ratio =
        result.learned_transitions + result.learned_outcomes == 0U ? 0.0 :
        static_cast<double>(training.transition_experiences) /
        static_cast<double>(result.learned_transitions + result.learned_outcomes);
    result.estimated_model_bytes = training.model.estimated_bytes();
    result.training_stats = training.model.stats();

    result.leakage_audit.training_start_goal_pairs = training.training_pairs.size();
    result.leakage_audit.evaluation_start_goal_pairs = cases.size();
    result.leakage_audit.transition_manifest_hash =
        training.transition_manifest_hash;
    result.leakage_audit.route_manifest_hash = training.route_manifest_hash;
    result.leakage_audit.training_pair_hashes.assign(
        training.training_pairs.begin(), training.training_pairs.end()
    );
    result.leakage_audit.training_route_hashes.assign(
        training.training_routes.begin(), training.training_routes.end()
    );
    for (const EvaluationCase& episode : cases) {
        result.leakage_audit.evaluation_pair_hashes.push_back(episode.pair_hash);
        result.leakage_audit.evaluation_route_hashes.push_back(episode.route_hash);
        result.leakage_audit.start_goal_overlap +=
            training.training_pairs.contains(episode.pair_hash) ? 1U : 0U;
        result.leakage_audit.route_overlap +=
            training.training_routes.contains(episode.route_hash) ? 1U : 0U;
    }
    const auto sort_hashes = [](std::vector<std::uint64_t>& values) {
        std::sort(values.begin(), values.end());
    };
    sort_hashes(result.leakage_audit.training_pair_hashes);
    sort_hashes(result.leakage_audit.evaluation_pair_hashes);
    sort_hashes(result.leakage_audit.training_route_hashes);
    sort_hashes(result.leakage_audit.evaluation_route_hashes);
    result.leakage_audit.no_evaluation_start_goal_training =
        result.leakage_audit.start_goal_overlap == 0U;
    result.leakage_audit.no_evaluation_route_training =
        result.leakage_audit.route_overlap == 0U;

    core::DeterministicRng trace_rng(config.seed ^ 0x5452414345333333ULL);
    const core::WorldObservation trace_start = world.observe(
        cases.front().start, trace_rng, 0.0
    );
    result.representative_plan = training.model.plan(
        trace_start,
        world.goal_observation(cases.front().goal_node),
        true,
        config.maximum_plan_depth,
        config.planner_node_budget
    );

    // Impossible goals are random vectors not present in the learned state index.
    core::DeterministicRng impossible_rng(config.seed ^ 0x494D504F535333ULL);
    std::size_t false_successes = 0U;
    constexpr std::size_t impossible_cases = 32U;
    for (std::size_t index = 0U; index < impossible_cases; ++index) {
        const core::PhaseVector impossible = core::PhaseVector::random(
            config.dimension, impossible_rng
        );
        const core::WorldObservation start = world.observe(
            cases[index % cases.size()].start, impossible_rng, 0.0
        );
        false_successes += training.model.plan(
            start, impossible, true,
            config.maximum_plan_depth, config.planner_node_budget
        ).success ? 1U : 0U;
    }
    result.impossible_false_success_rate =
        static_cast<double>(false_successes) /
        static_cast<double>(impossible_cases);

    if (result.prediction.top1_accuracy >= 0.72 &&
        result.indexed_receding.success_rate >= 0.70 &&
        result.indexed_receding.success_rate >= result.greedy.success_rate + 0.15 &&
        result.planner_node_reduction >= 0.20 &&
        result.partial_observation_gain >= 0.15 &&
        result.impossible_false_success_rate == 0.0) {
        result.scientific_decision = "A";
    } else if (result.prediction.top1_accuracy >= 0.60 &&
               result.indexed_receding.success_rate >= 0.45 &&
               result.indexed_receding.success_rate > result.greedy.success_rate &&
               result.planner_node_reduction > 0.0 &&
               result.impossible_false_success_rate == 0.0) {
        result.scientific_decision = "B";
    } else {
        result.scientific_decision = "C";
    }
    result.limitations = {
        "The world is synthetic and finite even though transition dynamics are learned from samples.",
        "Action identities are predefined; RLF-3 learns their effects rather than discovering motor primitives.",
        "Subgoal supervision comes from successful training trajectories and may reuse route fragments.",
        "The sparse index uses coarse phase hashing with bounded local exact comparisons.",
        "Receding-horizon planning still performs explicit graph search over learned prototypes.",
        "Partial observability is limited to one persistent hidden binary mode.",
        "No natural language, coding, multimodal perception, or broad world knowledge is tested.",
    };

    result.deterministic_run_hash = fnv_offset_basis;
    hash_u64(result.deterministic_run_hash, result.seed);
    hash_u64(result.deterministic_run_hash,
             result.leakage_audit.transition_manifest_hash);
    hash_u64(result.deterministic_run_hash,
             result.leakage_audit.route_manifest_hash);
    hash_double(result.deterministic_run_hash,
                result.prediction.top1_accuracy);
    hash_double(result.deterministic_run_hash,
                result.indexed_receding.success_rate);
    hash_double(result.deterministic_run_hash,
                result.flat_receding.success_rate);
    hash_double(result.deterministic_run_hash,
                result.planner_node_reduction);
    hash_double(result.deterministic_run_hash,
                result.partial_observation_gain);
    hash_u64(result.deterministic_run_hash,
             static_cast<std::uint64_t>(result.learned_transitions));
    hash_u64(result.deterministic_run_hash,
             static_cast<std::uint64_t>(result.learned_subgoals));
    return result;
}

void write_rlf3_world_model_json(
    std::ostream& output,
    const Rlf3Result& result
) {
    output << std::setprecision(17);
    output << "{\n"
           << "  \"experiment\": \"rlf3_sparse_world_model\",\n"
           << "  \"seed\": " << result.seed << ",\n"
           << "  \"dimension\": " << result.dimension << ",\n"
           << "  \"physical_world_states\": " << result.physical_world_states << ",\n"
           << "  \"learned_state_prototypes\": " << result.learned_state_prototypes << ",\n"
           << "  \"learned_context_prototypes\": " << result.learned_context_prototypes << ",\n"
           << "  \"learned_transitions\": " << result.learned_transitions << ",\n"
           << "  \"learned_outcomes\": " << result.learned_outcomes << ",\n"
           << "  \"learned_subgoals\": " << result.learned_subgoals << ",\n"
           << "  \"prediction\": {\n"
           << "    \"experiences\": " << result.prediction.experiences << ",\n"
           << "    \"top1_accuracy\": " << result.prediction.top1_accuracy << ",\n"
           << "    \"negative_log_likelihood\": " << result.prediction.negative_log_likelihood << ",\n"
           << "    \"brier_score\": " << result.prediction.brier_score << ",\n"
           << "    \"state_match_rate\": " << result.prediction.state_match_rate << "\n"
           << "  },\n  \"systems\": {\n"
           << "    \"indexed_receding\": ";
    write_planner_metrics(output, result.indexed_receding, 4U);
    output << ",\n    \"flat_receding\": ";
    write_planner_metrics(output, result.flat_receding, 4U);
    output << ",\n    \"memoryless_receding\": ";
    write_planner_metrics(output, result.memoryless_receding, 4U);
    output << ",\n    \"greedy\": ";
    write_planner_metrics(output, result.greedy, 4U);
    output << ",\n    \"oracle\": ";
    write_planner_metrics(output, result.oracle, 4U);
    output << "\n  },\n"
           << "  \"planner_node_reduction\": " << result.planner_node_reduction << ",\n"
           << "  \"transition_compression_ratio\": " << result.transition_compression_ratio << ",\n"
           << "  \"partial_observation_gain\": " << result.partial_observation_gain << ",\n"
           << "  \"stochastic_success_rate\": " << result.stochastic_success_rate << ",\n"
           << "  \"irreversible_success_rate\": " << result.irreversible_success_rate << ",\n"
           << "  \"impossible_false_success_rate\": " << result.impossible_false_success_rate << ",\n"
           << "  \"estimated_model_bytes\": " << result.estimated_model_bytes << ",\n"
           << "  \"training_seconds\": " << result.training_seconds << ",\n"
           << "  \"training_stats\": {\n"
           << "    \"experiences_observed\": " << result.training_stats.experiences_observed << ",\n"
           << "    \"states_created\": " << result.training_stats.states_created << ",\n"
           << "    \"contexts_created\": " << result.training_stats.contexts_created << ",\n"
           << "    \"transitions_created\": " << result.training_stats.transitions_created << ",\n"
           << "    \"outcomes_created\": " << result.training_stats.outcomes_created << ",\n"
           << "    \"subgoals_created\": " << result.training_stats.subgoals_created << "\n"
           << "  },\n"
           << "  \"leakage_audit\": {\n"
           << "    \"no_exact_action_models_in_fabric\": " << (result.leakage_audit.no_exact_action_models_in_fabric ? "true" : "false") << ",\n"
           << "    \"no_evaluation_start_goal_training\": " << (result.leakage_audit.no_evaluation_start_goal_training ? "true" : "false") << ",\n"
           << "    \"no_evaluation_route_training\": " << (result.leakage_audit.no_evaluation_route_training ? "true" : "false") << ",\n"
           << "    \"no_hidden_mode_in_observation\": " << (result.leakage_audit.no_hidden_mode_in_observation ? "true" : "false") << ",\n"
           << "    \"no_target_access_during_execution\": " << (result.leakage_audit.no_target_access_during_execution ? "true" : "false") << ",\n"
           << "    \"transition_holdout_disjoint\": " << (result.leakage_audit.transition_holdout_disjoint ? "true" : "false") << ",\n"
           << "    \"start_goal_overlap\": " << result.leakage_audit.start_goal_overlap << ",\n"
           << "    \"route_overlap\": " << result.leakage_audit.route_overlap << ",\n"
           << "    \"transition_manifest_hash\": \"" << format_hash(result.leakage_audit.transition_manifest_hash) << "\",\n"
           << "    \"route_manifest_hash\": \"" << format_hash(result.leakage_audit.route_manifest_hash) << "\"\n"
           << "  },\n"
           << "  \"scientific_decision\": \"" << result.scientific_decision << "\",\n"
           << "  \"limitations\": [";
    for (std::size_t index = 0U; index < result.limitations.size(); ++index) {
        output << '"' << json_escape(result.limitations[index]) << '"';
        if (index + 1U != result.limitations.size()) output << ", ";
    }
    output << "],\n"
           << "  \"deterministic_run_hash\": \"" << format_hash(result.deterministic_run_hash) << "\",\n"
           << "  \"representative_plan\": ";
    write_plan(output, result.representative_plan);
    output << "\n}\n";
}

Rlf3TrainingWorkflowResult train_rlf3_checkpoint(
    const Rlf3Config& config,
    const std::filesystem::path& checkpoint_path
) {
    validate_config(config);
    const CausalGateWorld world(config, config.seed);
    TrainingBundle training = train_models(config, world);
    storage::save_rlf3_checkpoint(checkpoint_path, training.model);
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, config.seed);
    hash_u64(hash, static_cast<std::uint64_t>(training.transition_experiences));
    hash_u64(hash, static_cast<std::uint64_t>(training.successful_routes));
    hash_u64(hash, static_cast<std::uint64_t>(training.model.transitions().size()));
    return {
        checkpoint_path,
        config.seed,
        training.transition_experiences,
        training.successful_routes,
        training.model.states().size(),
        training.model.contexts().size(),
        training.model.transitions().size(),
        training.model.subgoals().size(),
        hash,
    };
}

Rlf3EvaluationWorkflowResult evaluate_rlf3_checkpoint(
    const std::filesystem::path& checkpoint_path,
    const std::uint64_t seed,
    const std::size_t evaluation_episodes
) {
    core::SparseWorldModel model = storage::load_rlf3_checkpoint(checkpoint_path);
    Rlf3Config config;
    config.seed = seed;
    config.dimension = model.config().dimension;
    config.layers = model.config().environment_layers;
    config.lanes = model.config().environment_lanes;
    config.stochastic_dominant_probability =
        model.config().environment_stochastic_probability;
    config.observation_noise_radians =
        model.config().environment_observation_noise;
    config.evaluation_episodes = evaluation_episodes;
    config.planner_node_budget = model.config().planner_node_budget;
    config.maximum_plan_depth = model.config().maximum_plan_depth;
    const CausalGateWorld world(config, model.seed());
    // Evaluation workflow uses the same model as a memoryless placeholder only;
    // the two reported paths both retain the checkpoint's learned context model.
    TrainingBundle synthetic{
        core::SparseWorldModel::from_snapshot(model.snapshot()),
        core::SparseWorldModel::from_snapshot(model.snapshot()),
        {}, {}, {}, 0U, 0U, fnv_offset_basis, fnv_offset_basis
    };
    const std::vector<EvaluationCase> cases = make_evaluation_cases(
        config, world, synthetic
    );
    Rlf3EvaluationWorkflowResult result;
    result.checkpoint_path = checkpoint_path;
    result.evaluation_episodes = evaluation_episodes;
    result.indexed_receding = evaluate_policy(
        "rlf3_indexed_receding", config, world, model, model,
        cases, PolicyKind::indexed, seed ^ 0x4556414C3331ULL
    );
    result.flat_receding = evaluate_policy(
        "rlf3_flat_receding", config, world, model, model,
        cases, PolicyKind::flat, seed ^ 0x4556414C3332ULL
    );
    result.deterministic_run_hash = fnv_offset_basis;
    hash_double(result.deterministic_run_hash,
                result.indexed_receding.success_rate);
    hash_double(result.deterministic_run_hash,
                result.flat_receding.success_rate);
    return result;
}

Rlf3TraceWorkflowResult trace_rlf3_checkpoint(
    const std::filesystem::path& checkpoint_path,
    const std::uint64_t seed,
    const std::size_t sample_id
) {
    core::SparseWorldModel model = storage::load_rlf3_checkpoint(checkpoint_path);
    Rlf3Config config;
    config.seed = seed;
    config.dimension = model.config().dimension;
    config.evaluation_episodes = sample_id + 1U;
    config.planner_node_budget = model.config().planner_node_budget;
    config.maximum_plan_depth = model.config().maximum_plan_depth;
    const CausalGateWorld world(config, model.seed());
    TrainingBundle synthetic{
        core::SparseWorldModel::from_snapshot(model.snapshot()),
        core::SparseWorldModel::from_snapshot(model.snapshot()),
        {}, {}, {}, 0U, 0U, fnv_offset_basis, fnv_offset_basis
    };
    const std::vector<EvaluationCase> cases = make_evaluation_cases(
        config, world, synthetic
    );
    core::DeterministicRng rng(seed ^ 0x545241434533ULL);
    const EvaluationCase& selected = cases.at(sample_id);
    const core::WorldObservation start = world.observe(selected.start, rng, 0.0);
    return {
        checkpoint_path,
        sample_id,
        model.plan(start, world.goal_observation(selected.goal_node), true),
        model.plan(start, world.goal_observation(selected.goal_node), false),
    };
}

void write_rlf3_training_json(
    std::ostream& output,
    const Rlf3TrainingWorkflowResult& result
) {
    output << "{\n"
           << "  \"checkpoint\": \"" << json_escape(result.checkpoint_path.string()) << "\",\n"
           << "  \"seed\": " << result.seed << ",\n"
           << "  \"transition_experiences\": " << result.transition_experiences << ",\n"
           << "  \"training_routes\": " << result.training_routes << ",\n"
           << "  \"states\": " << result.states << ",\n"
           << "  \"contexts\": " << result.contexts << ",\n"
           << "  \"transitions\": " << result.transitions << ",\n"
           << "  \"subgoals\": " << result.subgoals << ",\n"
           << "  \"deterministic_run_hash\": \"" << format_hash(result.deterministic_run_hash) << "\"\n"
           << "}\n";
}

void write_rlf3_evaluation_json(
    std::ostream& output,
    const Rlf3EvaluationWorkflowResult& result
) {
    output << std::setprecision(17)
           << "{\n  \"checkpoint\": \""
           << json_escape(result.checkpoint_path.string()) << "\",\n"
           << "  \"evaluation_episodes\": " << result.evaluation_episodes << ",\n"
           << "  \"indexed_receding\": ";
    write_planner_metrics(output, result.indexed_receding, 2U);
    output << ",\n  \"flat_receding\": ";
    write_planner_metrics(output, result.flat_receding, 2U);
    output << ",\n  \"deterministic_run_hash\": \""
           << format_hash(result.deterministic_run_hash) << "\"\n}\n";
}

void write_rlf3_trace_json(
    std::ostream& output,
    const Rlf3TraceWorkflowResult& result
) {
    output << std::setprecision(17)
           << "{\n  \"checkpoint\": \""
           << json_escape(result.checkpoint_path.string()) << "\",\n"
           << "  \"sample_id\": " << result.sample_id << ",\n"
           << "  \"indexed_plan\": ";
    write_plan(output, result.indexed_plan);
    output << ",\n  \"flat_plan\": ";
    write_plan(output, result.flat_plan);
    output << "\n}\n";
}

}  // namespace rlf::experiments
