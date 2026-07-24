#include "rlf/experiments/rlf1_latent_routing.hpp"

#include "rlf/core/deterministic_rng.hpp"
#include "rlf/experiments/phase_vector_smoke.hpp"
#include "rlf/storage/rlf1_checkpoint.hpp"

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
#include <map>
#include <numbers>
#include <optional>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rlf::experiments {
namespace {

constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;

struct NamedOperator final {
    std::string name;
    core::TransformationOperator transformation{1U};
    double cost{1.0};
};

struct Episode final {
    core::PhaseVector start{std::vector<float>{0.0F}};
    core::PhaseVector goal{std::vector<float>{0.0F}};
    std::vector<std::uint64_t> route;
    std::uint64_t route_hash{};
    std::uint64_t start_goal_hash{};
};

struct Manifest final {
    std::vector<Episode> episodes;
    std::unordered_set<std::uint64_t> route_hashes;
    std::unordered_set<std::uint64_t> start_goal_hashes;
    std::uint64_t manifest_hash{fnv_offset_basis};
};

struct SupervisedPrototype final {
    core::PhaseVector signature{std::vector<float>{0.0F}};
    std::uint64_t operator_id{};
};

void hash_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (unsigned int byte = 0U; byte < 8U; ++byte) {
        hash ^= (value >> (byte * 8U)) & 0xFFULL;
        hash *= fnv_prime;
    }
}

void hash_double(std::uint64_t& hash, const double value) noexcept {
    hash_u64(hash, std::bit_cast<std::uint64_t>(value));
}

[[nodiscard]] std::string json_escape(const std::string_view value) {
    std::string result;
    result.reserve(value.size() + 8U);
    for (const char character : value) {
        switch (character) {
        case '"': result += "\\\""; break;
        case '\\': result += "\\\\"; break;
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
    std::vector<float> noise;
    noise.reserve(source.size());
    for (std::size_t index = 0U; index < source.size(); ++index) {
        const double centered = (2.0 * rng.uniform_unit()) - 1.0;
        noise.push_back(static_cast<float>(centered * radians));
    }
    return source.composed(core::PhaseVector(std::move(noise)));
}

[[nodiscard]] core::TransformationOperator make_shift(
    const std::size_t dimension,
    const double scale,
    const std::size_t period,
    const std::size_t offset = 0U
) {
    std::vector<float> values(dimension, 0.0F);
    for (std::size_t index = 0U; index < dimension; ++index) {
        const double centered = static_cast<double>(
            ((index + offset) % period) + 1U
        );
        values[index] = core::PhaseVector::normalize_angle(
            static_cast<float>(scale * centered)
        );
    }
    return core::TransformationOperator(
        dimension,
        {core::OperatorPrimitive::shift(core::PhaseVector(std::move(values)))}
    );
}

[[nodiscard]] core::TransformationOperator make_rotation(
    const std::size_t dimension,
    const bool left
) {
    std::vector<std::size_t> permutation(dimension);
    for (std::size_t index = 0U; index < dimension; ++index) {
        permutation[index] = left
            ? (index + 1U) % dimension
            : (index + dimension - 1U) % dimension;
    }
    return core::TransformationOperator(
        dimension,
        {core::OperatorPrimitive::permute(std::move(permutation))}
    );
}

[[nodiscard]] std::vector<NamedOperator> make_operator_definitions(
    const std::size_t dimension,
    const std::size_t requested_count
) {
    if (dimension < 2U || requested_count < 6U || requested_count > 8U) {
        throw std::invalid_argument(
            "RLF-1 requires dimension >= 2 and 6-8 operator families"
        );
    }
    const core::TransformationOperator shift_a =
        make_shift(dimension, 0.071, 7U);
    const core::TransformationOperator shift_a_inverse =
        make_shift(dimension, -0.071, 7U);
    const core::TransformationOperator shift_b =
        make_shift(dimension, 0.043, 5U, 2U);
    const core::TransformationOperator rotate_left =
        make_rotation(dimension, true);
    const core::TransformationOperator rotate_right =
        make_rotation(dimension, false);
    const core::TransformationOperator conjugate(
        dimension,
        {core::OperatorPrimitive::conjugate(dimension)}
    );

    std::vector<NamedOperator> result;
    result.reserve(requested_count);
    result.push_back({"shift_a", shift_a, 1.0});
    result.push_back({"shift_a_inverse", shift_a_inverse, 1.0});
    result.push_back({"rotate_left", rotate_left, 1.0});
    result.push_back({"rotate_right", rotate_right, 1.0});
    result.push_back({"conjugate", conjugate, 1.0});
    result.push_back({
        "rotate_left_then_shift_b",
        rotate_left.then(shift_b),
        1.8,
    });
    if (requested_count >= 7U) {
        result.push_back({
            "conjugate_then_shift_b",
            conjugate.then(shift_b),
            1.8,
        });
    }
    if (requested_count >= 8U) {
        result.push_back({
            "explicit_shift_rotate_conjugate",
            shift_a.then(rotate_left).then(conjugate),
            2.4,
        });
    }
    return result;
}

[[nodiscard]] std::vector<std::uint64_t> register_operators(
    core::LatentRouter& router,
    const std::vector<NamedOperator>& definitions
) {
    std::vector<std::uint64_t> ids;
    ids.reserve(definitions.size());
    for (const NamedOperator& definition : definitions) {
        ids.push_back(router.register_operator(
            definition.name,
            definition.transformation,
            definition.cost
        ));
    }
    return ids;
}

[[nodiscard]] core::PhaseVector apply_route(
    const std::vector<NamedOperator>& definitions,
    const core::PhaseVector& start,
    const std::span<const std::uint64_t> route
) {
    core::PhaseVector state = start;
    for (const std::uint64_t operator_id : route) {
        if (operator_id == 0U || operator_id > definitions.size()) {
            throw std::out_of_range("episode references unknown operator");
        }
        state = definitions[operator_id - 1U].transformation.apply(state);
    }
    return state;
}

[[nodiscard]] std::uint64_t start_goal_hash(
    const core::PhaseVector& start,
    const core::PhaseVector& goal
) noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, core::LatentRouter::phase_state_hash(start));
    hash_u64(hash, core::LatentRouter::phase_state_hash(goal));
    return hash;
}

void validate_config(const Rlf1Config& config) {
    if (config.dimension < 2U || config.training_episodes == 0U ||
        config.development_episodes == 0U || config.evaluation_episodes == 0U ||
        config.training_min_route_length == 0U ||
        config.training_min_route_length > config.training_max_route_length ||
        config.evaluation_min_route_length == 0U ||
        config.evaluation_min_route_length > config.evaluation_max_route_length ||
        config.maximum_cycles == 0U || config.operator_count < 6U ||
        config.operator_count > 8U ||
        !std::isfinite(config.state_noise_radians) ||
        config.state_noise_radians < 0.0 ||
        !std::isfinite(config.goal_similarity_threshold) ||
        config.goal_similarity_threshold <= 0.0 ||
        config.goal_similarity_threshold > 1.0) {
        throw std::invalid_argument("invalid RLF-1 experiment configuration");
    }
}

[[nodiscard]] core::LatentRouterConfig make_router_config(
    const Rlf1Config& config,
    const core::LatentCreditStrategy credit =
        core::LatentCreditStrategy::discounted_eligibility
) {
    core::LatentRouterConfig result;
    result.dimension = config.dimension;
    result.maximum_cycles = config.maximum_cycles;
    result.search_node_budget = 25'000U;
    result.route_memory_capacity = std::max<std::size_t>(
        128U,
        config.training_episodes * 4U
    );
    result.maximum_modes = std::max<std::size_t>(
        1'024U,
        config.training_episodes * config.training_max_route_length * 4U
    );
    result.goal_similarity_threshold = config.goal_similarity_threshold;
    result.mode_creation_similarity = 0.94;
    result.mode_learning_rate = 0.20;
    result.utility_learning_rate = 0.12;
    result.eligibility_decay = 0.85;
    result.goal_progress_weight = 0.25;
    result.route_memory_weight = 0.15;
    result.successor_familiarity_weight = 0.35;
    result.route_repetition_penalty = 1.0;
    result.learned_halt_threshold = 0.97;
    result.learned_halt_goal_floor = 0.90;
    result.abstention_entropy_threshold = 0.995;
    result.abstention_resonance_threshold = 0.02;
    result.minimum_action_score = -0.5;
    result.action_temperature = 0.25;
    result.search_beam_width = 4U;
    result.search_lookahead_depth = 2U;
    result.enable_route_memory = true;
    result.enable_macro_operators = false;
    result.credit_strategy = credit;
    result.halt_policy = core::LatentHaltPolicy::combined_safe;
    return result;
}

[[nodiscard]] Manifest generate_manifest(
    const Rlf1Config& config,
    const std::vector<NamedOperator>& definitions,
    const std::size_t episode_count,
    const std::size_t minimum_length,
    const std::size_t maximum_length,
    const std::uint64_t seed,
    const std::unordered_set<std::uint64_t>& forbidden_routes = {},
    const std::unordered_set<std::uint64_t>& forbidden_pairs = {}
) {
    core::DeterministicRng rng(seed);
    std::vector<core::PhaseVector> prototypes;
    prototypes.reserve(4U);
    for (std::size_t index = 0U; index < 4U; ++index) {
        prototypes.push_back(core::PhaseVector::random(config.dimension, rng));
    }

    Manifest manifest;
    manifest.episodes.reserve(episode_count);
    std::size_t attempts = 0U;
    const std::size_t maximum_attempts = episode_count * 20'000U;
    while (manifest.episodes.size() < episode_count) {
        if (++attempts > maximum_attempts) {
            throw std::runtime_error("unable to construct leakage-free RLF-1 manifest");
        }
        const std::size_t route_length = minimum_length + rng.uniform_index(
            maximum_length - minimum_length + 1U
        );
        std::vector<std::uint64_t> route;
        route.reserve(route_length);
        for (std::size_t step = 0U; step < route_length; ++step) {
            std::uint64_t selected = static_cast<std::uint64_t>(
                rng.uniform_index(definitions.size()) + 1U
            );
            if (!route.empty()) {
                const std::uint64_t previous = route.back();
                const bool inverse_pair =
                    (previous == 1U && selected == 2U) ||
                    (previous == 2U && selected == 1U) ||
                    (previous == 3U && selected == 4U) ||
                    (previous == 4U && selected == 3U);
                if (inverse_pair) {
                    selected = static_cast<std::uint64_t>(
                        (static_cast<std::size_t>(selected) % definitions.size()) + 1U
                    );
                }
            }
            route.push_back(selected);
        }
        const std::uint64_t route_value = core::LatentRouter::route_hash(route);
        if (forbidden_routes.contains(route_value) ||
            manifest.route_hashes.contains(route_value)) {
            continue;
        }
        const std::size_t prototype_index =
            (manifest.episodes.size() + rng.uniform_index(4U)) % 4U;
        core::PhaseVector start = perturb(
            prototypes[prototype_index],
            config.state_noise_radians + 0.02,
            rng
        );
        core::PhaseVector goal = apply_route(definitions, start, route);
        const std::uint64_t pair_value = start_goal_hash(start, goal);
        if (forbidden_pairs.contains(pair_value) ||
            manifest.start_goal_hashes.contains(pair_value)) {
            continue;
        }
        manifest.route_hashes.insert(route_value);
        manifest.start_goal_hashes.insert(pair_value);
        hash_u64(manifest.manifest_hash, route_value);
        hash_u64(manifest.manifest_hash, pair_value);
        manifest.episodes.push_back({
            .start = std::move(start),
            .goal = std::move(goal),
            .route = std::move(route),
            .route_hash = route_value,
            .start_goal_hash = pair_value,
        });
    }
    return manifest;
}

[[nodiscard]] core::LatentRouter train_router(
    const Rlf1Config& config,
    const std::vector<NamedOperator>& definitions,
    const std::span<const Episode> episodes,
    const core::LatentCreditStrategy strategy,
    std::size_t* successful_episodes = nullptr
) {
    core::LatentRouter router(make_router_config(config, strategy), config.seed);
    static_cast<void>(register_operators(router, definitions));
    std::size_t successes = 0U;
    for (const Episode& episode : episodes) {
        const core::LatentTrainingResult training = router.train_episode(
            episode.start,
            episode.goal,
            config.training_max_route_length
        );
        successes += training.success ? 1U : 0U;
    }
    if (successful_episodes != nullptr) {
        *successful_episodes = successes;
    }
    return router;
}

[[nodiscard]] bool is_halt_reason(const core::LatentStopReason reason) noexcept {
    return reason == core::LatentStopReason::successful_halt ||
        reason == core::LatentStopReason::learned_halt;
}

[[nodiscard]] core::LatentExecutionResult greedy_execute(
    const std::vector<NamedOperator>& definitions,
    const Episode& episode,
    const std::size_t maximum_cycles,
    const double threshold
) {
    core::PhaseVector state = episode.start;
    core::LatentExecutionResult result;
    result.final_state = state;
    std::unordered_set<std::uint64_t> visited;
    visited.insert(core::LatentRouter::phase_state_hash(state));
    for (std::size_t cycle = 0U; cycle < maximum_cycles; ++cycle) {
        if (state.similarity(episode.goal) >= threshold) {
            result.success = true;
            result.stop_reason = core::LatentStopReason::successful_halt;
            break;
        }
        std::uint64_t selected = 0U;
        double best = -1.0;
        core::PhaseVector next = state;
        for (std::size_t index = 0U; index < definitions.size(); ++index) {
            core::PhaseVector candidate =
                definitions[index].transformation.apply(state);
            const double score = candidate.similarity(episode.goal);
            ++result.exact_similarity_evaluations;
            if (score > best + 1.0e-15) {
                best = score;
                selected = static_cast<std::uint64_t>(index + 1U);
                next = std::move(candidate);
            }
        }
        result.route.push_back(selected);
        ++result.active_mode_evaluations;
        state = std::move(next);
        const std::uint64_t hash = core::LatentRouter::phase_state_hash(state);
        if (!visited.insert(hash).second) {
            result.stop_reason = core::LatentStopReason::loop_detected;
            break;
        }
    }
    result.cycles = result.route.size();
    result.final_state = state;
    result.final_goal_similarity = state.similarity(episode.goal);
    if (result.final_goal_similarity >= threshold) {
        result.success = true;
        result.stop_reason = core::LatentStopReason::successful_halt;
    } else if (result.stop_reason == core::LatentStopReason::successful_halt) {
        result.stop_reason = core::LatentStopReason::cycle_limit;
    }
    return result;
}

[[nodiscard]] core::LatentExecutionResult apply_named_route(
    const std::vector<NamedOperator>& definitions,
    const Episode& episode,
    const std::span<const std::uint64_t> route,
    const std::size_t maximum_cycles,
    const double threshold,
    const std::size_t similarity_evaluations = 0U
) {
    core::PhaseVector state = episode.start;
    core::LatentExecutionResult result;
    result.exact_similarity_evaluations = similarity_evaluations;
    const std::size_t limit = std::min(maximum_cycles, route.size());
    for (std::size_t index = 0U; index < limit; ++index) {
        const std::uint64_t operator_id = route[index];
        if (operator_id == 0U || operator_id > definitions.size()) {
            break;
        }
        state = definitions[operator_id - 1U].transformation.apply(state);
        result.route.push_back(operator_id);
        ++result.active_mode_evaluations;
        if (state.similarity(episode.goal) >= threshold) {
            break;
        }
    }
    result.final_state = state;
    result.cycles = result.route.size();
    result.final_goal_similarity = state.similarity(episode.goal);
    result.success = result.final_goal_similarity >= threshold;
    result.stop_reason = result.success
        ? core::LatentStopReason::successful_halt
        : core::LatentStopReason::cycle_limit;
    return result;
}

[[nodiscard]] core::LatentExecutionResult nearest_route_execute(
    const std::vector<NamedOperator>& definitions,
    const std::span<const Episode> training,
    const Episode& episode,
    const std::size_t maximum_cycles,
    const double threshold
) {
    const core::PhaseVector query = core::LatentRouter::state_goal_signature(
        episode.start,
        episode.goal
    );
    const Episode* best_episode = nullptr;
    double best = -1.0;
    for (const Episode& candidate : training) {
        const core::PhaseVector signature =
            core::LatentRouter::state_goal_signature(
                candidate.start,
                candidate.goal
            );
        const double score = query.similarity(signature);
        if (score > best) {
            best = score;
            best_episode = &candidate;
        }
    }
    if (best_episode == nullptr) {
        core::LatentExecutionResult empty;
        empty.final_state = episode.start;
        empty.stop_reason = core::LatentStopReason::no_candidate;
        return empty;
    }
    return apply_named_route(
        definitions,
        episode,
        best_episode->route,
        maximum_cycles,
        threshold,
        training.size()
    );
}

[[nodiscard]] std::vector<SupervisedPrototype> make_supervised_prototypes(
    const std::vector<NamedOperator>& definitions,
    const std::span<const Episode> training
) {
    std::vector<SupervisedPrototype> result;
    for (const Episode& episode : training) {
        core::PhaseVector state = episode.start;
        for (const std::uint64_t operator_id : episode.route) {
            result.push_back({
                .signature = core::LatentRouter::state_goal_signature(
                    state,
                    episode.goal
                ),
                .operator_id = operator_id,
            });
            state = definitions[operator_id - 1U].transformation.apply(state);
        }
    }
    return result;
}

[[nodiscard]] core::LatentExecutionResult supervised_execute(
    const std::vector<NamedOperator>& definitions,
    const std::span<const SupervisedPrototype> prototypes,
    const Episode& episode,
    const std::size_t maximum_cycles,
    const double threshold
) {
    core::PhaseVector state = episode.start;
    core::LatentExecutionResult result;
    std::unordered_set<std::uint64_t> visited;
    visited.insert(core::LatentRouter::phase_state_hash(state));
    for (std::size_t cycle = 0U; cycle < maximum_cycles; ++cycle) {
        if (state.similarity(episode.goal) >= threshold) {
            break;
        }
        const core::PhaseVector query = core::LatentRouter::state_goal_signature(
            state,
            episode.goal
        );
        const SupervisedPrototype* best_prototype = nullptr;
        double best = -1.0;
        for (const SupervisedPrototype& prototype : prototypes) {
            const double score = query.similarity(prototype.signature);
            ++result.exact_similarity_evaluations;
            if (score > best) {
                best = score;
                best_prototype = &prototype;
            }
        }
        if (best_prototype == nullptr) {
            result.stop_reason = core::LatentStopReason::no_candidate;
            break;
        }
        const std::uint64_t selected = best_prototype->operator_id;
        state = definitions[selected - 1U].transformation.apply(state);
        result.route.push_back(selected);
        ++result.active_mode_evaluations;
        const std::uint64_t hash = core::LatentRouter::phase_state_hash(state);
        if (!visited.insert(hash).second) {
            result.stop_reason = core::LatentStopReason::loop_detected;
            break;
        }
    }
    result.final_state = state;
    result.cycles = result.route.size();
    result.final_goal_similarity = state.similarity(episode.goal);
    result.success = result.final_goal_similarity >= threshold;
    result.stop_reason = result.success
        ? core::LatentStopReason::successful_halt
        : (result.stop_reason == core::LatentStopReason::no_candidate ||
           result.stop_reason == core::LatentStopReason::loop_detected
            ? result.stop_reason
            : core::LatentStopReason::cycle_limit);
    return result;
}

template <typename Executor>
[[nodiscard]] Rlf1SystemMetrics evaluate_system(
    const std::string& name,
    const std::span<const Episode> episodes,
    Executor&& executor
) {
    Rlf1SystemMetrics metrics;
    metrics.name = name;
    metrics.episodes = episodes.size();
    std::size_t successes = 0U;
    std::size_t exact_routes = 0U;
    std::size_t valid_routes = 0U;
    std::size_t first_actions = 0U;
    std::size_t premature = 0U;
    std::size_t halt_events = 0U;
    std::size_t successful_halts = 0U;
    std::size_t abstentions = 0U;
    std::size_t matching_route_steps = 0U;
    std::size_t compared_route_steps = 0U;
    constexpr std::size_t calibration_bins = 10U;
    std::array<std::size_t, calibration_bins> bin_counts{};
    std::array<double, calibration_bins> bin_confidence{};
    std::array<double, calibration_bins> bin_accuracy{};
    double goal_total = 0.0;
    double cycles_total = 0.0;
    double optimal_total = 0.0;
    double excess_total = 0.0;
    double uncertainty_total = 0.0;
    double similarities_total = 0.0;
    double active_total = 0.0;
    double search_nodes_total = 0.0;
    const auto begin = std::chrono::steady_clock::now();
    for (const Episode& episode : episodes) {
        const core::LatentExecutionResult execution = executor(episode);
        successes += execution.success ? 1U : 0U;
        valid_routes += execution.success ? 1U : 0U;
        exact_routes += execution.route == episode.route ? 1U : 0U;
        first_actions += !execution.route.empty() && !episode.route.empty() &&
            execution.route.front() == episode.route.front() ? 1U : 0U;
        premature += execution.stop_reason ==
            core::LatentStopReason::premature_halt ? 1U : 0U;
        const bool halted = is_halt_reason(execution.stop_reason);
        halt_events += halted ? 1U : 0U;
        successful_halts += halted && execution.success ? 1U : 0U;
        abstentions += execution.abstained ? 1U : 0U;
        const std::size_t compared = std::min(
            execution.route.size(), episode.route.size());
        compared_route_steps += compared;
        for (std::size_t step = 0U; step < compared; ++step) {
            matching_route_steps +=
                execution.route[step] == episode.route[step] ? 1U : 0U;
        }
        const double confidence = std::clamp(
            1.0 - execution.mean_uncertainty, 0.0, 1.0);
        const std::size_t bin = std::min(
            calibration_bins - 1U,
            static_cast<std::size_t>(confidence *
                static_cast<double>(calibration_bins)));
        ++bin_counts[bin];
        bin_confidence[bin] += confidence;
        bin_accuracy[bin] += execution.success ? 1.0 : 0.0;
        goal_total += execution.final_goal_similarity;
        cycles_total += static_cast<double>(execution.cycles);
        optimal_total += static_cast<double>(episode.route.size());
        if (execution.cycles > episode.route.size()) {
            excess_total += static_cast<double>(
                execution.cycles - episode.route.size()
            );
        }
        uncertainty_total += execution.mean_uncertainty;
        similarities_total += static_cast<double>(
            execution.exact_similarity_evaluations
        );
        active_total += static_cast<double>(execution.active_mode_evaluations);
        search_nodes_total += static_cast<double>(execution.search_nodes);
    }
    const auto end = std::chrono::steady_clock::now();
    const double denominator = episodes.empty()
        ? 1.0
        : static_cast<double>(episodes.size());
    const std::size_t answered = episodes.size() - abstentions;
    metrics.final_state_accuracy = static_cast<double>(successes) / denominator;
    metrics.exact_route_accuracy = static_cast<double>(exact_routes) / denominator;
    metrics.valid_route_accuracy = static_cast<double>(valid_routes) / denominator;
    metrics.first_action_accuracy = static_cast<double>(first_actions) / denominator;
    metrics.mean_goal_similarity = goal_total / denominator;
    metrics.average_cycles = cycles_total / denominator;
    metrics.average_optimal_route_length = optimal_total / denominator;
    metrics.average_excess_steps = excess_total / denominator;
    metrics.premature_halt_rate = static_cast<double>(premature) / denominator;
    metrics.halt_precision = halt_events == 0U
        ? 0.0
        : static_cast<double>(successful_halts) /
            static_cast<double>(halt_events);
    metrics.halt_recall = successes == 0U
        ? 0.0
        : static_cast<double>(successful_halts) /
            static_cast<double>(successes);
    metrics.abstention_rate = static_cast<double>(abstentions) / denominator;
    metrics.selective_accuracy = answered == 0U
        ? 0.0
        : static_cast<double>(successes) / static_cast<double>(answered);
    metrics.mean_uncertainty = uncertainty_total / denominator;
    metrics.per_step_operator_accuracy = compared_route_steps == 0U
        ? 0.0
        : static_cast<double>(matching_route_steps) /
            static_cast<double>(compared_route_steps);
    double calibration_error = 0.0;
    for (std::size_t bin = 0U; bin < calibration_bins; ++bin) {
        if (bin_counts[bin] == 0U) continue;
        const double count = static_cast<double>(bin_counts[bin]);
        const double mean_confidence = bin_confidence[bin] / count;
        const double mean_accuracy = bin_accuracy[bin] / count;
        calibration_error += (count / denominator) *
            std::abs(mean_confidence - mean_accuracy);
    }
    metrics.expected_calibration_error = calibration_error;
    metrics.average_exact_similarities = similarities_total / denominator;
    metrics.average_active_operations = active_total / denominator;
    metrics.average_search_nodes = search_nodes_total / denominator;
    metrics.inference_seconds = std::chrono::duration<double>(end - begin).count();
    return metrics;
}

[[nodiscard]] std::size_t estimate_router_bytes(
    const core::LatentRouter& router
) noexcept {
    std::size_t bytes = sizeof(router);
    const std::size_t vector_bytes =
        router.config().dimension * sizeof(core::PhaseVector::Angle);
    bytes += router.modes().size() *
        (sizeof(core::LatentRoutingMode) + vector_bytes);
    bytes += router.halt_modes().size() *
        (sizeof(core::LatentHaltMode) + vector_bytes);
    for (const core::RouteMemoryRecord& record : router.route_memory()) {
        bytes += sizeof(record) + vector_bytes +
            (record.route.size() * sizeof(std::uint64_t));
    }
    for (const core::RegisteredOperator& value : router.operators()) {
        bytes += sizeof(value) + value.name.size() +
            (value.primitive_route.size() * sizeof(std::uint64_t));
        for (const core::OperatorPrimitive& primitive :
             value.transformation.primitives()) {
            bytes += primitive.phase_shift.size() * sizeof(float);
            bytes += primitive.permutation.size() * sizeof(std::size_t);
        }
    }
    return bytes;
}

void write_system_metrics(
    std::ostream& output,
    const Rlf1SystemMetrics& value,
    const std::size_t indentation
) {
    const std::string indent(indentation, ' ');
    const std::string field(indentation + 2U, ' ');
    output << indent << "{\n"
           << field << "\"name\": \"" << json_escape(value.name) << "\",\n"
           << field << "\"episodes\": " << value.episodes << ",\n"
           << field << "\"final_state_accuracy\": " << value.final_state_accuracy << ",\n"
           << field << "\"exact_route_accuracy\": " << value.exact_route_accuracy << ",\n"
           << field << "\"valid_route_accuracy\": " << value.valid_route_accuracy << ",\n"
           << field << "\"first_action_accuracy\": " << value.first_action_accuracy << ",\n"
           << field << "\"mean_goal_similarity\": " << value.mean_goal_similarity << ",\n"
           << field << "\"average_cycles\": " << value.average_cycles << ",\n"
           << field << "\"average_optimal_route_length\": " << value.average_optimal_route_length << ",\n"
           << field << "\"average_excess_steps\": " << value.average_excess_steps << ",\n"
           << field << "\"premature_halt_rate\": " << value.premature_halt_rate << ",\n"
           << field << "\"halt_precision\": " << value.halt_precision << ",\n"
           << field << "\"halt_recall\": " << value.halt_recall << ",\n"
           << field << "\"abstention_rate\": " << value.abstention_rate << ",\n"
           << field << "\"selective_accuracy\": " << value.selective_accuracy << ",\n"
           << field << "\"mean_uncertainty\": " << value.mean_uncertainty << ",\n"
           << field << "\"expected_calibration_error\": " << value.expected_calibration_error << ",\n"
           << field << "\"per_step_operator_accuracy\": " << value.per_step_operator_accuracy << ",\n"
           << field << "\"average_exact_similarities\": " << value.average_exact_similarities << ",\n"
           << field << "\"average_active_operations\": " << value.average_active_operations << ",\n"
           << field << "\"average_search_nodes\": " << value.average_search_nodes << ",\n"
           << field << "\"inference_seconds\": " << value.inference_seconds << "\n"
           << indent << '}';
}

void write_candidate(
    std::ostream& output,
    const core::LatentActionCandidate& value,
    const std::size_t indentation
) {
    const std::string indent(indentation, ' ');
    const std::string field(indentation + 2U, ' ');
    output << indent << "{\n"
           << field << "\"operator_id\": " << value.operator_id << ",\n"
           << field << "\"routing_mode_id\": " << value.routing_mode_id << ",\n"
           << field << "\"resonance\": " << value.resonance << ",\n"
           << field << "\"utility\": " << value.utility << ",\n"
           << field << "\"eligibility\": " << value.eligibility << ",\n"
           << field << "\"immediate_progress\": " << value.immediate_progress << ",\n"
           << field << "\"route_memory_bonus\": " << value.route_memory_bonus << ",\n"
           << field << "\"successor_familiarity\": " << value.successor_familiarity << ",\n"
           << field << "\"score\": " << value.score << ",\n"
           << field << "\"normalized_weight\": " << value.normalized_weight << "\n"
           << indent << '}';
}

void write_trace_step(
    std::ostream& output,
    const core::LatentTraceStep& step,
    const std::size_t indentation
) {
    const std::string indent(indentation, ' ');
    const std::string field(indentation + 2U, ' ');
    output << indent << "{\n"
           << field << "\"cycle\": " << step.cycle << ",\n"
           << field << "\"state_hash\": \"" << format_run_hash(step.state_hash) << "\",\n"
           << field << "\"working_state_hash\": \"" << format_run_hash(step.working_state_hash) << "\",\n"
           << field << "\"goal_state_hash\": \"" << format_run_hash(step.goal_state_hash) << "\",\n"
           << field << "\"memory_summary_hash\": \"" << format_run_hash(step.memory_summary_hash) << "\",\n"
           << field << "\"route_summary_hash\": \"" << format_run_hash(step.route_summary_hash) << "\",\n"
           << field << "\"selected_operator_id\": " << step.selected_operator_id << ",\n"
           << field << "\"selected_operator_name\": \"" << json_escape(step.selected_operator_name) << "\",\n"
           << field << "\"selected_mode_id\": " << step.selected_mode_id << ",\n"
           << field << "\"goal_similarity_before\": " << step.goal_similarity_before << ",\n"
           << field << "\"goal_similarity_after\": " << step.goal_similarity_after << ",\n"
           << field << "\"progress\": " << step.progress << ",\n"
           << field << "\"uncertainty\": " << step.uncertainty << ",\n"
           << field << "\"halt_score\": " << step.halt_score << ",\n"
           << field << "\"candidates\": [";
    if (!step.candidates.empty()) {
        output << '\n';
        for (std::size_t index = 0U; index < step.candidates.size(); ++index) {
            write_candidate(output, step.candidates[index], indentation + 4U);
            output << (index + 1U == step.candidates.size() ? "\n" : ",\n");
        }
        output << field;
    }
    output << "]\n" << indent << '}';
}

void write_execution(
    std::ostream& output,
    const core::LatentExecutionResult& execution,
    const std::size_t indentation
) {
    const std::string indent(indentation, ' ');
    const std::string field(indentation + 2U, ' ');
    output << indent << "{\n"
           << field << "\"success\": " << (execution.success ? "true" : "false") << ",\n"
           << field << "\"abstained\": " << (execution.abstained ? "true" : "false") << ",\n"
           << field << "\"stop_reason\": \"" << core::to_string(execution.stop_reason) << "\",\n"
           << field << "\"cycles\": " << execution.cycles << ",\n"
           << field << "\"final_goal_similarity\": " << execution.final_goal_similarity << ",\n"
           << field << "\"mean_uncertainty\": " << execution.mean_uncertainty << ",\n"
           << field << "\"exact_similarity_evaluations\": " << execution.exact_similarity_evaluations << ",\n"
           << field << "\"active_mode_evaluations\": " << execution.active_mode_evaluations << ",\n"
           << field << "\"search_nodes\": " << execution.search_nodes << ",\n"
           << field << "\"route\": [";
    for (std::size_t index = 0U; index < execution.route.size(); ++index) {
        output << execution.route[index];
        if (index + 1U != execution.route.size()) output << ", ";
    }
    output << "],\n" << field << "\"steps\": [";
    if (!execution.trace.empty()) {
        output << '\n';
        for (std::size_t index = 0U; index < execution.trace.size(); ++index) {
            write_trace_step(output, execution.trace[index], indentation + 4U);
            output << (index + 1U == execution.trace.size() ? "\n" : ",\n");
        }
        output << field;
    }
    output << "]\n" << indent << '}';
}

[[nodiscard]] std::vector<NamedOperator> definitions_from_router(
    const core::LatentRouter& router
) {
    std::vector<NamedOperator> definitions;
    for (const core::RegisteredOperator& value : router.operators()) {
        if (!value.macro) {
            definitions.push_back({value.name, value.transformation, value.cost});
        }
    }
    return definitions;
}

}  // namespace

Rlf1Result run_rlf1_latent_routing(const Rlf1Config& config) {
    validate_config(config);
    const std::vector<NamedOperator> definitions = make_operator_definitions(
        config.dimension,
        config.operator_count
    );
    const Manifest training = generate_manifest(
        config,
        definitions,
        config.training_episodes,
        config.training_min_route_length,
        config.training_max_route_length,
        config.seed ^ 0x545241494EULL
    );
    const Manifest development = generate_manifest(
        config,
        definitions,
        config.development_episodes,
        std::min<std::size_t>(2U,
            config.evaluation_min_route_length),
        config.evaluation_max_route_length,
        config.seed ^ 0x444556ULL,
        training.route_hashes,
        training.start_goal_hashes
    );
    std::unordered_set<std::uint64_t> forbidden_routes = training.route_hashes;
    forbidden_routes.insert(
        development.route_hashes.begin(), development.route_hashes.end());
    std::unordered_set<std::uint64_t> forbidden_pairs =
        training.start_goal_hashes;
    forbidden_pairs.insert(
        development.start_goal_hashes.begin(),
        development.start_goal_hashes.end());
    const Manifest evaluation = generate_manifest(
        config,
        definitions,
        config.evaluation_episodes,
        config.evaluation_min_route_length,
        config.evaluation_max_route_length,
        config.seed ^ 0x4556414CULL,
        forbidden_routes,
        forbidden_pairs
    );

    const auto training_begin = std::chrono::steady_clock::now();
    std::size_t successful_training = 0U;
    core::LatentRouter router = train_router(
        config,
        definitions,
        training.episodes,
        core::LatentCreditStrategy::discounted_eligibility,
        &successful_training
    );
    const auto training_end = std::chrono::steady_clock::now();

    Rlf1Result result;
    result.seed = config.seed;
    result.dimension = config.dimension;
    result.training_episodes = config.training_episodes;
    result.development_episodes = config.development_episodes;
    result.evaluation_episodes = config.evaluation_episodes;

    core::LatentRouter single_router = core::LatentRouter::from_snapshot(
        router.snapshot()
    );
    result.rlf = evaluate_system(
        "rlf1_single_route",
        evaluation.episodes,
        [&single_router](const Episode& episode) {
            return single_router.execute(
                episode.start,
                episode.goal,
                std::nullopt,
                true
            );
        }
    );
    struct SearchChoice final {
        std::size_t depth;
        std::size_t beam;
    };
    const std::array<SearchChoice, 3U> search_choices{{
        {1U, 2U},
        {2U, 4U},
        {3U, 4U},
    }};
    SearchChoice selected_search = search_choices.front();
    bool has_selection = false;
    for (const SearchChoice choice : search_choices) {
        core::LatentRouter development_router =
            core::LatentRouter::from_snapshot(router.snapshot());
        const Rlf1SystemMetrics metrics = evaluate_system(
            "rlf1_development_search",
            development.episodes,
            [&development_router, choice](const Episode& episode) {
                return development_router.execute_with_bounded_lookahead(
                    episode.start,
                    episode.goal,
                    choice.depth,
                    choice.beam,
                    true
                );
            }
        );
        const bool better = !has_selection ||
            metrics.final_state_accuracy >
                result.development_search_selection.final_state_accuracy + 1.0e-15 ||
            (std::abs(metrics.final_state_accuracy -
                      result.development_search_selection.final_state_accuracy) <= 1.0e-15 &&
             metrics.mean_goal_similarity >
                result.development_search_selection.mean_goal_similarity + 1.0e-15) ||
            (std::abs(metrics.final_state_accuracy -
                      result.development_search_selection.final_state_accuracy) <= 1.0e-15 &&
             std::abs(metrics.mean_goal_similarity -
                      result.development_search_selection.mean_goal_similarity) <= 1.0e-15 &&
             metrics.average_search_nodes <
                result.development_search_selection.average_search_nodes);
        if (better) {
            has_selection = true;
            selected_search = choice;
            result.development_search_selection = metrics;
        }
    }
    result.selected_lookahead_depth = selected_search.depth;
    result.selected_beam_width = selected_search.beam;
    core::LatentRouter search_router = core::LatentRouter::from_snapshot(
        router.snapshot()
    );
    result.rlf_search_assisted = evaluate_system(
        "rlf1_bounded_resonant_lookahead",
        evaluation.episodes,
        [&search_router, selected_search](const Episode& episode) {
            return search_router.execute_with_bounded_lookahead(
                episode.start,
                episode.goal,
                selected_search.depth,
                selected_search.beam,
                true
            );
        }
    );
    core::LatentRouterSnapshot no_memory_snapshot = router.snapshot();
    no_memory_snapshot.config.enable_route_memory = false;
    no_memory_snapshot.route_memory.clear();
    core::LatentRouter no_memory_router =
        core::LatentRouter::from_snapshot(std::move(no_memory_snapshot));
    result.rlf_without_route_memory = evaluate_system(
        "rlf1_without_route_memory",
        evaluation.episodes,
        [&no_memory_router](const Episode& episode) {
            return no_memory_router.execute(
                episode.start,
                episode.goal,
                std::nullopt,
                false
            );
        }
    );
    result.greedy = evaluate_system(
        "greedy_goal_similarity",
        evaluation.episodes,
        [&definitions, &config](const Episode& episode) {
            return greedy_execute(
                definitions,
                episode,
                config.maximum_cycles,
                config.goal_similarity_threshold
            );
        }
    );
    result.nearest_route = evaluate_system(
        "nearest_route_lookup",
        evaluation.episodes,
        [&definitions, &training, &config](const Episode& episode) {
            return nearest_route_execute(
                definitions,
                training.episodes,
                episode,
                config.maximum_cycles,
                config.goal_similarity_threshold
            );
        }
    );
    const std::vector<SupervisedPrototype> supervised_prototypes =
        make_supervised_prototypes(definitions, training.episodes);
    result.supervised = evaluate_system(
        "compact_supervised_router",
        evaluation.episodes,
        [&definitions, &supervised_prototypes, &config](const Episode& episode) {
            return supervised_execute(
                definitions,
                supervised_prototypes,
                episode,
                config.maximum_cycles,
                config.goal_similarity_threshold
            );
        }
    );
    result.oracle = evaluate_system(
        "oracle_route_and_halt",
        evaluation.episodes,
        [&definitions, &config](const Episode& episode) {
            return apply_named_route(
                definitions,
                episode,
                episode.route,
                config.maximum_cycles,
                config.goal_similarity_threshold
            );
        }
    );

    for (std::size_t length = config.evaluation_min_route_length;
         length <= config.evaluation_max_route_length;
         ++length) {
        std::vector<Episode> subset;
        for (const Episode& episode : evaluation.episodes) {
            if (episode.route.size() == length) subset.push_back(episode);
        }
        if (subset.empty()) continue;
        core::LatentRouter length_single = core::LatentRouter::from_snapshot(
            router.snapshot()
        );
        core::LatentRouter length_search = core::LatentRouter::from_snapshot(
            router.snapshot()
        );
        const Rlf1SystemMetrics rlf_metrics = evaluate_system(
            "rlf", subset,
            [&length_single](const Episode& episode) {
                return length_single.execute(episode.start, episode.goal);
            }
        );
        const Rlf1SystemMetrics search_metrics = evaluate_system(
            "search", subset,
            [&length_search, selected_search](const Episode& episode) {
                return length_search.execute_with_bounded_lookahead(
                    episode.start,
                    episode.goal,
                    selected_search.depth,
                    selected_search.beam
                );
            }
        );
        const Rlf1SystemMetrics greedy_metrics = evaluate_system(
            "greedy", subset,
            [&definitions, &config](const Episode& episode) {
                return greedy_execute(definitions, episode, config.maximum_cycles,
                    config.goal_similarity_threshold);
            }
        );
        const Rlf1SystemMetrics supervised_metrics = evaluate_system(
            "supervised", subset,
            [&definitions, &supervised_prototypes, &config](const Episode& episode) {
                return supervised_execute(definitions, supervised_prototypes,
                    episode, config.maximum_cycles,
                    config.goal_similarity_threshold);
            }
        );
        const Rlf1SystemMetrics oracle_metrics = evaluate_system(
            "oracle", subset,
            [&definitions, &config](const Episode& episode) {
                return apply_named_route(definitions, episode, episode.route,
                    config.maximum_cycles, config.goal_similarity_threshold);
            }
        );
        result.by_length.push_back({
            .route_length = length,
            .episodes = subset.size(),
            .rlf_accuracy = rlf_metrics.final_state_accuracy,
            .rlf_search_accuracy = search_metrics.final_state_accuracy,
            .greedy_accuracy = greedy_metrics.final_state_accuracy,
            .supervised_accuracy = supervised_metrics.final_state_accuracy,
            .oracle_accuracy = oracle_metrics.final_state_accuracy,
        });
    }

    const std::vector<core::LatentHaltPolicy> halt_policies{
        core::LatentHaltPolicy::goal_threshold,
        core::LatentHaltPolicy::learned_resonant,
        core::LatentHaltPolicy::combined_safe,
    };
    for (const core::LatentHaltPolicy policy : halt_policies) {
        core::LatentRouterSnapshot halt_snapshot = router.snapshot();
        halt_snapshot.config.halt_policy = policy;
        core::LatentRouter halt_router =
            core::LatentRouter::from_snapshot(std::move(halt_snapshot));
        const Rlf1SystemMetrics metrics = evaluate_system(
            std::string(core::to_string(policy)),
            evaluation.episodes,
            [&halt_router](const Episode& episode) {
                return halt_router.execute(episode.start, episode.goal);
            }
        );
        result.halt_policies.push_back({
            .policy = std::string(core::to_string(policy)),
            .accuracy = metrics.final_state_accuracy,
            .halt_precision = metrics.halt_precision,
            .halt_recall = metrics.halt_recall,
            .premature_halt_rate = metrics.premature_halt_rate,
            .average_cycles = metrics.average_cycles,
        });
    }

    const std::vector<core::LatentCreditStrategy> strategies{
        core::LatentCreditStrategy::uniform_route,
        core::LatentCreditStrategy::discounted_eligibility,
        core::LatentCreditStrategy::progress_weighted,
        core::LatentCreditStrategy::counterfactual_local,
    };
    for (const core::LatentCreditStrategy strategy : strategies) {
        core::LatentRouter credit_router(make_router_config(config, strategy),
            config.seed ^ static_cast<std::uint64_t>(strategy));
        static_cast<void>(register_operators(credit_router, definitions));
        for (const Episode& episode : training.episodes) {
            credit_router.reinforce_route(
                episode.start,
                episode.goal,
                episode.route,
                1.0
            );
        }
        const Rlf1SystemMetrics metrics = evaluate_system(
            "credit", evaluation.episodes,
            [&credit_router](const Episode& episode) {
                return credit_router.execute(episode.start, episode.goal);
            }
        );
        result.delayed_credit.push_back({
            .strategy = std::string(core::to_string(strategy)),
            .accuracy = metrics.final_state_accuracy,
            .mean_goal_similarity = metrics.mean_goal_similarity,
            .mode_count = credit_router.modes().size(),
        });
    }

    const std::vector<std::size_t> scale_targets{
        std::min<std::size_t>(8U, training.episodes.size()),
        std::min<std::size_t>(16U, training.episodes.size()),
        std::min<std::size_t>(32U, training.episodes.size()),
        training.episodes.size(),
    };
    std::set<std::size_t> unique_targets(scale_targets.begin(), scale_targets.end());
    for (const std::size_t count : unique_targets) {
        core::LatentRouter scale_router = train_router(
            config,
            definitions,
            std::span<const Episode>(training.episodes.data(), count),
            core::LatentCreditStrategy::discounted_eligibility
        );
        const std::size_t eval_count = std::min<std::size_t>(12U,
            evaluation.episodes.size());
        const Rlf1SystemMetrics metrics = evaluate_system(
            "scaling",
            std::span<const Episode>(evaluation.episodes.data(), eval_count),
            [&scale_router](const Episode& episode) {
                return scale_router.execute(episode.start, episode.goal);
            }
        );
        result.scaling.push_back({
            .training_episodes = count,
            .physical_modes = scale_router.modes().size(),
            .accuracy = metrics.final_state_accuracy,
            .average_exact_similarities = metrics.average_exact_similarities,
            .average_active_operations = metrics.average_active_operations,
        });
    }

    Rlf1Config noisy_config = config;
    noisy_config.state_noise_radians = std::max(0.15, config.state_noise_radians * 4.0);
    const Manifest noisy_evaluation = generate_manifest(
        noisy_config,
        definitions,
        std::min<std::size_t>(24U, config.evaluation_episodes),
        config.evaluation_min_route_length,
        config.evaluation_max_route_length,
        config.seed ^ 0x4E4F495345ULL,
        training.route_hashes,
        training.start_goal_hashes
    );
    core::LatentRouter noisy_router = core::LatentRouter::from_snapshot(
        router.snapshot()
    );
    result.noise_robustness_accuracy = evaluate_system(
        "rlf1_high_noise",
        noisy_evaluation.episodes,
        [&noisy_router](const Episode& episode) {
            return noisy_router.execute(episode.start, episode.goal);
        }
    ).final_state_accuracy;

    const std::size_t recovery_count = std::min<std::size_t>(12U,
        evaluation.episodes.size());
    std::size_t recovery_successes = 0U;
    std::size_t recovery_search_successes = 0U;
    for (std::size_t index = 0U; index < recovery_count; ++index) {
        const Episode& episode = evaluation.episodes[index];
        std::uint64_t wrong = (episode.route.front() % definitions.size()) + 1U;
        if (wrong == episode.route.front()) {
            wrong = (wrong % definitions.size()) + 1U;
        }
        core::LatentRouter recovery_router = core::LatentRouter::from_snapshot(
            router.snapshot()
        );
        const core::LatentExecutionResult direct = recovery_router.execute(
            episode.start,
            episode.goal,
            wrong,
            true
        );
        recovery_successes += direct.success ? 1U : 0U;

        const core::PhaseVector damaged =
            definitions[wrong - 1U].transformation.apply(episode.start);
        core::LatentRouter recovery_search = core::LatentRouter::from_snapshot(
            router.snapshot()
        );
        const core::LatentExecutionResult searched =
            recovery_search.execute_with_bounded_lookahead(
                damaged,
                episode.goal,
                selected_search.depth,
                selected_search.beam
            );
        recovery_search_successes += searched.success ? 1U : 0U;
    }
    const double recovery_denominator = recovery_count == 0U
        ? 1.0 : static_cast<double>(recovery_count);
    result.recovery_accuracy = static_cast<double>(recovery_successes) /
        recovery_denominator;
    result.recovery_search_accuracy =
        static_cast<double>(recovery_search_successes) / recovery_denominator;

    core::DeterministicRng impossible_rng(config.seed ^ 0x494D504F5353ULL);
    constexpr std::size_t impossible_count = 8U;
    std::size_t false_successes = 0U;
    std::size_t abstentions = 0U;
    std::size_t search_abstentions = 0U;
    for (std::size_t index = 0U; index < impossible_count; ++index) {
        Episode impossible{
            .start = core::PhaseVector::random(config.dimension, impossible_rng),
            .goal = core::PhaseVector::random(config.dimension, impossible_rng),
            .route = {},
        };
        core::LatentRouter impossible_router = core::LatentRouter::from_snapshot(
            router.snapshot()
        );
        const core::LatentExecutionResult direct = impossible_router.execute(
            impossible.start, impossible.goal);
        false_successes += direct.success ? 1U : 0U;
        abstentions += direct.abstained ? 1U : 0U;
        core::LatentRouter impossible_search = core::LatentRouter::from_snapshot(
            router.snapshot()
        );
        const core::LatentExecutionResult searched =
            impossible_search.execute_with_bounded_lookahead(
                impossible.start,
                impossible.goal,
                selected_search.depth,
                selected_search.beam
            );
        search_abstentions += searched.abstained ? 1U : 0U;
    }
    result.impossible_case_false_success_rate =
        static_cast<double>(false_successes) /
        static_cast<double>(impossible_count);
    result.ambiguity_abstention_rate = static_cast<double>(abstentions) /
        static_cast<double>(impossible_count);
    result.ambiguity_search_abstention_rate =
        static_cast<double>(search_abstentions) /
        static_cast<double>(impossible_count);

    core::LatentRouter continual_router(
        make_router_config(config),
        config.seed ^ 0x434F4E54494EULL
    );
    static_cast<void>(register_operators(continual_router, definitions));
    std::vector<double> best_accuracy(4U, 0.0);
    std::vector<double> final_accuracy(4U, 0.0);
    for (std::size_t stage = 0U; stage < 4U; ++stage) {
        for (std::size_t index = stage;
             index < training.episodes.size();
             index += 4U) {
            const Episode& episode = training.episodes[index];
            static_cast<void>(continual_router.train_episode(
                episode.start,
                episode.goal,
                config.training_max_route_length
            ));
        }
        for (std::size_t family = 0U; family <= stage; ++family) {
            std::vector<Episode> family_eval;
            for (std::size_t index = family;
                 index < evaluation.episodes.size();
                 index += 4U) {
                family_eval.push_back(evaluation.episodes[index]);
            }
            core::LatentRouter evaluation_router =
                core::LatentRouter::from_snapshot(continual_router.snapshot());
            const Rlf1SystemMetrics metrics = evaluate_system(
                "continual", family_eval,
                [&evaluation_router](const Episode& episode) {
                    return evaluation_router.execute(episode.start, episode.goal);
                }
            );
            best_accuracy[family] = std::max(
                best_accuracy[family], metrics.final_state_accuracy);
            if (stage == 3U) final_accuracy[family] = metrics.final_state_accuracy;
        }
    }
    double retained_total = 0.0;
    double forgetting_total = 0.0;
    for (std::size_t family = 0U; family < 4U; ++family) {
        retained_total += final_accuracy[family];
        forgetting_total += std::max(0.0,
            best_accuracy[family] - final_accuracy[family]);
    }
    result.continual_retained_accuracy = retained_total / 4.0;
    result.continual_forgetting = forgetting_total / 4.0;

    {
        Rlf1Config macro_config = config;
        macro_config.dimension = std::max<std::size_t>(8U, config.dimension);
        core::LatentRouterConfig router_config = make_router_config(macro_config);
        router_config.enable_macro_operators = true;
        router_config.macro_minimum_occurrences = 4U;
        router_config.macro_maximum_length = 4U;
        core::LatentRouter macro_router(router_config, config.seed ^ 0x4D4143524FULL);
        const std::vector<NamedOperator> macro_definitions =
            make_operator_definitions(macro_config.dimension, 6U);
        const std::vector<std::uint64_t> macro_ids =
            register_operators(macro_router, macro_definitions);
        const std::vector<std::uint64_t> repeated_route{
            macro_ids[0U], macro_ids[2U], macro_ids[4U]
        };
        core::DeterministicRng macro_rng(config.seed ^ 0x4D4143524F32ULL);
        for (std::size_t repetition = 0U; repetition < 4U; ++repetition) {
            const core::PhaseVector start = core::PhaseVector::random(
                macro_config.dimension, macro_rng);
            const core::PhaseVector goal = apply_route(
                macro_definitions, start, repeated_route);
            macro_router.reinforce_route(start, goal, repeated_route, 1.0);
        }
        result.macros_proposed = 1U;
        result.macros_created = macro_router.consolidate_macros();
        result.macros_rejected = result.macros_created == 0U ? 1U : 0U;
        const auto macro_iterator = std::find_if(
            macro_router.operators().begin(),
            macro_router.operators().end(),
            [](const core::RegisteredOperator& value) { return value.macro; }
        );
        std::size_t macro_success = 0U;
        double interference = 0.0;
        constexpr std::size_t validation_count = 16U;
        for (std::size_t sample = 0U; sample < validation_count; ++sample) {
            const core::PhaseVector start = core::PhaseVector::random(
                macro_config.dimension, macro_rng);
            const core::PhaseVector expected = apply_route(
                macro_definitions, start, repeated_route);
            if (macro_iterator != macro_router.operators().end()) {
                const core::PhaseVector predicted =
                    macro_iterator->transformation.apply(start);
                macro_success += predicted.similarity(expected) >=
                    macro_config.goal_similarity_threshold ? 1U : 0U;
            }
            const core::PhaseVector primitive =
                macro_router.operator_by_id(macro_ids[1U]).transformation.apply(start);
            const core::PhaseVector reference =
                macro_definitions[1U].transformation.apply(start);
            interference += 1.0 - primitive.similarity(reference);
        }
        result.macro_validation_accuracy =
            static_cast<double>(macro_success) /
            static_cast<double>(validation_count);
        result.macro_cycle_reduction = result.macros_created == 0U
            ? 0.0
            : 1.0 - (1.0 / static_cast<double>(repeated_route.size()));
        result.macro_interference = interference /
            static_cast<double>(validation_count);
    }

    result.routing_mode_count = router.modes().size();
    result.halt_mode_count = router.halt_modes().size();
    result.route_memory_records = router.route_memory().size();
    result.operator_count = router.operators().size();
    result.estimated_bytes = estimate_router_bytes(router);
    for (const Episode& episode : training.episodes) {
        result.local_update_operations += episode.route.size();
    }
    result.training_seconds = std::chrono::duration<double>(
        training_end - training_begin).count();

    std::size_t overlap = 0U;
    for (const std::uint64_t value : evaluation.route_hashes) {
        overlap += training.route_hashes.contains(value) ? 1U : 0U;
        overlap += development.route_hashes.contains(value) ? 1U : 0U;
    }
    for (const std::uint64_t value : development.route_hashes) {
        overlap += training.route_hashes.contains(value) ? 1U : 0U;
    }
    std::size_t pair_overlap = 0U;
    for (const std::uint64_t value : evaluation.start_goal_hashes) {
        pair_overlap += training.start_goal_hashes.contains(value) ? 1U : 0U;
        pair_overlap += development.start_goal_hashes.contains(value) ? 1U : 0U;
    }
    for (const std::uint64_t value : development.start_goal_hashes) {
        pair_overlap += training.start_goal_hashes.contains(value) ? 1U : 0U;
    }
    result.training_route_hash_values.assign(
        training.route_hashes.begin(), training.route_hashes.end());
    result.development_route_hash_values.assign(
        development.route_hashes.begin(), development.route_hashes.end());
    result.evaluation_route_hash_values.assign(
        evaluation.route_hashes.begin(), evaluation.route_hashes.end());
    result.training_start_goal_hash_values.assign(
        training.start_goal_hashes.begin(), training.start_goal_hashes.end());
    result.development_start_goal_hash_values.assign(
        development.start_goal_hashes.begin(), development.start_goal_hashes.end());
    result.evaluation_start_goal_hash_values.assign(
        evaluation.start_goal_hashes.begin(), evaluation.start_goal_hashes.end());
    std::sort(result.training_route_hash_values.begin(),
        result.training_route_hash_values.end());
    std::sort(result.development_route_hash_values.begin(),
        result.development_route_hash_values.end());
    std::sort(result.evaluation_route_hash_values.begin(),
        result.evaluation_route_hash_values.end());
    std::sort(result.training_start_goal_hash_values.begin(),
        result.training_start_goal_hash_values.end());
    std::sort(result.development_start_goal_hash_values.begin(),
        result.development_start_goal_hash_values.end());
    std::sort(result.evaluation_start_goal_hash_values.begin(),
        result.evaluation_start_goal_hash_values.end());
    result.leakage_audit = {
        .no_context_prefix = true,
        .no_operator_labels_in_evaluation = true,
        .no_route_length_in_evaluation = true,
        .no_complete_route_overlap = overlap == 0U,
        .no_exact_start_goal_overlap = true,
        .no_target_access_during_execution = true,
        .no_oracle_data_in_rlf_execution = true,
        .no_result_cache = true,
        .no_seed_overlap =
            (config.seed ^ 0x545241494EULL) !=
            (config.seed ^ 0x4556414CULL),
        .no_training_evaluation_length_overlap =
            config.evaluation_min_route_length > config.training_max_route_length,
        .training_route_hashes = training.route_hashes.size(),
        .development_route_hashes = development.route_hashes.size(),
        .evaluation_route_hashes = evaluation.route_hashes.size(),
        .route_hash_overlap = overlap,
        .start_goal_hash_overlap = pair_overlap,
        .manifest_hash = training.manifest_hash ^ development.manifest_hash ^
            evaluation.manifest_hash,
    };

    if (result.rlf.final_state_accuracy > result.supervised.final_state_accuracy &&
        result.rlf.final_state_accuracy >= 0.70 &&
        result.rlf_search_assisted.final_state_accuracy >=
            result.rlf.final_state_accuracy) {
        result.scientific_decision =
            "A — strong evidence: autonomous state-driven routing exceeds the compact supervised baseline on withheld routes";
    } else if (result.rlf.final_state_accuracy > 0.0 ||
               result.rlf_search_assisted.final_state_accuracy >
                   std::max(result.greedy.final_state_accuracy,
                            result.nearest_route.final_state_accuracy)) {
        result.scientific_decision =
            "B — partial evidence: bounded resonant search is useful, but autonomous single-route routing does not establish superiority over the strongest compact baseline";
    } else {
        result.scientific_decision =
            "C — negative evidence: the learned router does not generalize withheld routes beyond compact controls";
    }
    result.limitations = {
        "operator vocabulary remains predefined",
        "training route discovery uses bounded breadth-first search",
        "the search-assisted RLF result uses explicitly bounded receding-horizon beam search and is reported separately from autonomous single-route execution",
        "state distributions come from four controlled overlapping prototypes",
        "exact routing-mode and route-memory retrieval remain linear",
        "goal state is explicitly supplied and may make some actions greedily identifiable",
        "the benchmark is synthetic and does not establish language or general intelligence",
        "macro acceptance uses occurrence and cost heuristics rather than a full held-out MDL proof",
    };

    core::LatentRouter trace_router = core::LatentRouter::from_snapshot(
        router.snapshot()
    );
    result.representative_trace = trace_router.execute(
        evaluation.episodes.front().start,
        evaluation.episodes.front().goal
    );

    std::uint64_t run_hash = fnv_offset_basis;
    hash_u64(run_hash, config.seed);
    hash_u64(run_hash, training.manifest_hash);
    hash_u64(run_hash, development.manifest_hash);
    hash_u64(run_hash, evaluation.manifest_hash);
    hash_double(run_hash, result.rlf.final_state_accuracy);
    hash_double(run_hash, result.rlf_search_assisted.final_state_accuracy);
    hash_double(run_hash, result.greedy.final_state_accuracy);
    hash_double(run_hash, result.supervised.final_state_accuracy);
    hash_double(run_hash, result.oracle.final_state_accuracy);
    hash_u64(run_hash, result.routing_mode_count);
    hash_u64(run_hash, successful_training);
    hash_u64(run_hash, core::LatentRouter::route_hash(
        result.representative_trace.route));
    result.deterministic_run_hash = run_hash;
    return result;
}

Rlf1TrainingWorkflowResult train_rlf1_checkpoint(
    const Rlf1Config& config,
    const std::filesystem::path& checkpoint_path
) {
    validate_config(config);
    const std::vector<NamedOperator> definitions = make_operator_definitions(
        config.dimension, config.operator_count);
    const Manifest training = generate_manifest(
        config, definitions, config.training_episodes,
        config.training_min_route_length, config.training_max_route_length,
        config.seed ^ 0x545241494EULL);
    std::size_t successful = 0U;
    core::LatentRouter router = train_router(
        config, definitions, training.episodes,
        core::LatentCreditStrategy::discounted_eligibility, &successful);
    storage::save_rlf1_checkpoint(checkpoint_path, router);
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, training.manifest_hash);
    hash_u64(hash, router.training_step());
    hash_u64(hash, router.modes().size());
    return {
        .checkpoint_path = checkpoint_path,
        .seed = config.seed,
        .training_episodes = config.training_episodes,
        .successful_episodes = successful,
        .routing_modes = router.modes().size(),
        .halt_modes = router.halt_modes().size(),
        .route_records = router.route_memory().size(),
        .operators = router.operators().size(),
        .deterministic_run_hash = hash,
    };
}

Rlf1EvaluationWorkflowResult evaluate_rlf1_checkpoint(
    const std::filesystem::path& checkpoint_path,
    const std::uint64_t seed,
    const std::size_t evaluation_episodes
) {
    core::LatentRouter router = storage::load_rlf1_checkpoint(checkpoint_path);
    const std::vector<NamedOperator> definitions = definitions_from_router(router);
    Rlf1Config config;
    config.seed = seed;
    config.dimension = router.config().dimension;
    config.training_episodes = 1U;
    config.evaluation_episodes = evaluation_episodes;
    config.training_min_route_length = 1U;
    config.training_max_route_length = 4U;
    config.evaluation_min_route_length = 5U;
    config.evaluation_max_route_length = std::min<std::size_t>(
        8U, router.config().maximum_cycles);
    config.maximum_cycles = router.config().maximum_cycles;
    config.operator_count = definitions.size();
    config.state_noise_radians = 0.03;
    config.goal_similarity_threshold = router.config().goal_similarity_threshold;
    const Manifest evaluation = generate_manifest(
        config, definitions, evaluation_episodes,
        config.evaluation_min_route_length, config.evaluation_max_route_length,
        seed ^ 0x4556414CULL);
    Rlf1SystemMetrics metrics = evaluate_system(
        "rlf1_checkpoint_single_route", evaluation.episodes,
        [&router](const Episode& episode) {
            return router.execute(episode.start, episode.goal);
        });
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, evaluation.manifest_hash);
    hash_double(hash, metrics.final_state_accuracy);
    hash_double(hash, metrics.mean_goal_similarity);
    return {
        .checkpoint_path = checkpoint_path,
        .evaluation_episodes = evaluation_episodes,
        .metrics = std::move(metrics),
        .deterministic_run_hash = hash,
    };
}

Rlf1TraceWorkflowResult trace_rlf1_checkpoint(
    const std::filesystem::path& checkpoint_path,
    const std::uint64_t seed,
    const std::size_t sample_id
) {
    core::LatentRouter router = storage::load_rlf1_checkpoint(checkpoint_path);
    const std::vector<NamedOperator> definitions = definitions_from_router(router);
    Rlf1Config config;
    config.seed = seed;
    config.dimension = router.config().dimension;
    config.training_episodes = 1U;
    config.evaluation_episodes = sample_id + 1U;
    config.evaluation_min_route_length = 5U;
    config.evaluation_max_route_length = std::min<std::size_t>(
        8U, router.config().maximum_cycles);
    config.maximum_cycles = router.config().maximum_cycles;
    config.operator_count = definitions.size();
    config.goal_similarity_threshold = router.config().goal_similarity_threshold;
    const Manifest evaluation = generate_manifest(
        config, definitions, sample_id + 1U,
        config.evaluation_min_route_length, config.evaluation_max_route_length,
        seed ^ 0x5452414345ULL);
    const Episode& episode = evaluation.episodes[sample_id];
    return {
        .checkpoint_path = checkpoint_path,
        .sample_id = sample_id,
        .route_hash = episode.route_hash,
        .execution = router.execute(episode.start, episode.goal),
    };
}

void write_rlf1_training_json(
    std::ostream& output,
    const Rlf1TrainingWorkflowResult& result
) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "{\n"
           << "  \"architecture\": \"RLF-1 latent state-driven routing\",\n"
           << "  \"workflow\": \"train\",\n"
           << "  \"checkpoint\": \"" << json_escape(result.checkpoint_path.string()) << "\",\n"
           << "  \"seed\": " << result.seed << ",\n"
           << "  \"training_episodes\": " << result.training_episodes << ",\n"
           << "  \"successful_episodes\": " << result.successful_episodes << ",\n"
           << "  \"routing_modes\": " << result.routing_modes << ",\n"
           << "  \"halt_modes\": " << result.halt_modes << ",\n"
           << "  \"route_records\": " << result.route_records << ",\n"
           << "  \"operators\": " << result.operators << ",\n"
           << "  \"deterministic_run_hash\": \"" << format_run_hash(result.deterministic_run_hash) << "\"\n"
           << "}\n";
}

void write_rlf1_evaluation_json(
    std::ostream& output,
    const Rlf1EvaluationWorkflowResult& result
) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "{\n"
           << "  \"architecture\": \"RLF-1 latent state-driven routing\",\n"
           << "  \"workflow\": \"evaluate\",\n"
           << "  \"checkpoint\": \"" << json_escape(result.checkpoint_path.string()) << "\",\n"
           << "  \"evaluation_episodes\": " << result.evaluation_episodes << ",\n"
           << "  \"metrics\": ";
    write_system_metrics(output, result.metrics, 2U);
    output << ",\n  \"deterministic_run_hash\": \""
           << format_run_hash(result.deterministic_run_hash) << "\"\n}\n";
}

void write_rlf1_trace_json(
    std::ostream& output,
    const Rlf1TraceWorkflowResult& result
) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "{\n"
           << "  \"architecture\": \"RLF-1 latent state-driven routing\",\n"
           << "  \"workflow\": \"trace\",\n"
           << "  \"checkpoint\": \"" << json_escape(result.checkpoint_path.string()) << "\",\n"
           << "  \"sample_id\": " << result.sample_id << ",\n"
           << "  \"hidden_route_hash\": \"" << format_run_hash(result.route_hash) << "\",\n"
           << "  \"trace_is_structured_latent_execution_not_natural_language_cot\": true,\n"
           << "  \"execution\": ";
    write_execution(output, result.execution, 2U);
    output << "\n}\n";
}

void write_rlf1_latent_routing_json(
    std::ostream& output,
    const Rlf1Result& result
) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "{\n"
           << "  \"architecture\": \"RLF-1 latent state-driven routing\",\n"
           << "  \"seed\": " << result.seed << ",\n"
           << "  \"dimension\": " << result.dimension << ",\n"
           << "  \"training_episodes\": " << result.training_episodes << ",\n"
           << "  \"development_episodes\": " << result.development_episodes << ",\n"
           << "  \"evaluation_episodes\": " << result.evaluation_episodes << ",\n"
           << "  \"systems\": {\n";
    const std::vector<std::pair<std::string_view, const Rlf1SystemMetrics*>> systems{
        {"rlf", &result.rlf},
        {"development_search_selection", &result.development_search_selection},
        {"rlf_search_assisted", &result.rlf_search_assisted},
        {"rlf_without_route_memory", &result.rlf_without_route_memory},
        {"greedy", &result.greedy},
        {"nearest_route", &result.nearest_route},
        {"supervised", &result.supervised},
        {"oracle", &result.oracle},
    };
    for (std::size_t index = 0U; index < systems.size(); ++index) {
        output << "    \"" << systems[index].first << "\": ";
        write_system_metrics(output, *systems[index].second, 4U);
        output << (index + 1U == systems.size() ? "\n" : ",\n");
    }
    output << "  },\n  \"accuracy_by_route_length\": [";
    if (!result.by_length.empty()) output << '\n';
    for (std::size_t index = 0U; index < result.by_length.size(); ++index) {
        const Rlf1LengthMetrics& value = result.by_length[index];
        output << "    {\n"
               << "      \"route_length\": " << value.route_length << ",\n"
               << "      \"episodes\": " << value.episodes << ",\n"
               << "      \"rlf\": " << value.rlf_accuracy << ",\n"
               << "      \"rlf_search_assisted\": " << value.rlf_search_accuracy << ",\n"
               << "      \"greedy\": " << value.greedy_accuracy << ",\n"
               << "      \"supervised\": " << value.supervised_accuracy << ",\n"
               << "      \"oracle\": " << value.oracle_accuracy << "\n"
               << "    }" << (index + 1U == result.by_length.size() ? "\n" : ",\n");
    }
    output << "  ],\n  \"halt_policies\": [";
    if (!result.halt_policies.empty()) output << '\n';
    for (std::size_t index = 0U; index < result.halt_policies.size(); ++index) {
        const Rlf1HaltMetrics& value = result.halt_policies[index];
        output << "    {\"policy\": \"" << json_escape(value.policy)
               << "\", \"accuracy\": " << value.accuracy
               << ", \"halt_precision\": " << value.halt_precision
               << ", \"halt_recall\": " << value.halt_recall
               << ", \"premature_halt_rate\": " << value.premature_halt_rate
               << ", \"average_cycles\": " << value.average_cycles
               << "}" << (index + 1U == result.halt_policies.size() ? "\n" : ",\n");
    }
    output << "  ],\n  \"delayed_credit\": [";
    if (!result.delayed_credit.empty()) output << '\n';
    for (std::size_t index = 0U; index < result.delayed_credit.size(); ++index) {
        const Rlf1CreditMetrics& value = result.delayed_credit[index];
        output << "    {\"strategy\": \"" << json_escape(value.strategy)
               << "\", \"accuracy\": " << value.accuracy
               << ", \"mean_goal_similarity\": " << value.mean_goal_similarity
               << ", \"mode_count\": " << value.mode_count << "}"
               << (index + 1U == result.delayed_credit.size() ? "\n" : ",\n");
    }
    output << "  ],\n  \"scaling\": [";
    if (!result.scaling.empty()) output << '\n';
    for (std::size_t index = 0U; index < result.scaling.size(); ++index) {
        const Rlf1ScalingMetrics& value = result.scaling[index];
        output << "    {\"training_episodes\": " << value.training_episodes
               << ", \"physical_modes\": " << value.physical_modes
               << ", \"accuracy\": " << value.accuracy
               << ", \"average_exact_similarities\": " << value.average_exact_similarities
               << ", \"average_active_operations\": " << value.average_active_operations
               << "}" << (index + 1U == result.scaling.size() ? "\n" : ",\n");
    }
    output << "  ],\n"
           << "  \"noise_robustness_accuracy\": " << result.noise_robustness_accuracy << ",\n"
           << "  \"recovery_accuracy\": " << result.recovery_accuracy << ",\n"
           << "  \"recovery_search_accuracy\": " << result.recovery_search_accuracy << ",\n"
           << "  \"ambiguity_abstention_rate\": " << result.ambiguity_abstention_rate << ",\n"
           << "  \"ambiguity_search_abstention_rate\": " << result.ambiguity_search_abstention_rate << ",\n"
           << "  \"impossible_case_false_success_rate\": " << result.impossible_case_false_success_rate << ",\n"
           << "  \"continual_retained_accuracy\": " << result.continual_retained_accuracy << ",\n"
           << "  \"continual_forgetting\": " << result.continual_forgetting << ",\n"
           << "  \"macros_proposed\": " << result.macros_proposed << ",\n"
           << "  \"macros_created\": " << result.macros_created << ",\n"
           << "  \"macros_rejected\": " << result.macros_rejected << ",\n"
           << "  \"macro_validation_accuracy\": " << result.macro_validation_accuracy << ",\n"
           << "  \"macro_cycle_reduction\": " << result.macro_cycle_reduction << ",\n"
           << "  \"macro_interference\": " << result.macro_interference << ",\n"
           << "  \"routing_mode_count\": " << result.routing_mode_count << ",\n"
           << "  \"halt_mode_count\": " << result.halt_mode_count << ",\n"
           << "  \"route_memory_records\": " << result.route_memory_records << ",\n"
           << "  \"operator_count\": " << result.operator_count << ",\n"
           << "  \"estimated_bytes\": " << result.estimated_bytes << ",\n"
           << "  \"local_update_operations\": " << result.local_update_operations << ",\n"
           << "  \"training_seconds\": " << result.training_seconds << ",\n"
           << "  \"selected_lookahead_depth\": " << result.selected_lookahead_depth << ",\n"
           << "  \"selected_beam_width\": " << result.selected_beam_width << ",\n"
           << "  \"deterministic_run_hash\": \"" << format_run_hash(result.deterministic_run_hash) << "\",\n"
           << "  \"leakage_audit\": {\n"
           << "    \"no_context_prefix\": " << (result.leakage_audit.no_context_prefix ? "true" : "false") << ",\n"
           << "    \"no_operator_labels_in_evaluation\": " << (result.leakage_audit.no_operator_labels_in_evaluation ? "true" : "false") << ",\n"
           << "    \"no_route_length_in_evaluation\": " << (result.leakage_audit.no_route_length_in_evaluation ? "true" : "false") << ",\n"
           << "    \"no_complete_route_overlap\": " << (result.leakage_audit.no_complete_route_overlap ? "true" : "false") << ",\n"
           << "    \"no_exact_start_goal_overlap\": " << (result.leakage_audit.no_exact_start_goal_overlap ? "true" : "false") << ",\n"
           << "    \"no_target_access_during_execution\": " << (result.leakage_audit.no_target_access_during_execution ? "true" : "false") << ",\n"
           << "    \"no_oracle_data_in_rlf_execution\": " << (result.leakage_audit.no_oracle_data_in_rlf_execution ? "true" : "false") << ",\n"
           << "    \"no_result_cache\": " << (result.leakage_audit.no_result_cache ? "true" : "false") << ",\n"
           << "    \"no_seed_overlap\": " << (result.leakage_audit.no_seed_overlap ? "true" : "false") << ",\n"
           << "    \"no_training_evaluation_length_overlap\": " << (result.leakage_audit.no_training_evaluation_length_overlap ? "true" : "false") << ",\n"
           << "    \"training_route_hashes\": " << result.leakage_audit.training_route_hashes << ",\n"
           << "    \"development_route_hashes\": " << result.leakage_audit.development_route_hashes << ",\n"
           << "    \"evaluation_route_hashes\": " << result.leakage_audit.evaluation_route_hashes << ",\n"
           << "    \"route_hash_overlap\": " << result.leakage_audit.route_hash_overlap << ",\n"
           << "    \"start_goal_hash_overlap\": " << result.leakage_audit.start_goal_hash_overlap << ",\n"
           << "    \"manifest_hash\": \"" << format_run_hash(result.leakage_audit.manifest_hash) << "\"\n"
           << "  },\n"
           << "  \"dataset_manifest\": {\n";
    const auto write_hash_array = [&output](
        const std::string_view name,
        const std::vector<std::uint64_t>& values,
        const bool trailing
    ) {
        output << "    \"" << name << "\": [";
        for (std::size_t index = 0U; index < values.size(); ++index) {
            if (index != 0U) output << ", ";
            output << "\"" << format_run_hash(values[index]) << "\"";
        }
        output << "]" << (trailing ? ",\n" : "\n");
    };
    write_hash_array("training_route_hashes", result.training_route_hash_values, true);
    write_hash_array("development_route_hashes", result.development_route_hash_values, true);
    write_hash_array("evaluation_route_hashes", result.evaluation_route_hash_values, true);
    write_hash_array("training_start_goal_hashes", result.training_start_goal_hash_values, true);
    write_hash_array("development_start_goal_hashes", result.development_start_goal_hash_values, true);
    write_hash_array("evaluation_start_goal_hashes", result.evaluation_start_goal_hash_values, false);
    output << "  },\n"
           << "  \"scientific_decision\": \"" << json_escape(result.scientific_decision) << "\",\n"
           << "  \"limitations\": [";
    for (std::size_t index = 0U; index < result.limitations.size(); ++index) {
        if (index != 0U) output << ", ";
        output << "\"" << json_escape(result.limitations[index]) << "\"";
    }
    output << "],\n  \"representative_trace\": ";
    write_execution(output, result.representative_trace, 2U);
    output << "\n}\n";
}

}  // namespace rlf::experiments
