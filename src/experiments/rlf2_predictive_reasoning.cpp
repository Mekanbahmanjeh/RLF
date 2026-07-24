#include "rlf/experiments/rlf2_predictive_reasoning.hpp"

#include "rlf/core/deterministic_rng.hpp"
#include "rlf/core/latent_routing.hpp"
#include "rlf/experiments/phase_vector_smoke.hpp"
#include "rlf/storage/rlf2_checkpoint.hpp"

#include <algorithm>
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
            "RLF-2 requires dimension >= 2 and 6-8 operators"
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
    core::PredictiveSkillFabric& fabric,
    const std::vector<NamedOperator>& definitions
) {
    std::vector<std::uint64_t> ids;
    ids.reserve(definitions.size());
    for (const NamedOperator& definition : definitions) {
        ids.push_back(fabric.register_operator(
            definition.name,
            definition.transformation,
            definition.cost
        ));
    }
    return ids;
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
        if (operator_id == 0ULL || operator_id > definitions.size()) {
            throw std::out_of_range("RLF-2 route references unknown operator");
        }
        state = definitions[operator_id - 1U].transformation.apply(state);
    }
    return state;
}

[[nodiscard]] std::uint64_t route_hash(
    const std::span<const std::uint64_t> route
) noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, static_cast<std::uint64_t>(route.size()));
    for (const std::uint64_t value : route) {
        hash_u64(hash, value);
    }
    return hash;
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

void validate_config(const Rlf2Config& config) {
    if (config.dimension < 2U || config.training_episodes == 0U ||
        config.development_episodes == 0U ||
        config.evaluation_episodes == 0U ||
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
        throw std::invalid_argument("invalid RLF-2 configuration");
    }
}

[[nodiscard]] Manifest generate_manifest(
    const Rlf2Config& config,
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
            throw std::runtime_error(
                "unable to construct leakage-free RLF-2 manifest"
            );
        }
        const std::size_t length = minimum_length + rng.uniform_index(
            maximum_length - minimum_length + 1U
        );
        std::vector<std::uint64_t> route;
        route.reserve(length);
        for (std::size_t step = 0U; step < length; ++step) {
            std::uint64_t selected = static_cast<std::uint64_t>(
                rng.uniform_index(definitions.size()) + 1U
            );
            if (!route.empty()) {
                const std::uint64_t previous = route.back();
                const bool inverse_pair =
                    (previous == 1ULL && selected == 2ULL) ||
                    (previous == 2ULL && selected == 1ULL) ||
                    (previous == 3ULL && selected == 4ULL) ||
                    (previous == 4ULL && selected == 3ULL);
                if (inverse_pair) {
                    selected = static_cast<std::uint64_t>(
                        (static_cast<std::size_t>(selected) %
                         definitions.size()) + 1U
                    );
                }
            }
            route.push_back(selected);
        }
        const std::uint64_t route_value = route_hash(route);
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
        if (start.similarity(goal) >= config.goal_similarity_threshold) {
            continue;
        }
        const std::uint64_t pair_value = start_goal_hash(start, goal);
        if (forbidden_pairs.contains(pair_value) ||
            manifest.start_goal_hashes.contains(pair_value)) {
            continue;
        }
        manifest.route_hashes.insert(route_value);
        manifest.start_goal_hashes.insert(pair_value);
        manifest.episodes.push_back({
            .start = std::move(start),
            .goal = std::move(goal),
            .route = route,
            .route_hash = route_value,
            .start_goal_hash = pair_value,
        });
        hash_u64(manifest.manifest_hash, route_value);
        hash_u64(manifest.manifest_hash, pair_value);
    }
    return manifest;
}

[[nodiscard]] core::PredictiveSkillConfig make_fabric_config(
    const Rlf2Config& config
) {
    core::PredictiveSkillConfig result;
    result.dimension = config.dimension;
    result.maximum_cycles = config.maximum_cycles;
    result.maximum_route_depth = std::max(
        config.training_max_route_length,
        config.evaluation_max_route_length + 2U
    );
    result.planner_node_budget = 750'000U;
    result.maximum_skills = 8'192U;
    result.maximum_subgoal_prototypes = std::max<std::size_t>(
        8'192U,
        config.training_episodes * config.training_max_route_length * 4U
    );
    result.maximum_skill_length = config.training_max_route_length;
    result.minimum_skill_support = 2U;
    result.nearest_prototypes = 5U;
    result.goal_similarity_threshold = config.goal_similarity_threshold;
    result.prototype_merge_distance = 0.012;
    result.enable_skill_consolidation = true;
    result.enable_intervention_credit = true;
    return result;
}

[[nodiscard]] core::LatentRouterConfig make_rlf1_config(
    const Rlf2Config& config
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
    result.enable_route_memory = true;
    result.enable_macro_operators = false;
    result.credit_strategy = core::LatentCreditStrategy::discounted_eligibility;
    result.halt_policy = core::LatentHaltPolicy::combined_safe;
    return result;
}

[[nodiscard]] core::PredictiveSkillFabric train_fabric(
    const Rlf2Config& config,
    const std::vector<NamedOperator>& definitions,
    const std::vector<Episode>& episodes,
    std::size_t* const successes
) {
    core::PredictiveSkillFabric fabric(
        make_fabric_config(config),
        config.seed ^ 0x50524544494354ULL
    );
    static_cast<void>(register_operators(fabric, definitions));
    std::size_t successful = 0U;
    for (const Episode& episode : episodes) {
        std::size_t nodes = 0U;
        const auto discovered = fabric.plan_primitive_bridge(
            episode.start,
            episode.goal,
            config.training_max_route_length,
            &nodes,
            nullptr,
            nullptr,
            false
        );
        if (!discovered.has_value()) {
            continue;
        }
        if (discovered->empty()) {
            ++successful;
            continue;
        }
        fabric.observe_successful_route(
            episode.start,
            episode.goal,
            *discovered
        );
        ++successful;
    }
    static_cast<void>(fabric.consolidate_skills());
    if (successes != nullptr) {
        *successes = successful;
    }
    return fabric;
}

[[nodiscard]] core::LatentRouter train_rlf1(
    const Rlf2Config& config,
    const std::vector<NamedOperator>& definitions,
    const std::vector<Episode>& episodes
) {
    core::LatentRouter router(
        make_rlf1_config(config),
        config.seed ^ 0x524C463142415345ULL
    );
    static_cast<void>(register_operators(router, definitions));
    for (const Episode& episode : episodes) {
        static_cast<void>(router.train_episode(
            episode.start,
            episode.goal,
            config.training_max_route_length
        ));
    }
    return router;
}

[[nodiscard]] core::Rlf2ExecutionResult greedy_execute(
    const Rlf2Config& config,
    const std::vector<NamedOperator>& definitions,
    const Episode& episode
) {
    core::Rlf2ExecutionResult result;
    result.final_state = episode.start;
    std::unordered_set<std::uint64_t> visited;
    visited.insert(core::LatentRouter::phase_state_hash(result.final_state));
    for (std::size_t cycle = 0U; cycle < config.maximum_cycles; ++cycle) {
        if (result.final_state.similarity(episode.goal) >=
            config.goal_similarity_threshold) {
            result.success = true;
            result.stop_reason = core::Rlf2StopReason::successful_halt;
            break;
        }
        std::size_t best_index = 0U;
        double best_similarity = -1.0;
        for (std::size_t index = 0U; index < definitions.size(); ++index) {
            const core::PhaseVector candidate =
                definitions[index].transformation.apply(result.final_state);
            const double similarity = candidate.similarity(episode.goal);
            if (similarity > best_similarity + 1.0e-12 ||
                (std::abs(similarity - best_similarity) <= 1.0e-12 &&
                 index < best_index)) {
                best_index = index;
                best_similarity = similarity;
            }
        }
        result.final_state =
            definitions[best_index].transformation.apply(result.final_state);
        result.primitive_route.push_back(
            static_cast<std::uint64_t>(best_index + 1U)
        );
        ++result.primitive_steps;
        ++result.cycles;
        const std::uint64_t hash =
            core::LatentRouter::phase_state_hash(result.final_state);
        if (!visited.insert(hash).second) {
            result.stop_reason = core::Rlf2StopReason::loop_detected;
            break;
        }
    }
    result.success = result.final_state.similarity(episode.goal) >=
        config.goal_similarity_threshold;
    if (result.success) {
        result.stop_reason = core::Rlf2StopReason::successful_halt;
    }
    result.final_goal_similarity = result.final_state.similarity(episode.goal);
    return result;
}

[[nodiscard]] core::Rlf2ExecutionResult convert_rlf1(
    const core::LatentExecutionResult& source
) {
    core::Rlf2ExecutionResult result;
    result.final_state = source.final_state;
    result.primitive_route = source.route;
    result.success = source.success;
    result.abstained = source.abstained;
    result.cycles = source.cycles;
    result.primitive_steps = source.route.size();
    result.planner_nodes = source.search_nodes;
    result.final_goal_similarity = source.final_goal_similarity;
    result.mean_uncertainty = source.mean_uncertainty;
    if (source.success) {
        result.stop_reason = core::Rlf2StopReason::successful_halt;
    } else if (source.abstained) {
        result.stop_reason = core::Rlf2StopReason::abstained;
    } else {
        result.stop_reason = core::Rlf2StopReason::cycle_limit;
    }
    return result;
}

[[nodiscard]] core::Rlf2ExecutionResult oracle_execute(
    const std::vector<NamedOperator>& definitions,
    const Episode& episode
) {
    core::Rlf2ExecutionResult result;
    result.final_state = apply_route(definitions, episode.start, episode.route);
    result.primitive_route = episode.route;
    result.primitive_steps = episode.route.size();
    result.cycles = episode.route.size();
    result.success = true;
    result.stop_reason = core::Rlf2StopReason::successful_halt;
    result.final_goal_similarity = result.final_state.similarity(episode.goal);
    return result;
}

template <typename Executor>
[[nodiscard]] Rlf2SystemMetrics evaluate_system(
    std::string name,
    const std::vector<Episode>& episodes,
    Executor&& executor
) {
    const auto begin = std::chrono::steady_clock::now();
    Rlf2SystemMetrics metrics;
    metrics.name = std::move(name);
    metrics.episodes = episodes.size();
    std::size_t successes = 0U;
    std::size_t exact_routes = 0U;
    std::size_t first_actions = 0U;
    std::size_t abstentions = 0U;
    double similarity_total = 0.0;
    double cycles_total = 0.0;
    double primitive_total = 0.0;
    double planner_total = 0.0;
    double forward_total = 0.0;
    double backward_total = 0.0;
    double subgoal_total = 0.0;
    double uncertainty_total = 0.0;
    for (const Episode& episode : episodes) {
        const core::Rlf2ExecutionResult result = executor(episode);
        successes += result.success ? 1U : 0U;
        exact_routes += result.primitive_route == episode.route ? 1U : 0U;
        first_actions += !result.primitive_route.empty() &&
            !episode.route.empty() &&
            result.primitive_route.front() == episode.route.front() ? 1U : 0U;
        abstentions += result.abstained ? 1U : 0U;
        similarity_total += result.final_goal_similarity;
        cycles_total += static_cast<double>(result.cycles);
        primitive_total += static_cast<double>(result.primitive_steps);
        planner_total += static_cast<double>(result.planner_nodes);
        forward_total += static_cast<double>(result.forward_nodes);
        backward_total += static_cast<double>(result.backward_nodes);
        subgoal_total += static_cast<double>(result.subgoals_considered);
        uncertainty_total += result.mean_uncertainty;
    }
    const auto end = std::chrono::steady_clock::now();
    const double denominator = episodes.empty()
        ? 1.0
        : static_cast<double>(episodes.size());
    metrics.final_state_accuracy = static_cast<double>(successes) / denominator;
    metrics.exact_route_accuracy = static_cast<double>(exact_routes) / denominator;
    metrics.first_action_accuracy = static_cast<double>(first_actions) / denominator;
    metrics.mean_goal_similarity = similarity_total / denominator;
    metrics.average_cycles = cycles_total / denominator;
    metrics.average_primitive_steps = primitive_total / denominator;
    metrics.average_planner_nodes = planner_total / denominator;
    metrics.average_forward_nodes = forward_total / denominator;
    metrics.average_backward_nodes = backward_total / denominator;
    metrics.average_subgoals = subgoal_total / denominator;
    metrics.abstention_rate = static_cast<double>(abstentions) / denominator;
    metrics.mean_uncertainty = uncertainty_total / denominator;
    metrics.inference_seconds = std::chrono::duration<double>(end - begin).count();
    return metrics;
}

[[nodiscard]] std::size_t estimate_bytes(
    const core::PredictiveSkillFabric& fabric
) {
    std::size_t bytes = sizeof(fabric);
    for (const core::PredictiveOperator& value : fabric.operators()) {
        bytes += sizeof(value) + value.name.capacity();
        for (const core::OperatorPrimitive& primitive :
             value.forward.primitives()) {
            bytes += primitive.phase_shift.size() * sizeof(float);
            bytes += primitive.permutation.size() * sizeof(std::size_t);
        }
    }
    for (const core::CausalSkill& skill : fabric.skills()) {
        bytes += sizeof(skill) + skill.name.capacity();
        bytes += skill.primitive_route.capacity() * sizeof(std::uint64_t);
    }
    for (const core::SubgoalPrototype& prototype : fabric.prototypes()) {
        bytes += sizeof(prototype);
        bytes += prototype.response_profile.capacity() * sizeof(float);
    }
    return bytes;
}

void write_system_metrics(
    std::ostream& output,
    const Rlf2SystemMetrics& value,
    const std::size_t indentation
) {
    const std::string indent(indentation, ' ');
    const std::string field(indentation + 2U, ' ');
    output << indent << "{\n"
           << field << "\"name\": \"" << json_escape(value.name) << "\",\n"
           << field << "\"episodes\": " << value.episodes << ",\n"
           << field << "\"final_state_accuracy\": " << value.final_state_accuracy << ",\n"
           << field << "\"exact_route_accuracy\": " << value.exact_route_accuracy << ",\n"
           << field << "\"first_action_accuracy\": " << value.first_action_accuracy << ",\n"
           << field << "\"mean_goal_similarity\": " << value.mean_goal_similarity << ",\n"
           << field << "\"average_cycles\": " << value.average_cycles << ",\n"
           << field << "\"average_primitive_steps\": " << value.average_primitive_steps << ",\n"
           << field << "\"average_planner_nodes\": " << value.average_planner_nodes << ",\n"
           << field << "\"average_forward_nodes\": " << value.average_forward_nodes << ",\n"
           << field << "\"average_backward_nodes\": " << value.average_backward_nodes << ",\n"
           << field << "\"average_subgoals\": " << value.average_subgoals << ",\n"
           << field << "\"abstention_rate\": " << value.abstention_rate << ",\n"
           << field << "\"mean_uncertainty\": " << value.mean_uncertainty << ",\n"
           << field << "\"inference_seconds\": " << value.inference_seconds << "\n"
           << indent << '}';
}

void write_execution(
    std::ostream& output,
    const core::Rlf2ExecutionResult& value,
    const std::size_t indentation
) {
    const std::string indent(indentation, ' ');
    const std::string field(indentation + 2U, ' ');
    output << indent << "{\n"
           << field << "\"success\": " << (value.success ? "true" : "false") << ",\n"
           << field << "\"abstained\": " << (value.abstained ? "true" : "false") << ",\n"
           << field << "\"stop_reason\": \"" << core::to_string(value.stop_reason) << "\",\n"
           << field << "\"cycles\": " << value.cycles << ",\n"
           << field << "\"primitive_steps\": " << value.primitive_steps << ",\n"
           << field << "\"planner_nodes\": " << value.planner_nodes << ",\n"
           << field << "\"forward_nodes\": " << value.forward_nodes << ",\n"
           << field << "\"backward_nodes\": " << value.backward_nodes << ",\n"
           << field << "\"subgoals_considered\": " << value.subgoals_considered << ",\n"
           << field << "\"final_goal_similarity\": " << value.final_goal_similarity << ",\n"
           << field << "\"primitive_route\": [";
    for (std::size_t index = 0U; index < value.primitive_route.size(); ++index) {
        output << value.primitive_route[index];
        if (index + 1U != value.primitive_route.size()) output << ", ";
    }
    output << "],\n" << field << "\"skill_route\": [";
    for (std::size_t index = 0U; index < value.skill_route.size(); ++index) {
        output << value.skill_route[index];
        if (index + 1U != value.skill_route.size()) output << ", ";
    }
    output << "],\n" << field << "\"trace\": [";
    if (!value.trace.empty()) {
        output << '\n';
        for (std::size_t index = 0U; index < value.trace.size(); ++index) {
            const core::Rlf2TraceStep& step = value.trace[index];
            const std::string step_indent(indentation + 4U, ' ');
            const std::string step_field(indentation + 6U, ' ');
            output << step_indent << "{\n"
                   << step_field << "\"cycle\": " << step.cycle << ",\n"
                   << step_field << "\"state_hash\": \"" << format_run_hash(step.state_hash) << "\",\n"
                   << step_field << "\"selected_skill_id\": " << step.selected_skill_id << ",\n"
                   << step_field << "\"selected_skill_name\": \"" << json_escape(step.selected_skill_name) << "\",\n"
                   << step_field << "\"goal_similarity_before\": " << step.goal_similarity_before << ",\n"
                   << step_field << "\"goal_similarity_after\": " << step.goal_similarity_after << ",\n"
                   << step_field << "\"uncertainty\": " << step.uncertainty << ",\n"
                   << step_field << "\"predicted_remaining_steps\": " << step.predicted_remaining_steps << ",\n"
                   << step_field << "\"bridge_subgoal\": " << (step.bridge_subgoal ? "true" : "false") << "\n"
                   << step_indent << '}';
            output << (index + 1U == value.trace.size() ? "\n" : ",\n");
        }
        output << field;
    }
    output << "]\n" << indent << '}';
}

}  // namespace

Rlf2Result run_rlf2_predictive_reasoning(const Rlf2Config& config) {
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
        config.seed ^ 0x545241494E32ULL
    );
    const Manifest development = generate_manifest(
        config,
        definitions,
        config.development_episodes,
        2U,
        config.evaluation_max_route_length,
        config.seed ^ 0x44455632ULL,
        training.route_hashes,
        training.start_goal_hashes
    );
    std::unordered_set<std::uint64_t> forbidden_routes = training.route_hashes;
    forbidden_routes.insert(
        development.route_hashes.begin(),
        development.route_hashes.end()
    );
    std::unordered_set<std::uint64_t> forbidden_pairs = training.start_goal_hashes;
    forbidden_pairs.insert(
        development.start_goal_hashes.begin(),
        development.start_goal_hashes.end()
    );
    const Manifest evaluation = generate_manifest(
        config,
        definitions,
        config.evaluation_episodes,
        config.evaluation_min_route_length,
        config.evaluation_max_route_length,
        config.seed ^ 0x4556414C32ULL,
        forbidden_routes,
        forbidden_pairs
    );

    const auto training_begin = std::chrono::steady_clock::now();
    std::size_t training_successes = 0U;
    core::PredictiveSkillFabric fabric = train_fabric(
        config,
        definitions,
        training.episodes,
        &training_successes
    );
    core::LatentRouter rlf1 = train_rlf1(
        config,
        definitions,
        training.episodes
    );
    const auto training_end = std::chrono::steady_clock::now();

    core::PredictiveSkillFabric autonomous_fabric =
        core::PredictiveSkillFabric::from_snapshot(fabric.snapshot());
    core::PredictiveSkillFabric bridge_fabric =
        core::PredictiveSkillFabric::from_snapshot(fabric.snapshot());
    core::PredictiveSkillFabric untrained(
        make_fabric_config(config),
        config.seed ^ 0x554E545241494EULL
    );
    static_cast<void>(register_operators(untrained, definitions));
    core::LatentRouter rlf1_eval = core::LatentRouter::from_snapshot(
        rlf1.snapshot()
    );

    Rlf2Result result;
    result.seed = config.seed;
    result.dimension = config.dimension;
    result.training_episodes = config.training_episodes;
    result.evaluation_episodes = config.evaluation_episodes;
    result.rlf2_autonomous = evaluate_system(
        "rlf2_autonomous",
        evaluation.episodes,
        [&autonomous_fabric](const Episode& episode) {
            return autonomous_fabric.execute_autonomous(
                episode.start,
                episode.goal
            );
        }
    );
    result.rlf2_subgoal_bridge = evaluate_system(
        "rlf2_skill_compressed_subgoal_bridge",
        evaluation.episodes,
        [&bridge_fabric, &config](const Episode& episode) {
            return bridge_fabric.execute_subgoal_bridge(
                episode.start,
                episode.goal,
                config.evaluation_max_route_length,
                false
            );
        }
    );
    result.untrained_bidirectional_bridge = evaluate_system(
        "untrained_bidirectional_bridge",
        evaluation.episodes,
        [&untrained, &config](const Episode& episode) {
            return untrained.execute_subgoal_bridge(
                episode.start,
                episode.goal,
                config.evaluation_max_route_length,
                false
            );
        }
    );
    result.rlf1_autonomous = evaluate_system(
        "rlf1_autonomous",
        evaluation.episodes,
        [&rlf1_eval](const Episode& episode) {
            return convert_rlf1(rlf1_eval.execute(
                episode.start,
                episode.goal
            ));
        }
    );
    result.greedy = evaluate_system(
        "greedy_goal_similarity",
        evaluation.episodes,
        [&config, &definitions](const Episode& episode) {
            return greedy_execute(config, definitions, episode);
        }
    );
    result.oracle = evaluate_system(
        "oracle_route",
        evaluation.episodes,
        [&definitions](const Episode& episode) {
            return oracle_execute(definitions, episode);
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
        core::PredictiveSkillFabric a =
            core::PredictiveSkillFabric::from_snapshot(fabric.snapshot());
        core::PredictiveSkillFabric b =
            core::PredictiveSkillFabric::from_snapshot(fabric.snapshot());
        core::PredictiveSkillFabric u(
            make_fabric_config(config),
            config.seed ^ static_cast<std::uint64_t>(length)
        );
        static_cast<void>(register_operators(u, definitions));
        core::LatentRouter l = core::LatentRouter::from_snapshot(rlf1.snapshot());
        const Rlf2SystemMetrics am = evaluate_system(
            "a", subset,
            [&a](const Episode& episode) {
                return a.execute_autonomous(episode.start, episode.goal);
            }
        );
        const Rlf2SystemMetrics bm = evaluate_system(
            "b", subset,
            [&b, length](const Episode& episode) {
                return b.execute_subgoal_bridge(
                    episode.start, episode.goal, length, false);
            }
        );
        const Rlf2SystemMetrics um = evaluate_system(
            "u", subset,
            [&u, length](const Episode& episode) {
                return u.execute_subgoal_bridge(
                    episode.start, episode.goal, length, false);
            }
        );
        const Rlf2SystemMetrics lm = evaluate_system(
            "l", subset,
            [&l](const Episode& episode) {
                return convert_rlf1(l.execute(episode.start, episode.goal));
            }
        );
        const Rlf2SystemMetrics gm = evaluate_system(
            "g", subset,
            [&config, &definitions](const Episode& episode) {
                return greedy_execute(config, definitions, episode);
            }
        );
        result.by_length.push_back({
            .route_length = length,
            .episodes = subset.size(),
            .autonomous_accuracy = am.final_state_accuracy,
            .bridge_accuracy = bm.final_state_accuracy,
            .untrained_bridge_accuracy = um.final_state_accuracy,
            .rlf1_accuracy = lm.final_state_accuracy,
            .greedy_accuracy = gm.final_state_accuracy,
        });
    }

    const std::size_t recovery_count = std::min<std::size_t>(
        12U,
        evaluation.episodes.size()
    );
    std::size_t recovery_successes = 0U;
    for (std::size_t index = 0U; index < recovery_count; ++index) {
        const Episode& episode = evaluation.episodes[index];
        std::uint64_t wrong =
            (episode.route.front() % definitions.size()) + 1ULL;
        if (wrong == episode.route.front()) {
            wrong = (wrong % definitions.size()) + 1ULL;
        }
        const core::PhaseVector damaged =
            definitions[wrong - 1U].transformation.apply(episode.start);
        core::PredictiveSkillFabric recovery =
            core::PredictiveSkillFabric::from_snapshot(fabric.snapshot());
        const core::Rlf2ExecutionResult repaired =
            recovery.execute_subgoal_bridge(
                damaged,
                episode.goal,
                config.evaluation_max_route_length + 1U,
                false
            );
        recovery_successes += repaired.success ? 1U : 0U;
    }
    result.recovery_accuracy = recovery_count == 0U
        ? 0.0
        : static_cast<double>(recovery_successes) /
            static_cast<double>(recovery_count);

    core::DeterministicRng impossible_rng(config.seed ^ 0x494D504F535332ULL);
    constexpr std::size_t impossible_count = 8U;
    std::size_t false_successes = 0U;
    std::size_t abstentions = 0U;
    for (std::size_t index = 0U; index < impossible_count; ++index) {
        const core::PhaseVector start =
            core::PhaseVector::random(config.dimension, impossible_rng);
        const core::PhaseVector goal =
            core::PhaseVector::random(config.dimension, impossible_rng);
        core::PredictiveSkillFabric direct =
            core::PredictiveSkillFabric::from_snapshot(fabric.snapshot());
        const core::Rlf2ExecutionResult autonomous =
            direct.execute_autonomous(start, goal);
        false_successes += autonomous.success ? 1U : 0U;
        abstentions += autonomous.abstained ? 1U : 0U;
    }
    result.impossible_false_success_rate =
        static_cast<double>(false_successes) /
        static_cast<double>(impossible_count);
    result.impossible_abstention_rate =
        static_cast<double>(abstentions) /
        static_cast<double>(impossible_count);

    result.route_cycle_compression =
        result.rlf2_subgoal_bridge.average_primitive_steps <= 0.0
            ? 0.0
            : 1.0 - result.rlf2_subgoal_bridge.average_cycles /
                result.rlf2_subgoal_bridge.average_primitive_steps;

    core::DeterministicRng validation_rng(config.seed ^ 0x534B494C4C56414CULL);
    std::size_t skill_successes = 0U;
    std::size_t skill_trials = 0U;
    double causal_total = 0.0;
    std::size_t causal_count = 0U;
    for (const core::CausalSkill& skill : fabric.skills()) {
        causal_total += skill.mean_causal_advantage;
        ++causal_count;
        if (skill.primitive_length <= 1U) continue;
        for (std::size_t sample = 0U; sample < 4U; ++sample) {
            const core::PhaseVector start =
                core::PhaseVector::random(config.dimension, validation_rng);
            core::PhaseVector expected = start;
            for (const std::uint64_t operator_id : skill.primitive_route) {
                expected = fabric.operator_by_id(operator_id).forward.apply(expected);
            }
            const core::PhaseVector predicted = skill.forward.apply(start);
            skill_successes += predicted.similarity(expected) >=
                config.goal_similarity_threshold ? 1U : 0U;
            ++skill_trials;
        }
    }
    result.skill_validation_accuracy = skill_trials == 0U
        ? 1.0
        : static_cast<double>(skill_successes) /
            static_cast<double>(skill_trials);
    result.mean_causal_advantage = causal_count == 0U
        ? 0.0
        : causal_total / static_cast<double>(causal_count);
    result.operator_count = fabric.operators().size();
    result.skill_count = fabric.skills().size();
    result.compound_skill_count = static_cast<std::size_t>(std::count_if(
        fabric.skills().begin(),
        fabric.skills().end(),
        [](const core::CausalSkill& value) {
            return value.primitive_length > 1U;
        }
    ));
    result.prototype_count = fabric.prototypes().size();
    result.estimated_bytes = estimate_bytes(fabric);
    result.training_stats = fabric.training_stats();
    result.training_seconds = std::chrono::duration<double>(
        training_end - training_begin
    ).count();

    result.leakage_audit.training_routes = training.route_hashes.size();
    result.leakage_audit.development_routes = development.route_hashes.size();
    result.leakage_audit.evaluation_routes = evaluation.route_hashes.size();
    for (const std::uint64_t hash : evaluation.route_hashes) {
        result.leakage_audit.route_overlap +=
            training.route_hashes.contains(hash) ||
            development.route_hashes.contains(hash) ? 1U : 0U;
    }
    for (const std::uint64_t hash : evaluation.start_goal_hashes) {
        result.leakage_audit.start_goal_overlap +=
            training.start_goal_hashes.contains(hash) ||
            development.start_goal_hashes.contains(hash) ? 1U : 0U;
    }
    result.leakage_audit.no_complete_route_overlap =
        result.leakage_audit.route_overlap == 0U;
    result.leakage_audit.no_exact_start_goal_overlap =
        result.leakage_audit.start_goal_overlap == 0U;
    const auto sorted_hashes = [](const auto& values) {
        std::vector<std::uint64_t> hashes(values.begin(), values.end());
        std::sort(hashes.begin(), hashes.end());
        return hashes;
    };
    result.leakage_audit.training_route_hashes =
        sorted_hashes(training.route_hashes);
    result.leakage_audit.development_route_hashes =
        sorted_hashes(development.route_hashes);
    result.leakage_audit.evaluation_route_hashes =
        sorted_hashes(evaluation.route_hashes);
    result.leakage_audit.training_start_goal_hashes =
        sorted_hashes(training.start_goal_hashes);
    result.leakage_audit.development_start_goal_hashes =
        sorted_hashes(development.start_goal_hashes);
    result.leakage_audit.evaluation_start_goal_hashes =
        sorted_hashes(evaluation.start_goal_hashes);
    result.leakage_audit.manifest_hash = fnv_offset_basis;
    hash_u64(result.leakage_audit.manifest_hash, training.manifest_hash);
    hash_u64(result.leakage_audit.manifest_hash, development.manifest_hash);
    hash_u64(result.leakage_audit.manifest_hash, evaluation.manifest_hash);

    core::PredictiveSkillFabric trace_fabric =
        core::PredictiveSkillFabric::from_snapshot(fabric.snapshot());
    result.representative_trace = trace_fabric.execute_subgoal_bridge(
        evaluation.episodes.front().start,
        evaluation.episodes.front().goal,
        config.evaluation_max_route_length,
        false
    );

    if (result.rlf2_autonomous.final_state_accuracy >= 0.50 &&
        result.rlf2_autonomous.final_state_accuracy >
            result.rlf1_autonomous.final_state_accuracy + 0.10 &&
        result.rlf2_subgoal_bridge.final_state_accuracy >= 0.95) {
        result.scientific_decision = "A";
    } else if (result.rlf2_subgoal_bridge.final_state_accuracy >= 0.90 &&
               result.skill_validation_accuracy >= 0.99 &&
               result.route_cycle_compression > 0.0) {
        result.scientific_decision = "B";
    } else {
        result.scientific_decision = "C";
    }
    result.limitations = {
        "The high-accuracy path depends on bounded bidirectional subgoal search.",
        "Primitive operator families remain predefined and exactly reversible.",
        "Autonomous prototype routing remains substantially weaker than planning.",
        "The benchmark is synthetic and uses fixed-dimensional phase vectors.",
        "Subgoal prototypes use response-profile nearest-neighbor estimation.",
        "No natural language, broad knowledge, coding, or multimodal data is tested.",
        "Exact bridge search grows exponentially with half the route depth.",
    };

    result.deterministic_run_hash = fnv_offset_basis;
    hash_u64(result.deterministic_run_hash, result.seed);
    hash_u64(result.deterministic_run_hash, result.leakage_audit.manifest_hash);
    hash_double(result.deterministic_run_hash,
                result.rlf2_autonomous.final_state_accuracy);
    hash_double(result.deterministic_run_hash,
                result.rlf2_subgoal_bridge.final_state_accuracy);
    hash_double(result.deterministic_run_hash,
                result.untrained_bidirectional_bridge.final_state_accuracy);
    hash_double(result.deterministic_run_hash, result.route_cycle_compression);
    hash_u64(result.deterministic_run_hash,
             static_cast<std::uint64_t>(result.skill_count));
    hash_u64(result.deterministic_run_hash,
             static_cast<std::uint64_t>(result.prototype_count));
    hash_u64(result.deterministic_run_hash,
             static_cast<std::uint64_t>(training_successes));
    return result;
}

void write_rlf2_predictive_reasoning_json(
    std::ostream& output,
    const Rlf2Result& result
) {
    output << std::setprecision(17);
    output << "{\n"
           << "  \"experiment\": \"rlf2_predictive_reasoning\",\n"
           << "  \"seed\": " << result.seed << ",\n"
           << "  \"dimension\": " << result.dimension << ",\n"
           << "  \"training_episodes\": " << result.training_episodes << ",\n"
           << "  \"evaluation_episodes\": " << result.evaluation_episodes << ",\n"
           << "  \"systems\": {\n"
           << "    \"rlf2_autonomous\": ";
    write_system_metrics(output, result.rlf2_autonomous, 4U);
    output << ",\n    \"rlf2_subgoal_bridge\": ";
    write_system_metrics(output, result.rlf2_subgoal_bridge, 4U);
    output << ",\n    \"untrained_bidirectional_bridge\": ";
    write_system_metrics(output, result.untrained_bidirectional_bridge, 4U);
    output << ",\n    \"rlf1_autonomous\": ";
    write_system_metrics(output, result.rlf1_autonomous, 4U);
    output << ",\n    \"greedy\": ";
    write_system_metrics(output, result.greedy, 4U);
    output << ",\n    \"oracle\": ";
    write_system_metrics(output, result.oracle, 4U);
    output << "\n  },\n  \"by_length\": [";
    if (!result.by_length.empty()) output << '\n';
    for (std::size_t index = 0U; index < result.by_length.size(); ++index) {
        const Rlf2LengthMetrics& value = result.by_length[index];
        output << "    {\"route_length\": " << value.route_length
               << ", \"episodes\": " << value.episodes
               << ", \"autonomous_accuracy\": " << value.autonomous_accuracy
               << ", \"bridge_accuracy\": " << value.bridge_accuracy
               << ", \"untrained_bridge_accuracy\": " << value.untrained_bridge_accuracy
               << ", \"rlf1_accuracy\": " << value.rlf1_accuracy
               << ", \"greedy_accuracy\": " << value.greedy_accuracy << '}';
        output << (index + 1U == result.by_length.size() ? "\n" : ",\n");
    }
    output << "  ],\n"
           << "  \"recovery_accuracy\": " << result.recovery_accuracy << ",\n"
           << "  \"impossible_false_success_rate\": " << result.impossible_false_success_rate << ",\n"
           << "  \"impossible_abstention_rate\": " << result.impossible_abstention_rate << ",\n"
           << "  \"route_cycle_compression\": " << result.route_cycle_compression << ",\n"
           << "  \"skill_validation_accuracy\": " << result.skill_validation_accuracy << ",\n"
           << "  \"mean_causal_advantage\": " << result.mean_causal_advantage << ",\n"
           << "  \"operator_count\": " << result.operator_count << ",\n"
           << "  \"skill_count\": " << result.skill_count << ",\n"
           << "  \"compound_skill_count\": " << result.compound_skill_count << ",\n"
           << "  \"prototype_count\": " << result.prototype_count << ",\n"
           << "  \"estimated_bytes\": " << result.estimated_bytes << ",\n"
           << "  \"training_seconds\": " << result.training_seconds << ",\n"
           << "  \"training_stats\": {\n"
           << "    \"observed_routes\": " << result.training_stats.observed_routes << ",\n"
           << "    \"observed_transitions\": " << result.training_stats.observed_transitions << ",\n"
           << "    \"intervention_tests\": " << result.training_stats.intervention_tests << ",\n"
           << "    \"intervention_alternative_successes\": " << result.training_stats.intervention_alternative_successes << ",\n"
           << "    \"prototypes_created\": " << result.training_stats.prototypes_created << ",\n"
           << "    \"prototypes_merged\": " << result.training_stats.prototypes_merged << ",\n"
           << "    \"skills_proposed\": " << result.training_stats.skills_proposed << ",\n"
           << "    \"skills_accepted\": " << result.training_stats.skills_accepted << ",\n"
           << "    \"routes_segmented\": " << result.training_stats.routes_segmented << "\n"
           << "  },\n"
           << "  \"leakage_audit\": {\n"
           << "    \"no_context_prefix\": " << (result.leakage_audit.no_context_prefix ? "true" : "false") << ",\n"
           << "    \"no_operator_labels_in_evaluation\": " << (result.leakage_audit.no_operator_labels_in_evaluation ? "true" : "false") << ",\n"
           << "    \"no_route_length_in_evaluation\": " << (result.leakage_audit.no_route_length_in_evaluation ? "true" : "false") << ",\n"
           << "    \"no_complete_route_overlap\": " << (result.leakage_audit.no_complete_route_overlap ? "true" : "false") << ",\n"
           << "    \"no_exact_start_goal_overlap\": " << (result.leakage_audit.no_exact_start_goal_overlap ? "true" : "false") << ",\n"
           << "    \"no_target_access_during_execution\": " << (result.leakage_audit.no_target_access_during_execution ? "true" : "false") << ",\n"
           << "    \"no_evaluation_route_training\": " << (result.leakage_audit.no_evaluation_route_training ? "true" : "false") << ",\n"
           << "    \"training_routes\": " << result.leakage_audit.training_routes << ",\n"
           << "    \"development_routes\": " << result.leakage_audit.development_routes << ",\n"
           << "    \"evaluation_routes\": " << result.leakage_audit.evaluation_routes << ",\n"
           << "    \"route_overlap\": " << result.leakage_audit.route_overlap << ",\n"
           << "    \"start_goal_overlap\": " << result.leakage_audit.start_goal_overlap << ",\n"
           << "    \"manifest_hash\": \"" << format_run_hash(result.leakage_audit.manifest_hash) << "\",\n";
    const auto write_hashes = [&output](const std::string_view name,
                                        const std::vector<std::uint64_t>& hashes,
                                        const bool trailing_comma) {
        output << "    \"" << name << "\": [";
        for (std::size_t index = 0U; index < hashes.size(); ++index) {
            output << '\"' << format_run_hash(hashes[index]) << '\"';
            if (index + 1U != hashes.size()) output << ", ";
        }
        output << ']' << (trailing_comma ? ",\n" : "\n");
    };
    write_hashes("training_route_hashes", result.leakage_audit.training_route_hashes, true);
    write_hashes("development_route_hashes", result.leakage_audit.development_route_hashes, true);
    write_hashes("evaluation_route_hashes", result.leakage_audit.evaluation_route_hashes, true);
    write_hashes("training_start_goal_hashes", result.leakage_audit.training_start_goal_hashes, true);
    write_hashes("development_start_goal_hashes", result.leakage_audit.development_start_goal_hashes, true);
    write_hashes("evaluation_start_goal_hashes", result.leakage_audit.evaluation_start_goal_hashes, false);
    output << "  },\n"
           << "  \"scientific_decision\": \"" << result.scientific_decision << "\",\n"
           << "  \"limitations\": [";
    for (std::size_t index = 0U; index < result.limitations.size(); ++index) {
        output << '"' << json_escape(result.limitations[index]) << '"';
        if (index + 1U != result.limitations.size()) output << ", ";
    }
    output << "],\n  \"deterministic_run_hash\": \""
           << format_run_hash(result.deterministic_run_hash) << "\",\n"
           << "  \"representative_trace\": ";
    write_execution(output, result.representative_trace, 2U);
    output << "\n}\n";
}

Rlf2TrainingWorkflowResult train_rlf2_checkpoint(
    const Rlf2Config& config,
    const std::filesystem::path& checkpoint_path
) {
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
        config.seed ^ 0x545241494E32ULL
    );
    std::size_t successes = 0U;
    core::PredictiveSkillFabric fabric = train_fabric(
        config,
        definitions,
        training.episodes,
        &successes
    );
    storage::save_rlf2_checkpoint(checkpoint_path, fabric);
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, config.seed);
    hash_u64(hash, static_cast<std::uint64_t>(successes));
    hash_u64(hash, static_cast<std::uint64_t>(fabric.skills().size()));
    hash_u64(hash, static_cast<std::uint64_t>(fabric.prototypes().size()));
    return {
        .checkpoint_path = checkpoint_path,
        .seed = config.seed,
        .successful_training_episodes = successes,
        .operators = fabric.operators().size(),
        .skills = fabric.skills().size(),
        .prototypes = fabric.prototypes().size(),
        .deterministic_run_hash = hash,
    };
}

Rlf2EvaluationWorkflowResult evaluate_rlf2_checkpoint(
    const std::filesystem::path& checkpoint_path,
    const std::uint64_t seed,
    const std::size_t evaluation_episodes
) {
    core::PredictiveSkillFabric fabric = storage::load_rlf2_checkpoint(
        checkpoint_path
    );
    Rlf2Config config;
    config.seed = seed;
    config.dimension = fabric.config().dimension;
    config.evaluation_episodes = evaluation_episodes;
    config.evaluation_max_route_length = std::min<std::size_t>(
        8U,
        fabric.config().maximum_route_depth
    );
    config.maximum_cycles = fabric.config().maximum_cycles;
    config.operator_count = fabric.operators().size();
    config.goal_similarity_threshold = fabric.config().goal_similarity_threshold;
    const std::vector<NamedOperator> definitions = make_operator_definitions(
        config.dimension,
        config.operator_count
    );
    const Manifest evaluation = generate_manifest(
        config,
        definitions,
        evaluation_episodes,
        config.evaluation_min_route_length,
        config.evaluation_max_route_length,
        seed ^ 0x4556414C574632ULL
    );
    core::PredictiveSkillFabric autonomous =
        core::PredictiveSkillFabric::from_snapshot(fabric.snapshot());
    core::PredictiveSkillFabric bridge =
        core::PredictiveSkillFabric::from_snapshot(fabric.snapshot());
    Rlf2EvaluationWorkflowResult result;
    result.checkpoint_path = checkpoint_path;
    result.evaluation_episodes = evaluation_episodes;
    result.autonomous = evaluate_system(
        "rlf2_autonomous",
        evaluation.episodes,
        [&autonomous](const Episode& episode) {
            return autonomous.execute_autonomous(episode.start, episode.goal);
        }
    );
    result.subgoal_bridge = evaluate_system(
        "rlf2_subgoal_bridge",
        evaluation.episodes,
        [&bridge, &config](const Episode& episode) {
            return bridge.execute_subgoal_bridge(
                episode.start,
                episode.goal,
                config.evaluation_max_route_length,
                false
            );
        }
    );
    result.deterministic_run_hash = fnv_offset_basis;
    hash_double(result.deterministic_run_hash,
                result.autonomous.final_state_accuracy);
    hash_double(result.deterministic_run_hash,
                result.subgoal_bridge.final_state_accuracy);
    return result;
}

Rlf2TraceWorkflowResult trace_rlf2_checkpoint(
    const std::filesystem::path& checkpoint_path,
    const std::uint64_t seed,
    const std::size_t sample_id
) {
    core::PredictiveSkillFabric fabric = storage::load_rlf2_checkpoint(
        checkpoint_path
    );
    Rlf2Config config;
    config.seed = seed;
    config.dimension = fabric.config().dimension;
    config.operator_count = fabric.operators().size();
    config.evaluation_episodes = sample_id + 1U;
    config.evaluation_max_route_length = std::min<std::size_t>(
        8U,
        fabric.config().maximum_route_depth
    );
    config.goal_similarity_threshold = fabric.config().goal_similarity_threshold;
    const std::vector<NamedOperator> definitions = make_operator_definitions(
        config.dimension,
        config.operator_count
    );
    const Manifest evaluation = generate_manifest(
        config,
        definitions,
        sample_id + 1U,
        config.evaluation_min_route_length,
        config.evaluation_max_route_length,
        seed ^ 0x545241434532ULL
    );
    const Episode& episode = evaluation.episodes.at(sample_id);
    core::PredictiveSkillFabric autonomous =
        core::PredictiveSkillFabric::from_snapshot(fabric.snapshot());
    core::PredictiveSkillFabric bridge =
        core::PredictiveSkillFabric::from_snapshot(fabric.snapshot());
    return {
        .checkpoint_path = checkpoint_path,
        .sample_id = sample_id,
        .autonomous = autonomous.execute_autonomous(
            episode.start,
            episode.goal
        ),
        .subgoal_bridge = bridge.execute_subgoal_bridge(
            episode.start,
            episode.goal,
            config.evaluation_max_route_length,
            false
        ),
    };
}

void write_rlf2_training_json(
    std::ostream& output,
    const Rlf2TrainingWorkflowResult& result
) {
    output << "{\n"
           << "  \"checkpoint\": \"" << json_escape(result.checkpoint_path.string()) << "\",\n"
           << "  \"seed\": " << result.seed << ",\n"
           << "  \"successful_training_episodes\": " << result.successful_training_episodes << ",\n"
           << "  \"operators\": " << result.operators << ",\n"
           << "  \"skills\": " << result.skills << ",\n"
           << "  \"prototypes\": " << result.prototypes << ",\n"
           << "  \"deterministic_run_hash\": \"" << format_run_hash(result.deterministic_run_hash) << "\"\n"
           << "}\n";
}

void write_rlf2_evaluation_json(
    std::ostream& output,
    const Rlf2EvaluationWorkflowResult& result
) {
    output << std::setprecision(17);
    output << "{\n  \"checkpoint\": \""
           << json_escape(result.checkpoint_path.string()) << "\",\n"
           << "  \"evaluation_episodes\": " << result.evaluation_episodes << ",\n"
           << "  \"autonomous\": ";
    write_system_metrics(output, result.autonomous, 2U);
    output << ",\n  \"subgoal_bridge\": ";
    write_system_metrics(output, result.subgoal_bridge, 2U);
    output << ",\n  \"deterministic_run_hash\": \""
           << format_run_hash(result.deterministic_run_hash) << "\"\n}\n";
}

void write_rlf2_trace_json(
    std::ostream& output,
    const Rlf2TraceWorkflowResult& result
) {
    output << std::setprecision(17);
    output << "{\n  \"checkpoint\": \""
           << json_escape(result.checkpoint_path.string()) << "\",\n"
           << "  \"sample_id\": " << result.sample_id << ",\n"
           << "  \"autonomous\": ";
    write_execution(output, result.autonomous, 2U);
    output << ",\n  \"subgoal_bridge\": ";
    write_execution(output, result.subgoal_bridge, 2U);
    output << "\n}\n";
}

}  // namespace rlf::experiments
