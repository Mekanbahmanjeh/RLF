#include "rlf/storage/rlf3_checkpoint.hpp"

#include "storage/binary_codec.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rlf::storage {
namespace {

constexpr std::array<std::uint8_t, 8U> magic{
    'R', 'L', 'F', '3', 'C', 'K', 'P', '5'
};
constexpr std::size_t header_size = 8U + 4U + 8U + 8U;

void write_phase_vector(
    detail::BufferWriter& writer,
    const core::PhaseVector& value
) {
    writer.write_u64(static_cast<std::uint64_t>(value.size()));
    for (const float angle : value.angles()) {
        writer.write_float(angle);
    }
}

[[nodiscard]] core::PhaseVector read_phase_vector(
    detail::BufferReader& reader,
    const std::size_t expected_dimension,
    const Rlf3CheckpointLoadOptions& options
) {
    const std::size_t dimension = detail::checked_size(
        reader.read_u64(),
        options.maximum_dimension,
        "RLF-3 phase-vector dimension"
    );
    if (dimension != expected_dimension) {
        throw std::runtime_error("RLF-3 checkpoint dimension mismatch");
    }
    constexpr float tau = 2.0F * std::numbers::pi_v<float>;
    std::vector<float> values;
    values.reserve(dimension);
    for (std::size_t index = 0U; index < dimension; ++index) {
        const float angle = reader.read_float();
        if (!std::isfinite(angle) || angle < 0.0F || angle >= tau) {
            throw std::runtime_error("invalid RLF-3 phase angle");
        }
        values.push_back(angle);
    }
    return core::PhaseVector(std::move(values));
}

void write_config(
    detail::BufferWriter& writer,
    const core::SparseWorldModelConfig& config
) {
    writer.write_u64(config.dimension);
    writer.write_u64(config.maximum_states);
    writer.write_u64(config.maximum_contexts);
    writer.write_u64(config.maximum_transitions);
    writer.write_u64(config.maximum_outcomes_per_transition);
    writer.write_u64(config.maximum_subgoals);
    writer.write_u64(config.hash_dimensions);
    writer.write_u64(config.phase_bins);
    writer.write_u64(config.maximum_bucket_candidates);
    writer.write_u64(config.nearest_subgoals);
    writer.write_u64(config.planner_node_budget);
    writer.write_u64(config.maximum_plan_depth);
    writer.write_u64(config.minimum_transition_support);
    writer.write_u64(config.minimum_subgoal_support);
    writer.write_double(config.state_merge_distance);
    writer.write_double(config.context_merge_distance);
    writer.write_double(config.minimum_outcome_probability);
    writer.write_double(config.risk_penalty);
    writer.write_double(config.uncertainty_penalty);
    writer.write_double(config.heuristic_scale);
    writer.write_u64(config.environment_layers);
    writer.write_u64(config.environment_lanes);
    writer.write_double(config.environment_stochastic_probability);
    writer.write_double(config.environment_observation_noise);
}

[[nodiscard]] core::SparseWorldModelConfig read_config(
    detail::BufferReader& reader,
    const Rlf3CheckpointLoadOptions& options
) {
    core::SparseWorldModelConfig config;
    config.dimension = detail::checked_size(
        reader.read_u64(), options.maximum_dimension, "RLF-3 dimension");
    config.maximum_states = detail::checked_size(
        reader.read_u64(), options.maximum_states, "RLF-3 maximum states");
    config.maximum_contexts = detail::checked_size(
        reader.read_u64(), options.maximum_contexts, "RLF-3 maximum contexts");
    config.maximum_transitions = detail::checked_size(
        reader.read_u64(), options.maximum_transitions,
        "RLF-3 maximum transitions");
    config.maximum_outcomes_per_transition = detail::checked_size(
        reader.read_u64(), options.maximum_outcomes,
        "RLF-3 maximum outcomes per transition");
    config.maximum_subgoals = detail::checked_size(
        reader.read_u64(), options.maximum_subgoals,
        "RLF-3 maximum subgoals");
    config.hash_dimensions = detail::checked_size(
        reader.read_u64(), config.dimension, "RLF-3 hash dimensions");
    config.phase_bins = detail::checked_size(
        reader.read_u64(), 256U, "RLF-3 phase bins");
    config.maximum_bucket_candidates = detail::checked_size(
        reader.read_u64(), options.maximum_states,
        "RLF-3 bucket candidates");
    config.nearest_subgoals = detail::checked_size(
        reader.read_u64(), options.maximum_subgoals,
        "RLF-3 nearest subgoals");
    config.planner_node_budget = detail::checked_size(
        reader.read_u64(), std::numeric_limits<std::size_t>::max(),
        "RLF-3 planner node budget");
    config.maximum_plan_depth = detail::checked_size(
        reader.read_u64(), std::numeric_limits<std::size_t>::max(),
        "RLF-3 maximum plan depth");
    config.minimum_transition_support = detail::checked_size(
        reader.read_u64(), std::numeric_limits<std::size_t>::max(),
        "RLF-3 minimum transition support");
    config.minimum_subgoal_support = detail::checked_size(
        reader.read_u64(), std::numeric_limits<std::size_t>::max(),
        "RLF-3 minimum subgoal support");
    config.state_merge_distance = reader.read_double();
    config.context_merge_distance = reader.read_double();
    config.minimum_outcome_probability = reader.read_double();
    config.risk_penalty = reader.read_double();
    config.uncertainty_penalty = reader.read_double();
    config.heuristic_scale = reader.read_double();
    config.environment_layers = detail::checked_size(
        reader.read_u64(), 1'000'000U, "RLF-3 environment layers");
    config.environment_lanes = detail::checked_size(
        reader.read_u64(), 1'000'000U, "RLF-3 environment lanes");
    config.environment_stochastic_probability = reader.read_double();
    config.environment_observation_noise = reader.read_double();
    return config;
}

void write_stats(
    detail::BufferWriter& writer,
    const core::SparseWorldModelStats& stats
) {
    writer.write_u64(stats.experiences_observed);
    writer.write_u64(stats.states_created);
    writer.write_u64(stats.states_merged);
    writer.write_u64(stats.contexts_created);
    writer.write_u64(stats.contexts_merged);
    writer.write_u64(stats.transitions_created);
    writer.write_u64(stats.outcomes_created);
    writer.write_u64(stats.subgoals_created);
    writer.write_u64(stats.subgoals_merged);
    writer.write_u64(stats.state_index_queries);
    writer.write_u64(stats.state_index_comparisons);
    writer.write_u64(stats.state_index_fallbacks);
}

[[nodiscard]] core::SparseWorldModelStats read_stats(
    detail::BufferReader& reader
) {
    return {
        .experiences_observed = reader.read_u64(),
        .states_created = reader.read_u64(),
        .states_merged = reader.read_u64(),
        .contexts_created = reader.read_u64(),
        .contexts_merged = reader.read_u64(),
        .transitions_created = reader.read_u64(),
        .outcomes_created = reader.read_u64(),
        .subgoals_created = reader.read_u64(),
        .subgoals_merged = reader.read_u64(),
        .state_index_queries = reader.read_u64(),
        .state_index_comparisons = reader.read_u64(),
        .state_index_fallbacks = reader.read_u64(),
    };
}

[[nodiscard]] std::vector<std::uint8_t> encode_payload(
    const core::SparseWorldModel& model
) {
    const core::SparseWorldModelSnapshot snapshot = model.snapshot();
    detail::BufferWriter writer;
    write_config(writer, snapshot.config);
    writer.write_u64(snapshot.seed);
    writer.write_u64(snapshot.training_step);
    writer.write_u64(snapshot.next_action_id);
    writer.write_u64(snapshot.next_state_id);
    writer.write_u64(snapshot.next_context_id);
    writer.write_u64(snapshot.next_transition_id);
    writer.write_u64(snapshot.next_subgoal_id);
    write_stats(writer, snapshot.stats);

    writer.write_u64(snapshot.actions.size());
    for (const core::WorldAction& action : snapshot.actions) {
        writer.write_u64(action.id);
        writer.write_string(action.name);
        writer.write_double(action.cost);
    }

    writer.write_u64(snapshot.states.size());
    for (const core::WorldStatePrototype& state : snapshot.states) {
        writer.write_u64(state.id);
        write_phase_vector(writer, state.key);
        writer.write_u64(state.support);
        writer.write_u64(state.creation_step);
        writer.write_u64(state.last_used_step);
    }

    writer.write_u64(snapshot.contexts.size());
    for (const core::WorldContextPrototype& context : snapshot.contexts) {
        writer.write_u64(context.id);
        write_phase_vector(writer, context.key);
        writer.write_u64(context.support);
        writer.write_u64(context.creation_step);
        writer.write_u64(context.last_used_step);
    }

    writer.write_u64(snapshot.transitions.size());
    for (const core::WorldTransition& transition : snapshot.transitions) {
        writer.write_u64(transition.id);
        writer.write_u64(transition.state_id);
        writer.write_u64(transition.context_id);
        writer.write_u64(transition.action_id);
        writer.write_u64(transition.support);
        writer.write_double(transition.mean_reward);
        writer.write_double(transition.mean_prediction_error);
        writer.write_u64(transition.outcomes.size());
        for (const core::WorldTransitionOutcome& outcome : transition.outcomes) {
            writer.write_u64(outcome.next_state_id);
            writer.write_u64(outcome.next_context_id);
            writer.write_u64(outcome.count);
            writer.write_double(outcome.mean_reward);
            writer.write_double(outcome.terminal_probability);
        }
    }

    writer.write_u64(snapshot.subgoals.size());
    for (const core::SparseSubgoal& subgoal : snapshot.subgoals) {
        writer.write_u64(subgoal.id);
        writer.write_u64(subgoal.state_id);
        writer.write_u64(subgoal.goal_state_id);
        writer.write_u64(subgoal.preferred_action_id);
        writer.write_double(subgoal.remaining_steps);
        writer.write_double(subgoal.success_probability);
        writer.write_u64(subgoal.support);
        writer.write_u64(subgoal.creation_step);
        writer.write_u64(subgoal.last_used_step);
    }
    return writer.take();
}

[[nodiscard]] core::SparseWorldModelSnapshot decode_payload(
    const std::span<const std::uint8_t> payload,
    const Rlf3CheckpointLoadOptions& options
) {
    detail::BufferReader reader(payload);
    core::SparseWorldModelSnapshot snapshot;
    snapshot.config = read_config(reader, options);
    snapshot.seed = reader.read_u64();
    snapshot.training_step = reader.read_u64();
    snapshot.next_action_id = reader.read_u64();
    snapshot.next_state_id = reader.read_u64();
    snapshot.next_context_id = reader.read_u64();
    snapshot.next_transition_id = reader.read_u64();
    snapshot.next_subgoal_id = reader.read_u64();
    snapshot.stats = read_stats(reader);

    const std::size_t action_count = detail::checked_size(
        reader.read_u64(), options.maximum_actions, "RLF-3 action count");
    snapshot.actions.reserve(action_count);
    for (std::size_t index = 0U; index < action_count; ++index) {
        snapshot.actions.push_back({
            reader.read_u64(),
            reader.read_string(options.maximum_string_bytes),
            reader.read_double(),
        });
    }

    const std::size_t state_count = detail::checked_size(
        reader.read_u64(), options.maximum_states, "RLF-3 state count");
    snapshot.states.reserve(state_count);
    for (std::size_t index = 0U; index < state_count; ++index) {
        snapshot.states.push_back({
            reader.read_u64(),
            read_phase_vector(reader, snapshot.config.dimension, options),
            reader.read_u64(),
            reader.read_u64(),
            reader.read_u64(),
        });
    }

    const std::size_t context_count = detail::checked_size(
        reader.read_u64(), options.maximum_contexts, "RLF-3 context count");
    snapshot.contexts.reserve(context_count);
    for (std::size_t index = 0U; index < context_count; ++index) {
        snapshot.contexts.push_back({
            reader.read_u64(),
            read_phase_vector(reader, snapshot.config.dimension, options),
            reader.read_u64(),
            reader.read_u64(),
            reader.read_u64(),
        });
    }

    const std::size_t transition_count = detail::checked_size(
        reader.read_u64(), options.maximum_transitions,
        "RLF-3 transition count");
    snapshot.transitions.reserve(transition_count);
    std::size_t total_outcomes = 0U;
    for (std::size_t index = 0U; index < transition_count; ++index) {
        core::WorldTransition transition;
        transition.id = reader.read_u64();
        transition.state_id = reader.read_u64();
        transition.context_id = reader.read_u64();
        transition.action_id = reader.read_u64();
        transition.support = reader.read_u64();
        transition.mean_reward = reader.read_double();
        transition.mean_prediction_error = reader.read_double();
        const std::size_t outcome_count = detail::checked_size(
            reader.read_u64(), options.maximum_outcomes,
            "RLF-3 outcome count");
        if (outcome_count > snapshot.config.maximum_outcomes_per_transition ||
            total_outcomes > options.maximum_outcomes - outcome_count) {
            throw std::runtime_error("RLF-3 outcomes exceed configured limit");
        }
        total_outcomes += outcome_count;
        transition.outcomes.reserve(outcome_count);
        for (std::size_t outcome_index = 0U;
             outcome_index < outcome_count; ++outcome_index) {
            transition.outcomes.push_back({
                reader.read_u64(),
                reader.read_u64(),
                reader.read_u64(),
                reader.read_double(),
                reader.read_double(),
            });
        }
        snapshot.transitions.push_back(std::move(transition));
    }

    const std::size_t subgoal_count = detail::checked_size(
        reader.read_u64(), options.maximum_subgoals, "RLF-3 subgoal count");
    snapshot.subgoals.reserve(subgoal_count);
    for (std::size_t index = 0U; index < subgoal_count; ++index) {
        snapshot.subgoals.push_back({
            reader.read_u64(),
            reader.read_u64(),
            reader.read_u64(),
            reader.read_u64(),
            reader.read_double(),
            reader.read_double(),
            reader.read_u64(),
            reader.read_u64(),
            reader.read_u64(),
        });
    }
    if (!reader.empty()) {
        throw std::runtime_error("trailing RLF-3 checkpoint payload data");
    }
    return snapshot;
}

struct DecodedFile final {
    std::vector<std::uint8_t> payload;
    std::uint64_t checksum{};
    std::size_t file_bytes{};
};

[[nodiscard]] DecodedFile decode_file(
    const std::filesystem::path& path,
    const Rlf3CheckpointLoadOptions& options
) {
    const std::vector<std::uint8_t> bytes = detail::read_file(
        path, options.maximum_file_bytes
    );
    if (bytes.size() < header_size) {
        throw std::runtime_error("truncated RLF-3 checkpoint header");
    }
    detail::BufferReader reader(bytes);
    for (const std::uint8_t expected : magic) {
        if (reader.read_u8() != expected) {
            throw std::runtime_error("invalid RLF-3 checkpoint magic");
        }
    }
    const std::uint32_t version = reader.read_u32();
    if (version != rlf3_checkpoint_format_version) {
        throw std::runtime_error("unsupported RLF-3 checkpoint version");
    }
    const std::size_t payload_size = detail::checked_size(
        reader.read_u64(),
        options.maximum_file_bytes - header_size,
        "RLF-3 payload size"
    );
    const std::uint64_t stored_checksum = reader.read_u64();
    if (payload_size != reader.remaining()) {
        throw std::runtime_error("RLF-3 checkpoint payload-size mismatch");
    }
    std::vector<std::uint8_t> payload = reader.read_bytes(payload_size);
    const std::uint64_t actual_checksum = detail::checksum(payload);
    if (actual_checksum != stored_checksum) {
        throw std::runtime_error("RLF-3 checkpoint checksum mismatch");
    }
    return {std::move(payload), stored_checksum, bytes.size()};
}

}  // namespace

void save_rlf3_checkpoint(
    const std::filesystem::path& path,
    const core::SparseWorldModel& model
) {
    const std::vector<std::uint8_t> payload = encode_payload(model);
    detail::BufferWriter writer;
    writer.write_bytes(magic);
    writer.write_u32(rlf3_checkpoint_format_version);
    writer.write_u64(payload.size());
    writer.write_u64(detail::checksum(payload));
    writer.write_bytes(payload);
    detail::write_file_transactionally(path, writer.bytes());
}

core::SparseWorldModel load_rlf3_checkpoint(
    const std::filesystem::path& path,
    const Rlf3CheckpointLoadOptions& options
) {
    const DecodedFile file = decode_file(path, options);
    return core::SparseWorldModel::from_snapshot(
        decode_payload(file.payload, options)
    );
}

Rlf3CheckpointSummary inspect_rlf3_checkpoint(
    const std::filesystem::path& path,
    const Rlf3CheckpointLoadOptions& options
) {
    const DecodedFile file = decode_file(path, options);
    const core::SparseWorldModelSnapshot snapshot = decode_payload(
        file.payload, options
    );
    std::size_t outcomes = 0U;
    std::size_t compounds = 0U;
    static_cast<void>(compounds);
    for (const core::WorldTransition& transition : snapshot.transitions) {
        outcomes += transition.outcomes.size();
    }
    return {
        rlf3_checkpoint_format_version,
        snapshot.seed,
        snapshot.training_step,
        snapshot.config.dimension,
        snapshot.actions.size(),
        snapshot.states.size(),
        snapshot.contexts.size(),
        snapshot.transitions.size(),
        outcomes,
        snapshot.subgoals.size(),
        file.checksum,
        file.file_bytes,
    };
}

}  // namespace rlf::storage
