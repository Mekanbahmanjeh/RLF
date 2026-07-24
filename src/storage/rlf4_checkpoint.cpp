#include "rlf/storage/rlf4_checkpoint.hpp"

#include "storage/binary_codec.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rlf::storage {
namespace {

constexpr std::array<std::uint8_t, 8U> magic{
    'R', 'L', 'F', '4', 'C', 'K', 'P', '6'
};
constexpr std::size_t header_size = 8U + 4U + 8U + 8U;

void write_phase_vector(
    detail::BufferWriter& writer,
    const core::PhaseVector& value
) {
    writer.write_u64(value.size());
    for (const float angle : value.angles()) {
        writer.write_float(angle);
    }
}

[[nodiscard]] core::PhaseVector read_phase_vector(
    detail::BufferReader& reader,
    const std::size_t expected_dimension,
    const Rlf4CheckpointLoadOptions& options
) {
    const std::size_t dimension = detail::checked_size(
        reader.read_u64(), options.maximum_dimension,
        "RLF-4 phase-vector dimension"
    );
    if (dimension != expected_dimension) {
        throw std::runtime_error("RLF-4 checkpoint dimension mismatch");
    }
    constexpr float tau = 2.0F * std::numbers::pi_v<float>;
    std::vector<float> angles;
    angles.reserve(dimension);
    for (std::size_t index = 0U; index < dimension; ++index) {
        const float angle = reader.read_float();
        if (!std::isfinite(angle) || angle < 0.0F || angle >= tau) {
            throw std::runtime_error("invalid RLF-4 phase angle");
        }
        angles.push_back(angle);
    }
    return core::PhaseVector(std::move(angles));
}

void write_config(
    detail::BufferWriter& writer,
    const core::TemporalFabricConfig& config
) {
    writer.write_u64(config.dimension);
    writer.write_u64(config.maximum_prototypes);
    writer.write_u64(config.maximum_contexts);
    writer.write_u64(config.maximum_context_order);
    writer.write_u64(config.minimum_context_support);
    writer.write_u64(config.maximum_options);
    writer.write_u64(config.minimum_option_length);
    writer.write_u64(config.maximum_option_length);
    writer.write_u64(config.minimum_option_support);
    writer.write_u64(config.option_prefix_minimum);
    writer.write_u64(config.maximum_prediction_outcomes);
    writer.write_double(config.prototype_merge_distance);
    writer.write_double(config.recent_decay);
    writer.write_double(config.recent_weight);
    writer.write_double(config.smoothing);
    writer.write_double(config.minimum_option_confidence);
    writer.write_double(config.minimum_option_gain);
    writer.write_double(config.surprise_slow_rate);
    writer.write_double(config.surprise_fast_rate);
    writer.write_double(config.change_threshold);
    writer.write_u64(config.change_cooldown);
}

[[nodiscard]] core::TemporalFabricConfig read_config(
    detail::BufferReader& reader,
    const Rlf4CheckpointLoadOptions& options
) {
    core::TemporalFabricConfig config;
    config.dimension = detail::checked_size(
        reader.read_u64(), options.maximum_dimension, "RLF-4 dimension");
    config.maximum_prototypes = detail::checked_size(
        reader.read_u64(), options.maximum_prototypes,
        "RLF-4 maximum prototypes");
    config.maximum_contexts = detail::checked_size(
        reader.read_u64(), options.maximum_contexts,
        "RLF-4 maximum contexts");
    config.maximum_context_order = detail::checked_size(
        reader.read_u64(), options.maximum_sequence_length,
        "RLF-4 maximum context order");
    config.minimum_context_support = detail::checked_size(
        reader.read_u64(), std::numeric_limits<std::size_t>::max(),
        "RLF-4 minimum context support");
    config.maximum_options = detail::checked_size(
        reader.read_u64(), options.maximum_options,
        "RLF-4 maximum options");
    config.minimum_option_length = detail::checked_size(
        reader.read_u64(), options.maximum_sequence_length,
        "RLF-4 minimum option length");
    config.maximum_option_length = detail::checked_size(
        reader.read_u64(), options.maximum_sequence_length,
        "RLF-4 maximum option length");
    config.minimum_option_support = detail::checked_size(
        reader.read_u64(), std::numeric_limits<std::size_t>::max(),
        "RLF-4 minimum option support");
    config.option_prefix_minimum = detail::checked_size(
        reader.read_u64(), options.maximum_sequence_length,
        "RLF-4 option prefix minimum");
    config.maximum_prediction_outcomes = detail::checked_size(
        reader.read_u64(), options.maximum_prototypes,
        "RLF-4 maximum prediction outcomes");
    config.prototype_merge_distance = reader.read_double();
    config.recent_decay = reader.read_double();
    config.recent_weight = reader.read_double();
    config.smoothing = reader.read_double();
    config.minimum_option_confidence = reader.read_double();
    config.minimum_option_gain = reader.read_double();
    config.surprise_slow_rate = reader.read_double();
    config.surprise_fast_rate = reader.read_double();
    config.change_threshold = reader.read_double();
    config.change_cooldown = detail::checked_size(
        reader.read_u64(), std::numeric_limits<std::size_t>::max(),
        "RLF-4 change cooldown");
    return config;
}

void write_stats(
    detail::BufferWriter& writer,
    const core::TemporalFabricStats& stats
) {
    writer.write_u64(stats.observations_seen);
    writer.write_u64(stats.sequences_seen);
    writer.write_u64(stats.prototypes_created);
    writer.write_u64(stats.prototypes_merged);
    writer.write_u64(stats.contexts_created);
    writer.write_u64(stats.contexts_updated);
    writer.write_u64(stats.option_candidates);
    writer.write_u64(stats.options_created);
    writer.write_u64(stats.prediction_queries);
    writer.write_u64(stats.context_comparisons);
    writer.write_u64(stats.option_comparisons);
    writer.write_u64(stats.change_points_detected);
}

[[nodiscard]] core::TemporalFabricStats read_stats(
    detail::BufferReader& reader
) {
    return {
        reader.read_u64(), reader.read_u64(), reader.read_u64(),
        reader.read_u64(), reader.read_u64(), reader.read_u64(),
        reader.read_u64(), reader.read_u64(), reader.read_u64(),
        reader.read_u64(), reader.read_u64(), reader.read_u64(),
    };
}

[[nodiscard]] std::vector<std::uint8_t> encode_payload(
    const core::TemporalPredictiveFabric& fabric
) {
    const auto snapshot = fabric.snapshot();
    detail::BufferWriter writer;
    write_config(writer, snapshot.config);
    writer.write_u64(snapshot.seed);
    writer.write_u64(snapshot.training_step);
    writer.write_u64(snapshot.next_prototype_id);
    writer.write_u64(snapshot.next_context_id);
    writer.write_u64(snapshot.next_option_id);
    writer.write_double(snapshot.slow_surprise);
    writer.write_double(snapshot.fast_surprise);
    writer.write_bool(snapshot.surprise_initialized);
    writer.write_u64(snapshot.last_change_step);
    write_stats(writer, snapshot.stats);

    writer.write_u64(snapshot.prototypes.size());
    for (const auto& prototype : snapshot.prototypes) {
        writer.write_u64(prototype.id);
        write_phase_vector(writer, prototype.key);
        writer.write_u64(prototype.support);
        writer.write_u64(prototype.creation_step);
        writer.write_u64(prototype.last_used_step);
    }

    writer.write_u64(snapshot.contexts.size());
    for (const auto& context : snapshot.contexts) {
        writer.write_u64(context.id);
        writer.write_u64(context.history.size());
        for (const auto id : context.history) {
            writer.write_u64(id);
        }
        writer.write_u64(context.support);
        writer.write_u64(context.creation_step);
        writer.write_u64(context.last_used_step);
        writer.write_u64(context.outcomes.size());
        for (const auto& outcome : context.outcomes) {
            writer.write_u64(outcome.prototype_id);
            writer.write_u64(outcome.total_count);
            writer.write_double(outcome.recent_count);
            writer.write_u64(outcome.last_update_step);
        }
    }

    writer.write_u64(snapshot.options.size());
    for (const auto& option : snapshot.options) {
        writer.write_u64(option.id);
        writer.write_u64(option.sequence.size());
        for (const auto id : option.sequence) {
            writer.write_u64(id);
        }
        writer.write_u64(option.support);
        writer.write_double(option.confidence);
        writer.write_double(option.predictive_gain);
        writer.write_double(option.compression_gain);
        writer.write_u64(option.creation_step);
        writer.write_u64(option.last_used_step);
    }

    writer.write_u64(snapshot.recent_history.size());
    for (const auto id : snapshot.recent_history) {
        writer.write_u64(id);
    }
    return writer.take();
}

[[nodiscard]] core::TemporalFabricSnapshot decode_payload(
    const std::span<const std::uint8_t> payload,
    const Rlf4CheckpointLoadOptions& options
) {
    detail::BufferReader reader(payload);
    core::TemporalFabricSnapshot snapshot;
    snapshot.config = read_config(reader, options);
    snapshot.seed = reader.read_u64();
    snapshot.training_step = reader.read_u64();
    snapshot.next_prototype_id = reader.read_u64();
    snapshot.next_context_id = reader.read_u64();
    snapshot.next_option_id = reader.read_u64();
    snapshot.slow_surprise = reader.read_double();
    snapshot.fast_surprise = reader.read_double();
    snapshot.surprise_initialized = reader.read_bool();
    snapshot.last_change_step = reader.read_u64();
    snapshot.stats = read_stats(reader);

    const std::size_t prototype_count = detail::checked_size(
        reader.read_u64(), options.maximum_prototypes,
        "RLF-4 prototype count");
    snapshot.prototypes.reserve(prototype_count);
    for (std::size_t index = 0U; index < prototype_count; ++index) {
        snapshot.prototypes.push_back({
            reader.read_u64(),
            read_phase_vector(reader, snapshot.config.dimension, options),
            reader.read_u64(), reader.read_u64(), reader.read_u64(),
        });
    }

    const std::size_t context_count = detail::checked_size(
        reader.read_u64(), options.maximum_contexts,
        "RLF-4 context count");
    snapshot.contexts.reserve(context_count);
    std::size_t total_outcomes = 0U;
    for (std::size_t index = 0U; index < context_count; ++index) {
        core::TemporalContext context;
        context.id = reader.read_u64();
        const std::size_t history_size = detail::checked_size(
            reader.read_u64(), options.maximum_sequence_length,
            "RLF-4 context history");
        context.history.reserve(history_size);
        for (std::size_t item = 0U; item < history_size; ++item) {
            context.history.push_back(reader.read_u64());
        }
        context.support = reader.read_u64();
        context.creation_step = reader.read_u64();
        context.last_used_step = reader.read_u64();
        const std::size_t outcome_count = detail::checked_size(
            reader.read_u64(), options.maximum_outcomes,
            "RLF-4 context outcomes");
        if (total_outcomes > options.maximum_outcomes - outcome_count) {
            throw std::runtime_error("RLF-4 outcomes exceed configured limit");
        }
        total_outcomes += outcome_count;
        context.outcomes.reserve(outcome_count);
        for (std::size_t item = 0U; item < outcome_count; ++item) {
            context.outcomes.push_back({
                reader.read_u64(), reader.read_u64(), reader.read_double(),
                reader.read_u64(),
            });
        }
        snapshot.contexts.push_back(std::move(context));
    }

    const std::size_t option_count = detail::checked_size(
        reader.read_u64(), options.maximum_options, "RLF-4 option count");
    snapshot.options.reserve(option_count);
    for (std::size_t index = 0U; index < option_count; ++index) {
        core::TemporalOption option;
        option.id = reader.read_u64();
        const std::size_t sequence_size = detail::checked_size(
            reader.read_u64(), options.maximum_sequence_length,
            "RLF-4 option sequence");
        option.sequence.reserve(sequence_size);
        for (std::size_t item = 0U; item < sequence_size; ++item) {
            option.sequence.push_back(reader.read_u64());
        }
        option.support = reader.read_u64();
        option.confidence = reader.read_double();
        option.predictive_gain = reader.read_double();
        option.compression_gain = reader.read_double();
        option.creation_step = reader.read_u64();
        option.last_used_step = reader.read_u64();
        snapshot.options.push_back(std::move(option));
    }

    const std::size_t history_size = detail::checked_size(
        reader.read_u64(), options.maximum_sequence_length,
        "RLF-4 recent history");
    snapshot.recent_history.reserve(history_size);
    for (std::size_t index = 0U; index < history_size; ++index) {
        snapshot.recent_history.push_back(reader.read_u64());
    }
    if (!reader.empty()) {
        throw std::runtime_error("RLF-4 checkpoint contains trailing payload");
    }
    return snapshot;
}

struct ParsedCheckpoint final {
    std::vector<std::uint8_t> payload;
    std::uint64_t checksum{};
    std::size_t file_bytes{};
};

[[nodiscard]] ParsedCheckpoint parse_file(
    const std::filesystem::path& path,
    const Rlf4CheckpointLoadOptions& options
) {
    const auto bytes = detail::read_file(path, options.maximum_file_bytes);
    if (bytes.size() < header_size) {
        throw std::runtime_error("truncated RLF-4 checkpoint header");
    }
    detail::BufferReader reader(bytes);
    for (const std::uint8_t expected : magic) {
        if (reader.read_u8() != expected) {
            throw std::runtime_error("invalid RLF-4 checkpoint magic");
        }
    }
    const std::uint32_t version = reader.read_u32();
    if (version != rlf4_checkpoint_format_version) {
        throw std::runtime_error("unsupported RLF-4 checkpoint version");
    }
    const std::size_t payload_size = detail::checked_size(
        reader.read_u64(), options.maximum_file_bytes,
        "RLF-4 checkpoint payload");
    const std::uint64_t expected_checksum = reader.read_u64();
    if (payload_size != reader.remaining()) {
        throw std::runtime_error("RLF-4 checkpoint payload size mismatch");
    }
    std::vector<std::uint8_t> payload = reader.read_bytes(payload_size);
    const std::uint64_t actual_checksum = detail::checksum(payload);
    if (actual_checksum != expected_checksum) {
        throw std::runtime_error("RLF-4 checkpoint checksum mismatch");
    }
    return {std::move(payload), actual_checksum, bytes.size()};
}

}  // namespace

void save_rlf4_checkpoint(
    const std::filesystem::path& path,
    const core::TemporalPredictiveFabric& fabric
) {
    const std::vector<std::uint8_t> payload = encode_payload(fabric);
    detail::BufferWriter writer;
    writer.write_bytes(magic);
    writer.write_u32(rlf4_checkpoint_format_version);
    writer.write_u64(payload.size());
    writer.write_u64(detail::checksum(payload));
    writer.write_bytes(payload);
    detail::write_file_transactionally(path, writer.bytes());
}

core::TemporalPredictiveFabric load_rlf4_checkpoint(
    const std::filesystem::path& path,
    const Rlf4CheckpointLoadOptions& options
) {
    const ParsedCheckpoint parsed = parse_file(path, options);
    return core::TemporalPredictiveFabric::from_snapshot(
        decode_payload(parsed.payload, options)
    );
}

Rlf4CheckpointSummary inspect_rlf4_checkpoint(
    const std::filesystem::path& path,
    const Rlf4CheckpointLoadOptions& options
) {
    const ParsedCheckpoint parsed = parse_file(path, options);
    const auto snapshot = decode_payload(parsed.payload, options);
    std::size_t outcomes = 0U;
    for (const auto& context : snapshot.contexts) {
        outcomes += context.outcomes.size();
    }
    return {
        rlf4_checkpoint_format_version,
        snapshot.seed,
        snapshot.training_step,
        snapshot.config.dimension,
        snapshot.prototypes.size(),
        snapshot.contexts.size(),
        outcomes,
        snapshot.options.size(),
        parsed.checksum,
        parsed.file_bytes,
    };
}

}  // namespace rlf::storage
