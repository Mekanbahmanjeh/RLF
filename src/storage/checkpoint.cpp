#include "rlf/storage/checkpoint.hpp"

#include "storage/binary_codec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace rlf::storage {
namespace {

constexpr std::array<std::uint8_t, 8> checkpoint_magic{
    'R', 'L', 'F', 'C', 'H', 'K', '0', '1'
};
constexpr std::size_t header_size = 8U + 4U + 8U + 8U;

struct ParsedCheckpoint final {
    CheckpointData data;
    std::uint64_t payload_checksum;
    std::size_t file_bytes;
};

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
    const CheckpointLoadOptions& options
) {
    const std::size_t dimension = detail::checked_size(
        reader.read_u64(),
        options.maximum_dimension,
        "checkpoint phase-vector dimension"
    );
    if (dimension != expected_dimension) {
        throw std::runtime_error(
            "checkpoint phase-vector dimension is incompatible"
        );
    }
    std::vector<float> angles;
    angles.reserve(dimension);
    constexpr float tau = 2.0F * std::numbers::pi_v<float>;
    for (std::size_t index = 0U; index < dimension; ++index) {
        const float angle = reader.read_float();
        if (!std::isfinite(angle) || angle < 0.0F || angle >= tau) {
            throw std::runtime_error("invalid checkpoint phase angle");
        }
        angles.push_back(angle);
    }
    return core::PhaseVector(std::move(angles));
}

void write_settling_config(
    detail::BufferWriter& writer,
    const core::SettlingConfig& config
) {
    writer.write_u64(static_cast<std::uint64_t>(config.candidate_count));
    writer.write_u64(static_cast<std::uint64_t>(config.active_count));
    writer.write_u64(static_cast<std::uint64_t>(config.maximum_cycles));
    writer.write_u64(static_cast<std::uint64_t>(config.minimum_cycles));
    writer.write_double(config.minimum_resonance);
    writer.write_double(config.convergence_tolerance_radians);
    writer.write_bool(config.confidence_threshold.has_value());
    if (config.confidence_threshold.has_value()) {
        writer.write_double(*config.confidence_threshold);
    }
    writer.write_double(config.input_weight);
    writer.write_double(config.previous_state_weight);
    writer.write_double(config.proposal_weight_scale);
    writer.write_double(config.utility_weight);
}

[[nodiscard]] core::SettlingConfig read_settling_config(
    detail::BufferReader& reader
) {
    core::SettlingConfig config;
    config.candidate_count = detail::checked_size(
        reader.read_u64(),
        std::numeric_limits<std::size_t>::max(),
        "candidate count"
    );
    config.active_count = detail::checked_size(
        reader.read_u64(),
        std::numeric_limits<std::size_t>::max(),
        "active count"
    );
    config.maximum_cycles = detail::checked_size(
        reader.read_u64(),
        std::numeric_limits<std::size_t>::max(),
        "maximum settling cycles"
    );
    config.minimum_cycles = detail::checked_size(
        reader.read_u64(),
        std::numeric_limits<std::size_t>::max(),
        "minimum settling cycles"
    );
    config.minimum_resonance = reader.read_double();
    config.convergence_tolerance_radians = reader.read_double();
    if (reader.read_bool()) {
        config.confidence_threshold = reader.read_double();
    } else {
        config.confidence_threshold.reset();
    }
    config.input_weight = reader.read_double();
    config.previous_state_weight = reader.read_double();
    config.proposal_weight_scale = reader.read_double();
    config.utility_weight = reader.read_double();
    return config;
}

void write_structural_config(
    detail::BufferWriter& writer,
    const learning::StructuralLearningConfig& config
) {
    writer.write_bool(config.enabled);
    writer.write_bool(config.enable_creation);
    writer.write_bool(config.enable_splitting);
    writer.write_bool(config.enable_merging);
    writer.write_bool(config.enable_pruning);
    writer.write_double(config.creation_minimum_resonance);
    writer.write_double(config.creation_prediction_error_threshold);
    writer.write_float(config.creation_confidence);
    writer.write_float(config.creation_utility);
    writer.write_float(config.creation_selectivity);
    writer.write_u64(
        static_cast<std::uint64_t>(
            config.correction_history_capacity
        )
    );
    writer.write_u64(
        static_cast<std::uint64_t>(config.split_minimum_samples)
    );
    writer.write_u64(
        static_cast<std::uint64_t>(
            config.split_minimum_cluster_size
        )
    );
    writer.write_u64(
        static_cast<std::uint64_t>(config.split_kmeans_iterations)
    );
    writer.write_double(
        config.split_minimum_transformation_separation_radians
    );
    writer.write_double(
        config.split_minimum_context_separation_radians
    );
    writer.write_double(
        config.split_minimum_validation_gain_radians
    );
    writer.write_double(config.split_context_distance_weight);
    writer.write_float(config.split_child_confidence_scale);
    writer.write_double(config.merge_maximum_key_error_radians);
    writer.write_double(
        config.merge_maximum_transformation_error_radians
    );
    writer.write_double(
        config.merge_maximum_history_dispersion_radians
    );
    writer.write_u64(config.pruning_minimum_age_steps);
    writer.write_u64(config.pruning_maximum_inactive_steps);
    writer.write_u64(config.pruning_minimum_activation_count);
    writer.write_double(config.pruning_maximum_utility);
    writer.write_double(config.pruning_harmful_update_ratio);
    writer.write_u64(config.pruning_disabled_grace_steps);
    writer.write_u64(
        static_cast<std::uint64_t>(config.minimum_retained_modes)
    );
    writer.write_u64(
        static_cast<std::uint64_t>(config.memory_budget_bytes)
    );
}

[[nodiscard]] learning::StructuralLearningConfig read_structural_config(
    detail::BufferReader& reader
) {
    learning::StructuralLearningConfig config;
    config.enabled = reader.read_bool();
    config.enable_creation = reader.read_bool();
    config.enable_splitting = reader.read_bool();
    config.enable_merging = reader.read_bool();
    config.enable_pruning = reader.read_bool();
    config.creation_minimum_resonance = reader.read_double();
    config.creation_prediction_error_threshold = reader.read_double();
    config.creation_confidence = reader.read_float();
    config.creation_utility = reader.read_float();
    config.creation_selectivity = reader.read_float();
    config.correction_history_capacity = detail::checked_size(
        reader.read_u64(),
        std::numeric_limits<std::size_t>::max(),
        "correction history capacity"
    );
    config.split_minimum_samples = detail::checked_size(
        reader.read_u64(),
        std::numeric_limits<std::size_t>::max(),
        "split minimum samples"
    );
    config.split_minimum_cluster_size = detail::checked_size(
        reader.read_u64(),
        std::numeric_limits<std::size_t>::max(),
        "split minimum cluster size"
    );
    config.split_kmeans_iterations = detail::checked_size(
        reader.read_u64(),
        std::numeric_limits<std::size_t>::max(),
        "split k-means iterations"
    );
    config.split_minimum_transformation_separation_radians =
        reader.read_double();
    config.split_minimum_context_separation_radians =
        reader.read_double();
    config.split_minimum_validation_gain_radians =
        reader.read_double();
    config.split_context_distance_weight = reader.read_double();
    config.split_child_confidence_scale = reader.read_float();
    config.merge_maximum_key_error_radians = reader.read_double();
    config.merge_maximum_transformation_error_radians =
        reader.read_double();
    config.merge_maximum_history_dispersion_radians =
        reader.read_double();
    config.pruning_minimum_age_steps = reader.read_u64();
    config.pruning_maximum_inactive_steps = reader.read_u64();
    config.pruning_minimum_activation_count = reader.read_u64();
    config.pruning_maximum_utility = reader.read_double();
    config.pruning_harmful_update_ratio = reader.read_double();
    config.pruning_disabled_grace_steps = reader.read_u64();
    config.minimum_retained_modes = detail::checked_size(
        reader.read_u64(),
        std::numeric_limits<std::size_t>::max(),
        "minimum retained modes"
    );
    config.memory_budget_bytes = detail::checked_size(
        reader.read_u64(),
        std::numeric_limits<std::size_t>::max(),
        "structural memory budget"
    );
    return config;
}

void write_fabric_config(
    detail::BufferWriter& writer,
    const core::FabricConfig& config
) {
    writer.write_u64(static_cast<std::uint64_t>(config.dimension));
    writer.write_u64(static_cast<std::uint64_t>(config.maximum_modes));
    write_settling_config(writer, config.settling);
    write_structural_config(writer, config.structural_learning);
}

[[nodiscard]] core::FabricConfig read_fabric_config(
    detail::BufferReader& reader,
    const CheckpointLoadOptions& options
) {
    core::FabricConfig config;
    config.dimension = detail::checked_size(
        reader.read_u64(),
        options.maximum_dimension,
        "checkpoint fabric dimension"
    );
    if (config.dimension == 0U) {
        throw std::runtime_error(
            "checkpoint fabric dimension must be positive"
        );
    }
    if (options.expected_dimension.has_value() &&
        config.dimension != *options.expected_dimension) {
        throw std::runtime_error(
            "checkpoint dimension does not match expected dimension"
        );
    }
    config.maximum_modes = detail::checked_size(
        reader.read_u64(),
        options.maximum_modes,
        "checkpoint maximum modes"
    );
    if (config.maximum_modes == 0U) {
        throw std::runtime_error(
            "checkpoint maximum modes must be positive"
        );
    }
    config.settling = read_settling_config(reader);
    config.structural_learning = read_structural_config(reader);
    return config;
}

void write_correction(
    detail::BufferWriter& writer,
    const core::CorrectionSummary& correction
) {
    write_phase_vector(writer, correction.context);
    write_phase_vector(writer, correction.desired_transformation);
    writer.write_double(correction.proposal_quality);
    writer.write_bool(correction.improved_prediction);
    writer.write_u64(correction.step);
}

[[nodiscard]] core::CorrectionSummary read_correction(
    detail::BufferReader& reader,
    const std::size_t dimension,
    const CheckpointLoadOptions& options
) {
    core::PhaseVector context = read_phase_vector(
        reader,
        dimension,
        options
    );
    core::PhaseVector desired_transformation = read_phase_vector(
        reader,
        dimension,
        options
    );
    const double quality = reader.read_double();
    if (!std::isfinite(quality) || quality < 0.0 || quality > 1.0) {
        throw std::runtime_error(
            "invalid correction proposal quality"
        );
    }
    const bool improved_prediction = reader.read_bool();
    const std::uint64_t step = reader.read_u64();
    return {
        .context = std::move(context),
        .desired_transformation = std::move(desired_transformation),
        .proposal_quality = quality,
        .improved_prediction = improved_prediction,
        .step = step,
    };
}

void write_mode(
    detail::BufferWriter& writer,
    const core::ResonantMode& mode
) {
    writer.write_u64(mode.id);
    write_phase_vector(writer, mode.context_key);
    write_phase_vector(writer, mode.transformation);
    writer.write_float(mode.selectivity);
    writer.write_float(mode.confidence);
    writer.write_float(mode.utility);
    writer.write_u64(mode.activation_count);
    writer.write_u64(mode.successful_update_count);
    writer.write_u64(mode.unsuccessful_update_count);
    writer.write_u64(mode.creation_step);
    writer.write_u64(mode.last_used_step);
    writer.write_bool(mode.enabled);
    writer.write_u64(
        static_cast<std::uint64_t>(mode.recent_corrections.size())
    );
    for (const core::CorrectionSummary& correction :
         mode.recent_corrections) {
        write_correction(writer, correction);
    }
}

[[nodiscard]] core::ResonantMode read_mode(
    detail::BufferReader& reader,
    const std::size_t dimension,
    const CheckpointLoadOptions& options
) {
    const std::uint64_t id = reader.read_u64();
    core::PhaseVector context_key = read_phase_vector(
        reader,
        dimension,
        options
    );
    core::PhaseVector transformation = read_phase_vector(
        reader,
        dimension,
        options
    );
    const float selectivity = reader.read_float();
    const float confidence = reader.read_float();
    const float utility = reader.read_float();
    const std::uint64_t activation_count = reader.read_u64();
    const std::uint64_t successful_update_count = reader.read_u64();
    const std::uint64_t unsuccessful_update_count = reader.read_u64();
    const std::uint64_t creation_step = reader.read_u64();
    const std::uint64_t last_used_step = reader.read_u64();
    const bool enabled = reader.read_bool();
    const std::size_t correction_count = detail::checked_size(
        reader.read_u64(),
        options.maximum_corrections_per_mode,
        "mode correction count"
    );

    core::ResonantMode mode(
        id,
        std::move(context_key),
        std::move(transformation),
        selectivity,
        confidence,
        utility,
        creation_step
    );
    mode.activation_count = activation_count;
    mode.successful_update_count = successful_update_count;
    mode.unsuccessful_update_count = unsuccessful_update_count;
    mode.last_used_step = last_used_step;
    mode.enabled = enabled;
    mode.recent_corrections.reserve(correction_count);
    for (std::size_t correction_index = 0U;
         correction_index < correction_count;
         ++correction_index) {
        mode.recent_corrections.push_back(
            read_correction(reader, dimension, options)
        );
    }
    return mode;
}

void write_associative_value(
    detail::BufferWriter& writer,
    const memory::AssociativeValue& value
) {
    if (std::holds_alternative<core::PhaseVector>(value)) {
        writer.write_u8(0U);
        write_phase_vector(
            writer,
            std::get<core::PhaseVector>(value)
        );
    } else {
        writer.write_u8(1U);
        const memory::BytePayload& bytes =
            std::get<memory::BytePayload>(value);
        writer.write_u64(static_cast<std::uint64_t>(bytes.size()));
        writer.write_bytes(bytes);
    }
}

[[nodiscard]] memory::AssociativeValue read_associative_value(
    detail::BufferReader& reader,
    const std::size_t dimension,
    const CheckpointLoadOptions& options
) {
    const std::uint8_t type = reader.read_u8();
    if (type == 0U) {
        return read_phase_vector(reader, dimension, options);
    }
    if (type == 1U) {
        const std::size_t payload_size = detail::checked_size(
            reader.read_u64(),
            options.maximum_payload_bytes,
            "checkpoint associative byte payload"
        );
        return reader.read_bytes(payload_size);
    }
    throw std::runtime_error(
        "unsupported checkpoint associative value type"
    );
}

void write_memory(
    detail::BufferWriter& writer,
    const memory::AssociativeMemory& memory
) {
    writer.write_u64(static_cast<std::uint64_t>(memory.dimension()));
    writer.write_u64(static_cast<std::uint64_t>(memory.capacity()));
    writer.write_u64(memory.next_record_id());
    writer.write_u64(memory.sequence());
    writer.write_u64(static_cast<std::uint64_t>(memory.size()));
    for (const memory::AssociativeRecord& record : memory.records()) {
        writer.write_u64(record.id);
        write_phase_vector(writer, record.key);
        write_associative_value(writer, record.value);
        writer.write_float(record.confidence);
        writer.write_u64(record.timestamp);
        writer.write_u64(record.access_count);
        writer.write_u64(record.last_access_sequence);
    }
}

[[nodiscard]] memory::AssociativeMemory read_memory(
    detail::BufferReader& reader,
    const std::size_t fabric_dimension,
    const CheckpointLoadOptions& options
) {
    const std::size_t dimension = detail::checked_size(
        reader.read_u64(),
        options.maximum_dimension,
        "checkpoint associative-memory dimension"
    );
    if (dimension != fabric_dimension) {
        throw std::runtime_error(
            "checkpoint associative-memory dimension is incompatible"
        );
    }
    const std::size_t capacity = detail::checked_size(
        reader.read_u64(),
        options.maximum_memory_capacity,
        "checkpoint associative-memory capacity"
    );
    if (capacity == 0U) {
        throw std::runtime_error(
            "checkpoint associative-memory capacity must be positive"
        );
    }
    const std::uint64_t next_record_id = reader.read_u64();
    const std::uint64_t sequence = reader.read_u64();
    const std::size_t record_count = detail::checked_size(
        reader.read_u64(),
        std::min(options.maximum_memory_records, capacity),
        "checkpoint associative-memory records"
    );

    std::vector<memory::AssociativeRecord> records;
    records.reserve(record_count);
    for (std::size_t record_index = 0U;
         record_index < record_count;
         ++record_index) {
        const std::uint64_t id = reader.read_u64();
        core::PhaseVector key = read_phase_vector(
            reader,
            dimension,
            options
        );
        memory::AssociativeValue value = read_associative_value(
            reader,
            dimension,
            options
        );
        const float confidence = reader.read_float();
        const std::uint64_t timestamp = reader.read_u64();
        const std::uint64_t access_count = reader.read_u64();
        const std::uint64_t last_access_sequence = reader.read_u64();
        records.push_back({
            .id = id,
            .key = std::move(key),
            .value = std::move(value),
            .confidence = confidence,
            .timestamp = timestamp,
            .access_count = access_count,
            .last_access_sequence = last_access_sequence,
        });
    }
    return memory::AssociativeMemory::restore(
        dimension,
        capacity,
        std::move(records),
        next_record_id,
        sequence
    );
}

void write_structural_statistics(
    detail::BufferWriter& writer,
    const learning::StructuralStatistics& statistics
) {
    writer.write_u64(statistics.modes_created);
    writer.write_u64(statistics.modes_split);
    writer.write_u64(statistics.modes_merged);
    writer.write_u64(statistics.modes_pruned);
}

[[nodiscard]] learning::StructuralStatistics read_structural_statistics(
    detail::BufferReader& reader
) {
    return {
        .modes_created = reader.read_u64(),
        .modes_split = reader.read_u64(),
        .modes_merged = reader.read_u64(),
        .modes_pruned = reader.read_u64(),
    };
}

void validate_checkpoint_for_save(const CheckpointData& checkpoint) {
    if (checkpoint.config.dimension == 0U ||
        checkpoint.config.maximum_modes == 0U) {
        throw std::invalid_argument(
            "checkpoint fabric configuration is invalid"
        );
    }
    if (checkpoint.modes.size() > checkpoint.config.maximum_modes) {
        throw std::invalid_argument(
            "checkpoint modes exceed configured maximum"
        );
    }
    if (checkpoint.associative_memory.dimension() !=
        checkpoint.config.dimension) {
        throw std::invalid_argument(
            "checkpoint associative-memory dimension is incompatible"
        );
    }
    if (checkpoint.update_strategy.empty()) {
        throw std::invalid_argument(
            "checkpoint update strategy must not be empty"
        );
    }

    core::ResonantFabric validation_fabric(checkpoint.config);
    for (const core::ResonantMode& mode : checkpoint.modes) {
        validation_fabric.add_mode(mode);
    }
}

[[nodiscard]] std::vector<std::uint8_t> build_payload(
    const CheckpointData& checkpoint
) {
    detail::BufferWriter writer;
    write_fabric_config(writer, checkpoint.config);
    writer.write_u64(checkpoint.master_seed);
    writer.write_u64(checkpoint.training_step);
    writer.write_string(checkpoint.update_strategy);
    writer.write_u64(
        static_cast<std::uint64_t>(checkpoint.modes.size())
    );
    for (const core::ResonantMode& mode : checkpoint.modes) {
        write_mode(writer, mode);
    }
    write_memory(writer, checkpoint.associative_memory);
    write_structural_statistics(
        writer,
        checkpoint.structural_statistics
    );
    writer.write_u64(
        static_cast<std::uint64_t>(
            checkpoint.experiment_metadata.size()
        )
    );
    for (const auto& [key, value] :
         checkpoint.experiment_metadata) {
        writer.write_string(key);
        writer.write_string(value);
    }
    return writer.take();
}

[[nodiscard]] std::vector<std::uint8_t> build_file(
    const std::span<const std::uint8_t> payload
) {
    detail::BufferWriter writer;
    writer.write_bytes(checkpoint_magic);
    writer.write_u32(checkpoint_format_version);
    writer.write_u64(static_cast<std::uint64_t>(payload.size()));
    writer.write_u64(detail::checksum(payload));
    writer.write_bytes(payload);
    return writer.take();
}

[[nodiscard]] ParsedCheckpoint parse_checkpoint(
    const std::filesystem::path& path,
    const CheckpointLoadOptions& options
) {
    const std::vector<std::uint8_t> file = detail::read_file(
        path,
        options.maximum_file_bytes
    );
    if (file.size() < header_size) {
        throw std::runtime_error("truncated checkpoint header");
    }
    detail::BufferReader header(file);
    for (const std::uint8_t expected : checkpoint_magic) {
        if (header.read_u8() != expected) {
            throw std::runtime_error("invalid checkpoint magic");
        }
    }
    const std::uint32_t version = header.read_u32();
    if (version != checkpoint_format_version) {
        throw std::runtime_error(
            "unsupported checkpoint format version"
        );
    }
    const std::size_t payload_size = detail::checked_size(
        header.read_u64(),
        file.size() - header_size,
        "checkpoint payload"
    );
    const std::uint64_t expected_checksum = header.read_u64();
    if (payload_size != header.remaining()) {
        throw std::runtime_error("checkpoint payload size mismatch");
    }
    const std::span<const std::uint8_t> payload =
        std::span<const std::uint8_t>(file).subspan(
            header_size,
            payload_size
        );
    const std::uint64_t actual_checksum = detail::checksum(payload);
    if (actual_checksum != expected_checksum) {
        throw std::runtime_error("checkpoint checksum mismatch");
    }

    detail::BufferReader reader(payload);
    core::FabricConfig config = read_fabric_config(reader, options);
    const std::uint64_t master_seed = reader.read_u64();
    const std::uint64_t training_step = reader.read_u64();
    std::string update_strategy = reader.read_string(
        options.maximum_string_bytes
    );
    if (update_strategy.empty()) {
        throw std::runtime_error(
            "checkpoint update strategy is empty"
        );
    }

    const std::size_t mode_count = detail::checked_size(
        reader.read_u64(),
        std::min(options.maximum_modes, config.maximum_modes),
        "checkpoint mode count"
    );
    std::vector<core::ResonantMode> modes;
    modes.reserve(mode_count);
    for (std::size_t mode_index = 0U;
         mode_index < mode_count;
         ++mode_index) {
        modes.push_back(
            read_mode(reader, config.dimension, options)
        );
    }

    memory::AssociativeMemory associative_memory = read_memory(
        reader,
        config.dimension,
        options
    );
    const learning::StructuralStatistics structural_statistics =
        read_structural_statistics(reader);
    const std::size_t metadata_count = detail::checked_size(
        reader.read_u64(),
        options.maximum_metadata_entries,
        "checkpoint metadata entry count"
    );
    std::map<std::string, std::string> experiment_metadata;
    for (std::size_t metadata_index = 0U;
         metadata_index < metadata_count;
         ++metadata_index) {
        std::string key = reader.read_string(
            options.maximum_string_bytes
        );
        std::string value = reader.read_string(
            options.maximum_string_bytes
        );
        if (key.empty()) {
            throw std::runtime_error(
                "checkpoint metadata key is empty"
            );
        }
        if (!experiment_metadata.emplace(
                std::move(key),
                std::move(value)
            ).second) {
            throw std::runtime_error(
                "checkpoint metadata keys are not unique"
            );
        }
    }
    if (!reader.empty()) {
        throw std::runtime_error(
            "checkpoint contains trailing payload data"
        );
    }

    CheckpointData data{
        .config = std::move(config),
        .master_seed = master_seed,
        .training_step = training_step,
        .modes = std::move(modes),
        .associative_memory = std::move(associative_memory),
        .structural_statistics = structural_statistics,
        .update_strategy = std::move(update_strategy),
        .experiment_metadata = std::move(experiment_metadata),
    };
    validate_checkpoint_for_save(data);
    return {
        .data = std::move(data),
        .payload_checksum = actual_checksum,
        .file_bytes = file.size(),
    };
}

}  // namespace

void save_checkpoint(
    const std::filesystem::path& path,
    const CheckpointData& checkpoint
) {
    validate_checkpoint_for_save(checkpoint);
    const std::vector<std::uint8_t> payload =
        build_payload(checkpoint);
    const std::vector<std::uint8_t> file = build_file(payload);
    detail::write_file_transactionally(path, file);
}

CheckpointData load_checkpoint(
    const std::filesystem::path& path,
    const CheckpointLoadOptions& options
) {
    return parse_checkpoint(path, options).data;
}

CheckpointSummary inspect_checkpoint(
    const std::filesystem::path& path,
    const CheckpointLoadOptions& options
) {
    ParsedCheckpoint parsed = parse_checkpoint(path, options);
    const std::size_t enabled_mode_count =
        static_cast<std::size_t>(std::count_if(
            parsed.data.modes.begin(),
            parsed.data.modes.end(),
            [](const core::ResonantMode& mode) {
                return mode.enabled;
            }
        ));
    return {
        .format_version = checkpoint_format_version,
        .master_seed = parsed.data.master_seed,
        .training_step = parsed.data.training_step,
        .dimension = parsed.data.config.dimension,
        .mode_count = parsed.data.modes.size(),
        .enabled_mode_count = enabled_mode_count,
        .associative_record_count =
            parsed.data.associative_memory.size(),
        .associative_capacity =
            parsed.data.associative_memory.capacity(),
        .structural_statistics =
            parsed.data.structural_statistics,
        .update_strategy = std::move(parsed.data.update_strategy),
        .experiment_metadata =
            std::move(parsed.data.experiment_metadata),
        .payload_checksum = parsed.payload_checksum,
        .file_bytes = parsed.file_bytes,
    };
}

}  // namespace rlf::storage
