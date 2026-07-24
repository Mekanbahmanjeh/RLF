#include "rlf/storage/rlf1_checkpoint.hpp"

#include "storage/binary_codec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rlf::storage {
namespace {

constexpr std::array<std::uint8_t, 8U> magic{
    'R', 'L', 'F', '1', 'C', 'K', 'P', '3'
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
    const Rlf1CheckpointLoadOptions& options
) {
    const std::size_t dimension = detail::checked_size(
        reader.read_u64(),
        options.maximum_dimension,
        "RLF-1 phase-vector dimension"
    );
    if (dimension != expected_dimension) {
        throw std::runtime_error("RLF-1 checkpoint dimension mismatch");
    }
    constexpr float tau = 2.0F * std::numbers::pi_v<float>;
    std::vector<float> values;
    values.reserve(dimension);
    for (std::size_t index = 0U; index < dimension; ++index) {
        const float angle = reader.read_float();
        if (!std::isfinite(angle) || angle < 0.0F || angle >= tau) {
            throw std::runtime_error("invalid RLF-1 phase angle");
        }
        values.push_back(angle);
    }
    return core::PhaseVector(std::move(values));
}

void write_config(
    detail::BufferWriter& writer,
    const core::LatentRouterConfig& config
) {
    writer.write_u64(static_cast<std::uint64_t>(config.dimension));
    writer.write_u64(static_cast<std::uint64_t>(config.maximum_cycles));
    writer.write_u64(static_cast<std::uint64_t>(config.search_node_budget));
    writer.write_u64(static_cast<std::uint64_t>(config.route_memory_capacity));
    writer.write_u64(static_cast<std::uint64_t>(config.maximum_modes));
    writer.write_u64(static_cast<std::uint64_t>(config.macro_minimum_occurrences));
    writer.write_u64(static_cast<std::uint64_t>(config.macro_maximum_length));
    writer.write_double(config.goal_similarity_threshold);
    writer.write_double(config.mode_creation_similarity);
    writer.write_double(config.mode_learning_rate);
    writer.write_double(config.utility_learning_rate);
    writer.write_double(config.eligibility_decay);
    writer.write_double(config.goal_progress_weight);
    writer.write_double(config.route_memory_weight);
    writer.write_double(config.successor_familiarity_weight);
    writer.write_double(config.route_repetition_penalty);
    writer.write_double(config.learned_halt_threshold);
    writer.write_double(config.learned_halt_goal_floor);
    writer.write_double(config.abstention_entropy_threshold);
    writer.write_double(config.minimum_action_score);
    writer.write_double(config.action_temperature);
    writer.write_double(config.abstention_resonance_threshold);
    writer.write_u64(static_cast<std::uint64_t>(config.search_beam_width));
    writer.write_u64(static_cast<std::uint64_t>(config.search_lookahead_depth));
    writer.write_bool(config.enable_route_memory);
    writer.write_bool(config.enable_macro_operators);
    writer.write_u32(static_cast<std::uint32_t>(config.credit_strategy));
    writer.write_u32(static_cast<std::uint32_t>(config.halt_policy));
}

[[nodiscard]] core::LatentRouterConfig read_config(
    detail::BufferReader& reader,
    const Rlf1CheckpointLoadOptions& options
) {
    core::LatentRouterConfig config;
    config.dimension = detail::checked_size(
        reader.read_u64(),
        options.maximum_dimension,
        "RLF-1 dimension"
    );
    config.maximum_cycles = detail::checked_size(
        reader.read_u64(),
        std::numeric_limits<std::size_t>::max(),
        "RLF-1 maximum cycles"
    );
    config.search_node_budget = detail::checked_size(
        reader.read_u64(),
        std::numeric_limits<std::size_t>::max(),
        "RLF-1 search budget"
    );
    config.route_memory_capacity = detail::checked_size(
        reader.read_u64(),
        options.maximum_route_records,
        "RLF-1 route memory capacity"
    );
    config.maximum_modes = detail::checked_size(
        reader.read_u64(),
        options.maximum_modes,
        "RLF-1 maximum modes"
    );
    config.macro_minimum_occurrences = detail::checked_size(
        reader.read_u64(),
        std::numeric_limits<std::size_t>::max(),
        "RLF-1 macro minimum occurrences"
    );
    config.macro_maximum_length = detail::checked_size(
        reader.read_u64(),
        options.maximum_route_length,
        "RLF-1 macro maximum length"
    );
    config.goal_similarity_threshold = reader.read_double();
    config.mode_creation_similarity = reader.read_double();
    config.mode_learning_rate = reader.read_double();
    config.utility_learning_rate = reader.read_double();
    config.eligibility_decay = reader.read_double();
    config.goal_progress_weight = reader.read_double();
    config.route_memory_weight = reader.read_double();
    config.successor_familiarity_weight = reader.read_double();
    config.route_repetition_penalty = reader.read_double();
    config.learned_halt_threshold = reader.read_double();
    config.learned_halt_goal_floor = reader.read_double();
    config.abstention_entropy_threshold = reader.read_double();
    config.minimum_action_score = reader.read_double();
    config.action_temperature = reader.read_double();
    config.abstention_resonance_threshold = reader.read_double();
    config.search_beam_width = detail::checked_size(
        reader.read_u64(),
        std::numeric_limits<std::size_t>::max(),
        "RLF-1 search beam width"
    );
    config.search_lookahead_depth = detail::checked_size(
        reader.read_u64(),
        options.maximum_route_length,
        "RLF-1 search lookahead depth"
    );
    config.enable_route_memory = reader.read_bool();
    config.enable_macro_operators = reader.read_bool();
    const std::uint32_t credit = reader.read_u32();
    const std::uint32_t halt = reader.read_u32();
    if (credit > static_cast<std::uint32_t>(
            core::LatentCreditStrategy::counterfactual_local) ||
        halt > static_cast<std::uint32_t>(
            core::LatentHaltPolicy::combined_safe)) {
        throw std::runtime_error("invalid RLF-1 policy enum");
    }
    config.credit_strategy = static_cast<core::LatentCreditStrategy>(credit);
    config.halt_policy = static_cast<core::LatentHaltPolicy>(halt);
    return config;
}

void write_operator_primitive(
    detail::BufferWriter& writer,
    const core::OperatorPrimitive& primitive
) {
    writer.write_u32(static_cast<std::uint32_t>(primitive.kind));
    writer.write_u64(static_cast<std::uint64_t>(primitive.begin_index));
    write_phase_vector(writer, primitive.phase_shift);
    writer.write_u64(static_cast<std::uint64_t>(primitive.permutation.size()));
    for (const std::size_t value : primitive.permutation) {
        writer.write_u64(static_cast<std::uint64_t>(value));
    }
}

[[nodiscard]] core::OperatorPrimitive read_operator_primitive(
    detail::BufferReader& reader,
    const std::size_t dimension,
    const Rlf1CheckpointLoadOptions& options
) {
    const std::uint32_t kind_value = reader.read_u32();
    if (kind_value > static_cast<std::uint32_t>(
            core::OperatorPrimitiveKind::conjugation)) {
        throw std::runtime_error("invalid RLF-1 operator primitive kind");
    }
    const auto kind = static_cast<core::OperatorPrimitiveKind>(kind_value);
    const std::size_t begin = detail::checked_size(
        reader.read_u64(),
        dimension,
        "RLF-1 operator begin index"
    );
    const core::PhaseVector phase = read_phase_vector(
        reader,
        dimension,
        options
    );
    const std::size_t permutation_count = detail::checked_size(
        reader.read_u64(),
        dimension,
        "RLF-1 permutation count"
    );
    std::vector<std::size_t> permutation;
    permutation.reserve(permutation_count);
    for (std::size_t index = 0U; index < permutation_count; ++index) {
        permutation.push_back(detail::checked_size(
            reader.read_u64(),
            dimension == 0U ? 0U : dimension - 1U,
            "RLF-1 permutation value"
        ));
    }
    switch (kind) {
    case core::OperatorPrimitiveKind::phase_shift:
        if (permutation_count != 0U) {
            throw std::runtime_error("shift primitive contains a permutation");
        }
        return core::OperatorPrimitive::shift(phase, begin);
    case core::OperatorPrimitiveKind::coordinate_permutation:
        if (permutation_count != dimension) {
            throw std::runtime_error("invalid RLF-1 permutation size");
        }
        return core::OperatorPrimitive::permute(
            std::move(permutation),
            begin
        );
    case core::OperatorPrimitiveKind::conjugation:
        if (permutation_count != 0U) {
            throw std::runtime_error("conjugation primitive contains a permutation");
        }
        return core::OperatorPrimitive::conjugate(dimension, begin);
    }
    throw std::runtime_error("invalid RLF-1 operator primitive");
}

void write_operator(
    detail::BufferWriter& writer,
    const core::RegisteredOperator& value
) {
    writer.write_u64(value.id);
    writer.write_string(value.name);
    writer.write_double(value.cost);
    writer.write_bool(value.macro);
    writer.write_u64(static_cast<std::uint64_t>(value.primitive_route.size()));
    for (const std::uint64_t operator_id : value.primitive_route) {
        writer.write_u64(operator_id);
    }
    writer.write_u64(static_cast<std::uint64_t>(
        value.transformation.primitives().size()
    ));
    for (const core::OperatorPrimitive& primitive :
         value.transformation.primitives()) {
        write_operator_primitive(writer, primitive);
    }
}

[[nodiscard]] core::RegisteredOperator read_operator(
    detail::BufferReader& reader,
    const std::size_t dimension,
    const Rlf1CheckpointLoadOptions& options
) {
    const std::uint64_t id = reader.read_u64();
    const std::string name = reader.read_string(options.maximum_string_bytes);
    const double cost = reader.read_double();
    const bool macro = reader.read_bool();
    if (id == 0ULL || name.empty() || !std::isfinite(cost) || cost <= 0.0) {
        throw std::runtime_error("invalid RLF-1 operator metadata");
    }
    const std::size_t route_size = detail::checked_size(
        reader.read_u64(),
        options.maximum_route_length,
        "RLF-1 primitive route size"
    );
    std::vector<std::uint64_t> route;
    route.reserve(route_size);
    for (std::size_t index = 0U; index < route_size; ++index) {
        route.push_back(reader.read_u64());
    }
    const std::size_t primitive_count = detail::checked_size(
        reader.read_u64(),
        options.maximum_route_length,
        "RLF-1 operator primitive count"
    );
    std::vector<core::OperatorPrimitive> primitives;
    primitives.reserve(primitive_count);
    for (std::size_t index = 0U; index < primitive_count; ++index) {
        primitives.push_back(read_operator_primitive(
            reader,
            dimension,
            options
        ));
    }
    return {
        .id = id,
        .name = name,
        .transformation = core::TransformationOperator(
            dimension,
            std::move(primitives)
        ),
        .cost = cost,
        .macro = macro,
        .primitive_route = std::move(route),
    };
}

void write_mode(
    detail::BufferWriter& writer,
    const core::LatentRoutingMode& mode
) {
    writer.write_u64(mode.id);
    write_phase_vector(writer, mode.key);
    writer.write_u64(mode.operator_id);
    writer.write_double(mode.utility);
    writer.write_double(mode.confidence);
    writer.write_double(mode.eligibility);
    writer.write_u64(mode.activation_count);
    writer.write_u64(mode.success_count);
    writer.write_u64(mode.failure_count);
    writer.write_u64(mode.creation_step);
    writer.write_u64(mode.last_used_step);
}

[[nodiscard]] core::LatentRoutingMode read_mode(
    detail::BufferReader& reader,
    const std::size_t dimension,
    const Rlf1CheckpointLoadOptions& options
) {
    core::LatentRoutingMode mode{
        .id = reader.read_u64(),
        .key = read_phase_vector(reader, dimension, options),
        .operator_id = reader.read_u64(),
        .utility = reader.read_double(),
        .confidence = reader.read_double(),
        .eligibility = reader.read_double(),
        .activation_count = reader.read_u64(),
        .success_count = reader.read_u64(),
        .failure_count = reader.read_u64(),
        .creation_step = reader.read_u64(),
        .last_used_step = reader.read_u64(),
    };
    if (mode.id == 0ULL || mode.operator_id == 0ULL ||
        !std::isfinite(mode.utility) ||
        !std::isfinite(mode.confidence) ||
        !std::isfinite(mode.eligibility)) {
        throw std::runtime_error("invalid RLF-1 routing mode");
    }
    return mode;
}

void write_halt_mode(
    detail::BufferWriter& writer,
    const core::LatentHaltMode& mode
) {
    writer.write_u64(mode.id);
    write_phase_vector(writer, mode.key);
    writer.write_double(mode.confidence);
    writer.write_double(mode.utility);
    writer.write_u64(mode.activation_count);
}

[[nodiscard]] core::LatentHaltMode read_halt_mode(
    detail::BufferReader& reader,
    const std::size_t dimension,
    const Rlf1CheckpointLoadOptions& options
) {
    core::LatentHaltMode mode{
        .id = reader.read_u64(),
        .key = read_phase_vector(reader, dimension, options),
        .confidence = reader.read_double(),
        .utility = reader.read_double(),
        .activation_count = reader.read_u64(),
    };
    if (mode.id == 0ULL || !std::isfinite(mode.confidence) ||
        !std::isfinite(mode.utility)) {
        throw std::runtime_error("invalid RLF-1 halt mode");
    }
    return mode;
}

void write_route_record(
    detail::BufferWriter& writer,
    const core::RouteMemoryRecord& record
) {
    writer.write_u64(record.id);
    write_phase_vector(writer, record.start_goal_signature);
    writer.write_u64(static_cast<std::uint64_t>(record.route.size()));
    for (const std::uint64_t operator_id : record.route) {
        writer.write_u64(operator_id);
    }
    writer.write_double(record.confidence);
    writer.write_double(record.utility);
    writer.write_u64(record.usage_count);
    writer.write_u64(record.observation_count);
    writer.write_u64(record.creation_step);
    writer.write_u64(record.last_used_step);
    writer.write_u64(record.route_hash);
}

[[nodiscard]] core::RouteMemoryRecord read_route_record(
    detail::BufferReader& reader,
    const std::size_t dimension,
    const Rlf1CheckpointLoadOptions& options
) {
    const std::uint64_t id = reader.read_u64();
    core::PhaseVector signature = read_phase_vector(
        reader,
        dimension,
        options
    );
    const std::size_t route_size = detail::checked_size(
        reader.read_u64(),
        options.maximum_route_length,
        "RLF-1 route record length"
    );
    std::vector<std::uint64_t> route;
    route.reserve(route_size);
    for (std::size_t index = 0U; index < route_size; ++index) {
        route.push_back(reader.read_u64());
    }
    core::RouteMemoryRecord record{
        .id = id,
        .start_goal_signature = std::move(signature),
        .route = std::move(route),
        .confidence = reader.read_double(),
        .utility = reader.read_double(),
        .usage_count = reader.read_u64(),
        .observation_count = reader.read_u64(),
        .creation_step = reader.read_u64(),
        .last_used_step = reader.read_u64(),
        .route_hash = reader.read_u64(),
    };
    if (record.id == 0ULL || record.route.empty() ||
        record.observation_count == 0ULL ||
        !std::isfinite(record.confidence) ||
        !std::isfinite(record.utility) ||
        record.route_hash != core::LatentRouter::route_hash(record.route)) {
        throw std::runtime_error("invalid RLF-1 route memory record");
    }
    return record;
}

[[nodiscard]] std::vector<std::uint8_t> make_payload(
    const core::LatentRouterSnapshot& snapshot
) {
    detail::BufferWriter writer;
    write_config(writer, snapshot.config);
    writer.write_u64(snapshot.seed);
    writer.write_u64(snapshot.training_step);
    writer.write_u64(snapshot.next_operator_id);
    writer.write_u64(snapshot.next_mode_id);
    writer.write_u64(snapshot.next_halt_mode_id);
    writer.write_u64(snapshot.next_route_record_id);
    writer.write_u64(static_cast<std::uint64_t>(snapshot.operators.size()));
    for (const core::RegisteredOperator& value : snapshot.operators) {
        write_operator(writer, value);
    }
    writer.write_u64(static_cast<std::uint64_t>(snapshot.modes.size()));
    for (const core::LatentRoutingMode& value : snapshot.modes) {
        write_mode(writer, value);
    }
    writer.write_u64(static_cast<std::uint64_t>(snapshot.halt_modes.size()));
    for (const core::LatentHaltMode& value : snapshot.halt_modes) {
        write_halt_mode(writer, value);
    }
    writer.write_u64(static_cast<std::uint64_t>(snapshot.route_memory.size()));
    for (const core::RouteMemoryRecord& value : snapshot.route_memory) {
        write_route_record(writer, value);
    }
    return writer.take();
}

struct Parsed final {
    core::LatentRouterSnapshot snapshot;
    std::uint64_t checksum{};
    std::size_t file_bytes{};
};

[[nodiscard]] Parsed parse(
    const std::filesystem::path& path,
    const Rlf1CheckpointLoadOptions& options
) {
    const std::vector<std::uint8_t> bytes = detail::read_file(
        path,
        options.maximum_file_bytes
    );
    if (bytes.size() < header_size) {
        throw std::runtime_error("truncated RLF-1 checkpoint header");
    }
    detail::BufferReader header(bytes);
    for (const std::uint8_t expected : magic) {
        if (header.read_u8() != expected) {
            throw std::runtime_error("invalid RLF-1 checkpoint magic");
        }
    }
    const std::uint32_t version = header.read_u32();
    if (version != rlf1_checkpoint_format_version) {
        throw std::runtime_error("unsupported RLF-1 checkpoint version");
    }
    const std::size_t payload_size = detail::checked_size(
        header.read_u64(),
        options.maximum_file_bytes,
        "RLF-1 payload size"
    );
    const std::uint64_t expected_checksum = header.read_u64();
    if (payload_size != header.remaining()) {
        throw std::runtime_error("RLF-1 checkpoint payload-size mismatch");
    }
    const std::span<const std::uint8_t> payload(
        bytes.data() + static_cast<std::ptrdiff_t>(header_size),
        payload_size
    );
    const std::uint64_t actual_checksum = detail::checksum(payload);
    if (actual_checksum != expected_checksum) {
        throw std::runtime_error("RLF-1 checkpoint checksum mismatch");
    }
    detail::BufferReader reader(payload);
    core::LatentRouterSnapshot snapshot;
    snapshot.config = read_config(reader, options);
    snapshot.seed = reader.read_u64();
    snapshot.training_step = reader.read_u64();
    snapshot.next_operator_id = reader.read_u64();
    snapshot.next_mode_id = reader.read_u64();
    snapshot.next_halt_mode_id = reader.read_u64();
    snapshot.next_route_record_id = reader.read_u64();

    const std::size_t operator_count = detail::checked_size(
        reader.read_u64(),
        options.maximum_operators,
        "RLF-1 operator count"
    );
    snapshot.operators.reserve(operator_count);
    std::unordered_set<std::uint64_t> operator_ids;
    std::set<std::string> operator_names;
    for (std::size_t index = 0U; index < operator_count; ++index) {
        core::RegisteredOperator value = read_operator(
            reader,
            snapshot.config.dimension,
            options
        );
        if (!operator_ids.insert(value.id).second ||
            !operator_names.insert(value.name).second) {
            throw std::runtime_error("duplicate RLF-1 operator identity");
        }
        snapshot.operators.push_back(std::move(value));
    }
    for (const core::RegisteredOperator& value : snapshot.operators) {
        for (const std::uint64_t primitive_id : value.primitive_route) {
            if (!operator_ids.contains(primitive_id)) {
                throw std::runtime_error("RLF-1 macro references unknown operator");
            }
        }
    }

    const std::size_t mode_count = detail::checked_size(
        reader.read_u64(),
        options.maximum_modes,
        "RLF-1 routing mode count"
    );
    snapshot.modes.reserve(mode_count);
    std::unordered_set<std::uint64_t> mode_ids;
    for (std::size_t index = 0U; index < mode_count; ++index) {
        core::LatentRoutingMode value = read_mode(
            reader,
            snapshot.config.dimension,
            options
        );
        if (!mode_ids.insert(value.id).second ||
            !operator_ids.contains(value.operator_id)) {
            throw std::runtime_error("invalid RLF-1 routing mode reference");
        }
        snapshot.modes.push_back(std::move(value));
    }

    const std::size_t halt_count = detail::checked_size(
        reader.read_u64(),
        options.maximum_halt_modes,
        "RLF-1 halt mode count"
    );
    snapshot.halt_modes.reserve(halt_count);
    std::unordered_set<std::uint64_t> halt_ids;
    for (std::size_t index = 0U; index < halt_count; ++index) {
        core::LatentHaltMode value = read_halt_mode(
            reader,
            snapshot.config.dimension,
            options
        );
        if (!halt_ids.insert(value.id).second) {
            throw std::runtime_error("duplicate RLF-1 halt mode ID");
        }
        snapshot.halt_modes.push_back(std::move(value));
    }

    const std::size_t route_count = detail::checked_size(
        reader.read_u64(),
        options.maximum_route_records,
        "RLF-1 route-memory count"
    );
    snapshot.route_memory.reserve(route_count);
    std::unordered_set<std::uint64_t> route_ids;
    for (std::size_t index = 0U; index < route_count; ++index) {
        core::RouteMemoryRecord value = read_route_record(
            reader,
            snapshot.config.dimension,
            options
        );
        if (!route_ids.insert(value.id).second) {
            throw std::runtime_error("duplicate RLF-1 route record ID");
        }
        for (const std::uint64_t operator_id : value.route) {
            if (!operator_ids.contains(operator_id)) {
                throw std::runtime_error("RLF-1 route references unknown operator");
            }
        }
        snapshot.route_memory.push_back(std::move(value));
    }
    if (!reader.empty()) {
        throw std::runtime_error("trailing RLF-1 checkpoint payload data");
    }
    return {
        .snapshot = std::move(snapshot),
        .checksum = actual_checksum,
        .file_bytes = bytes.size(),
    };
}

}  // namespace

void save_rlf1_checkpoint(
    const std::filesystem::path& path,
    const core::LatentRouter& router
) {
    const std::vector<std::uint8_t> payload = make_payload(router.snapshot());
    detail::BufferWriter file;
    file.write_bytes(magic);
    file.write_u32(rlf1_checkpoint_format_version);
    file.write_u64(static_cast<std::uint64_t>(payload.size()));
    file.write_u64(detail::checksum(payload));
    file.write_bytes(payload);
    detail::write_file_transactionally(path, file.bytes());
}

core::LatentRouter load_rlf1_checkpoint(
    const std::filesystem::path& path,
    const Rlf1CheckpointLoadOptions& options
) {
    Parsed parsed = parse(path, options);
    return core::LatentRouter::from_snapshot(std::move(parsed.snapshot));
}

Rlf1CheckpointSummary inspect_rlf1_checkpoint(
    const std::filesystem::path& path,
    const Rlf1CheckpointLoadOptions& options
) {
    const Parsed parsed = parse(path, options);
    std::size_t macros = 0U;
    for (const core::RegisteredOperator& value : parsed.snapshot.operators) {
        macros += value.macro ? 1U : 0U;
    }
    return {
        .format_version = rlf1_checkpoint_format_version,
        .seed = parsed.snapshot.seed,
        .training_step = parsed.snapshot.training_step,
        .dimension = parsed.snapshot.config.dimension,
        .operator_count = parsed.snapshot.operators.size(),
        .macro_operator_count = macros,
        .routing_mode_count = parsed.snapshot.modes.size(),
        .halt_mode_count = parsed.snapshot.halt_modes.size(),
        .route_memory_count = parsed.snapshot.route_memory.size(),
        .payload_checksum = parsed.checksum,
        .file_bytes = parsed.file_bytes,
    };
}

}  // namespace rlf::storage
