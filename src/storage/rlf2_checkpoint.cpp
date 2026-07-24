#include "rlf/storage/rlf2_checkpoint.hpp"

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
    'R', 'L', 'F', '2', 'C', 'K', 'P', '4'
};
constexpr std::size_t header_size = 8U + 4U + 8U + 8U;

void write_phase_vector(
    detail::BufferWriter& writer,
    const core::PhaseVector& value
) {
    writer.write_u64(static_cast<std::uint64_t>(value.size()));
    for (const float angle : value.angles()) writer.write_float(angle);
}

[[nodiscard]] core::PhaseVector read_phase_vector(
    detail::BufferReader& reader,
    const std::size_t expected_dimension,
    const Rlf2CheckpointLoadOptions& options
) {
    const std::size_t dimension = detail::checked_size(
        reader.read_u64(),
        options.maximum_dimension,
        "RLF-2 phase-vector dimension"
    );
    if (dimension != expected_dimension) {
        throw std::runtime_error("RLF-2 checkpoint dimension mismatch");
    }
    constexpr float tau = 2.0F * std::numbers::pi_v<float>;
    std::vector<float> values;
    values.reserve(dimension);
    for (std::size_t index = 0U; index < dimension; ++index) {
        const float angle = reader.read_float();
        if (!std::isfinite(angle) || angle < 0.0F || angle >= tau) {
            throw std::runtime_error("invalid RLF-2 phase angle");
        }
        values.push_back(angle);
    }
    return core::PhaseVector(std::move(values));
}

void write_config(
    detail::BufferWriter& writer,
    const core::PredictiveSkillConfig& config
) {
    writer.write_u64(config.dimension);
    writer.write_u64(config.maximum_cycles);
    writer.write_u64(config.maximum_route_depth);
    writer.write_u64(config.planner_node_budget);
    writer.write_u64(config.maximum_skills);
    writer.write_u64(config.maximum_subgoal_prototypes);
    writer.write_u64(config.maximum_skill_length);
    writer.write_u64(config.minimum_skill_support);
    writer.write_u64(config.nearest_prototypes);
    writer.write_double(config.goal_similarity_threshold);
    writer.write_double(config.prototype_merge_distance);
    writer.write_double(config.prototype_distance_scale);
    writer.write_double(config.learned_value_weight);
    writer.write_double(config.successor_value_weight);
    writer.write_double(config.direct_progress_weight);
    writer.write_double(config.causal_advantage_weight);
    writer.write_double(config.skill_cost_weight);
    writer.write_double(config.repetition_penalty);
    writer.write_double(config.action_temperature);
    writer.write_double(config.abstention_uncertainty_threshold);
    writer.write_double(config.abstention_value_threshold);
    writer.write_bool(config.enable_skill_consolidation);
    writer.write_bool(config.enable_intervention_credit);
}

[[nodiscard]] core::PredictiveSkillConfig read_config(
    detail::BufferReader& reader,
    const Rlf2CheckpointLoadOptions& options
) {
    core::PredictiveSkillConfig config;
    config.dimension = detail::checked_size(
        reader.read_u64(), options.maximum_dimension, "RLF-2 dimension");
    config.maximum_cycles = detail::checked_size(
        reader.read_u64(), std::numeric_limits<std::size_t>::max(),
        "RLF-2 maximum cycles");
    config.maximum_route_depth = detail::checked_size(
        reader.read_u64(), options.maximum_route_length,
        "RLF-2 maximum route depth");
    config.planner_node_budget = detail::checked_size(
        reader.read_u64(), std::numeric_limits<std::size_t>::max(),
        "RLF-2 planner budget");
    config.maximum_skills = detail::checked_size(
        reader.read_u64(), options.maximum_skills,
        "RLF-2 maximum skills");
    config.maximum_subgoal_prototypes = detail::checked_size(
        reader.read_u64(), options.maximum_prototypes,
        "RLF-2 maximum prototypes");
    config.maximum_skill_length = detail::checked_size(
        reader.read_u64(), options.maximum_route_length,
        "RLF-2 maximum skill length");
    config.minimum_skill_support = detail::checked_size(
        reader.read_u64(), std::numeric_limits<std::size_t>::max(),
        "RLF-2 minimum skill support");
    config.nearest_prototypes = detail::checked_size(
        reader.read_u64(), options.maximum_prototypes,
        "RLF-2 nearest prototypes");
    config.goal_similarity_threshold = reader.read_double();
    config.prototype_merge_distance = reader.read_double();
    config.prototype_distance_scale = reader.read_double();
    config.learned_value_weight = reader.read_double();
    config.successor_value_weight = reader.read_double();
    config.direct_progress_weight = reader.read_double();
    config.causal_advantage_weight = reader.read_double();
    config.skill_cost_weight = reader.read_double();
    config.repetition_penalty = reader.read_double();
    config.action_temperature = reader.read_double();
    config.abstention_uncertainty_threshold = reader.read_double();
    config.abstention_value_threshold = reader.read_double();
    config.enable_skill_consolidation = reader.read_bool();
    config.enable_intervention_credit = reader.read_bool();
    return config;
}

void write_primitive(
    detail::BufferWriter& writer,
    const core::OperatorPrimitive& primitive
) {
    writer.write_u32(static_cast<std::uint32_t>(primitive.kind));
    writer.write_u64(primitive.begin_index);
    write_phase_vector(writer, primitive.phase_shift);
    writer.write_u64(primitive.permutation.size());
    for (const std::size_t value : primitive.permutation) writer.write_u64(value);
}

[[nodiscard]] core::OperatorPrimitive read_primitive(
    detail::BufferReader& reader,
    const std::size_t dimension,
    const Rlf2CheckpointLoadOptions& options
) {
    const std::uint32_t kind_value = reader.read_u32();
    if (kind_value > static_cast<std::uint32_t>(
            core::OperatorPrimitiveKind::conjugation)) {
        throw std::runtime_error("invalid RLF-2 primitive kind");
    }
    const auto kind = static_cast<core::OperatorPrimitiveKind>(kind_value);
    const std::size_t begin = detail::checked_size(
        reader.read_u64(), dimension, "RLF-2 primitive begin");
    core::PhaseVector phase = read_phase_vector(reader, dimension, options);
    const std::size_t permutation_count = detail::checked_size(
        reader.read_u64(), dimension, "RLF-2 permutation count");
    std::vector<std::size_t> permutation;
    permutation.reserve(permutation_count);
    for (std::size_t index = 0U; index < permutation_count; ++index) {
        permutation.push_back(detail::checked_size(
            reader.read_u64(), dimension - 1U, "RLF-2 permutation value"));
    }
    switch (kind) {
    case core::OperatorPrimitiveKind::phase_shift:
        if (!permutation.empty()) throw std::runtime_error("shift has permutation");
        return core::OperatorPrimitive::shift(std::move(phase), begin);
    case core::OperatorPrimitiveKind::coordinate_permutation:
        if (permutation.size() != dimension) {
            throw std::runtime_error("invalid RLF-2 permutation size");
        }
        return core::OperatorPrimitive::permute(std::move(permutation), begin);
    case core::OperatorPrimitiveKind::conjugation:
        if (!permutation.empty()) {
            throw std::runtime_error("conjugation has permutation");
        }
        return core::OperatorPrimitive::conjugate(dimension, begin);
    }
    throw std::runtime_error("unreachable RLF-2 primitive kind");
}

void write_transformation(
    detail::BufferWriter& writer,
    const core::TransformationOperator& value
) {
    writer.write_u64(value.dimension());
    writer.write_u64(value.primitives().size());
    for (const core::OperatorPrimitive& primitive : value.primitives()) {
        write_primitive(writer, primitive);
    }
}

[[nodiscard]] core::TransformationOperator read_transformation(
    detail::BufferReader& reader,
    const std::size_t dimension,
    const Rlf2CheckpointLoadOptions& options
) {
    const std::size_t stored_dimension = detail::checked_size(
        reader.read_u64(), options.maximum_dimension,
        "RLF-2 transformation dimension");
    if (stored_dimension != dimension) {
        throw std::runtime_error("RLF-2 transformation dimension mismatch");
    }
    const std::size_t primitive_count = detail::checked_size(
        reader.read_u64(), options.maximum_route_length * 8U,
        "RLF-2 primitive count");
    std::vector<core::OperatorPrimitive> primitives;
    primitives.reserve(primitive_count);
    for (std::size_t index = 0U; index < primitive_count; ++index) {
        primitives.push_back(read_primitive(reader, dimension, options));
    }
    return core::TransformationOperator(dimension, std::move(primitives));
}

void write_stats(
    detail::BufferWriter& writer,
    const core::Rlf2TrainingStats& stats
) {
    writer.write_u64(stats.observed_routes);
    writer.write_u64(stats.observed_transitions);
    writer.write_u64(stats.intervention_tests);
    writer.write_u64(stats.intervention_alternative_successes);
    writer.write_u64(stats.prototypes_created);
    writer.write_u64(stats.prototypes_merged);
    writer.write_u64(stats.skills_proposed);
    writer.write_u64(stats.skills_accepted);
    writer.write_u64(stats.routes_segmented);
}

[[nodiscard]] core::Rlf2TrainingStats read_stats(
    detail::BufferReader& reader
) {
    return {
        .observed_routes = reader.read_u64(),
        .observed_transitions = reader.read_u64(),
        .intervention_tests = reader.read_u64(),
        .intervention_alternative_successes = reader.read_u64(),
        .prototypes_created = reader.read_u64(),
        .prototypes_merged = reader.read_u64(),
        .skills_proposed = reader.read_u64(),
        .skills_accepted = reader.read_u64(),
        .routes_segmented = reader.read_u64(),
    };
}

[[nodiscard]] std::vector<std::uint8_t> encode_payload(
    const core::PredictiveSkillFabric& fabric
) {
    const core::PredictiveSkillSnapshot snapshot = fabric.snapshot();
    detail::BufferWriter writer;
    write_config(writer, snapshot.config);
    writer.write_u64(snapshot.seed);
    writer.write_u64(snapshot.training_step);
    writer.write_u64(snapshot.next_operator_id);
    writer.write_u64(snapshot.next_skill_id);
    writer.write_u64(snapshot.next_prototype_id);
    write_stats(writer, snapshot.training_stats);

    writer.write_u64(snapshot.operators.size());
    for (const core::PredictiveOperator& value : snapshot.operators) {
        writer.write_u64(value.id);
        writer.write_string(value.name);
        writer.write_double(value.cost);
        write_transformation(writer, value.forward);
    }
    writer.write_u64(snapshot.skills.size());
    for (const core::CausalSkill& value : snapshot.skills) {
        writer.write_u64(value.id);
        writer.write_string(value.name);
        writer.write_u64(value.primitive_route.size());
        for (const std::uint64_t id : value.primitive_route) writer.write_u64(id);
        write_transformation(writer, value.forward);
        writer.write_u64(value.primitive_length);
        writer.write_u64(value.support);
        writer.write_u64(value.success_count);
        writer.write_u64(value.failure_count);
        writer.write_double(value.utility);
        writer.write_double(value.mean_causal_advantage);
        writer.write_bool(value.accepted);
    }
    writer.write_u64(snapshot.prototypes.size());
    for (const core::SubgoalPrototype& value : snapshot.prototypes) {
        writer.write_u64(value.id);
        writer.write_u64(value.response_profile.size());
        for (const float item : value.response_profile) writer.write_float(item);
        writer.write_u64(value.skill_id);
        writer.write_double(value.remaining_steps);
        writer.write_double(value.terminal_value);
        writer.write_double(value.causal_advantage);
        writer.write_double(value.confidence);
        writer.write_u64(value.support);
        writer.write_u64(value.creation_step);
        writer.write_u64(value.last_used_step);
    }
    return writer.take();
}

[[nodiscard]] core::PredictiveSkillSnapshot decode_payload(
    const std::span<const std::uint8_t> bytes,
    const Rlf2CheckpointLoadOptions& options
) {
    detail::BufferReader reader(bytes);
    core::PredictiveSkillSnapshot snapshot;
    snapshot.config = read_config(reader, options);
    snapshot.seed = reader.read_u64();
    snapshot.training_step = reader.read_u64();
    snapshot.next_operator_id = reader.read_u64();
    snapshot.next_skill_id = reader.read_u64();
    snapshot.next_prototype_id = reader.read_u64();
    snapshot.training_stats = read_stats(reader);

    const std::size_t operator_count = detail::checked_size(
        reader.read_u64(), options.maximum_operators,
        "RLF-2 operator count");
    snapshot.operators.reserve(operator_count);
    std::unordered_set<std::uint64_t> operator_ids;
    for (std::size_t index = 0U; index < operator_count; ++index) {
        const std::uint64_t id = reader.read_u64();
        if (id == 0ULL || !operator_ids.insert(id).second) {
            throw std::runtime_error("invalid or duplicate RLF-2 operator ID");
        }
        std::string name = reader.read_string(options.maximum_string_bytes);
        const double cost = reader.read_double();
        if (name.empty() || !std::isfinite(cost) || cost <= 0.0) {
            throw std::runtime_error("invalid RLF-2 operator metadata");
        }
        core::TransformationOperator forward = read_transformation(
            reader, snapshot.config.dimension, options);
        snapshot.operators.push_back({
            .id = id,
            .name = std::move(name),
            .forward = forward,
            .inverse = forward.inverse(),
            .cost = cost,
        });
    }

    const std::size_t skill_count = detail::checked_size(
        reader.read_u64(), options.maximum_skills, "RLF-2 skill count");
    snapshot.skills.reserve(skill_count);
    std::unordered_set<std::uint64_t> skill_ids;
    for (std::size_t index = 0U; index < skill_count; ++index) {
        const std::uint64_t id = reader.read_u64();
        if (id == 0ULL || !skill_ids.insert(id).second) {
            throw std::runtime_error("invalid or duplicate RLF-2 skill ID");
        }
        std::string name = reader.read_string(options.maximum_string_bytes);
        const std::size_t route_count = detail::checked_size(
            reader.read_u64(), options.maximum_route_length,
            "RLF-2 skill route length");
        if (route_count == 0U) throw std::runtime_error("empty RLF-2 skill route");
        std::vector<std::uint64_t> route;
        route.reserve(route_count);
        for (std::size_t route_index = 0U;
             route_index < route_count;
             ++route_index) {
            const std::uint64_t operator_id = reader.read_u64();
            if (!operator_ids.contains(operator_id)) {
                throw std::runtime_error("RLF-2 skill references unknown operator");
            }
            route.push_back(operator_id);
        }
        core::TransformationOperator forward = read_transformation(
            reader, snapshot.config.dimension, options);
        const std::size_t primitive_length = detail::checked_size(
            reader.read_u64(), options.maximum_route_length,
            "RLF-2 primitive length");
        const std::uint64_t support = reader.read_u64();
        const std::uint64_t success_count = reader.read_u64();
        const std::uint64_t failure_count = reader.read_u64();
        const double utility = reader.read_double();
        const double advantage = reader.read_double();
        const bool accepted = reader.read_bool();
        if (name.empty() || primitive_length != route.size() ||
            !std::isfinite(utility) || !std::isfinite(advantage)) {
            throw std::runtime_error("invalid RLF-2 skill metadata");
        }
        snapshot.skills.push_back({
            .id = id,
            .name = std::move(name),
            .primitive_route = std::move(route),
            .forward = forward,
            .inverse = forward.inverse(),
            .primitive_length = primitive_length,
            .support = support,
            .success_count = success_count,
            .failure_count = failure_count,
            .utility = utility,
            .mean_causal_advantage = advantage,
            .accepted = accepted,
        });
    }

    const std::size_t prototype_count = detail::checked_size(
        reader.read_u64(), options.maximum_prototypes,
        "RLF-2 prototype count");
    snapshot.prototypes.reserve(prototype_count);
    std::unordered_set<std::uint64_t> prototype_ids;
    for (std::size_t index = 0U; index < prototype_count; ++index) {
        const std::uint64_t id = reader.read_u64();
        if (id == 0ULL || !prototype_ids.insert(id).second) {
            throw std::runtime_error("invalid or duplicate RLF-2 prototype ID");
        }
        const std::size_t profile_size = detail::checked_size(
            reader.read_u64(), options.maximum_profile_values,
            "RLF-2 profile size");
        if (profile_size == 0U) throw std::runtime_error("empty RLF-2 profile");
        std::vector<float> profile;
        profile.reserve(profile_size);
        for (std::size_t value_index = 0U;
             value_index < profile_size;
             ++value_index) {
            const float value = reader.read_float();
            if (!std::isfinite(value)) {
                throw std::runtime_error("non-finite RLF-2 profile value");
            }
            profile.push_back(value);
        }
        const std::uint64_t skill_id = reader.read_u64();
        if (!skill_ids.contains(skill_id)) {
            throw std::runtime_error("RLF-2 prototype references unknown skill");
        }
        const double remaining = reader.read_double();
        const double terminal = reader.read_double();
        const double advantage = reader.read_double();
        const double confidence = reader.read_double();
        const std::uint64_t support = reader.read_u64();
        const std::uint64_t creation = reader.read_u64();
        const std::uint64_t last_used = reader.read_u64();
        if (!std::isfinite(remaining) || remaining < 0.0 ||
            !std::isfinite(terminal) || !std::isfinite(advantage) ||
            !std::isfinite(confidence) || confidence < 0.0 || confidence > 1.0 ||
            support == 0ULL) {
            throw std::runtime_error("invalid RLF-2 prototype metadata");
        }
        snapshot.prototypes.push_back({
            .id = id,
            .response_profile = std::move(profile),
            .skill_id = skill_id,
            .remaining_steps = remaining,
            .terminal_value = terminal,
            .causal_advantage = advantage,
            .confidence = confidence,
            .support = support,
            .creation_step = creation,
            .last_used_step = last_used,
        });
    }
    if (!reader.empty()) {
        throw std::runtime_error("trailing RLF-2 checkpoint payload data");
    }
    const auto maximum_id = [](const auto& values) {
        std::uint64_t maximum = 0ULL;
        for (const auto& value : values) maximum = std::max(maximum, value.id);
        return maximum;
    };
    if (snapshot.next_operator_id <= maximum_id(snapshot.operators) ||
        snapshot.next_skill_id <= maximum_id(snapshot.skills) ||
        snapshot.next_prototype_id <= maximum_id(snapshot.prototypes)) {
        throw std::runtime_error("invalid RLF-2 next-ID counter");
    }
    return snapshot;
}

struct DecodedFile final {
    core::PredictiveSkillSnapshot snapshot;
    std::uint64_t checksum{};
    std::size_t file_bytes{};
};

[[nodiscard]] DecodedFile decode_file(
    const std::filesystem::path& path,
    const Rlf2CheckpointLoadOptions& options
) {
    const std::vector<std::uint8_t> bytes = detail::read_file(
        path, options.maximum_file_bytes);
    if (bytes.size() < header_size) {
        throw std::runtime_error("truncated RLF-2 checkpoint header");
    }
    detail::BufferReader reader(bytes);
    for (const std::uint8_t expected : magic) {
        if (reader.read_u8() != expected) {
            throw std::runtime_error("invalid RLF-2 checkpoint magic");
        }
    }
    const std::uint32_t version = reader.read_u32();
    if (version != rlf2_checkpoint_format_version) {
        throw std::runtime_error("unsupported RLF-2 checkpoint version");
    }
    const std::size_t payload_size = detail::checked_size(
        reader.read_u64(), options.maximum_file_bytes,
        "RLF-2 payload size");
    const std::uint64_t expected_checksum = reader.read_u64();
    if (reader.remaining() != payload_size) {
        throw std::runtime_error("RLF-2 checkpoint payload-size mismatch");
    }
    const std::vector<std::uint8_t> payload = reader.read_bytes(payload_size);
    const std::uint64_t actual_checksum = detail::checksum(payload);
    if (actual_checksum != expected_checksum) {
        throw std::runtime_error("RLF-2 checkpoint checksum mismatch");
    }
    return {
        .snapshot = decode_payload(payload, options),
        .checksum = actual_checksum,
        .file_bytes = bytes.size(),
    };
}

}  // namespace

void save_rlf2_checkpoint(
    const std::filesystem::path& path,
    const core::PredictiveSkillFabric& fabric
) {
    const std::vector<std::uint8_t> payload = encode_payload(fabric);
    detail::BufferWriter writer;
    writer.write_bytes(magic);
    writer.write_u32(rlf2_checkpoint_format_version);
    writer.write_u64(payload.size());
    writer.write_u64(detail::checksum(payload));
    writer.write_bytes(payload);
    const std::vector<std::uint8_t> bytes = writer.take();
    detail::write_file_transactionally(path, bytes);
}

core::PredictiveSkillFabric load_rlf2_checkpoint(
    const std::filesystem::path& path,
    const Rlf2CheckpointLoadOptions& options
) {
    DecodedFile decoded = decode_file(path, options);
    return core::PredictiveSkillFabric::from_snapshot(
        std::move(decoded.snapshot));
}

Rlf2CheckpointSummary inspect_rlf2_checkpoint(
    const std::filesystem::path& path,
    const Rlf2CheckpointLoadOptions& options
) {
    const DecodedFile decoded = decode_file(path, options);
    return {
        .format_version = rlf2_checkpoint_format_version,
        .seed = decoded.snapshot.seed,
        .training_step = decoded.snapshot.training_step,
        .dimension = decoded.snapshot.config.dimension,
        .operator_count = decoded.snapshot.operators.size(),
        .skill_count = decoded.snapshot.skills.size(),
        .compound_skill_count = static_cast<std::size_t>(std::count_if(
            decoded.snapshot.skills.begin(),
            decoded.snapshot.skills.end(),
            [](const core::CausalSkill& value) {
                return value.primitive_length > 1U;
            })),
        .prototype_count = decoded.snapshot.prototypes.size(),
        .payload_checksum = decoded.checksum,
        .file_bytes = decoded.file_bytes,
    };
}

}  // namespace rlf::storage
