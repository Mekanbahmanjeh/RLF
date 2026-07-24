#include "rlf/storage/rlf6_checkpoint.hpp"

#include "storage/binary_codec.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rlf::storage {
namespace {

constexpr std::array<std::uint8_t, 8U> magic{
    'R', 'L', 'F', '6', 'C', 'K', 'P', '8'
};
constexpr std::size_t header_size = 8U + 4U + 8U + 8U;

template <typename Enum>
void write_enum(detail::BufferWriter& writer, const Enum value) {
    writer.write_u8(static_cast<std::uint8_t>(value));
}

template <typename Enum>
[[nodiscard]] Enum read_enum(
    detail::BufferReader& reader,
    const std::uint8_t maximum,
    const char* label
) {
    const auto value = reader.read_u8();
    if (value > maximum) {
        throw std::runtime_error(std::string("invalid RLF-6 ") + label);
    }
    return static_cast<Enum>(value);
}

void validate_probability(const double value, const char* label) {
    if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
        throw std::runtime_error(std::string("invalid RLF-6 ") + label);
    }
}

void validate_nonnegative(const double value, const char* label) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::runtime_error(std::string("invalid RLF-6 ") + label);
    }
}

void write_fact(detail::BufferWriter& writer, const agent::Fact& fact) {
    writer.write_string(fact.key);
    writer.write_string(fact.value);
    writer.write_bool(fact.negated);
}

[[nodiscard]] agent::Fact read_fact(
    detail::BufferReader& reader,
    const Rlf6CheckpointLoadOptions& options
) {
    agent::Fact fact;
    fact.key = reader.read_string(options.maximum_string_bytes);
    fact.value = reader.read_string(options.maximum_string_bytes);
    fact.negated = reader.read_bool();
    if (fact.key.empty()) {
        throw std::runtime_error("invalid RLF-6 empty fact key");
    }
    return fact;
}

void write_fact_vector(
    detail::BufferWriter& writer,
    const std::vector<agent::Fact>& facts
) {
    writer.write_u64(facts.size());
    for (const auto& fact : facts) write_fact(writer, fact);
}

[[nodiscard]] std::vector<agent::Fact> read_fact_vector(
    detail::BufferReader& reader,
    const Rlf6CheckpointLoadOptions& options
) {
    const std::size_t count = detail::checked_size(
        reader.read_u64(), options.maximum_nested_records,
        "RLF-6 fact count"
    );
    std::vector<agent::Fact> result;
    result.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        result.push_back(read_fact(reader, options));
    }
    return result;
}

void write_u64_vector(
    detail::BufferWriter& writer,
    const std::vector<std::uint64_t>& values
) {
    writer.write_u64(values.size());
    for (const auto value : values) writer.write_u64(value);
}

[[nodiscard]] std::vector<std::uint64_t> read_u64_vector(
    detail::BufferReader& reader,
    const Rlf6CheckpointLoadOptions& options,
    const char* label
) {
    const std::size_t count = detail::checked_size(
        reader.read_u64(), options.maximum_route_length, label
    );
    std::vector<std::uint64_t> result;
    result.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        result.push_back(reader.read_u64());
    }
    return result;
}

void write_size_vector(
    detail::BufferWriter& writer,
    const std::vector<std::size_t>& values
) {
    writer.write_u64(values.size());
    for (const auto value : values) writer.write_u64(value);
}

[[nodiscard]] std::vector<std::size_t> read_size_vector(
    detail::BufferReader& reader,
    const Rlf6CheckpointLoadOptions& options,
    const char* label
) {
    const std::size_t count = detail::checked_size(
        reader.read_u64(), options.maximum_route_length, label
    );
    std::vector<std::size_t> result;
    result.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        result.push_back(detail::checked_size(
            reader.read_u64(), options.maximum_route_length, label
        ));
    }
    return result;
}

void write_config(
    detail::BufferWriter& writer,
    const agent::AgentConfig& config
) {
    writer.write_u64(config.maximum_observations);
    writer.write_u64(config.maximum_beliefs);
    writer.write_u64(config.maximum_goals);
    writer.write_u64(config.maximum_memory_records);
    writer.write_u64(config.maximum_skills);
    writer.write_u64(config.maximum_errors);
    writer.write_u64(config.maximum_transition_records);
    writer.write_u64(config.maximum_tool_reliability_records);
    writer.write_u64(config.maximum_plan_depth);
    writer.write_u64(config.maximum_candidate_actions);
    writer.write_double(config.belief_acceptance_threshold);
    writer.write_double(config.irreversible_confidence_threshold);
    writer.write_double(config.verification_threshold);
    writer.write_double(config.skill_mdl_minimum_gain);
    writer.write_double(config.recent_decay);
    writer.write_double(config.change_detection_threshold);
}

[[nodiscard]] agent::AgentConfig read_config(
    detail::BufferReader& reader,
    const Rlf6CheckpointLoadOptions& options
) {
    agent::AgentConfig config;
    config.maximum_observations = detail::checked_size(
        reader.read_u64(), options.maximum_records, "RLF-6 maximum observations"
    );
    config.maximum_beliefs = detail::checked_size(
        reader.read_u64(), options.maximum_records, "RLF-6 maximum beliefs"
    );
    config.maximum_goals = detail::checked_size(
        reader.read_u64(), options.maximum_records, "RLF-6 maximum goals"
    );
    config.maximum_memory_records = detail::checked_size(
        reader.read_u64(), options.maximum_records, "RLF-6 maximum memory"
    );
    config.maximum_skills = detail::checked_size(
        reader.read_u64(), options.maximum_records, "RLF-6 maximum skills"
    );
    config.maximum_errors = detail::checked_size(
        reader.read_u64(), options.maximum_records, "RLF-6 maximum errors"
    );
    config.maximum_transition_records = detail::checked_size(
        reader.read_u64(), options.maximum_records, "RLF-6 maximum transitions"
    );
    config.maximum_tool_reliability_records = detail::checked_size(
        reader.read_u64(), options.maximum_records,
        "RLF-6 maximum tool reliability"
    );
    config.maximum_plan_depth = detail::checked_size(
        reader.read_u64(), options.maximum_route_length,
        "RLF-6 maximum plan depth"
    );
    config.maximum_candidate_actions = detail::checked_size(
        reader.read_u64(), options.maximum_records,
        "RLF-6 maximum candidate actions"
    );
    config.belief_acceptance_threshold = reader.read_double();
    config.irreversible_confidence_threshold = reader.read_double();
    config.verification_threshold = reader.read_double();
    config.skill_mdl_minimum_gain = reader.read_double();
    config.recent_decay = reader.read_double();
    config.change_detection_threshold = reader.read_double();
    validate_probability(config.belief_acceptance_threshold, "belief threshold");
    validate_probability(
        config.irreversible_confidence_threshold, "irreversible threshold"
    );
    validate_probability(config.verification_threshold, "verification threshold");
    validate_nonnegative(config.skill_mdl_minimum_gain, "skill MDL gain");
    validate_probability(config.recent_decay, "recent decay");
    validate_probability(config.change_detection_threshold, "change threshold");
    return config;
}

void write_evidence(
    detail::BufferWriter& writer,
    const agent::EvidenceRecord& record
) {
    writer.write_u64(record.stable_id);
    write_fact(writer, record.fact);
    write_enum(writer, record.kind);
    writer.write_double(record.confidence);
    writer.write_double(record.source_reliability);
    writer.write_u64(record.creation_step);
    writer.write_u64(record.last_update_step);
    writer.write_string(record.provenance);
    writer.write_bool(record.verified);
    writer.write_bool(record.stale);
}

[[nodiscard]] agent::EvidenceRecord read_evidence(
    detail::BufferReader& reader,
    const Rlf6CheckpointLoadOptions& options
) {
    agent::EvidenceRecord record;
    record.stable_id = reader.read_u64();
    record.fact = read_fact(reader, options);
    record.kind = read_enum<agent::EvidenceKind>(
        reader, static_cast<std::uint8_t>(agent::EvidenceKind::stale_information),
        "evidence kind"
    );
    record.confidence = reader.read_double();
    record.source_reliability = reader.read_double();
    validate_probability(record.confidence, "evidence confidence");
    validate_probability(record.source_reliability, "source reliability");
    record.creation_step = reader.read_u64();
    record.last_update_step = reader.read_u64();
    record.provenance = reader.read_string(options.maximum_string_bytes);
    record.verified = reader.read_bool();
    record.stale = reader.read_bool();
    return record;
}

void write_belief(
    detail::BufferWriter& writer,
    const agent::BeliefHypothesis& record
) {
    writer.write_u64(record.stable_id);
    write_fact(writer, record.fact);
    writer.write_double(record.support);
    writer.write_u64(record.contradiction_count);
    writer.write_u64(record.recency);
    writer.write_double(record.source_reliability);
    writer.write_double(record.uncertainty);
    writer.write_bool(record.verified);
    writer.write_bool(record.stale);
}

[[nodiscard]] agent::BeliefHypothesis read_belief(
    detail::BufferReader& reader,
    const Rlf6CheckpointLoadOptions& options
) {
    agent::BeliefHypothesis record;
    record.stable_id = reader.read_u64();
    record.fact = read_fact(reader, options);
    record.support = reader.read_double();
    record.contradiction_count = reader.read_u64();
    record.recency = reader.read_u64();
    record.source_reliability = reader.read_double();
    record.uncertainty = reader.read_double();
    validate_probability(record.support, "belief support");
    validate_probability(record.source_reliability, "belief reliability");
    validate_probability(record.uncertainty, "belief uncertainty");
    record.verified = reader.read_bool();
    record.stale = reader.read_bool();
    return record;
}

void write_goal(detail::BufferWriter& writer, const agent::Goal& goal) {
    writer.write_u64(goal.stable_id);
    writer.write_string(goal.specification);
    write_fact_vector(writer, goal.completion_conditions);
    write_fact_vector(writer, goal.failure_conditions);
    write_u64_vector(writer, goal.dependencies);
    writer.write_double(goal.priority);
    writer.write_u64(goal.creation_step);
    writer.write_u64(goal.deadline_step);
    writer.write_u64(goal.resource_budget);
    write_enum(writer, goal.status);
    writer.write_double(goal.confidence);
    writer.write_string(goal.provenance);
    writer.write_bool(goal.optional);
    writer.write_bool(goal.ordered);
}

[[nodiscard]] agent::Goal read_goal(
    detail::BufferReader& reader,
    const Rlf6CheckpointLoadOptions& options
) {
    agent::Goal goal;
    goal.stable_id = reader.read_u64();
    goal.specification = reader.read_string(options.maximum_string_bytes);
    goal.completion_conditions = read_fact_vector(reader, options);
    goal.failure_conditions = read_fact_vector(reader, options);
    goal.dependencies = read_u64_vector(
        reader, options, "RLF-6 goal dependencies"
    );
    goal.priority = reader.read_double();
    validate_nonnegative(goal.priority, "goal priority");
    goal.creation_step = reader.read_u64();
    goal.deadline_step = reader.read_u64();
    goal.resource_budget = reader.read_u64();
    goal.status = read_enum<agent::GoalStatus>(
        reader, static_cast<std::uint8_t>(agent::GoalStatus::abandoned),
        "goal status"
    );
    goal.confidence = reader.read_double();
    validate_probability(goal.confidence, "goal confidence");
    goal.provenance = reader.read_string(options.maximum_string_bytes);
    goal.optional = reader.read_bool();
    goal.ordered = reader.read_bool();
    if (goal.completion_conditions.empty()) {
        throw std::runtime_error("RLF-6 goal has no completion conditions");
    }
    return goal;
}

void write_memory(
    detail::BufferWriter& writer,
    const agent::MemoryRecord& record
) {
    writer.write_u64(record.stable_id);
    write_enum(writer, record.memory_class);
    writer.write_string(record.key);
    writer.write_string(record.payload);
    writer.write_double(record.confidence);
    writer.write_double(record.utility);
    writer.write_string(record.provenance);
    writer.write_u64(record.creation_step);
    writer.write_u64(record.last_use_step);
    writer.write_bool(record.verified);
    writer.write_bool(record.invalidated);
}

[[nodiscard]] agent::MemoryRecord read_memory(
    detail::BufferReader& reader,
    const Rlf6CheckpointLoadOptions& options
) {
    agent::MemoryRecord record;
    record.stable_id = reader.read_u64();
    record.memory_class = read_enum<agent::MemoryClass>(
        reader, static_cast<std::uint8_t>(agent::MemoryClass::safety),
        "memory class"
    );
    record.key = reader.read_string(options.maximum_string_bytes);
    record.payload = reader.read_string(options.maximum_string_bytes);
    record.confidence = reader.read_double();
    record.utility = reader.read_double();
    validate_probability(record.confidence, "memory confidence");
    if (!std::isfinite(record.utility)) {
        throw std::runtime_error("invalid RLF-6 memory utility");
    }
    record.provenance = reader.read_string(options.maximum_string_bytes);
    record.creation_step = reader.read_u64();
    record.last_use_step = reader.read_u64();
    record.verified = reader.read_bool();
    record.invalidated = reader.read_bool();
    if (record.memory_class == agent::MemoryClass::skill) {
        throw std::runtime_error("skill encoded in ordinary memory vector");
    }
    return record;
}

void write_tool(
    detail::BufferWriter& writer,
    const agent::ToolDefinition& tool
) {
    writer.write_u64(tool.stable_id);
    writer.write_string(tool.name);
    writer.write_u64(tool.input_schema.size());
    for (const auto& field : tool.input_schema) writer.write_string(field);
    writer.write_u64(tool.output_schema.size());
    for (const auto& field : tool.output_schema) writer.write_string(field);
    write_fact_vector(writer, tool.preconditions);
    writer.write_bool(tool.has_side_effects);
    writer.write_bool(tool.reversible);
    writer.write_double(tool.cost);
    writer.write_u64(tool.latency);
    writer.write_u64(tool.failure_modes.size());
    for (const auto mode : tool.failure_modes) write_enum(writer, mode);
    writer.write_double(tool.declared_reliability);
    write_enum(writer, tool.safety_class);
    writer.write_string(tool.required_permission);
}

[[nodiscard]] agent::ToolDefinition read_tool(
    detail::BufferReader& reader,
    const Rlf6CheckpointLoadOptions& options
) {
    agent::ToolDefinition tool;
    tool.stable_id = reader.read_u64();
    tool.name = reader.read_string(options.maximum_string_bytes);
    const std::size_t input_count = detail::checked_size(
        reader.read_u64(), options.maximum_nested_records,
        "RLF-6 tool input schema"
    );
    tool.input_schema.reserve(input_count);
    for (std::size_t index = 0U; index < input_count; ++index) {
        tool.input_schema.push_back(reader.read_string(options.maximum_string_bytes));
    }
    const std::size_t output_count = detail::checked_size(
        reader.read_u64(), options.maximum_nested_records,
        "RLF-6 tool output schema"
    );
    tool.output_schema.reserve(output_count);
    for (std::size_t index = 0U; index < output_count; ++index) {
        tool.output_schema.push_back(reader.read_string(options.maximum_string_bytes));
    }
    tool.preconditions = read_fact_vector(reader, options);
    tool.has_side_effects = reader.read_bool();
    tool.reversible = reader.read_bool();
    tool.cost = reader.read_double();
    validate_nonnegative(tool.cost, "tool cost");
    tool.latency = reader.read_u64();
    const std::size_t failure_count = detail::checked_size(
        reader.read_u64(), options.maximum_nested_records,
        "RLF-6 tool failure modes"
    );
    tool.failure_modes.reserve(failure_count);
    for (std::size_t index = 0U; index < failure_count; ++index) {
        tool.failure_modes.push_back(read_enum<agent::ToolFailure>(
            reader, static_cast<std::uint8_t>(agent::ToolFailure::internal_error),
            "tool failure mode"
        ));
    }
    tool.declared_reliability = reader.read_double();
    validate_probability(tool.declared_reliability, "tool reliability");
    tool.safety_class = read_enum<agent::SafetyClass>(
        reader, static_cast<std::uint8_t>(agent::SafetyClass::prohibited),
        "tool safety class"
    );
    tool.required_permission = reader.read_string(options.maximum_string_bytes);
    if (tool.name.empty()) {
        throw std::runtime_error("empty RLF-6 tool name");
    }
    return tool;
}

void write_transition(
    detail::BufferWriter& writer,
    const agent::TransitionRecord& record
) {
    writer.write_string(record.context_key);
    writer.write_string(record.action_signature);
    writer.write_u64(record.outcomes.size());
    for (const auto& outcome : record.outcomes) {
        write_fact_vector(writer, outcome.effects);
        writer.write_double(outcome.count);
        writer.write_double(outcome.recent_count);
        writer.write_double(outcome.total_cost);
        writer.write_u64(outcome.terminal_failures);
    }
    writer.write_u64(record.observations);
    writer.write_u64(record.model_version);
    writer.write_double(record.surprise_ema);
    writer.write_double(record.reliability);
}

[[nodiscard]] agent::TransitionRecord read_transition(
    detail::BufferReader& reader,
    const Rlf6CheckpointLoadOptions& options
) {
    agent::TransitionRecord record;
    record.context_key = reader.read_string(options.maximum_string_bytes);
    record.action_signature = reader.read_string(options.maximum_string_bytes);
    const std::size_t outcome_count = detail::checked_size(
        reader.read_u64(), options.maximum_nested_records,
        "RLF-6 transition outcomes"
    );
    record.outcomes.reserve(outcome_count);
    for (std::size_t index = 0U; index < outcome_count; ++index) {
        agent::TransitionOutcome outcome;
        outcome.effects = read_fact_vector(reader, options);
        outcome.count = reader.read_double();
        outcome.recent_count = reader.read_double();
        outcome.total_cost = reader.read_double();
        outcome.terminal_failures = reader.read_u64();
        validate_nonnegative(outcome.count, "transition count");
        validate_nonnegative(outcome.recent_count, "recent transition count");
        validate_nonnegative(outcome.total_cost, "transition cost");
        record.outcomes.push_back(std::move(outcome));
    }
    record.observations = reader.read_u64();
    record.model_version = reader.read_u64();
    record.surprise_ema = reader.read_double();
    record.reliability = reader.read_double();
    validate_probability(record.surprise_ema, "transition surprise");
    validate_probability(record.reliability, "transition reliability");
    if (record.context_key.empty() || record.action_signature.empty()) {
        throw std::runtime_error("invalid empty RLF-6 transition key");
    }
    return record;
}

void write_tool_reliability(
    detail::BufferWriter& writer,
    const agent::ToolReliability& reliability
) {
    writer.write_u64(reliability.tool_id);
    writer.write_string(reliability.context);
    writer.write_double(reliability.successes);
    writer.write_double(reliability.failures);
    writer.write_double(reliability.recent_successes);
    writer.write_double(reliability.recent_failures);
    writer.write_u64(reliability.last_update_step);
}

[[nodiscard]] agent::ToolReliability read_tool_reliability(
    detail::BufferReader& reader,
    const Rlf6CheckpointLoadOptions& options
) {
    agent::ToolReliability reliability;
    reliability.tool_id = reader.read_u64();
    reliability.context = reader.read_string(options.maximum_string_bytes);
    reliability.successes = reader.read_double();
    reliability.failures = reader.read_double();
    reliability.recent_successes = reader.read_double();
    reliability.recent_failures = reader.read_double();
    reliability.last_update_step = reader.read_u64();
    validate_nonnegative(reliability.successes, "tool successes");
    validate_nonnegative(reliability.failures, "tool failures");
    validate_nonnegative(reliability.recent_successes, "recent tool successes");
    validate_nonnegative(reliability.recent_failures, "recent tool failures");
    return reliability;
}

void write_skill(detail::BufferWriter& writer, const agent::Skill& skill) {
    writer.write_u64(skill.stable_id);
    writer.write_string(skill.goal_pattern);
    write_fact_vector(writer, skill.triggering_conditions);
    write_u64_vector(writer, skill.action_sequence);
    write_size_vector(writer, skill.verification_points);
    write_fact_vector(writer, skill.failure_conditions);
    write_u64_vector(writer, skill.fallback_actions);
    writer.write_double(skill.estimated_cost);
    writer.write_double(skill.confidence);
    writer.write_double(skill.utility);
    writer.write_u64(skill.support);
    writer.write_u64(skill.successful_reuses);
    writer.write_u64(skill.failed_reuses);
    writer.write_bool(skill.invalidated);
}

[[nodiscard]] agent::Skill read_skill(
    detail::BufferReader& reader,
    const Rlf6CheckpointLoadOptions& options
) {
    agent::Skill skill;
    skill.stable_id = reader.read_u64();
    skill.goal_pattern = reader.read_string(options.maximum_string_bytes);
    skill.triggering_conditions = read_fact_vector(reader, options);
    skill.action_sequence = read_u64_vector(
        reader, options, "RLF-6 skill action route"
    );
    skill.verification_points = read_size_vector(
        reader, options, "RLF-6 skill verification points"
    );
    skill.failure_conditions = read_fact_vector(reader, options);
    skill.fallback_actions = read_u64_vector(
        reader, options, "RLF-6 skill fallback route"
    );
    skill.estimated_cost = reader.read_double();
    skill.confidence = reader.read_double();
    skill.utility = reader.read_double();
    validate_nonnegative(skill.estimated_cost, "skill cost");
    validate_probability(skill.confidence, "skill confidence");
    if (!std::isfinite(skill.utility)) {
        throw std::runtime_error("invalid RLF-6 skill utility");
    }
    skill.support = reader.read_u64();
    skill.successful_reuses = reader.read_u64();
    skill.failed_reuses = reader.read_u64();
    skill.invalidated = reader.read_bool();
    if (skill.goal_pattern.empty() || skill.action_sequence.empty()) {
        throw std::runtime_error("invalid RLF-6 skill route");
    }
    for (const std::size_t point : skill.verification_points) {
        if (point >= skill.action_sequence.size()) {
            throw std::runtime_error("invalid RLF-6 skill verification point");
        }
    }
    return skill;
}

void write_error(detail::BufferWriter& writer, const agent::ErrorEvent& error) {
    writer.write_u64(error.stable_id);
    write_enum(writer, error.type);
    writer.write_u64(error.action_id);
    writer.write_u64(error.step);
    writer.write_string(error.context);
    writer.write_string(error.expected);
    writer.write_string(error.actual);
    writer.write_double(error.severity);
    writer.write_bool(error.recovered);
}

[[nodiscard]] agent::ErrorEvent read_error(
    detail::BufferReader& reader,
    const Rlf6CheckpointLoadOptions& options
) {
    agent::ErrorEvent error;
    error.stable_id = reader.read_u64();
    error.type = read_enum<agent::ErrorType>(
        reader, static_cast<std::uint8_t>(agent::ErrorType::false_completion),
        "error type"
    );
    error.action_id = reader.read_u64();
    error.step = reader.read_u64();
    error.context = reader.read_string(options.maximum_string_bytes);
    error.expected = reader.read_string(options.maximum_string_bytes);
    error.actual = reader.read_string(options.maximum_string_bytes);
    error.severity = reader.read_double();
    validate_nonnegative(error.severity, "error severity");
    error.recovered = reader.read_bool();
    return error;
}

void write_resources(
    detail::BufferWriter& writer,
    const agent::ResourceState& resources
) {
    writer.write_u64(resources.reasoning_cycles);
    writer.write_u64(resources.action_count);
    writer.write_u64(resources.tool_calls);
    writer.write_double(resources.tool_cost);
    writer.write_u64(resources.memory_reads);
    writer.write_u64(resources.memory_writes);
    writer.write_u64(resources.planning_nodes);
    writer.write_u64(resources.simulated_time);
    writer.write_double(resources.risk_used);
    writer.write_u64(resources.bytes_read);
    writer.write_u64(resources.bytes_written);
}

[[nodiscard]] agent::ResourceState read_resources(detail::BufferReader& reader) {
    agent::ResourceState resources;
    resources.reasoning_cycles = reader.read_u64();
    resources.action_count = reader.read_u64();
    resources.tool_calls = reader.read_u64();
    resources.tool_cost = reader.read_double();
    validate_nonnegative(resources.tool_cost, "resource tool cost");
    resources.memory_reads = reader.read_u64();
    resources.memory_writes = reader.read_u64();
    resources.planning_nodes = reader.read_u64();
    resources.simulated_time = reader.read_u64();
    resources.risk_used = reader.read_double();
    validate_nonnegative(resources.risk_used, "resource risk");
    resources.bytes_read = reader.read_u64();
    resources.bytes_written = reader.read_u64();
    return resources;
}

void write_uncertainty(
    detail::BufferWriter& writer,
    const agent::UncertaintyState& uncertainty
) {
    writer.write_double(uncertainty.belief_uncertainty);
    writer.write_double(uncertainty.prediction_uncertainty);
    writer.write_double(uncertainty.action_uncertainty);
    writer.write_double(uncertainty.tool_uncertainty);
    writer.write_double(uncertainty.plan_uncertainty);
    writer.write_double(uncertainty.goal_completion_uncertainty);
    writer.write_double(uncertainty.memory_uncertainty);
    writer.write_double(uncertainty.language_uncertainty);
    writer.write_double(uncertainty.safety_uncertainty);
}

[[nodiscard]] agent::UncertaintyState read_uncertainty(
    detail::BufferReader& reader
) {
    agent::UncertaintyState uncertainty{
        reader.read_double(), reader.read_double(), reader.read_double(),
        reader.read_double(), reader.read_double(), reader.read_double(),
        reader.read_double(), reader.read_double(), reader.read_double(),
    };
    const std::array<double, 9U> values{
        uncertainty.belief_uncertainty, uncertainty.prediction_uncertainty,
        uncertainty.action_uncertainty, uncertainty.tool_uncertainty,
        uncertainty.plan_uncertainty, uncertainty.goal_completion_uncertainty,
        uncertainty.memory_uncertainty, uncertainty.language_uncertainty,
        uncertainty.safety_uncertainty,
    };
    for (const double value : values) validate_probability(value, "uncertainty");
    return uncertainty;
}

template <typename Record, typename Writer>
void write_records(
    detail::BufferWriter& writer,
    const std::vector<Record>& records,
    Writer write_record
) {
    writer.write_u64(records.size());
    for (const auto& record : records) write_record(writer, record);
}

template <typename Record, typename Reader>
[[nodiscard]] std::vector<Record> read_records(
    detail::BufferReader& reader,
    const Rlf6CheckpointLoadOptions& options,
    const char* label,
    Reader read_record
) {
    const std::size_t count = detail::checked_size(
        reader.read_u64(), options.maximum_records, label
    );
    std::vector<Record> records;
    records.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        records.push_back(read_record(reader, options));
    }
    return records;
}

[[nodiscard]] std::vector<std::uint8_t> encode_payload(
    const agent::AgentFabric& fabric
) {
    const agent::AgentSnapshot snapshot = fabric.snapshot();
    const auto& state = snapshot.state;
    detail::BufferWriter writer;
    write_config(writer, snapshot.config);
    writer.write_u64(snapshot.seed);
    writer.write_u64(snapshot.next_id);
    writer.write_u64(state.active_goal);
    writer.write_u64(state.step_index);
    writer.write_u64(state.episode_id);
    writer.write_double(state.progress_state);
    write_records(writer, state.observation_state, write_evidence);
    write_records(writer, state.belief_state, write_belief);
    write_records(writer, state.goal_stack, write_goal);
    writer.write_u64(state.subgoal_graph.size());
    for (const auto& [parent, child] : state.subgoal_graph) {
        writer.write_u64(parent);
        writer.write_u64(child);
    }
    write_records(writer, state.working_memory, write_memory);
    write_records(writer, state.episodic_memory, write_memory);
    write_records(writer, state.semantic_memory, write_memory);
    write_records(writer, state.safety_memory, write_memory);
    write_records(writer, state.skill_memory, write_skill);
    write_records(writer, state.tool_state, write_tool);
    write_records(writer, state.world_model_state, write_transition);
    write_records(writer, state.tool_reliability, write_tool_reliability);
    write_uncertainty(writer, state.uncertainty_state);
    write_resources(writer, state.resource_state);
    write_u64_vector(writer, state.action_history_summary);
    write_records(writer, state.failure_history_summary, write_error);
    write_fact_vector(writer, state.verified_facts);
    writer.write_u64(fabric.deterministic_hash());
    return writer.take();
}

[[nodiscard]] agent::AgentSnapshot decode_payload(
    const std::span<const std::uint8_t> payload,
    const Rlf6CheckpointLoadOptions& options,
    std::uint64_t* const stored_hash
) {
    detail::BufferReader reader(payload);
    agent::AgentSnapshot snapshot;
    snapshot.config = read_config(reader, options);
    snapshot.seed = reader.read_u64();
    snapshot.next_id = reader.read_u64();
    snapshot.state.active_goal = reader.read_u64();
    snapshot.state.step_index = reader.read_u64();
    snapshot.state.episode_id = reader.read_u64();
    snapshot.state.progress_state = reader.read_double();
    validate_probability(snapshot.state.progress_state, "progress state");
    snapshot.state.observation_state = read_records<agent::EvidenceRecord>(
        reader, options, "RLF-6 observations", read_evidence
    );
    snapshot.state.belief_state = read_records<agent::BeliefHypothesis>(
        reader, options, "RLF-6 beliefs", read_belief
    );
    snapshot.state.goal_stack = read_records<agent::Goal>(
        reader, options, "RLF-6 goals", read_goal
    );
    const std::size_t edge_count = detail::checked_size(
        reader.read_u64(), options.maximum_nested_records,
        "RLF-6 subgoal edges"
    );
    snapshot.state.subgoal_graph.reserve(edge_count);
    for (std::size_t index = 0U; index < edge_count; ++index) {
        snapshot.state.subgoal_graph.emplace_back(
            reader.read_u64(), reader.read_u64()
        );
    }
    snapshot.state.working_memory = read_records<agent::MemoryRecord>(
        reader, options, "RLF-6 working memory", read_memory
    );
    snapshot.state.episodic_memory = read_records<agent::MemoryRecord>(
        reader, options, "RLF-6 episodic memory", read_memory
    );
    snapshot.state.semantic_memory = read_records<agent::MemoryRecord>(
        reader, options, "RLF-6 semantic memory", read_memory
    );
    snapshot.state.safety_memory = read_records<agent::MemoryRecord>(
        reader, options, "RLF-6 safety memory", read_memory
    );
    snapshot.state.skill_memory = read_records<agent::Skill>(
        reader, options, "RLF-6 skills", read_skill
    );
    snapshot.state.tool_state = read_records<agent::ToolDefinition>(
        reader, options, "RLF-6 tools", read_tool
    );
    snapshot.state.world_model_state = read_records<agent::TransitionRecord>(
        reader, options, "RLF-6 transitions", read_transition
    );
    snapshot.state.tool_reliability = read_records<agent::ToolReliability>(
        reader, options, "RLF-6 tool reliability", read_tool_reliability
    );
    snapshot.state.uncertainty_state = read_uncertainty(reader);
    snapshot.state.resource_state = read_resources(reader);
    snapshot.state.action_history_summary = read_u64_vector(
        reader, options, "RLF-6 action history"
    );
    snapshot.state.failure_history_summary = read_records<agent::ErrorEvent>(
        reader, options, "RLF-6 errors", read_error
    );
    snapshot.state.verified_facts = read_fact_vector(reader, options);
    *stored_hash = reader.read_u64();
    if (!reader.empty()) {
        throw std::runtime_error("unexpected trailing RLF-6 checkpoint payload");
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
    const Rlf6CheckpointLoadOptions& options
) {
    const auto bytes = detail::read_file(path, options.maximum_file_bytes);
    if (bytes.size() < header_size) {
        throw std::runtime_error("truncated RLF-6 checkpoint header");
    }
    detail::BufferReader reader(bytes);
    for (const std::uint8_t expected : magic) {
        if (reader.read_u8() != expected) {
            throw std::runtime_error("invalid RLF-6 checkpoint magic");
        }
    }
    const std::uint32_t version = reader.read_u32();
    if (version != rlf6_checkpoint_format_version) {
        throw std::runtime_error("unsupported RLF-6 checkpoint version");
    }
    const std::size_t payload_size = detail::checked_size(
        reader.read_u64(), options.maximum_file_bytes, "RLF-6 payload size"
    );
    const std::uint64_t stored_checksum = reader.read_u64();
    if (payload_size != reader.remaining()) {
        throw std::runtime_error("RLF-6 checkpoint payload length mismatch");
    }
    auto payload = reader.read_bytes(payload_size);
    const std::uint64_t actual_checksum = detail::checksum(payload);
    if (actual_checksum != stored_checksum) {
        throw std::runtime_error("RLF-6 checkpoint checksum mismatch");
    }
    return {std::move(payload), stored_checksum, bytes.size()};
}

}  // namespace

void save_rlf6_checkpoint(
    const std::filesystem::path& path,
    const agent::AgentFabric& fabric
) {
    const auto payload = encode_payload(fabric);
    detail::BufferWriter writer;
    for (const std::uint8_t byte : magic) writer.write_u8(byte);
    writer.write_u32(rlf6_checkpoint_format_version);
    writer.write_u64(payload.size());
    writer.write_u64(detail::checksum(payload));
    writer.write_bytes(payload);
    detail::write_file_transactionally(path, writer.bytes());
}

agent::AgentFabric load_rlf6_checkpoint(
    const std::filesystem::path& path,
    const Rlf6CheckpointLoadOptions& options
) {
    const auto decoded = decode_file(path, options);
    std::uint64_t stored_hash = 0U;
    const auto snapshot = decode_payload(decoded.payload, options, &stored_hash);
    auto fabric = agent::AgentFabric::from_snapshot(snapshot);
    if (fabric.deterministic_hash() != stored_hash) {
        throw std::runtime_error("RLF-6 checkpoint deterministic hash mismatch");
    }
    return fabric;
}

Rlf6CheckpointSummary inspect_rlf6_checkpoint(
    const std::filesystem::path& path,
    const Rlf6CheckpointLoadOptions& options
) {
    const auto decoded = decode_file(path, options);
    std::uint64_t stored_hash = 0U;
    const auto snapshot = decode_payload(decoded.payload, options, &stored_hash);
    auto fabric = agent::AgentFabric::from_snapshot(snapshot);
    if (fabric.deterministic_hash() != stored_hash) {
        throw std::runtime_error("RLF-6 checkpoint deterministic hash mismatch");
    }
    const auto& state = fabric.state();
    return {
        rlf6_checkpoint_format_version,
        fabric.seed(),
        state.episode_id,
        state.step_index,
        state.observation_state.size(),
        state.belief_state.size(),
        state.goal_stack.size(),
        state.tool_state.size(),
        state.world_model_state.size(),
        state.working_memory.size() + state.episodic_memory.size() +
            state.semantic_memory.size() + state.safety_memory.size(),
        state.skill_memory.size(),
        state.failure_history_summary.size(),
        stored_hash,
        decoded.checksum,
        decoded.file_bytes,
    };
}

}  // namespace rlf::storage
