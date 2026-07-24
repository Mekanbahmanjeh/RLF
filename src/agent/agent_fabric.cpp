#include "rlf/agent/agent_fabric.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace rlf::agent {
namespace {

constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;
constexpr double probability_floor = 1.0e-9;

void hash_byte(std::uint64_t& hash, const std::uint8_t value) noexcept {
    hash ^= value;
    hash *= fnv_prime;
}

void hash_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (unsigned int index = 0U; index < 8U; ++index) {
        hash_byte(hash, static_cast<std::uint8_t>(
            (value >> (index * 8U)) & 0xFFULL
        ));
    }
}

void hash_double(std::uint64_t& hash, const double value) noexcept {
    hash_u64(hash, std::bit_cast<std::uint64_t>(value));
}

void hash_string(std::uint64_t& hash, const std::string_view value) noexcept {
    hash_u64(hash, value.size());
    for (const char character : value) {
        hash_byte(hash, static_cast<std::uint8_t>(character));
    }
}

[[nodiscard]] double clamp_probability(const double value) noexcept {
    return std::clamp(value, probability_floor, 1.0 - probability_floor);
}

[[nodiscard]] bool finite_probability(const double value) noexcept {
    return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

[[nodiscard]] std::string key_prefix(const Fact& fact) {
    return fact.key + "=";
}

[[nodiscard]] std::vector<MemoryRecord>* memory_vector(
    AgentState& state,
    const MemoryClass memory_class
) {
    switch (memory_class) {
        case MemoryClass::working: return &state.working_memory;
        case MemoryClass::episodic: return &state.episodic_memory;
        case MemoryClass::semantic: return &state.semantic_memory;
        case MemoryClass::safety: return &state.safety_memory;
        case MemoryClass::skill: return nullptr;
    }
    return nullptr;
}

template <typename Record>
void erase_oldest(std::vector<Record>& records, const std::size_t maximum) {
    if (records.size() <= maximum) {
        return;
    }
    std::stable_sort(
        records.begin(), records.end(),
        [](const Record& left, const Record& right) {
            if (left.last_update_step != right.last_update_step) {
                return left.last_update_step > right.last_update_step;
            }
            return left.stable_id < right.stable_id;
        }
    );
    records.resize(maximum);
}

void erase_oldest_memory(
    std::vector<MemoryRecord>& records,
    const std::size_t maximum
) {
    if (records.size() <= maximum) {
        return;
    }
    std::stable_sort(
        records.begin(), records.end(),
        [](const MemoryRecord& left, const MemoryRecord& right) {
            if (left.invalidated != right.invalidated) {
                return !left.invalidated;
            }
            if (left.utility != right.utility) {
                return left.utility > right.utility;
            }
            if (left.last_use_step != right.last_use_step) {
                return left.last_use_step > right.last_use_step;
            }
            return left.stable_id < right.stable_id;
        }
    );
    records.resize(maximum);
}

[[nodiscard]] std::string goal_pattern_from_goal(const Goal& goal) {
    std::string result;
    for (const auto& fact : goal.completion_conditions) {
        if (!result.empty()) {
            result.push_back('&');
        }
        result += fact.canonical();
    }
    return result;
}

}  // namespace

std::string Fact::canonical() const {
    return (negated ? "!" : "") + key + "=" + value;
}

bool Fact::operator<(const Fact& other) const {
    return canonical() < other.canonical();
}

double ToolReliability::mean() const noexcept {
    const double recent_total = recent_successes + recent_failures;
    const double long_total = successes + failures;
    const double recent = recent_total > 0.0
        ? recent_successes / recent_total
        : 0.5;
    const double long_term = long_total > 0.0 ? successes / long_total : 0.5;
    return std::clamp(0.7 * recent + 0.3 * long_term, 0.0, 1.0);
}

double ToolReliability::uncertainty() const noexcept {
    const double samples = successes + failures;
    return std::clamp(1.0 / std::sqrt(std::max(1.0, samples)), 0.0, 1.0);
}

AgentFabric::AgentFabric(AgentConfig config, const std::uint64_t seed)
    : config_(std::move(config)), seed_(seed) {
    if (config_.maximum_observations == 0U ||
        config_.maximum_beliefs == 0U || config_.maximum_goals == 0U ||
        config_.maximum_memory_records == 0U || config_.maximum_skills == 0U ||
        config_.maximum_errors == 0U ||
        config_.maximum_transition_records == 0U ||
        config_.maximum_tool_reliability_records == 0U ||
        config_.maximum_plan_depth == 0U ||
        config_.maximum_candidate_actions == 0U ||
        !finite_probability(config_.belief_acceptance_threshold) ||
        !finite_probability(config_.irreversible_confidence_threshold) ||
        !finite_probability(config_.verification_threshold) ||
        !std::isfinite(config_.skill_mdl_minimum_gain) ||
        config_.skill_mdl_minimum_gain < 0.0 ||
        !finite_probability(config_.recent_decay) ||
        !finite_probability(config_.change_detection_threshold)) {
        throw std::invalid_argument("invalid RLF-6 agent configuration");
    }
}

const AgentConfig& AgentFabric::config() const noexcept { return config_; }
std::uint64_t AgentFabric::seed() const noexcept { return seed_; }
const AgentState& AgentFabric::state() const noexcept { return state_; }
AgentState& AgentFabric::mutable_state() noexcept { return state_; }

std::uint64_t AgentFabric::allocate_id() {
    if (next_id_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("RLF-6 stable ID space exhausted");
    }
    return next_id_++;
}

Goal* AgentFabric::find_goal_mutable(const std::uint64_t goal_id) {
    const auto found = std::find_if(
        state_.goal_stack.begin(), state_.goal_stack.end(),
        [goal_id](const Goal& goal) { return goal.stable_id == goal_id; }
    );
    return found == state_.goal_stack.end() ? nullptr : &*found;
}

const Goal* AgentFabric::find_goal(const std::uint64_t goal_id) const {
    const auto found = std::find_if(
        state_.goal_stack.begin(), state_.goal_stack.end(),
        [goal_id](const Goal& goal) { return goal.stable_id == goal_id; }
    );
    return found == state_.goal_stack.end() ? nullptr : &*found;
}

std::uint64_t AgentFabric::add_goal(Goal goal) {
    if (state_.goal_stack.size() >= config_.maximum_goals) {
        throw std::runtime_error("RLF-6 goal capacity exceeded");
    }
    if (goal.stable_id == 0U) {
        goal.stable_id = allocate_id();
    } else {
        next_id_ = std::max(next_id_, goal.stable_id + 1U);
    }
    if (goal.completion_conditions.empty()) {
        throw std::invalid_argument("RLF-6 goal requires completion conditions");
    }
    if (!finite_probability(goal.confidence) || !std::isfinite(goal.priority)) {
        throw std::invalid_argument("invalid RLF-6 goal confidence or priority");
    }
    const bool duplicate = std::any_of(
        state_.goal_stack.begin(), state_.goal_stack.end(),
        [&goal](const Goal& item) { return item.stable_id == goal.stable_id; }
    );
    if (duplicate) {
        throw std::invalid_argument("duplicate RLF-6 goal ID");
    }
    if (goal.creation_step == 0U) {
        goal.creation_step = state_.step_index;
    }
    if (goal.status == GoalStatus::pending && state_.active_goal == 0U) {
        goal.status = GoalStatus::active;
        state_.active_goal = goal.stable_id;
    }
    state_.goal_stack.push_back(std::move(goal));
    static_cast<void>(detect_goal_conflicts());
    return state_.goal_stack.back().stable_id;
}

void AgentFabric::set_active_goal(const std::uint64_t goal_id) {
    Goal* goal = find_goal_mutable(goal_id);
    if (goal == nullptr) {
        throw std::out_of_range("unknown RLF-6 goal ID");
    }
    if (goal->status == GoalStatus::completed ||
        goal->status == GoalStatus::failed ||
        goal->status == GoalStatus::impossible ||
        goal->status == GoalStatus::conflicted) {
        throw std::invalid_argument("RLF-6 terminal goal cannot be activated");
    }
    if (Goal* current = find_goal_mutable(state_.active_goal);
        current != nullptr && current->status == GoalStatus::active) {
        current->status = GoalStatus::pending;
    }
    goal->status = GoalStatus::active;
    state_.active_goal = goal_id;
}

bool AgentFabric::detect_goal_conflicts() {
    bool conflict = false;
    for (std::size_t left_index = 0U; left_index < state_.goal_stack.size(); ++left_index) {
        Goal& left = state_.goal_stack[left_index];
        if (left.optional || left.status == GoalStatus::completed ||
            left.status == GoalStatus::failed || left.status == GoalStatus::abandoned) {
            continue;
        }
        for (std::size_t right_index = left_index + 1U;
             right_index < state_.goal_stack.size(); ++right_index) {
            Goal& right = state_.goal_stack[right_index];
            if (right.optional || right.status == GoalStatus::completed ||
                right.status == GoalStatus::failed ||
                right.status == GoalStatus::abandoned) {
                continue;
            }
            for (const Fact& left_fact : left.completion_conditions) {
                for (const Fact& right_fact : right.completion_conditions) {
                    const bool direct_conflict = left_fact.key == right_fact.key &&
                        (left_fact.value != right_fact.value ||
                         left_fact.negated != right_fact.negated);
                    if (direct_conflict) {
                        left.status = GoalStatus::conflicted;
                        right.status = GoalStatus::conflicted;
                        conflict = true;
                    }
                }
            }
        }
    }
    if (conflict) {
        ErrorEvent event;
        event.type = ErrorType::goal_conflict;
        event.step = state_.step_index;
        event.context = "incompatible active goal conditions";
        event.severity = 1.0;
        record_error(std::move(event));
    }
    return conflict;
}

bool AgentFabric::goal_conditions_satisfied(const std::uint64_t goal_id) const {
    const Goal* goal = find_goal(goal_id);
    if (goal == nullptr) {
        throw std::out_of_range("unknown RLF-6 goal ID");
    }
    const auto facts = accepted_facts();
    return std::all_of(
        goal->completion_conditions.begin(), goal->completion_conditions.end(),
        [&facts](const Fact& fact) {
            const bool present = facts.contains(Fact{fact.key, fact.value, false}.canonical());
            return fact.negated ? !present : present;
        }
    );
}

bool AgentFabric::goal_conditions_impossible(
    const std::uint64_t goal_id,
    const std::vector<Action>& actions
) const {
    const Goal* goal = find_goal(goal_id);
    if (goal == nullptr) {
        throw std::out_of_range("unknown RLF-6 goal ID");
    }
    const auto facts = accepted_facts();
    for (const Fact& condition : goal->completion_conditions) {
        const bool satisfied = condition.negated
            ? !facts.contains(Fact{condition.key, condition.value, false}.canonical())
            : facts.contains(condition.canonical());
        if (satisfied) {
            continue;
        }
        const bool producer = std::any_of(
            actions.begin(), actions.end(),
            [&condition](const Action& action) {
                return std::any_of(
                    action.expected_effects.begin(), action.expected_effects.end(),
                    [&condition](const Fact& effect) {
                        return effect.key == condition.key &&
                            effect.value == condition.value &&
                            effect.negated == condition.negated;
                    }
                );
            }
        );
        if (!producer) {
            return true;
        }
    }
    return false;
}

void AgentFabric::update_goal_statuses(const std::vector<Action>& actions) {
    const auto facts = accepted_facts();
    for (Goal& goal : state_.goal_stack) {
        if (goal.status == GoalStatus::completed ||
            goal.status == GoalStatus::failed ||
            goal.status == GoalStatus::conflicted ||
            goal.status == GoalStatus::abandoned) {
            continue;
        }
        const bool failure = std::any_of(
            goal.failure_conditions.begin(), goal.failure_conditions.end(),
            [&facts](const Fact& fact) {
                const bool present = facts.contains(Fact{fact.key, fact.value, false}.canonical());
                return fact.negated ? !present : present;
            }
        );
        if (failure) {
            goal.status = GoalStatus::failed;
            continue;
        }
        if (goal_conditions_satisfied(goal.stable_id)) {
            goal.status = GoalStatus::completed;
        } else if (goal_conditions_impossible(goal.stable_id, actions)) {
            goal.status = GoalStatus::impossible;
        }
    }
    if (const Goal* active = find_goal(state_.active_goal);
        active == nullptr || active->status != GoalStatus::active) {
        state_.active_goal = 0U;
        const auto next = std::max_element(
            state_.goal_stack.begin(), state_.goal_stack.end(),
            [](const Goal& left, const Goal& right) {
                const bool left_available = left.status == GoalStatus::pending;
                const bool right_available = right.status == GoalStatus::pending;
                if (left_available != right_available) {
                    return !left_available;
                }
                if (left.priority != right.priority) {
                    return left.priority < right.priority;
                }
                return left.stable_id > right.stable_id;
            }
        );
        if (next != state_.goal_stack.end() && next->status == GoalStatus::pending) {
            next->status = GoalStatus::active;
            state_.active_goal = next->stable_id;
        }
    }
    state_.progress_state = state_.active_goal == 0U
        ? 1.0
        : progress_for_goal(state_.active_goal);
}

void AgentFabric::ingest_evidence(EvidenceRecord evidence) {
    if (!finite_probability(evidence.confidence) ||
        !finite_probability(evidence.source_reliability)) {
        throw std::invalid_argument("invalid RLF-6 evidence probability");
    }
    if (evidence.stable_id == 0U) {
        evidence.stable_id = allocate_id();
    } else {
        next_id_ = std::max(next_id_, evidence.stable_id + 1U);
    }
    evidence.creation_step = evidence.creation_step == 0U
        ? state_.step_index : evidence.creation_step;
    evidence.last_update_step = state_.step_index;
    state_.observation_state.push_back(evidence);

    const double evidence_support = clamp_probability(
        evidence.confidence * evidence.source_reliability
    );
    bool matched = false;
    for (BeliefHypothesis& belief : state_.belief_state) {
        if (belief.fact.key != evidence.fact.key) {
            continue;
        }
        if (belief.fact.value == evidence.fact.value &&
            belief.fact.negated == evidence.fact.negated) {
            belief.support = 1.0 - (1.0 - belief.support) *
                (1.0 - evidence_support);
            belief.recency = state_.step_index;
            belief.source_reliability = evidence.source_reliability;
            belief.verified = belief.verified || evidence.verified ||
                evidence.kind == EvidenceKind::verified_fact;
            belief.stale = evidence.stale;
            belief.uncertainty = 1.0 - belief.support;
            matched = true;
        } else {
            belief.support *= (1.0 - 0.75 * evidence_support);
            belief.contradiction_count += 1U;
            belief.uncertainty = std::clamp(
                1.0 - belief.support + 0.05 * static_cast<double>(belief.contradiction_count),
                0.0,
                1.0
            );
        }
    }
    if (!matched) {
        state_.belief_state.push_back({
            allocate_id(), evidence.fact, evidence_support, 0U,
            state_.step_index, evidence.source_reliability,
            1.0 - evidence_support,
            evidence.verified || evidence.kind == EvidenceKind::verified_fact,
            evidence.stale,
        });
    }
    if (evidence.verified || evidence.kind == EvidenceKind::verified_fact) {
        const auto found = std::find(
            state_.verified_facts.begin(), state_.verified_facts.end(),
            evidence.fact
        );
        if (found == state_.verified_facts.end()) {
            state_.verified_facts.push_back(evidence.fact);
        }
    }
    double uncertainty_sum = 0.0;
    for (const auto& belief : state_.belief_state) {
        uncertainty_sum += belief.uncertainty;
    }
    state_.uncertainty_state.belief_uncertainty = state_.belief_state.empty()
        ? 1.0
        : uncertainty_sum / static_cast<double>(state_.belief_state.size());
    prune_to_bounds();
}

void AgentFabric::mark_stale(const std::string_view key) {
    for (BeliefHypothesis& belief : state_.belief_state) {
        if (belief.fact.key == key) {
            belief.stale = true;
            belief.uncertainty = std::min(1.0, belief.uncertainty + 0.25);
        }
    }
    for (EvidenceRecord& evidence : state_.observation_state) {
        if (evidence.fact.key == key) {
            evidence.stale = true;
            evidence.kind = EvidenceKind::stale_information;
        }
    }
}

std::optional<BeliefHypothesis> AgentFabric::best_belief(
    const std::string_view key
) const {
    std::optional<BeliefHypothesis> best;
    for (const BeliefHypothesis& belief : state_.belief_state) {
        if (belief.fact.key != key || belief.stale) {
            continue;
        }
        if (!best.has_value() || belief.support > best->support ||
            (belief.support == best->support && belief.stable_id < best->stable_id)) {
            best = belief;
        }
    }
    return best;
}

std::set<std::string> AgentFabric::accepted_facts() const {
    std::map<std::string, const BeliefHypothesis*> best_by_key;
    for (const auto& belief : state_.belief_state) {
        if (belief.stale || belief.support < config_.belief_acceptance_threshold) {
            continue;
        }
        const auto found = best_by_key.find(belief.fact.key);
        if (found == best_by_key.end() ||
            belief.support > found->second->support ||
            (belief.support == found->second->support &&
             belief.stable_id < found->second->stable_id)) {
            best_by_key[belief.fact.key] = &belief;
        }
    }
    std::set<std::string> facts;
    for (const auto& [key, belief] : best_by_key) {
        static_cast<void>(key);
        if (!belief->fact.negated) {
            facts.insert(belief->fact.canonical());
        }
    }
    return facts;
}

void AgentFabric::register_tool(ToolDefinition tool) {
    if (tool.stable_id == 0U) {
        tool.stable_id = allocate_id();
    } else {
        next_id_ = std::max(next_id_, tool.stable_id + 1U);
    }
    if (tool.name.empty() || !finite_probability(tool.declared_reliability) ||
        !std::isfinite(tool.cost) || tool.cost < 0.0) {
        throw std::invalid_argument("invalid RLF-6 tool definition");
    }
    if (std::any_of(
            state_.tool_state.begin(), state_.tool_state.end(),
            [&tool](const ToolDefinition& item) {
                return item.stable_id == tool.stable_id || item.name == tool.name;
            }
        )) {
        throw std::invalid_argument("duplicate RLF-6 tool ID or name");
    }
    std::set<std::string> schema_names;
    for (const std::string& field : tool.input_schema) {
        if (field.empty() || !schema_names.insert(field).second) {
            throw std::invalid_argument("invalid or duplicate tool schema field");
        }
    }
    state_.tool_state.push_back(std::move(tool));
}

const ToolDefinition* AgentFabric::find_tool(const std::uint64_t tool_id) const {
    const auto found = std::find_if(
        state_.tool_state.begin(), state_.tool_state.end(),
        [tool_id](const ToolDefinition& tool) { return tool.stable_id == tool_id; }
    );
    return found == state_.tool_state.end() ? nullptr : &*found;
}

bool AgentFabric::validate_tool_arguments(
    const Action& action,
    std::string* const reason
) const {
    if (action.action_type != ActionType::tool &&
        action.action_type != ActionType::information_seeking &&
        action.action_type != ActionType::verification) {
        if (reason != nullptr) {
            *reason = "action is not a tool action";
        }
        return false;
    }
    const ToolDefinition* tool = find_tool(action.tool_id);
    if (tool == nullptr) {
        if (reason != nullptr) {
            *reason = "unknown tool";
        }
        return false;
    }
    std::map<std::string, std::size_t> parameter_counts;
    for (const auto& [name, value] : action.parameters) {
        if (name.empty() || value.empty()) {
            if (reason != nullptr) {
                *reason = "empty tool argument";
            }
            return false;
        }
        ++parameter_counts[name];
    }
    for (const std::string& field : tool->input_schema) {
        if (parameter_counts[field] != 1U) {
            if (reason != nullptr) {
                *reason = "missing or duplicate required argument: " + field;
            }
            return false;
        }
    }
    for (const auto& [name, count] : parameter_counts) {
        if (count != 1U ||
            std::find(tool->input_schema.begin(), tool->input_schema.end(), name) ==
                tool->input_schema.end()) {
            if (reason != nullptr) {
                *reason = "unexpected or duplicate argument: " + name;
            }
            return false;
        }
    }
    return true;
}

void AgentFabric::update_tool_reliability(
    const std::uint64_t tool_id,
    std::string context,
    const bool success
) {
    if (find_tool(tool_id) == nullptr) {
        throw std::out_of_range("unknown RLF-6 tool ID");
    }
    auto found = std::find_if(
        state_.tool_reliability.begin(), state_.tool_reliability.end(),
        [tool_id, &context](const ToolReliability& item) {
            return item.tool_id == tool_id && item.context == context;
        }
    );
    if (found == state_.tool_reliability.end()) {
        if (state_.tool_reliability.size() >=
            config_.maximum_tool_reliability_records) {
            const auto oldest = std::min_element(
                state_.tool_reliability.begin(), state_.tool_reliability.end(),
                [](const ToolReliability& left, const ToolReliability& right) {
                    if (left.last_update_step != right.last_update_step) {
                        return left.last_update_step < right.last_update_step;
                    }
                    return left.tool_id < right.tool_id;
                }
            );
            state_.tool_reliability.erase(oldest);
        }
        const ToolDefinition* definition = find_tool(tool_id);
        const double prior = definition == nullptr
            ? 0.5 : definition->declared_reliability;
        constexpr double prior_strength = 2.0;
        state_.tool_reliability.push_back({
            tool_id, std::move(context), prior * prior_strength,
            (1.0 - prior) * prior_strength, prior * prior_strength,
            (1.0 - prior) * prior_strength, state_.step_index
        });
        found = std::prev(state_.tool_reliability.end());
    }
    found->recent_successes *= config_.recent_decay;
    found->recent_failures *= config_.recent_decay;
    if (success) {
        found->successes += 1.0;
        found->recent_successes += 1.0;
    } else {
        found->failures += 1.0;
        found->recent_failures += 1.0;
    }
    found->last_update_step = state_.step_index;
}

double AgentFabric::tool_reliability(
    const std::uint64_t tool_id,
    const std::string_view context
) const {
    const auto exact = std::find_if(
        state_.tool_reliability.begin(), state_.tool_reliability.end(),
        [tool_id, context](const ToolReliability& item) {
            return item.tool_id == tool_id && item.context == context;
        }
    );
    if (exact != state_.tool_reliability.end()) {
        return exact->mean();
    }
    const ToolDefinition* tool = find_tool(tool_id);
    return tool == nullptr ? 0.0 : tool->declared_reliability;
}

SafetyDecision AgentFabric::evaluate_safety(
    const Action& action,
    const std::vector<Action>& alternatives
) const {
    SafetyDecision decision;
    if (action.safety_class == SafetyClass::prohibited) {
        decision.allowed = false;
        decision.should_abstain = true;
        decision.reason = "action is prohibited";
        decision.confidence = 1.0;
    } else if (action.safety_class == SafetyClass::high_risk ||
               (!action.reversible &&
                action.confidence < config_.irreversible_confidence_threshold)) {
        decision.requires_verification = true;
        const bool verified = std::all_of(
            action.preconditions.begin(), action.preconditions.end(),
            [this](const Fact& precondition) {
                return std::find(
                    state_.verified_facts.begin(), state_.verified_facts.end(),
                    precondition
                ) != state_.verified_facts.end();
            }
        );
        if (!verified) {
            decision.allowed = false;
            decision.reason = "high-risk or irreversible action lacks verified preconditions";
            decision.confidence = 1.0 - state_.uncertainty_state.safety_uncertainty;
        }
    } else if (action.safety_class == SafetyClass::conditionally_allowed) {
        decision.requires_verification = action.confidence <
            config_.verification_threshold;
        if (decision.requires_verification) {
            decision.allowed = false;
            decision.reason = "conditional action requires verification";
        }
    }
    if (!decision.allowed) {
        const auto safer = std::min_element(
            alternatives.begin(), alternatives.end(),
            [](const Action& left, const Action& right) {
                if (left.safety_class != right.safety_class) {
                    return static_cast<unsigned int>(left.safety_class) <
                        static_cast<unsigned int>(right.safety_class);
                }
                if (left.estimated_risk != right.estimated_risk) {
                    return left.estimated_risk < right.estimated_risk;
                }
                return left.stable_id < right.stable_id;
            }
        );
        if (safer != alternatives.end() &&
            safer->safety_class != SafetyClass::prohibited &&
            safer->stable_id != action.stable_id) {
            decision.safer_alternative = safer->stable_id;
            decision.should_abstain = false;
        }
    }
    return decision;
}

std::vector<Goal> AgentFabric::discover_subgoals(
    const std::uint64_t goal_id,
    const std::vector<Action>& actions
) {
    const Goal* goal = find_goal(goal_id);
    if (goal == nullptr) {
        throw std::out_of_range("unknown RLF-6 goal ID");
    }
    std::vector<Goal> proposed;
    std::set<std::string> known = accepted_facts();
    std::set<std::string> proposed_facts;
    std::vector<Fact> frontier;
    for (const auto& condition : goal->completion_conditions) {
        if (!known.contains(condition.canonical())) {
            frontier.push_back(condition);
        }
    }
    for (std::size_t index = 0U; index < frontier.size() &&
         proposed.size() < config_.maximum_goals; ++index) {
        const Fact missing = frontier[index];
        for (const Action& action : actions) {
            const bool produces = std::find(
                action.expected_effects.begin(), action.expected_effects.end(), missing
            ) != action.expected_effects.end();
            if (!produces) {
                continue;
            }
            for (const Fact& precondition : action.preconditions) {
                if (known.contains(precondition.canonical()) ||
                    proposed_facts.contains(precondition.canonical())) {
                    continue;
                }
                proposed_facts.insert(precondition.canonical());
                frontier.push_back(precondition);
                Goal subgoal;
                subgoal.specification = "satisfy precondition " + precondition.canonical();
                subgoal.completion_conditions = {precondition};
                subgoal.dependencies = {};
                subgoal.priority = goal->priority + 0.01 *
                    static_cast<double>(frontier.size() - index);
                subgoal.creation_step = state_.step_index;
                subgoal.status = GoalStatus::pending;
                subgoal.confidence = action.confidence;
                subgoal.provenance = "graph-derived unresolved precondition";
                proposed.push_back(subgoal);
            }
        }
    }
    for (Goal& subgoal : proposed) {
        const std::uint64_t id = add_goal(subgoal);
        state_.subgoal_graph.emplace_back(goal_id, id);
        subgoal.stable_id = id;
    }
    return proposed;
}

std::string AgentFabric::action_signature(const Action& action) {
    std::ostringstream output;
    output << static_cast<unsigned int>(action.action_type) << ':' << action.name;
    for (const auto& [name, value] : action.parameters) {
        output << '|' << name << '=' << value;
    }
    return output.str();
}

std::string AgentFabric::state_key(const std::set<std::string>& facts) {
    std::string result;
    for (const auto& fact : facts) {
        result += fact;
        result.push_back(';');
    }
    return result;
}

bool AgentFabric::preconditions_satisfied(
    const std::vector<Fact>& preconditions,
    const std::set<std::string>& facts
) {
    return std::all_of(
        preconditions.begin(), preconditions.end(),
        [&facts](const Fact& fact) {
            const bool present = facts.contains(Fact{fact.key, fact.value, false}.canonical());
            return fact.negated ? !present : present;
        }
    );
}

std::set<std::string> AgentFabric::apply_effects(
    std::set<std::string> facts,
    const std::vector<Fact>& effects
) {
    for (const Fact& effect : effects) {
        const std::string prefix = key_prefix(effect);
        for (auto iterator = facts.begin(); iterator != facts.end();) {
            if (iterator->starts_with(prefix)) {
                iterator = facts.erase(iterator);
            } else {
                ++iterator;
            }
        }
        if (!effect.negated) {
            facts.insert(effect.canonical());
        }
    }
    return facts;
}

std::size_t AgentFabric::goal_satisfied_count(
    const Goal& goal,
    const std::set<std::string>& facts
) const {
    return static_cast<std::size_t>(std::count_if(
        goal.completion_conditions.begin(), goal.completion_conditions.end(),
        [&facts](const Fact& fact) {
            const bool present = facts.contains(Fact{fact.key, fact.value, false}.canonical());
            return fact.negated ? !present : present;
        }
    ));
}

std::vector<CandidateScore> AgentFabric::score_actions(
    const std::vector<Action>& actions,
    const std::uint64_t goal_id
) const {
    const Goal* goal = find_goal(goal_id);
    if (goal == nullptr) {
        throw std::out_of_range("unknown RLF-6 goal ID");
    }
    const auto facts = accepted_facts();
    const std::size_t before = goal_satisfied_count(*goal, facts);
    std::vector<CandidateScore> scores;
    scores.reserve(actions.size());
    const std::string context = state_key(facts);
    for (const Action& action : actions) {
        if (!preconditions_satisfied(action.preconditions, facts)) {
            continue;
        }
        const auto after_facts = apply_effects(facts, action.expected_effects);
        const std::size_t after = goal_satisfied_count(*goal, after_facts);
        const double progress = static_cast<double>(after) -
            static_cast<double>(before);
        const double success = predicted_action_success(action, context);
        const double uncertainty = predicted_action_uncertainty(action, context);
        const double tool_factor = action.tool_id == 0U
            ? 1.0
            : tool_reliability(action.tool_id, action.name);
        const double utility = 3.0 * progress +
            (action.information_value ? 0.75 * uncertainty : 0.0) +
            success * tool_factor - action.estimated_cost -
            2.0 * action.estimated_risk - uncertainty;
        scores.push_back({
            action.stable_id, progress, success, action.estimated_cost,
            action.estimated_risk, uncertainty, utility,
        });
    }
    std::stable_sort(
        scores.begin(), scores.end(),
        [](const CandidateScore& left, const CandidateScore& right) {
            if (left.utility != right.utility) {
                return left.utility > right.utility;
            }
            return left.action_id < right.action_id;
        }
    );
    return scores;
}

Plan AgentFabric::plan(const PlanningRequest& request) {
    if (request.actions.empty() ||
        request.actions.size() > config_.maximum_candidate_actions ||
        request.node_budget == 0U || request.depth_budget == 0U) {
        throw std::invalid_argument("invalid RLF-6 planning request");
    }
    const Goal* goal = find_goal(request.goal_id);
    if (goal == nullptr) {
        throw std::out_of_range("unknown RLF-6 planning goal");
    }
    Plan result;
    result.goal_id = request.goal_id;
    result.resource_budget = request.node_budget;
    const auto initial_facts = accepted_facts();
    if (goal_satisfied_count(*goal, initial_facts) ==
        goal->completion_conditions.size()) {
        result.found = true;
        result.predicted_success = 1.0;
        return result;
    }

    std::vector<Action> actions = request.actions;
    std::stable_sort(
        actions.begin(), actions.end(),
        [](const Action& left, const Action& right) {
            return left.stable_id < right.stable_id;
        }
    );

    if (request.allow_skills && request.policy == PlanningPolicy::skill_guided) {
        const std::string pattern = goal_pattern_from_goal(*goal);
        std::vector<std::uint64_t> preferred;
        for (const Skill& skill : state_.skill_memory) {
            if (!skill.invalidated && skill.goal_pattern == pattern &&
                skill.confidence >= config_.belief_acceptance_threshold) {
                preferred.insert(
                    preferred.end(), skill.action_sequence.begin(),
                    skill.action_sequence.end()
                );
            }
        }
        std::stable_sort(
            actions.begin(), actions.end(),
            [&preferred](const Action& left, const Action& right) {
                const auto left_position = std::find(
                    preferred.begin(), preferred.end(), left.stable_id
                );
                const auto right_position = std::find(
                    preferred.begin(), preferred.end(), right.stable_id
                );
                if ((left_position != preferred.end()) !=
                    (right_position != preferred.end())) {
                    return left_position != preferred.end();
                }
                if (left_position != preferred.end() &&
                    right_position != preferred.end() &&
                    left_position != right_position) {
                    return left_position < right_position;
                }
                return left.stable_id < right.stable_id;
            }
        );
    }

    if (request.policy == PlanningPolicy::reactive_one_step) {
        const auto scores = score_actions(actions, request.goal_id);
        if (!scores.empty() && scores.front().progress > 0.0) {
            result.action_ids = {scores.front().action_id};
            result.predicted_cost = scores.front().cost;
            result.predicted_success = scores.front().predicted_success;
            result.uncertainty = scores.front().uncertainty;
            result.nodes_expanded = scores.size();
            result.maximum_depth = 1U;
            result.found = true;
        }
        state_.resource_state.planning_nodes += result.nodes_expanded;
        state_.resource_state.reasoning_cycles += 1U;
        return result;
    }

    struct Node final {
        std::set<std::string> facts;
        std::vector<std::uint64_t> path;
        std::vector<std::uint64_t> state_hashes;
        double cost{};
        double success{1.0};
        double uncertainty{};
        double priority{};
        std::size_t depth{};
        std::uint64_t serial{};
    };
    struct NodeCompare final {
        bool operator()(const Node& left, const Node& right) const noexcept {
            if (left.priority != right.priority) {
                return left.priority > right.priority;
            }
            if (left.cost != right.cost) {
                return left.cost > right.cost;
            }
            return left.serial > right.serial;
        }
    };

    std::priority_queue<Node, std::vector<Node>, NodeCompare> frontier;
    std::uint64_t serial = 0U;
    const std::size_t initial_unsatisfied = goal->completion_conditions.size() -
        goal_satisfied_count(*goal, initial_facts);
    frontier.push({initial_facts, {}, {}, 0.0, 1.0, 0.0,
                   static_cast<double>(initial_unsatisfied), 0U, serial++});
    std::map<std::string, double> best_cost;
    best_cost.emplace(state_key(initial_facts), 0.0);
    const std::string context = state_key(initial_facts);
    Node best_partial = frontier.top();
    std::size_t best_partial_satisfied =
        goal_satisfied_count(*goal, best_partial.facts);

    while (!frontier.empty() && result.nodes_expanded < request.node_budget) {
        Node node = frontier.top();
        frontier.pop();
        ++result.nodes_expanded;
        result.maximum_depth = std::max(result.maximum_depth, node.depth);
        const std::size_t satisfied = goal_satisfied_count(*goal, node.facts);
        if (satisfied > best_partial_satisfied ||
            (satisfied == best_partial_satisfied &&
             node.path.size() > best_partial.path.size()) ||
            (satisfied == best_partial_satisfied &&
             node.path.size() == best_partial.path.size() &&
             node.cost < best_partial.cost)) {
            best_partial = node;
            best_partial_satisfied = satisfied;
        }
        if (satisfied == goal->completion_conditions.size()) {
            result.action_ids = std::move(node.path);
            result.predicted_state_hashes = std::move(node.state_hashes);
            result.predicted_cost = node.cost;
            result.predicted_success = node.success;
            result.uncertainty = node.path.empty()
                ? 0.0 : node.uncertainty / static_cast<double>(node.path.size());
            result.found = true;
            break;
        }
        if (node.depth >= std::min(request.depth_budget, config_.maximum_plan_depth)) {
            continue;
        }

        std::vector<const Action*> applicable;
        for (const Action& action : actions) {
            if (preconditions_satisfied(action.preconditions, node.facts) &&
                action.safety_class != SafetyClass::prohibited) {
                applicable.push_back(&action);
            }
        }
        if (request.policy == PlanningPolicy::greedy_progress) {
            std::stable_sort(
                applicable.begin(), applicable.end(),
                [this, goal, &node, &context](const Action* left, const Action* right) {
                    const auto left_facts = apply_effects(node.facts, left->expected_effects);
                    const auto right_facts = apply_effects(node.facts, right->expected_effects);
                    const double left_progress = static_cast<double>(
                        goal_satisfied_count(*goal, left_facts)
                    );
                    const double right_progress = static_cast<double>(
                        goal_satisfied_count(*goal, right_facts)
                    );
                    const double left_score = 4.0 * left_progress +
                        predicted_action_success(*left, context) -
                        left->estimated_cost - left->estimated_risk;
                    const double right_score = 4.0 * right_progress +
                        predicted_action_success(*right, context) -
                        right->estimated_cost - right->estimated_risk;
                    if (left_score != right_score) {
                        return left_score > right_score;
                    }
                    return left->stable_id < right->stable_id;
                }
            );
            if (applicable.size() > 1U) {
                applicable.resize(1U);
            }
        }

        for (const Action* action : applicable) {
            const auto next_facts = apply_effects(node.facts, action->expected_effects);
            if (next_facts == node.facts && !action->information_value) {
                continue;
            }
            const double success = request.use_learned_model
                ? predicted_action_success(*action, context)
                : clamp_probability(action->confidence);
            const double uncertainty = request.use_learned_model
                ? predicted_action_uncertainty(*action, context)
                : 1.0 - clamp_probability(action->confidence);
            double step_cost = action->estimated_cost +
                2.0 * action->estimated_risk + uncertainty +
                (1.0 - success);
            if (request.policy == PlanningPolicy::uncertainty_aware &&
                action->information_value) {
                step_cost -= 0.5 * uncertainty;
            }
            step_cost = std::max(0.01, step_cost);
            const double next_cost = node.cost + step_cost;
            const std::string next_key = state_key(next_facts);
            const auto visited = best_cost.find(next_key);
            if (visited != best_cost.end() && visited->second <= next_cost) {
                continue;
            }
            best_cost[next_key] = next_cost;
            Node next;
            next.facts = next_facts;
            next.path = node.path;
            next.path.push_back(action->stable_id);
            next.state_hashes = node.state_hashes;
            std::uint64_t state_hash = fnv_offset_basis;
            hash_string(state_hash, next_key);
            next.state_hashes.push_back(state_hash);
            next.cost = next_cost;
            next.success = node.success * success;
            next.uncertainty = node.uncertainty + uncertainty;
            next.depth = node.depth + 1U;
            const std::size_t unsatisfied = goal->completion_conditions.size() -
                goal_satisfied_count(*goal, next.facts);
            const double heuristic = static_cast<double>(unsatisfied);
            if (request.policy == PlanningPolicy::bounded_best_first) {
                next.priority = heuristic + 0.25 * next.cost;
            } else if (request.policy == PlanningPolicy::receding_horizon) {
                next.priority = heuristic + next.cost +
                    0.1 * static_cast<double>(next.depth);
            } else {
                next.priority = heuristic + next.cost;
            }
            next.serial = serial++;
            frontier.push(std::move(next));
        }
    }
    if (!result.found && !best_partial.path.empty() &&
        (request.policy == PlanningPolicy::receding_horizon ||
         request.policy == PlanningPolicy::skill_guided ||
         request.policy == PlanningPolicy::uncertainty_aware ||
         request.policy == PlanningPolicy::bounded_best_first ||
         request.policy == PlanningPolicy::bounded_astar)) {
        result.action_ids = std::move(best_partial.path);
        result.predicted_state_hashes = std::move(best_partial.state_hashes);
        result.predicted_cost = best_partial.cost;
        result.predicted_success = best_partial.success;
        result.uncertainty = result.action_ids.empty()
            ? 1.0
            : best_partial.uncertainty /
                static_cast<double>(result.action_ids.size());
        result.found = true;
    }
    state_.resource_state.planning_nodes += result.nodes_expanded;
    state_.resource_state.reasoning_cycles += 1U;
    state_.uncertainty_state.plan_uncertainty = result.found
        ? result.uncertainty : 1.0;
    return result;
}

void AgentFabric::record_transition(
    const std::string& context_key,
    const Action& action,
    const std::vector<Fact>& actual_effects,
    const bool success,
    const double cost,
    const bool terminal_failure
) {
    if (!std::isfinite(cost) || cost < 0.0) {
        throw std::invalid_argument("invalid RLF-6 transition cost");
    }
    const std::string signature = action_signature(action);
    auto found = std::find_if(
        state_.world_model_state.begin(), state_.world_model_state.end(),
        [&context_key, &signature](const TransitionRecord& record) {
            return record.context_key == context_key &&
                record.action_signature == signature;
        }
    );
    if (found == state_.world_model_state.end()) {
        if (state_.world_model_state.size() >= config_.maximum_transition_records) {
            const auto least = std::min_element(
                state_.world_model_state.begin(), state_.world_model_state.end(),
                [](const TransitionRecord& left, const TransitionRecord& right) {
                    if (left.observations != right.observations) {
                        return left.observations < right.observations;
                    }
                    return left.model_version < right.model_version;
                }
            );
            state_.world_model_state.erase(least);
        }
        state_.world_model_state.push_back(TransitionRecord{
            context_key, signature, {}, 0U, 0U, 0.0, 0.5
        });
        found = std::prev(state_.world_model_state.end());
    }
    for (TransitionOutcome& outcome : found->outcomes) {
        outcome.recent_count *= config_.recent_decay;
    }
    auto outcome = std::find_if(
        found->outcomes.begin(), found->outcomes.end(),
        [&actual_effects](const TransitionOutcome& candidate) {
            return candidate.effects == actual_effects;
        }
    );
    if (outcome == found->outcomes.end()) {
        found->outcomes.push_back({actual_effects});
        outcome = std::prev(found->outcomes.end());
    }
    outcome->count += 1.0;
    outcome->recent_count += 1.0;
    outcome->total_cost += cost;
    if (terminal_failure) {
        outcome->terminal_failures += 1U;
    }
    found->observations += 1U;
    const bool expected = actual_effects == action.expected_effects && success;
    const double surprise = expected ? 0.0 : 1.0;
    found->surprise_ema = 0.9 * found->surprise_ema + 0.1 * surprise;
    found->reliability = static_cast<double>(std::count_if(
        found->outcomes.begin(), found->outcomes.end(),
        [&action](const TransitionOutcome& item) {
            return item.effects == action.expected_effects;
        }
    ));
    double matching_count = 0.0;
    double total = 0.0;
    for (const auto& item : found->outcomes) {
        total += item.count;
        if (item.effects == action.expected_effects) {
            matching_count += item.count;
        }
    }
    found->reliability = (matching_count + 1.0) / (total + 2.0);
    if (found->surprise_ema > config_.change_detection_threshold) {
        found->model_version += 1U;
        for (TransitionOutcome& item : found->outcomes) {
            item.recent_count = std::max(1.0, item.recent_count);
        }
    }
}

double AgentFabric::predicted_action_success(
    const Action& action,
    const std::string_view context_key
) const {
    const std::string signature = action_signature(action);
    const auto exact = std::find_if(
        state_.world_model_state.begin(), state_.world_model_state.end(),
        [context_key, &signature](const TransitionRecord& record) {
            return record.context_key == context_key &&
                record.action_signature == signature;
        }
    );
    if (exact != state_.world_model_state.end()) {
        return clamp_probability(exact->reliability);
    }
    double successes = 1.0;
    double total = 2.0;
    for (const auto& record : state_.world_model_state) {
        if (record.action_signature == signature) {
            successes += record.reliability *
                static_cast<double>(record.observations);
            total += static_cast<double>(record.observations);
        }
    }
    if (total > 2.0) {
        return clamp_probability(successes / total);
    }
    return clamp_probability(action.confidence);
}

double AgentFabric::predicted_action_uncertainty(
    const Action& action,
    const std::string_view context_key
) const {
    const std::string signature = action_signature(action);
    const auto exact = std::find_if(
        state_.world_model_state.begin(), state_.world_model_state.end(),
        [context_key, &signature](const TransitionRecord& record) {
            return record.context_key == context_key &&
                record.action_signature == signature;
        }
    );
    if (exact == state_.world_model_state.end()) {
        return 1.0 - clamp_probability(action.confidence);
    }
    const double sample_uncertainty = 1.0 /
        std::sqrt(static_cast<double>(exact->observations) + 1.0);
    double distribution_uncertainty = 0.0;
    double total = 0.0;
    for (const auto& outcome : exact->outcomes) {
        total += outcome.recent_count;
    }
    if (total > 0.0) {
        for (const auto& outcome : exact->outcomes) {
            const double probability = outcome.recent_count / total;
            if (probability > 0.0) {
                distribution_uncertainty -= probability * std::log(probability);
            }
        }
        const double maximum_entropy = exact->outcomes.size() > 1U
            ? std::log(static_cast<double>(exact->outcomes.size()))
            : 1.0;
        distribution_uncertainty /= maximum_entropy;
    }
    return std::clamp(
        0.5 * sample_uncertainty + 0.5 * distribution_uncertainty,
        0.0,
        1.0
    );
}

std::uint64_t AgentFabric::record_error(ErrorEvent event) {
    if (!std::isfinite(event.severity) || event.severity < 0.0) {
        throw std::invalid_argument("invalid RLF-6 error severity");
    }
    if (event.stable_id == 0U) {
        event.stable_id = allocate_id();
    } else {
        next_id_ = std::max(next_id_, event.stable_id + 1U);
    }
    if (event.step == 0U) {
        event.step = state_.step_index;
    }
    state_.failure_history_summary.push_back(std::move(event));
    if (state_.failure_history_summary.size() > config_.maximum_errors) {
        state_.failure_history_summary.erase(state_.failure_history_summary.begin());
    }
    return state_.failure_history_summary.back().stable_id;
}

void AgentFabric::mark_error_recovered(const std::uint64_t error_id) {
    const auto found = std::find_if(
        state_.failure_history_summary.begin(),
        state_.failure_history_summary.end(),
        [error_id](const ErrorEvent& event) { return event.stable_id == error_id; }
    );
    if (found == state_.failure_history_summary.end()) {
        throw std::out_of_range("unknown RLF-6 error ID");
    }
    found->recovered = true;
}

std::vector<std::uint64_t> AgentFabric::counterfactual_actions(
    const std::vector<Action>& actions,
    const Action& failed_action,
    const std::uint64_t goal_id,
    const std::size_t maximum_alternatives
) const {
    if (maximum_alternatives == 0U) {
        return {};
    }
    auto scores = score_actions(actions, goal_id);
    scores.erase(
        std::remove_if(
            scores.begin(), scores.end(),
            [&failed_action](const CandidateScore& score) {
                return score.action_id == failed_action.stable_id;
            }
        ),
        scores.end()
    );
    if (scores.size() > maximum_alternatives) {
        scores.resize(maximum_alternatives);
    }
    std::vector<std::uint64_t> result;
    result.reserve(scores.size());
    for (const auto& score : scores) {
        result.push_back(score.action_id);
    }
    return result;
}

std::uint64_t AgentFabric::insert_memory(MemoryRecord record) {
    if (record.memory_class == MemoryClass::skill) {
        throw std::invalid_argument("use consolidate_skill for skill memory");
    }
    if (!finite_probability(record.confidence) ||
        !std::isfinite(record.utility)) {
        throw std::invalid_argument("invalid RLF-6 memory confidence or utility");
    }
    if (record.stable_id == 0U) {
        record.stable_id = allocate_id();
    } else {
        next_id_ = std::max(next_id_, record.stable_id + 1U);
    }
    record.creation_step = record.creation_step == 0U
        ? state_.step_index : record.creation_step;
    record.last_use_step = state_.step_index;
    auto* records = memory_vector(state_, record.memory_class);
    if (records == nullptr) {
        throw std::invalid_argument("invalid RLF-6 memory class");
    }
    records->push_back(std::move(record));
    state_.resource_state.memory_writes += 1U;
    state_.resource_state.bytes_written +=
        records->back().key.size() + records->back().payload.size();
    prune_to_bounds();
    return records->back().stable_id;
}

std::vector<MemoryRecord> AgentFabric::retrieve_memory(
    const MemoryClass memory_class,
    const std::string_view key,
    const std::size_t maximum_records
) {
    if (maximum_records == 0U) {
        return {};
    }
    auto* records = memory_vector(state_, memory_class);
    if (records == nullptr) {
        throw std::invalid_argument("skill retrieval uses skill memory directly");
    }
    std::vector<MemoryRecord*> matches;
    for (MemoryRecord& record : *records) {
        if (!record.invalidated &&
            (record.key == key || record.key.find(key) != std::string::npos ||
             std::string(key).find(record.key) != std::string::npos)) {
            matches.push_back(&record);
        }
    }
    std::stable_sort(
        matches.begin(), matches.end(),
        [](const MemoryRecord* left, const MemoryRecord* right) {
            const double left_score = left->confidence + left->utility;
            const double right_score = right->confidence + right->utility;
            if (left_score != right_score) {
                return left_score > right_score;
            }
            return left->stable_id < right->stable_id;
        }
    );
    if (matches.size() > maximum_records) {
        matches.resize(maximum_records);
    }
    std::vector<MemoryRecord> result;
    result.reserve(matches.size());
    for (MemoryRecord* record : matches) {
        record->last_use_step = state_.step_index;
        result.push_back(*record);
        state_.resource_state.bytes_read +=
            record->key.size() + record->payload.size();
    }
    state_.resource_state.memory_reads += matches.size();
    return result;
}

void AgentFabric::invalidate_memory(const std::uint64_t record_id) {
    const std::array<MemoryClass, 4U> classes{
        MemoryClass::working, MemoryClass::episodic,
        MemoryClass::semantic, MemoryClass::safety,
    };
    for (const MemoryClass memory_class : classes) {
        auto* records = memory_vector(state_, memory_class);
        if (records == nullptr) {
            continue;
        }
        const auto found = std::find_if(
            records->begin(), records->end(),
            [record_id](const MemoryRecord& record) {
                return record.stable_id == record_id;
            }
        );
        if (found != records->end()) {
            found->invalidated = true;
            return;
        }
    }
    throw std::out_of_range("unknown RLF-6 memory record ID");
}

std::optional<std::uint64_t> AgentFabric::consolidate_skill(
    std::string goal_pattern,
    const std::vector<std::uint64_t>& action_sequence,
    const double baseline_cost,
    const double realized_cost,
    const bool held_out_improvement
) {
    if (goal_pattern.empty() || action_sequence.size() < 2U ||
        !std::isfinite(baseline_cost) || !std::isfinite(realized_cost) ||
        baseline_cost < 0.0 || realized_cost < 0.0) {
        return std::nullopt;
    }
    const double gain = baseline_cost - realized_cost -
        0.05 * static_cast<double>(action_sequence.size());
    auto found = std::find_if(
        state_.skill_memory.begin(), state_.skill_memory.end(),
        [&goal_pattern, &action_sequence](const Skill& skill) {
            return skill.goal_pattern == goal_pattern &&
                skill.action_sequence == action_sequence;
        }
    );
    if (found == state_.skill_memory.end()) {
        if (state_.skill_memory.size() >= config_.maximum_skills) {
            return std::nullopt;
        }
        Skill candidate;
        candidate.stable_id = allocate_id();
        candidate.goal_pattern = std::move(goal_pattern);
        candidate.action_sequence = action_sequence;
        candidate.estimated_cost = realized_cost;
        candidate.confidence = 0.5;
        candidate.utility = gain;
        candidate.support = 1U;
        state_.skill_memory.push_back(std::move(candidate));
        return std::nullopt;
    }
    found->support += 1U;
    found->estimated_cost =
        (found->estimated_cost * static_cast<double>(found->support - 1U) +
         realized_cost) / static_cast<double>(found->support);
    found->utility = std::max(found->utility, gain);
    if (found->support < 2U || !held_out_improvement ||
        gain < config_.skill_mdl_minimum_gain) {
        return std::nullopt;
    }
    found->confidence = std::clamp(
        1.0 - 1.0 / static_cast<double>(found->support + 1U), 0.0, 1.0
    );
    return found->stable_id;
}

bool AgentFabric::within_budget(const ResourceBudget& budget) const noexcept {
    return state_.resource_state.reasoning_cycles <= budget.reasoning_cycles &&
        state_.resource_state.action_count <= budget.action_count &&
        state_.resource_state.tool_calls <= budget.tool_calls &&
        state_.resource_state.tool_cost <= budget.tool_cost &&
        state_.resource_state.memory_reads <= budget.memory_reads &&
        state_.resource_state.memory_writes <= budget.memory_writes &&
        state_.resource_state.planning_nodes <= budget.planning_nodes &&
        state_.resource_state.simulated_time <= budget.simulated_time &&
        state_.resource_state.risk_used <= budget.risk_budget;
}

void AgentFabric::advance_step() {
    state_.step_index += 1U;
    state_.resource_state.reasoning_cycles += 1U;
}

double AgentFabric::progress_for_goal(const std::uint64_t goal_id) const {
    const Goal* goal = find_goal(goal_id);
    if (goal == nullptr || goal->completion_conditions.empty()) {
        return 0.0;
    }
    return static_cast<double>(goal_satisfied_count(*goal, accepted_facts())) /
        static_cast<double>(goal->completion_conditions.size());
}

std::uint64_t AgentFabric::deterministic_hash() const noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, seed_);
    hash_u64(hash, next_id_);
    hash_u64(hash, state_.episode_id);
    hash_u64(hash, state_.step_index);
    hash_u64(hash, state_.active_goal);
    for (const auto& evidence : state_.observation_state) {
        hash_u64(hash, evidence.stable_id);
        hash_string(hash, evidence.fact.canonical());
        hash_u64(hash, static_cast<std::uint64_t>(evidence.kind));
        hash_double(hash, evidence.confidence);
        hash_double(hash, evidence.source_reliability);
        hash_u64(hash, evidence.verified ? 1U : 0U);
        hash_u64(hash, evidence.stale ? 1U : 0U);
    }
    for (const auto& belief : state_.belief_state) {
        hash_u64(hash, belief.stable_id);
        hash_string(hash, belief.fact.canonical());
        hash_double(hash, belief.support);
        hash_u64(hash, belief.contradiction_count);
        hash_double(hash, belief.uncertainty);
    }
    for (const auto& goal : state_.goal_stack) {
        hash_u64(hash, goal.stable_id);
        hash_string(hash, goal.specification);
        hash_u64(hash, static_cast<std::uint64_t>(goal.status));
        for (const auto& fact : goal.completion_conditions) {
            hash_string(hash, fact.canonical());
        }
    }
    for (const auto& transition : state_.world_model_state) {
        hash_string(hash, transition.context_key);
        hash_string(hash, transition.action_signature);
        hash_u64(hash, transition.observations);
        hash_u64(hash, transition.model_version);
        hash_double(hash, transition.reliability);
    }
    for (const auto& reliability : state_.tool_reliability) {
        hash_u64(hash, reliability.tool_id);
        hash_string(hash, reliability.context);
        hash_double(hash, reliability.successes);
        hash_double(hash, reliability.failures);
        hash_double(hash, reliability.recent_successes);
        hash_double(hash, reliability.recent_failures);
    }
    for (const auto& skill : state_.skill_memory) {
        hash_u64(hash, skill.stable_id);
        hash_string(hash, skill.goal_pattern);
        for (const auto action : skill.action_sequence) {
            hash_u64(hash, action);
        }
        hash_u64(hash, skill.support);
        hash_double(hash, skill.confidence);
    }
    for (const auto action : state_.action_history_summary) {
        hash_u64(hash, action);
    }
    for (const auto& error : state_.failure_history_summary) {
        hash_u64(hash, error.stable_id);
        hash_u64(hash, static_cast<std::uint64_t>(error.type));
        hash_u64(hash, error.action_id);
        hash_u64(hash, error.recovered ? 1U : 0U);
    }
    return hash;
}

std::size_t AgentFabric::estimated_persistent_bytes() const noexcept {
    std::size_t total = sizeof(*this);
    for (const auto& evidence : state_.observation_state) {
        total += sizeof(evidence) + evidence.fact.key.size() +
            evidence.fact.value.size() + evidence.provenance.size();
    }
    for (const auto& belief : state_.belief_state) {
        total += sizeof(belief) + belief.fact.key.size() + belief.fact.value.size();
    }
    for (const auto& goal : state_.goal_stack) {
        total += sizeof(goal) + goal.specification.size() + goal.provenance.size();
        total += goal.dependencies.size() * sizeof(std::uint64_t);
        for (const auto& fact : goal.completion_conditions) {
            total += sizeof(fact) + fact.key.size() + fact.value.size();
        }
        for (const auto& fact : goal.failure_conditions) {
            total += sizeof(fact) + fact.key.size() + fact.value.size();
        }
    }
    const auto memory_bytes = [](const std::vector<MemoryRecord>& records) {
        std::size_t bytes = records.size() * sizeof(MemoryRecord);
        for (const auto& record : records) {
            bytes += record.key.size() + record.payload.size() +
                record.provenance.size();
        }
        return bytes;
    };
    total += memory_bytes(state_.working_memory);
    total += memory_bytes(state_.episodic_memory);
    total += memory_bytes(state_.semantic_memory);
    total += memory_bytes(state_.safety_memory);
    for (const auto& skill : state_.skill_memory) {
        total += sizeof(skill) + skill.goal_pattern.size() +
            skill.action_sequence.size() * sizeof(std::uint64_t) +
            skill.fallback_actions.size() * sizeof(std::uint64_t);
    }
    for (const auto& tool : state_.tool_state) {
        total += sizeof(tool) + tool.name.size() + tool.required_permission.size();
        for (const auto& field : tool.input_schema) total += field.size();
        for (const auto& field : tool.output_schema) total += field.size();
    }
    for (const auto& transition : state_.world_model_state) {
        total += sizeof(transition) + transition.context_key.size() +
            transition.action_signature.size();
        for (const auto& outcome : transition.outcomes) {
            total += sizeof(outcome) + outcome.effects.size() * sizeof(Fact);
            for (const auto& fact : outcome.effects) {
                total += fact.key.size() + fact.value.size();
            }
        }
    }
    return total;
}

AgentSnapshot AgentFabric::snapshot() const {
    return {config_, seed_, next_id_, state_};
}

AgentFabric AgentFabric::from_snapshot(const AgentSnapshot& snapshot) {
    AgentFabric result(snapshot.config, snapshot.seed);
    result.next_id_ = snapshot.next_id;
    result.state_ = snapshot.state;
    result.validate_state();
    return result;
}

void AgentFabric::prune_to_bounds() {
    erase_oldest(state_.observation_state, config_.maximum_observations);
    if (state_.belief_state.size() > config_.maximum_beliefs) {
        std::stable_sort(
            state_.belief_state.begin(), state_.belief_state.end(),
            [](const BeliefHypothesis& left, const BeliefHypothesis& right) {
                if (left.verified != right.verified) return left.verified;
                if (left.stale != right.stale) return !left.stale;
                if (left.support != right.support) return left.support > right.support;
                return left.stable_id < right.stable_id;
            }
        );
        state_.belief_state.resize(config_.maximum_beliefs);
    }
    const std::size_t per_class = std::max<std::size_t>(
        1U, config_.maximum_memory_records / 4U
    );
    erase_oldest_memory(state_.working_memory, per_class);
    erase_oldest_memory(state_.episodic_memory, per_class);
    erase_oldest_memory(state_.semantic_memory, per_class);
    erase_oldest_memory(state_.safety_memory, per_class);
}

void AgentFabric::validate_state() const {
    if (next_id_ == 0U) {
        throw std::runtime_error("invalid RLF-6 next ID");
    }
    std::unordered_set<std::uint64_t> ids;
    const auto add_id = [&ids](const std::uint64_t id, const char* label) {
        if (id == 0U || !ids.insert(id).second) {
            throw std::runtime_error(std::string("invalid or duplicate RLF-6 ") + label + " ID");
        }
    };
    for (const auto& evidence : state_.observation_state) {
        add_id(evidence.stable_id, "evidence");
        if (!finite_probability(evidence.confidence) ||
            !finite_probability(evidence.source_reliability)) {
            throw std::runtime_error("invalid RLF-6 evidence probability");
        }
    }
    for (const auto& belief : state_.belief_state) {
        add_id(belief.stable_id, "belief");
        if (!finite_probability(belief.support) ||
            !finite_probability(belief.source_reliability) ||
            !finite_probability(belief.uncertainty)) {
            throw std::runtime_error("invalid RLF-6 belief probability");
        }
    }
    std::unordered_set<std::uint64_t> goal_ids;
    for (const auto& goal : state_.goal_stack) {
        add_id(goal.stable_id, "goal");
        goal_ids.insert(goal.stable_id);
        if (!finite_probability(goal.confidence)) {
            throw std::runtime_error("invalid RLF-6 goal confidence");
        }
    }
    for (const auto& goal : state_.goal_stack) {
        for (const auto dependency : goal.dependencies) {
            if (!goal_ids.contains(dependency)) {
                throw std::runtime_error("invalid RLF-6 goal dependency");
            }
        }
    }
    std::map<std::uint64_t, std::vector<std::uint64_t>> graph;
    for (const auto& goal : state_.goal_stack) {
        graph[goal.stable_id] = goal.dependencies;
    }
    std::unordered_set<std::uint64_t> visiting;
    std::unordered_set<std::uint64_t> visited;
    const std::function<void(std::uint64_t)> visit =
        [&](const std::uint64_t id) {
            if (visiting.contains(id)) {
                throw std::runtime_error("cyclic RLF-6 goal graph");
            }
            if (visited.contains(id)) return;
            visiting.insert(id);
            for (const auto dependency : graph[id]) visit(dependency);
            visiting.erase(id);
            visited.insert(id);
        };
    for (const auto& [id, dependencies] : graph) {
        static_cast<void>(dependencies);
        visit(id);
    }
    for (const auto& tool : state_.tool_state) {
        add_id(tool.stable_id, "tool");
        if (tool.name.empty() || !finite_probability(tool.declared_reliability)) {
            throw std::runtime_error("invalid RLF-6 tool definition");
        }
    }
    for (const auto& skill : state_.skill_memory) {
        add_id(skill.stable_id, "skill");
        if (skill.action_sequence.empty() || !finite_probability(skill.confidence)) {
            throw std::runtime_error("invalid RLF-6 skill route");
        }
    }
    const std::array<const std::vector<MemoryRecord>*, 4U> memories{
        &state_.working_memory, &state_.episodic_memory,
        &state_.semantic_memory, &state_.safety_memory,
    };
    for (const auto* records : memories) {
        for (const auto& record : *records) {
            add_id(record.stable_id, "memory");
            if (!finite_probability(record.confidence)) {
                throw std::runtime_error("invalid RLF-6 memory confidence");
            }
        }
    }
    for (const auto& error : state_.failure_history_summary) {
        add_id(error.stable_id, "error");
    }
    if (state_.active_goal != 0U && !goal_ids.contains(state_.active_goal)) {
        throw std::runtime_error("invalid RLF-6 active goal reference");
    }
}

std::string to_string(const EvidenceKind value) {
    switch (value) {
        case EvidenceKind::observation: return "observation";
        case EvidenceKind::belief: return "belief";
        case EvidenceKind::hypothesis: return "hypothesis";
        case EvidenceKind::goal: return "goal";
        case EvidenceKind::prediction: return "prediction";
        case EvidenceKind::tool_output: return "tool_output";
        case EvidenceKind::verified_fact: return "verified_fact";
        case EvidenceKind::stale_information: return "stale_information";
    }
    return "unknown";
}

std::string to_string(const GoalStatus value) {
    switch (value) {
        case GoalStatus::pending: return "pending";
        case GoalStatus::active: return "active";
        case GoalStatus::completed: return "completed";
        case GoalStatus::failed: return "failed";
        case GoalStatus::impossible: return "impossible";
        case GoalStatus::conflicted: return "conflicted";
        case GoalStatus::abandoned: return "abandoned";
    }
    return "unknown";
}

std::string to_string(const ActionType value) {
    switch (value) {
        case ActionType::primitive: return "primitive";
        case ActionType::tool: return "tool";
        case ActionType::information_seeking: return "information_seeking";
        case ActionType::verification: return "verification";
        case ActionType::reversible: return "reversible";
        case ActionType::irreversible: return "irreversible";
        case ActionType::compound_skill: return "compound_skill";
        case ActionType::abstain: return "abstain";
    }
    return "unknown";
}

std::string to_string(const SafetyClass value) {
    switch (value) {
        case SafetyClass::allowed: return "allowed";
        case SafetyClass::conditionally_allowed: return "conditionally_allowed";
        case SafetyClass::high_risk: return "high_risk";
        case SafetyClass::prohibited: return "prohibited";
    }
    return "unknown";
}

std::string to_string(const ToolFailure value) {
    switch (value) {
        case ToolFailure::none: return "none";
        case ToolFailure::unavailable: return "unavailable";
        case ToolFailure::invalid_arguments: return "invalid_arguments";
        case ToolFailure::precondition_failed: return "precondition_failed";
        case ToolFailure::timeout: return "timeout";
        case ToolFailure::noisy_result: return "noisy_result";
        case ToolFailure::permission_denied: return "permission_denied";
        case ToolFailure::unsafe_rejected: return "unsafe_rejected";
        case ToolFailure::internal_error: return "internal_error";
    }
    return "unknown";
}

std::string to_string(const ErrorType value) {
    switch (value) {
        case ErrorType::prediction_mismatch: return "prediction_mismatch";
        case ErrorType::invalid_precondition: return "invalid_precondition";
        case ErrorType::tool_failure: return "tool_failure";
        case ErrorType::argument_error: return "argument_error";
        case ErrorType::stale_belief: return "stale_belief";
        case ErrorType::plan_invalidation: return "plan_invalidation";
        case ErrorType::goal_conflict: return "goal_conflict";
        case ErrorType::resource_exhaustion: return "resource_exhaustion";
        case ErrorType::loop: return "loop";
        case ErrorType::no_progress: return "no_progress";
        case ErrorType::unsafe_action_rejection: return "unsafe_action_rejection";
        case ErrorType::false_completion: return "false_completion";
    }
    return "unknown";
}

std::string to_string(const MemoryClass value) {
    switch (value) {
        case MemoryClass::working: return "working";
        case MemoryClass::episodic: return "episodic";
        case MemoryClass::semantic: return "semantic";
        case MemoryClass::skill: return "skill";
        case MemoryClass::safety: return "safety";
    }
    return "unknown";
}

std::string to_string(const PlanningPolicy value) {
    switch (value) {
        case PlanningPolicy::reactive_one_step: return "reactive_one_step";
        case PlanningPolicy::greedy_progress: return "greedy_progress";
        case PlanningPolicy::bounded_best_first: return "bounded_best_first";
        case PlanningPolicy::bounded_astar: return "bounded_astar";
        case PlanningPolicy::receding_horizon: return "receding_horizon";
        case PlanningPolicy::skill_guided: return "skill_guided";
        case PlanningPolicy::uncertainty_aware: return "uncertainty_aware";
    }
    return "unknown";
}

}  // namespace rlf::agent
