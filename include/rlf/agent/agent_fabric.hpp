#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rlf::agent {

enum class EvidenceKind : std::uint8_t {
    observation,
    belief,
    hypothesis,
    goal,
    prediction,
    tool_output,
    verified_fact,
    stale_information,
};

enum class GoalStatus : std::uint8_t {
    pending,
    active,
    completed,
    failed,
    impossible,
    conflicted,
    abandoned,
};

enum class ActionType : std::uint8_t {
    primitive,
    tool,
    information_seeking,
    verification,
    reversible,
    irreversible,
    compound_skill,
    abstain,
};

enum class SafetyClass : std::uint8_t {
    allowed,
    conditionally_allowed,
    high_risk,
    prohibited,
};

enum class ToolFailure : std::uint8_t {
    none,
    unavailable,
    invalid_arguments,
    precondition_failed,
    timeout,
    noisy_result,
    permission_denied,
    unsafe_rejected,
    internal_error,
};

enum class ErrorType : std::uint8_t {
    prediction_mismatch,
    invalid_precondition,
    tool_failure,
    argument_error,
    stale_belief,
    plan_invalidation,
    goal_conflict,
    resource_exhaustion,
    loop,
    no_progress,
    unsafe_action_rejection,
    false_completion,
};

enum class MemoryClass : std::uint8_t {
    working,
    episodic,
    semantic,
    skill,
    safety,
};

enum class PlanningPolicy : std::uint8_t {
    reactive_one_step,
    greedy_progress,
    bounded_best_first,
    bounded_astar,
    receding_horizon,
    skill_guided,
    uncertainty_aware,
};

struct Fact final {
    std::string key;
    std::string value;
    bool negated{false};

    [[nodiscard]] std::string canonical() const;
    [[nodiscard]] bool operator==(const Fact&) const = default;
    [[nodiscard]] bool operator<(const Fact& other) const;
};

struct EvidenceRecord final {
    std::uint64_t stable_id{};
    Fact fact;
    EvidenceKind kind{EvidenceKind::observation};
    double confidence{1.0};
    double source_reliability{1.0};
    std::uint64_t creation_step{};
    std::uint64_t last_update_step{};
    std::string provenance;
    bool verified{false};
    bool stale{false};
};

struct BeliefHypothesis final {
    std::uint64_t stable_id{};
    Fact fact;
    double support{};
    std::uint64_t contradiction_count{};
    std::uint64_t recency{};
    double source_reliability{1.0};
    double uncertainty{1.0};
    bool verified{false};
    bool stale{false};
};

struct Goal final {
    std::uint64_t stable_id{};
    std::string specification;
    std::vector<Fact> completion_conditions;
    std::vector<Fact> failure_conditions;
    std::vector<std::uint64_t> dependencies;
    double priority{1.0};
    std::uint64_t creation_step{};
    std::uint64_t deadline_step{};
    std::uint64_t resource_budget{};
    GoalStatus status{GoalStatus::pending};
    double confidence{1.0};
    std::string provenance;
    bool optional{false};
    bool ordered{false};
};

struct Action final {
    std::uint64_t stable_id{};
    ActionType action_type{ActionType::primitive};
    std::string name;
    std::vector<std::pair<std::string, std::string>> parameters;
    std::vector<Fact> preconditions;
    std::vector<Fact> expected_effects;
    double estimated_cost{1.0};
    double estimated_risk{};
    double confidence{0.5};
    std::string provenance;
    std::uint64_t tool_id{};
    SafetyClass safety_class{SafetyClass::allowed};
    bool reversible{true};
    bool information_value{false};
};

struct Plan final {
    std::uint64_t goal_id{};
    std::vector<std::uint64_t> action_ids;
    std::vector<std::uint64_t> predicted_state_hashes;
    double predicted_cost{};
    double predicted_success{};
    double uncertainty{};
    std::vector<std::size_t> verification_points;
    std::vector<std::vector<std::uint64_t>> fallback_branches;
    std::uint64_t resource_budget{};
    std::size_t nodes_expanded{};
    std::size_t maximum_depth{};
    bool found{false};
};

struct ToolDefinition final {
    std::uint64_t stable_id{};
    std::string name;
    std::vector<std::string> input_schema;
    std::vector<std::string> output_schema;
    std::vector<Fact> preconditions;
    bool has_side_effects{false};
    bool reversible{true};
    double cost{1.0};
    std::uint64_t latency{};
    std::vector<ToolFailure> failure_modes;
    double declared_reliability{1.0};
    SafetyClass safety_class{SafetyClass::allowed};
    std::string required_permission;
};

struct ToolResult final {
    std::uint64_t tool_id{};
    ToolFailure failure{ToolFailure::none};
    std::vector<Fact> structured_output;
    std::string untrusted_text;
    double reported_confidence{1.0};
    double cost{};
    std::uint64_t latency{};
    bool side_effect_applied{false};
};

struct ToolReliability final {
    std::uint64_t tool_id{};
    std::string context;
    double successes{1.0};
    double failures{1.0};
    double recent_successes{1.0};
    double recent_failures{1.0};
    std::uint64_t last_update_step{};

    [[nodiscard]] double mean() const noexcept;
    [[nodiscard]] double uncertainty() const noexcept;
};

struct TransitionOutcome final {
    std::vector<Fact> effects;
    double count{};
    double recent_count{};
    double total_cost{};
    std::uint64_t terminal_failures{};
};

struct TransitionRecord final {
    std::string context_key;
    std::string action_signature;
    std::vector<TransitionOutcome> outcomes;
    std::uint64_t observations{};
    std::uint64_t model_version{};
    double surprise_ema{};
    double reliability{0.5};
};

struct MemoryRecord final {
    std::uint64_t stable_id{};
    MemoryClass memory_class{MemoryClass::working};
    std::string key;
    std::string payload;
    double confidence{1.0};
    double utility{};
    std::string provenance;
    std::uint64_t creation_step{};
    std::uint64_t last_use_step{};
    bool verified{false};
    bool invalidated{false};
};

struct Skill final {
    std::uint64_t stable_id{};
    std::string goal_pattern;
    std::vector<Fact> triggering_conditions;
    std::vector<std::uint64_t> action_sequence;
    std::vector<std::size_t> verification_points;
    std::vector<Fact> failure_conditions;
    std::vector<std::uint64_t> fallback_actions;
    double estimated_cost{};
    double confidence{0.5};
    double utility{};
    std::uint64_t support{};
    std::uint64_t successful_reuses{};
    std::uint64_t failed_reuses{};
    bool invalidated{false};
};

struct ErrorEvent final {
    std::uint64_t stable_id{};
    ErrorType type{ErrorType::prediction_mismatch};
    std::uint64_t action_id{};
    std::uint64_t step{};
    std::string context;
    std::string expected;
    std::string actual;
    double severity{};
    bool recovered{false};
};

struct ResourceState final {
    std::uint64_t reasoning_cycles{};
    std::uint64_t action_count{};
    std::uint64_t tool_calls{};
    double tool_cost{};
    std::uint64_t memory_reads{};
    std::uint64_t memory_writes{};
    std::uint64_t planning_nodes{};
    std::uint64_t simulated_time{};
    double risk_used{};
    std::uint64_t bytes_read{};
    std::uint64_t bytes_written{};
};

struct ResourceBudget final {
    std::uint64_t reasoning_cycles{10'000U};
    std::uint64_t action_count{1'000U};
    std::uint64_t tool_calls{256U};
    double tool_cost{10'000.0};
    std::uint64_t memory_reads{100'000U};
    std::uint64_t memory_writes{100'000U};
    std::uint64_t planning_nodes{100'000U};
    std::uint64_t simulated_time{1'000'000U};
    double risk_budget{100.0};
};

struct UncertaintyState final {
    double belief_uncertainty{1.0};
    double prediction_uncertainty{1.0};
    double action_uncertainty{1.0};
    double tool_uncertainty{1.0};
    double plan_uncertainty{1.0};
    double goal_completion_uncertainty{1.0};
    double memory_uncertainty{1.0};
    double language_uncertainty{1.0};
    double safety_uncertainty{1.0};
};

struct AgentConfig final {
    std::size_t maximum_observations{4'096U};
    std::size_t maximum_beliefs{4'096U};
    std::size_t maximum_goals{512U};
    std::size_t maximum_memory_records{32'768U};
    std::size_t maximum_skills{4'096U};
    std::size_t maximum_errors{8'192U};
    std::size_t maximum_transition_records{65'536U};
    std::size_t maximum_tool_reliability_records{4'096U};
    std::size_t maximum_plan_depth{256U};
    std::size_t maximum_candidate_actions{4'096U};
    double belief_acceptance_threshold{0.55};
    double irreversible_confidence_threshold{0.85};
    double verification_threshold{0.70};
    double skill_mdl_minimum_gain{2.0};
    double recent_decay{0.97};
    double change_detection_threshold{0.45};
};

struct AgentState final {
    std::vector<EvidenceRecord> observation_state;
    std::vector<BeliefHypothesis> belief_state;
    std::vector<Goal> goal_stack;
    std::uint64_t active_goal{};
    std::vector<std::pair<std::uint64_t, std::uint64_t>> subgoal_graph;
    std::vector<MemoryRecord> working_memory;
    std::vector<MemoryRecord> episodic_memory;
    std::vector<MemoryRecord> semantic_memory;
    std::vector<Skill> skill_memory;
    std::vector<MemoryRecord> safety_memory;
    std::vector<ToolDefinition> tool_state;
    std::vector<TransitionRecord> world_model_state;
    std::vector<ToolReliability> tool_reliability;
    UncertaintyState uncertainty_state;
    ResourceState resource_state;
    std::vector<std::uint64_t> action_history_summary;
    std::vector<ErrorEvent> failure_history_summary;
    std::vector<Fact> verified_facts;
    double progress_state{};
    std::uint64_t step_index{};
    std::uint64_t episode_id{};
};

struct CandidateScore final {
    std::uint64_t action_id{};
    double progress{};
    double predicted_success{};
    double cost{};
    double risk{};
    double uncertainty{};
    double utility{};
};

struct PlanningRequest final {
    std::vector<Action> actions;
    std::uint64_t goal_id{};
    PlanningPolicy policy{PlanningPolicy::uncertainty_aware};
    std::size_t node_budget{10'000U};
    std::size_t depth_budget{128U};
    bool allow_skills{true};
    bool use_learned_model{true};
};

struct SafetyDecision final {
    bool allowed{true};
    bool requires_verification{false};
    bool should_abstain{false};
    std::string reason;
    std::optional<std::uint64_t> safer_alternative;
    double confidence{1.0};
};

struct AgentSnapshot final {
    AgentConfig config;
    std::uint64_t seed{};
    std::uint64_t next_id{1U};
    AgentState state;
};

class AgentFabric final {
public:
    explicit AgentFabric(AgentConfig config = {}, std::uint64_t seed = 0x524C4636ULL);

    [[nodiscard]] const AgentConfig& config() const noexcept;
    [[nodiscard]] std::uint64_t seed() const noexcept;
    [[nodiscard]] const AgentState& state() const noexcept;
    [[nodiscard]] AgentState& mutable_state() noexcept;

    std::uint64_t add_goal(Goal goal);
    void set_active_goal(std::uint64_t goal_id);
    [[nodiscard]] bool detect_goal_conflicts();
    [[nodiscard]] bool goal_conditions_satisfied(std::uint64_t goal_id) const;
    [[nodiscard]] bool goal_conditions_impossible(
        std::uint64_t goal_id,
        const std::vector<Action>& actions
    ) const;
    void update_goal_statuses(const std::vector<Action>& actions);

    void ingest_evidence(EvidenceRecord evidence);
    void mark_stale(std::string_view key);
    [[nodiscard]] std::optional<BeliefHypothesis> best_belief(
        std::string_view key
    ) const;
    [[nodiscard]] std::set<std::string> accepted_facts() const;

    void register_tool(ToolDefinition tool);
    [[nodiscard]] const ToolDefinition* find_tool(std::uint64_t tool_id) const;
    [[nodiscard]] bool validate_tool_arguments(
        const Action& action,
        std::string* reason = nullptr
    ) const;
    void update_tool_reliability(
        std::uint64_t tool_id,
        std::string context,
        bool success
    );
    [[nodiscard]] double tool_reliability(
        std::uint64_t tool_id,
        std::string_view context
    ) const;

    [[nodiscard]] SafetyDecision evaluate_safety(
        const Action& action,
        const std::vector<Action>& alternatives
    ) const;

    [[nodiscard]] std::vector<Goal> discover_subgoals(
        std::uint64_t goal_id,
        const std::vector<Action>& actions
    );
    [[nodiscard]] Plan plan(const PlanningRequest& request);
    [[nodiscard]] std::vector<CandidateScore> score_actions(
        const std::vector<Action>& actions,
        std::uint64_t goal_id
    ) const;

    void record_transition(
        const std::string& context_key,
        const Action& action,
        const std::vector<Fact>& actual_effects,
        bool success,
        double cost,
        bool terminal_failure
    );
    [[nodiscard]] double predicted_action_success(
        const Action& action,
        std::string_view context_key
    ) const;
    [[nodiscard]] double predicted_action_uncertainty(
        const Action& action,
        std::string_view context_key
    ) const;

    std::uint64_t record_error(ErrorEvent event);
    void mark_error_recovered(std::uint64_t error_id);
    [[nodiscard]] std::vector<std::uint64_t> counterfactual_actions(
        const std::vector<Action>& actions,
        const Action& failed_action,
        std::uint64_t goal_id,
        std::size_t maximum_alternatives
    ) const;

    std::uint64_t insert_memory(MemoryRecord record);
    [[nodiscard]] std::vector<MemoryRecord> retrieve_memory(
        MemoryClass memory_class,
        std::string_view key,
        std::size_t maximum_records
    );
    void invalidate_memory(std::uint64_t record_id);
    [[nodiscard]] std::optional<std::uint64_t> consolidate_skill(
        std::string goal_pattern,
        const std::vector<std::uint64_t>& action_sequence,
        double baseline_cost,
        double realized_cost,
        bool held_out_improvement
    );

    [[nodiscard]] bool within_budget(const ResourceBudget& budget) const noexcept;
    void advance_step();
    [[nodiscard]] double progress_for_goal(std::uint64_t goal_id) const;
    [[nodiscard]] std::uint64_t deterministic_hash() const noexcept;
    [[nodiscard]] std::size_t estimated_persistent_bytes() const noexcept;

    [[nodiscard]] AgentSnapshot snapshot() const;
    [[nodiscard]] static AgentFabric from_snapshot(const AgentSnapshot& snapshot);

private:
    [[nodiscard]] std::uint64_t allocate_id();
    [[nodiscard]] Goal* find_goal_mutable(std::uint64_t goal_id);
    [[nodiscard]] const Goal* find_goal(std::uint64_t goal_id) const;
    [[nodiscard]] static std::string action_signature(const Action& action);
    [[nodiscard]] static std::string state_key(const std::set<std::string>& facts);
    [[nodiscard]] static bool preconditions_satisfied(
        const std::vector<Fact>& preconditions,
        const std::set<std::string>& facts
    );
    [[nodiscard]] static std::set<std::string> apply_effects(
        std::set<std::string> facts,
        const std::vector<Fact>& effects
    );
    [[nodiscard]] std::size_t goal_satisfied_count(
        const Goal& goal,
        const std::set<std::string>& facts
    ) const;
    void prune_to_bounds();
    void validate_state() const;

    AgentConfig config_;
    std::uint64_t seed_{};
    std::uint64_t next_id_{1U};
    AgentState state_;
};

[[nodiscard]] std::string to_string(EvidenceKind value);
[[nodiscard]] std::string to_string(GoalStatus value);
[[nodiscard]] std::string to_string(ActionType value);
[[nodiscard]] std::string to_string(SafetyClass value);
[[nodiscard]] std::string to_string(ToolFailure value);
[[nodiscard]] std::string to_string(ErrorType value);
[[nodiscard]] std::string to_string(MemoryClass value);
[[nodiscard]] std::string to_string(PlanningPolicy value);

}  // namespace rlf::agent
