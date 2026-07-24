#pragma once

#include "rlf/agent/agent_fabric.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace rlf::experiments {

struct Rlf6Config final {
    std::uint64_t seed{0x524C4636ULL};
    std::size_t training_episodes{120U};
    std::size_t evaluation_episodes{80U};
    std::size_t minimum_route_length{5U};
    std::size_t maximum_route_length{100U};
    std::size_t stress_episodes{8U};
    std::size_t stress_route_length{120U};
    std::size_t action_budget{320U};
    std::size_t planning_node_budget{50'000U};
    std::size_t tool_budget{128U};
    std::size_t memory_limit_records{32'768U};
    double tool_cost_budget{10'000.0};
    double risk_budget{25.0};
    std::size_t threads{1U};
    bool deterministic{true};
    bool include_stress{true};
    std::string experiment_name{"long_horizon_planning"};
};

struct Rlf6TaskMetrics final {
    std::size_t episodes{};
    std::size_t successes{};
    std::size_t partial_completions{};
    std::size_t false_completions{};
    std::size_t impossible_goals{};
    std::size_t impossible_goals_recognized{};
    std::size_t actions{};
    std::size_t optimal_actions{};
    std::size_t excess_actions{};
    std::size_t plans_generated{};
    std::size_t replans{};
    std::size_t planning_nodes{};
    std::size_t maximum_search_depth{};
    double mean_branch_factor{};
    double planning_seconds{};
    double task_success_rate{};
    double mean_partial_completion{};
};

struct Rlf6GoalMetrics final {
    std::size_t goals_created{};
    std::size_t goals_completed{};
    std::size_t goals_abandoned{};
    std::size_t distraction_failures{};
    std::size_t wrong_goal_substitutions{};
    std::size_t conflicts_present{};
    std::size_t conflicts_detected{};
    double retention_rate{};
};

struct Rlf6ToolMetrics final {
    std::size_t calls{};
    std::size_t correct_selection{};
    std::size_t valid_arguments{};
    std::size_t useful_calls{};
    std::size_t wasted_calls{};
    std::size_t failures{};
    std::size_t retries{};
    std::size_t verification_calls{};
    std::size_t repeated_useless_calls{};
    double total_cost{};
    double success_gain_from_tools{};
};

struct Rlf6CorrectionMetrics final {
    std::size_t failures_detected{};
    std::size_t failures_recovered{};
    std::size_t repeated_failures{};
    std::size_t recovery_steps{};
    std::size_t alternatives_evaluated{};
    std::size_t wrong_counterfactuals{};
    double recovery_rate{};
    double correction_cost{};
    double causal_credit_accuracy{};
};

struct Rlf6UncertaintyMetrics final {
    std::size_t predictions{};
    std::size_t confident_errors{};
    std::size_t abstentions{};
    std::size_t unnecessary_abstentions{};
    std::size_t information_actions{};
    double expected_calibration_error{};
    double brier_score{};
    double selective_accuracy{};
    double risk_coverage_area{};
};

struct Rlf6MemoryMetrics final {
    std::size_t reads{};
    std::size_t writes{};
    std::size_t hits{};
    std::size_t stale_hits{};
    std::size_t records{};
    std::size_t bytes{};
    std::size_t interference_events{};
    double retrieval_precision{};
};

struct Rlf6SkillMetrics final {
    std::size_t proposed{};
    std::size_t accepted{};
    std::size_t reused{};
    std::size_t transferred{};
    std::size_t rejected{};
    std::size_t harmful{};
    std::size_t invalidated{};
    double execution_reduction{};
    double planning_reduction{};
};

struct Rlf6SafetyMetrics final {
    std::size_t adversarial_episodes{};
    std::size_t attacks_succeeded{};
    std::size_t unsafe_actions_attempted{};
    std::size_t unsafe_actions_executed{};
    std::size_t goal_hijacks{};
    std::size_t false_success_signals{};
    std::size_t false_success_signals_rejected{};
    std::size_t safety_rejections{};
    double attack_success_rate{};
    double unsafe_action_rate{};
};

struct Rlf6EfficiencyMetrics final {
    std::size_t active_modes_per_cycle_peak{};
    double active_modes_per_cycle_mean{};
    std::size_t retrieved_candidates{};
    std::size_t exact_similarities{};
    std::size_t reasoning_cycles{};
    std::uint64_t total_operations{};
    std::uint64_t bytes_read{};
    std::uint64_t bytes_written{};
    std::size_t peak_rss_bytes{};
    std::size_t persistent_bytes{};
    double wall_seconds{};
    std::uint64_t deterministic_hash{};
};

struct Rlf6LengthBucket final {
    std::string name;
    std::size_t minimum_length{};
    std::size_t maximum_length{};
    std::size_t episodes{};
    std::size_t successes{};
    double success_rate{};
    double mean_actions{};
    double mean_planning_nodes{};
};

struct Rlf6FamilyResult final {
    std::string family;
    std::size_t episodes{};
    std::size_t successes{};
    double success_rate{};
    double mean_actions{};
    double mean_tool_calls{};
    double mean_replans{};
};

struct Rlf6BaselineResult final {
    std::string name;
    std::string category;
    std::string supervision;
    bool oracle{false};
    std::size_t episodes{};
    std::size_t successes{};
    double success_rate{};
    double mean_actions{};
    double mean_tool_calls{};
    double mean_planning_nodes{};
    std::size_t memory_bytes{};
};

struct Rlf6LeakageAudit final {
    bool no_optimal_route_in_task_input{true};
    bool no_hidden_next_action_labels{true};
    bool no_task_id_plan_mapping{true};
    bool no_route_length_in_agent_input{true};
    bool no_oracle_state_in_beliefs{true};
    bool no_ground_truth_hidden_state_exposed{true};
    bool no_evaluator_result_before_completion{true};
    bool evaluation_routes_absent_from_memory{true};
    bool no_oracle_tool_reliability{true};
    bool no_target_answer_during_tool_selection{true};
    bool search_reported_separately{true};
    bool tool_text_never_executed{true};
    bool no_cached_success{true};
    std::size_t route_hash_overlap{};
    std::size_t environment_hash_overlap{};
    std::size_t goal_hash_overlap{};
    std::size_t tool_pattern_hash_overlap{};
    std::uint64_t training_manifest_hash{};
    std::uint64_t evaluation_manifest_hash{};
};

struct Rlf6Projection final {
    std::size_t gpu_memory_gb{};
    std::uint64_t projected_hot_modes{};
    std::uint64_t projected_hot_working_set_bytes{};
    std::uint64_t addressable_mode_capacity{};
    std::uint64_t external_state_bytes{};
};

struct Rlf6Result final {
    Rlf6Config config;
    Rlf6TaskMetrics task;
    Rlf6GoalMetrics goals;
    Rlf6ToolMetrics tools;
    Rlf6CorrectionMetrics correction;
    Rlf6UncertaintyMetrics uncertainty;
    Rlf6MemoryMetrics memory;
    Rlf6SkillMetrics skills;
    Rlf6SafetyMetrics safety;
    Rlf6EfficiencyMetrics efficiency;
    Rlf6LeakageAudit leakage_audit;
    std::vector<Rlf6LengthBucket> length_buckets;
    std::vector<Rlf6FamilyResult> families;
    std::vector<Rlf6BaselineResult> baselines;
    std::vector<Rlf6Projection> gpu_projections;
    double continual_retention{};
    double backward_transfer{};
    double forward_transfer{};
    std::string scientific_decision;
    bool rlf7_scientifically_justified{false};
    std::vector<std::string> limitations;
    std::string interpretation;
};

struct Rlf6TrainingWorkflowResult final {
    std::filesystem::path checkpoint_path;
    std::uint64_t seed{};
    std::size_t episodes{};
    std::size_t tools{};
    std::size_t transitions{};
    std::size_t skills{};
    std::size_t memory_records{};
    std::uint64_t deterministic_hash{};
};

struct Rlf6EvaluationWorkflowResult final {
    std::filesystem::path checkpoint_path;
    Rlf6TaskMetrics task;
    Rlf6ToolMetrics tools;
    Rlf6CorrectionMetrics correction;
    Rlf6UncertaintyMetrics uncertainty;
    Rlf6SafetyMetrics safety;
    std::uint64_t deterministic_hash{};
};

struct Rlf6TraceStep final {
    std::uint64_t episode_id{};
    std::size_t step{};
    std::uint64_t observation_hash{};
    std::uint64_t belief_hash{};
    std::vector<std::uint64_t> active_goal_ids;
    std::vector<std::uint64_t> proposed_subgoals;
    std::vector<std::uint64_t> retrieved_modes;
    std::vector<std::uint64_t> candidate_actions;
    std::vector<double> candidate_scores;
    double uncertainty{};
    std::uint64_t selected_action{};
    std::uint64_t tool_call{};
    std::string predicted_result;
    std::string actual_result;
    double prediction_error{};
    std::size_t memory_reads{};
    std::size_t memory_writes{};
    double progress{};
    agent::ResourceState resources;
    std::string verification_state;
    std::string correction_state;
    std::string halt_decision;
    std::string safety_decision;
};

struct Rlf6TraceWorkflowResult final {
    std::filesystem::path checkpoint_path;
    std::size_t sample_id{};
    std::string family;
    std::size_t nominal_route_length{};
    bool success{false};
    std::vector<Rlf6TraceStep> steps;
    std::uint64_t deterministic_hash{};
};

[[nodiscard]] Rlf6Result run_rlf6_agent(const Rlf6Config& config);
void write_rlf6_result_json(std::ostream& output, const Rlf6Result& result);

[[nodiscard]] Rlf6TrainingWorkflowResult train_rlf6_checkpoint(
    const Rlf6Config& config,
    const std::filesystem::path& checkpoint_path
);
[[nodiscard]] Rlf6EvaluationWorkflowResult evaluate_rlf6_checkpoint(
    const std::filesystem::path& checkpoint_path,
    const Rlf6Config& config
);
[[nodiscard]] Rlf6TraceWorkflowResult trace_rlf6_checkpoint(
    const std::filesystem::path& checkpoint_path,
    const Rlf6Config& config,
    std::size_t sample_id
);

void write_rlf6_training_json(
    std::ostream& output,
    const Rlf6TrainingWorkflowResult& result
);
void write_rlf6_evaluation_json(
    std::ostream& output,
    const Rlf6EvaluationWorkflowResult& result
);
void write_rlf6_trace_json(
    std::ostream& output,
    const Rlf6TraceWorkflowResult& result
);

[[nodiscard]] bool is_rlf6_experiment_name(std::string_view name) noexcept;

}  // namespace rlf::experiments
