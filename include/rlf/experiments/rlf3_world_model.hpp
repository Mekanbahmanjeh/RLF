#pragma once

#include "rlf/core/sparse_world_model.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

namespace rlf::experiments {

struct Rlf3Config final {
    std::uint64_t seed{0x524C4633ULL};
    std::size_t dimension{32U};
    std::size_t layers{7U};
    std::size_t lanes{5U};
    std::size_t transition_samples_per_case{12U};
    std::size_t training_routes{240U};
    std::size_t evaluation_episodes{120U};
    std::size_t stochastic_rollouts_per_episode{3U};
    std::size_t maximum_execution_steps{20U};
    std::size_t planner_node_budget{50'000U};
    std::size_t maximum_plan_depth{16U};
    double observation_noise_radians{0.012};
    double stochastic_dominant_probability{0.82};
    double goal_similarity_threshold{0.999};
};

struct Rlf3PlannerMetrics final {
    std::string name;
    std::size_t episodes{};
    double success_rate{};
    double mean_goal_similarity{};
    double average_execution_steps{};
    double average_replans{};
    double average_planner_nodes{};
    double average_transition_evaluations{};
    double average_subgoal_queries{};
    double average_subgoal_comparisons{};
    double planning_failure_rate{};
    double trap_rate{};
    double inference_seconds{};
};

struct Rlf3PredictionMetrics final {
    std::size_t experiences{};
    double top1_accuracy{};
    double negative_log_likelihood{};
    double brier_score{};
    double state_match_rate{};
};

struct Rlf3LeakageAudit final {
    bool no_exact_action_models_in_fabric{true};
    bool no_evaluation_start_goal_training{true};
    bool no_evaluation_route_training{true};
    bool no_hidden_mode_in_observation{true};
    bool no_target_access_during_execution{true};
    bool transition_holdout_disjoint{true};
    std::size_t training_start_goal_pairs{};
    std::size_t evaluation_start_goal_pairs{};
    std::size_t start_goal_overlap{};
    std::size_t route_overlap{};
    std::uint64_t transition_manifest_hash{};
    std::uint64_t route_manifest_hash{};
    std::vector<std::uint64_t> training_pair_hashes;
    std::vector<std::uint64_t> evaluation_pair_hashes;
    std::vector<std::uint64_t> training_route_hashes;
    std::vector<std::uint64_t> evaluation_route_hashes;
};

struct Rlf3Result final {
    std::uint64_t seed{};
    std::size_t dimension{};
    std::size_t physical_world_states{};
    std::size_t learned_state_prototypes{};
    std::size_t learned_context_prototypes{};
    std::size_t learned_transitions{};
    std::size_t learned_outcomes{};
    std::size_t learned_subgoals{};
    Rlf3PredictionMetrics prediction;
    Rlf3PlannerMetrics indexed_receding;
    Rlf3PlannerMetrics flat_receding;
    Rlf3PlannerMetrics memoryless_receding;
    Rlf3PlannerMetrics greedy;
    Rlf3PlannerMetrics oracle;
    double planner_node_reduction{};
    double transition_compression_ratio{};
    double partial_observation_gain{};
    double stochastic_success_rate{};
    double irreversible_success_rate{};
    double impossible_false_success_rate{};
    std::size_t estimated_model_bytes{};
    core::SparseWorldModelStats training_stats;
    Rlf3LeakageAudit leakage_audit;
    double training_seconds{};
    std::uint64_t deterministic_run_hash{};
    std::string scientific_decision;
    std::vector<std::string> limitations;
    core::Rlf3PlanResult representative_plan;
};

struct Rlf3TrainingWorkflowResult final {
    std::filesystem::path checkpoint_path;
    std::uint64_t seed{};
    std::size_t transition_experiences{};
    std::size_t training_routes{};
    std::size_t states{};
    std::size_t contexts{};
    std::size_t transitions{};
    std::size_t subgoals{};
    std::uint64_t deterministic_run_hash{};
};

struct Rlf3EvaluationWorkflowResult final {
    std::filesystem::path checkpoint_path;
    std::size_t evaluation_episodes{};
    Rlf3PlannerMetrics indexed_receding;
    Rlf3PlannerMetrics flat_receding;
    std::uint64_t deterministic_run_hash{};
};

struct Rlf3TraceWorkflowResult final {
    std::filesystem::path checkpoint_path;
    std::size_t sample_id{};
    core::Rlf3PlanResult indexed_plan;
    core::Rlf3PlanResult flat_plan;
};

[[nodiscard]] Rlf3Result run_rlf3_world_model(const Rlf3Config& config);
void write_rlf3_world_model_json(std::ostream& output, const Rlf3Result& result);

[[nodiscard]] Rlf3TrainingWorkflowResult train_rlf3_checkpoint(
    const Rlf3Config& config,
    const std::filesystem::path& checkpoint_path
);
[[nodiscard]] Rlf3EvaluationWorkflowResult evaluate_rlf3_checkpoint(
    const std::filesystem::path& checkpoint_path,
    std::uint64_t seed,
    std::size_t evaluation_episodes
);
[[nodiscard]] Rlf3TraceWorkflowResult trace_rlf3_checkpoint(
    const std::filesystem::path& checkpoint_path,
    std::uint64_t seed,
    std::size_t sample_id
);
void write_rlf3_training_json(
    std::ostream& output,
    const Rlf3TrainingWorkflowResult& result
);
void write_rlf3_evaluation_json(
    std::ostream& output,
    const Rlf3EvaluationWorkflowResult& result
);
void write_rlf3_trace_json(
    std::ostream& output,
    const Rlf3TraceWorkflowResult& result
);

}  // namespace rlf::experiments
