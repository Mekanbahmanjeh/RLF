#pragma once

#include "rlf/core/predictive_skill_fabric.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

namespace rlf::experiments {

struct Rlf2Config final {
    std::uint64_t seed{0x524C4632ULL};
    std::size_t dimension{24U};
    std::size_t training_episodes{96U};
    std::size_t development_episodes{24U};
    std::size_t evaluation_episodes{48U};
    std::size_t training_min_route_length{1U};
    std::size_t training_max_route_length{4U};
    std::size_t evaluation_min_route_length{5U};
    std::size_t evaluation_max_route_length{8U};
    std::size_t maximum_cycles{20U};
    std::size_t operator_count{8U};
    double state_noise_radians{0.03};
    double goal_similarity_threshold{0.9995};
};

struct Rlf2SystemMetrics final {
    std::string name;
    std::size_t episodes{};
    double final_state_accuracy{};
    double exact_route_accuracy{};
    double first_action_accuracy{};
    double mean_goal_similarity{};
    double average_cycles{};
    double average_primitive_steps{};
    double average_planner_nodes{};
    double average_forward_nodes{};
    double average_backward_nodes{};
    double average_subgoals{};
    double abstention_rate{};
    double mean_uncertainty{};
    double inference_seconds{};
};

struct Rlf2LengthMetrics final {
    std::size_t route_length{};
    std::size_t episodes{};
    double autonomous_accuracy{};
    double bridge_accuracy{};
    double untrained_bridge_accuracy{};
    double rlf1_accuracy{};
    double greedy_accuracy{};
};

struct Rlf2LeakageAudit final {
    bool no_context_prefix{true};
    bool no_operator_labels_in_evaluation{true};
    bool no_route_length_in_evaluation{true};
    bool no_complete_route_overlap{true};
    bool no_exact_start_goal_overlap{true};
    bool no_target_access_during_execution{true};
    bool no_evaluation_route_training{true};
    std::size_t training_routes{};
    std::size_t development_routes{};
    std::size_t evaluation_routes{};
    std::size_t route_overlap{};
    std::size_t start_goal_overlap{};
    std::uint64_t manifest_hash{};
    std::vector<std::uint64_t> training_route_hashes;
    std::vector<std::uint64_t> development_route_hashes;
    std::vector<std::uint64_t> evaluation_route_hashes;
    std::vector<std::uint64_t> training_start_goal_hashes;
    std::vector<std::uint64_t> development_start_goal_hashes;
    std::vector<std::uint64_t> evaluation_start_goal_hashes;
};

struct Rlf2Result final {
    std::uint64_t seed{};
    std::size_t dimension{};
    std::size_t training_episodes{};
    std::size_t evaluation_episodes{};
    Rlf2SystemMetrics rlf2_autonomous;
    Rlf2SystemMetrics rlf2_subgoal_bridge;
    Rlf2SystemMetrics untrained_bidirectional_bridge;
    Rlf2SystemMetrics rlf1_autonomous;
    Rlf2SystemMetrics greedy;
    Rlf2SystemMetrics oracle;
    std::vector<Rlf2LengthMetrics> by_length;
    double recovery_accuracy{};
    double impossible_false_success_rate{};
    double impossible_abstention_rate{};
    double route_cycle_compression{};
    double skill_validation_accuracy{};
    double mean_causal_advantage{};
    std::size_t operator_count{};
    std::size_t skill_count{};
    std::size_t compound_skill_count{};
    std::size_t prototype_count{};
    std::size_t estimated_bytes{};
    core::Rlf2TrainingStats training_stats;
    Rlf2LeakageAudit leakage_audit;
    double training_seconds{};
    std::uint64_t deterministic_run_hash{};
    std::string scientific_decision;
    std::vector<std::string> limitations;
    core::Rlf2ExecutionResult representative_trace;
};

struct Rlf2TrainingWorkflowResult final {
    std::filesystem::path checkpoint_path;
    std::uint64_t seed{};
    std::size_t successful_training_episodes{};
    std::size_t operators{};
    std::size_t skills{};
    std::size_t prototypes{};
    std::uint64_t deterministic_run_hash{};
};

struct Rlf2EvaluationWorkflowResult final {
    std::filesystem::path checkpoint_path;
    std::size_t evaluation_episodes{};
    Rlf2SystemMetrics autonomous;
    Rlf2SystemMetrics subgoal_bridge;
    std::uint64_t deterministic_run_hash{};
};

struct Rlf2TraceWorkflowResult final {
    std::filesystem::path checkpoint_path;
    std::size_t sample_id{};
    core::Rlf2ExecutionResult autonomous;
    core::Rlf2ExecutionResult subgoal_bridge;
};

[[nodiscard]] Rlf2Result run_rlf2_predictive_reasoning(
    const Rlf2Config& config
);
void write_rlf2_predictive_reasoning_json(
    std::ostream& output,
    const Rlf2Result& result
);

[[nodiscard]] Rlf2TrainingWorkflowResult train_rlf2_checkpoint(
    const Rlf2Config& config,
    const std::filesystem::path& checkpoint_path
);
[[nodiscard]] Rlf2EvaluationWorkflowResult evaluate_rlf2_checkpoint(
    const std::filesystem::path& checkpoint_path,
    std::uint64_t seed,
    std::size_t evaluation_episodes
);
[[nodiscard]] Rlf2TraceWorkflowResult trace_rlf2_checkpoint(
    const std::filesystem::path& checkpoint_path,
    std::uint64_t seed,
    std::size_t sample_id
);
void write_rlf2_training_json(
    std::ostream& output,
    const Rlf2TrainingWorkflowResult& result
);
void write_rlf2_evaluation_json(
    std::ostream& output,
    const Rlf2EvaluationWorkflowResult& result
);
void write_rlf2_trace_json(
    std::ostream& output,
    const Rlf2TraceWorkflowResult& result
);

}  // namespace rlf::experiments
