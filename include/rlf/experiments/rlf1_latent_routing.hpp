#pragma once

#include "rlf/core/latent_routing.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

namespace rlf::experiments {

struct Rlf1Config final {
    std::uint64_t seed{0x524C4631ULL};
    std::size_t dimension{24U};
    std::size_t training_episodes{64U};
    std::size_t development_episodes{24U};
    std::size_t evaluation_episodes{48U};
    std::size_t training_min_route_length{1U};
    std::size_t training_max_route_length{4U};
    std::size_t evaluation_min_route_length{5U};
    std::size_t evaluation_max_route_length{8U};
    std::size_t maximum_cycles{16U};
    std::size_t operator_count{8U};
    double state_noise_radians{0.03};
    double goal_similarity_threshold{0.9995};
};

struct Rlf1SystemMetrics final {
    std::string name;
    std::size_t episodes{};
    double final_state_accuracy{};
    double exact_route_accuracy{};
    double valid_route_accuracy{};
    double first_action_accuracy{};
    double mean_goal_similarity{};
    double average_cycles{};
    double average_optimal_route_length{};
    double average_excess_steps{};
    double premature_halt_rate{};
    double halt_precision{};
    double halt_recall{};
    double abstention_rate{};
    double selective_accuracy{};
    double mean_uncertainty{};
    double expected_calibration_error{};
    double per_step_operator_accuracy{};
    double average_exact_similarities{};
    double average_active_operations{};
    double average_search_nodes{};
    double inference_seconds{};
};

struct Rlf1LengthMetrics final {
    std::size_t route_length{};
    std::size_t episodes{};
    double rlf_accuracy{};
    double rlf_search_accuracy{};
    double greedy_accuracy{};
    double supervised_accuracy{};
    double oracle_accuracy{};
};

struct Rlf1HaltMetrics final {
    std::string policy;
    double accuracy{};
    double halt_precision{};
    double halt_recall{};
    double premature_halt_rate{};
    double average_cycles{};
};

struct Rlf1CreditMetrics final {
    std::string strategy;
    double accuracy{};
    double mean_goal_similarity{};
    std::size_t mode_count{};
};

struct Rlf1ScalingMetrics final {
    std::size_t training_episodes{};
    std::size_t physical_modes{};
    double accuracy{};
    double average_exact_similarities{};
    double average_active_operations{};
};

struct Rlf1LeakageAudit final {
    bool no_context_prefix{true};
    bool no_operator_labels_in_evaluation{true};
    bool no_route_length_in_evaluation{true};
    bool no_complete_route_overlap{true};
    bool no_exact_start_goal_overlap{true};
    bool no_target_access_during_execution{true};
    bool no_oracle_data_in_rlf_execution{true};
    bool no_result_cache{true};
    bool no_seed_overlap{true};
    bool no_training_evaluation_length_overlap{true};
    std::size_t training_route_hashes{};
    std::size_t development_route_hashes{};
    std::size_t evaluation_route_hashes{};
    std::size_t route_hash_overlap{};
    std::size_t start_goal_hash_overlap{};
    std::uint64_t manifest_hash{};
};

struct Rlf1Result final {
    std::uint64_t seed{};
    std::size_t dimension{};
    std::size_t training_episodes{};
    std::size_t development_episodes{};
    std::size_t evaluation_episodes{};
    Rlf1SystemMetrics rlf;
    Rlf1SystemMetrics development_search_selection;
    Rlf1SystemMetrics rlf_search_assisted;
    Rlf1SystemMetrics rlf_without_route_memory;
    Rlf1SystemMetrics greedy;
    Rlf1SystemMetrics nearest_route;
    Rlf1SystemMetrics supervised;
    Rlf1SystemMetrics oracle;
    std::vector<Rlf1LengthMetrics> by_length;
    std::vector<Rlf1HaltMetrics> halt_policies;
    std::vector<Rlf1CreditMetrics> delayed_credit;
    std::vector<Rlf1ScalingMetrics> scaling;
    double noise_robustness_accuracy{};
    double recovery_accuracy{};
    double recovery_search_accuracy{};
    double ambiguity_abstention_rate{};
    double ambiguity_search_abstention_rate{};
    double impossible_case_false_success_rate{};
    double continual_retained_accuracy{};
    double continual_forgetting{};
    std::size_t macros_proposed{};
    std::size_t macros_created{};
    std::size_t macros_rejected{};
    double macro_validation_accuracy{};
    double macro_cycle_reduction{};
    double macro_interference{};
    std::size_t routing_mode_count{};
    std::size_t halt_mode_count{};
    std::size_t route_memory_records{};
    std::size_t operator_count{};
    std::size_t estimated_bytes{};
    std::size_t local_update_operations{};
    double training_seconds{};
    std::uint64_t deterministic_run_hash{};
    Rlf1LeakageAudit leakage_audit;
    std::size_t selected_lookahead_depth{};
    std::size_t selected_beam_width{};
    std::vector<std::uint64_t> training_route_hash_values;
    std::vector<std::uint64_t> development_route_hash_values;
    std::vector<std::uint64_t> evaluation_route_hash_values;
    std::vector<std::uint64_t> training_start_goal_hash_values;
    std::vector<std::uint64_t> development_start_goal_hash_values;
    std::vector<std::uint64_t> evaluation_start_goal_hash_values;
    std::string scientific_decision;
    std::vector<std::string> limitations;
    core::LatentExecutionResult representative_trace;
};


struct Rlf1TrainingWorkflowResult final {
    std::filesystem::path checkpoint_path;
    std::uint64_t seed{};
    std::size_t training_episodes{};
    std::size_t successful_episodes{};
    std::size_t routing_modes{};
    std::size_t halt_modes{};
    std::size_t route_records{};
    std::size_t operators{};
    std::uint64_t deterministic_run_hash{};
};

struct Rlf1EvaluationWorkflowResult final {
    std::filesystem::path checkpoint_path;
    std::size_t evaluation_episodes{};
    Rlf1SystemMetrics metrics;
    std::uint64_t deterministic_run_hash{};
};

struct Rlf1TraceWorkflowResult final {
    std::filesystem::path checkpoint_path;
    std::size_t sample_id{};
    std::uint64_t route_hash{};
    core::LatentExecutionResult execution;
};

[[nodiscard]] Rlf1TrainingWorkflowResult train_rlf1_checkpoint(
    const Rlf1Config& config,
    const std::filesystem::path& checkpoint_path
);
[[nodiscard]] Rlf1EvaluationWorkflowResult evaluate_rlf1_checkpoint(
    const std::filesystem::path& checkpoint_path,
    std::uint64_t seed,
    std::size_t evaluation_episodes
);
[[nodiscard]] Rlf1TraceWorkflowResult trace_rlf1_checkpoint(
    const std::filesystem::path& checkpoint_path,
    std::uint64_t seed,
    std::size_t sample_id
);
void write_rlf1_training_json(
    std::ostream& output,
    const Rlf1TrainingWorkflowResult& result
);
void write_rlf1_evaluation_json(
    std::ostream& output,
    const Rlf1EvaluationWorkflowResult& result
);
void write_rlf1_trace_json(
    std::ostream& output,
    const Rlf1TraceWorkflowResult& result
);

[[nodiscard]] Rlf1Result run_rlf1_latent_routing(
    const Rlf1Config& config
);
void write_rlf1_latent_routing_json(
    std::ostream& output,
    const Rlf1Result& result
);

}  // namespace rlf::experiments
