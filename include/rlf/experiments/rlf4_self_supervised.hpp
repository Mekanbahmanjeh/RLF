#pragma once

#include "rlf/core/temporal_predictive_fabric.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

namespace rlf::experiments {

struct Rlf4Config final {
    std::uint64_t seed{0x524C4634ULL};
    std::size_t dimension{48U};
    std::size_t symbol_count{18U};
    std::size_t training_tokens{24'000U};
    std::size_t evaluation_tokens{6'000U};
    std::size_t adaptation_tokens{6'000U};
    std::size_t maximum_context_order{12U};
    std::size_t minimum_context_support{2U};
    std::size_t maximum_options{2'048U};
    std::size_t minimum_option_support{6U};
    std::size_t forecast_horizon{8U};
    std::size_t forecast_samples{256U};
    std::size_t change_tolerance{96U};
    double training_noise_radians{0.018};
    double evaluation_noise_radians{0.035};
    double dominant_motif_probability{0.88};
    double prototype_merge_distance{0.10};
    double recent_decay{0.997};
    double recent_weight{0.70};
};

struct Rlf4PredictionMetrics final {
    std::string name;
    std::size_t predictions{};
    double top1_accuracy{};
    double negative_log_likelihood{};
    double perplexity{};
    double brier_score{};
    double mean_uncertainty{};
    double inference_seconds{};
    std::size_t estimated_bytes{};
};

struct Rlf4ForecastMetrics final {
    std::size_t samples{};
    std::size_t horizon{};
    double token_accuracy{};
    double exact_forecast_accuracy{};
    double average_decisions{};
    double decision_reduction{};
    double average_option_uses{};
};

struct Rlf4AdaptationMetrics final {
    double static_accuracy{};
    double adaptive_accuracy{};
    double adaptation_gain{};
    double first_window_accuracy{};
    double final_window_accuracy{};
    std::size_t recovery_tokens{};
    std::size_t true_changes{};
    std::size_t detected_changes{};
    double change_precision{};
    double change_recall{};
    double change_f1{};
};

struct Rlf4RepresentationMetrics final {
    std::size_t learned_prototypes{};
    std::size_t learned_contexts{};
    std::size_t learned_options{};
    double noisy_observation_match_rate{};
    double prototype_compression_ratio{};
    double option_mean_length{};
    double option_mean_confidence{};
};

struct Rlf4LeakageAudit final {
    bool no_hidden_labels_passed{true};
    bool no_reward_or_route_supervision{true};
    bool no_evaluation_updates_before_scoring{true};
    bool train_evaluation_seeds_disjoint{true};
    bool full_streams_distinct{true};
    std::size_t training_tokens{};
    std::size_t evaluation_tokens{};
    std::uint64_t training_stream_hash{};
    std::uint64_t evaluation_stream_hash{};
    std::uint64_t adaptation_stream_hash{};
};

struct Rlf4Result final {
    std::uint64_t seed{};
    std::size_t dimension{};
    std::size_t training_tokens{};
    std::size_t evaluation_tokens{};
    Rlf4PredictionMetrics full_fabric;
    Rlf4PredictionMetrics no_options_ablation;
    Rlf4PredictionMetrics fixed_order_1;
    Rlf4PredictionMetrics fixed_order_4;
    Rlf4PredictionMetrics fixed_order_12;
    Rlf4PredictionMetrics oracle;
    Rlf4ForecastMetrics forecast;
    Rlf4AdaptationMetrics adaptation;
    Rlf4RepresentationMetrics representation;
    Rlf4LeakageAudit leakage_audit;
    core::TemporalFabricStats training_stats;
    std::size_t estimated_model_bytes{};
    double training_seconds{};
    std::uint64_t deterministic_run_hash{};
    std::string scientific_decision;
    std::vector<std::string> limitations;
};

struct Rlf4TrainingWorkflowResult final {
    std::filesystem::path checkpoint_path;
    std::uint64_t seed{};
    std::size_t training_tokens{};
    std::size_t prototypes{};
    std::size_t contexts{};
    std::size_t options{};
    std::uint64_t deterministic_run_hash{};
};

struct Rlf4EvaluationWorkflowResult final {
    std::filesystem::path checkpoint_path;
    Rlf4PredictionMetrics full_fabric;
    Rlf4ForecastMetrics forecast;
    std::uint64_t deterministic_run_hash{};
};

struct Rlf4TraceWorkflowResult final {
    std::filesystem::path checkpoint_path;
    std::size_t sample_id{};
    std::vector<std::uint64_t> history;
    core::TemporalPrediction prediction;
    core::TemporalForecast forecast;
};

[[nodiscard]] Rlf4Result run_rlf4_self_supervised(const Rlf4Config& config);
void write_rlf4_result_json(std::ostream& output, const Rlf4Result& result);

[[nodiscard]] Rlf4TrainingWorkflowResult train_rlf4_checkpoint(
    const Rlf4Config& config,
    const std::filesystem::path& checkpoint_path
);
[[nodiscard]] Rlf4EvaluationWorkflowResult evaluate_rlf4_checkpoint(
    const std::filesystem::path& checkpoint_path,
    std::uint64_t seed,
    std::size_t evaluation_tokens
);
[[nodiscard]] Rlf4TraceWorkflowResult trace_rlf4_checkpoint(
    const std::filesystem::path& checkpoint_path,
    std::uint64_t seed,
    std::size_t sample_id
);
void write_rlf4_training_json(
    std::ostream& output,
    const Rlf4TrainingWorkflowResult& result
);
void write_rlf4_evaluation_json(
    std::ostream& output,
    const Rlf4EvaluationWorkflowResult& result
);
void write_rlf4_trace_json(
    std::ostream& output,
    const Rlf4TraceWorkflowResult& result
);

}  // namespace rlf::experiments
