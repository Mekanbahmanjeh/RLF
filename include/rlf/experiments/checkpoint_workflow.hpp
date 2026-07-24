#pragma once

#include "rlf/core/settling.hpp"
#include "rlf/experiments/metrics.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

namespace rlf::experiments {

inline constexpr const char* checkpoint_workflow_task =
    "global_phase_delta";

struct CheckpointTrainingConfig final {
    std::uint64_t seed{0x524C4639ULL};
    std::size_t dimension{1'024U};
    std::size_t training_examples{256U};
    std::size_t evaluation_examples{128U};
    std::filesystem::path checkpoint_path{"results/rlf_trained.rlf"};
};

struct CheckpointTrainingResult final {
    std::string task;
    std::uint64_t seed{};
    std::size_t dimension{};
    std::size_t training_examples{};
    std::size_t evaluation_examples{};
    double initial_mean_similarity{};
    double final_mean_similarity{};
    double final_accuracy{};
    std::size_t checkpoint_bytes{};
    std::uint64_t payload_checksum{};
    ExperimentMetrics metrics;
    std::uint64_t deterministic_run_hash{};
};

struct CheckpointEvaluationResult final {
    std::string task;
    std::uint64_t seed{};
    std::size_t dimension{};
    std::size_t evaluation_examples{};
    double mean_similarity{};
    double accuracy{};
    ExperimentMetrics metrics;
    std::uint64_t deterministic_run_hash{};
};

struct CheckpointTraceResult final {
    std::string task;
    std::uint64_t seed{};
    std::size_t dimension{};
    std::size_t sample_id{};
    double target_similarity{};
    std::string stopping_reason;
    std::vector<core::SettlingCycleTrace> cycles;
    std::uint64_t deterministic_run_hash{};
};

[[nodiscard]] CheckpointTrainingResult train_checkpoint_workflow(
    const CheckpointTrainingConfig& config
);
[[nodiscard]] CheckpointEvaluationResult evaluate_checkpoint_workflow(
    const std::filesystem::path& checkpoint_path,
    const std::string& task,
    std::size_t evaluation_examples
);
[[nodiscard]] CheckpointTraceResult trace_checkpoint_workflow(
    const std::filesystem::path& checkpoint_path,
    const std::string& task,
    std::size_t sample_id
);

void write_checkpoint_training_json(
    std::ostream& output,
    const CheckpointTrainingResult& result
);
void write_checkpoint_evaluation_json(
    std::ostream& output,
    const CheckpointEvaluationResult& result
);
void write_checkpoint_trace_json(
    std::ostream& output,
    const CheckpointTraceResult& result
);

}  // namespace rlf::experiments
