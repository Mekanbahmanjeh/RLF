#pragma once

#include "rlf/experiments/metrics.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace rlf::experiments {

struct TransformationLearningConfig final {
    std::uint64_t seed{0x524C4631ULL};
    std::size_t dimension{256U};
    std::size_t training_examples{256U};
    std::size_t evaluation_examples{128U};
};

struct TransformationStrategyResult final {
    std::string strategy;
    double initial_mean_similarity{};
    double final_mean_similarity{};
    double final_task_accuracy{};
    double mean_training_error{};
    double average_settling_cycles{};
    double update_operations_per_example{};
    double training_seconds{};
    double inference_seconds{};
    double training_examples_per_second{};
    double inference_examples_per_second{};
    std::size_t modes_created{};
    std::uint64_t deterministic_run_hash{};
};

struct TransformationTaskSystemResult final {
    std::string system;
    double familiar_mean_similarity{};
    double familiar_accuracy{};
    double unseen_mean_similarity{};
    double unseen_accuracy{};
    std::size_t learned_units{};
    ExperimentMetrics metrics;
};

struct TransformationTaskResult final {
    std::string name;
    std::string representation;
    bool expected_unseen_generalization{};
    TransformationTaskSystemResult rlf;
    TransformationTaskSystemResult baseline;
};

struct TransformationLearningResult final {
    std::uint64_t seed{};
    std::size_t dimension{};
    std::size_t training_examples{};
    std::size_t evaluation_examples{};
    std::vector<TransformationStrategyResult> strategies;
    std::vector<TransformationTaskResult> tasks;
    double supported_unseen_accuracy{};
    double unsupported_unseen_accuracy{};
    std::uint64_t deterministic_run_hash{};
};

[[nodiscard]] TransformationLearningResult run_transformation_learning(
    const TransformationLearningConfig& config
);
void write_transformation_learning_json(
    std::ostream& output,
    const TransformationLearningResult& result
);

}  // namespace rlf::experiments
