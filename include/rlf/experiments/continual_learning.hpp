#pragma once

#include "rlf/experiments/metrics.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace rlf::experiments {

struct ContinualLearningConfig final {
    std::uint64_t seed{0x524C4636ULL};
    std::size_t dimension{256U};
    std::size_t training_examples_per_task{256U};
    std::size_t evaluation_examples_per_task{128U};
    double context_noise_radians{0.05};
};

struct ContinualSystemResult final {
    std::string system;
    std::vector<std::vector<double>> accuracy_after_task;
    std::vector<std::size_t> mode_or_state_growth;
    std::vector<std::size_t> memory_growth_bytes;
    double average_retained_accuracy{};
    double backward_transfer{};
    double average_forgetting{};
    ExperimentMetrics metrics;
};

struct ContinualLearningResult final {
    std::uint64_t seed{};
    std::size_t dimension{};
    std::size_t task_count{4U};
    std::size_t training_examples_per_task{};
    std::size_t evaluation_examples_per_task{};
    double context_noise_radians{};
    ContinualSystemResult rlf;
    ContinualSystemResult baseline;
    std::uint64_t deterministic_run_hash{};
};

[[nodiscard]] ContinualLearningResult run_continual_learning(
    const ContinualLearningConfig& config
);
void write_continual_learning_json(
    std::ostream& output,
    const ContinualLearningResult& result
);

}  // namespace rlf::experiments
