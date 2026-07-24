#pragma once

#include "rlf/experiments/transformation_learning.hpp"

#include <cstdint>
#include <vector>

namespace rlf::experiments::detail {

struct TransformationTaskSuite final {
    std::vector<TransformationTaskResult> tasks;
    double supported_unseen_accuracy{};
    double unsupported_unseen_accuracy{};
    std::uint64_t deterministic_run_hash{};
};

[[nodiscard]] TransformationTaskSuite run_transformation_task_suite(
    const TransformationLearningConfig& config
);

}  // namespace rlf::experiments::detail
