#pragma once

#include "rlf/experiments/metrics.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace rlf::experiments {

struct CompositionalGeneralizationConfig final {
    std::uint64_t seed{0x524C4637ULL};
    std::size_t dimension{256U};
    std::size_t training_examples{128U};
    std::size_t evaluation_examples{128U};
};

struct CompositionSystemResult final {
    std::string system;
    double first_transformation_accuracy{};
    double second_transformation_accuracy{};
    double composed_mean_similarity{};
    double composed_accuracy{};
    std::size_t learned_units{};
    ExperimentMetrics metrics;
};

struct CompositionCaseResult final {
    std::string name;
    std::string transformation_class;
    bool expected_supported{};
    CompositionSystemResult rlf;
    CompositionSystemResult baseline;
};

struct CompositionalGeneralizationResult final {
    std::uint64_t seed{};
    std::size_t dimension{};
    std::size_t training_examples{};
    std::size_t evaluation_examples{};
    std::vector<CompositionCaseResult> cases;
    double supported_case_score{};
    double unsupported_case_score{};
    std::uint64_t deterministic_run_hash{};
};

[[nodiscard]] CompositionalGeneralizationResult
run_compositional_generalization(
    const CompositionalGeneralizationConfig& config
);
void write_compositional_generalization_json(
    std::ostream& output,
    const CompositionalGeneralizationResult& result
);

}  // namespace rlf::experiments
