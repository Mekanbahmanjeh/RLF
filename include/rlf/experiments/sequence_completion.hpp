#pragma once

#include "rlf/experiments/metrics.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace rlf::experiments {

struct SequenceCompletionConfig final {
    std::uint64_t seed{0x524C4635ULL};
    std::size_t dimension{256U};
    std::size_t symbol_count{8U};
    std::size_t training_examples{1'024U};
    std::size_t evaluation_examples{256U};
    double corruption_radians{0.12};
    double dominant_probability{0.8};
};

struct SequenceSystemResult final {
    std::string system;
    double next_symbol_accuracy{};
    double unseen_position_accuracy{};
    double corrupted_input_accuracy{};
    ExperimentMetrics metrics;
};

struct SequenceCaseResult final {
    std::string name;
    SequenceSystemResult rlf;
    SequenceSystemResult baseline;
};

struct SequenceCompletionResult final {
    std::uint64_t seed{};
    std::size_t dimension{};
    std::size_t symbol_count{};
    std::size_t training_examples{};
    std::size_t evaluation_examples{};
    double corruption_radians{};
    double dominant_probability{};
    std::vector<SequenceCaseResult> cases;
    std::uint64_t deterministic_run_hash{};
};

[[nodiscard]] SequenceCompletionResult run_sequence_completion(
    const SequenceCompletionConfig& config
);
void write_sequence_completion_json(
    std::ostream& output,
    const SequenceCompletionResult& result
);

}  // namespace rlf::experiments
