#pragma once

#include "rlf/experiments/metrics.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace rlf::experiments {

struct AssociativeRecallConfig final {
    std::uint64_t seed{0x524C4634ULL};
    std::size_t dimension{256U};
    std::size_t association_count{256U};
    double noise_radians{0.15};
};

struct RecallGrowthPoint final {
    std::size_t records{};
    double rlf_exact_recall{};
    double rlf_noisy_recall{};
    double baseline_exact_recall{};
    double baseline_noisy_recall{};
};

struct RecallSystemResult final {
    std::string system;
    double exact_recall{};
    double noisy_recall{};
    double initial_one_shot_recall{};
    double retained_accuracy{};
    ExperimentMetrics metrics;
};

struct AssociativeRecallResult final {
    std::uint64_t seed{};
    std::size_t dimension{};
    std::size_t association_count{};
    double noise_radians{};
    RecallSystemResult rlf;
    RecallSystemResult baseline;
    std::vector<RecallGrowthPoint> growth;
    std::uint64_t deterministic_run_hash{};
};

[[nodiscard]] AssociativeRecallResult run_associative_recall(
    const AssociativeRecallConfig& config
);
void write_associative_recall_json(
    std::ostream& output,
    const AssociativeRecallResult& result
);

}  // namespace rlf::experiments
