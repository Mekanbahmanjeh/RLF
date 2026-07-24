#pragma once

#include "rlf/experiments/metrics.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace rlf::experiments {

struct CapacityScalingConfig final {
    std::uint64_t seed{0x524C4638ULL};
    std::size_t dimension{64U};
    std::size_t evaluation_queries{16U};
    std::size_t candidate_count{256U};
    std::size_t active_count{1U};
    double noise_radians{0.03};
    std::vector<std::size_t> mode_counts{
        1'024U,
        4'096U,
        16'384U,
        65'536U,
    };
};

struct CapacitySystemResult final {
    std::string system;
    double clean_accuracy{};
    double noisy_accuracy{};
    double useful_transformations{};
    double exact_similarity_evaluations_per_inference{};
    double post_retrieval_operations_per_inference{};
    std::size_t maximum_candidates_returned{};
    ExperimentMetrics metrics;
};

struct CapacityScaleResult final {
    std::size_t mode_count{};
    CapacitySystemResult rlf;
    CapacitySystemResult baseline;
};

struct CapacityScalingResult final {
    std::uint64_t seed{};
    std::size_t dimension{};
    std::size_t evaluation_queries{};
    std::size_t candidate_count{};
    std::size_t active_count{};
    double noise_radians{};
    std::vector<CapacityScaleResult> scales;
    bool rlf_post_retrieval_work_bounded{};
    bool rlf_total_exact_work_bounded{};
    std::uint64_t deterministic_run_hash{};
};

[[nodiscard]] CapacityScalingResult run_capacity_scaling(
    const CapacityScalingConfig& config
);
void write_capacity_scaling_json(
    std::ostream& output,
    const CapacityScalingResult& result
);

}  // namespace rlf::experiments
