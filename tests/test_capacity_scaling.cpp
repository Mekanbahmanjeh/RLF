#include "test_framework.hpp"

#include "rlf/experiments/capacity_scaling.hpp"

RLF_TEST_CASE("capacity scaling preserves retrieval while reporting exact cost") {
    const rlf::experiments::CapacityScalingResult result =
        rlf::experiments::run_capacity_scaling({
            .seed = 0xCA9AC17EULL,
            .dimension = 32U,
            .evaluation_queries = 8U,
            .candidate_count = 16U,
            .active_count = 1U,
            .noise_radians = 0.02,
            .mode_counts = {32U, 128U, 512U},
        });

    RLF_CHECK(result.scales.size() == 3U);
    RLF_CHECK(result.rlf_post_retrieval_work_bounded);
    RLF_CHECK(!result.rlf_total_exact_work_bounded);
    for (const auto& scale : result.scales) {
        RLF_CHECK(scale.rlf.clean_accuracy == 1.0);
        RLF_CHECK(scale.rlf.noisy_accuracy == 1.0);
        RLF_CHECK(scale.baseline.clean_accuracy == 1.0);
        RLF_CHECK(scale.baseline.noisy_accuracy == 1.0);
        RLF_CHECK(
            scale.rlf.exact_similarity_evaluations_per_inference ==
            static_cast<double>(scale.mode_count)
        );
        RLF_CHECK(scale.rlf.maximum_candidates_returned <= 16U);
    }
}

RLF_TEST_CASE("capacity scaling deterministic hash repeats") {
    const rlf::experiments::CapacityScalingConfig config{
        .seed = 0xCA9AC17EULL,
        .dimension = 24U,
        .evaluation_queries = 4U,
        .candidate_count = 8U,
        .active_count = 1U,
        .noise_radians = 0.02,
        .mode_counts = {16U, 64U},
    };
    const auto first =
        rlf::experiments::run_capacity_scaling(config);
    const auto second =
        rlf::experiments::run_capacity_scaling(config);

    RLF_CHECK(
        first.deterministic_run_hash ==
        second.deterministic_run_hash
    );
}
