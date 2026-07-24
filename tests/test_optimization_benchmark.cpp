#include "test_framework.hpp"

#include "rlf/experiments/optimization_benchmark.hpp"

RLF_TEST_CASE("optimization benchmark preserves reference results") {
    const auto result =
        rlf::experiments::run_optimization_benchmark({
            .seed = 0x0B3ECA11ULL,
            .dimension = 64U,
            .mode_count = 128U,
            .query_count = 8U,
            .candidate_count = 16U,
            .thread_count = 2U,
            .similarity_iterations = 64U,
            .quantization_samples = 8U,
        });
    RLF_CHECK(result.optimized_matches_reference);
    RLF_CHECK(result.parallel_matches_reference);
    RLF_CHECK(result.maximum_similarity_difference < 2.0e-6);
    RLF_CHECK(result.retrieval.size() == 5U);
    for (const auto& item : result.retrieval) {
        RLF_CHECK(item.accuracy == 1.0);
    }
    RLF_CHECK(result.quantization.size() == 5U);
    for (const auto& item : result.quantization) {
        RLF_CHECK(item.transformation_accuracy == 1.0);
        RLF_CHECK(item.numerically_stable);
    }
    RLF_CHECK(!result.cuda_scientifically_justified);
}
