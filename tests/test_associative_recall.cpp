#include "test_framework.hpp"

#include "rlf/experiments/associative_recall.hpp"

RLF_TEST_CASE("associative recall retains one-shot associations") {
    const rlf::experiments::AssociativeRecallResult result =
        rlf::experiments::run_associative_recall({
            .seed = 900ULL,
            .dimension = 64U,
            .association_count = 32U,
            .noise_radians = 0.1,
        });

    RLF_CHECK(result.rlf.exact_recall == 1.0);
    RLF_CHECK(result.rlf.initial_one_shot_recall == 1.0);
    RLF_CHECK(result.rlf.retained_accuracy > 0.9);
    RLF_CHECK(result.baseline.exact_recall == 1.0);
    RLF_CHECK(result.growth.size() >= 2U);
}
