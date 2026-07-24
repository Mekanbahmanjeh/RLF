#include "test_framework.hpp"

#include "rlf/experiments/continual_learning.hpp"

RLF_TEST_CASE("continual learning retains four transformations without replay") {
    const rlf::experiments::ContinualLearningResult result =
        rlf::experiments::run_continual_learning({
            .seed = 902ULL,
            .dimension = 64U,
            .training_examples_per_task = 48U,
            .evaluation_examples_per_task = 16U,
            .context_noise_radians = 0.03,
        });

    RLF_CHECK(result.rlf.accuracy_after_task.size() == 4U);
    RLF_CHECK(result.rlf.average_retained_accuracy > 0.9);
    RLF_CHECK(result.rlf.average_forgetting < 0.1);
    RLF_CHECK(result.rlf.metrics.modes_created == 4ULL);
    RLF_CHECK(
        result.rlf.average_retained_accuracy >
        result.baseline.average_retained_accuracy
    );
}
