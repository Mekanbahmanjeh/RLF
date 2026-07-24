#include "test_framework.hpp"

#include "rlf/experiments/transformation_learning.hpp"

#include <algorithm>
#include <string>

RLF_TEST_CASE("transformation experiment learns held-out phase states") {
    const rlf::experiments::TransformationLearningResult result =
        rlf::experiments::run_transformation_learning({
            .seed = 91ULL,
            .dimension = 64U,
            .training_examples = 96U,
            .evaluation_examples = 16U,
        });

    RLF_CHECK(result.strategies.size() == 3U);
    RLF_CHECK(result.tasks.size() == 7U);
    for (const auto& strategy : result.strategies) {
        RLF_CHECK(
            strategy.final_mean_similarity >
            strategy.initial_mean_similarity
        );
        RLF_CHECK(strategy.final_task_accuracy > 0.9);
        RLF_CHECK(strategy.update_operations_per_example > 0.0);
    }
    RLF_CHECK(result.supported_unseen_accuracy > 0.99);
    RLF_CHECK(result.unsupported_unseen_accuracy < 0.5);
    const auto role_rebinding = std::find_if(
        result.tasks.begin(),
        result.tasks.end(),
        [](const auto& task) {
            return task.name == "role_value_rebinding";
        }
    );
    RLF_CHECK(role_rebinding != result.tasks.end());
    RLF_CHECK(role_rebinding->rlf.unseen_accuracy > 0.99);
    RLF_CHECK(
        role_rebinding->rlf.unseen_accuracy >
        role_rebinding->baseline.unseen_accuracy
    );
}

RLF_TEST_CASE("transformation experiment deterministic hash repeats") {
    const rlf::experiments::TransformationLearningConfig config{
        .seed = 17ULL,
        .dimension = 32U,
        .training_examples = 48U,
        .evaluation_examples = 8U,
    };
    const auto first =
        rlf::experiments::run_transformation_learning(config);
    const auto second =
        rlf::experiments::run_transformation_learning(config);

    RLF_CHECK(
        first.deterministic_run_hash == second.deterministic_run_hash
    );
}
