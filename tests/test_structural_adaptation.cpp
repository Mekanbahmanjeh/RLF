#include "test_framework.hpp"

#include "rlf/experiments/structural_adaptation.hpp"

RLF_TEST_CASE("structural adaptation specializes incompatible contexts") {
    const rlf::experiments::StructuralAdaptationResult result =
        rlf::experiments::run_structural_adaptation({
            .seed = 404ULL,
            .dimension = 64U,
            .training_examples = 24U,
            .evaluation_examples = 32U,
            .context_noise_radians = 0.03,
        });

    RLF_CHECK(result.modes_split >= 1ULL);
    RLF_CHECK(result.enabled_modes >= 2U);
    RLF_CHECK(
        result.final_mean_similarity >
        result.initial_mean_similarity
    );
    RLF_CHECK(result.final_task_accuracy > 0.9);
}

RLF_TEST_CASE("structural adaptation hash is deterministic") {
    const rlf::experiments::StructuralAdaptationConfig config{
        .seed = 405ULL,
        .dimension = 32U,
        .training_examples = 24U,
        .evaluation_examples = 16U,
        .context_noise_radians = 0.02,
    };
    const auto first =
        rlf::experiments::run_structural_adaptation(config);
    const auto second =
        rlf::experiments::run_structural_adaptation(config);

    RLF_CHECK(
        first.deterministic_run_hash == second.deterministic_run_hash
    );
}
