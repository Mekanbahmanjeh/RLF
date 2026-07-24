#include "test_framework.hpp"

#include "rlf/experiments/compositional_generalization.hpp"

RLF_TEST_CASE("composition reuses additive modes on unseen inputs") {
    const rlf::experiments::CompositionalGeneralizationResult result =
        rlf::experiments::run_compositional_generalization({
            .seed = 0xC011AB1EULL,
            .dimension = 64U,
            .training_examples = 48U,
            .evaluation_examples = 32U,
        });

    RLF_CHECK(result.cases.size() == 3U);
    RLF_CHECK(result.supported_case_score > 0.99);
    RLF_CHECK(result.cases[0U].rlf.learned_units == 2U);
    RLF_CHECK(
        result.cases[0U].baseline.learned_units == 96U
    );
    RLF_CHECK(
        result.cases[0U].rlf.composed_accuracy >
        result.cases[0U].baseline.composed_accuracy
    );
    RLF_CHECK(
        result.cases[1U].rlf.composed_accuracy > 0.99
    );
}

RLF_TEST_CASE("composition experiment exposes unsupported non-additive case") {
    const rlf::experiments::CompositionalGeneralizationConfig config{
        .seed = 0xC011AB1EULL,
        .dimension = 64U,
        .training_examples = 32U,
        .evaluation_examples = 24U,
    };
    const auto first =
        rlf::experiments::run_compositional_generalization(config);
    const auto second =
        rlf::experiments::run_compositional_generalization(config);

    RLF_CHECK(first.unsupported_case_score < 0.5);
    RLF_CHECK(
        first.deterministic_run_hash ==
        second.deterministic_run_hash
    );
}
