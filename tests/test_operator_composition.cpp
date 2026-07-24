#include "test_framework.hpp"

#include "rlf/experiments/operator_composition.hpp"

RLF_TEST_CASE("operator extension generalizes non-additive compositions") {
    const rlf::experiments::OperatorCompositionResult result =
        rlf::experiments::run_operator_composition({
            .seed = 0x0F3A4702ULL,
            .dimension = 24U,
            .context_dimensions = 6U,
            .training_examples = 12U,
            .evaluation_examples = 12U,
            .noise_radians = 0.05,
        });

    RLF_CHECK(result.operator_familiar_accuracy > 0.9);
    RLF_CHECK(result.operator_unseen_composition_accuracy > 0.9);
    RLF_CHECK(
        result.operator_unseen_composition_accuracy >
        result.phase_offset_unseen_composition_accuracy
    );
    RLF_CHECK(
        result.operator_unseen_composition_accuracy >
        result.nearest_neighbor_unseen_composition_accuracy
    );
    RLF_CHECK(result.supervised_unseen_composition_accuracy > 0.9);
    RLF_CHECK(result.oracle_unseen_composition_accuracy == 1.0);
}

RLF_TEST_CASE("operator experiment exposes ambiguous context limitation") {
    const rlf::experiments::OperatorCompositionConfig config{
        .seed = 0x0F3A4702ULL,
        .dimension = 24U,
        .context_dimensions = 6U,
        .training_examples = 12U,
        .evaluation_examples = 12U,
        .noise_radians = 0.05,
    };
    const auto first =
        rlf::experiments::run_operator_composition(config);
    const auto second =
        rlf::experiments::run_operator_composition(config);
    RLF_CHECK(first.ambiguous_context_accuracy < 0.75);
    RLF_CHECK(
        first.deterministic_run_hash ==
        second.deterministic_run_hash
    );
    RLF_CHECK(first.scientific_decision == "A");
}
