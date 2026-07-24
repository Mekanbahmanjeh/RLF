#include "test_framework.hpp"

#include "rlf/experiments/sequence_completion.hpp"

RLF_TEST_CASE("sequence completion learns deterministic and dominant transitions") {
    const rlf::experiments::SequenceCompletionResult result =
        rlf::experiments::run_sequence_completion({
            .seed = 901ULL,
            .dimension = 64U,
            .symbol_count = 6U,
            .training_examples = 256U,
            .evaluation_examples = 64U,
            .corruption_radians = 0.08,
            .dominant_probability = 0.8,
        });

    RLF_CHECK(result.cases.size() == 2U);
    RLF_CHECK(result.cases[0U].rlf.next_symbol_accuracy > 0.9);
    RLF_CHECK(result.cases[0U].rlf.corrupted_input_accuracy > 0.9);
    RLF_CHECK(result.cases[1U].rlf.next_symbol_accuracy > 0.7);
    RLF_CHECK(result.cases[0U].rlf.metrics.modes_created > 0ULL);
}
