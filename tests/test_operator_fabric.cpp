#include "test_framework.hpp"

#include "rlf/core/deterministic_rng.hpp"
#include "rlf/core/operator_fabric.hpp"

#include <cstddef>
#include <vector>

namespace {

[[nodiscard]] std::vector<std::size_t> rotated_payload(
    const std::size_t dimension,
    const std::size_t begin_index
) {
    std::vector<std::size_t> permutation(dimension);
    for (std::size_t index = 0U; index < dimension; ++index) {
        permutation[index] = index;
    }
    for (std::size_t index = begin_index; index < dimension; ++index) {
        permutation[index] = begin_index +
            ((index - begin_index + 1U) % (dimension - begin_index));
    }
    return permutation;
}

}  // namespace

RLF_TEST_CASE("transformation operators compose explicit sequences") {
    constexpr std::size_t dimension = 16U;
    constexpr std::size_t context_dimensions = 4U;
    rlf::core::DeterministicRng rng(0x0A11CEULL);
    const rlf::core::PhaseVector input =
        rlf::core::PhaseVector::random(dimension, rng);
    const rlf::core::PhaseVector shift =
        rlf::core::PhaseVector::random(dimension, rng);
    const rlf::core::TransformationOperator first(
        dimension,
        {rlf::core::OperatorPrimitive::permute(
            rotated_payload(dimension, context_dimensions),
            context_dimensions
        )}
    );
    const rlf::core::TransformationOperator second(
        dimension,
        {
            rlf::core::OperatorPrimitive::conjugate(
                dimension,
                context_dimensions
            ),
            rlf::core::OperatorPrimitive::shift(
                shift,
                context_dimensions
            ),
        }
    );
    const rlf::core::PhaseVector sequential =
        second.apply(first.apply(input));
    const rlf::core::PhaseVector composed =
        first.then(second).apply(input);
    RLF_CHECK(sequential.similarity(composed) > 0.999999);
}

RLF_TEST_CASE("operator fabric selects permutation shift without labels") {
    constexpr std::size_t dimension = 24U;
    constexpr std::size_t context_dimensions = 8U;
    rlf::core::DeterministicRng rng(0x0BADC0DEULL);
    const rlf::core::PhaseVector context =
        rlf::core::PhaseVector::random(dimension, rng);
    const rlf::core::PhaseVector shift =
        rlf::core::PhaseVector::random(dimension, rng);
    const rlf::core::TransformationOperator truth(
        dimension,
        {
            rlf::core::OperatorPrimitive::permute(
                rotated_payload(dimension, context_dimensions),
                context_dimensions
            ),
            rlf::core::OperatorPrimitive::shift(
                shift,
                context_dimensions
            ),
        }
    );
    rlf::core::OperatorFabric fabric({
        .dimension = dimension,
        .context_dimensions = context_dimensions,
        .history_capacity = 48U,
    });
    for (std::size_t example_index = 0U;
         example_index < 32U;
         ++example_index) {
        const rlf::core::PhaseVector random_input =
            rlf::core::PhaseVector::random(dimension, rng);
        std::vector<float> angles(
            random_input.angles().begin(),
            random_input.angles().end()
        );
        for (std::size_t index = 0U; index < context_dimensions; ++index) {
            angles[index] = context[index];
        }
        const rlf::core::PhaseVector input(std::move(angles));
        fabric.learn(input, truth.apply(input));
    }
    const rlf::core::PhaseVector random_evaluation_input =
        rlf::core::PhaseVector::random(dimension, rng);
    std::vector<float> evaluation_angles(
        random_evaluation_input.angles().begin(),
        random_evaluation_input.angles().end()
    );
    for (std::size_t index = 0U; index < context_dimensions; ++index) {
        evaluation_angles[index] = context[index];
    }
    const rlf::core::PhaseVector evaluation_input(
        std::move(evaluation_angles)
    );
    const rlf::core::OperatorPrediction prediction =
        fabric.predict(evaluation_input);
    RLF_CHECK(
        prediction.family ==
        rlf::core::OperatorFamily::permutation_then_phase_shift
    );
    RLF_CHECK(
        prediction.state.similarity(truth.apply(evaluation_input)) >
        0.99
    );
}

RLF_TEST_CASE("transformation operator inverse reconstructs composite inputs") {
    constexpr std::size_t dimension = 20U;
    constexpr std::size_t begin_index = 3U;
    rlf::core::DeterministicRng rng(0x1A2B3C4DULL);
    const rlf::core::PhaseVector input =
        rlf::core::PhaseVector::random(dimension, rng);
    const rlf::core::PhaseVector shift =
        rlf::core::PhaseVector::random(dimension, rng);
    const rlf::core::TransformationOperator transformation(
        dimension,
        {
            rlf::core::OperatorPrimitive::shift(shift, begin_index),
            rlf::core::OperatorPrimitive::permute(
                rotated_payload(dimension, begin_index),
                begin_index
            ),
            rlf::core::OperatorPrimitive::conjugate(
                dimension,
                begin_index
            ),
        }
    );
    const rlf::core::PhaseVector output = transformation.apply(input);
    const rlf::core::PhaseVector reconstructed =
        transformation.inverse().apply(output);
    RLF_CHECK(reconstructed.similarity(input) > 0.999999);
    RLF_CHECK(reconstructed.mean_angular_error(input) < 1.0e-5);
}
