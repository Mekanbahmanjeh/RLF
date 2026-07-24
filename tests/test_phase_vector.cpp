#include "test_framework.hpp"

#include "rlf/core/deterministic_rng.hpp"
#include "rlf/core/phase_vector.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double tolerance = 2.0e-6;

}  // namespace

RLF_TEST_CASE("phase normalization stays in the canonical interval") {
    constexpr float tau = 2.0F * std::numbers::pi_v<float>;
    const std::vector<float> inputs{
        -10.0F * tau,
        -tau,
        -0.25F,
        0.0F,
        0.25F,
        tau,
        10.0F * tau,
    };

    for (const float input : inputs) {
        const float normalized = rlf::core::PhaseVector::normalize_angle(input);
        RLF_CHECK(normalized >= 0.0F);
        RLF_CHECK(normalized < tau);
    }
    RLF_CHECK_NEAR(
        rlf::core::PhaseVector::normalize_angle(-0.25F),
        tau - 0.25F,
        1.0e-6F
    );
}

RLF_TEST_CASE("phase composition and difference reconstruct the target") {
    rlf::core::DeterministicRng random_number_generator(123ULL);
    for (std::size_t sample_index = 0U; sample_index < 200U; ++sample_index) {
        const rlf::core::PhaseVector source =
            rlf::core::PhaseVector::random(257U, random_number_generator);
        const rlf::core::PhaseVector target =
            rlf::core::PhaseVector::random(257U, random_number_generator);
        const rlf::core::PhaseVector difference =
            rlf::core::PhaseVector::phase_difference(source, target);
        const rlf::core::PhaseVector reconstructed =
            source.composed(difference);

        RLF_CHECK_NEAR(reconstructed.similarity(target), 1.0, tolerance);
        RLF_CHECK(
            reconstructed.mean_angular_error(target) < 5.0e-7
        );
    }
}

RLF_TEST_CASE("phase conjugation composes to the identity") {
    rlf::core::DeterministicRng random_number_generator(444ULL);
    const rlf::core::PhaseVector phase_vector =
        rlf::core::PhaseVector::random(128U, random_number_generator);
    const rlf::core::PhaseVector identity =
        phase_vector.composed(phase_vector.conjugated());
    const rlf::core::PhaseVector expected =
        rlf::core::PhaseVector::zeros(128U);

    RLF_CHECK_NEAR(identity.similarity(expected), 1.0, tolerance);
    RLF_CHECK(identity.mean_angular_error(expected) < 5.0e-7);
}

RLF_TEST_CASE("complex conversion preserves phase") {
    rlf::core::DeterministicRng random_number_generator(712ULL);
    const rlf::core::PhaseVector original =
        rlf::core::PhaseVector::random(512U, random_number_generator);
    const std::vector<std::complex<float>> complex_values =
        original.to_complex();
    const rlf::core::PhaseVector restored =
        rlf::core::PhaseVector::from_complex(complex_values);

    RLF_CHECK_NEAR(original.similarity(restored), 1.0, tolerance);
    RLF_CHECK(original.mean_angular_error(restored) < 5.0e-7);
}

RLF_TEST_CASE("zero-magnitude complex values are rejected") {
    const std::vector<std::complex<float>> values{
        std::complex<float>{0.0F, 0.0F}
    };
    RLF_CHECK_THROWS_AS(
        rlf::core::PhaseVector::from_complex(values),
        std::invalid_argument
    );
}

RLF_TEST_CASE("weighted circular averaging crosses the phase wrap") {
    constexpr float degrees_to_radians =
        std::numbers::pi_v<float> / 180.0F;
    const std::vector<rlf::core::PhaseVector> vectors{
        rlf::core::PhaseVector({359.0F * degrees_to_radians}),
        rlf::core::PhaseVector({1.0F * degrees_to_radians}),
    };
    const std::vector<float> weights{1.0F, 1.0F};
    const rlf::core::PhaseVector average =
        rlf::core::PhaseVector::weighted_circular_average(vectors, weights);
    const rlf::core::PhaseVector zero =
        rlf::core::PhaseVector::zeros(1U);

    RLF_CHECK(average.mean_angular_error(zero) < 1.0e-6);
}

RLF_TEST_CASE("degenerate circular averaging uses a deterministic fallback") {
    const std::vector<rlf::core::PhaseVector> vectors{
        rlf::core::PhaseVector({0.0F}),
        rlf::core::PhaseVector({std::numbers::pi_v<float>}),
    };
    const std::vector<float> weights{1.0F, 1.0F};
    const rlf::core::PhaseVector average =
        rlf::core::PhaseVector::weighted_circular_average(vectors, weights);

    RLF_CHECK_NEAR(average[0U], 0.0F, 1.0e-7F);
}

RLF_TEST_CASE("permutation is validated and deterministic") {
    const rlf::core::PhaseVector source({0.1F, 0.2F, 0.3F, 0.4F});
    const std::vector<std::size_t> permutation{2U, 0U, 3U, 1U};
    const rlf::core::PhaseVector result = source.permuted(permutation);

    RLF_CHECK_NEAR(result[0U], 0.3F, 1.0e-7F);
    RLF_CHECK_NEAR(result[1U], 0.1F, 1.0e-7F);
    RLF_CHECK_NEAR(result[2U], 0.4F, 1.0e-7F);
    RLF_CHECK_NEAR(result[3U], 0.2F, 1.0e-7F);

    const std::vector<std::size_t> duplicate{0U, 0U, 2U, 3U};
    RLF_CHECK_THROWS_AS(source.permuted(duplicate), std::invalid_argument);
}

RLF_TEST_CASE("similarity implements normalized squared complex alignment") {
    const rlf::core::PhaseVector reference({0.0F, 0.0F, 0.0F, 0.0F});
    const rlf::core::PhaseVector identical({0.0F, 0.0F, 0.0F, 0.0F});
    const rlf::core::PhaseVector globally_shifted({
        0.7F, 0.7F, 0.7F, 0.7F
    });
    const rlf::core::PhaseVector cancelling({
        0.0F,
        0.0F,
        std::numbers::pi_v<float>,
        std::numbers::pi_v<float>,
    });

    RLF_CHECK_NEAR(reference.similarity(identical), 1.0, tolerance);
    RLF_CHECK_NEAR(reference.similarity(globally_shifted), 1.0, tolerance);
    RLF_CHECK_NEAR(reference.similarity(cancelling), 0.0, tolerance);
}

RLF_TEST_CASE("phase-vector serialization round trips exactly") {
    rlf::core::DeterministicRng random_number_generator(999ULL);
    const rlf::core::PhaseVector original =
        rlf::core::PhaseVector::random(1'024U, random_number_generator);

    std::stringstream stream(
        std::ios::in | std::ios::out | std::ios::binary
    );
    original.serialize(stream);
    stream.seekg(0);
    const rlf::core::PhaseVector restored =
        rlf::core::PhaseVector::deserialize(stream);

    RLF_CHECK(original.angles().size() == restored.angles().size());
    for (std::size_t dimension_index = 0U;
         dimension_index < original.size();
         ++dimension_index) {
        RLF_CHECK(original[dimension_index] == restored[dimension_index]);
    }
}

RLF_TEST_CASE("phase-vector serialization rejects corruption and truncation") {
    const rlf::core::PhaseVector original({0.1F, 0.2F, 0.3F});
    std::stringstream stream(
        std::ios::in | std::ios::out | std::ios::binary
    );
    original.serialize(stream);
    std::string bytes = stream.str();

    std::string corrupted = bytes;
    corrupted[20U] = static_cast<char>(corrupted[20U] ^ 0x01);
    std::stringstream corrupted_stream(
        corrupted,
        std::ios::in | std::ios::binary
    );
    RLF_CHECK_THROWS_AS(
        rlf::core::PhaseVector::deserialize(corrupted_stream),
        std::runtime_error
    );

    bytes.resize(bytes.size() - 1U);
    std::stringstream truncated_stream(
        bytes,
        std::ios::in | std::ios::binary
    );
    RLF_CHECK_THROWS_AS(
        rlf::core::PhaseVector::deserialize(truncated_stream),
        std::runtime_error
    );
}

RLF_TEST_CASE("dimension mismatches and invalid weights are rejected") {
    const rlf::core::PhaseVector short_vector({0.0F});
    const rlf::core::PhaseVector long_vector({0.0F, 1.0F});
    RLF_CHECK_THROWS_AS(
        short_vector.composed(long_vector),
        std::invalid_argument
    );
    RLF_CHECK_THROWS_AS(
        short_vector.similarity(long_vector),
        std::invalid_argument
    );

    const std::vector<rlf::core::PhaseVector> vectors{short_vector};
    const std::vector<float> zero_weight{0.0F};
    RLF_CHECK_THROWS_AS(
        rlf::core::PhaseVector::weighted_circular_average(
            vectors,
            zero_weight
        ),
        std::invalid_argument
    );
}
