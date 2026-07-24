#include "test_framework.hpp"

#include "rlf/core/deterministic_rng.hpp"
#include "rlf/core/quantized_phase_vector.hpp"

#include <stdexcept>
#include <sstream>
#include <string>

RLF_TEST_CASE("quantized phase representations preserve phase accurately") {
    rlf::core::DeterministicRng rng(0x0A11CE55ULL);
    const rlf::core::PhaseVector value =
        rlf::core::PhaseVector::random(1024U, rng);
    for (const auto encoding : {
             rlf::core::PhaseEncoding::float32,
             rlf::core::PhaseEncoding::uint8_phase,
             rlf::core::PhaseEncoding::uint16_phase,
             rlf::core::PhaseEncoding::float16_compatible,
             rlf::core::PhaseEncoding::int8_residual,
         }) {
        const auto quantized =
            rlf::core::QuantizedPhaseVector::encode(
                value,
                encoding
            );
        const double error =
            quantized.mean_angular_error(value);
        if (encoding == rlf::core::PhaseEncoding::float32) {
            RLF_CHECK(error == 0.0);
        } else {
            RLF_CHECK(error < 0.014);
        }
        RLF_CHECK(
            quantized.decode().similarity(value) > 0.9997
        );
    }
}

RLF_TEST_CASE("quantized phase serialization round trips") {
    rlf::core::DeterministicRng rng(0x51A11CEULL);
    const rlf::core::PhaseVector value =
        rlf::core::PhaseVector::random(257U, rng);
    const auto original =
        rlf::core::QuantizedPhaseVector::encode(
            value,
            rlf::core::PhaseEncoding::uint16_phase
        );
    std::stringstream stream(
        std::ios::in | std::ios::out | std::ios::binary
    );
    original.serialize(stream);
    stream.seekg(0);
    const auto loaded =
        rlf::core::QuantizedPhaseVector::deserialize(stream);
    RLF_CHECK(
        loaded.encoding() ==
        rlf::core::PhaseEncoding::uint16_phase
    );
    RLF_CHECK(loaded.size() == value.size());
    RLF_CHECK(
        loaded.decode().similarity(original.decode()) == 1.0
    );
}

RLF_TEST_CASE("quantized phase serialization rejects malformed input") {
    rlf::core::DeterministicRng rng(0xC0FFEEULL);
    const auto original =
        rlf::core::QuantizedPhaseVector::encode(
            rlf::core::PhaseVector::random(64U, rng),
            rlf::core::PhaseEncoding::uint8_phase
        );
    std::stringstream stream(
        std::ios::in | std::ios::out | std::ios::binary
    );
    original.serialize(stream);
    const std::string serialized = stream.str();

    std::string corrupted_magic = serialized;
    corrupted_magic[0U] = 'X';
    std::stringstream corrupted_stream(
        corrupted_magic,
        std::ios::in | std::ios::binary
    );
    RLF_CHECK_THROWS_AS(
        rlf::core::QuantizedPhaseVector::deserialize(corrupted_stream),
        std::runtime_error
    );

    const std::string truncated =
        serialized.substr(0U, serialized.size() - 1U);
    std::stringstream truncated_stream(
        truncated,
        std::ios::in | std::ios::binary
    );
    RLF_CHECK_THROWS_AS(
        rlf::core::QuantizedPhaseVector::deserialize(truncated_stream),
        std::runtime_error
    );

    std::stringstream limited_stream(
        serialized,
        std::ios::in | std::ios::binary
    );
    RLF_CHECK_THROWS_AS(
        rlf::core::QuantizedPhaseVector::deserialize(
            limited_stream,
            32U
        ),
        std::runtime_error
    );
}
