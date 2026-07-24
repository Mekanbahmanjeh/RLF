#include "test_framework.hpp"

#include "rlf/backend/compute_backend.hpp"
#include "rlf/core/deterministic_rng.hpp"
#include "rlf/core/phase_vector.hpp"

#include <cstddef>

RLF_TEST_CASE("optimized CPU similarity matches scalar reference") {
    rlf::core::DeterministicRng rng(0xBACC3EEDULL);
    const rlf::backend::ScalarCpuBackend scalar;
    const rlf::backend::OptimizedCpuBackend optimized;
    for (std::size_t sample = 0U; sample < 64U; ++sample) {
        const rlf::core::PhaseVector left =
            rlf::core::PhaseVector::random(257U, rng);
        const rlf::core::PhaseVector right =
            rlf::core::PhaseVector::random(257U, rng);
        RLF_CHECK_NEAR(
            optimized.similarity(left, right),
            scalar.similarity(left, right),
            2.0e-6
        );
    }
}

RLF_TEST_CASE("future CUDA backend remains explicitly unavailable") {
    const auto cuda = rlf::backend::make_backend(
        rlf::backend::BackendKind::future_cuda
    );
    RLF_CHECK(!cuda->available());
    RLF_CHECK(
        cuda->name() == std::string_view("future_cuda")
    );
    const rlf::core::PhaseVector value =
        rlf::core::PhaseVector::zeros(4U);
    RLF_CHECK_THROWS_AS(
        cuda->similarity(value, value),
        std::runtime_error
    );
}
