#include "test_framework.hpp"

#include "rlf/core/deterministic_rng.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

RLF_TEST_CASE("deterministic RNG repeats a seeded stream") {
    rlf::core::DeterministicRng first(0x1234ULL);
    rlf::core::DeterministicRng second(0x1234ULL);

    for (std::size_t sample_index = 0U; sample_index < 1'000U; ++sample_index) {
        RLF_CHECK(first.next_u64() == second.next_u64());
    }
}

RLF_TEST_CASE("deterministic RNG separates different seeds") {
    rlf::core::DeterministicRng first(1ULL);
    rlf::core::DeterministicRng second(2ULL);

    std::size_t equal_values = 0U;
    for (std::size_t sample_index = 0U; sample_index < 64U; ++sample_index) {
        if (first.next_u64() == second.next_u64()) {
            ++equal_values;
        }
    }
    RLF_CHECK(equal_values == 0U);
}

RLF_TEST_CASE("deterministic RNG bounded values stay in range") {
    rlf::core::DeterministicRng random_number_generator(42ULL);
    std::array<bool, 7> observed{};

    for (std::size_t sample_index = 0U; sample_index < 10'000U; ++sample_index) {
        const double unit = random_number_generator.uniform_unit();
        RLF_CHECK(unit >= 0.0);
        RLF_CHECK(unit < 1.0);

        const std::size_t index = random_number_generator.uniform_index(
            observed.size()
        );
        RLF_CHECK(index < observed.size());
        observed[index] = true;
    }

    for (const bool was_observed : observed) {
        RLF_CHECK(was_observed);
    }
    RLF_CHECK_THROWS_AS(
        random_number_generator.uniform_index(0U),
        std::invalid_argument
    );
}
