#include "rlf/core/deterministic_rng.hpp"

#include <bit>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace rlf::core {
namespace {

[[nodiscard]] std::uint64_t split_mix_64(std::uint64_t& state) noexcept {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t value = state;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

}  // namespace

DeterministicRng::DeterministicRng(const std::uint64_t seed) noexcept
    : seed_(seed), state_{} {
    std::uint64_t split_mix_state = seed;
    for (std::uint64_t& state_word : state_) {
        state_word = split_mix_64(split_mix_state);
    }
}

std::uint64_t DeterministicRng::seed() const noexcept {
    return seed_;
}

std::uint64_t DeterministicRng::next_u64() noexcept {
    const std::uint64_t result =
        std::rotl(state_[1] * 5ULL, 7) * 9ULL;
    const std::uint64_t temporary = state_[1] << 17U;

    state_[2] ^= state_[0];
    state_[3] ^= state_[1];
    state_[1] ^= state_[2];
    state_[0] ^= state_[3];
    state_[2] ^= temporary;
    state_[3] = std::rotl(state_[3], 45);

    return result;
}

double DeterministicRng::uniform_unit() noexcept {
    constexpr double inverse_two_to_the_53 =
        1.0 / static_cast<double>(1ULL << 53U);
    return static_cast<double>(next_u64() >> 11U) * inverse_two_to_the_53;
}

float DeterministicRng::uniform_angle() noexcept {
    constexpr double tau = 2.0 * std::numbers::pi_v<double>;
    return static_cast<float>(uniform_unit() * tau);
}

std::size_t DeterministicRng::uniform_index(const std::size_t upper_bound) {
    if (upper_bound == 0U) {
        throw std::invalid_argument("uniform_index requires a non-zero upper bound");
    }

    static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));
    const auto bound = static_cast<std::uint64_t>(upper_bound);
    const std::uint64_t threshold = (0ULL - bound) % bound;

    while (true) {
        const std::uint64_t value = next_u64();
        if (value >= threshold) {
            return static_cast<std::size_t>(value % bound);
        }
    }
}

}  // namespace rlf::core
