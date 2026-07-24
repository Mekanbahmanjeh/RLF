#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace rlf::core {

class DeterministicRng final {
public:
    explicit DeterministicRng(std::uint64_t seed) noexcept;

    [[nodiscard]] std::uint64_t seed() const noexcept;
    [[nodiscard]] std::uint64_t next_u64() noexcept;
    [[nodiscard]] double uniform_unit() noexcept;
    [[nodiscard]] float uniform_angle() noexcept;
    [[nodiscard]] std::size_t uniform_index(std::size_t upper_bound);

private:
    std::uint64_t seed_;
    std::array<std::uint64_t, 4> state_;
};

}  // namespace rlf::core
