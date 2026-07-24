#pragma once

#include "rlf/core/phase_vector.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace rlf::baselines {

struct NearestNeighborRecord final {
    std::uint64_t id;
    core::PhaseVector key;
    core::PhaseVector value;
};

struct NearestNeighborMatch final {
    std::size_t record_index;
    std::uint64_t record_id;
    double similarity;
};

class NearestNeighborMemory final {
public:
    explicit NearestNeighborMemory(std::size_t dimension);

    [[nodiscard]] std::uint64_t insert(
        core::PhaseVector key,
        core::PhaseVector value
    );
    [[nodiscard]] std::vector<NearestNeighborMatch> retrieve(
        const core::PhaseVector& query,
        std::size_t count
    ) const;

    [[nodiscard]] std::size_t dimension() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::span<const NearestNeighborRecord>
    records() const noexcept;
    [[nodiscard]] std::size_t bytes_stored() const noexcept;

private:
    std::size_t dimension_;
    std::vector<NearestNeighborRecord> records_;
    std::uint64_t next_id_{1ULL};
};

}  // namespace rlf::baselines
