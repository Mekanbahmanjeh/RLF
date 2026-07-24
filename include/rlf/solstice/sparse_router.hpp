#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace rlf::solstice {

struct SparseRouterConfig final {
    std::size_t signature_bits{16U};
    std::size_t maximum_candidates{512U};
    std::size_t probe_radius{2U};
    std::uint64_t seed{0x524C46524F555445ULL};
};

struct SparseRouteResult final {
    std::vector<std::size_t> candidate_indices;
    std::uint64_t signatures_probed{};
    std::uint64_t candidates_examined{};
    std::uint64_t exhaustive_candidates{};
};

struct SparseRouterOperationStats final {
    std::uint64_t full_rebuilds{};
    std::uint64_t vectors_rebuilt{};
    std::uint64_t incremental_updates{};
    std::uint64_t vectors_incrementally_updated{};
    std::uint64_t vectors_appended{};
};

class SparseRoutingIndex final {
public:
    explicit SparseRoutingIndex(SparseRouterConfig config = {});

    void rebuild(
        std::span<const float> vectors,
        std::size_t vector_count,
        std::size_t dimension
    );
    void update(std::size_t index, std::span<const float> vector);
    void append(std::span<const float> vector);
    [[nodiscard]] SparseRouteResult route(
        std::span<const float> query
    ) const;
    [[nodiscard]] std::uint64_t signature(
        std::span<const float> vector
    ) const;
    [[nodiscard]] const SparseRouterConfig& config() const noexcept;
    [[nodiscard]] std::size_t vector_count() const noexcept;
    [[nodiscard]] std::size_t dimension() const noexcept;
    [[nodiscard]] SparseRouterOperationStats operation_stats() const noexcept;

private:
    [[nodiscard]] double projection_weight(
        std::size_t bit,
        std::size_t dimension_index
    ) const noexcept;

    SparseRouterConfig config_;
    std::size_t vector_count_{};
    std::size_t dimension_{};
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> buckets_;
    std::vector<std::uint64_t> signatures_;
    SparseRouterOperationStats operation_stats_{};
};

}  // namespace rlf::solstice
