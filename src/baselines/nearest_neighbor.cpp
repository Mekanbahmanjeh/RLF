#include "rlf/baselines/nearest_neighbor.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rlf::baselines {

NearestNeighborMemory::NearestNeighborMemory(
    const std::size_t dimension
)
    : dimension_(dimension) {
    if (dimension_ == 0U) {
        throw std::invalid_argument(
            "nearest-neighbor dimension must be positive"
        );
    }
}

std::uint64_t NearestNeighborMemory::insert(
    core::PhaseVector key,
    core::PhaseVector value
) {
    if (key.size() != dimension_ || value.size() != dimension_) {
        throw std::invalid_argument(
            "nearest-neighbor records must match the configured dimension"
        );
    }
    if (next_id_ == 0ULL) {
        throw std::overflow_error(
            "nearest-neighbor record ID space exhausted"
        );
    }
    const std::uint64_t id = next_id_;
    if (next_id_ == std::numeric_limits<std::uint64_t>::max()) {
        next_id_ = 0ULL;
    } else {
        ++next_id_;
    }
    records_.push_back({
        .id = id,
        .key = std::move(key),
        .value = std::move(value),
    });
    return id;
}

std::vector<NearestNeighborMatch> NearestNeighborMemory::retrieve(
    const core::PhaseVector& query,
    const std::size_t count
) const {
    if (query.size() != dimension_) {
        throw std::invalid_argument(
            "nearest-neighbor query dimension must match memory"
        );
    }
    if (count == 0U || records_.empty()) {
        return {};
    }
    std::vector<NearestNeighborMatch> matches;
    matches.reserve(records_.size());
    for (std::size_t index = 0U; index < records_.size(); ++index) {
        matches.push_back({
            .record_index = index,
            .record_id = records_[index].id,
            .similarity = query.similarity(records_[index].key),
        });
    }
    const auto strongest_first = [](
        const NearestNeighborMatch& left,
        const NearestNeighborMatch& right
    ) {
        if (left.similarity != right.similarity) {
            return left.similarity > right.similarity;
        }
        return left.record_id < right.record_id;
    };
    const std::size_t selected = std::min(count, matches.size());
    std::partial_sort(
        matches.begin(),
        matches.begin() + static_cast<std::ptrdiff_t>(selected),
        matches.end(),
        strongest_first
    );
    matches.resize(selected);
    return matches;
}

std::size_t NearestNeighborMemory::dimension() const noexcept {
    return dimension_;
}

std::size_t NearestNeighborMemory::size() const noexcept {
    return records_.size();
}

std::span<const NearestNeighborRecord>
NearestNeighborMemory::records() const noexcept {
    return records_;
}

std::size_t NearestNeighborMemory::bytes_stored() const noexcept {
    return sizeof(*this) +
        (records_.size() * sizeof(NearestNeighborRecord)) +
        (records_.size() * dimension_ *
         sizeof(core::PhaseVector::Angle) * 2U);
}

}  // namespace rlf::baselines
