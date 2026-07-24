#include "rlf/solstice/sparse_router.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace rlf::solstice {
namespace {

[[nodiscard]] std::uint64_t mix64(std::uint64_t value) noexcept {
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31U;
    return value;
}

void saturating_add(
    std::uint64_t& destination,
    const std::size_t amount
) noexcept {
    const auto converted = static_cast<std::uint64_t>(amount);
    destination = converted >
        std::numeric_limits<std::uint64_t>::max() - destination
        ? std::numeric_limits<std::uint64_t>::max()
        : destination + converted;
}

}  // namespace

SparseRoutingIndex::SparseRoutingIndex(SparseRouterConfig config)
    : config_(std::move(config)) {
    if (config_.signature_bits == 0U || config_.signature_bits > 63U ||
        config_.maximum_candidates == 0U || config_.probe_radius > 3U) {
        throw std::invalid_argument("invalid sparse routing configuration");
    }
}

double SparseRoutingIndex::projection_weight(
    const std::size_t bit,
    const std::size_t dimension_index
) const noexcept {
    const std::uint64_t mixed = mix64(
        config_.seed ^
        (static_cast<std::uint64_t>(bit) * 0x9e3779b97f4a7c15ULL) ^
        (static_cast<std::uint64_t>(dimension_index) * 0xd6e8feb86659fd93ULL)
    );
    const double magnitude = 0.5 +
        static_cast<double>((mixed >> 11U) & 0xFFFFULL) / 65535.0;
    return (mixed & 1ULL) == 0ULL ? magnitude : -magnitude;
}

std::uint64_t SparseRoutingIndex::signature(
    const std::span<const float> vector
) const {
    if (vector.empty()) {
        throw std::invalid_argument("sparse router query must be non-empty");
    }
    std::uint64_t result = 0U;
    for (std::size_t bit = 0U; bit < config_.signature_bits; ++bit) {
        double projection = 0.0;
        for (std::size_t index = 0U; index < vector.size(); ++index) {
            projection += static_cast<double>(vector[index]) *
                projection_weight(bit, index);
        }
        if (projection >= 0.0) {
            result |= 1ULL << bit;
        }
    }
    return result;
}

void SparseRoutingIndex::rebuild(
    const std::span<const float> vectors,
    const std::size_t vector_count,
    const std::size_t dimension
) {
    if (vector_count == 0U) {
        saturating_add(operation_stats_.full_rebuilds, 1U);
        vector_count_ = 0U;
        dimension_ = dimension;
        buckets_.clear();
        signatures_.clear();
        return;
    }
    if (dimension == 0U ||
        vector_count > std::numeric_limits<std::size_t>::max() / dimension ||
        vectors.size() != vector_count * dimension) {
        throw std::invalid_argument("sparse router matrix shape mismatch");
    }
    saturating_add(operation_stats_.full_rebuilds, 1U);
    saturating_add(operation_stats_.vectors_rebuilt, vector_count);
    vector_count_ = vector_count;
    dimension_ = dimension;
    buckets_.clear();
    buckets_.reserve(vector_count);
    signatures_.assign(vector_count, 0U);
    for (std::size_t index = 0U; index < vector_count; ++index) {
        const auto vector = vectors.subspan(index * dimension, dimension);
        const std::uint64_t value = signature(vector);
        signatures_[index] = value;
        buckets_[value].push_back(index);
    }
}

void SparseRoutingIndex::update(
    const std::size_t index,
    const std::span<const float> vector
) {
    if (index >= vector_count_ || vector.size() != dimension_ || dimension_ == 0U) {
        throw std::invalid_argument("sparse router incremental update is invalid");
    }
    saturating_add(operation_stats_.incremental_updates, 1U);
    saturating_add(operation_stats_.vectors_incrementally_updated, 1U);
    const std::uint64_t previous = signatures_[index];
    const std::uint64_t replacement = signature(vector);
    if (previous == replacement) {
        return;
    }
    auto old_bucket = buckets_.find(previous);
    if (old_bucket == buckets_.end()) {
        throw std::logic_error("sparse router signature table is inconsistent");
    }
    auto position = std::lower_bound(
        old_bucket->second.begin(), old_bucket->second.end(), index
    );
    if (position == old_bucket->second.end() || *position != index) {
        throw std::logic_error("sparse router bucket is inconsistent");
    }
    old_bucket->second.erase(position);
    if (old_bucket->second.empty()) {
        buckets_.erase(old_bucket);
    }
    auto& new_bucket = buckets_[replacement];
    new_bucket.insert(
        std::lower_bound(new_bucket.begin(), new_bucket.end(), index), index
    );
    signatures_[index] = replacement;
}

void SparseRoutingIndex::append(const std::span<const float> vector) {
    if (dimension_ == 0U || vector.size() != dimension_ ||
        vector_count_ == std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("sparse router append is invalid");
    }
    const std::uint64_t value = signature(vector);
    const std::size_t index = vector_count_;
    buckets_[value].push_back(index);
    signatures_.push_back(value);
    ++vector_count_;
    saturating_add(operation_stats_.incremental_updates, 1U);
    saturating_add(operation_stats_.vectors_appended, 1U);
}

SparseRouteResult SparseRoutingIndex::route(
    const std::span<const float> query
) const {
    if (dimension_ == 0U || query.size() != dimension_) {
        throw std::invalid_argument("sparse router query dimension mismatch");
    }
    SparseRouteResult result;
    result.exhaustive_candidates = vector_count_;
    if (vector_count_ == 0U) {
        return result;
    }
    const std::uint64_t base = signature(query);
    std::vector<std::uint64_t> signatures;
    signatures.push_back(base);
    if (config_.probe_radius >= 1U) {
        for (std::size_t first = 0U; first < config_.signature_bits; ++first) {
            signatures.push_back(base ^ (1ULL << first));
        }
    }
    if (config_.probe_radius >= 2U) {
        for (std::size_t first = 0U; first < config_.signature_bits; ++first) {
            for (std::size_t second = first + 1U;
                 second < config_.signature_bits;
                 ++second) {
                signatures.push_back(
                    base ^ (1ULL << first) ^ (1ULL << second)
                );
            }
        }
    }
    if (config_.probe_radius >= 3U) {
        for (std::size_t first = 0U; first < config_.signature_bits; ++first) {
            for (std::size_t second = first + 1U;
                 second < config_.signature_bits;
                 ++second) {
                for (std::size_t third = second + 1U;
                     third < config_.signature_bits;
                     ++third) {
                    signatures.push_back(
                        base ^ (1ULL << first) ^ (1ULL << second) ^
                        (1ULL << third)
                    );
                }
            }
        }
    }
    for (const std::uint64_t candidate_signature : signatures) {
        ++result.signatures_probed;
        const auto iterator = buckets_.find(candidate_signature);
        if (iterator == buckets_.end()) {
            continue;
        }
        for (const std::size_t index : iterator->second) {
            result.candidate_indices.push_back(index);
            if (result.candidate_indices.size() >=
                config_.maximum_candidates) {
                break;
            }
        }
        if (result.candidate_indices.size() >= config_.maximum_candidates) {
            break;
        }
    }
    if (result.candidate_indices.empty()) {
        const std::size_t count = std::min(
            vector_count_, config_.maximum_candidates
        );
        result.candidate_indices.resize(count);
        for (std::size_t index = 0U; index < count; ++index) {
            result.candidate_indices[index] = index;
        }
    }
    std::sort(
        result.candidate_indices.begin(),
        result.candidate_indices.end()
    );
    result.candidate_indices.erase(
        std::unique(
            result.candidate_indices.begin(),
            result.candidate_indices.end()
        ),
        result.candidate_indices.end()
    );
    result.candidates_examined = result.candidate_indices.size();
    return result;
}

const SparseRouterConfig& SparseRoutingIndex::config() const noexcept {
    return config_;
}

std::size_t SparseRoutingIndex::vector_count() const noexcept {
    return vector_count_;
}

std::size_t SparseRoutingIndex::dimension() const noexcept {
    return dimension_;
}

SparseRouterOperationStats SparseRoutingIndex::operation_stats() const noexcept {
    return operation_stats_;
}

}  // namespace rlf::solstice
