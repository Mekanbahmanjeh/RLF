#include "rlf/frontier/frontier_backend.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>

namespace rlf::frontier {
namespace {

void validate_batch(
    const std::span<const float> queries,
    const std::size_t query_count,
    const std::span<const float> candidates,
    const std::size_t candidate_count,
    const std::size_t dimension
) {
    if (query_count == 0U || candidate_count == 0U || dimension == 0U) {
        throw std::invalid_argument("batch dimensions must be non-zero");
    }
    if (query_count > std::numeric_limits<std::size_t>::max() / dimension ||
        candidate_count > std::numeric_limits<std::size_t>::max() / dimension ||
        queries.size() != query_count * dimension ||
        candidates.size() != candidate_count * dimension) {
        throw std::invalid_argument("batch storage does not match dimensions");
    }
}

void validate_indexed_batch(
    const std::span<const float> queries,
    const std::size_t query_count,
    const std::span<const float> candidates,
    const std::span<const std::size_t> candidate_query_indices,
    const std::size_t dimension
) {
    const std::size_t pair_count = candidate_query_indices.size();
    if (query_count == 0U || pair_count == 0U || dimension == 0U ||
        query_count > std::numeric_limits<std::size_t>::max() / dimension ||
        pair_count > std::numeric_limits<std::size_t>::max() / dimension ||
        queries.size() != query_count * dimension ||
        candidates.size() != pair_count * dimension ||
        std::any_of(
            candidate_query_indices.begin(), candidate_query_indices.end(),
            [query_count](const std::size_t query) { return query >= query_count; }
        )) {
        throw std::invalid_argument("indexed batch storage does not match dimensions");
    }
}

[[nodiscard]] float cosine(
    const float* left,
    const float* right,
    const std::size_t dimension
) noexcept {
    double dot = 0.0;
    double left_norm = 0.0;
    double right_norm = 0.0;
    for (std::size_t index = 0U; index < dimension; ++index) {
        const double l = static_cast<double>(left[index]);
        const double r = static_cast<double>(right[index]);
        dot += l * r;
        left_norm += l * l;
        right_norm += r * r;
    }
    if (left_norm <= std::numeric_limits<double>::epsilon() ||
        right_norm <= std::numeric_limits<double>::epsilon()) return 0.0F;
    return static_cast<float>(std::clamp(dot / std::sqrt(left_norm * right_norm), -1.0, 1.0));
}

class CpuFrontierBackend final : public FrontierComputeBackend {
public:
    explicit CpuFrontierBackend(const FrontierBackendKind kind) : kind_(kind) {}

    [[nodiscard]] FrontierBackendKind kind() const noexcept override { return kind_; }
    [[nodiscard]] std::string_view name() const noexcept override { return to_string(kind_); }
    [[nodiscard]] BackendCapabilities capabilities() const noexcept override {
        return {
            .available = true,
            .deterministic = true,
            .supports_batch = true,
            .supports_local_update = true,
            .supports_candidate_cache = true,
            .maximum_batch = std::numeric_limits<std::size_t>::max(),
            .device_memory_bytes = 0U,
        };
    }
    [[nodiscard]] BackendOperationStats operation_stats() const noexcept override {
        return operation_stats_;
    }
    [[nodiscard]] std::vector<float> batch_cosine(
        const std::span<const float> queries,
        const std::size_t query_count,
        const std::span<const float> candidates,
        const std::size_t candidate_count,
        const std::size_t dimension
    ) const override {
        validate_batch(queries, query_count, candidates, candidate_count, dimension);
        ++operation_stats_.batch_cosine_calls;
        ++operation_stats_.inline_norm_cosine_calls;
        ++operation_stats_.host_batch_cosine_calls;
        std::vector<float> result(query_count * candidate_count);
        const std::size_t pair_count = query_count * candidate_count;
        const auto evaluate_range = [&](const std::size_t begin, const std::size_t end) {
            for (std::size_t pair = begin; pair < end; ++pair) {
                const std::size_t query = pair / candidate_count;
                const std::size_t candidate = pair % candidate_count;
                result[pair] = cosine(
                    queries.data() + query * dimension,
                    candidates.data() + candidate * dimension,
                    dimension
                );
            }
        };
        if (kind_ == FrontierBackendKind::optimized_cpu && pair_count >= 4'096U) {
            constexpr std::size_t maximum_workers = 16U;
            const std::size_t available = std::min<std::size_t>(
                maximum_workers,
                std::max<std::size_t>(
                    1U,
                    static_cast<std::size_t>(std::thread::hardware_concurrency())
                )
            );
            const std::size_t worker_count = std::min<std::size_t>(available, pair_count);
            const std::size_t block = (pair_count + worker_count - 1U) / worker_count;
            std::vector<std::jthread> workers;
            workers.reserve(worker_count);
            for (std::size_t worker = 0U; worker < worker_count; ++worker) {
                const std::size_t begin = worker * block;
                const std::size_t end = std::min(pair_count, begin + block);
                if (begin >= end) break;
                workers.emplace_back(evaluate_range, begin, end);
            }
        } else {
            evaluate_range(0U, pair_count);
        }
        return result;
    }
    [[nodiscard]] std::vector<float> batch_cosine_indexed(
        const std::span<const float> queries,
        const std::size_t query_count,
        const std::span<const float> candidates,
        const std::span<const std::size_t> candidate_query_indices,
        const std::size_t dimension
    ) const override {
        validate_indexed_batch(
            queries, query_count, candidates, candidate_query_indices, dimension
        );
        ++operation_stats_.batch_cosine_calls;
        ++operation_stats_.indexed_batch_cosine_calls;
        operation_stats_.indexed_cosine_pairs += candidate_query_indices.size();
        ++operation_stats_.inline_norm_cosine_calls;
        ++operation_stats_.host_batch_cosine_calls;
        std::vector<float> result(candidate_query_indices.size());
        const auto evaluate_range = [&](const std::size_t begin, const std::size_t end) {
            for (std::size_t pair = begin; pair < end; ++pair) {
                result[pair] = cosine(
                    queries.data() + candidate_query_indices[pair] * dimension,
                    candidates.data() + pair * dimension,
                    dimension
                );
            }
        };
        if (kind_ == FrontierBackendKind::optimized_cpu && result.size() >= 65'536U) {
            constexpr std::size_t maximum_workers = 16U;
            const std::size_t available = std::min<std::size_t>(
                maximum_workers,
                std::max<std::size_t>(
                    1U, static_cast<std::size_t>(std::thread::hardware_concurrency())
                )
            );
            const std::size_t worker_count = std::min(available, result.size());
            const std::size_t block = (result.size() + worker_count - 1U) / worker_count;
            std::vector<std::jthread> workers;
            workers.reserve(worker_count);
            for (std::size_t worker = 0U; worker < worker_count; ++worker) {
                const std::size_t begin = worker * block;
                const std::size_t end = std::min(result.size(), begin + block);
                if (begin >= end) break;
                workers.emplace_back(evaluate_range, begin, end);
            }
        } else {
            evaluate_range(0U, result.size());
        }
        return result;
    }
    void prepare_candidate_cache(
        const std::span<const float> candidates,
        const std::size_t candidate_count,
        const std::size_t dimension,
        const std::uint64_t cache_key
    ) const override {
        if (candidate_count == 0U || dimension == 0U ||
            candidate_count > std::numeric_limits<std::size_t>::max() / dimension ||
            candidates.size() != candidate_count * dimension) {
            throw std::invalid_argument("invalid CPU candidate cache dimensions");
        }
        if (candidate_cache_key_ == cache_key &&
            candidate_cache_count_ == candidate_count &&
            candidate_cache_dimension_ == dimension) {
            ++operation_stats_.candidate_cache_hits;
            return;
        }
        ++operation_stats_.candidate_cache_uploads;
        candidate_cache_.assign(candidates.begin(), candidates.end());
        candidate_cache_count_ = candidate_count;
        candidate_cache_dimension_ = dimension;
        candidate_cache_key_ = cache_key;
    }
    [[nodiscard]] std::vector<float> batch_cosine_cached(
        const std::span<const float> queries,
        const std::size_t query_count,
        const std::size_t candidate_offset,
        const std::size_t candidate_count,
        const std::size_t dimension,
        const std::uint64_t cache_key
    ) const override {
        if (candidate_cache_key_ != cache_key ||
            candidate_cache_dimension_ != dimension ||
            candidate_offset > candidate_cache_count_ ||
            candidate_count > candidate_cache_count_ - candidate_offset) {
            throw std::runtime_error("CPU candidate cache is unavailable or stale");
        }
        ++operation_stats_.cached_batch_cosine_calls;
        return batch_cosine(
            queries,
            query_count,
            std::span<const float>(
                candidate_cache_.data() + candidate_offset * dimension,
                candidate_count * dimension
            ),
            candidate_count,
            dimension
        );
    }
    void local_average_update(
        const std::span<float> prototype,
        const std::span<const float> observation,
        const float learning_rate
    ) const override {
        if (prototype.size() != observation.size() || prototype.empty() ||
            !std::isfinite(static_cast<double>(learning_rate)) ||
            learning_rate < 0.0F || learning_rate > 1.0F) {
            throw std::invalid_argument("invalid local average update");
        }
        ++operation_stats_.local_update_calls;
        ++operation_stats_.host_local_update_calls;
        for (std::size_t index = 0U; index < prototype.size(); ++index) {
            prototype[index] += learning_rate * (observation[index] - prototype[index]);
        }
    }

private:
    FrontierBackendKind kind_;
    mutable std::vector<float> candidate_cache_;
    mutable std::size_t candidate_cache_count_{};
    mutable std::size_t candidate_cache_dimension_{};
    mutable std::uint64_t candidate_cache_key_{};
    mutable BackendOperationStats operation_stats_{};
};

#ifndef RLF_HAS_CUDA
class UnavailableCudaBackend final : public FrontierComputeBackend {
public:
    [[nodiscard]] FrontierBackendKind kind() const noexcept override { return FrontierBackendKind::cuda; }
    [[nodiscard]] std::string_view name() const noexcept override { return "cuda_unavailable"; }
    [[nodiscard]] BackendCapabilities capabilities() const noexcept override { return {}; }
    [[nodiscard]] BackendOperationStats operation_stats() const noexcept override { return {}; }
    [[nodiscard]] std::vector<float> batch_cosine(
        std::span<const float>, std::size_t, std::span<const float>, std::size_t, std::size_t
    ) const override {
        throw std::runtime_error("CUDA backend was not compiled; configure with -DRLF_ENABLE_CUDA=ON and a CUDA toolkit");
    }
    [[nodiscard]] std::vector<float> batch_cosine_indexed(
        std::span<const float>, std::size_t, std::span<const float>,
        std::span<const std::size_t>, std::size_t
    ) const override {
        throw std::runtime_error("CUDA backend was not compiled");
    }
    void prepare_candidate_cache(
        std::span<const float>, std::size_t, std::size_t, std::uint64_t
    ) const override {
        throw std::runtime_error("CUDA backend was not compiled");
    }
    [[nodiscard]] std::vector<float> batch_cosine_cached(
        std::span<const float>, std::size_t, std::size_t, std::size_t,
        std::size_t, std::uint64_t
    ) const override {
        throw std::runtime_error("CUDA backend was not compiled");
    }
    void local_average_update(
        std::span<float>, std::span<const float>, float
    ) const override {
        throw std::runtime_error("CUDA backend was not compiled");
    }
};
#endif

}  // namespace

#ifdef RLF_HAS_CUDA
[[nodiscard]] std::unique_ptr<FrontierComputeBackend> make_cuda_frontier_backend();
#endif

std::string_view to_string(const FrontierBackendKind kind) noexcept {
    switch (kind) {
    case FrontierBackendKind::scalar_cpu: return "scalar_cpu";
    case FrontierBackendKind::optimized_cpu: return "optimized_cpu";
    case FrontierBackendKind::cuda: return "cuda";
    }
    return "unknown";
}

std::unique_ptr<FrontierComputeBackend> make_frontier_backend(
    const FrontierBackendKind kind
) {
    switch (kind) {
    case FrontierBackendKind::scalar_cpu:
    case FrontierBackendKind::optimized_cpu:
        return std::make_unique<CpuFrontierBackend>(kind);
    case FrontierBackendKind::cuda:
#ifdef RLF_HAS_CUDA
        return make_cuda_frontier_backend();
#else
        return std::make_unique<UnavailableCudaBackend>();
#endif
    }
    throw std::invalid_argument("unknown Frontier backend kind");
}

}  // namespace rlf::frontier
