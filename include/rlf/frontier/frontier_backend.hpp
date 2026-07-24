#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace rlf::frontier {

enum class FrontierBackendKind {
    scalar_cpu,
    optimized_cpu,
    cuda,
};

struct BackendCapabilities final {
    bool available{};
    bool deterministic{};
    bool supports_batch{};
    bool supports_local_update{};
    bool supports_candidate_cache{};
    std::size_t maximum_batch{};
    std::uint64_t device_memory_bytes{};
};

struct BackendOperationStats final {
    std::uint64_t batch_cosine_calls{};
    std::uint64_t indexed_batch_cosine_calls{};
    std::uint64_t indexed_cosine_pairs{};
    std::uint64_t cached_batch_cosine_calls{};
    std::uint64_t candidate_cache_uploads{};
    std::uint64_t candidate_cache_hits{};
    std::uint64_t inline_norm_cosine_calls{};
    std::uint64_t precomputed_norm_cosine_calls{};
    std::uint64_t host_batch_cosine_calls{};
    std::uint64_t device_batch_cosine_calls{};
    std::uint64_t host_cosine_fma_operations{};
    std::uint64_t avoided_device_cosine_fma_operations{};
    std::uint64_t candidate_norm_cache_uploads{};
    std::uint64_t avoided_pairwise_norm_fma_operations{};
    std::uint64_t host_precomputed_norm_fma_operations{};
    std::uint64_t local_update_calls{};
    std::uint64_t host_local_update_calls{};
    std::uint64_t device_local_update_calls{};
    std::uint64_t host_to_device_bytes{};
    std::uint64_t device_to_host_bytes{};
    std::uint64_t kernel_launches{};
    std::uint64_t stream_synchronizations{};
    std::uint64_t avoided_host_to_device_bytes{};
    std::uint64_t avoided_device_to_host_bytes{};
    std::uint64_t avoided_kernel_launches{};
    std::uint64_t avoided_stream_synchronizations{};
    std::size_t host_local_update_maximum_dimensions{};
    std::size_t host_batch_cosine_maximum_fma_operations{};
    bool hybrid_local_updates{};
    bool hybrid_small_batch_cosine{};
    bool precomputed_cached_cosine_norms{};
};

class FrontierComputeBackend {
public:
    virtual ~FrontierComputeBackend() = default;
    [[nodiscard]] virtual FrontierBackendKind kind() const noexcept = 0;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual BackendCapabilities capabilities() const noexcept = 0;
    [[nodiscard]] virtual BackendOperationStats operation_stats() const noexcept = 0;
    [[nodiscard]] virtual std::vector<float> batch_cosine(
        std::span<const float> queries,
        std::size_t query_count,
        std::span<const float> candidates,
        std::size_t candidate_count,
        std::size_t dimension
    ) const = 0;
    [[nodiscard]] virtual std::vector<float> batch_cosine_indexed(
        std::span<const float> queries,
        std::size_t query_count,
        std::span<const float> candidates,
        std::span<const std::size_t> candidate_query_indices,
        std::size_t dimension
    ) const = 0;
    virtual void prepare_candidate_cache(
        std::span<const float> candidates,
        std::size_t candidate_count,
        std::size_t dimension,
        std::uint64_t cache_key
    ) const = 0;
    [[nodiscard]] virtual std::vector<float> batch_cosine_cached(
        std::span<const float> queries,
        std::size_t query_count,
        std::size_t candidate_offset,
        std::size_t candidate_count,
        std::size_t dimension,
        std::uint64_t cache_key
    ) const = 0;
    virtual void local_average_update(
        std::span<float> prototype,
        std::span<const float> observation,
        float learning_rate
    ) const = 0;
};

[[nodiscard]] std::unique_ptr<FrontierComputeBackend> make_frontier_backend(
    FrontierBackendKind kind
);
[[nodiscard]] std::string_view to_string(FrontierBackendKind kind) noexcept;

}  // namespace rlf::frontier
