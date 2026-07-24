#include "rlf/frontier/frontier_backend.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace rlf::frontier {
namespace {

void check_cuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__global__ void cosine_kernel(
    const float* queries,
    const std::size_t query_count,
    const float* candidates,
    const std::size_t candidate_count,
    const std::size_t dimension,
    float* output
) {
    const std::size_t pair = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t pair_count = query_count * candidate_count;
    if (pair >= pair_count) {
        return;
    }
    const std::size_t query = pair / candidate_count;
    const std::size_t candidate = pair % candidate_count;
    float dot = 0.0F;
    float query_norm = 0.0F;
    float candidate_norm = 0.0F;
    for (std::size_t index = 0U; index < dimension; ++index) {
        const float left = queries[query * dimension + index];
        const float right = candidates[candidate * dimension + index];
        dot = fmaf(left, right, dot);
        query_norm = fmaf(left, left, query_norm);
        candidate_norm = fmaf(right, right, candidate_norm);
    }
    output[pair] = query_norm > 0.0F && candidate_norm > 0.0F
        ? dot * rsqrtf(query_norm * candidate_norm)
        : 0.0F;
}

__global__ void cosine_indexed_kernel(
    const float* queries,
    const float* candidates,
    const std::size_t* candidate_query_indices,
    const std::size_t pair_count,
    const std::size_t dimension,
    float* output
) {
    const std::size_t pair = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= pair_count) return;
    const std::size_t query = candidate_query_indices[pair];
    float dot = 0.0F;
    float query_norm = 0.0F;
    float candidate_norm = 0.0F;
    for (std::size_t index = 0U; index < dimension; ++index) {
        const float left = queries[query * dimension + index];
        const float right = candidates[pair * dimension + index];
        dot = fmaf(left, right, dot);
        query_norm = fmaf(left, left, query_norm);
        candidate_norm = fmaf(right, right, candidate_norm);
    }
    output[pair] = query_norm > 0.0F && candidate_norm > 0.0F
        ? dot * rsqrtf(query_norm * candidate_norm)
        : 0.0F;
}

__global__ void cosine_precomputed_norms_kernel(
    const float* queries,
    const std::size_t query_count,
    const float* query_norms,
    const float* candidates,
    const float* candidate_norms,
    const std::size_t candidate_count,
    const std::size_t dimension,
    float* output
) {
    const std::size_t pair = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t pair_count = query_count * candidate_count;
    if (pair >= pair_count) {
        return;
    }
    const std::size_t query = pair / candidate_count;
    const std::size_t candidate = pair % candidate_count;
    float dot = 0.0F;
    for (std::size_t index = 0U; index < dimension; ++index) {
        dot = fmaf(
            queries[query * dimension + index],
            candidates[candidate * dimension + index],
            dot
        );
    }
    const float query_norm = query_norms[query];
    const float candidate_norm = candidate_norms[candidate];
    output[pair] = query_norm > 0.0F && candidate_norm > 0.0F
        ? dot * rsqrtf(query_norm * candidate_norm)
        : 0.0F;
}

__global__ void local_update_kernel(
    float* prototype,
    const float* observation,
    const std::size_t dimension,
    const float learning_rate
) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < dimension) {
        prototype[index] = fmaf(
            learning_rate,
            observation[index] - prototype[index],
            prototype[index]
        );
    }
}

[[nodiscard]] std::size_t growth_capacity(const std::size_t required) {
    std::size_t capacity = 1U;
    while (capacity < required) {
        if (capacity > std::numeric_limits<std::size_t>::max() / 2U) {
            return required;
        }
        capacity *= 2U;
    }
    return capacity;
}

constexpr std::size_t host_local_update_maximum_dimensions = 256U;
constexpr std::size_t host_batch_cosine_maximum_fma_operations = 98'304U;

[[nodiscard]] bool hybrid_local_updates_from_environment() {
    const char* const value = std::getenv("RLF_CUDA_LOCAL_UPDATE_POLICY");
    if (value == nullptr || std::string_view(value).empty() ||
        std::string_view(value) == "hybrid") {
        return true;
    }
    if (std::string_view(value) == "device") {
        return false;
    }
    throw std::invalid_argument(
        "RLF_CUDA_LOCAL_UPDATE_POLICY must be hybrid or device"
    );
}

[[nodiscard]] bool precomputed_cached_cosine_norms_from_environment() {
    const char* const value = std::getenv("RLF_CUDA_CACHED_COSINE_POLICY");
    if (value == nullptr || std::string_view(value).empty() ||
        std::string_view(value) == "precomputed_norms") {
        return true;
    }
    if (std::string_view(value) == "inline_norms") {
        return false;
    }
    throw std::invalid_argument(
        "RLF_CUDA_CACHED_COSINE_POLICY must be precomputed_norms or inline_norms"
    );
}

[[nodiscard]] bool hybrid_small_batch_cosine_from_environment() {
    const char* const value = std::getenv("RLF_CUDA_SMALL_COSINE_POLICY");
    if (value == nullptr || std::string_view(value).empty() ||
        std::string_view(value) == "hybrid") {
        return true;
    }
    if (std::string_view(value) == "device") {
        return false;
    }
    throw std::invalid_argument(
        "RLF_CUDA_SMALL_COSINE_POLICY must be hybrid or device"
    );
}

[[nodiscard]] std::vector<float> host_float_cosine(
    const std::span<const float> queries,
    const std::size_t query_count,
    const std::span<const float> candidates,
    const std::size_t candidate_count,
    const std::size_t dimension
) {
    std::vector<float> result(query_count * candidate_count, 0.0F);
    for (std::size_t query = 0U; query < query_count; ++query) {
        for (std::size_t candidate = 0U; candidate < candidate_count; ++candidate) {
            float dot = 0.0F;
            float query_norm = 0.0F;
            float candidate_norm = 0.0F;
            for (std::size_t index = 0U; index < dimension; ++index) {
                const float left = queries[query * dimension + index];
                const float right = candidates[candidate * dimension + index];
                dot = std::fma(left, right, dot);
                query_norm = std::fma(left, left, query_norm);
                candidate_norm = std::fma(right, right, candidate_norm);
            }
            result[query * candidate_count + candidate] =
                query_norm > 0.0F && candidate_norm > 0.0F
                ? dot / std::sqrt(query_norm * candidate_norm)
                : 0.0F;
        }
    }
    return result;
}

[[nodiscard]] std::vector<float> host_float_cosine_indexed(
    const std::span<const float> queries,
    const std::span<const float> candidates,
    const std::span<const std::size_t> candidate_query_indices,
    const std::size_t dimension
) {
    std::vector<float> result(candidate_query_indices.size(), 0.0F);
    for (std::size_t pair = 0U; pair < candidate_query_indices.size(); ++pair) {
        const std::size_t query = candidate_query_indices[pair];
        float dot = 0.0F;
        float query_norm = 0.0F;
        float candidate_norm = 0.0F;
        for (std::size_t index = 0U; index < dimension; ++index) {
            const float left = queries[query * dimension + index];
            const float right = candidates[pair * dimension + index];
            dot = std::fma(left, right, dot);
            query_norm = std::fma(left, left, query_norm);
            candidate_norm = std::fma(right, right, candidate_norm);
        }
        result[pair] = query_norm > 0.0F && candidate_norm > 0.0F
            ? dot / std::sqrt(query_norm * candidate_norm)
            : 0.0F;
    }
    return result;
}

void saturating_increment(std::uint64_t& value) noexcept {
    if (value != std::numeric_limits<std::uint64_t>::max()) {
        ++value;
    }
}

void saturating_add_bytes(
    std::uint64_t& destination,
    const std::size_t bytes
) noexcept {
    const auto amount = static_cast<std::uint64_t>(bytes);
    destination = amount > std::numeric_limits<std::uint64_t>::max() - destination
        ? std::numeric_limits<std::uint64_t>::max()
        : destination + amount;
}

void saturating_add_product(
    std::uint64_t& destination,
    const std::size_t first,
    const std::size_t second,
    const std::size_t third
) noexcept {
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    const auto a = static_cast<std::uint64_t>(first);
    const auto b = static_cast<std::uint64_t>(second);
    const auto c = static_cast<std::uint64_t>(third);
    if ((a != 0U && b > maximum / a) ||
        (a * b != 0U && c > maximum / (a * b))) {
        destination = maximum;
        return;
    }
    const std::uint64_t amount = a * b * c;
    destination = amount > maximum - destination ? maximum : destination + amount;
}

class CudaFrontierBackend final : public FrontierComputeBackend {
public:
    CudaFrontierBackend()
        : hybrid_local_updates_(hybrid_local_updates_from_environment()),
          precomputed_cached_cosine_norms_(
              precomputed_cached_cosine_norms_from_environment()
          ),
          hybrid_small_batch_cosine_(hybrid_small_batch_cosine_from_environment()) {
        int count = 0;
        check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        if (count == 0) {
            throw std::runtime_error("no CUDA device is available");
        }
        check_cuda(cudaSetDevice(0), "cudaSetDevice");
        check_cuda(cudaGetDeviceProperties(&properties_, 0), "cudaGetDeviceProperties");
        check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate");
    }

    ~CudaFrontierBackend() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (device_candidate_query_indices_ != nullptr) {
            static_cast<void>(cudaFree(device_candidate_query_indices_));
            device_candidate_query_indices_ = nullptr;
        }
        release_noexcept(device_output_);
        release_noexcept(device_candidate_norm_cache_);
        release_noexcept(device_query_norms_);
        release_noexcept(device_candidate_cache_);
        release_noexcept(device_candidates_);
        release_noexcept(device_queries_);
        if (stream_ != nullptr) {
            static_cast<void>(cudaStreamDestroy(stream_));
            stream_ = nullptr;
        }
    }

    CudaFrontierBackend(const CudaFrontierBackend&) = delete;
    CudaFrontierBackend& operator=(const CudaFrontierBackend&) = delete;

    [[nodiscard]] FrontierBackendKind kind() const noexcept override {
        return FrontierBackendKind::cuda;
    }

    [[nodiscard]] std::string_view name() const noexcept override {
        return "cuda-persistent";
    }

    [[nodiscard]] BackendCapabilities capabilities() const noexcept override {
        return {
            .available = true,
            .deterministic = true,
            .supports_batch = true,
            .supports_local_update = true,
            .supports_candidate_cache = true,
            .maximum_batch = static_cast<std::size_t>(properties_.maxGridSize[0]) * 256U,
            .device_memory_bytes = static_cast<std::uint64_t>(properties_.totalGlobalMem),
        };
    }

    [[nodiscard]] BackendOperationStats operation_stats() const noexcept override {
        std::lock_guard<std::mutex> lock(mutex_);
        BackendOperationStats result = operation_stats_;
        result.host_local_update_maximum_dimensions =
            host_local_update_maximum_dimensions;
        result.host_batch_cosine_maximum_fma_operations =
            host_batch_cosine_maximum_fma_operations;
        result.hybrid_local_updates = hybrid_local_updates_;
        result.hybrid_small_batch_cosine = hybrid_small_batch_cosine_;
        result.precomputed_cached_cosine_norms = precomputed_cached_cosine_norms_;
        return result;
    }

    [[nodiscard]] std::vector<float> batch_cosine(
        const std::span<const float> queries,
        const std::size_t query_count,
        const std::span<const float> candidates,
        const std::size_t candidate_count,
        const std::size_t dimension
    ) const override {
        if (query_count == 0U || candidate_count == 0U || dimension == 0U ||
            query_count > std::numeric_limits<std::size_t>::max() / dimension ||
            candidate_count > std::numeric_limits<std::size_t>::max() / dimension ||
            query_count > std::numeric_limits<std::size_t>::max() / candidate_count ||
            queries.size() != query_count * dimension ||
            candidates.size() != candidate_count * dimension) {
            throw std::invalid_argument("invalid CUDA batch dimensions");
        }

        const std::size_t output_count = query_count * candidate_count;
        std::lock_guard<std::mutex> lock(mutex_);
        saturating_increment(operation_stats_.batch_cosine_calls);
        saturating_increment(operation_stats_.inline_norm_cosine_calls);
        const bool use_host = hybrid_small_batch_cosine_ &&
            dimension <= host_batch_cosine_maximum_fma_operations / 3U &&
            output_count <=
                host_batch_cosine_maximum_fma_operations / (dimension * 3U);
        if (use_host) {
            saturating_increment(operation_stats_.host_batch_cosine_calls);
            saturating_add_product(
                operation_stats_.host_cosine_fma_operations,
                output_count, dimension, 3U
            );
            saturating_add_product(
                operation_stats_.avoided_device_cosine_fma_operations,
                output_count, dimension, 3U
            );
            saturating_add_bytes(
                operation_stats_.avoided_host_to_device_bytes,
                queries.size_bytes()
            );
            saturating_add_bytes(
                operation_stats_.avoided_host_to_device_bytes,
                candidates.size_bytes()
            );
            saturating_add_bytes(
                operation_stats_.avoided_device_to_host_bytes,
                output_count * sizeof(float)
            );
            saturating_increment(operation_stats_.avoided_kernel_launches);
            saturating_increment(operation_stats_.avoided_stream_synchronizations);
            return host_float_cosine(
                queries, query_count, candidates, candidate_count, dimension
            );
        }
        saturating_increment(operation_stats_.device_batch_cosine_calls);
        std::vector<float> result(output_count);
        ensure_capacity(device_queries_, query_capacity_, queries.size(), "cudaMalloc query workspace");
        ensure_capacity(
            device_candidates_, candidate_capacity_, candidates.size(),
            "cudaMalloc candidate workspace"
        );
        ensure_capacity(device_output_, output_capacity_, output_count, "cudaMalloc output workspace");

        check_cuda(
            cudaMemcpyAsync(
                device_queries_, queries.data(), queries.size_bytes(),
                cudaMemcpyHostToDevice, stream_
            ),
            "copy queries"
        );
        saturating_add_bytes(operation_stats_.host_to_device_bytes, queries.size_bytes());
        check_cuda(
            cudaMemcpyAsync(
                device_candidates_, candidates.data(), candidates.size_bytes(),
                cudaMemcpyHostToDevice, stream_
            ),
            "copy candidates"
        );
        saturating_add_bytes(operation_stats_.host_to_device_bytes, candidates.size_bytes());
        constexpr unsigned int threads = 256U;
        const auto blocks_required = (output_count + threads - 1U) / threads;
        if (blocks_required > static_cast<std::size_t>(properties_.maxGridSize[0])) {
            throw std::runtime_error("CUDA cosine output exceeds device grid capacity");
        }
        const auto blocks = static_cast<unsigned int>(blocks_required);
        cosine_kernel<<<blocks, threads, 0U, stream_>>>(
            device_queries_, query_count, device_candidates_, candidate_count,
            dimension, device_output_
        );
        saturating_increment(operation_stats_.kernel_launches);
        check_cuda(cudaGetLastError(), "cosine kernel launch");
        check_cuda(
            cudaMemcpyAsync(
                result.data(), device_output_, result.size() * sizeof(float),
                cudaMemcpyDeviceToHost, stream_
            ),
            "copy cosine output"
        );
        saturating_add_bytes(
            operation_stats_.device_to_host_bytes,
            result.size() * sizeof(float)
        );
        check_cuda(cudaStreamSynchronize(stream_), "cosine stream synchronize");
        saturating_increment(operation_stats_.stream_synchronizations);
        return result;
    }

    [[nodiscard]] std::vector<float> batch_cosine_indexed(
        const std::span<const float> queries,
        const std::size_t query_count,
        const std::span<const float> candidates,
        const std::span<const std::size_t> candidate_query_indices,
        const std::size_t dimension
    ) const override {
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
            throw std::invalid_argument("invalid indexed CUDA batch dimensions");
        }

        std::lock_guard<std::mutex> lock(mutex_);
        saturating_increment(operation_stats_.batch_cosine_calls);
        saturating_increment(operation_stats_.indexed_batch_cosine_calls);
        saturating_add_bytes(operation_stats_.indexed_cosine_pairs, pair_count);
        saturating_increment(operation_stats_.inline_norm_cosine_calls);
        const bool use_host = hybrid_small_batch_cosine_ &&
            dimension <= host_batch_cosine_maximum_fma_operations / 3U &&
            pair_count <= host_batch_cosine_maximum_fma_operations / (dimension * 3U);
        if (use_host) {
            saturating_increment(operation_stats_.host_batch_cosine_calls);
            saturating_add_product(
                operation_stats_.host_cosine_fma_operations,
                pair_count, dimension, 3U
            );
            saturating_add_product(
                operation_stats_.avoided_device_cosine_fma_operations,
                pair_count, dimension, 3U
            );
            saturating_add_bytes(
                operation_stats_.avoided_host_to_device_bytes, queries.size_bytes()
            );
            saturating_add_bytes(
                operation_stats_.avoided_host_to_device_bytes, candidates.size_bytes()
            );
            saturating_add_bytes(
                operation_stats_.avoided_host_to_device_bytes,
                candidate_query_indices.size_bytes()
            );
            saturating_add_bytes(
                operation_stats_.avoided_device_to_host_bytes,
                pair_count * sizeof(float)
            );
            saturating_increment(operation_stats_.avoided_kernel_launches);
            saturating_increment(operation_stats_.avoided_stream_synchronizations);
            return host_float_cosine_indexed(
                queries, candidates, candidate_query_indices, dimension
            );
        }

        saturating_increment(operation_stats_.device_batch_cosine_calls);
        std::vector<float> result(pair_count);
        ensure_capacity(device_queries_, query_capacity_, queries.size(),
                        "cudaMalloc indexed query workspace");
        ensure_capacity(device_candidates_, candidate_capacity_, candidates.size(),
                        "cudaMalloc indexed candidate workspace");
        ensure_index_capacity(pair_count);
        ensure_capacity(device_output_, output_capacity_, pair_count,
                        "cudaMalloc indexed output workspace");
        check_cuda(
            cudaMemcpyAsync(
                device_queries_, queries.data(), queries.size_bytes(),
                cudaMemcpyHostToDevice, stream_
            ),
            "copy indexed queries"
        );
        saturating_add_bytes(operation_stats_.host_to_device_bytes, queries.size_bytes());
        check_cuda(
            cudaMemcpyAsync(
                device_candidates_, candidates.data(), candidates.size_bytes(),
                cudaMemcpyHostToDevice, stream_
            ),
            "copy indexed candidates"
        );
        saturating_add_bytes(operation_stats_.host_to_device_bytes, candidates.size_bytes());
        check_cuda(
            cudaMemcpyAsync(
                device_candidate_query_indices_, candidate_query_indices.data(),
                candidate_query_indices.size_bytes(), cudaMemcpyHostToDevice, stream_
            ),
            "copy indexed query mapping"
        );
        saturating_add_bytes(
            operation_stats_.host_to_device_bytes, candidate_query_indices.size_bytes()
        );
        constexpr unsigned int threads = 256U;
        const auto blocks_required = (pair_count + threads - 1U) / threads;
        if (blocks_required > static_cast<std::size_t>(properties_.maxGridSize[0])) {
            throw std::runtime_error("indexed CUDA cosine output exceeds device grid capacity");
        }
        cosine_indexed_kernel<<<static_cast<unsigned int>(blocks_required), threads, 0U, stream_>>>(
            device_queries_, device_candidates_, device_candidate_query_indices_,
            pair_count, dimension, device_output_
        );
        saturating_increment(operation_stats_.kernel_launches);
        check_cuda(cudaGetLastError(), "indexed cosine kernel launch");
        check_cuda(
            cudaMemcpyAsync(
                result.data(), device_output_, result.size() * sizeof(float),
                cudaMemcpyDeviceToHost, stream_
            ),
            "copy indexed cosine output"
        );
        saturating_add_bytes(
            operation_stats_.device_to_host_bytes, result.size() * sizeof(float)
        );
        check_cuda(cudaStreamSynchronize(stream_), "indexed cosine stream synchronize");
        saturating_increment(operation_stats_.stream_synchronizations);
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
            throw std::invalid_argument("invalid CUDA candidate cache dimensions");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (candidate_cache_key_ == cache_key &&
            candidate_cache_count_ == candidate_count &&
            candidate_cache_dimension_ == dimension) {
            saturating_increment(operation_stats_.candidate_cache_hits);
            return;
        }
        saturating_increment(operation_stats_.candidate_cache_uploads);
        ensure_capacity(
            device_candidate_cache_, candidate_cache_capacity_, candidates.size(),
            "cudaMalloc persistent candidate cache"
        );
        saturating_add_bytes(
            operation_stats_.host_to_device_bytes,
            candidates.size_bytes()
        );
        check_cuda(
            cudaMemcpyAsync(
                device_candidate_cache_, candidates.data(), candidates.size_bytes(),
                cudaMemcpyHostToDevice, stream_
            ),
            "copy persistent candidate cache"
        );
        if (precomputed_cached_cosine_norms_) {
            std::vector<float> candidate_norms(candidate_count, 0.0F);
            for (std::size_t candidate = 0U; candidate < candidate_count; ++candidate) {
                float norm = 0.0F;
                for (std::size_t index = 0U; index < dimension; ++index) {
                    const float value = candidates[candidate * dimension + index];
                    norm = std::fma(value, value, norm);
                }
                candidate_norms[candidate] = norm;
            }
            ensure_capacity(
                device_candidate_norm_cache_, candidate_norm_cache_capacity_,
                candidate_norms.size(), "cudaMalloc persistent candidate norm cache"
            );
            check_cuda(
                cudaMemcpyAsync(
                    device_candidate_norm_cache_, candidate_norms.data(),
                    candidate_norms.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream_
                ),
                "copy persistent candidate norm cache"
            );
            saturating_add_bytes(
                operation_stats_.host_to_device_bytes,
                candidate_norms.size() * sizeof(float)
            );
            saturating_increment(operation_stats_.candidate_norm_cache_uploads);
            saturating_add_product(
                operation_stats_.host_precomputed_norm_fma_operations,
                candidate_count, dimension, 1U
            );
        }
        saturating_increment(operation_stats_.stream_synchronizations);
        check_cuda(
            cudaStreamSynchronize(stream_),
            "candidate cache stream synchronize"
        );
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
        if (query_count == 0U || candidate_count == 0U || dimension == 0U ||
            query_count > std::numeric_limits<std::size_t>::max() / dimension ||
            query_count > std::numeric_limits<std::size_t>::max() / candidate_count ||
            queries.size() != query_count * dimension) {
            throw std::invalid_argument("invalid cached CUDA batch dimensions");
        }
        const std::size_t output_count = query_count * candidate_count;
        std::vector<float> result(output_count);
        std::lock_guard<std::mutex> lock(mutex_);
        saturating_increment(operation_stats_.batch_cosine_calls);
        saturating_increment(operation_stats_.cached_batch_cosine_calls);
        if (candidate_cache_key_ != cache_key ||
            candidate_cache_dimension_ != dimension ||
            candidate_offset > candidate_cache_count_ ||
            candidate_count > candidate_cache_count_ - candidate_offset) {
            throw std::runtime_error("CUDA candidate cache is unavailable or stale");
        }
        ensure_capacity(device_queries_, query_capacity_, queries.size(),
                        "cudaMalloc query workspace");
        ensure_capacity(device_output_, output_capacity_, output_count,
                        "cudaMalloc output workspace");
        check_cuda(
            cudaMemcpyAsync(
                device_queries_, queries.data(), queries.size_bytes(),
                cudaMemcpyHostToDevice, stream_
            ),
            "copy cached queries"
        );
        saturating_add_bytes(operation_stats_.host_to_device_bytes, queries.size_bytes());
        constexpr unsigned int threads = 256U;
        const auto blocks_required = (output_count + threads - 1U) / threads;
        if (blocks_required > static_cast<std::size_t>(properties_.maxGridSize[0])) {
            throw std::runtime_error("cached CUDA cosine output exceeds device grid capacity");
        }
        const auto blocks = static_cast<unsigned int>(blocks_required);
        if (precomputed_cached_cosine_norms_) {
            std::vector<float> query_norms(query_count, 0.0F);
            for (std::size_t query = 0U; query < query_count; ++query) {
                float norm = 0.0F;
                for (std::size_t index = 0U; index < dimension; ++index) {
                    const float value = queries[query * dimension + index];
                    norm = std::fma(value, value, norm);
                }
                query_norms[query] = norm;
            }
            ensure_capacity(
                device_query_norms_, query_norm_capacity_, query_norms.size(),
                "cudaMalloc query norm workspace"
            );
            check_cuda(
                cudaMemcpyAsync(
                    device_query_norms_, query_norms.data(),
                    query_norms.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream_
                ),
                "copy cached query norms"
            );
            saturating_add_bytes(
                operation_stats_.host_to_device_bytes,
                query_norms.size() * sizeof(float)
            );
            saturating_add_product(
                operation_stats_.host_precomputed_norm_fma_operations,
                query_count, dimension, 1U
            );
            cosine_precomputed_norms_kernel<<<blocks, threads, 0U, stream_>>>(
                device_queries_, query_count, device_query_norms_,
                device_candidate_cache_ + candidate_offset * dimension,
                device_candidate_norm_cache_ + candidate_offset,
                candidate_count, dimension, device_output_
            );
            saturating_increment(operation_stats_.precomputed_norm_cosine_calls);
            saturating_add_product(
                operation_stats_.avoided_pairwise_norm_fma_operations,
                query_count, candidate_count, dimension
            );
            saturating_add_product(
                operation_stats_.avoided_pairwise_norm_fma_operations,
                query_count, candidate_count, dimension
            );
        } else {
            cosine_kernel<<<blocks, threads, 0U, stream_>>>(
                device_queries_, query_count,
                device_candidate_cache_ + candidate_offset * dimension,
                candidate_count, dimension, device_output_
            );
            saturating_increment(operation_stats_.inline_norm_cosine_calls);
        }
        saturating_increment(operation_stats_.kernel_launches);
        check_cuda(cudaGetLastError(), "cached cosine kernel launch");
        check_cuda(
            cudaMemcpyAsync(
                result.data(), device_output_, result.size() * sizeof(float),
                cudaMemcpyDeviceToHost, stream_
            ),
            "copy cached cosine output"
        );
        saturating_add_bytes(
            operation_stats_.device_to_host_bytes,
            result.size() * sizeof(float)
        );
        check_cuda(
            cudaStreamSynchronize(stream_),
            "cached cosine stream synchronize"
        );
        saturating_increment(operation_stats_.stream_synchronizations);
        return result;
    }

    void local_average_update(
        const std::span<float> prototype,
        const std::span<const float> observation,
        const float learning_rate
    ) const override {
        if (prototype.empty() || prototype.size() != observation.size() ||
            !std::isfinite(static_cast<double>(learning_rate)) ||
            learning_rate < 0.0F || learning_rate > 1.0F) {
            throw std::invalid_argument("invalid CUDA local update");
        }

        std::lock_guard<std::mutex> lock(mutex_);
        saturating_increment(operation_stats_.local_update_calls);
        if (hybrid_local_updates_ &&
            prototype.size() <= host_local_update_maximum_dimensions) {
            saturating_increment(operation_stats_.host_local_update_calls);
            saturating_add_bytes(
                operation_stats_.avoided_host_to_device_bytes,
                prototype.size_bytes()
            );
            saturating_add_bytes(
                operation_stats_.avoided_host_to_device_bytes,
                observation.size_bytes()
            );
            saturating_add_bytes(
                operation_stats_.avoided_device_to_host_bytes,
                prototype.size_bytes()
            );
            saturating_increment(operation_stats_.avoided_kernel_launches);
            saturating_increment(operation_stats_.avoided_stream_synchronizations);
            for (std::size_t index = 0U; index < prototype.size(); ++index) {
                prototype[index] = std::fma(
                    learning_rate,
                    observation[index] - prototype[index],
                    prototype[index]
                );
            }
            return;
        }
        saturating_increment(operation_stats_.device_local_update_calls);
        ensure_capacity(
            device_queries_, query_capacity_, prototype.size(),
            "cudaMalloc prototype workspace"
        );
        saturating_add_bytes(operation_stats_.host_to_device_bytes, prototype.size_bytes());
        ensure_capacity(
            device_candidates_, candidate_capacity_, observation.size(),
            "cudaMalloc observation workspace"
        );
        saturating_add_bytes(operation_stats_.host_to_device_bytes, observation.size_bytes());
        check_cuda(
            cudaMemcpyAsync(
                device_queries_, prototype.data(), prototype.size_bytes(),
                cudaMemcpyHostToDevice, stream_
            ),
            "copy prototype"
        );
        check_cuda(
            cudaMemcpyAsync(
                device_candidates_, observation.data(), observation.size_bytes(),
                cudaMemcpyHostToDevice, stream_
            ),
            "copy observation"
        );
        constexpr unsigned int threads = 256U;
        const auto blocks = static_cast<unsigned int>(
            (prototype.size() + threads - 1U) / threads
        );
        local_update_kernel<<<blocks, threads, 0U, stream_>>>(
            device_queries_, device_candidates_, prototype.size(), learning_rate
        );
        saturating_increment(operation_stats_.kernel_launches);
        check_cuda(cudaGetLastError(), "local update kernel launch");
        check_cuda(
            cudaMemcpyAsync(
                prototype.data(), device_queries_, prototype.size_bytes(),
                cudaMemcpyDeviceToHost, stream_
            ),
            "copy updated prototype"
        );
        saturating_add_bytes(operation_stats_.device_to_host_bytes, prototype.size_bytes());
        check_cuda(cudaStreamSynchronize(stream_), "local update stream synchronize");
        saturating_increment(operation_stats_.stream_synchronizations);
    }

private:
    void ensure_index_capacity(const std::size_t required) const {
        if (required <= candidate_query_index_capacity_) return;
        if (device_candidate_query_indices_ != nullptr) {
            check_cuda(cudaFree(device_candidate_query_indices_),
                       "cudaFree indexed query workspace");
            device_candidate_query_indices_ = nullptr;
        }
        candidate_query_index_capacity_ = 0U;
        const std::size_t next_capacity = growth_capacity(required);
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_candidate_query_indices_),
                next_capacity * sizeof(std::size_t)
            ),
            "cudaMalloc indexed query workspace"
        );
        candidate_query_index_capacity_ = next_capacity;
    }

    static void release_noexcept(float*& pointer) noexcept {
        if (pointer != nullptr) {
            static_cast<void>(cudaFree(pointer));
            pointer = nullptr;
        }
    }

    static void ensure_capacity(
        float*& pointer,
        std::size_t& capacity,
        const std::size_t required,
        const char* operation
    ) {
        if (required <= capacity) {
            return;
        }
        release_noexcept(pointer);
        capacity = 0U;
        const std::size_t next_capacity = growth_capacity(required);
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&pointer), next_capacity * sizeof(float)),
            operation
        );
        capacity = next_capacity;
    }

    cudaDeviceProp properties_{};
    mutable std::mutex mutex_;
    mutable cudaStream_t stream_{};
    mutable float* device_queries_{};
    mutable float* device_candidates_{};
    mutable float* device_candidate_cache_{};
    mutable float* device_candidate_norm_cache_{};
    mutable float* device_query_norms_{};
    mutable float* device_output_{};
    mutable std::size_t* device_candidate_query_indices_{};
    mutable std::size_t query_capacity_{};
    mutable std::size_t candidate_capacity_{};
    mutable std::size_t candidate_cache_capacity_{};
    mutable std::size_t candidate_norm_cache_capacity_{};
    mutable std::size_t query_norm_capacity_{};
    mutable std::size_t output_capacity_{};
    mutable std::size_t candidate_query_index_capacity_{};
    mutable std::size_t candidate_cache_count_{};
    mutable std::size_t candidate_cache_dimension_{};
    mutable std::uint64_t candidate_cache_key_{};
    const bool hybrid_local_updates_{};
    const bool precomputed_cached_cosine_norms_{};
    const bool hybrid_small_batch_cosine_{};
    mutable BackendOperationStats operation_stats_{};
};

}  // namespace

std::unique_ptr<FrontierComputeBackend> make_cuda_frontier_backend() {
    return std::make_unique<CudaFrontierBackend>();
}

}  // namespace rlf::frontier
