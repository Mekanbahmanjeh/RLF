#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace rlf::experiments {

struct OptimizationBenchmarkConfig final {
    std::uint64_t seed{0x524C4642ULL};
    std::size_t dimension{1'024U};
    std::size_t mode_count{4'096U};
    std::size_t query_count{32U};
    std::size_t candidate_count{256U};
    std::size_t thread_count{4U};
    std::size_t similarity_iterations{2'048U};
    std::size_t quantization_samples{256U};
};

struct RetrievalBenchmarkResult final {
    std::string implementation;
    std::size_t thread_count{};
    double accuracy{};
    double seconds{};
    double queries_per_second{};
    std::size_t bytes_stored{};
    std::uint64_t result_hash{};
};

struct QuantizationBenchmarkResult final {
    std::string encoding;
    double mean_angular_error{};
    double maximum_angular_error{};
    double mean_similarity{};
    double transformation_accuracy{};
    double encode_seconds{};
    double decode_seconds{};
    std::size_t bytes_stored{};
    double compression_ratio{};
    bool numerically_stable{};
};

struct OptimizationBenchmarkResult final {
    std::uint64_t seed{};
    std::size_t dimension{};
    std::size_t mode_count{};
    std::size_t query_count{};
    std::size_t candidate_count{};
    std::size_t thread_count{};
    double scalar_similarity_seconds{};
    double optimized_similarity_seconds{};
    double similarity_speedup{};
    double maximum_similarity_difference{};
    std::vector<RetrievalBenchmarkResult> retrieval;
    std::vector<QuantizationBenchmarkResult> quantization;
    std::string dominant_bottleneck;
    bool optimized_matches_reference{};
    bool parallel_matches_reference{};
    bool cuda_scientifically_justified{};
    std::uint64_t deterministic_run_hash{};
};

[[nodiscard]] OptimizationBenchmarkResult run_optimization_benchmark(
    const OptimizationBenchmarkConfig& config
);
void write_optimization_benchmark_json(
    std::ostream& output,
    const OptimizationBenchmarkResult& result
);

}  // namespace rlf::experiments
