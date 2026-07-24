#include "rlf/experiments/optimization_benchmark.hpp"

#include "rlf/backend/compute_backend.hpp"
#include "rlf/core/deterministic_rng.hpp"
#include "rlf/core/phase_vector.hpp"
#include "rlf/core/quantized_phase_vector.hpp"
#include "rlf/core/resonant_mode.hpp"
#include "rlf/experiments/phase_vector_smoke.hpp"
#include "rlf/retrieval/contiguous_mode_index.hpp"
#include "rlf/retrieval/mode_retriever.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rlf::experiments {
namespace {

constexpr double success_similarity = 0.95;
constexpr std::uint64_t fnv_offset_basis =
    14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;

void hash_u64(
    std::uint64_t& hash,
    const std::uint64_t value
) noexcept {
    for (unsigned int byte = 0U; byte < 8U; ++byte) {
        hash ^= (value >> (byte * 8U)) & 0xFFULL;
        hash *= fnv_prime;
    }
}

void hash_string(
    std::uint64_t& hash,
    const std::string& value
) noexcept {
    for (const char character : value) {
        hash ^= static_cast<std::uint8_t>(character);
        hash *= fnv_prime;
    }
}

[[nodiscard]] std::vector<core::ResonantMode> make_modes(
    const OptimizationBenchmarkConfig& config,
    core::DeterministicRng& rng
) {
    std::vector<core::ResonantMode> modes;
    modes.reserve(config.mode_count);
    for (std::size_t index = 0U;
         index < config.mode_count;
         ++index) {
        modes.emplace_back(
            static_cast<std::uint64_t>(index + 1U),
            core::PhaseVector::random(config.dimension, rng),
            core::PhaseVector::random(config.dimension, rng),
            1.0F,
            1.0F
        );
    }
    return modes;
}

[[nodiscard]] std::vector<core::PhaseVector> make_queries(
    const OptimizationBenchmarkConfig& config,
    const std::vector<core::ResonantMode>& modes
) {
    std::vector<core::PhaseVector> queries;
    queries.reserve(config.query_count);
    for (std::size_t index = 0U;
         index < config.query_count;
         ++index) {
        const std::size_t mode_index =
            (index * modes.size()) / config.query_count;
        queries.push_back(modes[mode_index].context_key);
    }
    return queries;
}

template <typename Retrieve>
[[nodiscard]] RetrievalBenchmarkResult benchmark_retrieval(
    const std::string& name,
    const std::size_t thread_count,
    const std::span<const core::PhaseVector> queries,
    const std::vector<core::ResonantMode>& modes,
    const std::size_t bytes_stored,
    Retrieve retrieve
) {
    std::size_t successes = 0U;
    std::uint64_t result_hash = fnv_offset_basis;
    const auto start = std::chrono::steady_clock::now();
    for (const core::PhaseVector& query : queries) {
        const std::vector<retrieval::RetrievedMode> result =
            retrieve(query);
        if (!result.empty() &&
            modes[result.front().mode_index]
                .context_key.similarity(query) >=
                success_similarity) {
            ++successes;
        }
        for (const retrieval::RetrievedMode& item : result) {
            hash_u64(result_hash, item.mode_id);
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const double seconds =
        std::chrono::duration<double>(end - start).count();
    return {
        .implementation = name,
        .thread_count = thread_count,
        .accuracy =
            static_cast<double>(successes) /
            static_cast<double>(queries.size()),
        .seconds = seconds,
        .queries_per_second =
            static_cast<double>(queries.size()) / seconds,
        .bytes_stored = bytes_stored,
        .result_hash = result_hash,
    };
}

[[nodiscard]] QuantizationBenchmarkResult benchmark_quantization(
    const core::PhaseEncoding encoding,
    const OptimizationBenchmarkConfig& config,
    core::DeterministicRng& rng
) {
    std::vector<core::PhaseVector> values;
    values.reserve(config.quantization_samples);
    for (std::size_t index = 0U;
         index < config.quantization_samples;
         ++index) {
        values.push_back(
            core::PhaseVector::random(config.dimension, rng)
        );
    }

    std::vector<core::QuantizedPhaseVector> encoded;
    encoded.reserve(values.size());
    const auto encode_start = std::chrono::steady_clock::now();
    for (const core::PhaseVector& value : values) {
        encoded.push_back(
            core::QuantizedPhaseVector::encode(value, encoding)
        );
    }
    const auto encode_end = std::chrono::steady_clock::now();

    double error_total = 0.0;
    double maximum_error = 0.0;
    double similarity_total = 0.0;
    std::size_t successes = 0U;
    std::size_t bytes = 0U;
    const core::PhaseVector input =
        core::PhaseVector::random(config.dimension, rng);
    const auto decode_start = std::chrono::steady_clock::now();
    for (std::size_t index = 0U;
         index < encoded.size();
         ++index) {
        const core::PhaseVector decoded = encoded[index].decode();
        const double error =
            decoded.mean_angular_error(values[index]);
        error_total += error;
        maximum_error = std::max(maximum_error, error);
        similarity_total += decoded.similarity(values[index]);
        const core::PhaseVector reference_target =
            input.composed(values[index]);
        const core::PhaseVector quantized_target =
            input.composed(decoded);
        if (quantized_target.similarity(reference_target) >=
            success_similarity) {
            ++successes;
        }
        bytes += encoded[index].bytes_stored();
    }
    const auto decode_end = std::chrono::steady_clock::now();

    const double count = static_cast<double>(values.size());
    const std::size_t float_bytes =
        values.size() * config.dimension * sizeof(float);
    return {
        .encoding = std::string(core::to_string(encoding)),
        .mean_angular_error = error_total / count,
        .maximum_angular_error = maximum_error,
        .mean_similarity = similarity_total / count,
        .transformation_accuracy =
            static_cast<double>(successes) / count,
        .encode_seconds = std::chrono::duration<double>(
            encode_end - encode_start
        ).count(),
        .decode_seconds = std::chrono::duration<double>(
            decode_end - decode_start
        ).count(),
        .bytes_stored = bytes,
        .compression_ratio =
            static_cast<double>(float_bytes) /
            static_cast<double>(bytes),
        .numerically_stable =
            std::isfinite(error_total) &&
            std::isfinite(similarity_total) &&
            maximum_error < 0.02,
    };
}

}  // namespace

OptimizationBenchmarkResult run_optimization_benchmark(
    const OptimizationBenchmarkConfig& config
) {
    if (config.dimension == 0U ||
        config.mode_count == 0U ||
        config.query_count == 0U ||
        config.candidate_count == 0U ||
        config.thread_count == 0U ||
        config.similarity_iterations == 0U ||
        config.quantization_samples == 0U) {
        throw std::invalid_argument(
            "invalid optimization benchmark configuration"
        );
    }
    core::DeterministicRng rng(config.seed);
    const std::vector<core::ResonantMode> modes =
        make_modes(config, rng);
    const std::vector<core::PhaseVector> queries =
        make_queries(config, modes);

    const auto scalar_backend = backend::make_backend(
        backend::BackendKind::scalar_cpu
    );
    const auto optimized_backend = backend::make_backend(
        backend::BackendKind::optimized_cpu
    );
    double maximum_similarity_difference = 0.0;
    volatile double similarity_sink = 0.0;
    const auto scalar_start = std::chrono::steady_clock::now();
    for (std::size_t iteration = 0U;
         iteration < config.similarity_iterations;
         ++iteration) {
        similarity_sink = similarity_sink +
            scalar_backend->similarity(
                modes[iteration % modes.size()].context_key,
                queries[iteration % queries.size()]
            );
    }
    const auto scalar_end = std::chrono::steady_clock::now();
    const auto optimized_start = std::chrono::steady_clock::now();
    for (std::size_t iteration = 0U;
         iteration < config.similarity_iterations;
         ++iteration) {
        similarity_sink = similarity_sink +
            optimized_backend->similarity(
                modes[iteration % modes.size()].context_key,
                queries[iteration % queries.size()]
            );
    }
    const auto optimized_end = std::chrono::steady_clock::now();
    static_cast<void>(similarity_sink);
    for (std::size_t index = 0U;
         index < std::min(modes.size(), queries.size());
         ++index) {
        maximum_similarity_difference = std::max(
            maximum_similarity_difference,
            std::abs(
                scalar_backend->similarity(
                    modes[index].context_key,
                    queries[index]
                ) -
                optimized_backend->similarity(
                    modes[index].context_key,
                    queries[index]
                )
            )
        );
    }
    const double scalar_similarity_seconds =
        std::chrono::duration<double>(
            scalar_end - scalar_start
        ).count();
    const double optimized_similarity_seconds =
        std::chrono::duration<double>(
            optimized_end - optimized_start
        ).count();

    const retrieval::ExactModeRetriever scalar_retriever(
        scalar_backend
    );
    const retrieval::ExactModeRetriever optimized_retriever(
        optimized_backend
    );
    const retrieval::ParallelExactModeRetriever parallel_retriever(
        config.thread_count,
        optimized_backend
    );
    const retrieval::ContiguousModeIndex contiguous_index(
        modes,
        optimized_backend
    );

    std::size_t object_bytes = sizeof(modes);
    for (const core::ResonantMode& mode : modes) {
        object_bytes += sizeof(mode) +
            ((mode.context_key.size() + mode.transformation.size()) *
             sizeof(float));
    }
    std::vector<RetrievalBenchmarkResult> retrieval_results;
    retrieval_results.push_back(benchmark_retrieval(
        "scalar_object_exact",
        1U,
        queries,
        modes,
        object_bytes,
        [&](const core::PhaseVector& query) {
            return scalar_retriever.retrieve(
                query,
                modes,
                config.candidate_count
            );
        }
    ));
    retrieval_results.push_back(benchmark_retrieval(
        "optimized_object_exact",
        1U,
        queries,
        modes,
        object_bytes,
        [&](const core::PhaseVector& query) {
            return optimized_retriever.retrieve(
                query,
                modes,
                config.candidate_count
            );
        }
    ));
    retrieval_results.push_back(benchmark_retrieval(
        "optimized_object_parallel",
        config.thread_count,
        queries,
        modes,
        object_bytes,
        [&](const core::PhaseVector& query) {
            return parallel_retriever.retrieve(
                query,
                modes,
                config.candidate_count
            );
        }
    ));
    retrieval_results.push_back(benchmark_retrieval(
        "optimized_contiguous",
        1U,
        queries,
        modes,
        contiguous_index.bytes_stored(),
        [&](const core::PhaseVector& query) {
            return contiguous_index.retrieve(
                query,
                config.candidate_count,
                1U
            );
        }
    ));
    retrieval_results.push_back(benchmark_retrieval(
        "optimized_contiguous_parallel",
        config.thread_count,
        queries,
        modes,
        contiguous_index.bytes_stored(),
        [&](const core::PhaseVector& query) {
            return contiguous_index.retrieve(
                query,
                config.candidate_count,
                config.thread_count
            );
        }
    ));

    std::vector<QuantizationBenchmarkResult> quantization;
    for (const core::PhaseEncoding encoding : {
             core::PhaseEncoding::float32,
             core::PhaseEncoding::uint8_phase,
             core::PhaseEncoding::uint16_phase,
             core::PhaseEncoding::float16_compatible,
             core::PhaseEncoding::int8_residual,
         }) {
        quantization.push_back(
            benchmark_quantization(encoding, config, rng)
        );
    }

    const std::uint64_t reference_hash =
        retrieval_results.front().result_hash;
    bool optimized_matches_reference = true;
    bool parallel_matches_reference = true;
    for (const RetrievalBenchmarkResult& result :
         retrieval_results) {
        if (result.implementation.find("parallel") !=
            std::string::npos) {
            parallel_matches_reference =
                parallel_matches_reference &&
                result.result_hash == reference_hash;
        } else {
            optimized_matches_reference =
                optimized_matches_reference &&
                result.result_hash == reference_hash;
        }
    }
    std::uint64_t run_hash = fnv_offset_basis;
    hash_u64(run_hash, config.seed);
    for (const RetrievalBenchmarkResult& result :
         retrieval_results) {
        hash_string(run_hash, result.implementation);
        hash_u64(run_hash, result.result_hash);
    }
    for (const QuantizationBenchmarkResult& result :
         quantization) {
        hash_string(run_hash, result.encoding);
        hash_u64(
            run_hash,
            std::bit_cast<std::uint64_t>(
                result.transformation_accuracy
            )
        );
    }

    return {
        .seed = config.seed,
        .dimension = config.dimension,
        .mode_count = config.mode_count,
        .query_count = config.query_count,
        .candidate_count = config.candidate_count,
        .thread_count = config.thread_count,
        .scalar_similarity_seconds = scalar_similarity_seconds,
        .optimized_similarity_seconds =
            optimized_similarity_seconds,
        .similarity_speedup =
            scalar_similarity_seconds /
            optimized_similarity_seconds,
        .maximum_similarity_difference =
            maximum_similarity_difference,
        .retrieval = std::move(retrieval_results),
        .quantization = std::move(quantization),
        .dominant_bottleneck =
            "exact_mode_scan_and_phase_similarity",
        .optimized_matches_reference =
            optimized_matches_reference,
        .parallel_matches_reference =
            parallel_matches_reference,
        .cuda_scientifically_justified = false,
        .deterministic_run_hash = run_hash,
    };
}

void write_optimization_benchmark_json(
    std::ostream& output,
    const OptimizationBenchmarkResult& result
) {
    output << std::setprecision(
        std::numeric_limits<double>::max_digits10
    );
    output
        << "{\n"
        << "  \"experiment\": \"optimization_benchmark\",\n"
        << "  \"seed\": " << result.seed << ",\n"
        << "  \"dimension\": " << result.dimension << ",\n"
        << "  \"mode_count\": " << result.mode_count << ",\n"
        << "  \"query_count\": " << result.query_count << ",\n"
        << "  \"candidate_count\": "
        << result.candidate_count << ",\n"
        << "  \"thread_count\": " << result.thread_count << ",\n"
        << "  \"scalar_similarity_seconds\": "
        << result.scalar_similarity_seconds << ",\n"
        << "  \"optimized_similarity_seconds\": "
        << result.optimized_similarity_seconds << ",\n"
        << "  \"similarity_speedup\": "
        << result.similarity_speedup << ",\n"
        << "  \"maximum_similarity_difference\": "
        << result.maximum_similarity_difference << ",\n"
        << "  \"dominant_bottleneck\": \""
        << result.dominant_bottleneck << "\",\n"
        << "  \"optimized_matches_reference\": "
        << (result.optimized_matches_reference ? "true" : "false")
        << ",\n"
        << "  \"parallel_matches_reference\": "
        << (result.parallel_matches_reference ? "true" : "false")
        << ",\n"
        << "  \"cuda_scientifically_justified\": "
        << (result.cuda_scientifically_justified
                ? "true"
                : "false")
        << ",\n"
        << "  \"retrieval\": [\n";
    for (std::size_t index = 0U;
         index < result.retrieval.size();
         ++index) {
        const RetrievalBenchmarkResult& item =
            result.retrieval[index];
        output
            << "    {\n"
            << "      \"implementation\": \""
            << item.implementation << "\",\n"
            << "      \"thread_count\": "
            << item.thread_count << ",\n"
            << "      \"accuracy\": "
            << item.accuracy << ",\n"
            << "      \"seconds\": "
            << item.seconds << ",\n"
            << "      \"queries_per_second\": "
            << item.queries_per_second << ",\n"
            << "      \"bytes_stored\": "
            << item.bytes_stored << ",\n"
            << "      \"result_hash\": \""
            << format_run_hash(item.result_hash) << "\"\n"
            << "    }";
        if (index + 1U != result.retrieval.size()) {
            output << ',';
        }
        output << '\n';
    }
    output << "  ],\n  \"quantization\": [\n";
    for (std::size_t index = 0U;
         index < result.quantization.size();
         ++index) {
        const QuantizationBenchmarkResult& item =
            result.quantization[index];
        output
            << "    {\n"
            << "      \"encoding\": \"" << item.encoding << "\",\n"
            << "      \"mean_angular_error\": "
            << item.mean_angular_error << ",\n"
            << "      \"maximum_angular_error\": "
            << item.maximum_angular_error << ",\n"
            << "      \"mean_similarity\": "
            << item.mean_similarity << ",\n"
            << "      \"transformation_accuracy\": "
            << item.transformation_accuracy << ",\n"
            << "      \"encode_seconds\": "
            << item.encode_seconds << ",\n"
            << "      \"decode_seconds\": "
            << item.decode_seconds << ",\n"
            << "      \"bytes_stored\": "
            << item.bytes_stored << ",\n"
            << "      \"compression_ratio\": "
            << item.compression_ratio << ",\n"
            << "      \"numerically_stable\": "
            << (item.numerically_stable ? "true" : "false")
            << "\n"
            << "    }";
        if (index + 1U != result.quantization.size()) {
            output << ',';
        }
        output << '\n';
    }
    output
        << "  ],\n"
        << "  \"deterministic_run_hash\": \""
        << format_run_hash(result.deterministic_run_hash)
        << "\"\n"
        << "}\n";
    if (!output) {
        throw std::runtime_error(
            "failed to write optimization benchmark result"
        );
    }
}

}  // namespace rlf::experiments
