#include "rlf/experiments/associative_recall.hpp"

#include "rlf/baselines/nearest_neighbor.hpp"
#include "rlf/core/deterministic_rng.hpp"
#include "rlf/core/phase_vector.hpp"
#include "rlf/experiments/phase_vector_smoke.hpp"
#include "rlf/memory/associative_memory.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rlf::experiments {
namespace {

constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;

struct Association final {
    core::PhaseVector key;
    core::PhaseVector value;
};

struct RecallMeasurement final {
    double exact;
    double noisy;
    std::size_t queries;
};

void hash_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (unsigned int byte_index = 0U; byte_index < 8U; ++byte_index) {
        hash ^= (value >> (byte_index * 8U)) & 0xFFULL;
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

[[nodiscard]] core::PhaseVector perturb(
    const core::PhaseVector& key,
    const double noise_radians,
    core::DeterministicRng& random_number_generator
) {
    std::vector<float> noise;
    noise.reserve(key.size());
    for (std::size_t index = 0U; index < key.size(); ++index) {
        const double signed_unit =
            (2.0 * random_number_generator.uniform_unit()) - 1.0;
        noise.push_back(
            static_cast<float>(signed_unit * noise_radians)
        );
    }
    return key.composed(core::PhaseVector(std::move(noise)));
}

[[nodiscard]] RecallMeasurement measure_rlf(
    memory::AssociativeMemory& memory,
    const std::vector<Association>& associations,
    const std::size_t count,
    const double noise_radians,
    const std::uint64_t noise_seed
) {
    core::DeterministicRng noise_rng(noise_seed);
    std::size_t exact_successes = 0U;
    std::size_t noisy_successes = 0U;
    for (std::size_t index = 0U; index < count; ++index) {
        const std::uint64_t expected_id =
            static_cast<std::uint64_t>(index + 1U);
        const auto exact = memory.retrieve(
            associations[index].key,
            1U
        );
        if (!exact.empty() &&
            exact[0U].record_id == expected_id) {
            ++exact_successes;
        }
        const core::PhaseVector noisy_key = perturb(
            associations[index].key,
            noise_radians,
            noise_rng
        );
        const auto noisy = memory.retrieve(noisy_key, 1U);
        if (!noisy.empty() &&
            noisy[0U].record_id == expected_id) {
            ++noisy_successes;
        }
    }
    return {
        .exact =
            static_cast<double>(exact_successes) /
            static_cast<double>(count),
        .noisy =
            static_cast<double>(noisy_successes) /
            static_cast<double>(count),
        .queries = count * 2U,
    };
}

[[nodiscard]] RecallMeasurement measure_baseline(
    const baselines::NearestNeighborMemory& memory,
    const std::vector<Association>& associations,
    const std::size_t count,
    const double noise_radians,
    const std::uint64_t noise_seed
) {
    core::DeterministicRng noise_rng(noise_seed);
    std::size_t exact_successes = 0U;
    std::size_t noisy_successes = 0U;
    for (std::size_t index = 0U; index < count; ++index) {
        const std::uint64_t expected_id =
            static_cast<std::uint64_t>(index + 1U);
        const auto exact = memory.retrieve(
            associations[index].key,
            1U
        );
        if (!exact.empty() &&
            exact[0U].record_id == expected_id) {
            ++exact_successes;
        }
        const core::PhaseVector noisy_key = perturb(
            associations[index].key,
            noise_radians,
            noise_rng
        );
        const auto noisy = memory.retrieve(noisy_key, 1U);
        if (!noisy.empty() &&
            noisy[0U].record_id == expected_id) {
            ++noisy_successes;
        }
    }
    return {
        .exact =
            static_cast<double>(exact_successes) /
            static_cast<double>(count),
        .noisy =
            static_cast<double>(noisy_successes) /
            static_cast<double>(count),
        .queries = count * 2U,
    };
}

[[nodiscard]] std::vector<std::size_t> growth_counts(
    const std::size_t total
) {
    std::vector<std::size_t> counts{
        std::max(std::size_t{1U}, total / 4U),
        std::max(std::size_t{1U}, total / 2U),
        std::max(std::size_t{1U}, (total * 3U) / 4U),
        total,
    };
    std::sort(counts.begin(), counts.end());
    counts.erase(std::unique(counts.begin(), counts.end()), counts.end());
    return counts;
}

[[nodiscard]] ExperimentMetrics make_metrics(
    const double noisy_recall,
    const double initial_one_shot_recall,
    const double retained_accuracy,
    const std::size_t bytes_stored,
    const std::size_t peak_resident_bytes,
    const double training_seconds,
    const double inference_seconds,
    const std::size_t training_examples,
    const std::size_t inference_queries,
    const double operations_per_inference,
    const std::uint64_t deterministic_hash
) {
    const double training_count =
        static_cast<double>(training_examples);
    const double inference_count =
        static_cast<double>(inference_queries);
    const double useful_retained =
        retained_accuracy * training_count;
    return {
        .task_accuracy = noisy_recall,
        .one_shot_recall = initial_one_shot_recall,
        .retained_accuracy = retained_accuracy,
        .catastrophic_forgetting_score =
            std::max(0.0, initial_one_shot_recall - retained_accuracy),
        .compositional_generalization_score = 0.0,
        .prediction_error = 1.0 - noisy_recall,
        .average_settling_cycles = 0.0,
        .average_modes_retrieved = 0.0,
        .average_modes_activated = 0.0,
        .modes_created = 0ULL,
        .modes_split = 0ULL,
        .modes_merged = 0ULL,
        .modes_pruned = 0ULL,
        .bytes_stored = bytes_stored,
        .peak_resident_bytes = peak_resident_bytes,
        .training_seconds = training_seconds,
        .inference_seconds = inference_seconds,
        .training_examples_per_second =
            training_seconds > 0.0
                ? training_count / training_seconds
                : 0.0,
        .inference_examples_per_second =
            inference_seconds > 0.0
                ? inference_count / inference_seconds
                : 0.0,
        .update_operations_per_example = 1.0,
        .active_operations_per_inference =
            operations_per_inference,
        .efficiency_score = provisional_efficiency_score(
            useful_retained,
            bytes_stored,
            operations_per_inference,
            training_seconds
        ),
        .deterministic_run_hash = deterministic_hash,
    };
}

}  // namespace

AssociativeRecallResult run_associative_recall(
    const AssociativeRecallConfig& config
) {
    if (config.dimension == 0U ||
        config.association_count < 4U ||
        !std::isfinite(config.noise_radians) ||
        config.noise_radians < 0.0) {
        throw std::invalid_argument(
            "invalid associative-recall configuration"
        );
    }

    core::DeterministicRng data_rng(config.seed);
    std::vector<Association> associations;
    associations.reserve(config.association_count);
    for (std::size_t index = 0U;
         index < config.association_count;
         ++index) {
        associations.push_back({
            .key = core::PhaseVector::random(
                config.dimension,
                data_rng
            ),
            .value = core::PhaseVector::random(
                config.dimension,
                data_rng
            ),
        });
    }

    memory::AssociativeMemory rlf_memory(
        config.dimension,
        config.association_count
    );
    baselines::NearestNeighborMemory baseline(config.dimension);
    const std::size_t initial_count = config.association_count / 2U;

    const auto rlf_training_start =
        std::chrono::steady_clock::now();
    for (std::size_t index = 0U;
         index < initial_count;
         ++index) {
        static_cast<void>(rlf_memory.insert(
            associations[index].key,
            associations[index].value
        ));
    }
    const RecallMeasurement rlf_initial = measure_rlf(
        rlf_memory,
        associations,
        initial_count,
        config.noise_radians,
        config.seed ^ 0x1001ULL
    );
    for (std::size_t index = initial_count;
         index < config.association_count;
         ++index) {
        static_cast<void>(rlf_memory.insert(
            associations[index].key,
            associations[index].value
        ));
    }
    const auto rlf_training_end =
        std::chrono::steady_clock::now();

    const auto baseline_training_start =
        std::chrono::steady_clock::now();
    for (std::size_t index = 0U;
         index < initial_count;
         ++index) {
        static_cast<void>(baseline.insert(
            associations[index].key,
            associations[index].value
        ));
    }
    const RecallMeasurement baseline_initial = measure_baseline(
        baseline,
        associations,
        initial_count,
        config.noise_radians,
        config.seed ^ 0x1001ULL
    );
    for (std::size_t index = initial_count;
         index < config.association_count;
         ++index) {
        static_cast<void>(baseline.insert(
            associations[index].key,
            associations[index].value
        ));
    }
    const auto baseline_training_end =
        std::chrono::steady_clock::now();

    const auto rlf_inference_start =
        std::chrono::steady_clock::now();
    const RecallMeasurement rlf_final = measure_rlf(
        rlf_memory,
        associations,
        config.association_count,
        config.noise_radians,
        config.seed ^ 0x2002ULL
    );
    const RecallMeasurement rlf_retained = measure_rlf(
        rlf_memory,
        associations,
        initial_count,
        config.noise_radians,
        config.seed ^ 0x3003ULL
    );
    const auto rlf_inference_end =
        std::chrono::steady_clock::now();

    const auto baseline_inference_start =
        std::chrono::steady_clock::now();
    const RecallMeasurement baseline_final = measure_baseline(
        baseline,
        associations,
        config.association_count,
        config.noise_radians,
        config.seed ^ 0x2002ULL
    );
    const RecallMeasurement baseline_retained = measure_baseline(
        baseline,
        associations,
        initial_count,
        config.noise_radians,
        config.seed ^ 0x3003ULL
    );
    const auto baseline_inference_end =
        std::chrono::steady_clock::now();

    std::vector<RecallGrowthPoint> growth;
    for (const std::size_t count :
         growth_counts(config.association_count)) {
        memory::AssociativeMemory growth_rlf(
            config.dimension,
            count
        );
        baselines::NearestNeighborMemory growth_baseline(
            config.dimension
        );
        for (std::size_t index = 0U; index < count; ++index) {
            static_cast<void>(growth_rlf.insert(
                associations[index].key,
                associations[index].value
            ));
            static_cast<void>(growth_baseline.insert(
                associations[index].key,
                associations[index].value
            ));
        }
        const std::uint64_t noise_seed =
            config.seed ^ static_cast<std::uint64_t>(count);
        const RecallMeasurement rlf_point = measure_rlf(
            growth_rlf,
            associations,
            count,
            config.noise_radians,
            noise_seed
        );
        const RecallMeasurement baseline_point = measure_baseline(
            growth_baseline,
            associations,
            count,
            config.noise_radians,
            noise_seed
        );
        growth.push_back({
            .records = count,
            .rlf_exact_recall = rlf_point.exact,
            .rlf_noisy_recall = rlf_point.noisy,
            .baseline_exact_recall = baseline_point.exact,
            .baseline_noisy_recall = baseline_point.noisy,
        });
    }

    const double rlf_training_seconds =
        std::chrono::duration<double>(
            rlf_training_end - rlf_training_start
        ).count();
    const double baseline_training_seconds =
        std::chrono::duration<double>(
            baseline_training_end - baseline_training_start
        ).count();
    const double rlf_inference_seconds =
        std::chrono::duration<double>(
            rlf_inference_end - rlf_inference_start
        ).count();
    const double baseline_inference_seconds =
        std::chrono::duration<double>(
            baseline_inference_end - baseline_inference_start
        ).count();

    std::uint64_t rlf_hash = fnv_offset_basis;
    std::uint64_t baseline_hash = fnv_offset_basis;
    hash_u64(
        rlf_hash,
        std::bit_cast<std::uint64_t>(rlf_final.noisy)
    );
    hash_u64(
        rlf_hash,
        std::bit_cast<std::uint64_t>(rlf_retained.noisy)
    );
    hash_u64(
        baseline_hash,
        std::bit_cast<std::uint64_t>(baseline_final.noisy)
    );
    hash_u64(
        baseline_hash,
        std::bit_cast<std::uint64_t>(baseline_retained.noisy)
    );
    hash_u64(rlf_hash, static_cast<std::uint64_t>(rlf_memory.size()));
    hash_u64(
        baseline_hash,
        static_cast<std::uint64_t>(baseline.size())
    );

    const std::size_t peak_resident = peak_resident_memory_bytes();
    RecallSystemResult rlf_result{
        .system = "rlf_associative_memory",
        .exact_recall = rlf_final.exact,
        .noisy_recall = rlf_final.noisy,
        .initial_one_shot_recall = rlf_initial.exact,
        .retained_accuracy = rlf_retained.noisy,
        .metrics = make_metrics(
            rlf_final.noisy,
            rlf_initial.exact,
            rlf_retained.noisy,
            rlf_memory.bytes_stored(),
            peak_resident,
            rlf_training_seconds,
            rlf_inference_seconds,
            config.association_count,
            rlf_final.queries + rlf_retained.queries,
            static_cast<double>(config.association_count),
            rlf_hash
        ),
    };
    RecallSystemResult baseline_result{
        .system = "nearest_neighbor_baseline",
        .exact_recall = baseline_final.exact,
        .noisy_recall = baseline_final.noisy,
        .initial_one_shot_recall = baseline_initial.exact,
        .retained_accuracy = baseline_retained.noisy,
        .metrics = make_metrics(
            baseline_final.noisy,
            baseline_initial.exact,
            baseline_retained.noisy,
            baseline.bytes_stored(),
            peak_resident,
            baseline_training_seconds,
            baseline_inference_seconds,
            config.association_count,
            baseline_final.queries + baseline_retained.queries,
            static_cast<double>(config.association_count),
            baseline_hash
        ),
    };

    std::uint64_t run_hash = fnv_offset_basis;
    hash_string(run_hash, rlf_result.system);
    hash_u64(run_hash, rlf_hash);
    hash_string(run_hash, baseline_result.system);
    hash_u64(run_hash, baseline_hash);
    for (const RecallGrowthPoint& point : growth) {
        hash_u64(run_hash, static_cast<std::uint64_t>(point.records));
        hash_u64(
            run_hash,
            std::bit_cast<std::uint64_t>(point.rlf_noisy_recall)
        );
        hash_u64(
            run_hash,
            std::bit_cast<std::uint64_t>(
                point.baseline_noisy_recall
            )
        );
    }

    return {
        .seed = config.seed,
        .dimension = config.dimension,
        .association_count = config.association_count,
        .noise_radians = config.noise_radians,
        .rlf = std::move(rlf_result),
        .baseline = std::move(baseline_result),
        .growth = std::move(growth),
        .deterministic_run_hash = run_hash,
    };
}

void write_associative_recall_json(
    std::ostream& output,
    const AssociativeRecallResult& result
) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "{\n"
           << "  \"experiment\": \"associative_recall\",\n"
           << "  \"status\": \"observed\",\n"
           << "  \"seed\": " << result.seed << ",\n"
           << "  \"dimension\": " << result.dimension << ",\n"
           << "  \"association_count\": "
           << result.association_count << ",\n"
           << "  \"noise_radians\": "
           << result.noise_radians << ",\n"
           << "  \"rlf\": {\n"
           << "    \"system\": \"" << result.rlf.system << "\",\n"
           << "    \"exact_recall\": "
           << result.rlf.exact_recall << ",\n"
           << "    \"noisy_recall\": "
           << result.rlf.noisy_recall << ",\n"
           << "    \"initial_one_shot_recall\": "
           << result.rlf.initial_one_shot_recall << ",\n"
           << "    \"retained_accuracy\": "
           << result.rlf.retained_accuracy << ",\n"
           << "    \"metrics\": ";
    write_metrics_json(output, result.rlf.metrics, 4U);
    output << "\n  },\n"
           << "  \"baseline\": {\n"
           << "    \"system\": \"" << result.baseline.system << "\",\n"
           << "    \"exact_recall\": "
           << result.baseline.exact_recall << ",\n"
           << "    \"noisy_recall\": "
           << result.baseline.noisy_recall << ",\n"
           << "    \"initial_one_shot_recall\": "
           << result.baseline.initial_one_shot_recall << ",\n"
           << "    \"retained_accuracy\": "
           << result.baseline.retained_accuracy << ",\n"
           << "    \"metrics\": ";
    write_metrics_json(output, result.baseline.metrics, 4U);
    output << "\n  },\n"
           << "  \"growth\": [\n";
    for (std::size_t index = 0U;
         index < result.growth.size();
         ++index) {
        const RecallGrowthPoint& point = result.growth[index];
        output << "    {\n"
               << "      \"records\": " << point.records << ",\n"
               << "      \"rlf_exact_recall\": "
               << point.rlf_exact_recall << ",\n"
               << "      \"rlf_noisy_recall\": "
               << point.rlf_noisy_recall << ",\n"
               << "      \"baseline_exact_recall\": "
               << point.baseline_exact_recall << ",\n"
               << "      \"baseline_noisy_recall\": "
               << point.baseline_noisy_recall << "\n"
               << "    }";
        if (index + 1U != result.growth.size()) {
            output << ',';
        }
        output << '\n';
    }
    output << "  ],\n"
           << "  \"deterministic_run_hash\": \""
           << format_run_hash(result.deterministic_run_hash)
           << "\"\n"
           << "}\n";
    if (!output) {
        throw std::runtime_error(
            "failed to write associative-recall result"
        );
    }
}

}  // namespace rlf::experiments
