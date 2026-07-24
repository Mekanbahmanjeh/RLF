#include "rlf/experiments/capacity_scaling.hpp"

#include "rlf/baselines/nearest_neighbor.hpp"
#include "rlf/core/deterministic_rng.hpp"
#include "rlf/core/phase_vector.hpp"
#include "rlf/core/resonant_mode.hpp"
#include "rlf/experiments/phase_vector_smoke.hpp"
#include "rlf/learning/structural_learning.hpp"
#include "rlf/retrieval/mode_retriever.hpp"

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
#include <string_view>
#include <utility>
#include <vector>

namespace rlf::experiments {
namespace {

constexpr double success_similarity = 0.95;
constexpr std::uint64_t fnv_offset_basis =
    14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;

struct Evaluation final {
    double clean_accuracy{};
    double noisy_accuracy{};
    double inference_seconds{};
    double candidates_returned{};
};

void hash_u64(
    std::uint64_t& hash,
    const std::uint64_t value
) noexcept {
    for (unsigned int byte_index = 0U;
         byte_index < 8U;
         ++byte_index) {
        hash ^= (value >> (byte_index * 8U)) & 0xFFULL;
        hash *= fnv_prime;
    }
}

void hash_string(
    std::uint64_t& hash,
    const std::string_view value
) noexcept {
    for (const char character : value) {
        hash ^= static_cast<std::uint8_t>(character);
        hash *= fnv_prime;
    }
}

void hash_phase_vector(
    std::uint64_t& hash,
    const core::PhaseVector& value
) noexcept {
    for (const float angle : value.angles()) {
        hash_u64(
            hash,
            static_cast<std::uint64_t>(
                std::bit_cast<std::uint32_t>(angle)
            )
        );
    }
}

[[nodiscard]] core::PhaseVector perturb(
    const core::PhaseVector& value,
    const double noise_radians,
    core::DeterministicRng& random_number_generator
) {
    std::vector<float> noise;
    noise.reserve(value.size());
    for (std::size_t index = 0U; index < value.size(); ++index) {
        const double signed_unit =
            (2.0 * random_number_generator.uniform_unit()) - 1.0;
        noise.push_back(
            static_cast<float>(signed_unit * noise_radians)
        );
    }
    return value.composed(core::PhaseVector(std::move(noise)));
}

[[nodiscard]] std::size_t query_mode_index(
    const std::size_t query_index,
    const std::size_t query_count,
    const std::size_t mode_count
) noexcept {
    return std::min(
        (query_index * mode_count) / query_count,
        mode_count - 1U
    );
}

[[nodiscard]] std::vector<core::ResonantMode> make_modes(
    const CapacityScalingConfig& config,
    const std::size_t mode_count,
    double& training_seconds,
    std::uint64_t& state_hash
) {
    core::DeterministicRng random_number_generator(config.seed);
    std::vector<core::ResonantMode> modes;
    modes.reserve(mode_count);
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t mode_index = 0U;
         mode_index < mode_count;
         ++mode_index) {
        core::PhaseVector context = core::PhaseVector::random(
            config.dimension,
            random_number_generator
        );
        core::PhaseVector transformation = core::PhaseVector::random(
            config.dimension,
            random_number_generator
        );
        modes.emplace_back(
            static_cast<std::uint64_t>(mode_index + 1U),
            std::move(context),
            std::move(transformation),
            1.0F,
            1.0F,
            0.0F,
            0ULL
        );
    }
    const auto end = std::chrono::steady_clock::now();
    training_seconds =
        std::chrono::duration<double>(end - start).count();

    state_hash = fnv_offset_basis;
    for (const core::ResonantMode& mode : modes) {
        hash_u64(state_hash, mode.id);
        hash_phase_vector(state_hash, mode.context_key);
        hash_phase_vector(state_hash, mode.transformation);
    }
    return modes;
}

[[nodiscard]] Evaluation evaluate_modes(
    const CapacityScalingConfig& config,
    const std::vector<core::ResonantMode>& modes
) {
    const retrieval::ExactModeRetriever retriever;
    core::DeterministicRng noise_rng(
        config.seed ^
        static_cast<std::uint64_t>(modes.size()) ^
        0xD1B54A32D192ED03ULL
    );
    std::size_t clean_successes = 0U;
    std::size_t noisy_successes = 0U;
    std::size_t total_candidates = 0U;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t query_index = 0U;
         query_index < config.evaluation_queries;
         ++query_index) {
        const std::size_t mode_index = query_mode_index(
            query_index,
            config.evaluation_queries,
            modes.size()
        );
        const core::ResonantMode& expected = modes[mode_index];
        const auto clean = retriever.retrieve(
            expected.context_key,
            modes,
            config.candidate_count
        );
        total_candidates += clean.size();
        if (!clean.empty()) {
            const core::ResonantMode& selected =
                modes[clean.front().mode_index];
            const core::PhaseVector prediction =
                selected.propose(expected.context_key);
            const core::PhaseVector target =
                expected.propose(expected.context_key);
            if (prediction.similarity(target) >=
                success_similarity) {
                ++clean_successes;
            }
        }

        const core::PhaseVector noisy_query = perturb(
            expected.context_key,
            config.noise_radians,
            noise_rng
        );
        const auto noisy = retriever.retrieve(
            noisy_query,
            modes,
            config.candidate_count
        );
        total_candidates += noisy.size();
        if (!noisy.empty()) {
            const core::ResonantMode& selected =
                modes[noisy.front().mode_index];
            const core::PhaseVector prediction =
                selected.propose(noisy_query);
            const core::PhaseVector target =
                noisy_query.composed(expected.transformation);
            if (prediction.similarity(target) >=
                success_similarity) {
                ++noisy_successes;
            }
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const double query_count =
        static_cast<double>(config.evaluation_queries);
    return {
        .clean_accuracy =
            static_cast<double>(clean_successes) / query_count,
        .noisy_accuracy =
            static_cast<double>(noisy_successes) / query_count,
        .inference_seconds =
            std::chrono::duration<double>(end - start).count(),
        .candidates_returned =
            static_cast<double>(total_candidates) /
            (query_count * 2.0),
    };
}

[[nodiscard]] baselines::NearestNeighborMemory make_baseline(
    const CapacityScalingConfig& config,
    const std::size_t mode_count,
    double& training_seconds,
    std::uint64_t& state_hash
) {
    core::DeterministicRng random_number_generator(config.seed);
    baselines::NearestNeighborMemory baseline(config.dimension);
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t mode_index = 0U;
         mode_index < mode_count;
         ++mode_index) {
        core::PhaseVector context = core::PhaseVector::random(
            config.dimension,
            random_number_generator
        );
        core::PhaseVector transformation = core::PhaseVector::random(
            config.dimension,
            random_number_generator
        );
        static_cast<void>(baseline.insert(
            std::move(context),
            std::move(transformation)
        ));
    }
    const auto end = std::chrono::steady_clock::now();
    training_seconds =
        std::chrono::duration<double>(end - start).count();

    state_hash = fnv_offset_basis;
    for (const baselines::NearestNeighborRecord& record :
         baseline.records()) {
        hash_u64(state_hash, record.id);
        hash_phase_vector(state_hash, record.key);
        hash_phase_vector(state_hash, record.value);
    }
    return baseline;
}

[[nodiscard]] Evaluation evaluate_baseline(
    const CapacityScalingConfig& config,
    const baselines::NearestNeighborMemory& baseline
) {
    core::DeterministicRng noise_rng(
        config.seed ^
        static_cast<std::uint64_t>(baseline.size()) ^
        0xD1B54A32D192ED03ULL
    );
    std::size_t clean_successes = 0U;
    std::size_t noisy_successes = 0U;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t query_index = 0U;
         query_index < config.evaluation_queries;
         ++query_index) {
        const std::size_t record_index = query_mode_index(
            query_index,
            config.evaluation_queries,
            baseline.size()
        );
        const baselines::NearestNeighborRecord& expected =
            baseline.records()[record_index];
        const auto clean = baseline.retrieve(expected.key, 1U);
        if (!clean.empty()) {
            const core::PhaseVector& transformation =
                baseline.records()[clean.front().record_index].value;
            const core::PhaseVector prediction =
                expected.key.composed(transformation);
            const core::PhaseVector target =
                expected.key.composed(expected.value);
            if (prediction.similarity(target) >=
                success_similarity) {
                ++clean_successes;
            }
        }

        const core::PhaseVector noisy_query = perturb(
            expected.key,
            config.noise_radians,
            noise_rng
        );
        const auto noisy = baseline.retrieve(noisy_query, 1U);
        if (!noisy.empty()) {
            const core::PhaseVector& transformation =
                baseline.records()[noisy.front().record_index].value;
            const core::PhaseVector prediction =
                noisy_query.composed(transformation);
            const core::PhaseVector target =
                noisy_query.composed(expected.value);
            if (prediction.similarity(target) >=
                success_similarity) {
                ++noisy_successes;
            }
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const double query_count =
        static_cast<double>(config.evaluation_queries);
    return {
        .clean_accuracy =
            static_cast<double>(clean_successes) / query_count,
        .noisy_accuracy =
            static_cast<double>(noisy_successes) / query_count,
        .inference_seconds =
            std::chrono::duration<double>(end - start).count(),
        .candidates_returned = 1.0,
    };
}

[[nodiscard]] CapacityScaleResult run_scale(
    const CapacityScalingConfig& config,
    const std::size_t mode_count
) {
    double rlf_training_seconds = 0.0;
    std::uint64_t rlf_hash = fnv_offset_basis;
    const std::vector<core::ResonantMode> modes = make_modes(
        config,
        mode_count,
        rlf_training_seconds,
        rlf_hash
    );
    const Evaluation rlf_evaluation =
        evaluate_modes(config, modes);
    std::size_t rlf_bytes = sizeof(modes);
    for (const core::ResonantMode& mode : modes) {
        rlf_bytes +=
            learning::StructuralLearner::estimate_mode_bytes(mode);
    }

    double baseline_training_seconds = 0.0;
    std::uint64_t baseline_hash = fnv_offset_basis;
    const baselines::NearestNeighborMemory baseline = make_baseline(
        config,
        mode_count,
        baseline_training_seconds,
        baseline_hash
    );
    const Evaluation baseline_evaluation =
        evaluate_baseline(config, baseline);

    const double training_examples =
        static_cast<double>(mode_count);
    const double inference_queries =
        static_cast<double>(config.evaluation_queries * 2U);
    const double rlf_post_retrieval_operations =
        rlf_evaluation.candidates_returned +
        static_cast<double>(config.active_count);
    const double rlf_total_operations =
        static_cast<double>(mode_count) +
        rlf_post_retrieval_operations;
    const double baseline_total_operations =
        static_cast<double>(mode_count) + 1.0;
    const std::size_t peak_resident = peak_resident_memory_bytes();

    return {
        .mode_count = mode_count,
        .rlf = {
            .system = "rlf_exact_retrieval",
            .clean_accuracy = rlf_evaluation.clean_accuracy,
            .noisy_accuracy = rlf_evaluation.noisy_accuracy,
            .useful_transformations =
                static_cast<double>(mode_count) *
                rlf_evaluation.noisy_accuracy,
            .exact_similarity_evaluations_per_inference =
                static_cast<double>(mode_count),
            .post_retrieval_operations_per_inference =
                rlf_post_retrieval_operations,
            .maximum_candidates_returned = std::min(
                config.candidate_count,
                mode_count
            ),
            .metrics = {
                .task_accuracy = rlf_evaluation.noisy_accuracy,
                .one_shot_recall = rlf_evaluation.clean_accuracy,
                .retained_accuracy = rlf_evaluation.noisy_accuracy,
                .catastrophic_forgetting_score = 0.0,
                .compositional_generalization_score = 0.0,
                .prediction_error =
                    1.0 - rlf_evaluation.noisy_accuracy,
                .average_settling_cycles = 1.0,
                .average_modes_retrieved =
                    rlf_evaluation.candidates_returned,
                .average_modes_activated =
                    static_cast<double>(config.active_count),
                .modes_created =
                    static_cast<std::uint64_t>(mode_count),
                .modes_split = 0ULL,
                .modes_merged = 0ULL,
                .modes_pruned = 0ULL,
                .bytes_stored = rlf_bytes,
                .peak_resident_bytes = peak_resident,
                .training_seconds = rlf_training_seconds,
                .inference_seconds =
                    rlf_evaluation.inference_seconds,
                .training_examples_per_second =
                    training_examples / rlf_training_seconds,
                .inference_examples_per_second =
                    inference_queries /
                    rlf_evaluation.inference_seconds,
                .update_operations_per_example = 1.0,
                .active_operations_per_inference =
                    rlf_total_operations,
                .efficiency_score = provisional_efficiency_score(
                    static_cast<double>(mode_count) *
                        rlf_evaluation.noisy_accuracy,
                    rlf_bytes,
                    rlf_total_operations,
                    rlf_training_seconds
                ),
                .deterministic_run_hash = rlf_hash,
            },
        },
        .baseline = {
            .system = "nearest_neighbor_transformation_memory",
            .clean_accuracy = baseline_evaluation.clean_accuracy,
            .noisy_accuracy = baseline_evaluation.noisy_accuracy,
            .useful_transformations =
                static_cast<double>(mode_count) *
                baseline_evaluation.noisy_accuracy,
            .exact_similarity_evaluations_per_inference =
                static_cast<double>(mode_count),
            .post_retrieval_operations_per_inference = 1.0,
            .maximum_candidates_returned = 1U,
            .metrics = {
                .task_accuracy = baseline_evaluation.noisy_accuracy,
                .one_shot_recall =
                    baseline_evaluation.clean_accuracy,
                .retained_accuracy =
                    baseline_evaluation.noisy_accuracy,
                .catastrophic_forgetting_score = 0.0,
                .compositional_generalization_score = 0.0,
                .prediction_error =
                    1.0 - baseline_evaluation.noisy_accuracy,
                .average_settling_cycles = 0.0,
                .average_modes_retrieved = 1.0,
                .average_modes_activated = 1.0,
                .modes_created = 0ULL,
                .modes_split = 0ULL,
                .modes_merged = 0ULL,
                .modes_pruned = 0ULL,
                .bytes_stored = baseline.bytes_stored(),
                .peak_resident_bytes = peak_resident,
                .training_seconds = baseline_training_seconds,
                .inference_seconds =
                    baseline_evaluation.inference_seconds,
                .training_examples_per_second =
                    training_examples / baseline_training_seconds,
                .inference_examples_per_second =
                    inference_queries /
                    baseline_evaluation.inference_seconds,
                .update_operations_per_example = 1.0,
                .active_operations_per_inference =
                    baseline_total_operations,
                .efficiency_score = provisional_efficiency_score(
                    static_cast<double>(mode_count) *
                        baseline_evaluation.noisy_accuracy,
                    baseline.bytes_stored(),
                    baseline_total_operations,
                    baseline_training_seconds
                ),
                .deterministic_run_hash = baseline_hash,
            },
        },
    };
}

void write_system(
    std::ostream& output,
    const CapacitySystemResult& system,
    const std::size_t indentation
) {
    const std::string indent(indentation, ' ');
    const std::string field_indent(indentation + 2U, ' ');
    output << indent << "{\n"
           << field_indent << "\"system\": \""
           << system.system << "\",\n"
           << field_indent << "\"clean_accuracy\": "
           << system.clean_accuracy << ",\n"
           << field_indent << "\"noisy_accuracy\": "
           << system.noisy_accuracy << ",\n"
           << field_indent << "\"useful_transformations\": "
           << system.useful_transformations << ",\n"
           << field_indent
           << "\"exact_similarity_evaluations_per_inference\": "
           << system.exact_similarity_evaluations_per_inference
           << ",\n"
           << field_indent
           << "\"post_retrieval_operations_per_inference\": "
           << system.post_retrieval_operations_per_inference
           << ",\n"
           << field_indent << "\"maximum_candidates_returned\": "
           << system.maximum_candidates_returned << ",\n"
           << field_indent << "\"metrics\": ";
    write_metrics_json(output, system.metrics, indentation + 2U);
    output << '\n' << indent << '}';
}

}  // namespace

CapacityScalingResult run_capacity_scaling(
    const CapacityScalingConfig& config
) {
    if (config.dimension == 0U ||
        config.evaluation_queries == 0U ||
        config.candidate_count == 0U ||
        config.active_count == 0U ||
        config.active_count > config.candidate_count ||
        !std::isfinite(config.noise_radians) ||
        config.noise_radians < 0.0 ||
        config.mode_counts.empty()) {
        throw std::invalid_argument(
            "invalid capacity-scaling configuration"
        );
    }
    for (const std::size_t mode_count : config.mode_counts) {
        if (mode_count == 0U) {
            throw std::invalid_argument(
                "capacity-scaling mode counts must be positive"
            );
        }
    }
    if (!std::is_sorted(
            config.mode_counts.begin(),
            config.mode_counts.end()
        )) {
        throw std::invalid_argument(
            "capacity-scaling mode counts must be sorted"
        );
    }

    CapacityScalingResult result{
        .seed = config.seed,
        .dimension = config.dimension,
        .evaluation_queries = config.evaluation_queries,
        .candidate_count = config.candidate_count,
        .active_count = config.active_count,
        .noise_radians = config.noise_radians,
        .scales = {},
        .rlf_post_retrieval_work_bounded = true,
        .rlf_total_exact_work_bounded = true,
        .deterministic_run_hash = fnv_offset_basis,
    };
    result.scales.reserve(config.mode_counts.size());
    for (const std::size_t mode_count : config.mode_counts) {
        result.scales.push_back(run_scale(config, mode_count));
    }

    const double first_post_work =
        result.scales.front().rlf
            .post_retrieval_operations_per_inference;
    const double first_total_work =
        result.scales.front().rlf.metrics
            .active_operations_per_inference;
    for (const CapacityScaleResult& scale : result.scales) {
        result.rlf_post_retrieval_work_bounded =
            result.rlf_post_retrieval_work_bounded &&
            scale.rlf.post_retrieval_operations_per_inference <=
                first_post_work + 1.0e-12;
        result.rlf_total_exact_work_bounded =
            result.rlf_total_exact_work_bounded &&
            scale.rlf.metrics.active_operations_per_inference <=
                first_total_work + 1.0e-12;
        hash_u64(
            result.deterministic_run_hash,
            static_cast<std::uint64_t>(scale.mode_count)
        );
        hash_string(
            result.deterministic_run_hash,
            scale.rlf.system
        );
        hash_u64(
            result.deterministic_run_hash,
            scale.rlf.metrics.deterministic_run_hash
        );
        hash_u64(
            result.deterministic_run_hash,
            scale.baseline.metrics.deterministic_run_hash
        );
    }
    hash_u64(result.deterministic_run_hash, config.seed);
    return result;
}

void write_capacity_scaling_json(
    std::ostream& output,
    const CapacityScalingResult& result
) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "{\n"
           << "  \"experiment\": \"capacity_scaling\",\n"
           << "  \"status\": \"observed\",\n"
           << "  \"seed\": " << result.seed << ",\n"
           << "  \"dimension\": " << result.dimension << ",\n"
           << "  \"evaluation_queries\": "
           << result.evaluation_queries << ",\n"
           << "  \"candidate_count\": "
           << result.candidate_count << ",\n"
           << "  \"active_count\": "
           << result.active_count << ",\n"
           << "  \"noise_radians\": "
           << result.noise_radians << ",\n"
           << "  \"rlf_post_retrieval_work_bounded\": "
           << (result.rlf_post_retrieval_work_bounded
                   ? "true"
                   : "false")
           << ",\n"
           << "  \"rlf_total_exact_work_bounded\": "
           << (result.rlf_total_exact_work_bounded
                   ? "true"
                   : "false")
           << ",\n"
           << "  \"scales\": [\n";
    for (std::size_t scale_index = 0U;
         scale_index < result.scales.size();
         ++scale_index) {
        const CapacityScaleResult& scale =
            result.scales[scale_index];
        output << "    {\n"
               << "      \"mode_count\": "
               << scale.mode_count << ",\n"
               << "      \"rlf\": ";
        write_system(output, scale.rlf, 6U);
        output << ",\n      \"baseline\": ";
        write_system(output, scale.baseline, 6U);
        output << "\n    }";
        if (scale_index + 1U != result.scales.size()) {
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
            "failed to write capacity-scaling result"
        );
    }
}

}  // namespace rlf::experiments
