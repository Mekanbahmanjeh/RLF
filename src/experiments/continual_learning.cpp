#include "rlf/experiments/continual_learning.hpp"

#include "rlf/core/deterministic_rng.hpp"
#include "rlf/core/phase_vector.hpp"
#include "rlf/core/resonant_fabric.hpp"
#include "rlf/experiments/phase_vector_smoke.hpp"
#include "rlf/learning/local_learning.hpp"
#include "rlf/learning/structural_learning.hpp"

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

constexpr std::size_t task_count = 4U;
constexpr double success_similarity = 0.95;
constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;

struct Example final {
    core::PhaseVector input;
    core::PhaseVector target;
};

struct RlfEvaluation final {
    double accuracy;
    double cycles;
    double retrieved;
    double activated;
};

class SingleTransformationBaseline final {
public:
    explicit SingleTransformationBaseline(
        const std::size_t dimension
    )
        : transformation_(core::PhaseVector::zeros(dimension)) {}

    [[nodiscard]] core::PhaseVector predict(
        const core::PhaseVector& input
    ) const {
        return input.composed(transformation_);
    }

    [[nodiscard]] double update(
        const core::PhaseVector& input,
        const core::PhaseVector& target,
        const double learning_rate
    ) {
        const double error = 1.0 - predict(input).similarity(target);
        const core::PhaseVector desired =
            core::PhaseVector::phase_difference(input, target);
        const std::vector<core::PhaseVector> values{
            transformation_,
            desired,
        };
        const std::vector<float> weights{
            static_cast<float>(1.0 - learning_rate),
            static_cast<float>(learning_rate),
        };
        transformation_ =
            core::PhaseVector::weighted_circular_average(
                values,
                weights
            );
        return error;
    }

    [[nodiscard]] const core::PhaseVector& transformation() const noexcept {
        return transformation_;
    }

    [[nodiscard]] std::size_t bytes_stored() const noexcept {
        return sizeof(*this) +
            (transformation_.size() *
             sizeof(core::PhaseVector::Angle));
    }

private:
    core::PhaseVector transformation_;
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

void hash_phase_vector(
    std::uint64_t& hash,
    const core::PhaseVector& value
) noexcept {
    for (const float angle : value.angles()) {
        hash_u64(
            hash,
            static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(angle))
        );
    }
}

[[nodiscard]] core::PhaseVector perturb(
    const core::PhaseVector& prototype,
    const double noise_radians,
    core::DeterministicRng& random_number_generator
) {
    std::vector<float> noise;
    noise.reserve(prototype.size());
    for (std::size_t index = 0U; index < prototype.size(); ++index) {
        const double signed_unit =
            (2.0 * random_number_generator.uniform_unit()) - 1.0;
        noise.push_back(
            static_cast<float>(signed_unit * noise_radians)
        );
    }
    return prototype.composed(core::PhaseVector(std::move(noise)));
}

[[nodiscard]] std::vector<std::vector<Example>> make_task_examples(
    const std::vector<core::PhaseVector>& contexts,
    const std::vector<core::PhaseVector>& transformations,
    const std::size_t examples_per_task,
    const double context_noise_radians,
    const std::uint64_t seed
) {
    std::vector<std::vector<Example>> task_examples(task_count);
    for (std::size_t task_index = 0U;
         task_index < task_count;
         ++task_index) {
        core::DeterministicRng task_rng(
            seed ^ static_cast<std::uint64_t>(
                (task_index + 1U) * 0x9E37U
            )
        );
        task_examples[task_index].reserve(examples_per_task);
        for (std::size_t example_index = 0U;
             example_index < examples_per_task;
             ++example_index) {
            core::PhaseVector input = perturb(
                contexts[task_index],
                context_noise_radians,
                task_rng
            );
            core::PhaseVector target = input.composed(
                transformations[task_index]
            );
            task_examples[task_index].push_back({
                .input = std::move(input),
                .target = std::move(target),
            });
        }
    }
    return task_examples;
}

[[nodiscard]] core::FabricConfig make_fabric_config(
    const ContinualLearningConfig& config
) {
    core::SettlingConfig settling;
    settling.candidate_count = task_count;
    settling.active_count = 1U;
    settling.maximum_cycles = 1U;
    settling.minimum_cycles = 1U;
    settling.minimum_resonance = 0.2;
    settling.convergence_tolerance_radians = 0.0;
    settling.input_weight = 0.0;
    settling.previous_state_weight = 0.0;
    settling.proposal_weight_scale = 1.0;
    settling.utility_weight = 0.0;

    learning::StructuralLearningConfig structural;
    structural.enabled = true;
    structural.enable_creation = true;
    structural.enable_splitting = false;
    structural.enable_merging = false;
    structural.enable_pruning = false;
    structural.creation_minimum_resonance = 0.2;
    structural.creation_prediction_error_threshold = 1.0;
    structural.creation_confidence = 1.0F;
    structural.creation_utility = 0.0F;
    structural.creation_selectivity = 1.0F;

    return {
        .dimension = config.dimension,
        .maximum_modes = task_count,
        .settling = settling,
        .structural_learning = structural,
    };
}

[[nodiscard]] RlfEvaluation evaluate_rlf_task(
    core::ResonantFabric& fabric,
    const std::vector<Example>& examples
) {
    std::size_t successes = 0U;
    double cycles_total = 0.0;
    double retrieved_total = 0.0;
    double activated_total = 0.0;
    for (const Example& example : examples) {
        const core::SettleResult prediction =
            fabric.settle(example.input, true);
        if (prediction.state.similarity(example.target) >=
            success_similarity) {
            ++successes;
        }
        cycles_total += static_cast<double>(prediction.cycles);
        if (prediction.trace.has_value() &&
            !prediction.trace->cycles.empty()) {
            retrieved_total += static_cast<double>(
                prediction.trace->cycles.back()
                    .retrieved_mode_ids.size()
            );
            activated_total += static_cast<double>(
                prediction.trace->cycles.back()
                    .active_mode_ids.size()
            );
        }
    }
    const double count = static_cast<double>(examples.size());
    return {
        .accuracy = static_cast<double>(successes) / count,
        .cycles = cycles_total / count,
        .retrieved = retrieved_total / count,
        .activated = activated_total / count,
    };
}

[[nodiscard]] double evaluate_baseline_task(
    const SingleTransformationBaseline& baseline,
    const std::vector<Example>& examples
) {
    std::size_t successes = 0U;
    for (const Example& example : examples) {
        if (baseline.predict(example.input).similarity(example.target) >=
            success_similarity) {
            ++successes;
        }
    }
    return static_cast<double>(successes) /
        static_cast<double>(examples.size());
}

[[nodiscard]] std::size_t fabric_bytes(
    const core::ResonantFabric& fabric
) {
    std::size_t bytes = sizeof(fabric);
    for (const core::ResonantMode& mode : fabric.modes()) {
        bytes += learning::StructuralLearner::estimate_mode_bytes(mode);
    }
    return bytes;
}

[[nodiscard]] double final_average(
    const std::vector<std::vector<double>>& matrix
) {
    const std::vector<double>& final_row = matrix.back();
    double total = 0.0;
    for (const double accuracy : final_row) {
        total += accuracy;
    }
    return total / static_cast<double>(final_row.size());
}

[[nodiscard]] double backward_transfer(
    const std::vector<std::vector<double>>& matrix
) {
    double total = 0.0;
    for (std::size_t task_index = 0U;
         task_index + 1U < task_count;
         ++task_index) {
        total += matrix.back()[task_index] -
            matrix[task_index][task_index];
    }
    return total / static_cast<double>(task_count - 1U);
}

[[nodiscard]] double average_forgetting(
    const std::vector<std::vector<double>>& matrix
) {
    double total = 0.0;
    for (std::size_t task_index = 0U;
         task_index + 1U < task_count;
         ++task_index) {
        double best = matrix[task_index][task_index];
        for (std::size_t row = task_index + 1U;
             row < matrix.size();
             ++row) {
            best = std::max(best, matrix[row][task_index]);
        }
        total += best - matrix.back()[task_index];
    }
    return total / static_cast<double>(task_count - 1U);
}

void write_matrix(
    std::ostream& output,
    const std::vector<std::vector<double>>& matrix,
    const std::size_t indentation
) {
    const std::string indent(indentation, ' ');
    const std::string row_indent(indentation + 2U, ' ');
    output << "[\n";
    for (std::size_t row = 0U; row < matrix.size(); ++row) {
        output << row_indent << '[';
        for (std::size_t column = 0U;
             column < matrix[row].size();
             ++column) {
            output << matrix[row][column];
            if (column + 1U != matrix[row].size()) {
                output << ", ";
            }
        }
        output << ']';
        if (row + 1U != matrix.size()) {
            output << ',';
        }
        output << '\n';
    }
    output << indent << ']';
}

void write_size_array(
    std::ostream& output,
    const std::vector<std::size_t>& values
) {
    output << '[';
    for (std::size_t index = 0U; index < values.size(); ++index) {
        output << values[index];
        if (index + 1U != values.size()) {
            output << ", ";
        }
    }
    output << ']';
}

}  // namespace

ContinualLearningResult run_continual_learning(
    const ContinualLearningConfig& config
) {
    if (config.dimension == 0U ||
        config.training_examples_per_task == 0U ||
        config.evaluation_examples_per_task == 0U ||
        !std::isfinite(config.context_noise_radians) ||
        config.context_noise_radians < 0.0) {
        throw std::invalid_argument(
            "invalid continual-learning configuration"
        );
    }

    core::DeterministicRng random_number_generator(config.seed);
    std::vector<core::PhaseVector> contexts;
    std::vector<core::PhaseVector> transformations;
    contexts.reserve(task_count);
    transformations.reserve(task_count);
    for (std::size_t task_index = 0U;
         task_index < task_count;
         ++task_index) {
        contexts.push_back(core::PhaseVector::random(
            config.dimension,
            random_number_generator
        ));
        transformations.push_back(core::PhaseVector::random(
            config.dimension,
            random_number_generator
        ));
    }
    const auto training_tasks = make_task_examples(
        contexts,
        transformations,
        config.training_examples_per_task,
        config.context_noise_radians,
        config.seed ^ 0x1111ULL
    );
    const auto evaluation_tasks = make_task_examples(
        contexts,
        transformations,
        config.evaluation_examples_per_task,
        config.context_noise_radians,
        config.seed ^ 0x2222ULL
    );

    core::ResonantFabric fabric(make_fabric_config(config));
    fabric.set_update_strategy(
        std::make_unique<learning::WinnerOnlyUpdate>()
    );
    SingleTransformationBaseline baseline(config.dimension);
    learning::LocalLearningConfig learning_config;
    learning_config.transformation_learning_rate = 0.2;
    learning_config.context_learning_rate = 0.0;
    learning_config.confidence_learning_rate = 0.0;
    learning_config.utility_learning_rate = 0.0;
    learning_config.use_utility = false;

    std::vector<std::vector<double>> rlf_matrix;
    std::vector<std::vector<double>> baseline_matrix;
    std::vector<std::size_t> rlf_growth;
    std::vector<std::size_t> baseline_growth;
    std::vector<std::size_t> rlf_memory_growth;
    std::vector<std::size_t> baseline_memory_growth;
    double rlf_training_seconds = 0.0;
    double baseline_training_seconds = 0.0;
    double rlf_inference_seconds = 0.0;
    double baseline_inference_seconds = 0.0;
    double rlf_prediction_error_total = 0.0;
    double baseline_prediction_error_total = 0.0;
    double rlf_cycles_total = 0.0;
    double rlf_retrieved_total = 0.0;
    double rlf_activated_total = 0.0;
    std::size_t rlf_evaluations = 0U;
    std::size_t update_operations = 0U;
    double rlf_one_shot_total = 0.0;
    double baseline_one_shot_total = 0.0;

    for (std::size_t task_index = 0U;
         task_index < task_count;
         ++task_index) {
        const auto rlf_training_start =
            std::chrono::steady_clock::now();
        for (std::size_t example_index = 0U;
             example_index <
                training_tasks[task_index].size();
             ++example_index) {
            const Example& example =
                training_tasks[task_index][example_index];
            const core::LearningResult update = fabric.learn(
                example.input,
                example.target,
                learning_config
            );
            rlf_prediction_error_total += update.prediction_error;
            update_operations += update.updated_mode_ids.size();
            update_operations += update.structural_events.size();
            if (example_index == 0U) {
                const RlfEvaluation one_shot =
                    evaluate_rlf_task(
                        fabric,
                        evaluation_tasks[task_index]
                    );
                rlf_one_shot_total += one_shot.accuracy;
            }
        }
        const auto rlf_training_end =
            std::chrono::steady_clock::now();
        rlf_training_seconds += std::chrono::duration<double>(
            rlf_training_end - rlf_training_start
        ).count();

        const auto baseline_training_start =
            std::chrono::steady_clock::now();
        for (std::size_t example_index = 0U;
             example_index <
                training_tasks[task_index].size();
             ++example_index) {
            const Example& example =
                training_tasks[task_index][example_index];
            baseline_prediction_error_total += baseline.update(
                example.input,
                example.target,
                0.2
            );
            if (example_index == 0U) {
                baseline_one_shot_total += evaluate_baseline_task(
                    baseline,
                    evaluation_tasks[task_index]
                );
            }
        }
        const auto baseline_training_end =
            std::chrono::steady_clock::now();
        baseline_training_seconds += std::chrono::duration<double>(
            baseline_training_end - baseline_training_start
        ).count();

        std::vector<double> rlf_row;
        std::vector<double> baseline_row;
        rlf_row.reserve(task_index + 1U);
        baseline_row.reserve(task_index + 1U);
        for (std::size_t evaluation_task = 0U;
             evaluation_task <= task_index;
             ++evaluation_task) {
            const auto rlf_inference_start =
                std::chrono::steady_clock::now();
            const RlfEvaluation rlf_evaluation = evaluate_rlf_task(
                fabric,
                evaluation_tasks[evaluation_task]
            );
            const auto rlf_inference_end =
                std::chrono::steady_clock::now();
            rlf_inference_seconds +=
                std::chrono::duration<double>(
                    rlf_inference_end - rlf_inference_start
                ).count();
            rlf_row.push_back(rlf_evaluation.accuracy);
            rlf_cycles_total += rlf_evaluation.cycles *
                static_cast<double>(
                    config.evaluation_examples_per_task
                );
            rlf_retrieved_total += rlf_evaluation.retrieved *
                static_cast<double>(
                    config.evaluation_examples_per_task
                );
            rlf_activated_total += rlf_evaluation.activated *
                static_cast<double>(
                    config.evaluation_examples_per_task
                );
            rlf_evaluations +=
                config.evaluation_examples_per_task;

            const auto baseline_inference_start =
                std::chrono::steady_clock::now();
            baseline_row.push_back(evaluate_baseline_task(
                baseline,
                evaluation_tasks[evaluation_task]
            ));
            const auto baseline_inference_end =
                std::chrono::steady_clock::now();
            baseline_inference_seconds +=
                std::chrono::duration<double>(
                    baseline_inference_end -
                    baseline_inference_start
                ).count();
        }
        rlf_matrix.push_back(std::move(rlf_row));
        baseline_matrix.push_back(std::move(baseline_row));
        rlf_growth.push_back(fabric.modes().size());
        baseline_growth.push_back(1U);
        rlf_memory_growth.push_back(fabric_bytes(fabric));
        baseline_memory_growth.push_back(baseline.bytes_stored());
    }

    const double rlf_retained = final_average(rlf_matrix);
    const double baseline_retained = final_average(baseline_matrix);
    const double rlf_bwt = backward_transfer(rlf_matrix);
    const double baseline_bwt = backward_transfer(baseline_matrix);
    const double rlf_forgetting = average_forgetting(rlf_matrix);
    const double baseline_forgetting =
        average_forgetting(baseline_matrix);
    const double total_training_examples = static_cast<double>(
        config.training_examples_per_task * task_count
    );
    const double total_evaluation_examples =
        static_cast<double>(rlf_evaluations);
    const double rlf_average_cycles =
        rlf_cycles_total / total_evaluation_examples;
    const double rlf_average_retrieved =
        rlf_retrieved_total / total_evaluation_examples;
    const double rlf_average_activated =
        rlf_activated_total / total_evaluation_examples;

    std::uint64_t rlf_hash = fnv_offset_basis;
    std::uint64_t baseline_hash = fnv_offset_basis;
    for (const core::ResonantMode& mode : fabric.modes()) {
        hash_u64(rlf_hash, mode.id);
        hash_phase_vector(rlf_hash, mode.context_key);
        hash_phase_vector(rlf_hash, mode.transformation);
    }
    hash_phase_vector(
        baseline_hash,
        baseline.transformation()
    );
    for (const auto& row : rlf_matrix) {
        for (const double accuracy : row) {
            hash_u64(
                rlf_hash,
                std::bit_cast<std::uint64_t>(accuracy)
            );
        }
    }
    for (const auto& row : baseline_matrix) {
        for (const double accuracy : row) {
            hash_u64(
                baseline_hash,
                std::bit_cast<std::uint64_t>(accuracy)
            );
        }
    }

    const std::size_t peak_resident = peak_resident_memory_bytes();
    ContinualSystemResult rlf_result{
        .system = "rlf_context_specialized_modes",
        .accuracy_after_task = std::move(rlf_matrix),
        .mode_or_state_growth = std::move(rlf_growth),
        .memory_growth_bytes = std::move(rlf_memory_growth),
        .average_retained_accuracy = rlf_retained,
        .backward_transfer = rlf_bwt,
        .average_forgetting = rlf_forgetting,
        .metrics = {
            .task_accuracy = rlf_retained,
            .one_shot_recall =
                rlf_one_shot_total /
                static_cast<double>(task_count),
            .retained_accuracy = rlf_retained,
            .catastrophic_forgetting_score = rlf_forgetting,
            .compositional_generalization_score = 0.0,
            .prediction_error =
                rlf_prediction_error_total /
                total_training_examples,
            .average_settling_cycles = rlf_average_cycles,
            .average_modes_retrieved = rlf_average_retrieved,
            .average_modes_activated = rlf_average_activated,
            .modes_created =
                fabric.structural_statistics().modes_created,
            .modes_split =
                fabric.structural_statistics().modes_split,
            .modes_merged =
                fabric.structural_statistics().modes_merged,
            .modes_pruned =
                fabric.structural_statistics().modes_pruned,
            .bytes_stored = fabric_bytes(fabric),
            .peak_resident_bytes = peak_resident,
            .training_seconds = rlf_training_seconds,
            .inference_seconds = rlf_inference_seconds,
            .training_examples_per_second =
                total_training_examples /
                rlf_training_seconds,
            .inference_examples_per_second =
                total_evaluation_examples /
                rlf_inference_seconds,
            .update_operations_per_example =
                static_cast<double>(update_operations) /
                total_training_examples,
            .active_operations_per_inference =
                rlf_average_retrieved +
                rlf_average_activated,
            .efficiency_score = provisional_efficiency_score(
                rlf_retained *
                    static_cast<double>(task_count),
                fabric_bytes(fabric),
                rlf_average_retrieved +
                    rlf_average_activated,
                rlf_training_seconds
            ),
            .deterministic_run_hash = rlf_hash,
        },
    };
    ContinualSystemResult baseline_result{
        .system = "single_prototype_transformation_baseline",
        .accuracy_after_task = std::move(baseline_matrix),
        .mode_or_state_growth = std::move(baseline_growth),
        .memory_growth_bytes = std::move(baseline_memory_growth),
        .average_retained_accuracy = baseline_retained,
        .backward_transfer = baseline_bwt,
        .average_forgetting = baseline_forgetting,
        .metrics = {
            .task_accuracy = baseline_retained,
            .one_shot_recall =
                baseline_one_shot_total /
                static_cast<double>(task_count),
            .retained_accuracy = baseline_retained,
            .catastrophic_forgetting_score =
                baseline_forgetting,
            .compositional_generalization_score = 0.0,
            .prediction_error =
                baseline_prediction_error_total /
                total_training_examples,
            .average_settling_cycles = 0.0,
            .average_modes_retrieved = 0.0,
            .average_modes_activated = 0.0,
            .modes_created = 0ULL,
            .modes_split = 0ULL,
            .modes_merged = 0ULL,
            .modes_pruned = 0ULL,
            .bytes_stored = baseline.bytes_stored(),
            .peak_resident_bytes = peak_resident,
            .training_seconds = baseline_training_seconds,
            .inference_seconds = baseline_inference_seconds,
            .training_examples_per_second =
                total_training_examples /
                baseline_training_seconds,
            .inference_examples_per_second =
                total_evaluation_examples /
                baseline_inference_seconds,
            .update_operations_per_example = 1.0,
            .active_operations_per_inference = 1.0,
            .efficiency_score = provisional_efficiency_score(
                baseline_retained,
                baseline.bytes_stored(),
                1.0,
                baseline_training_seconds
            ),
            .deterministic_run_hash = baseline_hash,
        },
    };

    std::uint64_t run_hash = fnv_offset_basis;
    hash_string(run_hash, rlf_result.system);
    hash_u64(run_hash, rlf_hash);
    hash_string(run_hash, baseline_result.system);
    hash_u64(run_hash, baseline_hash);
    return {
        .seed = config.seed,
        .dimension = config.dimension,
        .task_count = task_count,
        .training_examples_per_task =
            config.training_examples_per_task,
        .evaluation_examples_per_task =
            config.evaluation_examples_per_task,
        .context_noise_radians = config.context_noise_radians,
        .rlf = std::move(rlf_result),
        .baseline = std::move(baseline_result),
        .deterministic_run_hash = run_hash,
    };
}

void write_continual_learning_json(
    std::ostream& output,
    const ContinualLearningResult& result
) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "{\n"
           << "  \"experiment\": \"continual_learning\",\n"
           << "  \"status\": \"observed\",\n"
           << "  \"seed\": " << result.seed << ",\n"
           << "  \"dimension\": " << result.dimension << ",\n"
           << "  \"task_count\": " << result.task_count << ",\n"
           << "  \"training_examples_per_task\": "
           << result.training_examples_per_task << ",\n"
           << "  \"evaluation_examples_per_task\": "
           << result.evaluation_examples_per_task << ",\n"
           << "  \"context_noise_radians\": "
           << result.context_noise_radians << ",\n";

    const auto write_system = [&output](
        const std::string& field,
        const ContinualSystemResult& system,
        const bool trailing_comma
    ) {
        output << "  \"" << field << "\": {\n"
               << "    \"system\": \"" << system.system << "\",\n"
               << "    \"accuracy_after_task\": ";
        write_matrix(output, system.accuracy_after_task, 4U);
        output << ",\n"
               << "    \"mode_or_state_growth\": ";
        write_size_array(output, system.mode_or_state_growth);
        output << ",\n"
               << "    \"memory_growth_bytes\": ";
        write_size_array(output, system.memory_growth_bytes);
        output << ",\n"
               << "    \"average_retained_accuracy\": "
               << system.average_retained_accuracy << ",\n"
               << "    \"backward_transfer\": "
               << system.backward_transfer << ",\n"
               << "    \"average_forgetting\": "
               << system.average_forgetting << ",\n"
               << "    \"metrics\": ";
        write_metrics_json(output, system.metrics, 4U);
        output << "\n  }";
        if (trailing_comma) {
            output << ',';
        }
        output << '\n';
    };
    write_system("rlf", result.rlf, true);
    write_system("baseline", result.baseline, true);
    output << "  \"deterministic_run_hash\": \""
           << format_run_hash(result.deterministic_run_hash)
           << "\"\n"
           << "}\n";
    if (!output) {
        throw std::runtime_error(
            "failed to write continual-learning result"
        );
    }
}

}  // namespace rlf::experiments
