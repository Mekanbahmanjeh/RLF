#include "rlf/experiments/compositional_generalization.hpp"

#include "rlf/baselines/nearest_neighbor.hpp"
#include "rlf/core/deterministic_rng.hpp"
#include "rlf/core/encoding.hpp"
#include "rlf/core/phase_vector.hpp"
#include "rlf/core/resonant_fabric.hpp"
#include "rlf/core/resonant_mode.hpp"
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
#include <numeric>
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

enum class TransformationKind {
    additive,
    permutation,
    conjugation,
};

struct Transformation final {
    TransformationKind kind;
    core::PhaseVector additive;
    std::vector<std::size_t> permutation;
};

struct Example final {
    core::PhaseVector input;
    core::PhaseVector target;
};

struct Evaluation final {
    double mean_similarity{};
    double accuracy{};
    double seconds{};
};

struct TrainingMeasurement final {
    double seconds{};
    double prediction_error{};
    std::size_t update_operations{};
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

[[nodiscard]] core::PhaseVector apply_transformation(
    const Transformation& transformation,
    const core::PhaseVector& input
) {
    switch (transformation.kind) {
    case TransformationKind::additive:
        return input.composed(transformation.additive);
    case TransformationKind::permutation:
        return input.permuted(transformation.permutation);
    case TransformationKind::conjugation:
        return input.conjugated();
    }
    throw std::logic_error("unknown composition transformation kind");
}

[[nodiscard]] std::vector<Example> make_examples(
    const std::size_t count,
    const std::size_t dimension,
    const Transformation& transformation,
    core::DeterministicRng& random_number_generator
) {
    std::vector<Example> examples;
    examples.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        core::PhaseVector input = core::PhaseVector::random(
            dimension,
            random_number_generator
        );
        core::PhaseVector target = apply_transformation(
            transformation,
            input
        );
        examples.push_back({
            .input = std::move(input),
            .target = std::move(target),
        });
    }
    return examples;
}

[[nodiscard]] std::vector<Example> make_composed_examples(
    const std::size_t count,
    const std::size_t dimension,
    const Transformation& first,
    const Transformation& second,
    core::DeterministicRng& random_number_generator
) {
    std::vector<Example> examples;
    examples.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        core::PhaseVector input = core::PhaseVector::random(
            dimension,
            random_number_generator
        );
        core::PhaseVector target = apply_transformation(
            second,
            apply_transformation(first, input)
        );
        examples.push_back({
            .input = std::move(input),
            .target = std::move(target),
        });
    }
    return examples;
}

[[nodiscard]] core::FabricConfig make_fabric_config(
    const std::size_t dimension
) {
    core::SettlingConfig settling;
    settling.candidate_count = 1U;
    settling.active_count = 1U;
    settling.maximum_cycles = 1U;
    settling.minimum_cycles = 1U;
    settling.minimum_resonance = 0.0;
    settling.convergence_tolerance_radians = 0.0;
    settling.input_weight = 0.0;
    settling.previous_state_weight = 0.0;
    settling.proposal_weight_scale = 1.0;
    settling.utility_weight = 0.0;
    return {
        .dimension = dimension,
        .maximum_modes = 1U,
        .settling = settling,
    };
}

[[nodiscard]] core::ResonantFabric make_fabric(
    const std::size_t dimension,
    const core::PhaseVector& initial_context
) {
    core::ResonantFabric fabric(make_fabric_config(dimension));
    fabric.set_update_strategy(
        std::make_unique<learning::WinnerOnlyUpdate>()
    );
    fabric.add_mode(core::ResonantMode(
        1ULL,
        initial_context,
        core::PhaseVector::zeros(dimension),
        0.05F,
        1.0F,
        0.0F,
        0ULL
    ));
    return fabric;
}

[[nodiscard]] TrainingMeasurement train_fabric(
    core::ResonantFabric& fabric,
    const std::vector<Example>& examples
) {
    learning::LocalLearningConfig learning_config;
    learning_config.transformation_learning_rate = 0.35;
    learning_config.context_learning_rate = 0.0;
    learning_config.confidence_learning_rate = 0.0;
    learning_config.utility_learning_rate = 0.0;
    learning_config.use_utility = false;

    double prediction_error = 0.0;
    std::size_t update_operations = 0U;
    const auto start = std::chrono::steady_clock::now();
    for (const Example& example : examples) {
        const core::LearningResult update = fabric.learn(
            example.input,
            example.target,
            learning_config
        );
        prediction_error += update.prediction_error;
        update_operations += update.updated_mode_ids.size();
    }
    const auto end = std::chrono::steady_clock::now();
    return {
        .seconds = std::chrono::duration<double>(end - start).count(),
        .prediction_error =
            prediction_error / static_cast<double>(examples.size()),
        .update_operations = update_operations,
    };
}

[[nodiscard]] double train_baseline(
    baselines::NearestNeighborMemory& baseline,
    const std::vector<Example>& examples
) {
    const auto start = std::chrono::steady_clock::now();
    for (const Example& example : examples) {
        static_cast<void>(baseline.insert(
            example.input,
            example.target
        ));
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

[[nodiscard]] Evaluation evaluate_fabric(
    core::ResonantFabric& fabric,
    const std::vector<Example>& examples
) {
    double similarity_total = 0.0;
    std::size_t successes = 0U;
    const auto start = std::chrono::steady_clock::now();
    for (const Example& example : examples) {
        const core::SettleResult prediction =
            fabric.settle(example.input);
        const double similarity =
            prediction.state.similarity(example.target);
        similarity_total += similarity;
        if (similarity >= success_similarity) {
            ++successes;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    return {
        .mean_similarity =
            similarity_total / static_cast<double>(examples.size()),
        .accuracy =
            static_cast<double>(successes) /
            static_cast<double>(examples.size()),
        .seconds = std::chrono::duration<double>(end - start).count(),
    };
}

[[nodiscard]] Evaluation evaluate_fabric_composition(
    core::ResonantFabric& first,
    core::ResonantFabric& second,
    const std::vector<Example>& examples
) {
    double similarity_total = 0.0;
    std::size_t successes = 0U;
    const auto start = std::chrono::steady_clock::now();
    for (const Example& example : examples) {
        const core::PhaseVector intermediate =
            first.settle(example.input).state;
        const core::PhaseVector prediction =
            second.settle(intermediate).state;
        const double similarity = prediction.similarity(example.target);
        similarity_total += similarity;
        if (similarity >= success_similarity) {
            ++successes;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    return {
        .mean_similarity =
            similarity_total / static_cast<double>(examples.size()),
        .accuracy =
            static_cast<double>(successes) /
            static_cast<double>(examples.size()),
        .seconds = std::chrono::duration<double>(end - start).count(),
    };
}

[[nodiscard]] core::PhaseVector baseline_predict(
    const baselines::NearestNeighborMemory& baseline,
    const core::PhaseVector& input
) {
    const std::vector<baselines::NearestNeighborMatch> matches =
        baseline.retrieve(input, 1U);
    if (matches.empty()) {
        return input;
    }
    return baseline.records()[matches.front().record_index].value;
}

[[nodiscard]] Evaluation evaluate_baseline(
    const baselines::NearestNeighborMemory& baseline,
    const std::vector<Example>& examples
) {
    double similarity_total = 0.0;
    std::size_t successes = 0U;
    const auto start = std::chrono::steady_clock::now();
    for (const Example& example : examples) {
        const core::PhaseVector prediction = baseline_predict(
            baseline,
            example.input
        );
        const double similarity = prediction.similarity(example.target);
        similarity_total += similarity;
        if (similarity >= success_similarity) {
            ++successes;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    return {
        .mean_similarity =
            similarity_total / static_cast<double>(examples.size()),
        .accuracy =
            static_cast<double>(successes) /
            static_cast<double>(examples.size()),
        .seconds = std::chrono::duration<double>(end - start).count(),
    };
}

[[nodiscard]] Evaluation evaluate_baseline_composition(
    const baselines::NearestNeighborMemory& first,
    const baselines::NearestNeighborMemory& second,
    const std::vector<Example>& examples
) {
    double similarity_total = 0.0;
    std::size_t successes = 0U;
    const auto start = std::chrono::steady_clock::now();
    for (const Example& example : examples) {
        const core::PhaseVector intermediate = baseline_predict(
            first,
            example.input
        );
        const core::PhaseVector prediction = baseline_predict(
            second,
            intermediate
        );
        const double similarity = prediction.similarity(example.target);
        similarity_total += similarity;
        if (similarity >= success_similarity) {
            ++successes;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    return {
        .mean_similarity =
            similarity_total / static_cast<double>(examples.size()),
        .accuracy =
            static_cast<double>(successes) /
            static_cast<double>(examples.size()),
        .seconds = std::chrono::duration<double>(end - start).count(),
    };
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

[[nodiscard]] CompositionCaseResult run_case(
    const CompositionalGeneralizationConfig& config,
    const std::string& name,
    const std::string& transformation_class,
    const bool expected_supported,
    const Transformation& first_transformation,
    const Transformation& second_transformation,
    const std::uint64_t case_seed
) {
    core::DeterministicRng training_rng(case_seed);
    core::DeterministicRng evaluation_rng(
        case_seed ^ 0xA0761D6478BD642FULL
    );
    const std::vector<Example> first_training = make_examples(
        config.training_examples,
        config.dimension,
        first_transformation,
        training_rng
    );
    const std::vector<Example> second_training = make_examples(
        config.training_examples,
        config.dimension,
        second_transformation,
        training_rng
    );
    const std::vector<Example> first_evaluation = make_examples(
        config.evaluation_examples,
        config.dimension,
        first_transformation,
        evaluation_rng
    );
    const std::vector<Example> second_evaluation = make_examples(
        config.evaluation_examples,
        config.dimension,
        second_transformation,
        evaluation_rng
    );
    const std::vector<Example> composed_evaluation =
        make_composed_examples(
            config.evaluation_examples,
            config.dimension,
            first_transformation,
            second_transformation,
            evaluation_rng
        );

    core::ResonantFabric first_fabric = make_fabric(
        config.dimension,
        first_training.front().input
    );
    core::ResonantFabric second_fabric = make_fabric(
        config.dimension,
        second_training.front().input
    );
    const TrainingMeasurement first_training_measurement =
        train_fabric(first_fabric, first_training);
    const TrainingMeasurement second_training_measurement =
        train_fabric(second_fabric, second_training);

    baselines::NearestNeighborMemory first_baseline(config.dimension);
    baselines::NearestNeighborMemory second_baseline(config.dimension);
    const double baseline_training_seconds =
        train_baseline(first_baseline, first_training) +
        train_baseline(second_baseline, second_training);

    const Evaluation first_rlf =
        evaluate_fabric(first_fabric, first_evaluation);
    const Evaluation second_rlf =
        evaluate_fabric(second_fabric, second_evaluation);
    const Evaluation composed_rlf = evaluate_fabric_composition(
        first_fabric,
        second_fabric,
        composed_evaluation
    );
    const Evaluation first_baseline_evaluation =
        evaluate_baseline(first_baseline, first_evaluation);
    const Evaluation second_baseline_evaluation =
        evaluate_baseline(second_baseline, second_evaluation);
    const Evaluation composed_baseline =
        evaluate_baseline_composition(
            first_baseline,
            second_baseline,
            composed_evaluation
        );

    std::uint64_t rlf_hash = fnv_offset_basis;
    hash_string(rlf_hash, name);
    hash_phase_vector(
        rlf_hash,
        first_fabric.modes().front().transformation
    );
    hash_phase_vector(
        rlf_hash,
        second_fabric.modes().front().transformation
    );
    hash_u64(
        rlf_hash,
        std::bit_cast<std::uint64_t>(composed_rlf.mean_similarity)
    );
    hash_u64(
        rlf_hash,
        std::bit_cast<std::uint64_t>(composed_rlf.accuracy)
    );

    std::uint64_t baseline_hash = fnv_offset_basis;
    hash_string(baseline_hash, name);
    for (const baselines::NearestNeighborRecord& record :
         first_baseline.records()) {
        hash_u64(baseline_hash, record.id);
        hash_phase_vector(baseline_hash, record.key);
        hash_phase_vector(baseline_hash, record.value);
    }
    for (const baselines::NearestNeighborRecord& record :
         second_baseline.records()) {
        hash_u64(baseline_hash, record.id);
        hash_phase_vector(baseline_hash, record.key);
        hash_phase_vector(baseline_hash, record.value);
    }
    hash_u64(
        baseline_hash,
        std::bit_cast<std::uint64_t>(composed_baseline.accuracy)
    );

    const double rlf_training_seconds =
        first_training_measurement.seconds +
        second_training_measurement.seconds;
    const double rlf_inference_seconds =
        first_rlf.seconds + second_rlf.seconds + composed_rlf.seconds;
    const double baseline_inference_seconds =
        first_baseline_evaluation.seconds +
        second_baseline_evaluation.seconds +
        composed_baseline.seconds;
    const double total_training_examples =
        static_cast<double>(config.training_examples * 2U);
    const double total_evaluation_examples =
        static_cast<double>(config.evaluation_examples * 4U);
    const std::size_t rlf_bytes =
        fabric_bytes(first_fabric) + fabric_bytes(second_fabric);
    const std::size_t baseline_bytes =
        first_baseline.bytes_stored() +
        second_baseline.bytes_stored();
    const std::size_t peak_resident = peak_resident_memory_bytes();

    return {
        .name = name,
        .transformation_class = transformation_class,
        .expected_supported = expected_supported,
        .rlf = {
            .system = "rlf_two_reusable_modes",
            .first_transformation_accuracy = first_rlf.accuracy,
            .second_transformation_accuracy = second_rlf.accuracy,
            .composed_mean_similarity =
                composed_rlf.mean_similarity,
            .composed_accuracy = composed_rlf.accuracy,
            .learned_units = 2U,
            .metrics = {
                .task_accuracy = composed_rlf.accuracy,
                .one_shot_recall = 0.0,
                .retained_accuracy = composed_rlf.accuracy,
                .catastrophic_forgetting_score = 0.0,
                .compositional_generalization_score =
                    composed_rlf.accuracy,
                .prediction_error =
                    (first_training_measurement.prediction_error +
                     second_training_measurement.prediction_error) /
                    2.0,
                .average_settling_cycles = 1.0,
                .average_modes_retrieved = 1.0,
                .average_modes_activated = 1.0,
                .modes_created = 2ULL,
                .modes_split = 0ULL,
                .modes_merged = 0ULL,
                .modes_pruned = 0ULL,
                .bytes_stored = rlf_bytes,
                .peak_resident_bytes = peak_resident,
                .training_seconds = rlf_training_seconds,
                .inference_seconds = rlf_inference_seconds,
                .training_examples_per_second =
                    total_training_examples / rlf_training_seconds,
                .inference_examples_per_second =
                    total_evaluation_examples /
                    rlf_inference_seconds,
                .update_operations_per_example =
                    static_cast<double>(
                        first_training_measurement.update_operations +
                        second_training_measurement.update_operations
                    ) /
                    total_training_examples,
                .active_operations_per_inference = 4.0,
                .efficiency_score = provisional_efficiency_score(
                    composed_rlf.accuracy * 2.0,
                    rlf_bytes,
                    4.0,
                    rlf_training_seconds
                ),
                .deterministic_run_hash = rlf_hash,
            },
        },
        .baseline = {
            .system = "two_nearest_neighbor_lookup_tables",
            .first_transformation_accuracy =
                first_baseline_evaluation.accuracy,
            .second_transformation_accuracy =
                second_baseline_evaluation.accuracy,
            .composed_mean_similarity =
                composed_baseline.mean_similarity,
            .composed_accuracy = composed_baseline.accuracy,
            .learned_units = config.training_examples * 2U,
            .metrics = {
                .task_accuracy = composed_baseline.accuracy,
                .one_shot_recall = 0.0,
                .retained_accuracy = composed_baseline.accuracy,
                .catastrophic_forgetting_score = 0.0,
                .compositional_generalization_score =
                    composed_baseline.accuracy,
                .prediction_error = 0.0,
                .average_settling_cycles = 0.0,
                .average_modes_retrieved = 0.0,
                .average_modes_activated = 0.0,
                .modes_created = 0ULL,
                .modes_split = 0ULL,
                .modes_merged = 0ULL,
                .modes_pruned = 0ULL,
                .bytes_stored = baseline_bytes,
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
                .active_operations_per_inference =
                    static_cast<double>(
                        config.training_examples * 4U
                    ),
                .efficiency_score = provisional_efficiency_score(
                    composed_baseline.accuracy,
                    baseline_bytes,
                    static_cast<double>(
                        config.training_examples * 4U
                    ),
                    baseline_training_seconds
                ),
                .deterministic_run_hash = baseline_hash,
            },
        },
    };
}

[[nodiscard]] std::vector<std::size_t> cyclic_rotation(
    const std::size_t dimension
) {
    std::vector<std::size_t> permutation(dimension);
    for (std::size_t output_index = 0U;
         output_index < dimension;
         ++output_index) {
        permutation[output_index] =
            (output_index + dimension - 1U) % dimension;
    }
    return permutation;
}

void write_system(
    std::ostream& output,
    const CompositionSystemResult& system,
    const std::size_t indentation
) {
    const std::string indent(indentation, ' ');
    const std::string field_indent(indentation + 2U, ' ');
    output << indent << "{\n"
           << field_indent << "\"system\": \""
           << system.system << "\",\n"
           << field_indent
           << "\"first_transformation_accuracy\": "
           << system.first_transformation_accuracy << ",\n"
           << field_indent
           << "\"second_transformation_accuracy\": "
           << system.second_transformation_accuracy << ",\n"
           << field_indent << "\"composed_mean_similarity\": "
           << system.composed_mean_similarity << ",\n"
           << field_indent << "\"composed_accuracy\": "
           << system.composed_accuracy << ",\n"
           << field_indent << "\"learned_units\": "
           << system.learned_units << ",\n"
           << field_indent << "\"metrics\": ";
    write_metrics_json(output, system.metrics, indentation + 2U);
    output << '\n' << indent << '}';
}

}  // namespace

CompositionalGeneralizationResult run_compositional_generalization(
    const CompositionalGeneralizationConfig& config
) {
    if (config.dimension < 2U ||
        config.training_examples == 0U ||
        config.evaluation_examples == 0U) {
        throw std::invalid_argument(
            "invalid compositional-generalization configuration"
        );
    }

    core::DeterministicRng random_number_generator(config.seed);
    const core::PhaseVector rotation =
        core::PhaseVector::random(
            config.dimension,
            random_number_generator
        );
    const Transformation rotate{
        .kind = TransformationKind::additive,
        .additive = rotation,
        .permutation = {},
    };
    const Transformation inverse_rotate{
        .kind = TransformationKind::additive,
        .additive = rotation.conjugated(),
        .permutation = {},
    };

    const core::PhaseVector role =
        core::PhaseVector::random(
            config.dimension,
            random_number_generator
        );
    const core::PhaseVector substitution =
        core::PhaseVector::random(
            config.dimension,
            random_number_generator
        );
    const Transformation role_binding{
        .kind = TransformationKind::additive,
        .additive = role,
        .permutation = {},
    };
    const Transformation value_substitution{
        .kind = TransformationKind::additive,
        .additive = substitution,
        .permutation = {},
    };

    const Transformation coordinate_rotation{
        .kind = TransformationKind::permutation,
        .additive = core::PhaseVector::zeros(config.dimension),
        .permutation = cyclic_rotation(config.dimension),
    };
    const Transformation conjugation{
        .kind = TransformationKind::conjugation,
        .additive = core::PhaseVector::zeros(config.dimension),
        .permutation = {},
    };

    CompositionalGeneralizationResult result{
        .seed = config.seed,
        .dimension = config.dimension,
        .training_examples = config.training_examples,
        .evaluation_examples = config.evaluation_examples,
        .cases = {},
        .supported_case_score = 0.0,
        .unsupported_case_score = 0.0,
        .deterministic_run_hash = fnv_offset_basis,
    };
    result.cases.push_back(run_case(
        config,
        "rotation_then_inverse",
        "fixed_phase_delta",
        true,
        rotate,
        inverse_rotate,
        config.seed ^ 0x1111111111111111ULL
    ));
    result.cases.push_back(run_case(
        config,
        "role_binding_then_value_substitution",
        "fixed_phase_delta",
        true,
        role_binding,
        value_substitution,
        config.seed ^ 0x2222222222222222ULL
    ));
    result.cases.push_back(run_case(
        config,
        "coordinate_rotation_then_conjugation",
        "non_additive_stress_case",
        false,
        coordinate_rotation,
        conjugation,
        config.seed ^ 0x3333333333333333ULL
    ));

    double supported_total = 0.0;
    std::size_t supported_count = 0U;
    double unsupported_total = 0.0;
    std::size_t unsupported_count = 0U;
    for (const CompositionCaseResult& case_result : result.cases) {
        if (case_result.expected_supported) {
            supported_total += case_result.rlf.composed_accuracy;
            ++supported_count;
        } else {
            unsupported_total += case_result.rlf.composed_accuracy;
            ++unsupported_count;
        }
        hash_string(result.deterministic_run_hash, case_result.name);
        hash_u64(
            result.deterministic_run_hash,
            case_result.rlf.metrics.deterministic_run_hash
        );
        hash_u64(
            result.deterministic_run_hash,
            case_result.baseline.metrics.deterministic_run_hash
        );
    }
    result.supported_case_score =
        supported_total / static_cast<double>(supported_count);
    result.unsupported_case_score =
        unsupported_total / static_cast<double>(unsupported_count);
    hash_u64(result.deterministic_run_hash, config.seed);
    hash_u64(
        result.deterministic_run_hash,
        static_cast<std::uint64_t>(config.dimension)
    );
    return result;
}

void write_compositional_generalization_json(
    std::ostream& output,
    const CompositionalGeneralizationResult& result
) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "{\n"
           << "  \"experiment\": \"compositional_generalization\",\n"
           << "  \"status\": \"observed\",\n"
           << "  \"seed\": " << result.seed << ",\n"
           << "  \"dimension\": " << result.dimension << ",\n"
           << "  \"training_examples\": "
           << result.training_examples << ",\n"
           << "  \"evaluation_examples\": "
           << result.evaluation_examples << ",\n"
           << "  \"success_similarity\": "
           << success_similarity << ",\n"
           << "  \"supported_case_score\": "
           << result.supported_case_score << ",\n"
           << "  \"unsupported_case_score\": "
           << result.unsupported_case_score << ",\n"
           << "  \"cases\": [\n";
    for (std::size_t case_index = 0U;
         case_index < result.cases.size();
         ++case_index) {
        const CompositionCaseResult& case_result =
            result.cases[case_index];
        output << "    {\n"
               << "      \"name\": \"" << case_result.name << "\",\n"
               << "      \"transformation_class\": \""
               << case_result.transformation_class << "\",\n"
               << "      \"expected_supported\": "
               << (case_result.expected_supported ? "true" : "false")
               << ",\n"
               << "      \"rlf\": ";
        write_system(output, case_result.rlf, 6U);
        output << ",\n      \"baseline\": ";
        write_system(output, case_result.baseline, 6U);
        output << "\n    }";
        if (case_index + 1U != result.cases.size()) {
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
            "failed to write compositional-generalization result"
        );
    }
}

}  // namespace rlf::experiments
