#include "experiments/transformation_task_suite.hpp"

#include "rlf/baselines/nearest_neighbor.hpp"
#include "rlf/core/deterministic_rng.hpp"
#include "rlf/core/encoding.hpp"
#include "rlf/core/phase_vector.hpp"
#include "rlf/core/resonant_fabric.hpp"
#include "rlf/core/resonant_mode.hpp"
#include "rlf/learning/local_learning.hpp"
#include "rlf/learning/structural_learning.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rlf::experiments::detail {
namespace {

constexpr double success_similarity = 0.95;
constexpr std::uint64_t fnv_offset_basis =
    14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;

struct Example final {
    core::PhaseVector input;
    core::PhaseVector target;
};

struct Evaluation final {
    double mean_similarity{};
    double accuracy{};
    double average_cycles{};
    double average_retrieved{};
    double average_activated{};
    double seconds{};
};

struct Training final {
    double mean_prediction_error{};
    double seconds{};
    std::size_t update_operations{};
};

enum class OperationKind {
    additive,
    permutation,
    conjugation,
};

struct Operation final {
    OperationKind kind;
    core::PhaseVector phase_delta;
    std::vector<std::size_t> permutation;
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

[[nodiscard]] double safe_rate(
    const double examples,
    const double seconds
) noexcept {
    return seconds > 0.0
        ? examples / seconds
        : std::numeric_limits<double>::infinity();
}

[[nodiscard]] core::PhaseVector apply_operation(
    const Operation& operation,
    const core::PhaseVector& input
) {
    switch (operation.kind) {
    case OperationKind::additive:
        return input.composed(operation.phase_delta);
    case OperationKind::permutation:
        return input.permuted(operation.permutation);
    case OperationKind::conjugation:
        return input.conjugated();
    }
    throw std::logic_error("unknown transformation task operation");
}

[[nodiscard]] std::vector<Example> make_operation_examples(
    const std::size_t count,
    const std::size_t dimension,
    const Operation& operation,
    core::DeterministicRng& random_number_generator
) {
    std::vector<Example> examples;
    examples.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        core::PhaseVector input = core::PhaseVector::random(
            dimension,
            random_number_generator
        );
        core::PhaseVector target = apply_operation(operation, input);
        examples.push_back({
            .input = std::move(input),
            .target = std::move(target),
        });
    }
    return examples;
}

[[nodiscard]] std::vector<Example> familiar_subset(
    const std::vector<Example>& training,
    const std::size_t count
) {
    std::vector<Example> familiar;
    familiar.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        familiar.push_back(training[index % training.size()]);
    }
    return familiar;
}

[[nodiscard]] core::FabricConfig broad_fabric_config(
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

[[nodiscard]] core::FabricConfig contextual_fabric_config(
    const std::size_t dimension,
    const std::size_t maximum_modes
) {
    core::SettlingConfig settling;
    settling.candidate_count = std::min(
        maximum_modes,
        std::size_t{256U}
    );
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
        .dimension = dimension,
        .maximum_modes = maximum_modes,
        .settling = settling,
        .structural_learning = structural,
    };
}

[[nodiscard]] core::ResonantFabric make_broad_fabric(
    const std::size_t dimension,
    const core::PhaseVector& initial_context
) {
    core::ResonantFabric fabric(broad_fabric_config(dimension));
    fabric.set_update_strategy(
        std::make_unique<learning::WinnerOnlyUpdate>()
    );
    fabric.add_mode(core::ResonantMode(
        1ULL,
        initial_context,
        core::PhaseVector::zeros(dimension),
        0.01F,
        1.0F,
        0.0F,
        0ULL
    ));
    return fabric;
}

[[nodiscard]] core::ResonantFabric make_contextual_fabric(
    const std::size_t dimension,
    const std::size_t maximum_modes
) {
    core::ResonantFabric fabric(
        contextual_fabric_config(dimension, maximum_modes)
    );
    fabric.set_update_strategy(
        std::make_unique<learning::WinnerOnlyUpdate>()
    );
    return fabric;
}

[[nodiscard]] Training train_fabric(
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
        update_operations += update.structural_events.size();
    }
    const auto end = std::chrono::steady_clock::now();
    return {
        .mean_prediction_error =
            prediction_error / static_cast<double>(examples.size()),
        .seconds =
            std::chrono::duration<double>(end - start).count(),
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
    double cycles_total = 0.0;
    double retrieved_total = 0.0;
    double activated_total = 0.0;
    const auto start = std::chrono::steady_clock::now();
    for (const Example& example : examples) {
        const core::SettleResult prediction =
            fabric.settle(example.input, true);
        const double similarity =
            prediction.state.similarity(example.target);
        similarity_total += similarity;
        if (similarity >= success_similarity) {
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
    const auto end = std::chrono::steady_clock::now();
    const double count = static_cast<double>(examples.size());
    return {
        .mean_similarity = similarity_total / count,
        .accuracy = static_cast<double>(successes) / count,
        .average_cycles = cycles_total / count,
        .average_retrieved = retrieved_total / count,
        .average_activated = activated_total / count,
        .seconds =
            std::chrono::duration<double>(end - start).count(),
    };
}

[[nodiscard]] Evaluation evaluate_baseline(
    const baselines::NearestNeighborMemory& baseline,
    const std::vector<Example>& examples
) {
    double similarity_total = 0.0;
    std::size_t successes = 0U;
    const auto start = std::chrono::steady_clock::now();
    for (const Example& example : examples) {
        const auto matches = baseline.retrieve(example.input, 1U);
        const core::PhaseVector prediction = matches.empty()
            ? example.input
            : baseline.records()[matches.front().record_index].value;
        const double similarity =
            prediction.similarity(example.target);
        similarity_total += similarity;
        if (similarity >= success_similarity) {
            ++successes;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const double count = static_cast<double>(examples.size());
    return {
        .mean_similarity = similarity_total / count,
        .accuracy = static_cast<double>(successes) / count,
        .average_cycles = 0.0,
        .average_retrieved = 1.0,
        .average_activated = 1.0,
        .seconds =
            std::chrono::duration<double>(end - start).count(),
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

[[nodiscard]] std::uint64_t fabric_hash(
    const core::ResonantFabric& fabric,
    const std::string_view task_name
) noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_string(hash, task_name);
    for (const core::ResonantMode& mode : fabric.modes()) {
        hash_u64(hash, mode.id);
        hash_u64(hash, mode.enabled ? 1ULL : 0ULL);
        hash_phase_vector(hash, mode.context_key);
        hash_phase_vector(hash, mode.transformation);
    }
    return hash;
}

[[nodiscard]] std::uint64_t baseline_hash(
    const baselines::NearestNeighborMemory& baseline,
    const std::string_view task_name
) noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_string(hash, task_name);
    for (const baselines::NearestNeighborRecord& record :
         baseline.records()) {
        hash_u64(hash, record.id);
        hash_phase_vector(hash, record.key);
        hash_phase_vector(hash, record.value);
    }
    return hash;
}

[[nodiscard]] TransformationTaskResult evaluate_task(
    const TransformationLearningConfig& config,
    std::string name,
    std::string representation,
    const bool expected_unseen_generalization,
    const bool broad_mode,
    std::vector<Example> training,
    std::vector<Example> familiar,
    std::vector<Example> unseen
) {
    core::ResonantFabric fabric = broad_mode
        ? make_broad_fabric(config.dimension, training.front().input)
        : make_contextual_fabric(
              config.dimension,
              std::max(training.size(), std::size_t{1U})
          );
    const Training rlf_training = train_fabric(fabric, training);
    baselines::NearestNeighborMemory baseline(config.dimension);
    const double baseline_training_seconds =
        train_baseline(baseline, training);

    const Evaluation rlf_familiar =
        evaluate_fabric(fabric, familiar);
    const Evaluation rlf_unseen =
        evaluate_fabric(fabric, unseen);
    const Evaluation baseline_familiar =
        evaluate_baseline(baseline, familiar);
    const Evaluation baseline_unseen =
        evaluate_baseline(baseline, unseen);

    const double training_count =
        static_cast<double>(training.size());
    const double evaluation_count =
        static_cast<double>(familiar.size() + unseen.size());
    const std::size_t rlf_bytes = fabric_bytes(fabric);
    const std::size_t baseline_bytes = baseline.bytes_stored();
    const double rlf_inference_seconds =
        rlf_familiar.seconds + rlf_unseen.seconds;
    const double baseline_inference_seconds =
        baseline_familiar.seconds + baseline_unseen.seconds;
    const double average_cycles =
        ((rlf_familiar.average_cycles *
          static_cast<double>(familiar.size())) +
         (rlf_unseen.average_cycles *
          static_cast<double>(unseen.size()))) /
        evaluation_count;
    const double average_retrieved =
        ((rlf_familiar.average_retrieved *
          static_cast<double>(familiar.size())) +
         (rlf_unseen.average_retrieved *
          static_cast<double>(unseen.size()))) /
        evaluation_count;
    const double average_activated =
        ((rlf_familiar.average_activated *
          static_cast<double>(familiar.size())) +
         (rlf_unseen.average_activated *
          static_cast<double>(unseen.size()))) /
        evaluation_count;
    const double rlf_active_operations =
        static_cast<double>(fabric.modes().size()) +
        average_retrieved + average_activated;
    const double baseline_active_operations =
        static_cast<double>(baseline.size()) + 1.0;
    const std::uint64_t rlf_run_hash = fabric_hash(fabric, name);
    const std::uint64_t baseline_run_hash =
        baseline_hash(baseline, name);
    const std::size_t peak_resident = peak_resident_memory_bytes();

    return {
        .name = std::move(name),
        .representation = std::move(representation),
        .expected_unseen_generalization =
            expected_unseen_generalization,
        .rlf = {
            .system = broad_mode
                ? "rlf_single_reusable_phase_delta"
                : "rlf_context_specific_modes",
            .familiar_mean_similarity =
                rlf_familiar.mean_similarity,
            .familiar_accuracy = rlf_familiar.accuracy,
            .unseen_mean_similarity =
                rlf_unseen.mean_similarity,
            .unseen_accuracy = rlf_unseen.accuracy,
            .learned_units = fabric.modes().size(),
            .metrics = {
                .task_accuracy = rlf_unseen.accuracy,
                .one_shot_recall = rlf_familiar.accuracy,
                .retained_accuracy = rlf_unseen.accuracy,
                .catastrophic_forgetting_score = 0.0,
                .compositional_generalization_score =
                    rlf_unseen.accuracy,
                .prediction_error =
                    rlf_training.mean_prediction_error,
                .average_settling_cycles = average_cycles,
                .average_modes_retrieved = average_retrieved,
                .average_modes_activated = average_activated,
                .modes_created = broad_mode
                    ? 1ULL
                    : fabric.structural_statistics().modes_created,
                .modes_split =
                    fabric.structural_statistics().modes_split,
                .modes_merged =
                    fabric.structural_statistics().modes_merged,
                .modes_pruned =
                    fabric.structural_statistics().modes_pruned,
                .bytes_stored = rlf_bytes,
                .peak_resident_bytes = peak_resident,
                .training_seconds = rlf_training.seconds,
                .inference_seconds = rlf_inference_seconds,
                .training_examples_per_second = safe_rate(
                    training_count,
                    rlf_training.seconds
                ),
                .inference_examples_per_second = safe_rate(
                    evaluation_count,
                    rlf_inference_seconds
                ),
                .update_operations_per_example =
                    static_cast<double>(
                        rlf_training.update_operations
                    ) /
                    training_count,
                .active_operations_per_inference =
                    rlf_active_operations,
                .efficiency_score = provisional_efficiency_score(
                    rlf_unseen.accuracy,
                    rlf_bytes,
                    rlf_active_operations,
                    rlf_training.seconds
                ),
                .deterministic_run_hash = rlf_run_hash,
            },
        },
        .baseline = {
            .system = "nearest_neighbor_example_memory",
            .familiar_mean_similarity =
                baseline_familiar.mean_similarity,
            .familiar_accuracy = baseline_familiar.accuracy,
            .unseen_mean_similarity =
                baseline_unseen.mean_similarity,
            .unseen_accuracy = baseline_unseen.accuracy,
            .learned_units = baseline.size(),
            .metrics = {
                .task_accuracy = baseline_unseen.accuracy,
                .one_shot_recall = baseline_familiar.accuracy,
                .retained_accuracy = baseline_unseen.accuracy,
                .catastrophic_forgetting_score = 0.0,
                .compositional_generalization_score =
                    baseline_unseen.accuracy,
                .prediction_error = 0.0,
                .average_settling_cycles = 0.0,
                .average_modes_retrieved = 1.0,
                .average_modes_activated = 1.0,
                .modes_created = 0ULL,
                .modes_split = 0ULL,
                .modes_merged = 0ULL,
                .modes_pruned = 0ULL,
                .bytes_stored = baseline_bytes,
                .peak_resident_bytes = peak_resident,
                .training_seconds = baseline_training_seconds,
                .inference_seconds = baseline_inference_seconds,
                .training_examples_per_second = safe_rate(
                    training_count,
                    baseline_training_seconds
                ),
                .inference_examples_per_second = safe_rate(
                    evaluation_count,
                    baseline_inference_seconds
                ),
                .update_operations_per_example = 1.0,
                .active_operations_per_inference =
                    baseline_active_operations,
                .efficiency_score = provisional_efficiency_score(
                    baseline_unseen.accuracy,
                    baseline_bytes,
                    baseline_active_operations,
                    baseline_training_seconds
                ),
                .deterministic_run_hash = baseline_run_hash,
            },
        },
    };
}

[[nodiscard]] std::vector<std::size_t> reversal_permutation(
    const std::size_t dimension
) {
    std::vector<std::size_t> permutation(dimension);
    for (std::size_t output_index = 0U;
         output_index < dimension;
         ++output_index) {
        permutation[output_index] = dimension - 1U - output_index;
    }
    return permutation;
}

[[nodiscard]] std::vector<std::size_t> rotation_permutation(
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

[[nodiscard]] std::vector<Example> make_role_rebinding_examples(
    const std::size_t count,
    const std::size_t dimension,
    const core::PhaseVector& source_role,
    const core::PhaseVector& target_role,
    core::DeterministicRng& random_number_generator
) {
    std::vector<Example> examples;
    examples.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        const core::PhaseVector value = core::PhaseVector::random(
            dimension,
            random_number_generator
        );
        examples.push_back({
            .input = core::bind(source_role, value),
            .target = core::bind(target_role, value),
        });
    }
    return examples;
}

[[nodiscard]] std::vector<std::string> symbol_names(
    const std::size_t count
) {
    std::vector<std::string> names;
    names.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        names.push_back("symbol_" + std::to_string(index));
    }
    return names;
}

[[nodiscard]] std::vector<Example> repeat_examples(
    const std::vector<Example>& prototypes,
    const std::size_t count
) {
    std::vector<Example> examples;
    examples.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        examples.push_back(prototypes[index % prototypes.size()]);
    }
    return examples;
}

[[nodiscard]] TransformationTaskResult make_symbol_substitution_task(
    const TransformationLearningConfig& config
) {
    constexpr std::size_t symbol_count = 16U;
    const core::SymbolEncoder encoder(
        config.dimension,
        config.seed ^ 0x510B5717ULL,
        symbol_names(symbol_count)
    );
    std::vector<Example> training_prototypes;
    std::vector<Example> unseen_prototypes;
    for (std::size_t source = 0U; source < 12U; source += 2U) {
        training_prototypes.push_back({
            .input = encoder.encode(encoder.symbols()[source]),
            .target = encoder.encode(encoder.symbols()[source + 1U]),
        });
    }
    for (std::size_t source = 12U;
         source < symbol_count;
         source += 2U) {
        unseen_prototypes.push_back({
            .input = encoder.encode(encoder.symbols()[source]),
            .target = encoder.encode(encoder.symbols()[source + 1U]),
        });
    }
    return evaluate_task(
        config,
        "symbol_substitution",
        "random_symbol_codebook_context_modes",
        false,
        false,
        repeat_examples(training_prototypes, config.training_examples),
        repeat_examples(
            training_prototypes,
            config.evaluation_examples
        ),
        repeat_examples(
            unseen_prototypes,
            config.evaluation_examples
        )
    );
}

[[nodiscard]] TransformationTaskResult make_arithmetic_task(
    const TransformationLearningConfig& config
) {
    constexpr std::int64_t minimum = 0;
    constexpr std::int64_t maximum = 15;
    const core::IntegerEncoder encoder(
        config.dimension,
        config.seed ^ 0xA8174E71CULL,
        minimum,
        maximum
    );
    std::vector<Example> training_prototypes;
    std::vector<Example> unseen_prototypes;
    for (std::int64_t value = minimum; value < 12; ++value) {
        training_prototypes.push_back({
            .input = encoder.encode(value),
            .target = encoder.encode(value + 1),
        });
    }
    for (std::int64_t value = 12; value < maximum; ++value) {
        unseen_prototypes.push_back({
            .input = encoder.encode(value),
            .target = encoder.encode(value + 1),
        });
    }
    return evaluate_task(
        config,
        "bounded_arithmetic_increment",
        "random_bounded_integer_codebook_context_modes",
        false,
        false,
        repeat_examples(training_prototypes, config.training_examples),
        repeat_examples(
            training_prototypes,
            config.evaluation_examples
        ),
        repeat_examples(
            unseen_prototypes,
            config.evaluation_examples
        )
    );
}

}  // namespace

TransformationTaskSuite run_transformation_task_suite(
    const TransformationLearningConfig& config
) {
    core::DeterministicRng random_number_generator(
        config.seed ^ 0x745A5B17EULL
    );
    std::vector<TransformationTaskResult> tasks;
    tasks.reserve(7U);

    const Operation substitution{
        .kind = OperationKind::additive,
        .phase_delta = core::PhaseVector::random(
            config.dimension,
            random_number_generator
        ),
        .permutation = {},
    };
    std::vector<Example> substitution_training =
        make_operation_examples(
            config.training_examples,
            config.dimension,
            substitution,
            random_number_generator
        );
    tasks.push_back(evaluate_task(
        config,
        "distributed_phase_substitution",
        "single_global_phase_delta",
        true,
        true,
        substitution_training,
        familiar_subset(
            substitution_training,
            config.evaluation_examples
        ),
        make_operation_examples(
            config.evaluation_examples,
            config.dimension,
            substitution,
            random_number_generator
        )
    ));

    const core::PermutationFamily permutation_family(
        config.dimension,
        config.seed ^ 0x9E2A2D1FULL,
        1U
    );
    const Operation permutation{
        .kind = OperationKind::permutation,
        .phase_delta = core::PhaseVector::zeros(config.dimension),
        .permutation = std::vector<std::size_t>(
            permutation_family.permutation(0U).begin(),
            permutation_family.permutation(0U).end()
        ),
    };
    std::vector<Example> permutation_training =
        make_operation_examples(
            config.training_examples,
            config.dimension,
            permutation,
            random_number_generator
        );
    tasks.push_back(evaluate_task(
        config,
        "coordinate_permutation",
        "context_specific_phase_differences",
        false,
        false,
        permutation_training,
        familiar_subset(
            permutation_training,
            config.evaluation_examples
        ),
        make_operation_examples(
            config.evaluation_examples,
            config.dimension,
            permutation,
            random_number_generator
        )
    ));

    const Operation reversal{
        .kind = OperationKind::permutation,
        .phase_delta = core::PhaseVector::zeros(config.dimension),
        .permutation = reversal_permutation(config.dimension),
    };
    std::vector<Example> reversal_training =
        make_operation_examples(
            config.training_examples,
            config.dimension,
            reversal,
            random_number_generator
        );
    tasks.push_back(evaluate_task(
        config,
        "coordinate_reversal",
        "context_specific_phase_differences",
        false,
        false,
        reversal_training,
        familiar_subset(
            reversal_training,
            config.evaluation_examples
        ),
        make_operation_examples(
            config.evaluation_examples,
            config.dimension,
            reversal,
            random_number_generator
        )
    ));

    const Operation rotation{
        .kind = OperationKind::permutation,
        .phase_delta = core::PhaseVector::zeros(config.dimension),
        .permutation = rotation_permutation(config.dimension),
    };
    std::vector<Example> rotation_training =
        make_operation_examples(
            config.training_examples,
            config.dimension,
            rotation,
            random_number_generator
        );
    tasks.push_back(evaluate_task(
        config,
        "coordinate_rotation",
        "context_specific_phase_differences",
        false,
        false,
        rotation_training,
        familiar_subset(
            rotation_training,
            config.evaluation_examples
        ),
        make_operation_examples(
            config.evaluation_examples,
            config.dimension,
            rotation,
            random_number_generator
        )
    ));

    tasks.push_back(make_symbol_substitution_task(config));
    tasks.push_back(make_arithmetic_task(config));

    const core::PhaseVector source_role =
        core::PhaseVector::random(
            config.dimension,
            random_number_generator
        );
    const core::PhaseVector target_role =
        core::PhaseVector::random(
            config.dimension,
            random_number_generator
        );
    std::vector<Example> rebinding_training =
        make_role_rebinding_examples(
            config.training_examples,
            config.dimension,
            source_role,
            target_role,
            random_number_generator
        );
    tasks.push_back(evaluate_task(
        config,
        "role_value_rebinding",
        "role_composition_with_fixed_role_delta",
        true,
        true,
        rebinding_training,
        familiar_subset(
            rebinding_training,
            config.evaluation_examples
        ),
        make_role_rebinding_examples(
            config.evaluation_examples,
            config.dimension,
            source_role,
            target_role,
            random_number_generator
        )
    ));

    double supported_total = 0.0;
    std::size_t supported_count = 0U;
    double unsupported_total = 0.0;
    std::size_t unsupported_count = 0U;
    std::uint64_t run_hash = fnv_offset_basis;
    for (const TransformationTaskResult& task : tasks) {
        if (task.expected_unseen_generalization) {
            supported_total += task.rlf.unseen_accuracy;
            ++supported_count;
        } else {
            unsupported_total += task.rlf.unseen_accuracy;
            ++unsupported_count;
        }
        hash_string(run_hash, task.name);
        hash_u64(
            run_hash,
            task.rlf.metrics.deterministic_run_hash
        );
        hash_u64(
            run_hash,
            task.baseline.metrics.deterministic_run_hash
        );
        hash_u64(
            run_hash,
            std::bit_cast<std::uint64_t>(task.rlf.unseen_accuracy)
        );
    }
    hash_u64(run_hash, config.seed);
    return {
        .tasks = std::move(tasks),
        .supported_unseen_accuracy =
            supported_total / static_cast<double>(supported_count),
        .unsupported_unseen_accuracy =
            unsupported_total /
            static_cast<double>(unsupported_count),
        .deterministic_run_hash = run_hash,
    };
}

}  // namespace rlf::experiments::detail
