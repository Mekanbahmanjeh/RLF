#include "rlf/experiments/operator_composition.hpp"

#include "rlf/baselines/nearest_neighbor.hpp"
#include "rlf/core/deterministic_rng.hpp"
#include "rlf/core/operator_fabric.hpp"
#include "rlf/core/transformation_operator.hpp"
#include "rlf/experiments/phase_vector_smoke.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <map>
#include <numbers>
#include <ostream>
#include <set>
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

struct Task final {
    std::string name;
    core::PhaseVector context;
    core::TransformationOperator truth;
    core::TransformationOperator phase_baseline;
    core::TransformationOperator supervised_baseline;
    std::vector<core::OperatorTrainingExample> training;
    baselines::NearestNeighborMemory nearest_neighbor;
};

struct Evaluation final {
    double operator_extension{};
    double phase_offset{};
    double nearest_neighbor{};
    double supervised{};
    double oracle{};
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
    const std::string& value
) noexcept {
    for (const char character : value) {
        hash ^= static_cast<std::uint8_t>(character);
        hash *= fnv_prime;
    }
}

[[nodiscard]] std::vector<std::size_t> payload_rotation(
    const std::size_t dimension,
    const std::size_t context_dimensions,
    const std::size_t amount
) {
    std::vector<std::size_t> permutation(dimension);
    for (std::size_t index = 0U; index < dimension; ++index) {
        permutation[index] = index;
    }
    const std::size_t payload_size = dimension - context_dimensions;
    for (std::size_t index = context_dimensions;
         index < dimension;
         ++index) {
        permutation[index] = context_dimensions +
            ((index - context_dimensions + amount) % payload_size);
    }
    return permutation;
}

[[nodiscard]] core::PhaseVector payload_shift(
    const std::size_t dimension,
    const std::size_t context_dimensions,
    core::DeterministicRng& rng
) {
    std::vector<float> angles(dimension, 0.0F);
    for (std::size_t index = context_dimensions;
         index < dimension;
         ++index) {
        angles[index] = rng.uniform_angle();
    }
    return core::PhaseVector(std::move(angles));
}

[[nodiscard]] core::PhaseVector with_context(
    const core::PhaseVector& payload,
    const core::PhaseVector& context,
    const std::size_t context_dimensions
) {
    std::vector<float> angles(
        payload.angles().begin(),
        payload.angles().end()
    );
    for (std::size_t index = 0U;
         index < context_dimensions;
         ++index) {
        angles[index] = context[index];
    }
    return core::PhaseVector(std::move(angles));
}

[[nodiscard]] core::PhaseVector noisy(
    const core::PhaseVector& input,
    const double radians,
    core::DeterministicRng& rng
) {
    std::vector<float> angles(
        input.angles().begin(),
        input.angles().end()
    );
    for (float& angle : angles) {
        angle = core::PhaseVector::normalize_angle(
            angle + static_cast<float>(
                ((rng.uniform_unit() * 2.0) - 1.0) * radians
            )
        );
    }
    return core::PhaseVector(std::move(angles));
}

[[nodiscard]] core::PhaseVector nearest_predict(
    baselines::NearestNeighborMemory& memory,
    const core::PhaseVector& input
) {
    const std::vector<baselines::NearestNeighborMatch> matches =
        memory.retrieve(input, 1U);
    if (matches.empty()) {
        throw std::runtime_error(
            "nearest-neighbor operator baseline is empty"
        );
    }
    return memory.records()[matches.front().record_index].value;
}

[[nodiscard]] Task make_task(
    std::string name,
    core::PhaseVector context,
    core::TransformationOperator truth,
    const OperatorCompositionConfig& config,
    core::DeterministicRng& rng
) {
    std::vector<core::OperatorTrainingExample> training;
    training.reserve(config.training_examples);
    baselines::NearestNeighborMemory nearest_neighbor(config.dimension);
    for (std::size_t example_index = 0U;
         example_index < config.training_examples;
         ++example_index) {
        const core::PhaseVector random =
            core::PhaseVector::random(config.dimension, rng);
        core::PhaseVector input = with_context(
            random,
            context,
            config.context_dimensions
        );
        core::PhaseVector target = truth.apply(input);
        static_cast<void>(nearest_neighbor.insert(input, target));
        training.push_back({
            .input = std::move(input),
            .target = std::move(target),
        });
    }
    return {
        .name = std::move(name),
        .context = std::move(context),
        .truth = std::move(truth),
        .phase_baseline = core::fit_operator(
            core::OperatorFamily::phase_shift,
            training,
            config.context_dimensions
        ),
        .supervised_baseline = core::fit_operator(
            core::OperatorFamily::explicit_sequence,
            training,
            config.context_dimensions
        ),
        .training = std::move(training),
        .nearest_neighbor = std::move(nearest_neighbor),
    };
}

[[nodiscard]] double success(
    const core::PhaseVector& prediction,
    const core::PhaseVector& target
) {
    return prediction.similarity(target) >= success_similarity
        ? 1.0
        : 0.0;
}

[[nodiscard]] Evaluation evaluate_single_task(
    core::OperatorFabric& fabric,
    Task& task,
    const OperatorCompositionConfig& config,
    core::DeterministicRng& rng,
    const bool add_noise
) {
    Evaluation totals;
    for (std::size_t example_index = 0U;
         example_index < config.evaluation_examples;
         ++example_index) {
        const core::PhaseVector clean_input = with_context(
            core::PhaseVector::random(config.dimension, rng),
            task.context,
            config.context_dimensions
        );
        const core::PhaseVector inference_input = add_noise
            ? noisy(clean_input, config.noise_radians, rng)
            : clean_input;
        const core::PhaseVector target = task.truth.apply(clean_input);
        totals.operator_extension += success(
            fabric.predict(inference_input).state,
            target
        );
        totals.phase_offset += success(
            task.phase_baseline.apply(inference_input),
            target
        );
        totals.nearest_neighbor += success(
            nearest_predict(task.nearest_neighbor, inference_input),
            target
        );
        totals.supervised += success(
            task.supervised_baseline.apply(inference_input),
            target
        );
        totals.oracle += success(
            task.truth.apply(inference_input),
            target
        );
    }
    const double count =
        static_cast<double>(config.evaluation_examples);
    totals.operator_extension /= count;
    totals.phase_offset /= count;
    totals.nearest_neighbor /= count;
    totals.supervised /= count;
    totals.oracle /= count;
    return totals;
}

[[nodiscard]] core::PhaseVector apply_operator_chain(
    core::OperatorFabric& fabric,
    const std::vector<Task*>& tasks,
    core::PhaseVector state,
    const std::size_t context_dimensions
) {
    for (const Task* task : tasks) {
        state = with_context(
            state,
            task->context,
            context_dimensions
        );
        state = fabric.predict(state).state;
    }
    return state;
}

[[nodiscard]] core::PhaseVector apply_truth_chain(
    const std::vector<Task*>& tasks,
    core::PhaseVector state,
    const std::size_t context_dimensions
) {
    for (const Task* task : tasks) {
        state = with_context(
            state,
            task->context,
            context_dimensions
        );
        state = task->truth.apply(state);
    }
    return state;
}

[[nodiscard]] core::PhaseVector apply_phase_chain(
    const std::vector<Task*>& tasks,
    core::PhaseVector state,
    const std::size_t context_dimensions
) {
    for (const Task* task : tasks) {
        state = with_context(
            state,
            task->context,
            context_dimensions
        );
        state = task->phase_baseline.apply(state);
    }
    return state;
}

[[nodiscard]] core::PhaseVector apply_supervised_chain(
    const std::vector<Task*>& tasks,
    core::PhaseVector state,
    const std::size_t context_dimensions
) {
    for (const Task* task : tasks) {
        state = with_context(
            state,
            task->context,
            context_dimensions
        );
        state = task->supervised_baseline.apply(state);
    }
    return state;
}

[[nodiscard]] core::PhaseVector apply_nearest_chain(
    const std::vector<Task*>& tasks,
    core::PhaseVector state,
    const std::size_t context_dimensions
) {
    for (Task* task : tasks) {
        state = with_context(
            state,
            task->context,
            context_dimensions
        );
        state = nearest_predict(task->nearest_neighbor, state);
    }
    return state;
}

[[nodiscard]] Evaluation evaluate_chain(
    core::OperatorFabric& fabric,
    const std::vector<Task*>& tasks,
    const OperatorCompositionConfig& config,
    core::DeterministicRng& rng
) {
    Evaluation totals;
    for (std::size_t example_index = 0U;
         example_index < config.evaluation_examples;
         ++example_index) {
        const core::PhaseVector input =
            core::PhaseVector::random(config.dimension, rng);
        const core::PhaseVector target = apply_truth_chain(
            tasks,
            input,
            config.context_dimensions
        );
        totals.operator_extension += success(
            apply_operator_chain(
                fabric,
                tasks,
                input,
                config.context_dimensions
            ),
            target
        );
        totals.phase_offset += success(
            apply_phase_chain(
                tasks,
                input,
                config.context_dimensions
            ),
            target
        );
        totals.nearest_neighbor += success(
            apply_nearest_chain(
                tasks,
                input,
                config.context_dimensions
            ),
            target
        );
        totals.supervised += success(
            apply_supervised_chain(
                tasks,
                input,
                config.context_dimensions
            ),
            target
        );
        totals.oracle += 1.0;
    }
    const double count =
        static_cast<double>(config.evaluation_examples);
    totals.operator_extension /= count;
    totals.phase_offset /= count;
    totals.nearest_neighbor /= count;
    totals.supervised /= count;
    totals.oracle /= count;
    return totals;
}

[[nodiscard]] OperatorCompositionCaseResult make_case(
    std::string name,
    std::string category,
    const Evaluation& evaluation
) {
    return {
        .name = std::move(name),
        .category = std::move(category),
        .operator_extension_accuracy =
            evaluation.operator_extension,
        .phase_offset_accuracy = evaluation.phase_offset,
        .nearest_neighbor_accuracy =
            evaluation.nearest_neighbor,
        .supervised_operator_accuracy =
            evaluation.supervised,
        .oracle_accuracy = evaluation.oracle,
    };
}

[[nodiscard]] double category_mean(
    const std::vector<OperatorCompositionCaseResult>& cases,
    const std::string& category,
    const double OperatorCompositionCaseResult::*member
) {
    double total = 0.0;
    std::size_t count = 0U;
    for (const OperatorCompositionCaseResult& item : cases) {
        if (item.category == category) {
            total += item.*member;
            ++count;
        }
    }
    return count == 0U ? 0.0 : total / static_cast<double>(count);
}

[[nodiscard]] std::string decide(
    const double unseen_accuracy,
    const double supervised_accuracy,
    const double noisy_accuracy,
    const double conflicting_accuracy
) {
    if (unseen_accuracy >= 0.9 &&
        supervised_accuracy >= 0.9 &&
        noisy_accuracy >= 0.9 &&
        conflicting_accuracy >= 0.9) {
        return "A";
    }
    if (unseen_accuracy >= 0.75 &&
        std::abs(unseen_accuracy - supervised_accuracy) <= 0.05) {
        return "B";
    }
    return "C";
}

}  // namespace

OperatorCompositionResult run_operator_composition(
    const OperatorCompositionConfig& config
) {
    if (config.dimension < 8U ||
        config.context_dimensions == 0U ||
        config.context_dimensions >= config.dimension ||
        config.training_examples < 4U ||
        config.evaluation_examples == 0U ||
        !std::isfinite(config.noise_radians) ||
        config.noise_radians < 0.0 ||
        config.noise_radians > std::numbers::pi_v<double>) {
        throw std::invalid_argument(
            "invalid operator composition configuration"
        );
    }

    core::DeterministicRng training_rng(config.seed);
    core::DeterministicRng evaluation_rng(
        config.seed ^ 0x9E3779B97F4A7C15ULL
    );
    const core::PhaseVector shift_a = payload_shift(
        config.dimension,
        config.context_dimensions,
        training_rng
    );
    const core::PhaseVector shift_b = payload_shift(
        config.dimension,
        config.context_dimensions,
        training_rng
    );
    const core::PhaseVector context_shift =
        core::PhaseVector::random(config.dimension, training_rng);
    const core::PhaseVector context_permutation =
        core::PhaseVector::random(config.dimension, training_rng);
    const core::PhaseVector context_conjugation =
        core::PhaseVector::random(config.dimension, training_rng);
    const core::PhaseVector context_permutation_shift =
        core::PhaseVector::random(config.dimension, training_rng);
    const core::PhaseVector context_conjugation_shift =
        core::PhaseVector::random(config.dimension, training_rng);
    const core::PhaseVector context_explicit_sequence =
        core::PhaseVector::random(config.dimension, training_rng);

    std::vector<Task> tasks;
    tasks.reserve(6U);
    tasks.push_back(make_task(
        "phase_shift",
        context_shift,
        core::TransformationOperator(
            config.dimension,
            {core::OperatorPrimitive::shift(
                shift_a,
                config.context_dimensions
            )}
        ),
        config,
        training_rng
    ));
    tasks.push_back(make_task(
        "coordinate_permutation",
        context_permutation,
        core::TransformationOperator(
            config.dimension,
            {core::OperatorPrimitive::permute(
                payload_rotation(
                    config.dimension,
                    config.context_dimensions,
                    3U
                ),
                config.context_dimensions
            )}
        ),
        config,
        training_rng
    ));
    tasks.push_back(make_task(
        "conjugation",
        context_conjugation,
        core::TransformationOperator(
            config.dimension,
            {core::OperatorPrimitive::conjugate(
                config.dimension,
                config.context_dimensions
            )}
        ),
        config,
        training_rng
    ));
    tasks.push_back(make_task(
        "permutation_then_phase_shift",
        context_permutation_shift,
        core::TransformationOperator(
            config.dimension,
            {
                core::OperatorPrimitive::permute(
                    payload_rotation(
                        config.dimension,
                        config.context_dimensions,
                        5U
                    ),
                    config.context_dimensions
                ),
                core::OperatorPrimitive::shift(
                    shift_b,
                    config.context_dimensions
                ),
            }
        ),
        config,
        training_rng
    ));
    tasks.push_back(make_task(
        "conjugation_then_phase_shift",
        context_conjugation_shift,
        core::TransformationOperator(
            config.dimension,
            {
                core::OperatorPrimitive::conjugate(
                    config.dimension,
                    config.context_dimensions
                ),
                core::OperatorPrimitive::shift(
                    shift_b,
                    config.context_dimensions
                ),
            }
        ),
        config,
        training_rng
    ));
    tasks.push_back(make_task(
        "explicit_conjugate_permute_shift_sequence",
        context_explicit_sequence,
        core::TransformationOperator(
            config.dimension,
            {
                core::OperatorPrimitive::conjugate(
                    config.dimension,
                    config.context_dimensions
                ),
                core::OperatorPrimitive::permute(
                    payload_rotation(
                        config.dimension,
                        config.context_dimensions,
                        7U
                    ),
                    config.context_dimensions
                ),
                core::OperatorPrimitive::shift(
                    shift_a,
                    config.context_dimensions
                ),
            }
        ),
        config,
        training_rng
    ));

    core::OperatorFabric fabric({
        .dimension = config.dimension,
        .context_dimensions = config.context_dimensions,
        .history_capacity = config.training_examples,
        .candidate_count = 16U,
        .creation_resonance_threshold = 0.8,
        .context_learning_rate = 0.05,
        .complexity_penalty = 0.002,
    });
    const auto training_start = std::chrono::steady_clock::now();
    for (const Task& task : tasks) {
        for (const core::OperatorTrainingExample& example :
             task.training) {
            fabric.learn(example.input, example.target);
        }
    }
    const auto training_end = std::chrono::steady_clock::now();

    std::vector<OperatorCompositionCaseResult> cases;
    const auto inference_start = std::chrono::steady_clock::now();
    for (Task& task : tasks) {
        cases.push_back(make_case(
            task.name,
            "familiar_operator_held_out_states",
            evaluate_single_task(
                fabric,
                task,
                config,
                evaluation_rng,
                false
            )
        ));
    }
    cases.push_back(make_case(
        "permutation_then_conjugation_shift",
        "unseen_composition",
        evaluate_chain(
            fabric,
            {&tasks[1U], &tasks[4U]},
            config,
            evaluation_rng
        )
    ));
    cases.push_back(make_case(
        "shift_then_permutation",
        "unseen_composition",
        evaluate_chain(
            fabric,
            {&tasks[0U], &tasks[1U]},
            config,
            evaluation_rng
        )
    ));
    cases.push_back(make_case(
        "compound_then_conjugation",
        "unseen_composition",
        evaluate_chain(
            fabric,
            {&tasks[3U], &tasks[2U]},
            config,
            evaluation_rng
        )
    ));
    cases.push_back(make_case(
        "explicit_sequence_then_shift",
        "unseen_composition",
        evaluate_chain(
            fabric,
            {&tasks[5U], &tasks[0U]},
            config,
            evaluation_rng
        )
    ));
    cases.push_back(make_case(
        "repeated_permutation_three_times",
        "unseen_composition",
        evaluate_chain(
            fabric,
            {&tasks[1U], &tasks[1U], &tasks[1U]},
            config,
            evaluation_rng
        )
    ));
    cases.push_back(make_case(
        "repeated_conjugation_identity",
        "unseen_composition",
        evaluate_chain(
            fabric,
            {&tasks[2U], &tasks[2U]},
            config,
            evaluation_rng
        )
    ));
    const Evaluation noisy_evaluation = evaluate_single_task(
        fabric,
        tasks[3U],
        config,
        evaluation_rng,
        true
    );
    cases.push_back(make_case(
        "noisy_permutation_shift",
        "noisy_input",
        noisy_evaluation
    ));

    core::OperatorFabric ambiguous_fabric({
        .dimension = config.dimension,
        .context_dimensions = config.context_dimensions,
        .history_capacity = config.training_examples * 2U,
        .candidate_count = 16U,
        .creation_resonance_threshold = 0.8,
        .context_learning_rate = 0.0,
        .complexity_penalty = 0.002,
    });
    for (std::size_t index = 0U;
         index < config.training_examples;
         ++index) {
        ambiguous_fabric.learn(
            tasks[0U].training[index].input,
            tasks[0U].training[index].target
        );
        ambiguous_fabric.learn(
            with_context(
                tasks[1U].training[index].input,
                tasks[0U].context,
                config.context_dimensions
            ),
            tasks[1U].truth.apply(with_context(
                tasks[1U].training[index].input,
                tasks[0U].context,
                config.context_dimensions
            ))
        );
    }
    double ambiguous_accuracy = 0.0;
    for (std::size_t index = 0U;
         index < config.evaluation_examples;
         ++index) {
        const core::PhaseVector input = with_context(
            core::PhaseVector::random(config.dimension, evaluation_rng),
            tasks[0U].context,
            config.context_dimensions
        );
        ambiguous_accuracy += success(
            ambiguous_fabric.predict(input).state,
            tasks[0U].truth.apply(input)
        );
    }
    ambiguous_accuracy /=
        static_cast<double>(config.evaluation_examples);
    cases.push_back({
        .name = "identical_context_conflict",
        .category = "ambiguous_context",
        .operator_extension_accuracy = ambiguous_accuracy,
        .phase_offset_accuracy = 0.0,
        .nearest_neighbor_accuracy = 0.0,
        .supervised_operator_accuracy = 0.0,
        .oracle_accuracy = 1.0,
    });

    core::OperatorFabric conflict_fabric({
        .dimension = config.dimension,
        .context_dimensions = config.context_dimensions,
        .history_capacity = config.training_examples,
        .candidate_count = 16U,
        .creation_resonance_threshold = 0.999999,
        .context_learning_rate = 0.0,
        .complexity_penalty = 0.002,
    });
    core::PhaseVector nearby_context = noisy(
        tasks[0U].context,
        0.03,
        evaluation_rng
    );
    for (std::size_t index = 0U;
         index < config.training_examples;
         ++index) {
        conflict_fabric.learn(
            tasks[0U].training[index].input,
            tasks[0U].training[index].target
        );
        const core::PhaseVector input = with_context(
            tasks[1U].training[index].input,
            nearby_context,
            config.context_dimensions
        );
        conflict_fabric.learn(input, tasks[1U].truth.apply(input));
    }
    double conflict_accuracy = 0.0;
    for (std::size_t index = 0U;
         index < config.evaluation_examples;
         ++index) {
        const core::PhaseVector input = with_context(
            core::PhaseVector::random(config.dimension, evaluation_rng),
            tasks[0U].context,
            config.context_dimensions
        );
        conflict_accuracy += success(
            conflict_fabric.predict(input).state,
            tasks[0U].truth.apply(input)
        );
    }
    conflict_accuracy /=
        static_cast<double>(config.evaluation_examples);
    cases.push_back({
        .name = "nearby_context_competing_modes",
        .category = "conflicting_modes",
        .operator_extension_accuracy = conflict_accuracy,
        .phase_offset_accuracy = 0.0,
        .nearest_neighbor_accuracy = 0.0,
        .supervised_operator_accuracy = 0.0,
        .oracle_accuracy = 1.0,
    });
    const auto inference_end = std::chrono::steady_clock::now();

    const double operator_familiar = category_mean(
        cases,
        "familiar_operator_held_out_states",
        &OperatorCompositionCaseResult::operator_extension_accuracy
    );
    const double operator_unseen = category_mean(
        cases,
        "unseen_composition",
        &OperatorCompositionCaseResult::operator_extension_accuracy
    );
    const double supervised_unseen = category_mean(
        cases,
        "unseen_composition",
        &OperatorCompositionCaseResult::supervised_operator_accuracy
    );
    const double phase_familiar = category_mean(
        cases,
        "familiar_operator_held_out_states",
        &OperatorCompositionCaseResult::phase_offset_accuracy
    );
    const double phase_unseen = category_mean(
        cases,
        "unseen_composition",
        &OperatorCompositionCaseResult::phase_offset_accuracy
    );
    const double nearest_familiar = category_mean(
        cases,
        "familiar_operator_held_out_states",
        &OperatorCompositionCaseResult::nearest_neighbor_accuracy
    );
    const double nearest_unseen = category_mean(
        cases,
        "unseen_composition",
        &OperatorCompositionCaseResult::nearest_neighbor_accuracy
    );
    const double supervised_familiar = category_mean(
        cases,
        "familiar_operator_held_out_states",
        &OperatorCompositionCaseResult::supervised_operator_accuracy
    );
    const double oracle_unseen = category_mean(
        cases,
        "unseen_composition",
        &OperatorCompositionCaseResult::oracle_accuracy
    );
    std::set<core::OperatorFamily> selected_families;
    for (Task& task : tasks) {
        const core::PhaseVector input = with_context(
            core::PhaseVector::random(config.dimension, evaluation_rng),
            task.context,
            config.context_dimensions
        );
        selected_families.insert(fabric.predict(input).family);
    }

    std::size_t operator_bytes = sizeof(fabric);
    for (const core::OperatorMode& mode : fabric.modes()) {
        operator_bytes += sizeof(mode);
        operator_bytes += mode.context_key.size() * sizeof(float);
        for (const core::OperatorPrimitive& primitive :
             mode.transformation.primitives()) {
            operator_bytes += sizeof(primitive);
            operator_bytes +=
                primitive.phase_shift.size() * sizeof(float);
            operator_bytes +=
                primitive.permutation.size() * sizeof(std::size_t);
        }
        for (const core::OperatorTrainingExample& example :
             mode.recent_examples) {
            operator_bytes +=
                (example.input.size() + example.target.size()) *
                sizeof(float);
        }
    }

    std::uint64_t run_hash = fnv_offset_basis;
    hash_u64(run_hash, config.seed);
    for (const OperatorCompositionCaseResult& item : cases) {
        hash_string(run_hash, item.name);
        hash_u64(
            run_hash,
            std::bit_cast<std::uint64_t>(
                item.operator_extension_accuracy
            )
        );
        hash_u64(
            run_hash,
            std::bit_cast<std::uint64_t>(
                item.phase_offset_accuracy
            )
        );
    }

    return {
        .seed = config.seed,
        .dimension = config.dimension,
        .context_dimensions = config.context_dimensions,
        .training_examples = config.training_examples,
        .evaluation_examples = config.evaluation_examples,
        .cases = std::move(cases),
        .operator_familiar_accuracy = operator_familiar,
        .operator_unseen_composition_accuracy = operator_unseen,
        .phase_offset_familiar_accuracy = phase_familiar,
        .phase_offset_unseen_composition_accuracy = phase_unseen,
        .nearest_neighbor_familiar_accuracy = nearest_familiar,
        .nearest_neighbor_unseen_composition_accuracy = nearest_unseen,
        .supervised_familiar_accuracy = supervised_familiar,
        .supervised_unseen_composition_accuracy = supervised_unseen,
        .oracle_unseen_composition_accuracy = oracle_unseen,
        .noisy_input_accuracy = noisy_evaluation.operator_extension,
        .ambiguous_context_accuracy = ambiguous_accuracy,
        .conflicting_mode_accuracy = conflict_accuracy,
        .operator_mode_count = fabric.modes().size(),
        .operator_selected_family_count = selected_families.size(),
        .operator_bytes = operator_bytes,
        .training_seconds = std::chrono::duration<double>(
            training_end - training_start
        ).count(),
        .inference_seconds = std::chrono::duration<double>(
            inference_end - inference_start
        ).count(),
        .deterministic_run_hash = run_hash,
        .scientific_decision = decide(
            operator_unseen,
            supervised_unseen,
            noisy_evaluation.operator_extension,
            conflict_accuracy
        ),
    };
}

void write_operator_composition_json(
    std::ostream& output,
    const OperatorCompositionResult& result
) {
    output << std::setprecision(17);
    output
        << "{\n"
        << "  \"experiment\": \"operator_composition\",\n"
        << "  \"seed\": " << result.seed << ",\n"
        << "  \"dimension\": " << result.dimension << ",\n"
        << "  \"context_dimensions\": "
        << result.context_dimensions << ",\n"
        << "  \"training_examples\": "
        << result.training_examples << ",\n"
        << "  \"evaluation_examples\": "
        << result.evaluation_examples << ",\n"
        << "  \"operator_familiar_accuracy\": "
        << result.operator_familiar_accuracy << ",\n"
        << "  \"operator_unseen_composition_accuracy\": "
        << result.operator_unseen_composition_accuracy << ",\n"
        << "  \"phase_offset_familiar_accuracy\": "
        << result.phase_offset_familiar_accuracy << ",\n"
        << "  \"phase_offset_unseen_composition_accuracy\": "
        << result.phase_offset_unseen_composition_accuracy << ",\n"
        << "  \"nearest_neighbor_familiar_accuracy\": "
        << result.nearest_neighbor_familiar_accuracy << ",\n"
        << "  \"nearest_neighbor_unseen_composition_accuracy\": "
        << result.nearest_neighbor_unseen_composition_accuracy << ",\n"
        << "  \"supervised_familiar_accuracy\": "
        << result.supervised_familiar_accuracy << ",\n"
        << "  \"supervised_unseen_composition_accuracy\": "
        << result.supervised_unseen_composition_accuracy << ",\n"
        << "  \"oracle_unseen_composition_accuracy\": "
        << result.oracle_unseen_composition_accuracy << ",\n"
        << "  \"noisy_input_accuracy\": "
        << result.noisy_input_accuracy << ",\n"
        << "  \"ambiguous_context_accuracy\": "
        << result.ambiguous_context_accuracy << ",\n"
        << "  \"conflicting_mode_accuracy\": "
        << result.conflicting_mode_accuracy << ",\n"
        << "  \"operator_mode_count\": "
        << result.operator_mode_count << ",\n"
        << "  \"operator_selected_family_count\": "
        << result.operator_selected_family_count << ",\n"
        << "  \"operator_bytes\": "
        << result.operator_bytes << ",\n"
        << "  \"training_seconds\": "
        << result.training_seconds << ",\n"
        << "  \"inference_seconds\": "
        << result.inference_seconds << ",\n"
        << "  \"deterministic_run_hash\": \""
        << format_run_hash(result.deterministic_run_hash)
        << "\",\n"
        << "  \"scientific_decision\": \""
        << result.scientific_decision << "\",\n"
        << "  \"cases\": [\n";
    for (std::size_t index = 0U;
         index < result.cases.size();
         ++index) {
        const OperatorCompositionCaseResult& item =
            result.cases[index];
        output
            << "    {\n"
            << "      \"name\": \"" << item.name << "\",\n"
            << "      \"category\": \"" << item.category << "\",\n"
            << "      \"operator_extension_accuracy\": "
            << item.operator_extension_accuracy << ",\n"
            << "      \"phase_offset_accuracy\": "
            << item.phase_offset_accuracy << ",\n"
            << "      \"nearest_neighbor_accuracy\": "
            << item.nearest_neighbor_accuracy << ",\n"
            << "      \"supervised_operator_accuracy\": "
            << item.supervised_operator_accuracy << ",\n"
            << "      \"oracle_accuracy\": "
            << item.oracle_accuracy << "\n"
            << "    }";
        if (index + 1U != result.cases.size()) {
            output << ',';
        }
        output << '\n';
    }
    output << "  ]\n}\n";
}

}  // namespace rlf::experiments
