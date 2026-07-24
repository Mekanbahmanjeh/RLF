#include "rlf/experiments/checkpoint_workflow.hpp"

#include "rlf/core/deterministic_rng.hpp"
#include "rlf/core/phase_vector.hpp"
#include "rlf/core/resonant_fabric.hpp"
#include "rlf/core/resonant_mode.hpp"
#include "rlf/experiments/phase_vector_smoke.hpp"
#include "rlf/learning/local_learning.hpp"
#include "rlf/learning/structural_learning.hpp"
#include "rlf/memory/associative_memory.hpp"
#include "rlf/storage/checkpoint.hpp"

#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
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
constexpr std::uint64_t transformation_seed_mask =
    0x6A09E667F3BCC909ULL;
constexpr std::uint64_t training_seed_mask =
    0xBB67AE8584CAA73BULL;
constexpr std::uint64_t evaluation_seed_mask =
    0x3C6EF372FE94F82BULL;
constexpr std::size_t maximum_trace_sample_id = 1'000'000U;

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

[[nodiscard]] core::PhaseVector task_transformation(
    const std::uint64_t seed,
    const std::size_t dimension
) {
    core::DeterministicRng random_number_generator(
        seed ^ transformation_seed_mask
    );
    return core::PhaseVector::random(
        dimension,
        random_number_generator
    );
}

[[nodiscard]] std::vector<Example> make_examples(
    const std::uint64_t seed,
    const std::size_t dimension,
    const std::size_t count,
    const std::uint64_t seed_mask
) {
    const core::PhaseVector transformation =
        task_transformation(seed, dimension);
    core::DeterministicRng random_number_generator(seed ^ seed_mask);
    std::vector<Example> examples;
    examples.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        core::PhaseVector input = core::PhaseVector::random(
            dimension,
            random_number_generator
        );
        core::PhaseVector target = input.composed(transformation);
        examples.push_back({
            .input = std::move(input),
            .target = std::move(target),
        });
    }
    return examples;
}

[[nodiscard]] Example make_trace_example(
    const std::uint64_t seed,
    const std::size_t dimension,
    const std::size_t sample_id
) {
    const core::PhaseVector transformation =
        task_transformation(seed, dimension);
    core::DeterministicRng random_number_generator(
        seed ^ evaluation_seed_mask
    );
    core::PhaseVector input = core::PhaseVector::zeros(dimension);
    for (std::size_t index = 0U; index <= sample_id; ++index) {
        input = core::PhaseVector::random(
            dimension,
            random_number_generator
        );
    }
    core::PhaseVector target = input.composed(transformation);
    return {
        .input = std::move(input),
        .target = std::move(target),
    };
}

[[nodiscard]] core::FabricConfig workflow_fabric_config(
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

[[nodiscard]] std::unique_ptr<learning::LocalUpdateStrategy>
make_update_strategy(const std::string_view name) {
    if (name == "resonance_weighted") {
        return std::make_unique<learning::ResonanceWeightedUpdate>();
    }
    if (name == "winner_only") {
        return std::make_unique<learning::WinnerOnlyUpdate>();
    }
    if (name == "normalized_responsibility") {
        return std::make_unique<
            learning::NormalizedResponsibilityUpdate
        >();
    }
    throw std::runtime_error(
        "checkpoint contains an unsupported update strategy"
    );
}

[[nodiscard]] core::ResonantFabric restore_fabric(
    const storage::CheckpointData& checkpoint
) {
    core::ResonantFabric fabric(checkpoint.config);
    fabric.set_update_strategy(
        make_update_strategy(checkpoint.update_strategy)
    );
    for (const core::ResonantMode& mode : checkpoint.modes) {
        fabric.add_mode(mode);
    }
    return fabric;
}

void validate_task(
    const storage::CheckpointData& checkpoint,
    const std::string& task
) {
    if (task != checkpoint_workflow_task) {
        throw std::invalid_argument(
            "unsupported evaluation task: " + task
        );
    }
    const auto metadata_task =
        checkpoint.experiment_metadata.find("task");
    if (metadata_task == checkpoint.experiment_metadata.end() ||
        metadata_task->second != task) {
        throw std::runtime_error(
            "checkpoint task metadata does not match --task"
        );
    }
}

[[nodiscard]] Evaluation evaluate(
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

[[nodiscard]] std::size_t fabric_bytes(
    const core::ResonantFabric& fabric
) {
    std::size_t bytes = sizeof(fabric);
    for (const core::ResonantMode& mode : fabric.modes()) {
        bytes += learning::StructuralLearner::estimate_mode_bytes(mode);
    }
    return bytes;
}

[[nodiscard]] std::uint64_t checkpoint_state_hash(
    const std::uint64_t seed,
    const core::ResonantFabric& fabric
) noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, seed);
    hash_string(hash, checkpoint_workflow_task);
    for (const core::ResonantMode& mode : fabric.modes()) {
        hash_u64(hash, mode.id);
        hash_phase_vector(hash, mode.context_key);
        hash_phase_vector(hash, mode.transformation);
    }
    return hash;
}

void write_number_array(
    std::ostream& output,
    const std::vector<double>& values
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

void write_id_array(
    std::ostream& output,
    const std::vector<std::uint64_t>& values
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

CheckpointTrainingResult train_checkpoint_workflow(
    const CheckpointTrainingConfig& config
) {
    if (config.dimension == 0U ||
        config.training_examples == 0U ||
        config.evaluation_examples == 0U ||
        config.checkpoint_path.empty()) {
        throw std::invalid_argument(
            "invalid checkpoint training configuration"
        );
    }

    const std::vector<Example> training = make_examples(
        config.seed,
        config.dimension,
        config.training_examples,
        training_seed_mask
    );
    const std::vector<Example> evaluation_examples = make_examples(
        config.seed,
        config.dimension,
        config.evaluation_examples,
        evaluation_seed_mask
    );
    core::ResonantFabric fabric =
        core::ResonantFabric(workflow_fabric_config(config.dimension));
    fabric.set_update_strategy(
        std::make_unique<learning::WinnerOnlyUpdate>()
    );
    fabric.add_mode(core::ResonantMode(
        1ULL,
        training.front().input,
        core::PhaseVector::zeros(config.dimension),
        0.01F,
        1.0F,
        0.0F,
        0ULL
    ));

    const Evaluation initial =
        evaluate(fabric, evaluation_examples);
    learning::LocalLearningConfig learning_config;
    learning_config.transformation_learning_rate = 0.35;
    learning_config.context_learning_rate = 0.0;
    learning_config.confidence_learning_rate = 0.0;
    learning_config.utility_learning_rate = 0.0;
    learning_config.use_utility = false;

    double prediction_error_total = 0.0;
    std::size_t update_operations = 0U;
    const auto training_start = std::chrono::steady_clock::now();
    for (const Example& example : training) {
        const core::LearningResult update = fabric.learn(
            example.input,
            example.target,
            learning_config
        );
        prediction_error_total += update.prediction_error;
        update_operations += update.updated_mode_ids.size();
    }
    const auto training_end = std::chrono::steady_clock::now();
    const double training_seconds =
        std::chrono::duration<double>(
            training_end - training_start
        ).count();
    const Evaluation final = evaluate(fabric, evaluation_examples);

    storage::CheckpointData checkpoint{
        .config = fabric.config(),
        .master_seed = config.seed,
        .training_step = fabric.training_step(),
        .modes = std::vector<core::ResonantMode>(
            fabric.modes().begin(),
            fabric.modes().end()
        ),
        .associative_memory =
            memory::AssociativeMemory(config.dimension, 1U),
        .structural_statistics = fabric.structural_statistics(),
        .update_strategy = std::string(fabric.update_strategy_name()),
        .experiment_metadata = {
            {"experiment", "checkpoint_training"},
            {"task", checkpoint_workflow_task},
            {"training_examples",
             std::to_string(config.training_examples)},
            {"evaluation_examples",
             std::to_string(config.evaluation_examples)},
        },
    };
    storage::save_checkpoint(config.checkpoint_path, checkpoint);
    const storage::CheckpointSummary summary =
        storage::inspect_checkpoint(config.checkpoint_path);

    const double training_count =
        static_cast<double>(config.training_examples);
    const double evaluation_count =
        static_cast<double>(config.evaluation_examples);
    const std::size_t bytes = fabric_bytes(fabric);
    const std::uint64_t run_hash =
        checkpoint_state_hash(config.seed, fabric);
    const std::size_t peak_resident = peak_resident_memory_bytes();
    const double active_operations = 2.0;
    return {
        .task = checkpoint_workflow_task,
        .seed = config.seed,
        .dimension = config.dimension,
        .training_examples = config.training_examples,
        .evaluation_examples = config.evaluation_examples,
        .initial_mean_similarity = initial.mean_similarity,
        .final_mean_similarity = final.mean_similarity,
        .final_accuracy = final.accuracy,
        .checkpoint_bytes = summary.file_bytes,
        .payload_checksum = summary.payload_checksum,
        .metrics = {
            .task_accuracy = final.accuracy,
            .one_shot_recall = 0.0,
            .retained_accuracy = final.accuracy,
            .catastrophic_forgetting_score = 0.0,
            .compositional_generalization_score = 0.0,
            .prediction_error =
                prediction_error_total / training_count,
            .average_settling_cycles = final.average_cycles,
            .average_modes_retrieved = final.average_retrieved,
            .average_modes_activated = final.average_activated,
            .modes_created = 1ULL,
            .modes_split = 0ULL,
            .modes_merged = 0ULL,
            .modes_pruned = 0ULL,
            .bytes_stored = bytes,
            .peak_resident_bytes = peak_resident,
            .training_seconds = training_seconds,
            .inference_seconds = final.seconds,
            .training_examples_per_second =
                training_count / training_seconds,
            .inference_examples_per_second =
                evaluation_count / final.seconds,
            .update_operations_per_example =
                static_cast<double>(update_operations) /
                training_count,
            .active_operations_per_inference = active_operations,
            .efficiency_score = provisional_efficiency_score(
                final.accuracy,
                bytes,
                active_operations,
                training_seconds
            ),
            .deterministic_run_hash = run_hash,
        },
        .deterministic_run_hash = run_hash,
    };
}

CheckpointEvaluationResult evaluate_checkpoint_workflow(
    const std::filesystem::path& checkpoint_path,
    const std::string& task,
    const std::size_t evaluation_examples
) {
    if (checkpoint_path.empty() || evaluation_examples == 0U) {
        throw std::invalid_argument(
            "checkpoint evaluation requires a path and examples"
        );
    }
    const storage::CheckpointData checkpoint =
        storage::load_checkpoint(checkpoint_path);
    validate_task(checkpoint, task);
    core::ResonantFabric fabric = restore_fabric(checkpoint);
    const std::vector<Example> examples = make_examples(
        checkpoint.master_seed,
        checkpoint.config.dimension,
        evaluation_examples,
        evaluation_seed_mask
    );
    const Evaluation evaluation = evaluate(fabric, examples);
    const double count = static_cast<double>(evaluation_examples);
    const std::size_t bytes = fabric_bytes(fabric);
    std::uint64_t run_hash =
        checkpoint_state_hash(checkpoint.master_seed, fabric);
    hash_u64(
        run_hash,
        std::bit_cast<std::uint64_t>(evaluation.mean_similarity)
    );
    hash_u64(
        run_hash,
        std::bit_cast<std::uint64_t>(evaluation.accuracy)
    );
    return {
        .task = task,
        .seed = checkpoint.master_seed,
        .dimension = checkpoint.config.dimension,
        .evaluation_examples = evaluation_examples,
        .mean_similarity = evaluation.mean_similarity,
        .accuracy = evaluation.accuracy,
        .metrics = {
            .task_accuracy = evaluation.accuracy,
            .one_shot_recall = 0.0,
            .retained_accuracy = evaluation.accuracy,
            .catastrophic_forgetting_score = 0.0,
            .compositional_generalization_score = 0.0,
            .prediction_error = 1.0 - evaluation.mean_similarity,
            .average_settling_cycles = evaluation.average_cycles,
            .average_modes_retrieved = evaluation.average_retrieved,
            .average_modes_activated = evaluation.average_activated,
            .modes_created =
                checkpoint.structural_statistics.modes_created,
            .modes_split =
                checkpoint.structural_statistics.modes_split,
            .modes_merged =
                checkpoint.structural_statistics.modes_merged,
            .modes_pruned =
                checkpoint.structural_statistics.modes_pruned,
            .bytes_stored = bytes,
            .peak_resident_bytes = peak_resident_memory_bytes(),
            .training_seconds = 0.0,
            .inference_seconds = evaluation.seconds,
            .training_examples_per_second = 0.0,
            .inference_examples_per_second =
                count / evaluation.seconds,
            .update_operations_per_example = 0.0,
            .active_operations_per_inference = 2.0,
            .efficiency_score = 0.0,
            .deterministic_run_hash = run_hash,
        },
        .deterministic_run_hash = run_hash,
    };
}

CheckpointTraceResult trace_checkpoint_workflow(
    const std::filesystem::path& checkpoint_path,
    const std::string& task,
    const std::size_t sample_id
) {
    if (checkpoint_path.empty() ||
        sample_id > maximum_trace_sample_id) {
        throw std::invalid_argument(
            "invalid checkpoint trace request"
        );
    }
    const storage::CheckpointData checkpoint =
        storage::load_checkpoint(checkpoint_path);
    validate_task(checkpoint, task);
    core::ResonantFabric fabric = restore_fabric(checkpoint);
    const Example example = make_trace_example(
        checkpoint.master_seed,
        checkpoint.config.dimension,
        sample_id
    );
    const core::SettleResult prediction =
        fabric.settle(example.input, true);
    if (!prediction.trace.has_value()) {
        throw std::runtime_error(
            "checkpoint trace capture did not produce a trace"
        );
    }
    const double target_similarity =
        prediction.state.similarity(example.target);
    std::uint64_t run_hash =
        checkpoint_state_hash(checkpoint.master_seed, fabric);
    hash_u64(run_hash, static_cast<std::uint64_t>(sample_id));
    hash_u64(
        run_hash,
        std::bit_cast<std::uint64_t>(target_similarity)
    );
    for (const core::SettlingCycleTrace& cycle :
         prediction.trace->cycles) {
        hash_u64(
            run_hash,
            static_cast<std::uint64_t>(cycle.cycle_number)
        );
        for (const std::uint64_t mode_id :
             cycle.retrieved_mode_ids) {
            hash_u64(run_hash, mode_id);
        }
        for (const std::uint64_t mode_id :
             cycle.active_mode_ids) {
            hash_u64(run_hash, mode_id);
        }
    }
    return {
        .task = task,
        .seed = checkpoint.master_seed,
        .dimension = checkpoint.config.dimension,
        .sample_id = sample_id,
        .target_similarity = target_similarity,
        .stopping_reason =
            std::string(core::to_string(prediction.stopping_reason)),
        .cycles = prediction.trace->cycles,
        .deterministic_run_hash = run_hash,
    };
}

void write_checkpoint_training_json(
    std::ostream& output,
    const CheckpointTrainingResult& result
) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "{\n"
           << "  \"command\": \"train\",\n"
           << "  \"status\": \"observed\",\n"
           << "  \"task\": \"" << result.task << "\",\n"
           << "  \"seed\": " << result.seed << ",\n"
           << "  \"dimension\": " << result.dimension << ",\n"
           << "  \"training_examples\": "
           << result.training_examples << ",\n"
           << "  \"evaluation_examples\": "
           << result.evaluation_examples << ",\n"
           << "  \"initial_mean_similarity\": "
           << result.initial_mean_similarity << ",\n"
           << "  \"final_mean_similarity\": "
           << result.final_mean_similarity << ",\n"
           << "  \"final_accuracy\": "
           << result.final_accuracy << ",\n"
           << "  \"checkpoint_bytes\": "
           << result.checkpoint_bytes << ",\n"
           << "  \"payload_checksum\": \""
           << format_run_hash(result.payload_checksum) << "\",\n"
           << "  \"metrics\": ";
    write_metrics_json(output, result.metrics, 2U);
    output << ",\n"
           << "  \"deterministic_run_hash\": \""
           << format_run_hash(result.deterministic_run_hash)
           << "\"\n"
           << "}\n";
}

void write_checkpoint_evaluation_json(
    std::ostream& output,
    const CheckpointEvaluationResult& result
) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "{\n"
           << "  \"command\": \"evaluate\",\n"
           << "  \"status\": \"observed\",\n"
           << "  \"task\": \"" << result.task << "\",\n"
           << "  \"seed\": " << result.seed << ",\n"
           << "  \"dimension\": " << result.dimension << ",\n"
           << "  \"evaluation_examples\": "
           << result.evaluation_examples << ",\n"
           << "  \"mean_similarity\": "
           << result.mean_similarity << ",\n"
           << "  \"accuracy\": " << result.accuracy << ",\n"
           << "  \"metrics\": ";
    write_metrics_json(output, result.metrics, 2U);
    output << ",\n"
           << "  \"deterministic_run_hash\": \""
           << format_run_hash(result.deterministic_run_hash)
           << "\"\n"
           << "}\n";
}

void write_checkpoint_trace_json(
    std::ostream& output,
    const CheckpointTraceResult& result
) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "{\n"
           << "  \"command\": \"trace\",\n"
           << "  \"status\": \"observed\",\n"
           << "  \"task\": \"" << result.task << "\",\n"
           << "  \"seed\": " << result.seed << ",\n"
           << "  \"dimension\": " << result.dimension << ",\n"
           << "  \"sample_id\": " << result.sample_id << ",\n"
           << "  \"target_similarity\": "
           << result.target_similarity << ",\n"
           << "  \"stopping_reason\": \""
           << result.stopping_reason << "\",\n"
           << "  \"cycles\": [\n";
    for (std::size_t index = 0U;
         index < result.cycles.size();
         ++index) {
        const core::SettlingCycleTrace& cycle = result.cycles[index];
        output << "    {\n"
               << "      \"cycle_number\": "
               << cycle.cycle_number << ",\n"
               << "      \"retrieved_mode_ids\": ";
        write_id_array(output, cycle.retrieved_mode_ids);
        output << ",\n      \"resonance_scores\": ";
        write_number_array(output, cycle.resonance_scores);
        output << ",\n      \"active_mode_ids\": ";
        write_id_array(output, cycle.active_mode_ids);
        output << ",\n      \"proposal_weights\": ";
        write_number_array(output, cycle.proposal_weights);
        output << ",\n      \"state_change_radians\": "
               << cycle.state_change_radians << ",\n"
               << "      \"coherence\": "
               << cycle.coherence << ",\n"
               << "      \"confidence\": "
               << cycle.confidence << ",\n"
               << "      \"stopping_reason\": \""
               << core::to_string(cycle.stopping_reason)
               << "\"\n"
               << "    }";
        if (index + 1U != result.cycles.size()) {
            output << ',';
        }
        output << '\n';
    }
    output << "  ],\n"
           << "  \"deterministic_run_hash\": \""
           << format_run_hash(result.deterministic_run_hash)
           << "\"\n"
           << "}\n";
}

}  // namespace rlf::experiments
