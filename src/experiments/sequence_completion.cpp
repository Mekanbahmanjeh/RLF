#include "rlf/experiments/sequence_completion.hpp"

#include "rlf/baselines/transition_table.hpp"
#include "rlf/core/deterministic_rng.hpp"
#include "rlf/core/encoding.hpp"
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

constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;

enum class SequenceKind {
    deterministic,
    probabilistic,
};

struct TransitionExample final {
    std::size_t position;
    std::size_t current;
    std::size_t next;
};

struct TrainingMeasurement final {
    double one_shot_accuracy;
    double prediction_error;
    double average_cycles;
    double update_operations_per_example;
    double seconds;
};

struct EvaluationMeasurement final {
    double accuracy;
    double average_cycles;
    double average_retrieved;
    double average_activated;
    double seconds;
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

[[nodiscard]] std::vector<std::string> make_symbols(
    const std::size_t count
) {
    std::vector<std::string> symbols;
    symbols.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        symbols.push_back("s" + std::to_string(index));
    }
    return symbols;
}

[[nodiscard]] std::size_t sample_next(
    const SequenceKind kind,
    const std::size_t current,
    const std::size_t symbol_count,
    const double dominant_probability,
    core::DeterministicRng& random_number_generator
) {
    const std::size_t dominant = (current + 1U) % symbol_count;
    if (kind == SequenceKind::deterministic ||
        random_number_generator.uniform_unit() <
            dominant_probability) {
        return dominant;
    }
    return (current + 2U) % symbol_count;
}

[[nodiscard]] std::vector<TransitionExample> make_examples(
    const SequenceKind kind,
    const std::size_t start_position,
    const std::size_t count,
    const std::size_t symbol_count,
    const double dominant_probability,
    const std::uint64_t seed
) {
    core::DeterministicRng random_number_generator(seed);
    std::vector<TransitionExample> examples;
    examples.reserve(count);
    for (std::size_t offset = 0U; offset < count; ++offset) {
        const std::size_t position = start_position + offset;
        const std::size_t current = position % symbol_count;
        examples.push_back({
            .position = position,
            .current = current,
            .next = sample_next(
                kind,
                current,
                symbol_count,
                dominant_probability,
                random_number_generator
            ),
        });
    }
    return examples;
}

[[nodiscard]] core::PhaseVector perturb(
    const core::PhaseVector& value,
    const double corruption_radians,
    core::DeterministicRng& random_number_generator
) {
    std::vector<float> noise;
    noise.reserve(value.size());
    for (std::size_t index = 0U; index < value.size(); ++index) {
        const double signed_unit =
            (2.0 * random_number_generator.uniform_unit()) - 1.0;
        noise.push_back(
            static_cast<float>(signed_unit * corruption_radians)
        );
    }
    return value.composed(core::PhaseVector(std::move(noise)));
}

[[nodiscard]] core::FabricConfig make_fabric_config(
    const SequenceCompletionConfig& config
) {
    core::SettlingConfig settling;
    settling.candidate_count = config.symbol_count;
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
    structural.correction_history_capacity = 32U;
    structural.split_minimum_samples = 12U;
    structural.split_minimum_cluster_size = 4U;
    structural.split_minimum_transformation_separation_radians = 0.4;
    structural.split_minimum_context_separation_radians = 0.25;
    structural.split_minimum_validation_gain_radians = 0.1;
    structural.merge_maximum_key_error_radians = 0.01;
    structural.merge_maximum_transformation_error_radians = 0.01;
    structural.pruning_minimum_age_steps =
        std::numeric_limits<std::uint64_t>::max();
    structural.pruning_maximum_inactive_steps =
        std::numeric_limits<std::uint64_t>::max();
    structural.pruning_disabled_grace_steps =
        std::numeric_limits<std::uint64_t>::max();

    return {
        .dimension = config.dimension,
        .maximum_modes = config.symbol_count * 2U,
        .settling = settling,
        .structural_learning = structural,
    };
}

[[nodiscard]] TrainingMeasurement train_rlf(
    core::ResonantFabric& fabric,
    const core::SymbolEncoder& encoder,
    const std::vector<TransitionExample>& examples,
    const std::size_t symbol_count
) {
    learning::LocalLearningConfig learning_config;
    learning_config.transformation_learning_rate = 0.2;
    learning_config.context_learning_rate = 0.0;
    learning_config.confidence_learning_rate = 0.0;
    learning_config.utility_learning_rate = 0.0;
    learning_config.use_utility = false;

    std::vector<bool> seen(symbol_count, false);
    std::size_t one_shot_successes = 0U;
    std::size_t one_shot_trials = 0U;
    double prediction_error_total = 0.0;
    double cycles_total = 0.0;
    std::size_t update_operations = 0U;
    const auto start = std::chrono::steady_clock::now();
    for (const TransitionExample& example : examples) {
        const core::PhaseVector& input =
            encoder.encode(encoder.symbols()[example.current]);
        const core::PhaseVector& target =
            encoder.encode(encoder.symbols()[example.next]);
        const core::LearningResult update = fabric.learn(
            input,
            target,
            learning_config
        );
        prediction_error_total += update.prediction_error;
        cycles_total +=
            static_cast<double>(update.prediction.cycles);
        update_operations += update.updated_mode_ids.size();
        update_operations += update.structural_events.size();

        if (!seen[example.current]) {
            seen[example.current] = true;
            ++one_shot_trials;
            const core::SettleResult prediction = fabric.settle(input);
            const core::DecodedSymbol decoded =
                encoder.decode(prediction.state);
            if (decoded.symbol == encoder.symbols()[example.next]) {
                ++one_shot_successes;
            }
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const double example_count =
        static_cast<double>(examples.size());
    return {
        .one_shot_accuracy =
            static_cast<double>(one_shot_successes) /
            static_cast<double>(one_shot_trials),
        .prediction_error =
            prediction_error_total / example_count,
        .average_cycles = cycles_total / example_count,
        .update_operations_per_example =
            static_cast<double>(update_operations) / example_count,
        .seconds =
            std::chrono::duration<double>(end - start).count(),
    };
}

[[nodiscard]] double train_baseline(
    baselines::TransitionTablePredictor& predictor,
    const std::vector<TransitionExample>& examples,
    const std::size_t symbol_count,
    double& one_shot_accuracy
) {
    std::vector<bool> seen(symbol_count, false);
    std::size_t one_shot_successes = 0U;
    std::size_t one_shot_trials = 0U;
    const auto start = std::chrono::steady_clock::now();
    for (const TransitionExample& example : examples) {
        predictor.observe(
            static_cast<std::uint64_t>(example.current),
            static_cast<std::uint64_t>(example.next)
        );
        if (!seen[example.current]) {
            seen[example.current] = true;
            ++one_shot_trials;
            const auto prediction = predictor.predict(
                static_cast<std::uint64_t>(example.current)
            );
            if (prediction.has_value() &&
                *prediction ==
                    static_cast<std::uint64_t>(example.next)) {
                ++one_shot_successes;
            }
        }
    }
    const auto end = std::chrono::steady_clock::now();
    one_shot_accuracy =
        static_cast<double>(one_shot_successes) /
        static_cast<double>(one_shot_trials);
    return std::chrono::duration<double>(end - start).count();
}

[[nodiscard]] EvaluationMeasurement evaluate_rlf(
    core::ResonantFabric& fabric,
    const core::SymbolEncoder& encoder,
    const std::vector<TransitionExample>& examples,
    const double corruption_radians,
    const std::uint64_t corruption_seed
) {
    core::DeterministicRng corruption_rng(corruption_seed);
    std::size_t successes = 0U;
    double cycles_total = 0.0;
    double retrieved_total = 0.0;
    double activated_total = 0.0;
    const auto start = std::chrono::steady_clock::now();
    for (const TransitionExample& example : examples) {
        core::PhaseVector input =
            encoder.encode(encoder.symbols()[example.current]);
        if (corruption_radians > 0.0) {
            input = perturb(
                input,
                corruption_radians,
                corruption_rng
            );
        }
        const core::SettleResult prediction =
            fabric.settle(input, true);
        const core::DecodedSymbol decoded =
            encoder.decode(prediction.state);
        if (decoded.symbol == encoder.symbols()[example.next]) {
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
        .accuracy =
            static_cast<double>(successes) / count,
        .average_cycles = cycles_total / count,
        .average_retrieved = retrieved_total / count,
        .average_activated = activated_total / count,
        .seconds =
            std::chrono::duration<double>(end - start).count(),
    };
}

[[nodiscard]] EvaluationMeasurement evaluate_baseline(
    const baselines::TransitionTablePredictor& predictor,
    const core::SymbolEncoder& encoder,
    const std::vector<TransitionExample>& examples,
    const double corruption_radians,
    const std::uint64_t corruption_seed
) {
    core::DeterministicRng corruption_rng(corruption_seed);
    std::size_t successes = 0U;
    const auto start = std::chrono::steady_clock::now();
    for (const TransitionExample& example : examples) {
        std::size_t current = example.current;
        if (corruption_radians > 0.0) {
            const core::PhaseVector corrupted = perturb(
                encoder.encode(encoder.symbols()[example.current]),
                corruption_radians,
                corruption_rng
            );
            const core::DecodedSymbol decoded = encoder.decode(corrupted);
            const auto found = std::find(
                encoder.symbols().begin(),
                encoder.symbols().end(),
                decoded.symbol
            );
            current = static_cast<std::size_t>(
                std::distance(encoder.symbols().begin(), found)
            );
        }
        const auto prediction = predictor.predict(
            static_cast<std::uint64_t>(current)
        );
        if (prediction.has_value() &&
            *prediction ==
                static_cast<std::uint64_t>(example.next)) {
            ++successes;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const double count = static_cast<double>(examples.size());
    return {
        .accuracy =
            static_cast<double>(successes) / count,
        .average_cycles = 0.0,
        .average_retrieved = 0.0,
        .average_activated = 0.0,
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

[[nodiscard]] SequenceSystemResult make_rlf_result(
    const std::string& system,
    const core::ResonantFabric& fabric,
    const TrainingMeasurement& training,
    const EvaluationMeasurement& clean,
    const EvaluationMeasurement& unseen,
    const EvaluationMeasurement& corrupted,
    const std::size_t training_examples,
    const std::size_t evaluation_examples,
    const std::uint64_t run_hash
) {
    const double inference_seconds =
        clean.seconds + unseen.seconds + corrupted.seconds;
    const double inference_count =
        static_cast<double>(evaluation_examples * 3U);
    const std::size_t bytes = fabric_bytes(fabric);
    const double useful_retained =
        clean.accuracy *
        static_cast<double>(fabric.modes().size());
    return {
        .system = system,
        .next_symbol_accuracy = clean.accuracy,
        .unseen_position_accuracy = unseen.accuracy,
        .corrupted_input_accuracy = corrupted.accuracy,
        .metrics = {
            .task_accuracy = clean.accuracy,
            .one_shot_recall = training.one_shot_accuracy,
            .retained_accuracy = clean.accuracy,
            .catastrophic_forgetting_score = 0.0,
            .compositional_generalization_score = 0.0,
            .prediction_error = training.prediction_error,
            .average_settling_cycles = clean.average_cycles,
            .average_modes_retrieved = clean.average_retrieved,
            .average_modes_activated = clean.average_activated,
            .modes_created =
                fabric.structural_statistics().modes_created,
            .modes_split =
                fabric.structural_statistics().modes_split,
            .modes_merged =
                fabric.structural_statistics().modes_merged,
            .modes_pruned =
                fabric.structural_statistics().modes_pruned,
            .bytes_stored = bytes,
            .peak_resident_bytes =
                peak_resident_memory_bytes(),
            .training_seconds = training.seconds,
            .inference_seconds = inference_seconds,
            .training_examples_per_second =
                training.seconds > 0.0
                    ? static_cast<double>(training_examples) /
                        training.seconds
                    : 0.0,
            .inference_examples_per_second =
                inference_seconds > 0.0
                    ? inference_count / inference_seconds
                    : 0.0,
            .update_operations_per_example =
                training.update_operations_per_example,
            .active_operations_per_inference =
                clean.average_retrieved +
                clean.average_activated,
            .efficiency_score = provisional_efficiency_score(
                useful_retained,
                bytes,
                clean.average_retrieved +
                    clean.average_activated,
                training.seconds
            ),
            .deterministic_run_hash = run_hash,
        },
    };
}

[[nodiscard]] SequenceSystemResult make_baseline_result(
    const std::string& system,
    const baselines::TransitionTablePredictor& predictor,
    const double one_shot_accuracy,
    const double training_seconds,
    const EvaluationMeasurement& clean,
    const EvaluationMeasurement& unseen,
    const EvaluationMeasurement& corrupted,
    const std::size_t training_examples,
    const std::size_t evaluation_examples,
    const std::uint64_t run_hash
) {
    const double inference_seconds =
        clean.seconds + unseen.seconds + corrupted.seconds;
    const double inference_count =
        static_cast<double>(evaluation_examples * 3U);
    const std::size_t bytes = predictor.bytes_stored();
    return {
        .system = system,
        .next_symbol_accuracy = clean.accuracy,
        .unseen_position_accuracy = unseen.accuracy,
        .corrupted_input_accuracy = corrupted.accuracy,
        .metrics = {
            .task_accuracy = clean.accuracy,
            .one_shot_recall = one_shot_accuracy,
            .retained_accuracy = clean.accuracy,
            .catastrophic_forgetting_score = 0.0,
            .compositional_generalization_score = 0.0,
            .prediction_error = 1.0 - clean.accuracy,
            .average_settling_cycles = 0.0,
            .average_modes_retrieved = 0.0,
            .average_modes_activated = 0.0,
            .modes_created = 0ULL,
            .modes_split = 0ULL,
            .modes_merged = 0ULL,
            .modes_pruned = 0ULL,
            .bytes_stored = bytes,
            .peak_resident_bytes =
                peak_resident_memory_bytes(),
            .training_seconds = training_seconds,
            .inference_seconds = inference_seconds,
            .training_examples_per_second =
                training_seconds > 0.0
                    ? static_cast<double>(training_examples) /
                        training_seconds
                    : 0.0,
            .inference_examples_per_second =
                inference_seconds > 0.0
                    ? inference_count / inference_seconds
                    : 0.0,
            .update_operations_per_example = 1.0,
            .active_operations_per_inference = 1.0,
            .efficiency_score = provisional_efficiency_score(
                clean.accuracy *
                    static_cast<double>(predictor.transitions()),
                bytes,
                1.0,
                training_seconds
            ),
            .deterministic_run_hash = run_hash,
        },
    };
}

[[nodiscard]] SequenceCaseResult run_case(
    const SequenceCompletionConfig& config,
    const core::SymbolEncoder& encoder,
    const SequenceKind kind,
    const std::string& name,
    const std::uint64_t seed_offset
) {
    const std::vector<TransitionExample> training = make_examples(
        kind,
        0U,
        config.training_examples,
        config.symbol_count,
        config.dominant_probability,
        config.seed ^ seed_offset
    );
    const std::vector<TransitionExample> clean = make_examples(
        kind,
        config.training_examples,
        config.evaluation_examples,
        config.symbol_count,
        config.dominant_probability,
        config.seed ^ (seed_offset + 1ULL)
    );
    const std::vector<TransitionExample> unseen = make_examples(
        kind,
        config.training_examples + config.evaluation_examples,
        config.evaluation_examples,
        config.symbol_count,
        config.dominant_probability,
        config.seed ^ (seed_offset + 2ULL)
    );

    core::ResonantFabric fabric(make_fabric_config(config));
    fabric.set_update_strategy(
        std::make_unique<learning::WinnerOnlyUpdate>()
    );
    const TrainingMeasurement rlf_training = train_rlf(
        fabric,
        encoder,
        training,
        config.symbol_count
    );

    baselines::TransitionTablePredictor baseline;
    double baseline_one_shot = 0.0;
    const double baseline_training_seconds = train_baseline(
        baseline,
        training,
        config.symbol_count,
        baseline_one_shot
    );

    const EvaluationMeasurement rlf_clean = evaluate_rlf(
        fabric,
        encoder,
        clean,
        0.0,
        config.seed ^ (seed_offset + 3ULL)
    );
    const EvaluationMeasurement rlf_unseen = evaluate_rlf(
        fabric,
        encoder,
        unseen,
        0.0,
        config.seed ^ (seed_offset + 4ULL)
    );
    const EvaluationMeasurement rlf_corrupted = evaluate_rlf(
        fabric,
        encoder,
        clean,
        config.corruption_radians,
        config.seed ^ (seed_offset + 5ULL)
    );
    const EvaluationMeasurement baseline_clean = evaluate_baseline(
        baseline,
        encoder,
        clean,
        0.0,
        config.seed ^ (seed_offset + 3ULL)
    );
    const EvaluationMeasurement baseline_unseen = evaluate_baseline(
        baseline,
        encoder,
        unseen,
        0.0,
        config.seed ^ (seed_offset + 4ULL)
    );
    const EvaluationMeasurement baseline_corrupted = evaluate_baseline(
        baseline,
        encoder,
        clean,
        config.corruption_radians,
        config.seed ^ (seed_offset + 5ULL)
    );

    std::uint64_t rlf_hash = fnv_offset_basis;
    std::uint64_t baseline_hash = fnv_offset_basis;
    hash_string(rlf_hash, name);
    hash_string(baseline_hash, name);
    for (const core::ResonantMode& mode : fabric.modes()) {
        hash_u64(rlf_hash, mode.id);
        hash_phase_vector(rlf_hash, mode.context_key);
        hash_phase_vector(rlf_hash, mode.transformation);
    }
    hash_u64(
        rlf_hash,
        std::bit_cast<std::uint64_t>(rlf_clean.accuracy)
    );
    hash_u64(
        rlf_hash,
        std::bit_cast<std::uint64_t>(rlf_unseen.accuracy)
    );
    hash_u64(
        rlf_hash,
        std::bit_cast<std::uint64_t>(rlf_corrupted.accuracy)
    );
    hash_u64(
        baseline_hash,
        std::bit_cast<std::uint64_t>(baseline_clean.accuracy)
    );
    hash_u64(
        baseline_hash,
        std::bit_cast<std::uint64_t>(baseline_unseen.accuracy)
    );
    hash_u64(
        baseline_hash,
        std::bit_cast<std::uint64_t>(
            baseline_corrupted.accuracy
        )
    );
    hash_u64(
        baseline_hash,
        static_cast<std::uint64_t>(baseline.transitions())
    );

    return {
        .name = name,
        .rlf = make_rlf_result(
            "rlf_resonant_modes",
            fabric,
            rlf_training,
            rlf_clean,
            rlf_unseen,
            rlf_corrupted,
            config.training_examples,
            config.evaluation_examples,
            rlf_hash
        ),
        .baseline = make_baseline_result(
            "transition_table_baseline",
            baseline,
            baseline_one_shot,
            baseline_training_seconds,
            baseline_clean,
            baseline_unseen,
            baseline_corrupted,
            config.training_examples,
            config.evaluation_examples,
            baseline_hash
        ),
    };
}

}  // namespace

SequenceCompletionResult run_sequence_completion(
    const SequenceCompletionConfig& config
) {
    if (config.dimension == 0U ||
        config.symbol_count < 3U ||
        config.training_examples < config.symbol_count ||
        config.evaluation_examples == 0U ||
        !std::isfinite(config.corruption_radians) ||
        config.corruption_radians < 0.0 ||
        !std::isfinite(config.dominant_probability) ||
        config.dominant_probability <= 0.5 ||
        config.dominant_probability > 1.0) {
        throw std::invalid_argument(
            "invalid sequence-completion configuration"
        );
    }

    const core::SymbolEncoder encoder(
        config.dimension,
        config.seed ^ 0x5A5AULL,
        make_symbols(config.symbol_count)
    );
    std::vector<SequenceCaseResult> cases;
    cases.push_back(run_case(
        config,
        encoder,
        SequenceKind::deterministic,
        "deterministic_cycle",
        0x1000ULL
    ));
    cases.push_back(run_case(
        config,
        encoder,
        SequenceKind::probabilistic,
        "probabilistic_dominant_transition",
        0x2000ULL
    ));

    std::uint64_t run_hash = fnv_offset_basis;
    for (const SequenceCaseResult& sequence_case : cases) {
        hash_string(run_hash, sequence_case.name);
        hash_u64(
            run_hash,
            sequence_case.rlf.metrics.deterministic_run_hash
        );
        hash_u64(
            run_hash,
            sequence_case.baseline.metrics.deterministic_run_hash
        );
    }
    return {
        .seed = config.seed,
        .dimension = config.dimension,
        .symbol_count = config.symbol_count,
        .training_examples = config.training_examples,
        .evaluation_examples = config.evaluation_examples,
        .corruption_radians = config.corruption_radians,
        .dominant_probability = config.dominant_probability,
        .cases = std::move(cases),
        .deterministic_run_hash = run_hash,
    };
}

void write_sequence_completion_json(
    std::ostream& output,
    const SequenceCompletionResult& result
) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "{\n"
           << "  \"experiment\": \"sequence_completion\",\n"
           << "  \"status\": \"observed\",\n"
           << "  \"seed\": " << result.seed << ",\n"
           << "  \"dimension\": " << result.dimension << ",\n"
           << "  \"symbol_count\": " << result.symbol_count << ",\n"
           << "  \"training_examples\": "
           << result.training_examples << ",\n"
           << "  \"evaluation_examples\": "
           << result.evaluation_examples << ",\n"
           << "  \"corruption_radians\": "
           << result.corruption_radians << ",\n"
           << "  \"dominant_probability\": "
           << result.dominant_probability << ",\n"
           << "  \"cases\": [\n";
    for (std::size_t case_index = 0U;
         case_index < result.cases.size();
         ++case_index) {
        const SequenceCaseResult& sequence_case =
            result.cases[case_index];
        output << "    {\n"
               << "      \"name\": \"" << sequence_case.name << "\",\n"
               << "      \"rlf\": {\n"
               << "        \"system\": \""
               << sequence_case.rlf.system << "\",\n"
               << "        \"next_symbol_accuracy\": "
               << sequence_case.rlf.next_symbol_accuracy << ",\n"
               << "        \"unseen_position_accuracy\": "
               << sequence_case.rlf.unseen_position_accuracy << ",\n"
               << "        \"corrupted_input_accuracy\": "
               << sequence_case.rlf.corrupted_input_accuracy << ",\n"
               << "        \"metrics\": ";
        write_metrics_json(output, sequence_case.rlf.metrics, 8U);
        output << "\n      },\n"
               << "      \"baseline\": {\n"
               << "        \"system\": \""
               << sequence_case.baseline.system << "\",\n"
               << "        \"next_symbol_accuracy\": "
               << sequence_case.baseline.next_symbol_accuracy
               << ",\n"
               << "        \"unseen_position_accuracy\": "
               << sequence_case.baseline.unseen_position_accuracy
               << ",\n"
               << "        \"corrupted_input_accuracy\": "
               << sequence_case.baseline.corrupted_input_accuracy
               << ",\n"
               << "        \"metrics\": ";
        write_metrics_json(
            output,
            sequence_case.baseline.metrics,
            8U
        );
        output << "\n      }\n"
               << "    }";
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
            "failed to write sequence-completion result"
        );
    }
}

}  // namespace rlf::experiments
