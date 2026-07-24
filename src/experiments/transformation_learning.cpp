#include "rlf/experiments/transformation_learning.hpp"

#include "experiments/transformation_task_suite.hpp"
#include "rlf/core/deterministic_rng.hpp"
#include "rlf/core/phase_vector.hpp"
#include "rlf/core/resonant_fabric.hpp"
#include "rlf/core/resonant_mode.hpp"
#include "rlf/experiments/phase_vector_smoke.hpp"
#include "rlf/learning/local_learning.hpp"
#include "rlf/retrieval/mode_retriever.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace rlf::experiments {
namespace {

constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;
constexpr double task_success_similarity = 0.95;

struct Example final {
    core::PhaseVector input;
    core::PhaseVector target;
};

void hash_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (unsigned int byte_index = 0U; byte_index < 8U; ++byte_index) {
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
    const core::PhaseVector& phase_vector
) noexcept {
    for (const float angle : phase_vector.angles()) {
        hash_u64(
            hash,
            static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(angle))
        );
    }
}

[[nodiscard]] std::vector<Example> make_examples(
    const std::size_t count,
    const std::size_t dimension,
    const core::PhaseVector& transformation,
    core::DeterministicRng& random_number_generator
) {
    std::vector<Example> examples;
    examples.reserve(count);
    for (std::size_t example_index = 0U;
         example_index < count;
         ++example_index) {
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

[[nodiscard]] core::FabricConfig fabric_config(
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

[[nodiscard]] std::vector<
    std::unique_ptr<learning::LocalUpdateStrategy>
> make_strategies() {
    std::vector<std::unique_ptr<learning::LocalUpdateStrategy>> strategies;
    strategies.push_back(
        std::make_unique<learning::ResonanceWeightedUpdate>()
    );
    strategies.push_back(std::make_unique<learning::WinnerOnlyUpdate>());
    strategies.push_back(
        std::make_unique<learning::NormalizedResponsibilityUpdate>()
    );
    return strategies;
}

[[nodiscard]] std::pair<double, double> evaluate(
    core::ResonantFabric& fabric,
    const std::vector<Example>& examples
) {
    double similarity_total = 0.0;
    std::size_t successes = 0U;
    for (const Example& example : examples) {
        const core::SettleResult prediction = fabric.settle(example.input);
        const double similarity = prediction.state.similarity(example.target);
        similarity_total += similarity;
        if (similarity >= task_success_similarity) {
            ++successes;
        }
    }
    return {
        similarity_total / static_cast<double>(examples.size()),
        static_cast<double>(successes) /
            static_cast<double>(examples.size()),
    };
}

[[nodiscard]] TransformationStrategyResult run_strategy(
    const TransformationLearningConfig& config,
    const core::PhaseVector& true_transformation,
    const std::vector<Example>& training_examples,
    const std::vector<Example>& evaluation_examples,
    std::unique_ptr<learning::LocalUpdateStrategy> strategy
) {
    const std::string strategy_name(strategy->name());
    core::ResonantFabric fabric(
        fabric_config(config.dimension),
        std::make_unique<retrieval::ExactModeRetriever>(),
        std::make_unique<core::StableCircularSettlingPolicy>(),
        std::move(strategy)
    );
    fabric.add_mode(core::ResonantMode(
        1ULL,
        training_examples.front().input,
        core::PhaseVector::zeros(config.dimension),
        0.05F,
        1.0F,
        0.0F,
        0ULL
    ));

    const auto [initial_similarity, initial_accuracy] =
        evaluate(fabric, evaluation_examples);
    static_cast<void>(initial_accuracy);

    learning::LocalLearningConfig learning_config;
    learning_config.transformation_learning_rate = 0.35;
    learning_config.context_learning_rate = 0.0;
    learning_config.confidence_learning_rate = 0.05;
    learning_config.utility_learning_rate = 0.05;
    learning_config.utility_influence = 0.25;
    learning_config.harmful_update_multiplier = 1.25;
    learning_config.use_utility = true;

    double training_error_total = 0.0;
    double training_cycles_total = 0.0;
    std::size_t update_operations = 0U;
    const auto training_start = std::chrono::steady_clock::now();
    for (const Example& example : training_examples) {
        const core::LearningResult update = fabric.learn(
            example.input,
            example.target,
            learning_config
        );
        training_error_total += update.prediction_error;
        training_cycles_total +=
            static_cast<double>(update.prediction.cycles);
        update_operations += update.updated_mode_ids.size();
    }
    const auto training_end = std::chrono::steady_clock::now();

    const auto inference_start = std::chrono::steady_clock::now();
    const auto [final_similarity, final_accuracy] =
        evaluate(fabric, evaluation_examples);
    const auto inference_end = std::chrono::steady_clock::now();

    const double training_seconds =
        std::chrono::duration<double>(training_end - training_start).count();
    const double inference_seconds =
        std::chrono::duration<double>(inference_end - inference_start).count();

    std::uint64_t run_hash = fnv_offset_basis;
    hash_string(run_hash, strategy_name);
    hash_phase_vector(run_hash, true_transformation);
    hash_phase_vector(run_hash, fabric.modes().front().transformation);
    hash_u64(
        run_hash,
        std::bit_cast<std::uint64_t>(initial_similarity)
    );
    hash_u64(run_hash, std::bit_cast<std::uint64_t>(final_similarity));
    hash_u64(run_hash, std::bit_cast<std::uint64_t>(final_accuracy));
    hash_u64(run_hash, static_cast<std::uint64_t>(update_operations));

    const double training_count =
        static_cast<double>(training_examples.size());
    const double evaluation_count =
        static_cast<double>(evaluation_examples.size());
    return {
        .strategy = strategy_name,
        .initial_mean_similarity = initial_similarity,
        .final_mean_similarity = final_similarity,
        .final_task_accuracy = final_accuracy,
        .mean_training_error = training_error_total / training_count,
        .average_settling_cycles = training_cycles_total / training_count,
        .update_operations_per_example =
            static_cast<double>(update_operations) / training_count,
        .training_seconds = training_seconds,
        .inference_seconds = inference_seconds,
        .training_examples_per_second =
            training_seconds > 0.0
                ? training_count / training_seconds
                : std::numeric_limits<double>::infinity(),
        .inference_examples_per_second =
            inference_seconds > 0.0
                ? evaluation_count / inference_seconds
                : std::numeric_limits<double>::infinity(),
        .modes_created = 1U,
        .deterministic_run_hash = run_hash,
    };
}

}  // namespace

TransformationLearningResult run_transformation_learning(
    const TransformationLearningConfig& config
) {
    if (config.dimension == 0U ||
        config.training_examples == 0U ||
        config.evaluation_examples == 0U) {
        throw std::invalid_argument(
            "transformation experiment dimensions and counts must be positive"
        );
    }

    core::DeterministicRng random_number_generator(config.seed);
    const core::PhaseVector true_transformation =
        core::PhaseVector::random(
            config.dimension,
            random_number_generator
        );
    const std::vector<Example> training_examples = make_examples(
        config.training_examples,
        config.dimension,
        true_transformation,
        random_number_generator
    );
    const std::vector<Example> evaluation_examples = make_examples(
        config.evaluation_examples,
        config.dimension,
        true_transformation,
        random_number_generator
    );

    TransformationLearningResult result{
        .seed = config.seed,
        .dimension = config.dimension,
        .training_examples = config.training_examples,
        .evaluation_examples = config.evaluation_examples,
        .strategies = {},
        .tasks = {},
        .supported_unseen_accuracy = 0.0,
        .unsupported_unseen_accuracy = 0.0,
        .deterministic_run_hash = fnv_offset_basis,
    };
    for (auto& strategy : make_strategies()) {
        result.strategies.push_back(run_strategy(
            config,
            true_transformation,
            training_examples,
            evaluation_examples,
            std::move(strategy)
        ));
    }
    for (const TransformationStrategyResult& strategy : result.strategies) {
        hash_u64(
            result.deterministic_run_hash,
            strategy.deterministic_run_hash
        );
    }
    detail::TransformationTaskSuite task_suite =
        detail::run_transformation_task_suite(config);
    result.tasks = std::move(task_suite.tasks);
    result.supported_unseen_accuracy =
        task_suite.supported_unseen_accuracy;
    result.unsupported_unseen_accuracy =
        task_suite.unsupported_unseen_accuracy;
    hash_u64(
        result.deterministic_run_hash,
        task_suite.deterministic_run_hash
    );
    hash_u64(result.deterministic_run_hash, config.seed);
    hash_u64(
        result.deterministic_run_hash,
        static_cast<std::uint64_t>(config.dimension)
    );
    hash_u64(
        result.deterministic_run_hash,
        static_cast<std::uint64_t>(config.training_examples)
    );
    hash_u64(
        result.deterministic_run_hash,
        static_cast<std::uint64_t>(config.evaluation_examples)
    );
    return result;
}

void write_transformation_learning_json(
    std::ostream& output,
    const TransformationLearningResult& result
) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "{\n"
           << "  \"experiment\": \"transformation_learning\",\n"
           << "  \"status\": \"observed\",\n"
           << "  \"seed\": " << result.seed << ",\n"
           << "  \"dimension\": " << result.dimension << ",\n"
           << "  \"training_examples\": "
           << result.training_examples << ",\n"
           << "  \"evaluation_examples\": "
           << result.evaluation_examples << ",\n"
           << "  \"task_success_similarity\": "
           << task_success_similarity << ",\n"
           << "  \"strategies\": [\n";
    for (std::size_t strategy_index = 0U;
         strategy_index < result.strategies.size();
         ++strategy_index) {
        const TransformationStrategyResult& strategy =
            result.strategies[strategy_index];
        output
            << "    {\n"
            << "      \"strategy\": \"" << strategy.strategy << "\",\n"
            << "      \"initial_mean_similarity\": "
            << strategy.initial_mean_similarity << ",\n"
            << "      \"final_mean_similarity\": "
            << strategy.final_mean_similarity << ",\n"
            << "      \"final_task_accuracy\": "
            << strategy.final_task_accuracy << ",\n"
            << "      \"mean_training_error\": "
            << strategy.mean_training_error << ",\n"
            << "      \"average_settling_cycles\": "
            << strategy.average_settling_cycles << ",\n"
            << "      \"update_operations_per_example\": "
            << strategy.update_operations_per_example << ",\n"
            << "      \"training_seconds\": "
            << strategy.training_seconds << ",\n"
            << "      \"inference_seconds\": "
            << strategy.inference_seconds << ",\n"
            << "      \"training_examples_per_second\": "
            << strategy.training_examples_per_second << ",\n"
            << "      \"inference_examples_per_second\": "
            << strategy.inference_examples_per_second << ",\n"
            << "      \"modes_created\": "
            << strategy.modes_created << ",\n"
            << "      \"deterministic_run_hash\": \""
            << format_run_hash(strategy.deterministic_run_hash)
            << "\"\n"
            << "    }";
        if (strategy_index + 1U != result.strategies.size()) {
            output << ',';
        }
        output << '\n';
    }
    output
        << "  ],\n"
        << "  \"supported_unseen_accuracy\": "
        << result.supported_unseen_accuracy << ",\n"
        << "  \"unsupported_unseen_accuracy\": "
        << result.unsupported_unseen_accuracy << ",\n"
        << "  \"tasks\": [\n";
    for (std::size_t task_index = 0U;
         task_index < result.tasks.size();
         ++task_index) {
        const TransformationTaskResult& task =
            result.tasks[task_index];
        const auto write_system = [&output](
            const TransformationTaskSystemResult& system,
            const std::size_t indentation
        ) {
            const std::string indent(indentation, ' ');
            const std::string field_indent(indentation + 2U, ' ');
            output
                << indent << "{\n"
                << field_indent << "\"system\": \""
                << system.system << "\",\n"
                << field_indent
                << "\"familiar_mean_similarity\": "
                << system.familiar_mean_similarity << ",\n"
                << field_indent << "\"familiar_accuracy\": "
                << system.familiar_accuracy << ",\n"
                << field_indent << "\"unseen_mean_similarity\": "
                << system.unseen_mean_similarity << ",\n"
                << field_indent << "\"unseen_accuracy\": "
                << system.unseen_accuracy << ",\n"
                << field_indent << "\"learned_units\": "
                << system.learned_units << ",\n"
                << field_indent << "\"metrics\": ";
            write_metrics_json(
                output,
                system.metrics,
                indentation + 2U
            );
            output << '\n' << indent << '}';
        };
        output
            << "    {\n"
            << "      \"name\": \"" << task.name << "\",\n"
            << "      \"representation\": \""
            << task.representation << "\",\n"
            << "      \"expected_unseen_generalization\": "
            << (task.expected_unseen_generalization
                    ? "true"
                    : "false")
            << ",\n"
            << "      \"rlf\": ";
        write_system(task.rlf, 6U);
        output << ",\n      \"baseline\": ";
        write_system(task.baseline, 6U);
        output << "\n    }";
        if (task_index + 1U != result.tasks.size()) {
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
            "failed to write transformation-learning result"
        );
    }
}

}  // namespace rlf::experiments
