#include "rlf/experiments/structural_adaptation.hpp"

#include "rlf/core/deterministic_rng.hpp"
#include "rlf/core/phase_vector.hpp"
#include "rlf/core/resonant_fabric.hpp"
#include "rlf/core/resonant_mode.hpp"
#include "rlf/experiments/phase_vector_smoke.hpp"
#include "rlf/learning/local_learning.hpp"
#include "rlf/learning/structural_learning.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <numbers>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rlf::experiments {
namespace {

constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;
constexpr double task_success_similarity = 0.95;

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

[[nodiscard]] core::PhaseVector perturb(
    const core::PhaseVector& prototype,
    const double noise_radians,
    core::DeterministicRng& random_number_generator
) {
    std::vector<float> noise;
    noise.reserve(prototype.size());
    for (std::size_t dimension_index = 0U;
         dimension_index < prototype.size();
         ++dimension_index) {
        const double signed_unit =
            (2.0 * random_number_generator.uniform_unit()) - 1.0;
        noise.push_back(
            static_cast<float>(signed_unit * noise_radians)
        );
    }
    return prototype.composed(core::PhaseVector(std::move(noise)));
}

[[nodiscard]] core::FabricConfig make_fabric_config(
    const StructuralAdaptationConfig& experiment_config
) {
    core::SettlingConfig settling;
    settling.candidate_count = 3U;
    settling.active_count = 1U;
    settling.maximum_cycles = 1U;
    settling.minimum_cycles = 1U;
    settling.minimum_resonance = 0.0;
    settling.convergence_tolerance_radians = 0.0;
    settling.input_weight = 0.0;
    settling.previous_state_weight = 0.0;
    settling.proposal_weight_scale = 1.0;
    settling.utility_weight = 0.0;

    learning::StructuralLearningConfig structural;
    structural.enabled = true;
    structural.creation_minimum_resonance = 0.0;
    structural.creation_prediction_error_threshold = 1.0;
    structural.correction_history_capacity = 24U;
    structural.split_minimum_samples = 12U;
    structural.split_minimum_cluster_size = 4U;
    structural.split_kmeans_iterations = 8U;
    structural.split_minimum_transformation_separation_radians = 0.5;
    structural.split_minimum_context_separation_radians = 0.5;
    structural.split_minimum_validation_gain_radians = 0.1;
    structural.split_context_distance_weight = 0.5;
    structural.merge_maximum_key_error_radians = 0.05;
    structural.merge_maximum_transformation_error_radians = 0.05;
    structural.merge_maximum_history_dispersion_radians = 0.1;
    structural.pruning_minimum_age_steps =
        std::numeric_limits<std::uint64_t>::max();
    structural.pruning_maximum_inactive_steps =
        std::numeric_limits<std::uint64_t>::max();
    structural.pruning_disabled_grace_steps =
        std::numeric_limits<std::uint64_t>::max();
    structural.minimum_retained_modes = 1U;

    return {
        .dimension = experiment_config.dimension,
        .maximum_modes = 16U,
        .settling = settling,
        .structural_learning = structural,
    };
}

struct Evaluation final {
    double mean_similarity;
    double accuracy;
};

[[nodiscard]] Evaluation evaluate(
    core::ResonantFabric& fabric,
    const core::PhaseVector& first_context,
    const core::PhaseVector& second_context,
    const core::PhaseVector& first_transformation,
    const core::PhaseVector& second_transformation,
    const std::size_t example_count,
    const double context_noise_radians,
    core::DeterministicRng& random_number_generator
) {
    double similarity_total = 0.0;
    std::size_t successes = 0U;
    for (std::size_t example_index = 0U;
         example_index < example_count;
         ++example_index) {
        const bool first_task = (example_index % 2U) == 0U;
        const core::PhaseVector input = perturb(
            first_task ? first_context : second_context,
            context_noise_radians,
            random_number_generator
        );
        const core::PhaseVector& transformation =
            first_task ? first_transformation : second_transformation;
        const core::PhaseVector target = input.composed(transformation);
        const core::SettleResult prediction = fabric.settle(input);
        const double similarity = prediction.state.similarity(target);
        similarity_total += similarity;
        if (similarity >= task_success_similarity) {
            ++successes;
        }
    }
    return {
        .mean_similarity =
            similarity_total / static_cast<double>(example_count),
        .accuracy =
            static_cast<double>(successes) /
            static_cast<double>(example_count),
    };
}

}  // namespace

StructuralAdaptationResult run_structural_adaptation(
    const StructuralAdaptationConfig& config
) {
    if (config.dimension == 0U ||
        config.training_examples < 12U ||
        config.evaluation_examples == 0U ||
        !std::isfinite(config.context_noise_radians) ||
        config.context_noise_radians < 0.0 ||
        config.context_noise_radians >= std::numbers::pi_v<double>) {
        throw std::invalid_argument(
            "invalid structural-adaptation experiment configuration"
        );
    }

    core::DeterministicRng random_number_generator(config.seed);
    const core::PhaseVector first_context = core::PhaseVector::random(
        config.dimension,
        random_number_generator
    );
    const core::PhaseVector second_context = core::PhaseVector::random(
        config.dimension,
        random_number_generator
    );
    const core::PhaseVector first_transformation =
        core::PhaseVector::random(
            config.dimension,
            random_number_generator
        );
    const core::PhaseVector second_transformation =
        core::PhaseVector::random(
            config.dimension,
            random_number_generator
        );

    core::ResonantFabric fabric(make_fabric_config(config));
    fabric.set_update_strategy(
        std::make_unique<learning::WinnerOnlyUpdate>()
    );
    fabric.add_mode(core::ResonantMode(
        1ULL,
        first_context,
        core::PhaseVector::zeros(config.dimension),
        1.0F,
        1.0F,
        0.0F,
        0ULL
    ));

    core::DeterministicRng initial_evaluation_rng(
        config.seed ^ 0xA11CEULL
    );
    const Evaluation initial = evaluate(
        fabric,
        first_context,
        second_context,
        first_transformation,
        second_transformation,
        config.evaluation_examples,
        config.context_noise_radians,
        initial_evaluation_rng
    );

    learning::LocalLearningConfig learning_config;
    learning_config.transformation_learning_rate = 0.15;
    learning_config.context_learning_rate = 0.0;
    learning_config.confidence_learning_rate = 0.0;
    learning_config.utility_learning_rate = 0.0;
    for (std::size_t example_index = 0U;
         example_index < config.training_examples;
         ++example_index) {
        const bool first_task = (example_index % 2U) == 0U;
        const core::PhaseVector& input =
            first_task ? first_context : second_context;
        const core::PhaseVector& transformation =
            first_task ? first_transformation : second_transformation;
        const core::PhaseVector target = input.composed(transformation);
        static_cast<void>(
            fabric.learn(input, target, learning_config)
        );
    }

    core::DeterministicRng final_evaluation_rng(
        config.seed ^ 0xF1A1ULL
    );
    const Evaluation final = evaluate(
        fabric,
        first_context,
        second_context,
        first_transformation,
        second_transformation,
        config.evaluation_examples,
        config.context_noise_radians,
        final_evaluation_rng
    );

    StructuralAdaptationResult result{
        .seed = config.seed,
        .dimension = config.dimension,
        .training_examples = config.training_examples,
        .evaluation_examples = config.evaluation_examples,
        .initial_mean_similarity = initial.mean_similarity,
        .final_mean_similarity = final.mean_similarity,
        .final_task_accuracy = final.accuracy,
        .total_modes = fabric.modes().size(),
        .enabled_modes = static_cast<std::size_t>(std::count_if(
            fabric.modes().begin(),
            fabric.modes().end(),
            [](const core::ResonantMode& mode) {
                return mode.enabled;
            }
        )),
        .modes_created =
            fabric.structural_statistics().modes_created,
        .modes_split = fabric.structural_statistics().modes_split,
        .modes_merged = fabric.structural_statistics().modes_merged,
        .modes_pruned = fabric.structural_statistics().modes_pruned,
        .events = {},
        .deterministic_run_hash = fnv_offset_basis,
    };
    result.events.reserve(fabric.structural_events().size());
    for (const learning::StructuralEvent& event :
         fabric.structural_events()) {
        result.events.push_back({
            .type = std::string(learning::to_string(event.type)),
            .step = event.step,
            .primary_mode_id = event.primary_mode_id,
            .related_mode_ids = event.related_mode_ids,
            .reason = event.reason,
            .metric = event.metric,
        });
        hash_string(result.deterministic_run_hash, result.events.back().type);
        hash_u64(result.deterministic_run_hash, event.step);
        hash_u64(
            result.deterministic_run_hash,
            event.primary_mode_id
        );
        for (const std::uint64_t related_id : event.related_mode_ids) {
            hash_u64(result.deterministic_run_hash, related_id);
        }
    }
    hash_phase_vector(result.deterministic_run_hash, first_context);
    hash_phase_vector(result.deterministic_run_hash, second_context);
    hash_phase_vector(result.deterministic_run_hash, first_transformation);
    hash_phase_vector(result.deterministic_run_hash, second_transformation);
    for (const core::ResonantMode& mode : fabric.modes()) {
        hash_u64(result.deterministic_run_hash, mode.id);
        hash_u64(
            result.deterministic_run_hash,
            mode.enabled ? 1ULL : 0ULL
        );
        hash_phase_vector(
            result.deterministic_run_hash,
            mode.context_key
        );
        hash_phase_vector(
            result.deterministic_run_hash,
            mode.transformation
        );
    }
    hash_u64(
        result.deterministic_run_hash,
        std::bit_cast<std::uint64_t>(result.final_mean_similarity)
    );
    hash_u64(
        result.deterministic_run_hash,
        std::bit_cast<std::uint64_t>(result.final_task_accuracy)
    );
    return result;
}

void write_structural_adaptation_json(
    std::ostream& output,
    const StructuralAdaptationResult& result
) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "{\n"
           << "  \"experiment\": \"structural_adaptation\",\n"
           << "  \"status\": \"observed\",\n"
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
           << "  \"final_task_accuracy\": "
           << result.final_task_accuracy << ",\n"
           << "  \"total_modes\": " << result.total_modes << ",\n"
           << "  \"enabled_modes\": " << result.enabled_modes << ",\n"
           << "  \"modes_created\": " << result.modes_created << ",\n"
           << "  \"modes_split\": " << result.modes_split << ",\n"
           << "  \"modes_merged\": " << result.modes_merged << ",\n"
           << "  \"modes_pruned\": " << result.modes_pruned << ",\n"
           << "  \"events\": [\n";
    for (std::size_t event_index = 0U;
         event_index < result.events.size();
         ++event_index) {
        const StructuralEventResult& event = result.events[event_index];
        output << "    {\n"
               << "      \"type\": \"" << event.type << "\",\n"
               << "      \"step\": " << event.step << ",\n"
               << "      \"primary_mode_id\": "
               << event.primary_mode_id << ",\n"
               << "      \"related_mode_ids\": [";
        for (std::size_t id_index = 0U;
             id_index < event.related_mode_ids.size();
             ++id_index) {
            output << event.related_mode_ids[id_index];
            if (id_index + 1U != event.related_mode_ids.size()) {
                output << ", ";
            }
        }
        output << "],\n"
               << "      \"reason\": \"" << event.reason << "\",\n"
               << "      \"metric\": " << event.metric << "\n"
               << "    }";
        if (event_index + 1U != result.events.size()) {
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
            "failed to write structural-adaptation result"
        );
    }
}

}  // namespace rlf::experiments
