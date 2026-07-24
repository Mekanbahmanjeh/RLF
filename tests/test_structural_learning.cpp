#include "test_framework.hpp"

#include "rlf/core/phase_vector.hpp"
#include "rlf/core/resonant_mode.hpp"
#include "rlf/learning/structural_learning.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

[[nodiscard]] rlf::learning::StructuralLearningConfig structural_config() {
    rlf::learning::StructuralLearningConfig config;
    config.enabled = true;
    config.creation_minimum_resonance = 0.2;
    config.creation_prediction_error_threshold = 0.6;
    config.correction_history_capacity = 16U;
    config.split_minimum_samples = 8U;
    config.split_minimum_cluster_size = 4U;
    config.split_kmeans_iterations = 8U;
    config.split_minimum_transformation_separation_radians = 0.5;
    config.split_minimum_context_separation_radians = 0.5;
    config.split_minimum_validation_gain_radians = 0.1;
    config.split_context_distance_weight = 0.25;
    config.merge_maximum_key_error_radians = 0.02;
    config.merge_maximum_transformation_error_radians = 0.02;
    config.merge_maximum_history_dispersion_radians = 0.05;
    config.pruning_minimum_age_steps = 10ULL;
    config.pruning_maximum_inactive_steps = 20ULL;
    config.pruning_minimum_activation_count = 2ULL;
    config.pruning_maximum_utility = 0.0;
    config.pruning_harmful_update_ratio = 0.75;
    config.pruning_disabled_grace_steps = 4ULL;
    config.minimum_retained_modes = 1U;
    return config;
}

[[nodiscard]] std::size_t enabled_count(
    const std::vector<rlf::core::ResonantMode>& modes
) {
    return static_cast<std::size_t>(std::count_if(
        modes.begin(),
        modes.end(),
        [](const rlf::core::ResonantMode& mode) {
            return mode.enabled;
        }
    ));
}

}  // namespace

RLF_TEST_CASE("structural creation responds to low resonance") {
    rlf::learning::StructuralLearner learner(structural_config());
    std::vector<rlf::core::ResonantMode> modes;
    std::uint64_t next_mode_id = 1ULL;
    const rlf::core::PhaseVector input =
        rlf::core::PhaseVector::zeros(8U);
    const rlf::core::PhaseVector transformation(
        std::vector<float>(8U, 0.4F)
    );

    const bool created = learner.consider_creation(
        modes,
        next_mode_id,
        8U,
        input,
        transformation,
        0.1,
        0.1,
        3ULL
    );

    RLF_CHECK(created);
    RLF_CHECK(modes.size() == 1U);
    RLF_CHECK(modes[0U].id == 1ULL);
    RLF_CHECK(next_mode_id == 2ULL);
    RLF_CHECK(learner.statistics().modes_created == 1ULL);
    RLF_CHECK(
        learner.events()[0U].type ==
        rlf::learning::StructuralEventType::created
    );
}

RLF_TEST_CASE("structural splitting validates two incompatible clusters") {
    rlf::learning::StructuralLearningConfig config = structural_config();
    config.merge_maximum_key_error_radians = 0.001;
    rlf::learning::StructuralLearner learner(config);
    std::vector<rlf::core::ResonantMode> modes;
    modes.emplace_back(
        1ULL,
        rlf::core::PhaseVector::zeros(8U),
        rlf::core::PhaseVector::zeros(8U),
        1.0F,
        0.8F
    );
    std::uint64_t next_mode_id = 2ULL;

    const rlf::core::PhaseVector first_context(
        std::vector<float>(8U, 0.1F)
    );
    const rlf::core::PhaseVector second_context(
        std::vector<float>(8U, 2.0F)
    );
    const rlf::core::PhaseVector first_transformation(
        std::vector<float>(8U, 0.25F)
    );
    const rlf::core::PhaseVector second_transformation(
        std::vector<float>(8U, 2.5F)
    );
    for (std::size_t sample_index = 0U;
         sample_index < 4U;
         ++sample_index) {
        learner.record_correction(
            modes[0U],
            first_context,
            first_transformation,
            0.9,
            true,
            static_cast<std::uint64_t>(sample_index + 1U)
        );
        learner.record_correction(
            modes[0U],
            second_context,
            second_transformation,
            0.9,
            true,
            static_cast<std::uint64_t>(sample_index + 5U)
        );
    }

    learner.maintain(modes, next_mode_id, 8U, 10ULL);

    RLF_CHECK(modes.size() == 3U);
    RLF_CHECK(!modes[0U].enabled);
    RLF_CHECK(modes[1U].enabled);
    RLF_CHECK(modes[2U].enabled);
    RLF_CHECK(modes[1U].id == 2ULL);
    RLF_CHECK(modes[2U].id == 3ULL);
    RLF_CHECK(learner.statistics().modes_split == 1ULL);
    RLF_CHECK(learner.statistics().modes_created == 2ULL);
    RLF_CHECK(
        modes[1U].transformation.mean_angular_error(
            modes[2U].transformation
        ) > 1.0
    );
}

RLF_TEST_CASE("structural merging conservatively combines redundant modes") {
    rlf::learning::StructuralLearningConfig config = structural_config();
    config.pruning_minimum_age_steps = 1'000ULL;
    config.pruning_maximum_inactive_steps = 1'000ULL;
    rlf::learning::StructuralLearner learner(config);
    std::vector<rlf::core::ResonantMode> modes;
    modes.emplace_back(
        2ULL,
        rlf::core::PhaseVector(std::vector<float>(8U, 0.1F)),
        rlf::core::PhaseVector(std::vector<float>(8U, 0.2F)),
        1.0F,
        0.7F
    );
    modes.emplace_back(
        1ULL,
        rlf::core::PhaseVector(std::vector<float>(8U, 0.105F)),
        rlf::core::PhaseVector(std::vector<float>(8U, 0.205F)),
        1.0F,
        0.9F
    );
    modes[0U].activation_count = 4ULL;
    modes[1U].activation_count = 8ULL;
    std::uint64_t next_mode_id = 3ULL;

    learner.maintain(modes, next_mode_id, 8U, 5ULL);

    RLF_CHECK(enabled_count(modes) == 1U);
    RLF_CHECK(modes[1U].enabled);
    RLF_CHECK(!modes[0U].enabled);
    RLF_CHECK(modes[1U].activation_count == 12ULL);
    RLF_CHECK(learner.statistics().modes_merged == 1ULL);
}

RLF_TEST_CASE("structural pruning removes old harmful low-value modes") {
    rlf::learning::StructuralLearningConfig config = structural_config();
    config.merge_maximum_key_error_radians = 0.001;
    config.merge_maximum_transformation_error_radians = 0.001;
    rlf::learning::StructuralLearner learner(config);
    std::vector<rlf::core::ResonantMode> modes;
    modes.emplace_back(
        1ULL,
        rlf::core::PhaseVector::zeros(8U),
        rlf::core::PhaseVector::zeros(8U),
        1.0F,
        0.2F,
        -0.5F,
        0ULL
    );
    modes.emplace_back(
        2ULL,
        rlf::core::PhaseVector(std::vector<float>(8U, 1.0F)),
        rlf::core::PhaseVector(std::vector<float>(8U, 1.0F)),
        1.0F,
        0.9F,
        0.8F,
        0ULL
    );
    modes[0U].activation_count = 1ULL;
    modes[0U].unsuccessful_update_count = 9ULL;
    modes[0U].successful_update_count = 1ULL;
    modes[1U].activation_count = 20ULL;
    modes[1U].last_used_step = 29ULL;
    std::uint64_t next_mode_id = 3ULL;

    learner.maintain(modes, next_mode_id, 8U, 30ULL);

    RLF_CHECK(!modes[0U].enabled);
    RLF_CHECK(modes[1U].enabled);
    RLF_CHECK(learner.statistics().modes_pruned == 1ULL);
    RLF_CHECK(
        learner.events().back().type ==
        rlf::learning::StructuralEventType::pruned
    );
}

RLF_TEST_CASE("structural pruning enforces the physical memory budget") {
    rlf::learning::StructuralLearningConfig config = structural_config();
    config.merge_maximum_key_error_radians = 0.001;
    config.merge_maximum_transformation_error_radians = 0.001;
    config.pruning_minimum_age_steps = 1'000ULL;
    config.pruning_maximum_inactive_steps = 1'000ULL;
    config.pruning_harmful_update_ratio = 1.0;
    config.memory_budget_bytes = 1U;
    rlf::learning::StructuralLearner learner(config);
    std::vector<rlf::core::ResonantMode> modes;
    modes.emplace_back(
        1ULL,
        rlf::core::PhaseVector::zeros(8U),
        rlf::core::PhaseVector::zeros(8U),
        1.0F,
        0.2F,
        -0.5F,
        0ULL
    );
    modes.emplace_back(
        2ULL,
        rlf::core::PhaseVector(std::vector<float>(8U, 1.0F)),
        rlf::core::PhaseVector(std::vector<float>(8U, 1.0F)),
        1.0F,
        0.9F,
        0.8F,
        0ULL
    );
    modes[0U].activation_count = 1ULL;
    modes[1U].activation_count = 20ULL;
    std::uint64_t next_mode_id = 3ULL;

    learner.maintain(modes, next_mode_id, 8U, 2ULL);

    RLF_CHECK(enabled_count(modes) == 1U);
    RLF_CHECK(!modes[0U].enabled);
    RLF_CHECK(modes[1U].enabled);
    RLF_CHECK(learner.statistics().modes_pruned == 1ULL);
    RLF_CHECK(
        learner.events().back().reason == "memory_budget_exceeded"
    );
}
