#include "test_framework.hpp"

#include "rlf/core/deterministic_rng.hpp"
#include "rlf/core/phase_vector.hpp"
#include "rlf/core/resonant_fabric.hpp"
#include "rlf/core/resonant_mode.hpp"
#include "rlf/learning/local_learning.hpp"

#include <cstddef>
#include <memory>

namespace {

[[nodiscard]] rlf::core::FabricConfig one_cycle_config(
    const std::size_t dimension
) {
    rlf::core::SettlingConfig settling;
    settling.candidate_count = 2U;
    settling.active_count = 2U;
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
        .maximum_modes = 8U,
        .settling = settling,
    };
}

}  // namespace

RLF_TEST_CASE("settling captures a complete one-cycle trace") {
    constexpr std::size_t dimension = 16U;
    rlf::core::DeterministicRng random_number_generator(19ULL);
    const rlf::core::PhaseVector input =
        rlf::core::PhaseVector::random(dimension, random_number_generator);
    const rlf::core::PhaseVector transformation =
        rlf::core::PhaseVector::random(dimension, random_number_generator);

    rlf::core::ResonantFabric fabric(one_cycle_config(dimension));
    fabric.add_mode(rlf::core::ResonantMode(
        1ULL,
        input,
        transformation,
        1.0F,
        1.0F
    ));

    const rlf::core::SettleResult result = fabric.settle(input, true);
    RLF_CHECK(result.cycles == 1U);
    RLF_CHECK(result.stopping_reason == rlf::core::StopReason::cycle_limit);
    RLF_CHECK(result.trace.has_value());
    RLF_CHECK(result.trace->cycles.size() == 1U);
    RLF_CHECK(result.trace->cycles[0U].retrieved_mode_ids.size() == 1U);
    RLF_CHECK(result.trace->cycles[0U].active_mode_ids.size() == 1U);
    RLF_CHECK(result.trace->cycles[0U].proposal_weights.size() == 1U);
    RLF_CHECK(result.state.similarity(input.composed(transformation)) >
              0.999999);
}

RLF_TEST_CASE("settling reports no active modes without a learned mode") {
    constexpr std::size_t dimension = 8U;
    rlf::core::ResonantFabric fabric(one_cycle_config(dimension));
    const rlf::core::PhaseVector input =
        rlf::core::PhaseVector::zeros(dimension);

    const rlf::core::SettleResult result = fabric.settle(input, true);
    RLF_CHECK(
        result.stopping_reason == rlf::core::StopReason::no_active_modes
    );
    RLF_CHECK(result.state.similarity(input) == 1.0);
    RLF_CHECK(result.trace->cycles[0U].stopping_reason ==
              rlf::core::StopReason::no_active_modes);
}

RLF_TEST_CASE("settling converges on an identity transformation") {
    constexpr std::size_t dimension = 8U;
    rlf::core::FabricConfig config = one_cycle_config(dimension);
    config.settling.maximum_cycles = 8U;
    config.settling.convergence_tolerance_radians = 1.0e-7;

    rlf::core::ResonantFabric fabric(config);
    const rlf::core::PhaseVector input =
        rlf::core::PhaseVector::zeros(dimension);
    fabric.add_mode(rlf::core::ResonantMode(
        1ULL,
        input,
        rlf::core::PhaseVector::zeros(dimension),
        1.0F,
        1.0F
    ));

    const rlf::core::SettleResult result = fabric.settle(input, true);
    RLF_CHECK(result.cycles == 1U);
    RLF_CHECK(result.stopping_reason == rlf::core::StopReason::converged);
    RLF_CHECK(result.trace.has_value());
    RLF_CHECK(
        result.trace->cycles[0U].stopping_reason ==
        rlf::core::StopReason::converged
    );
    RLF_CHECK(result.state.similarity(input) == 1.0);
}

RLF_TEST_CASE("settling stops at the configured confidence threshold") {
    constexpr std::size_t dimension = 8U;
    rlf::core::FabricConfig config = one_cycle_config(dimension);
    config.settling.maximum_cycles = 8U;
    config.settling.confidence_threshold = 0.9;

    rlf::core::ResonantFabric fabric(config);
    const rlf::core::PhaseVector input =
        rlf::core::PhaseVector::zeros(dimension);
    fabric.add_mode(rlf::core::ResonantMode(
        1ULL,
        input,
        rlf::core::PhaseVector(
            std::vector<float>(dimension, 0.25F)
        ),
        1.0F,
        1.0F
    ));

    const rlf::core::SettleResult result = fabric.settle(input, true);
    RLF_CHECK(result.cycles == 1U);
    RLF_CHECK(
        result.stopping_reason ==
        rlf::core::StopReason::confidence_threshold
    );
    RLF_CHECK(result.confidence >= 0.9);
    RLF_CHECK(
        result.trace->cycles[0U].stopping_reason ==
        rlf::core::StopReason::confidence_threshold
    );
}

RLF_TEST_CASE("explicit halt condition stops recurrent settling") {
    constexpr std::size_t dimension = 8U;
    rlf::core::FabricConfig config = one_cycle_config(dimension);
    config.settling.maximum_cycles = 8U;
    config.settling.convergence_tolerance_radians = 0.0;

    rlf::core::ResonantFabric fabric(config);
    const rlf::core::PhaseVector input =
        rlf::core::PhaseVector::zeros(dimension);
    fabric.add_mode(rlf::core::ResonantMode(
        1ULL,
        input,
        rlf::core::PhaseVector(
            std::vector<float>(dimension, 0.1F)
        ),
        1.0F,
        1.0F
    ));

    const rlf::core::SettleResult result = fabric.settle(
        input,
        true,
        [](const rlf::core::SettlingCycleTrace& cycle) {
            return cycle.cycle_number == 1U;
        }
    );
    RLF_CHECK(result.cycles == 2U);
    RLF_CHECK(
        result.stopping_reason == rlf::core::StopReason::explicit_halt
    );
}

RLF_TEST_CASE("winner-only local learning improves a reusable transformation") {
    constexpr std::size_t dimension = 64U;
    rlf::core::DeterministicRng random_number_generator(88ULL);
    const rlf::core::PhaseVector transformation =
        rlf::core::PhaseVector::random(dimension, random_number_generator);
    const rlf::core::PhaseVector first_input =
        rlf::core::PhaseVector::random(dimension, random_number_generator);

    rlf::core::ResonantFabric fabric(one_cycle_config(dimension));
    fabric.set_update_strategy(
        std::make_unique<rlf::learning::WinnerOnlyUpdate>()
    );
    fabric.add_mode(rlf::core::ResonantMode(
        1ULL,
        first_input,
        rlf::core::PhaseVector::zeros(dimension),
        0.05F,
        1.0F
    ));

    rlf::learning::LocalLearningConfig learning_config;
    learning_config.transformation_learning_rate = 0.5;
    learning_config.context_learning_rate = 0.0;
    for (std::size_t example_index = 0U;
         example_index < 24U;
         ++example_index) {
        const rlf::core::PhaseVector input =
            rlf::core::PhaseVector::random(
                dimension,
                random_number_generator
            );
        const rlf::core::PhaseVector target =
            input.composed(transformation);
        static_cast<void>(
            fabric.learn(input, target, learning_config)
        );
    }

    const rlf::core::PhaseVector held_out =
        rlf::core::PhaseVector::random(dimension, random_number_generator);
    const rlf::core::PhaseVector target =
        held_out.composed(transformation);
    const rlf::core::SettleResult prediction = fabric.settle(held_out);

    RLF_CHECK(prediction.state.similarity(target) > 0.999);
    RLF_CHECK(fabric.modes()[0U].successful_update_count > 0ULL);
    RLF_CHECK(fabric.training_step() == 24ULL);
}

RLF_TEST_CASE("local learning updates only participating modes") {
    constexpr std::size_t dimension = 16U;
    rlf::core::FabricConfig config = one_cycle_config(dimension);
    config.settling.active_count = 1U;
    rlf::core::ResonantFabric fabric(config);
    fabric.set_update_strategy(
        std::make_unique<rlf::learning::WinnerOnlyUpdate>()
    );

    const rlf::core::PhaseVector input =
        rlf::core::PhaseVector::zeros(dimension);
    const rlf::core::PhaseVector target(
        std::vector<float>(dimension, 0.7F)
    );
    fabric.add_mode(rlf::core::ResonantMode(
        1ULL,
        input,
        rlf::core::PhaseVector::zeros(dimension),
        1.0F,
        1.0F
    ));
    fabric.add_mode(rlf::core::ResonantMode(
        2ULL,
        rlf::core::PhaseVector(
            std::vector<float>(dimension, 0.2F)
        ),
        rlf::core::PhaseVector::zeros(dimension),
        1.0F,
        1.0F
    ));

    const rlf::core::LearningResult update = fabric.learn(input, target);
    RLF_CHECK(update.updated_mode_ids.size() == 1U);
    RLF_CHECK(update.updated_mode_ids[0U] == 1ULL);
    RLF_CHECK(
        fabric.modes()[1U].transformation.similarity(
            rlf::core::PhaseVector::zeros(dimension)
        ) == 1.0
    );
}
