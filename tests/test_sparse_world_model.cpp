#include "test_framework.hpp"

#include "rlf/core/deterministic_rng.hpp"
#include "rlf/core/sparse_world_model.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

[[nodiscard]] rlf::core::WorldObservation observation(
    const rlf::core::PhaseVector& visible,
    const rlf::core::PhaseVector& memory
) {
    return {visible, memory};
}

}  // namespace

RLF_TEST_CASE("sparse world model learns stochastic local transitions") {
    constexpr std::size_t dimension = 8U;
    rlf::core::SparseWorldModelConfig config;
    config.dimension = dimension;
    config.hash_dimensions = 3U;
    config.state_merge_distance = 0.05;
    config.context_merge_distance = 0.05;
    config.minimum_transition_support = 1U;
    rlf::core::SparseWorldModel model(config, 0x524C4633ULL);
    const std::uint64_t action = model.register_action("advance");

    rlf::core::DeterministicRng rng(0x12345678ULL);
    const auto state0 = rlf::core::PhaseVector::random(dimension, rng);
    const auto state1 = rlf::core::PhaseVector::random(dimension, rng);
    const auto state2 = rlf::core::PhaseVector::random(dimension, rng);
    const auto memory = rlf::core::PhaseVector::random(dimension, rng);
    for (std::size_t index = 0U; index < 10U; ++index) {
        model.observe_transition({
            observation(state0, memory),
            action,
            observation(index < 8U ? state1 : state2, memory),
            -0.1,
            false,
        });
    }
    const auto prediction = model.predict(observation(state0, memory), action);
    RLF_CHECK(prediction.has_value());
    RLF_CHECK(prediction->outcomes.size() == 2U);
    RLF_CHECK_NEAR(prediction->outcomes.front().probability, 0.8, 1.0e-12);
    const auto expected_state = model.match_state(state1);
    RLF_CHECK(expected_state.has_value());
    RLF_CHECK(prediction->outcomes.front().next_state_id == *expected_state);
}

RLF_TEST_CASE("sparse subgoal index guides a learned transition plan") {
    constexpr std::size_t dimension = 10U;
    rlf::core::SparseWorldModelConfig config;
    config.dimension = dimension;
    config.hash_dimensions = 3U;
    config.minimum_transition_support = 1U;
    config.maximum_plan_depth = 8U;
    config.planner_node_budget = 1'000U;
    rlf::core::SparseWorldModel model(config, 0xABCDEFULL);
    const std::uint64_t advance = model.register_action("advance");
    const std::uint64_t distract = model.register_action("distract");

    rlf::core::DeterministicRng rng(0xCAFEULL);
    const auto state0 = rlf::core::PhaseVector::random(dimension, rng);
    const auto state1 = rlf::core::PhaseVector::random(dimension, rng);
    const auto state2 = rlf::core::PhaseVector::random(dimension, rng);
    const auto dead = rlf::core::PhaseVector::random(dimension, rng);
    const auto memory = rlf::core::PhaseVector::random(dimension, rng);
    for (std::size_t repetition = 0U; repetition < 4U; ++repetition) {
        model.observe_transition({
            observation(state0, memory), advance,
            observation(state1, memory), -0.1, false});
        model.observe_transition({
            observation(state1, memory), advance,
            observation(state2, memory), 1.0, true});
        model.observe_transition({
            observation(state0, memory), distract,
            observation(dead, memory), -1.0, true});
    }
    model.observe_successful_route({
        {observation(state0, memory), observation(state1, memory),
         observation(state2, memory)},
        {advance, advance},
        true,
    });
    const auto indexed = model.plan(observation(state0, memory), state2, true);
    const auto flat = model.plan(observation(state0, memory), state2, false);
    RLF_CHECK(indexed.success);
    RLF_CHECK(flat.success);
    RLF_CHECK(indexed.actions.size() == 2U);
    RLF_CHECK(indexed.actions.front() == advance);
    RLF_CHECK(indexed.nodes_expanded <= flat.nodes_expanded);
    RLF_CHECK(indexed.subgoal_queries > 0U);
}
