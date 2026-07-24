#include "test_framework.hpp"

#include "rlf/core/deterministic_rng.hpp"
#include "rlf/core/predictive_skill_fabric.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

[[nodiscard]] rlf::core::TransformationOperator make_shift(
    const std::size_t dimension,
    const float amount
) {
    return rlf::core::TransformationOperator(
        dimension,
        {rlf::core::OperatorPrimitive::shift(
            rlf::core::PhaseVector(
                std::vector<float>(dimension, amount)
            )
        )}
    );
}

[[nodiscard]] rlf::core::TransformationOperator make_rotation(
    const std::size_t dimension
) {
    std::vector<std::size_t> permutation(dimension);
    for (std::size_t index = 0U; index < dimension; ++index) {
        permutation[index] = (index + 1U) % dimension;
    }
    return rlf::core::TransformationOperator(
        dimension,
        {rlf::core::OperatorPrimitive::permute(std::move(permutation))}
    );
}

[[nodiscard]] rlf::core::PhaseVector apply_route(
    const rlf::core::PredictiveSkillFabric& fabric,
    const rlf::core::PhaseVector& start,
    const std::vector<std::uint64_t>& route
) {
    rlf::core::PhaseVector state = start;
    for (const std::uint64_t operator_id : route) {
        state = fabric.operator_by_id(operator_id).forward.apply(state);
    }
    return state;
}

}  // namespace

RLF_TEST_CASE("predictive subgoal bridge discovers withheld routes") {
    constexpr std::size_t dimension = 12U;
    rlf::core::PredictiveSkillConfig config;
    config.dimension = dimension;
    config.maximum_route_depth = 8U;
    config.planner_node_budget = 100'000U;
    rlf::core::PredictiveSkillFabric fabric(config, 0x524C4632ULL);
    const std::uint64_t shift = fabric.register_operator(
        "shift",
        make_shift(dimension, 0.17F)
    );
    const std::uint64_t inverse_shift = fabric.register_operator(
        "inverse_shift",
        make_shift(dimension, -0.17F)
    );
    const std::uint64_t rotate = fabric.register_operator(
        "rotate",
        make_rotation(dimension)
    );
    static_cast<void>(inverse_shift);

    rlf::core::DeterministicRng rng(0x12345678ULL);
    const rlf::core::PhaseVector start =
        rlf::core::PhaseVector::random(dimension, rng);
    const std::vector<std::uint64_t> hidden_route{
        shift, rotate, shift, rotate, shift
    };
    const rlf::core::PhaseVector goal = apply_route(
        fabric,
        start,
        hidden_route
    );
    std::size_t nodes = 0U;
    const auto planned = fabric.plan_primitive_bridge(
        start,
        goal,
        hidden_route.size(),
        &nodes
    );
    RLF_CHECK(planned.has_value());
    RLF_CHECK(nodes > 0U);
    RLF_CHECK(
        apply_route(fabric, start, *planned).similarity(goal) > 0.99999
    );
}

RLF_TEST_CASE("predictive skill consolidation reduces execution cycles") {
    constexpr std::size_t dimension = 10U;
    rlf::core::PredictiveSkillConfig config;
    config.dimension = dimension;
    config.maximum_route_depth = 8U;
    config.minimum_skill_support = 2U;
    config.maximum_skill_length = 4U;
    rlf::core::PredictiveSkillFabric fabric(config, 0xABCDEFULL);
    const std::uint64_t shift = fabric.register_operator(
        "shift",
        make_shift(dimension, 0.11F)
    );
    const std::uint64_t rotate = fabric.register_operator(
        "rotate",
        make_rotation(dimension)
    );
    const std::vector<std::uint64_t> route{shift, rotate, shift};
    rlf::core::DeterministicRng rng(0x424242ULL);
    for (std::size_t repetition = 0U; repetition < 3U; ++repetition) {
        const rlf::core::PhaseVector start =
            rlf::core::PhaseVector::random(dimension, rng);
        const rlf::core::PhaseVector goal = apply_route(fabric, start, route);
        fabric.observe_successful_route(start, goal, route);
    }
    const std::vector<std::uint64_t> segmented = fabric.segment_route(route);
    RLF_CHECK(segmented.size() < route.size());

    const rlf::core::PhaseVector start =
        rlf::core::PhaseVector::random(dimension, rng);
    const rlf::core::PhaseVector goal = apply_route(fabric, start, route);
    const rlf::core::Rlf2ExecutionResult result =
        fabric.execute_subgoal_bridge(start, goal, route.size());
    RLF_CHECK(result.success);
    RLF_CHECK(result.cycles < result.primitive_steps);
}

RLF_TEST_CASE("predictive snapshot preserves learned skills and profiles") {
    constexpr std::size_t dimension = 8U;
    rlf::core::PredictiveSkillConfig config;
    config.dimension = dimension;
    config.maximum_route_depth = 4U;
    rlf::core::PredictiveSkillFabric fabric(config, 0xC0FFEEULL);
    const std::uint64_t shift = fabric.register_operator(
        "shift",
        make_shift(dimension, 0.09F)
    );
    rlf::core::DeterministicRng rng(0xBAD5EEDULL);
    const rlf::core::PhaseVector start =
        rlf::core::PhaseVector::random(dimension, rng);
    const std::vector<std::uint64_t> route{shift};
    const rlf::core::PhaseVector goal = apply_route(fabric, start, route);
    fabric.observe_successful_route(start, goal, route);
    const std::vector<float> before = fabric.response_profile(start, goal);

    rlf::core::PredictiveSkillFabric restored =
        rlf::core::PredictiveSkillFabric::from_snapshot(fabric.snapshot());
    const std::vector<float> after = restored.response_profile(start, goal);
    RLF_CHECK(before == after);
    RLF_CHECK(restored.skills().size() == fabric.skills().size());
    RLF_CHECK(restored.prototypes().size() == fabric.prototypes().size());
    RLF_CHECK(
        restored.execute_subgoal_bridge(start, goal, 1U).success
    );
}
