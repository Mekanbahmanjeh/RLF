#include "test_framework.hpp"

#include "rlf/core/latent_routing.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

[[nodiscard]] rlf::core::TransformationOperator shift_operator(
    const std::size_t dimension,
    const float shift
) {
    std::vector<float> values(dimension, 0.0F);
    for (std::size_t index = 0U; index < dimension; ++index) {
        values[index] = shift * static_cast<float>((index % 5U) + 1U);
    }
    return rlf::core::TransformationOperator(
        dimension,
        {rlf::core::OperatorPrimitive::shift(
            rlf::core::PhaseVector(std::move(values))
        )}
    );
}

}  // namespace

RLF_TEST_CASE("latent routing learns a multi-step route without a context prefix") {
    constexpr std::size_t dimension = 12U;
    rlf::core::LatentRouter router(
        {
            .dimension = dimension,
            .maximum_cycles = 8U,
            .search_node_budget = 1'000U,
            .goal_similarity_threshold = 0.9999,
            .goal_progress_weight = 0.5,
            .enable_route_memory = false,
            .enable_macro_operators = false,
        },
        123ULL
    );
    const std::uint64_t forward = router.register_operator(
        "forward",
        shift_operator(dimension, 0.25F)
    );
    static_cast<void>(router.register_operator(
        "backward",
        shift_operator(dimension, -0.25F)
    ));
    const rlf::core::PhaseVector start =
        rlf::core::PhaseVector::zeros(dimension);
    const rlf::core::PhaseVector goal =
        router.operator_by_id(forward).transformation.apply(
            router.operator_by_id(forward).transformation.apply(start)
        );

    const rlf::core::LatentTrainingResult training =
        router.train_episode(start, goal, 3U);
    RLF_CHECK(training.success);
    RLF_CHECK(training.discovered_route.size() == 2U);
    RLF_CHECK(router.modes().size() >= 1U);

    const rlf::core::LatentExecutionResult result =
        router.execute(start, goal, std::nullopt, false);
    RLF_CHECK(result.success);
    RLF_CHECK(result.cycles == 2U);
    RLF_CHECK(result.final_goal_similarity >= 0.9999);
}

RLF_TEST_CASE("latent routing eligibility assigns delayed utility") {
    constexpr std::size_t dimension = 8U;
    rlf::core::LatentRouter router(
        {
            .dimension = dimension,
            .maximum_cycles = 6U,
            .credit_strategy =
                rlf::core::LatentCreditStrategy::discounted_eligibility,
        },
        456ULL
    );
    const std::uint64_t first = router.register_operator(
        "first",
        shift_operator(dimension, 0.1F)
    );
    const std::uint64_t second = router.register_operator(
        "second",
        shift_operator(dimension, 0.2F)
    );
    const rlf::core::PhaseVector start =
        rlf::core::PhaseVector::zeros(dimension);
    const std::vector<std::uint64_t> route{first, second};
    const rlf::core::PhaseVector goal =
        router.operator_by_id(second).transformation.apply(
            router.operator_by_id(first).transformation.apply(start)
        );
    router.reinforce_route(start, goal, route, 1.0);
    RLF_CHECK(router.modes().size() == 2U);
    RLF_CHECK(router.modes()[0U].utility > 0.0);
    RLF_CHECK(router.modes()[1U].utility > 0.0);
    RLF_CHECK(router.halt_modes().size() == 1U);
}

RLF_TEST_CASE("latent routing snapshot round trip preserves behavior") {
    constexpr std::size_t dimension = 8U;
    rlf::core::LatentRouter router(
        {.dimension = dimension, .maximum_cycles = 4U},
        789ULL
    );
    const std::uint64_t action = router.register_operator(
        "action",
        shift_operator(dimension, 0.15F)
    );
    const rlf::core::PhaseVector start =
        rlf::core::PhaseVector::zeros(dimension);
    const rlf::core::PhaseVector goal =
        router.operator_by_id(action).transformation.apply(start);
    router.reinforce_route(start, goal, std::vector<std::uint64_t>{action}, 1.0);

    rlf::core::LatentRouter restored =
        rlf::core::LatentRouter::from_snapshot(router.snapshot());
    const auto original = router.execute(start, goal);
    const auto replay = restored.execute(start, goal);
    RLF_CHECK(original.success == replay.success);
    RLF_CHECK(original.route == replay.route);
    RLF_CHECK_NEAR(
        original.final_goal_similarity,
        replay.final_goal_similarity,
        1.0e-12
    );
}

RLF_TEST_CASE("latent routing bounded lookahead reports bounded search work") {
    constexpr std::size_t dimension = 10U;
    rlf::core::LatentRouter router(
        {
            .dimension = dimension,
            .maximum_cycles = 8U,
            .search_node_budget = 2'000U,
            .search_beam_width = 4U,
            .search_lookahead_depth = 2U,
            .enable_route_memory = true,
            .enable_macro_operators = false,
        },
        991ULL
    );
    const std::uint64_t first = router.register_operator(
        "first", shift_operator(dimension, 0.11F));
    const std::uint64_t second = router.register_operator(
        "second", shift_operator(dimension, 0.19F));
    static_cast<void>(router.register_operator(
        "distractor", shift_operator(dimension, -0.17F)));
    const rlf::core::PhaseVector start =
        rlf::core::PhaseVector::zeros(dimension);
    const std::vector<std::uint64_t> route{first, second};
    const rlf::core::PhaseVector goal =
        router.operator_by_id(second).transformation.apply(
            router.operator_by_id(first).transformation.apply(start));
    for (std::size_t repetition = 0U; repetition < 4U; ++repetition) {
        router.reinforce_route(start, goal, route, 1.0);
    }
    const auto result = router.execute_with_bounded_lookahead(
        start, goal, 2U, 4U, true);
    RLF_CHECK(result.search_nodes > 0U);
    RLF_CHECK(result.search_nodes <= router.config().search_node_budget);
    RLF_CHECK(result.exact_similarity_evaluations > 0U);
}
