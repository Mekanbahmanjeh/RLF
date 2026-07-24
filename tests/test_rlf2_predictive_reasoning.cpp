#include "test_framework.hpp"

#include "rlf/experiments/rlf2_predictive_reasoning.hpp"

RLF_TEST_CASE("RLF-2 predictive subgoal bridge solves withheld compositions") {
    const rlf::experiments::Rlf2Config config{
        .seed = 0x524C463254455354ULL,
        .dimension = 12U,
        .training_episodes = 32U,
        .development_episodes = 8U,
        .evaluation_episodes = 12U,
        .training_min_route_length = 1U,
        .training_max_route_length = 3U,
        .evaluation_min_route_length = 4U,
        .evaluation_max_route_length = 5U,
        .maximum_cycles = 12U,
        .operator_count = 6U,
        .state_noise_radians = 0.02,
        .goal_similarity_threshold = 0.9995,
    };
    const auto first =
        rlf::experiments::run_rlf2_predictive_reasoning(config);
    const auto second =
        rlf::experiments::run_rlf2_predictive_reasoning(config);
    RLF_CHECK(first.rlf2_subgoal_bridge.final_state_accuracy > 0.90);
    RLF_CHECK(first.skill_validation_accuracy > 0.99);
    RLF_CHECK(first.leakage_audit.route_overlap == 0U);
    RLF_CHECK(first.leakage_audit.start_goal_overlap == 0U);
    RLF_CHECK(first.route_cycle_compression >= 0.0);
    RLF_CHECK(first.deterministic_run_hash == second.deterministic_run_hash);
}
