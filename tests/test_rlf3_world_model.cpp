#include "test_framework.hpp"

#include "rlf/experiments/rlf3_world_model.hpp"

RLF_TEST_CASE("RLF-3 learns a sparse world model without evaluation leakage") {
    const rlf::experiments::Rlf3Config config{
        .seed = 0x524C463354455354ULL,
        .dimension = 16U,
        .layers = 6U,
        .lanes = 4U,
        .transition_samples_per_case = 6U,
        .training_routes = 80U,
        .evaluation_episodes = 20U,
        .stochastic_rollouts_per_episode = 1U,
        .maximum_execution_steps = 16U,
        .planner_node_budget = 20'000U,
        .maximum_plan_depth = 14U,
        .observation_noise_radians = 0.01,
        .stochastic_dominant_probability = 0.82,
        .goal_similarity_threshold = 0.999,
    };
    const auto first = rlf::experiments::run_rlf3_world_model(config);
    const auto second = rlf::experiments::run_rlf3_world_model(config);
    RLF_CHECK(first.prediction.top1_accuracy > 0.70);
    RLF_CHECK(first.indexed_receding.success_rate > first.greedy.success_rate);
    RLF_CHECK(first.planner_node_reduction > 0.10);
    RLF_CHECK(first.partial_observation_gain > 0.10);
    RLF_CHECK(first.impossible_false_success_rate == 0.0);
    RLF_CHECK(first.leakage_audit.start_goal_overlap == 0U);
    RLF_CHECK(first.leakage_audit.route_overlap == 0U);
    RLF_CHECK(first.deterministic_run_hash == second.deterministic_run_hash);
}
