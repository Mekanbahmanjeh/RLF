#include "test_framework.hpp"

#include "rlf/experiments/rlf1_latent_routing.hpp"

#include <filesystem>
#include <sstream>
#include <string>

RLF_TEST_CASE("RLF-1 benchmark enforces leakage boundaries") {
    const rlf::experiments::Rlf1Result result =
        rlf::experiments::run_rlf1_latent_routing({
            .seed = 0xABCDEFULL,
            .dimension = 12U,
            .training_episodes = 12U,
            .evaluation_episodes = 8U,
            .training_min_route_length = 1U,
            .training_max_route_length = 3U,
            .evaluation_min_route_length = 4U,
            .evaluation_max_route_length = 5U,
            .maximum_cycles = 10U,
            .operator_count = 8U,
            .state_noise_radians = 0.01,
            .goal_similarity_threshold = 0.999,
        });
    RLF_CHECK(result.leakage_audit.no_context_prefix);
    RLF_CHECK(result.leakage_audit.no_complete_route_overlap);
    RLF_CHECK(result.leakage_audit.no_exact_start_goal_overlap);
    RLF_CHECK(result.leakage_audit.route_hash_overlap == 0U);
    RLF_CHECK(result.leakage_audit.start_goal_hash_overlap == 0U);
    RLF_CHECK(result.leakage_audit.no_seed_overlap);
    RLF_CHECK(result.leakage_audit.no_training_evaluation_length_overlap);
    RLF_CHECK(result.operator_count >= 6U);
    RLF_CHECK(result.halt_policies.size() == 3U);
    RLF_CHECK(result.training_route_hash_values.size() == 12U);
    RLF_CHECK(result.evaluation_route_hash_values.size() == 8U);
    RLF_CHECK(result.macros_proposed >= result.macros_created);
    RLF_CHECK(!result.rlf_without_route_memory.name.empty());
    RLF_CHECK(!result.scientific_decision.empty());
    if (!result.representative_trace.trace.empty()) {
        const auto& step = result.representative_trace.trace.front();
        RLF_CHECK(step.state_hash != 0U);
        RLF_CHECK(step.working_state_hash != 0U);
        RLF_CHECK(step.goal_state_hash != 0U);
        RLF_CHECK(step.memory_summary_hash != 0U);
        RLF_CHECK(step.route_summary_hash != 0U);
    }

    std::ostringstream output;
    rlf::experiments::write_rlf1_latent_routing_json(output, result);
    const std::string json = output.str();
    RLF_CHECK(json.find("\"architecture\"") != std::string::npos);
    RLF_CHECK(json.find("\"leakage_audit\"") != std::string::npos);
    RLF_CHECK(json.find("\"scientific_decision\"") != std::string::npos);
    RLF_CHECK(json.find("\"dataset_manifest\"") != std::string::npos);
    RLF_CHECK(json.find("\"halt_policies\"") != std::string::npos);
    RLF_CHECK(json.find("\"expected_calibration_error\"") != std::string::npos);
}

RLF_TEST_CASE("RLF-1 train evaluate and trace workflows persist version three state") {
    const std::filesystem::path checkpoint =
        std::filesystem::temp_directory_path() / "rlf1_workflow_test.rlf";
    const rlf::experiments::Rlf1Config config{
        .seed = 0x12345678ULL,
        .dimension = 8U,
        .training_episodes = 8U,
        .evaluation_episodes = 4U,
        .training_min_route_length = 1U,
        .training_max_route_length = 2U,
        .evaluation_min_route_length = 3U,
        .evaluation_max_route_length = 3U,
        .maximum_cycles = 6U,
        .operator_count = 8U,
        .state_noise_radians = 0.01,
        .goal_similarity_threshold = 0.999,
    };
    const auto trained = rlf::experiments::train_rlf1_checkpoint(
        config, checkpoint);
    RLF_CHECK(std::filesystem::exists(checkpoint));
    RLF_CHECK(trained.operators == 8U);
    const auto evaluated = rlf::experiments::evaluate_rlf1_checkpoint(
        checkpoint, 0x9876ULL, 4U);
    RLF_CHECK(evaluated.evaluation_episodes == 4U);
    const auto trace = rlf::experiments::trace_rlf1_checkpoint(
        checkpoint, 0x9876ULL, 1U);
    RLF_CHECK(trace.sample_id == 1U);
    RLF_CHECK(trace.route_hash != 0U);
    std::filesystem::remove(checkpoint);
}
