#include "test_framework.hpp"

#include "rlf/experiments/rlf4_self_supervised.hpp"

RLF_TEST_CASE("RLF-4 self-supervised temporal experiment is deterministic") {
    const rlf::experiments::Rlf4Config config{
        .seed = 0x524C463454455354ULL,
        .dimension = 16U,
        .symbol_count = 18U,
        .training_tokens = 3'000U,
        .evaluation_tokens = 1'000U,
        .adaptation_tokens = 1'200U,
        .maximum_context_order = 10U,
        .minimum_context_support = 2U,
        .maximum_options = 256U,
        .minimum_option_support = 4U,
        .forecast_horizon = 6U,
        .forecast_samples = 48U,
        .change_tolerance = 64U,
        .training_noise_radians = 0.012,
        .evaluation_noise_radians = 0.025,
        .dominant_motif_probability = 0.88,
        .prototype_merge_distance = 0.10,
        .recent_decay = 0.997,
        .recent_weight = 0.70,
    };
    const auto first = rlf::experiments::run_rlf4_self_supervised(config);
    const auto second = rlf::experiments::run_rlf4_self_supervised(config);
    RLF_CHECK(first.representation.learned_prototypes >= 18U);
    RLF_CHECK(first.representation.learned_options > 0U);
    RLF_CHECK(first.full_fabric.top1_accuracy > first.fixed_order_1.top1_accuracy);
    RLF_CHECK(first.forecast.decision_reduction > 0.0);
    RLF_CHECK(first.leakage_audit.full_streams_distinct);
    RLF_CHECK(first.deterministic_run_hash == second.deterministic_run_hash);
}

RLF_TEST_CASE("RLF-4 train evaluate and trace workflows round trip") {
    const rlf::experiments::Rlf4Config config{
        .seed = 0x524C4634574F524BULL,
        .dimension = 16U,
        .symbol_count = 18U,
        .training_tokens = 2'000U,
        .evaluation_tokens = 600U,
        .adaptation_tokens = 600U,
        .maximum_context_order = 8U,
        .minimum_context_support = 2U,
        .maximum_options = 128U,
        .minimum_option_support = 4U,
        .forecast_horizon = 6U,
        .forecast_samples = 32U,
        .change_tolerance = 48U,
        .training_noise_radians = 0.012,
        .evaluation_noise_radians = 0.025,
        .dominant_motif_probability = 0.88,
        .prototype_merge_distance = 0.10,
        .recent_decay = 0.997,
        .recent_weight = 0.70,
    };
    const auto checkpoint = std::filesystem::temp_directory_path() /
        "rlf4_workflow_roundtrip.rlf";
    const auto training = rlf::experiments::train_rlf4_checkpoint(
        config, checkpoint
    );
    RLF_CHECK(training.prototypes >= 18U);
    RLF_CHECK(training.contexts > 0U);
    const auto evaluation = rlf::experiments::evaluate_rlf4_checkpoint(
        checkpoint, config.seed, 600U
    );
    RLF_CHECK(evaluation.full_fabric.predictions > 0U);
    RLF_CHECK(evaluation.full_fabric.top1_accuracy > 0.5);
    const auto trace = rlf::experiments::trace_rlf4_checkpoint(
        checkpoint, config.seed, 7U
    );
    RLF_CHECK(!trace.history.empty());
    RLF_CHECK(!trace.prediction.outcomes.empty());
    std::filesystem::remove(checkpoint);
}
