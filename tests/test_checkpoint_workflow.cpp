#include "test_framework.hpp"

#include "rlf/experiments/checkpoint_workflow.hpp"
#include "rlf/storage/checkpoint.hpp"

#include <filesystem>

RLF_TEST_CASE("checkpoint workflow trains evaluates and traces") {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        "rlf-checkpoint-workflow.rlf";
    std::filesystem::remove(path);

    const rlf::experiments::CheckpointTrainingResult trained =
        rlf::experiments::train_checkpoint_workflow({
            .seed = 0x7A11ULL,
            .dimension = 64U,
            .training_examples = 48U,
            .evaluation_examples = 24U,
            .checkpoint_path = path,
        });
    RLF_CHECK(trained.final_accuracy > 0.99);
    RLF_CHECK(trained.final_mean_similarity >
              trained.initial_mean_similarity);
    RLF_CHECK(std::filesystem::exists(path));

    const rlf::experiments::CheckpointEvaluationResult evaluated =
        rlf::experiments::evaluate_checkpoint_workflow(
            path,
            rlf::experiments::checkpoint_workflow_task,
            24U
        );
    RLF_CHECK(evaluated.accuracy > 0.99);

    const rlf::experiments::CheckpointTraceResult trace =
        rlf::experiments::trace_checkpoint_workflow(
            path,
            rlf::experiments::checkpoint_workflow_task,
            3U
        );
    RLF_CHECK(trace.target_similarity > 0.99);
    RLF_CHECK(trace.cycles.size() == 1U);
    RLF_CHECK(trace.cycles.front().active_mode_ids.size() == 1U);

    const rlf::storage::CheckpointSummary summary =
        rlf::storage::inspect_checkpoint(path);
    RLF_CHECK(summary.training_step == 48ULL);
    RLF_CHECK(summary.mode_count == 1U);
    std::filesystem::remove(path);
}

RLF_TEST_CASE("checkpoint workflow rejects mismatched task") {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        "rlf-checkpoint-workflow-task.rlf";
    std::filesystem::remove(path);
    static_cast<void>(
        rlf::experiments::train_checkpoint_workflow({
            .seed = 0x7A12ULL,
            .dimension = 32U,
            .training_examples = 24U,
            .evaluation_examples = 8U,
            .checkpoint_path = path,
        })
    );
    RLF_CHECK_THROWS_AS(
        rlf::experiments::evaluate_checkpoint_workflow(
            path,
            "unsupported",
            8U
        ),
        std::invalid_argument
    );
    std::filesystem::remove(path);
}
