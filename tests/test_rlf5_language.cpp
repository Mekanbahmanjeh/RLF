#include "test_framework.hpp"

#include "rlf/experiments/rlf5_language.hpp"
#include "rlf/storage/rlf5_checkpoint.hpp"

#include <filesystem>
#include <sstream>

namespace {

rlf::experiments::Rlf5Config compact_config() {
    rlf::experiments::Rlf5Config config;
    config.seed = 0x524C463554455354ULL;
    config.phase_dimension = 16U;
    config.raw_training_sentences = 1'200U;
    config.supervised_training_examples = 1'000U;
    config.evaluation_examples = 120U;
    config.qa_episodes = 40U;
    config.free_generation_samples = 16U;
    config.maximum_lexemes = 640U;
    config.maximum_merges = 180U;
    config.minimum_pair_support = 3U;
    config.maximum_context_order = 6U;
    config.minimum_context_support = 1U;
    config.maximum_constructions = 1'024U;
    config.minimum_construction_support = 2U;
    config.maximum_generation_tokens = 64U;
    config.holdout_modulus = 7U;
    return config;
}

}  // namespace

RLF_TEST_CASE("RLF-5 controlled language experiment is deterministic and leakage-audited") {
    const auto config = compact_config();
    const auto first = rlf::experiments::run_rlf5_language(config);
    const auto second = rlf::experiments::run_rlf5_language(config);
    RLF_CHECK(first.deterministic_run_hash == second.deterministic_run_hash);
    RLF_CHECK(first.leakage_audit.evaluation_frames_withheld);
    RLF_CHECK(first.leakage_audit.exact_evaluation_text_withheld);
    RLF_CHECK(first.leakage_audit.train_evaluation_hashes_disjoint);
    RLF_CHECK(first.segmentation.exact_roundtrip_rate == 1.0);
    RLF_CHECK(first.learned_concepts > 0U);
    RLF_CHECK(first.learned_constructions > 0U);
    RLF_CHECK(first.semantics.frame_exact_accuracy >
        first.semantics.nearest_example_accuracy);
    RLF_CHECK(first.generation.semantic_roundtrip_accuracy > 0.50);
    RLF_CHECK(first.question_answering.answer_semantic_accuracy > 0.50);

    std::ostringstream json;
    rlf::experiments::write_rlf5_result_json(json, first);
    RLF_CHECK(json.str().find("\"architecture\": \"RLF-5\"") != std::string::npos);
}

RLF_TEST_CASE("RLF-5 train evaluate and trace workflows use checkpoint version 7") {
    const auto config = compact_config();
    const auto path = std::filesystem::temp_directory_path() /
        "rlf5_language_workflow_test.rlf";
    const auto training = rlf::experiments::train_rlf5_checkpoint(config, path);
    RLF_CHECK(training.lexemes > 256U);
    RLF_CHECK(training.concepts > 0U);
    const auto summary = rlf::storage::inspect_rlf5_checkpoint(path);
    RLF_CHECK(summary.format_version == 7U);

    const auto evaluation = rlf::experiments::evaluate_rlf5_checkpoint(
        path, config.seed + 1U, 96U
    );
    RLF_CHECK(evaluation.semantics.frame_exact_accuracy > 0.50);
    RLF_CHECK(evaluation.generation.semantic_roundtrip_accuracy > 0.50);
    RLF_CHECK(evaluation.question_answering.answer_semantic_accuracy > 0.50);

    const auto trace = rlf::experiments::trace_rlf5_checkpoint(
        path, config.seed + 2U, 3U
    );
    RLF_CHECK(!trace.statement.empty());
    RLF_CHECK(!trace.statement_tokens.empty());
    RLF_CHECK(trace.statement_parse.success);
    RLF_CHECK(trace.question_parse.success);
    RLF_CHECK(trace.answer.success);

    std::filesystem::remove(path);
}
