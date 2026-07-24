#pragma once

#include "rlf/core/language_fabric.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

namespace rlf::experiments {

struct Rlf5Config final {
    std::uint64_t seed{0x524C4635ULL};
    std::size_t phase_dimension{64U};
    std::size_t raw_training_sentences{12'000U};
    std::size_t supervised_training_examples{6'000U};
    std::size_t evaluation_examples{1'200U};
    std::size_t qa_episodes{400U};
    std::size_t free_generation_samples{128U};
    std::size_t maximum_lexemes{1'024U};
    std::size_t maximum_merges{512U};
    std::size_t minimum_pair_support{6U};
    std::size_t maximum_context_order{8U};
    std::size_t minimum_context_support{2U};
    std::size_t maximum_constructions{4'096U};
    std::size_t minimum_construction_support{3U};
    std::size_t maximum_generation_tokens{96U};
    std::size_t holdout_modulus{7U};
};

struct Rlf5SegmentationMetrics final {
    std::size_t raw_bytes{};
    std::size_t encoded_tokens{};
    std::size_t learned_lexemes{};
    std::size_t learned_merges{};
    double compression_ratio{};
    double boundary_precision{};
    double boundary_recall{};
    double boundary_f1{};
    double exact_roundtrip_rate{};
};

struct Rlf5LanguageModelMetrics final {
    std::size_t predictions{};
    double top1_accuracy{};
    double negative_log_likelihood{};
    double perplexity{};
    double bits_per_byte{};
    double char_ngram_bits_per_byte{};
    double oracle_word_ngram_bits_per_byte{};
    double free_generation_parse_rate{};
    double free_generation_unique_rate{};
    double mean_generated_bytes{};
};

struct Rlf5SemanticMetrics final {
    std::size_t examples{};
    double frame_exact_accuracy{};
    double act_accuracy{};
    double role_accuracy{};
    double construction_coverage{};
    double nearest_example_accuracy{};
    double bag_of_words_accuracy{};
    double unseen_composition_accuracy{};
};

struct Rlf5GenerationMetrics final {
    std::size_t examples{};
    double generation_success_rate{};
    double semantic_roundtrip_accuracy{};
    double exact_surface_match_rate{};
    double novel_sentence_rate{};
    double lexical_diversity{};
};

struct Rlf5QuestionAnswerMetrics final {
    std::size_t episodes{};
    double answer_success_rate{};
    double answer_semantic_accuracy{};
    double answer_text_roundtrip_accuracy{};
    double distractor_robustness{};
    double nearest_fact_baseline_accuracy{};
    double question_parse_rate{};
    double question_frame_exact_rate{};
    double target_fact_parse_rate{};
    double agent_query_accuracy{};
    double patient_query_accuracy{};
    double location_query_accuracy{};
};

struct Rlf5LeakageAudit final {
    bool raw_corpus_contains_no_semantic_labels{true};
    bool evaluation_frames_withheld{true};
    bool exact_evaluation_text_withheld{true};
    bool train_evaluation_hashes_disjoint{true};
    bool generation_not_cached{true};
    std::uint64_t raw_training_hash{};
    std::uint64_t supervised_training_hash{};
    std::uint64_t evaluation_hash{};
    std::size_t exact_text_overlap{};
    std::size_t exact_frame_overlap{};
};

struct Rlf5Result final {
    std::uint64_t seed{};
    std::size_t phase_dimension{};
    std::size_t raw_training_sentences{};
    std::size_t supervised_training_examples{};
    Rlf5SegmentationMetrics segmentation;
    Rlf5LanguageModelMetrics language_model;
    Rlf5SemanticMetrics semantics;
    Rlf5GenerationMetrics generation;
    Rlf5QuestionAnswerMetrics question_answering;
    Rlf5LeakageAudit leakage_audit;
    core::LanguageFabricStats training_stats;
    std::size_t learned_concepts{};
    std::size_t learned_constructions{};
    std::size_t estimated_model_bytes{};
    double training_seconds{};
    std::uint64_t deterministic_run_hash{};
    std::string scientific_decision;
    std::vector<std::string> limitations;
};

struct Rlf5TrainingWorkflowResult final {
    std::filesystem::path checkpoint_path;
    std::uint64_t seed{};
    std::size_t raw_training_sentences{};
    std::size_t lexemes{};
    std::size_t contexts{};
    std::size_t concepts{};
    std::size_t constructions{};
    std::uint64_t deterministic_run_hash{};
};

struct Rlf5EvaluationWorkflowResult final {
    std::filesystem::path checkpoint_path;
    Rlf5SemanticMetrics semantics;
    Rlf5GenerationMetrics generation;
    Rlf5QuestionAnswerMetrics question_answering;
    double language_bits_per_byte{};
    std::uint64_t deterministic_run_hash{};
};

struct Rlf5TraceWorkflowResult final {
    std::filesystem::path checkpoint_path;
    std::size_t sample_id{};
    std::string statement;
    std::vector<std::uint64_t> statement_tokens;
    core::LanguageParse statement_parse;
    std::string generated_statement;
    std::string question;
    core::LanguageParse question_parse;
    core::LanguageAnswer answer;
};

[[nodiscard]] Rlf5Result run_rlf5_language(const Rlf5Config& config);
void write_rlf5_result_json(std::ostream& output, const Rlf5Result& result);

[[nodiscard]] Rlf5TrainingWorkflowResult train_rlf5_checkpoint(
    const Rlf5Config& config,
    const std::filesystem::path& checkpoint_path
);
[[nodiscard]] Rlf5EvaluationWorkflowResult evaluate_rlf5_checkpoint(
    const std::filesystem::path& checkpoint_path,
    std::uint64_t seed,
    std::size_t evaluation_examples
);
[[nodiscard]] Rlf5TraceWorkflowResult trace_rlf5_checkpoint(
    const std::filesystem::path& checkpoint_path,
    std::uint64_t seed,
    std::size_t sample_id
);
void write_rlf5_training_json(
    std::ostream& output,
    const Rlf5TrainingWorkflowResult& result
);
void write_rlf5_evaluation_json(
    std::ostream& output,
    const Rlf5EvaluationWorkflowResult& result
);
void write_rlf5_trace_json(
    std::ostream& output,
    const Rlf5TraceWorkflowResult& result
);

}  // namespace rlf::experiments
