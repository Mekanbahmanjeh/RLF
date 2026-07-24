#include "rlf/solstice/efficiency_campaign.hpp"
#include "rlf/solstice/efficiency_corpus.hpp"

#include <charconv>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

template <typename T>
T parse_number(const std::string_view text, const char* name) {
    T value{};
    const char* const begin = text.data();
    const char* const end = begin + text.size();
    const auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc{} || result.ptr != end) {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return value;
}

bool parse_boolean(const std::string_view text, const char* name) {
    if (text == "true" || text == "1") return true;
    if (text == "false" || text == "0") return false;
    throw std::invalid_argument(std::string("invalid ") + name);
}

void usage() {
    std::cout
        << "Usage: rlf_efficiency_campaign COMMAND [options]\n"
        << "Commands:\n"
        << "  generate-corpus --records N --output NEW_DIRECTORY [--seed N] [--report FILE]\n"
        << "  materialize-evaluation --ledger FILE --output NEW_DIRECTORY [--report FILE]\n"
        << "  score-evaluation --expected FILE --results DIRECTORY --output NEW_DIRECTORY\n"
        << "  vision-grounding-ablation [--images N] [--width N] [--height N]"
           " [--repetitions N] [--output FILE]\n"
        << "  sparse-router-update-ablation [--vectors N] [--dimensions N]"
           " [--updates N] [--batches N] [--queries N] [--repetitions N]"
           " [--output FILE]\n"
        << "  sparse-rerank-batch-ablation [--vectors N] [--queries N]"
           " [--dimensions N] [--candidates N] [--repetitions N] [--output FILE]\n"
        << "  concept-update-ablation [--images N] [--width N] [--height N]"
           " [--concepts N] [--repetitions N] [--output FILE]\n"
        << "  example-duplicate-ablation [--examples N] [--repetitions N]"
           " [--output FILE]\n"
        << "  mode-id-index-ablation [--modes N] [--images N]"
           " [--repetitions N] [--output FILE]\n"
        << "  grounding-index-ablation [--links N] [--concepts-per-mode N]"
           " [--observations N] [--queries N] [--repetitions N] [--output FILE]\n"
        << "  language-outcome-index-ablation [--outcomes N] [--updates N]"
           " [--repetitions N] [--output FILE]\n"
        << "  preference-duplicate-index-ablation [--preferences N] [--updates N]"
           " [--repetitions N] [--output FILE]\n"
        << "  active-learning-duplicate-index-ablation [--items N] [--updates N]"
           " [--repetitions N] [--output FILE]\n"
        << "  instruction-duplicate-prefilter-ablation [--demonstrations N]"
           " [--updates N] [--repetitions N] [--output FILE]\n"
        << "  tool-keyword-index-ablation [--keywords N] [--updates N]"
           " [--repetitions N] [--output FILE]\n"
        << "  dialogue-encoding-ablation [--dialogues N] [--repetitions N]"
           " [--output FILE]\n"
        << "  fifty-million-contract [physical evidence options]"
           " [--required-target strict-18|stated-hour|physical-evidence|100000x]"
           " [--output FILE]\n"
        << "  physical-floor [options]\n"
        << "Required for nonzero work: matching throughput options.\n"
        << "  --dataset-bytes N --nvme-bytes-per-second N\n"
        << "  --decompressed-bytes N --decompression-bytes-per-second N\n"
        << "  --pcie-bytes N --pcie-bytes-per-second N\n"
        << "  --images N --image-decodes-per-second N\n"
        << "  --records N --preprocessing-records-per-second N\n"
        << "  --checkpoint-bytes N --checkpoint-write-bytes-per-second N\n"
        << "  --evaluation-seconds N [--baseline-seconds 1800000]\n"
        << "  [--target-speedup 100000] [--evidence-kind planning_assumption|measured]\n"
        << "  [--output results/efficiency/physical_runtime_lower_bound.json]\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            usage();
            return 2;
        }
        const std::string_view command = argv[1];
        if (command == "--help" || command == "-h") {
            usage();
            return 0;
        }
        if (command == "generate-corpus") {
            rlf::solstice::EfficiencyCorpusConfig config;
            std::string report_path;
            for (int index = 2; index < argc; ++index) {
                const std::string_view argument = argv[index];
                if (argument == "--help") {
                    usage();
                    return 0;
                }
                if (index + 1 >= argc) {
                    throw std::invalid_argument(
                        "option requires a value: " + std::string(argument)
                    );
                }
                const std::string_view value = argv[++index];
                if (argument == "--records") {
                    config.training_records = parse_number<std::size_t>(value, "record count");
                } else if (argument == "--output") {
                    config.output_directory = value;
                } else if (argument == "--seed") {
                    config.seed = parse_number<std::uint64_t>(value, "seed");
                } else if (argument == "--evaluation-records-per-category") {
                    config.evaluation_records_per_category =
                        parse_number<std::size_t>(value, "evaluation record count");
                } else if (argument == "--report") {
                    report_path = value;
                } else {
                    throw std::invalid_argument("unknown option: " + std::string(argument));
                }
            }
            const auto report = rlf::solstice::generate_efficiency_corpus(config);
            rlf::solstice::write_efficiency_corpus_report_json(std::cout, report);
            if (!report_path.empty()) {
                std::ofstream output(report_path);
                if (!output) throw std::runtime_error("unable to create corpus report");
                rlf::solstice::write_efficiency_corpus_report_json(output, report);
            }
            return 0;
        }
        if (command == "materialize-evaluation") {
            std::string ledger_path;
            std::string output_path;
            std::string report_path;
            for (int index = 2; index < argc; ++index) {
                const std::string_view argument = argv[index];
                if (argument == "--help") {
                    usage();
                    return 0;
                }
                if (index + 1 >= argc) {
                    throw std::invalid_argument(
                        "option requires a value: " + std::string(argument)
                    );
                }
                const std::string_view value = argv[++index];
                if (argument == "--ledger") ledger_path = value;
                else if (argument == "--output") output_path = value;
                else if (argument == "--report") report_path = value;
                else throw std::invalid_argument("unknown option: " + std::string(argument));
            }
            const auto report = rlf::solstice::materialize_efficiency_evaluation(
                ledger_path, output_path
            );
            rlf::solstice::write_efficiency_evaluation_report_json(std::cout, report);
            if (!report_path.empty()) {
                std::ofstream output(report_path);
                if (!output) throw std::runtime_error("unable to create evaluation report");
                rlf::solstice::write_efficiency_evaluation_report_json(output, report);
            }
            return 0;
        }
        if (command == "score-evaluation") {
            std::string expected_path;
            std::string results_path;
            std::string output_path;
            for (int index = 2; index < argc; ++index) {
                const std::string_view argument = argv[index];
                if (argument == "--help") {
                    usage();
                    return 0;
                }
                if (index + 1 >= argc) {
                    throw std::invalid_argument(
                        "option requires a value: " + std::string(argument)
                    );
                }
                const std::string_view value = argv[++index];
                if (argument == "--expected") expected_path = value;
                else if (argument == "--results") results_path = value;
                else if (argument == "--output") output_path = value;
                else throw std::invalid_argument("unknown option: " + std::string(argument));
            }
            const auto score = rlf::solstice::score_efficiency_evaluation(
                expected_path, results_path, output_path
            );
            rlf::solstice::write_efficiency_evaluation_score_json(std::cout, score);
            return 0;
        }
        if (command == "vision-grounding-ablation") {
            rlf::solstice::VisionGroundingAblationInputs inputs;
            std::string output_path;
            for (int index = 2; index < argc; ++index) {
                const std::string_view argument = argv[index];
                if (argument == "--help") {
                    usage();
                    return 0;
                }
                if (index + 1 >= argc) {
                    throw std::invalid_argument(
                        "option requires a value: " + std::string(argument)
                    );
                }
                const std::string_view value = argv[++index];
                if (argument == "--images") {
                    inputs.images = parse_number<std::size_t>(value, "image count");
                } else if (argument == "--width") {
                    inputs.width = parse_number<std::size_t>(value, "image width");
                } else if (argument == "--height") {
                    inputs.height = parse_number<std::size_t>(value, "image height");
                } else if (argument == "--repetitions") {
                    inputs.repetitions = parse_number<std::size_t>(value, "repetition count");
                } else if (argument == "--output") {
                    output_path = value;
                } else {
                    throw std::invalid_argument("unknown option: " + std::string(argument));
                }
            }
            const auto report = rlf::solstice::measure_vision_grounding_ablation(inputs);
            rlf::solstice::write_vision_grounding_ablation_json(std::cout, report);
            if (!output_path.empty()) {
                std::ofstream output(output_path);
                if (!output) throw std::runtime_error("unable to create ablation report");
                rlf::solstice::write_vision_grounding_ablation_json(output, report);
            }
            return report.analyses_identical && report.model_states_identical ? 0 : 1;
        }
        if (command == "sparse-router-update-ablation") {
            rlf::solstice::SparseRouterUpdateAblationInputs inputs;
            std::string output_path;
            for (int index = 2; index < argc; ++index) {
                const std::string_view argument = argv[index];
                if (argument == "--help") {
                    usage();
                    return 0;
                }
                if (index + 1 >= argc) {
                    throw std::invalid_argument(
                        "option requires a value: " + std::string(argument)
                    );
                }
                const std::string_view value = argv[++index];
                if (argument == "--vectors") {
                    inputs.vector_count = parse_number<std::size_t>(value, "vector count");
                } else if (argument == "--dimensions") {
                    inputs.dimensions = parse_number<std::size_t>(value, "dimension count");
                } else if (argument == "--updates") {
                    inputs.updates_per_batch = parse_number<std::size_t>(value, "update count");
                } else if (argument == "--batches") {
                    inputs.batches = parse_number<std::size_t>(value, "batch count");
                } else if (argument == "--queries") {
                    inputs.queries_per_batch = parse_number<std::size_t>(value, "query count");
                } else if (argument == "--repetitions") {
                    inputs.repetitions = parse_number<std::size_t>(value, "repetition count");
                } else if (argument == "--output") {
                    output_path = value;
                } else {
                    throw std::invalid_argument("unknown option: " + std::string(argument));
                }
            }
            const auto report =
                rlf::solstice::measure_sparse_router_update_ablation(inputs);
            rlf::solstice::write_sparse_router_update_ablation_json(
                std::cout, report
            );
            if (!output_path.empty()) {
                std::ofstream output(output_path);
                if (!output) throw std::runtime_error("unable to create ablation report");
                rlf::solstice::write_sparse_router_update_ablation_json(
                    output, report
                );
            }
            return report.all_routes_identical ? 0 : 1;
        }
        if (command == "sparse-rerank-batch-ablation") {
            rlf::solstice::SparseRerankBatchAblationInputs inputs;
            std::string output_path;
            for (int index = 2; index < argc; ++index) {
                const std::string_view argument = argv[index];
                if (argument == "--help") {
                    usage();
                    return 0;
                }
                if (index + 1 >= argc) {
                    throw std::invalid_argument(
                        "option requires a value: " + std::string(argument)
                    );
                }
                const std::string_view value = argv[++index];
                if (argument == "--vectors") {
                    inputs.vector_count = parse_number<std::size_t>(value, "vector count");
                } else if (argument == "--queries") {
                    inputs.query_count = parse_number<std::size_t>(value, "query count");
                } else if (argument == "--dimensions") {
                    inputs.dimensions = parse_number<std::size_t>(value, "dimension count");
                } else if (argument == "--candidates") {
                    inputs.maximum_candidates =
                        parse_number<std::size_t>(value, "candidate count");
                } else if (argument == "--repetitions") {
                    inputs.repetitions = parse_number<std::size_t>(value, "repetition count");
                } else if (argument == "--output") {
                    output_path = value;
                } else {
                    throw std::invalid_argument("unknown option: " + std::string(argument));
                }
            }
            const auto report =
                rlf::solstice::measure_sparse_rerank_batch_ablation(inputs);
            rlf::solstice::write_sparse_rerank_batch_ablation_json(std::cout, report);
            if (!output_path.empty()) {
                std::ofstream output(output_path);
                if (!output) throw std::runtime_error("unable to create ablation report");
                rlf::solstice::write_sparse_rerank_batch_ablation_json(output, report);
            }
            return report.results_identical ? 0 : 1;
        }
        if (command == "concept-update-ablation") {
            rlf::solstice::ConceptUpdateAblationInputs inputs;
            std::string output_path;
            for (int index = 2; index < argc; ++index) {
                const std::string_view argument = argv[index];
                if (argument == "--help") {
                    usage();
                    return 0;
                }
                if (index + 1 >= argc) {
                    throw std::invalid_argument(
                        "option requires a value: " + std::string(argument)
                    );
                }
                const std::string_view value = argv[++index];
                if (argument == "--images") {
                    inputs.images = parse_number<std::size_t>(value, "image count");
                } else if (argument == "--width") {
                    inputs.width = parse_number<std::size_t>(value, "image width");
                } else if (argument == "--height") {
                    inputs.height = parse_number<std::size_t>(value, "image height");
                } else if (argument == "--concepts") {
                    inputs.concepts = parse_number<std::size_t>(value, "concept count");
                } else if (argument == "--repetitions") {
                    inputs.repetitions = parse_number<std::size_t>(value, "repetition count");
                } else if (argument == "--output") {
                    output_path = value;
                } else {
                    throw std::invalid_argument("unknown option: " + std::string(argument));
                }
            }
            const auto report = rlf::solstice::measure_concept_update_ablation(inputs);
            rlf::solstice::write_concept_update_ablation_json(std::cout, report);
            if (!output_path.empty()) {
                std::ofstream output(output_path);
                if (!output) throw std::runtime_error("unable to create ablation report");
                rlf::solstice::write_concept_update_ablation_json(output, report);
            }
            return report.model_states_identical ? 0 : 1;
        }
        if (command == "example-duplicate-ablation") {
            rlf::solstice::ExampleDuplicateAblationInputs inputs;
            std::string output_path;
            for (int index = 2; index < argc; ++index) {
                const std::string_view argument = argv[index];
                if (argument == "--help") {
                    usage();
                    return 0;
                }
                if (index + 1 >= argc) {
                    throw std::invalid_argument(
                        "option requires a value: " + std::string(argument)
                    );
                }
                const std::string_view value = argv[++index];
                if (argument == "--examples") {
                    inputs.examples = parse_number<std::size_t>(
                        value, "example count"
                    );
                } else if (argument == "--repetitions") {
                    inputs.repetitions = parse_number<std::size_t>(
                        value, "repetition count"
                    );
                } else if (argument == "--output") {
                    output_path = value;
                } else {
                    throw std::invalid_argument(
                        "unknown option: " + std::string(argument)
                    );
                }
            }
            const auto report =
                rlf::solstice::measure_example_duplicate_ablation(inputs);
            rlf::solstice::write_example_duplicate_ablation_json(
                std::cout, report
            );
            if (!output_path.empty()) {
                std::ofstream output(output_path);
                if (!output) {
                    throw std::runtime_error(
                        "unable to create ablation report"
                    );
                }
                rlf::solstice::write_example_duplicate_ablation_json(
                    output, report
                );
            }
            return report.model_states_identical ? 0 : 1;
        }
        if (command == "mode-id-index-ablation") {
            rlf::solstice::ModeIdIndexAblationInputs inputs;
            std::string output_path;
            for (int index = 2; index < argc; ++index) {
                const std::string_view argument = argv[index];
                if (argument == "--help") {
                    usage();
                    return 0;
                }
                if (index + 1 >= argc) {
                    throw std::invalid_argument(
                        "option requires a value: " + std::string(argument)
                    );
                }
                const std::string_view value = argv[++index];
                if (argument == "--modes") {
                    inputs.modes = parse_number<std::size_t>(value, "mode count");
                } else if (argument == "--images") {
                    inputs.images = parse_number<std::size_t>(value, "image count");
                } else if (argument == "--repetitions") {
                    inputs.repetitions = parse_number<std::size_t>(
                        value, "repetition count"
                    );
                } else if (argument == "--output") {
                    output_path = value;
                } else {
                    throw std::invalid_argument(
                        "unknown option: " + std::string(argument)
                    );
                }
            }
            const auto report = rlf::solstice::measure_mode_id_index_ablation(
                inputs
            );
            rlf::solstice::write_mode_id_index_ablation_json(std::cout, report);
            if (!output_path.empty()) {
                std::ofstream output(output_path);
                if (!output) {
                    throw std::runtime_error(
                        "unable to create ablation report"
                    );
                }
                rlf::solstice::write_mode_id_index_ablation_json(output, report);
            }
            return report.model_states_identical && report.analyses_identical
                ? 0 : 1;
        }
        if (command == "grounding-index-ablation") {
            rlf::solstice::GroundingIndexAblationInputs inputs;
            std::string output_path;
            for (int index = 2; index < argc; ++index) {
                const std::string_view argument = argv[index];
                if (argument == "--help") {
                    usage();
                    return 0;
                }
                if (index + 1 >= argc) {
                    throw std::invalid_argument(
                        "option requires a value: " + std::string(argument)
                    );
                }
                const std::string_view value = argv[++index];
                if (argument == "--links") {
                    inputs.initial_links = parse_number<std::size_t>(
                        value, "link count"
                    );
                } else if (argument == "--concepts-per-mode") {
                    inputs.concepts_per_mode = parse_number<std::size_t>(
                        value, "concept count"
                    );
                } else if (argument == "--observations") {
                    inputs.observations = parse_number<std::size_t>(
                        value, "observation count"
                    );
                } else if (argument == "--queries") {
                    inputs.queries = parse_number<std::size_t>(
                        value, "query count"
                    );
                } else if (argument == "--repetitions") {
                    inputs.repetitions = parse_number<std::size_t>(
                        value, "repetition count"
                    );
                } else if (argument == "--output") {
                    output_path = value;
                } else {
                    throw std::invalid_argument(
                        "unknown option: " + std::string(argument)
                    );
                }
            }
            const auto report = rlf::solstice::measure_grounding_index_ablation(
                inputs
            );
            rlf::solstice::write_grounding_index_ablation_json(
                std::cout, report
            );
            if (!output_path.empty()) {
                std::ofstream output(output_path);
                if (!output) {
                    throw std::runtime_error(
                        "unable to create ablation report"
                    );
                }
                rlf::solstice::write_grounding_index_ablation_json(
                    output, report
                );
            }
            return report.model_states_identical &&
                    report.query_results_identical
                ? 0
                : 1;
        }
        if (command == "language-outcome-index-ablation") {
            rlf::solstice::LanguageOutcomeIndexAblationInputs inputs;
            std::string output_path;
            for (int index = 2; index < argc; ++index) {
                const std::string_view argument = argv[index];
                if (argument == "--help") {
                    usage();
                    return 0;
                }
                if (index + 1 >= argc) {
                    throw std::invalid_argument(
                        "option requires a value: " + std::string(argument)
                    );
                }
                const std::string_view value = argv[++index];
                if (argument == "--outcomes") {
                    inputs.outcomes = parse_number<std::size_t>(
                        value, "outcome count"
                    );
                } else if (argument == "--updates") {
                    inputs.updates = parse_number<std::size_t>(
                        value, "update count"
                    );
                } else if (argument == "--repetitions") {
                    inputs.repetitions = parse_number<std::size_t>(
                        value, "repetition count"
                    );
                } else if (argument == "--output") {
                    output_path = value;
                } else {
                    throw std::invalid_argument(
                        "unknown option: " + std::string(argument)
                    );
                }
            }
            const auto report =
                rlf::solstice::measure_language_outcome_index_ablation(inputs);
            rlf::solstice::write_language_outcome_index_ablation_json(
                std::cout, report
            );
            if (!output_path.empty()) {
                std::ofstream output(output_path);
                if (!output) {
                    throw std::runtime_error(
                        "unable to create ablation report"
                    );
                }
                rlf::solstice::write_language_outcome_index_ablation_json(
                    output, report
                );
            }
            return report.model_states_identical && report.predictions_identical
                ? 0
                : 1;
        }
        if (command == "preference-duplicate-index-ablation") {
            rlf::solstice::PreferenceDuplicateIndexAblationInputs inputs;
            std::string output_path;
            for (int index = 2; index < argc; ++index) {
                const std::string_view argument = argv[index];
                if (argument == "--help") {
                    usage();
                    return 0;
                }
                if (index + 1 >= argc) {
                    throw std::invalid_argument(
                        "option requires a value: " + std::string(argument)
                    );
                }
                const std::string_view value = argv[++index];
                if (argument == "--preferences") {
                    inputs.preferences = parse_number<std::size_t>(
                        value, "preference count"
                    );
                } else if (argument == "--updates") {
                    inputs.updates = parse_number<std::size_t>(
                        value, "update count"
                    );
                } else if (argument == "--repetitions") {
                    inputs.repetitions = parse_number<std::size_t>(
                        value, "repetition count"
                    );
                } else if (argument == "--output") {
                    output_path = value;
                } else {
                    throw std::invalid_argument(
                        "unknown option: " + std::string(argument)
                    );
                }
            }
            const auto report =
                rlf::solstice::measure_preference_duplicate_index_ablation(inputs);
            rlf::solstice::write_preference_duplicate_index_ablation_json(
                std::cout, report
            );
            if (!output_path.empty()) {
                std::ofstream output(output_path);
                if (!output) {
                    throw std::runtime_error(
                        "unable to create ablation report"
                    );
                }
                rlf::solstice::write_preference_duplicate_index_ablation_json(
                    output, report
                );
            }
            return report.model_states_identical && report.scores_identical
                ? 0
                : 1;
        }
        if (command == "active-learning-duplicate-index-ablation") {
            rlf::solstice::ActiveLearningDuplicateIndexAblationInputs inputs;
            std::string output_path;
            for (int index = 2; index < argc; ++index) {
                const std::string_view argument = argv[index];
                if (argument == "--help") {
                    usage();
                    return 0;
                }
                if (index + 1 >= argc) {
                    throw std::invalid_argument(
                        "option requires a value: " + std::string(argument)
                    );
                }
                const std::string_view value = argv[++index];
                if (argument == "--items") {
                    inputs.items = parse_number<std::size_t>(value, "item count");
                } else if (argument == "--updates") {
                    inputs.updates = parse_number<std::size_t>(
                        value, "update count"
                    );
                } else if (argument == "--repetitions") {
                    inputs.repetitions = parse_number<std::size_t>(
                        value, "repetition count"
                    );
                } else if (argument == "--output") {
                    output_path = value;
                } else {
                    throw std::invalid_argument(
                        "unknown option: " + std::string(argument)
                    );
                }
            }
            const auto report = rlf::solstice::
                measure_active_learning_duplicate_index_ablation(inputs);
            rlf::solstice::write_active_learning_duplicate_index_ablation_json(
                std::cout, report
            );
            if (!output_path.empty()) {
                std::ofstream output(output_path);
                if (!output) {
                    throw std::runtime_error(
                        "unable to create ablation report"
                    );
                }
                rlf::solstice::
                    write_active_learning_duplicate_index_ablation_json(
                        output, report
                    );
            }
            return report.model_states_identical ? 0 : 1;
        }
        if (command == "instruction-duplicate-prefilter-ablation") {
            rlf::solstice::InstructionDuplicatePrefilterAblationInputs inputs;
            std::string output_path;
            for (int index = 2; index < argc; ++index) {
                const std::string_view argument = argv[index];
                if (argument == "--help") {
                    usage();
                    return 0;
                }
                if (index + 1 >= argc) {
                    throw std::invalid_argument(
                        "option requires a value: " + std::string(argument)
                    );
                }
                const std::string_view value = argv[++index];
                if (argument == "--demonstrations") {
                    inputs.demonstrations = parse_number<std::size_t>(
                        value, "demonstration count"
                    );
                } else if (argument == "--updates") {
                    inputs.updates = parse_number<std::size_t>(
                        value, "update count"
                    );
                } else if (argument == "--repetitions") {
                    inputs.repetitions = parse_number<std::size_t>(
                        value, "repetition count"
                    );
                } else if (argument == "--output") {
                    output_path = value;
                } else {
                    throw std::invalid_argument(
                        "unknown option: " + std::string(argument)
                    );
                }
            }
            const auto report = rlf::solstice::
                measure_instruction_duplicate_prefilter_ablation(inputs);
            rlf::solstice::write_instruction_duplicate_prefilter_ablation_json(
                std::cout, report
            );
            if (!output_path.empty()) {
                std::ofstream output(output_path);
                if (!output) throw std::runtime_error("unable to create report");
                rlf::solstice::
                    write_instruction_duplicate_prefilter_ablation_json(
                        output, report
                    );
            }
            return report.model_states_identical ? 0 : 1;
        }
        if (command == "tool-keyword-index-ablation") {
            rlf::solstice::ToolKeywordIndexAblationInputs inputs;
            std::string output_path;
            for (int index = 2; index < argc; ++index) {
                const std::string_view argument = argv[index];
                if (argument == "--help") {
                    usage();
                    return 0;
                }
                if (index + 1 >= argc) {
                    throw std::invalid_argument(
                        "option requires a value: " + std::string(argument)
                    );
                }
                const std::string_view value = argv[++index];
                if (argument == "--keywords") {
                    inputs.keywords = parse_number<std::size_t>(
                        value, "keyword count"
                    );
                } else if (argument == "--updates") {
                    inputs.updates = parse_number<std::size_t>(
                        value, "update count"
                    );
                } else if (argument == "--repetitions") {
                    inputs.repetitions = parse_number<std::size_t>(
                        value, "repetition count"
                    );
                } else if (argument == "--output") {
                    output_path = value;
                } else {
                    throw std::invalid_argument(
                        "unknown option: " + std::string(argument)
                    );
                }
            }
            const auto report =
                rlf::solstice::measure_tool_keyword_index_ablation(inputs);
            rlf::solstice::write_tool_keyword_index_ablation_json(
                std::cout, report
            );
            if (!output_path.empty()) {
                std::ofstream output(output_path);
                if (!output) throw std::runtime_error("unable to create report");
                rlf::solstice::write_tool_keyword_index_ablation_json(
                    output, report
                );
            }
            return report.model_states_identical &&
                   report.keyword_capacity_skips == 0U
                ? 0
                : 1;
        }
        if (command == "dialogue-encoding-ablation") {
            rlf::solstice::DialogueEncodingAblationInputs inputs;
            std::string output_path;
            for (int index = 2; index < argc; ++index) {
                const std::string_view argument = argv[index];
                if (argument == "--help") {
                    usage();
                    return 0;
                }
                if (index + 1 >= argc) {
                    throw std::invalid_argument(
                        "option requires a value: " + std::string(argument)
                    );
                }
                const std::string_view value = argv[++index];
                if (argument == "--dialogues") {
                    inputs.dialogues = parse_number<std::size_t>(
                        value, "dialogue count"
                    );
                } else if (argument == "--repetitions") {
                    inputs.repetitions = parse_number<std::size_t>(
                        value, "repetition count"
                    );
                } else if (argument == "--output") {
                    output_path = value;
                } else {
                    throw std::invalid_argument(
                        "unknown option: " + std::string(argument)
                    );
                }
            }
            const auto report =
                rlf::solstice::measure_dialogue_encoding_ablation(inputs);
            rlf::solstice::write_dialogue_encoding_ablation_json(
                std::cout, report
            );
            if (!output_path.empty()) {
                std::ofstream output(output_path);
                if (!output) throw std::runtime_error("unable to create report");
                rlf::solstice::write_dialogue_encoding_ablation_json(
                    output, report
                );
            }
            return report.model_states_identical ? 0 : 1;
        }
        if (command == "fifty-million-contract") {
            rlf::solstice::FiftyMillionTrainingEvidence evidence;
            std::string required_target{"strict-18"};
            std::string output_path;
            for (int index = 2; index < argc; ++index) {
                const std::string_view argument = argv[index];
                if (argument == "--help") {
                    usage();
                    return 0;
                }
                if (index + 1 >= argc) {
                    throw std::invalid_argument(
                        "option requires a value: " + std::string(argument)
                    );
                }
                const std::string_view value = argv[++index];
                if (argument == "--hardware-profile") evidence.hardware_profile = value;
                else if (argument == "--device-name") evidence.device_name = value;
                else if (argument == "--device-uuid") evidence.device_uuid = value;
                else if (argument == "--backend") evidence.backend = value;
                else if (argument == "--device-count") evidence.device_count = parse_number<std::uint64_t>(value, "device count");
                else if (argument == "--mig-disabled") evidence.mig_disabled = parse_boolean(value, "MIG state");
                else if (argument == "--records") evidence.total_records = parse_number<std::uint64_t>(value, "record count");
                else if (argument == "--instruction-records") evidence.instruction_records = parse_number<std::uint64_t>(value, "instruction count");
                else if (argument == "--preference-records") evidence.preference_records = parse_number<std::uint64_t>(value, "preference count");
                else if (argument == "--tool-records") evidence.tool_records = parse_number<std::uint64_t>(value, "tool count");
                else if (argument == "--fact-records") evidence.fact_records = parse_number<std::uint64_t>(value, "fact count");
                else if (argument == "--vision-records") evidence.vision_records = parse_number<std::uint64_t>(value, "vision count");
                else if (argument == "--episode-capacity-skips") evidence.episode_capacity_skips = parse_number<std::uint64_t>(value, "episode capacity skips");
                else if (argument == "--context-capacity-skips") evidence.context_capacity_skips = parse_number<std::uint64_t>(value, "context capacity skips");
                else if (argument == "--tool-keyword-capacity-skips") evidence.tool_keyword_capacity_skips = parse_number<std::uint64_t>(value, "tool keyword capacity skips");
                else if (argument == "--wall-seconds") evidence.wall_seconds = parse_number<double>(value, "wall seconds");
                else if (argument == "--gpu-active-seconds") evidence.gpu_active_seconds = parse_number<double>(value, "GPU active seconds");
                else if (argument == "--peak-vram-bytes") evidence.peak_vram_bytes = parse_number<std::uint64_t>(value, "peak VRAM bytes");
                else if (argument == "--checkpoint-verified") evidence.checkpoint_verified = parse_boolean(value, "checkpoint verification");
                else if (argument == "--raw-gpu-trace-bound") evidence.raw_gpu_trace_bound = parse_boolean(value, "GPU trace binding");
                else if (argument == "--raw-energy-trace-bound") evidence.raw_energy_trace_bound = parse_boolean(value, "energy trace binding");
                else if (argument == "--baseline-gpu-hours") evidence.baseline_gpu_hours = parse_number<double>(value, "baseline GPU hours");
                else if (argument == "--candidate-gpu-hours") evidence.candidate_gpu_hours = parse_number<double>(value, "candidate GPU hours");
                else if (argument == "--matched-quality") evidence.matched_quality = parse_boolean(value, "matched quality");
                else if (argument == "--external-quality-gate-passed") evidence.external_quality_gate_passed = parse_boolean(value, "external quality gate");
                else if (argument == "--independent-reproduction-complete") evidence.independent_reproduction_complete = parse_boolean(value, "independent reproduction");
                else if (argument == "--required-target") required_target = value;
                else if (argument == "--output") output_path = value;
                else throw std::invalid_argument("unknown option: " + std::string(argument));
            }
            const auto report =
                rlf::solstice::evaluate_fifty_million_contract(evidence);
            rlf::solstice::write_fifty_million_contract_json(std::cout, report);
            if (!output_path.empty()) {
                std::ofstream output(output_path);
                if (!output) throw std::runtime_error("unable to open output path");
                rlf::solstice::write_fifty_million_contract_json(output, report);
            }
            bool passed = false;
            if (required_target == "strict-18") {
                passed = report.strict_eighteen_second_target_passed;
            } else if (required_target == "stated-hour") {
                passed = report.stated_hour_target_passed;
            } else if (required_target == "physical-evidence") {
                passed = report.physical_evidence_complete;
            } else if (required_target == "100000x") {
                passed = report.general_100000x_gpu_hour_efficiency_proven;
            } else {
                throw std::invalid_argument("invalid required target");
            }
            return passed ? 0 : 1;
        }
        if (command != "physical-floor") {
            usage();
            return 2;
        }
        rlf::solstice::PhysicalRuntimeInputs inputs;
        std::string output_path;
        for (int index = 2; index < argc; ++index) {
            const std::string_view argument = argv[index];
            if (argument == "--help") {
                usage();
                return 0;
            }
            if (index + 1 >= argc) {
                throw std::invalid_argument("option requires a value: " + std::string(argument));
            }
            const std::string_view value = argv[++index];
            if (argument == "--dataset-bytes") inputs.compressed_dataset_bytes = parse_number<std::uint64_t>(value, "dataset bytes");
            else if (argument == "--decompressed-bytes") inputs.decompressed_dataset_bytes = parse_number<std::uint64_t>(value, "decompressed bytes");
            else if (argument == "--pcie-bytes") inputs.pcie_transfer_bytes = parse_number<std::uint64_t>(value, "PCIe bytes");
            else if (argument == "--images") inputs.image_count = parse_number<std::uint64_t>(value, "image count");
            else if (argument == "--records") inputs.record_count = parse_number<std::uint64_t>(value, "record count");
            else if (argument == "--checkpoint-bytes") inputs.checkpoint_bytes = parse_number<std::uint64_t>(value, "checkpoint bytes");
            else if (argument == "--nvme-bytes-per-second") inputs.nvme_bytes_per_second = parse_number<double>(value, "NVMe rate");
            else if (argument == "--decompression-bytes-per-second") inputs.decompression_bytes_per_second = parse_number<double>(value, "decompression rate");
            else if (argument == "--pcie-bytes-per-second") inputs.pcie_bytes_per_second = parse_number<double>(value, "PCIe rate");
            else if (argument == "--image-decodes-per-second") inputs.image_decodes_per_second = parse_number<double>(value, "image decode rate");
            else if (argument == "--preprocessing-records-per-second") inputs.preprocessing_records_per_second = parse_number<double>(value, "preprocessing rate");
            else if (argument == "--checkpoint-write-bytes-per-second") inputs.checkpoint_write_bytes_per_second = parse_number<double>(value, "checkpoint write rate");
            else if (argument == "--evaluation-seconds") inputs.evaluation_seconds = parse_number<double>(value, "evaluation seconds");
            else if (argument == "--baseline-seconds") inputs.baseline_seconds = parse_number<double>(value, "baseline seconds");
            else if (argument == "--target-speedup") inputs.target_speedup = parse_number<double>(value, "target speedup");
            else if (argument == "--evidence-kind") inputs.evidence_kind = value;
            else if (argument == "--hardware-profile") inputs.hardware_profile = value;
            else if (argument == "--output") output_path = value;
            else throw std::invalid_argument("unknown option: " + std::string(argument));
        }
        if (inputs.evidence_kind != "planning_assumption" && inputs.evidence_kind != "measured") {
            throw std::invalid_argument("evidence-kind must be planning_assumption or measured");
        }
        const auto report = rlf::solstice::calculate_physical_runtime_floor(inputs);
        rlf::solstice::write_physical_runtime_floor_json(std::cout, report);
        if (!output_path.empty()) {
            std::ofstream output(output_path);
            if (!output) throw std::runtime_error("unable to open output path");
            rlf::solstice::write_physical_runtime_floor_json(output, report);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "efficiency campaign error: " << error.what() << '\n';
        return 1;
    }
}
