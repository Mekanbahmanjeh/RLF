#include "rlf/solstice/frontier_gate.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace rlf::solstice {
namespace {

[[nodiscard]] std::string escape_json(const std::string_view value) {
    std::string output;
    output.reserve(value.size() + 8U);
    for (const char character : value) {
        switch (character) {
            case '\\': output += "\\\\"; break;
            case '"': output += "\\\""; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default: output.push_back(character); break;
        }
    }
    return output;
}

[[nodiscard]] bool valid_evidence(const FrontierBenchmarkEvidence& item) noexcept {
    return !item.benchmark.empty() && !item.capability.empty() &&
        std::isfinite(item.score) && item.score >= 0.0 && item.score <= 1.0;
}

}  // namespace

std::vector<FrontierBenchmarkTarget> leading_system_targets_2026() {
    constexpr std::string_view gemini_source =
        "Google DeepMind Gemini 3.1 Pro model card, 2026-02-19, "
        "https://deepmind.google/models/model-cards/gemini-3-1-pro/";
    constexpr std::string_view gpt56_source =
        "OpenAI GPT-5.6 release, 2026-07-09, "
        "https://openai.com/index/gpt-5-6/";
    constexpr std::string_view arc_source =
        "ARC Prize verified GPT-5.6 results, 2026-07-09, "
        "https://arcprize.org/results/openai-gpt-5-6";
    constexpr std::string_view mythos_source =
        "Anthropic Claude Mythos Preview system card, 2026-04, "
        "https://www.anthropic.com/glasswing";
    constexpr std::string_view opus46_source =
        "Anthropic Claude Opus 4.6 system card, 2026-02, "
        "https://www.anthropic.com/claude/opus";
    return {
        {"hle", "academic_reasoning", 0.568, 500U, "2026-04", std::string(mythos_source)},
        {"arc_agi_2", "abstract_reasoning", 0.925, 120U, "2026-07", std::string(arc_source)},
        {"gpqa_diamond", "scientific_reasoning", 0.946, 198U, "2026-07", std::string(gpt56_source)},
        {"swe_bench_verified", "agentic_coding", 0.939, 500U, "2026-04", std::string(mythos_source)},
        {"mmmu_pro", "multimodal_reasoning", 0.830, 500U, "2026-07", std::string(gpt56_source)},
        {"mmmlu", "multilingual_knowledge", 0.927, 1'000U, "2026-04", std::string(mythos_source)},
        {"tau2_retail", "tool_use", 0.919, 100U, "2026-02", std::string(opus46_source)},
        {"mrcr_128k", "long_context", 0.849, 100U, "2026-02", std::string(gemini_source)},
    };
}

FrontierGateReport evaluate_frontier_gate(
    const std::span<const FrontierBenchmarkEvidence> evidence,
    const std::span<const FrontierBenchmarkTarget> targets
) {
    if (targets.empty()) {
        throw std::invalid_argument("frontier gate requires at least one target");
    }
    FrontierGateReport report;
    report.items.reserve(targets.size());
    for (const FrontierBenchmarkTarget& target : targets) {
        FrontierGateItem item;
        item.target = target;
        std::vector<const FrontierBenchmarkEvidence*> matches;
        for (const FrontierBenchmarkEvidence& candidate : evidence) {
            if (!valid_evidence(candidate) || candidate.benchmark != target.benchmark ||
                candidate.capability != target.capability) {
                continue;
            }
            matches.push_back(&candidate);
        }
        if (matches.empty()) {
            item.reason = "missing benchmark evidence";
            report.blocking_reasons.push_back(target.benchmark + ": missing evidence");
        } else if (matches.size() != 1U) {
            item.present = true;
            item.reason = "multiple evidence rows are ambiguous";
            report.blocking_reasons.push_back(target.benchmark + ": " + item.reason);
        } else {
            const FrontierBenchmarkEvidence* const best = matches.front();
            item.present = true;
            item.best_score = best->score;
            item.examples = best->examples;
            report.external_examples += best->external_dataset ? best->examples : 0U;
            if (!best->external_dataset) {
                item.reason = "dataset is internal or synthetic";
            } else if (!best->contamination_audited) {
                item.reason = "contamination audit missing";
            } else if (!best->independent_harness) {
                item.reason = "independent benchmark harness missing";
            } else if (best->examples < target.minimum_examples) {
                item.reason = "insufficient evaluated examples";
            } else if (best->score < target.minimum_score) {
                item.reason = "score below leading-system target";
            } else if (best->dataset_version.empty()) {
                item.reason = "dataset version missing";
            } else if (best->harness_name.empty() || best->harness_version.empty()) {
                item.reason = "benchmark harness identity missing";
            } else if (best->evaluator.empty()) {
                item.reason = "independent evaluator identity missing";
            } else if (best->evidence_path.empty()) {
                item.reason = "evidence artifact path missing";
            } else if (!best->evidence_artifact_present) {
                item.reason = "evidence artifact is missing, empty, or not a regular file";
            } else if (best->evidence_sha256.empty() ||
                       !best->evidence_sha256_verified) {
                item.reason = "evidence artifact SHA-256 is missing or mismatched";
            } else if (best->contamination_report_path.empty() ||
                       !best->contamination_report_present) {
                item.reason = "contamination report artifact is missing or empty";
            } else if (best->contamination_report_sha256.empty() ||
                       !best->contamination_sha256_verified) {
                item.reason = "contamination report SHA-256 is missing or mismatched";
            } else if (best->evidence_path == best->contamination_report_path) {
                item.reason = "raw evidence and contamination report must be distinct artifacts";
            } else if (best->model_checkpoint_sha256.empty() ||
                       best->training_manifest_sha256.empty() ||
                       !best->model_identity_verified) {
                item.reason = "model checkpoint/training manifest identity missing or mismatched";
            } else {
                item.passed = true;
                item.reason = "passed";
                ++report.passed_targets;
            }
            if (!item.passed) {
                report.blocking_reasons.push_back(
                    target.benchmark + ": " + item.reason
                );
            }
        }
        report.items.push_back(std::move(item));
    }
    report.all_targets_passed = report.passed_targets == targets.size();
    report.minimum_external_examples = std::accumulate(
        targets.begin(), targets.end(), std::size_t{0U},
        [](const std::size_t total, const FrontierBenchmarkTarget& target) {
            if (target.minimum_examples >
                std::numeric_limits<std::size_t>::max() - total) {
                throw std::overflow_error("frontier target example count overflow");
            }
            return total + target.minimum_examples;
        }
    );
    if (report.external_examples < report.minimum_external_examples) {
        report.blocking_reasons.push_back(
            "external evaluation coverage below required " +
            std::to_string(report.minimum_external_examples) + " examples"
        );
    }
    report.broad_frontier_parity_proven = report.all_targets_passed &&
        report.external_examples >= report.minimum_external_examples &&
        report.blocking_reasons.empty();
    return report;
}

std::string frontier_gate_json(const FrontierGateReport& report) {
    std::ostringstream output;
    output << std::boolalpha << std::fixed << std::setprecision(6);
    output << "{\n"
           << "  \"broad_frontier_parity_proven\": "
           << report.broad_frontier_parity_proven << ",\n"
           << "  \"all_targets_passed\": " << report.all_targets_passed << ",\n"
           << "  \"passed_targets\": " << report.passed_targets << ",\n"
           << "  \"external_examples\": " << report.external_examples << ",\n"
           << "  \"minimum_external_examples\": "
           << report.minimum_external_examples << ",\n"
           << "  \"targets\": [\n";
    for (std::size_t index = 0U; index < report.items.size(); ++index) {
        const FrontierGateItem& item = report.items[index];
        output << "    {\"benchmark\": \"" << escape_json(item.target.benchmark)
               << "\", \"capability\": \"" << escape_json(item.target.capability)
               << "\", \"target\": " << item.target.minimum_score
               << ", \"minimum_examples\": " << item.target.minimum_examples
               << ", \"reference_date\": \""
               << escape_json(item.target.reference_date)
               << "\", \"reference_source\": \""
               << escape_json(item.target.reference_source)
               << "\", \"score\": " << item.best_score
               << ", \"examples\": " << item.examples
               << ", \"present\": " << item.present
               << ", \"passed\": " << item.passed
               << ", \"reason\": \"" << escape_json(item.reason) << "\"}";
        if (index + 1U != report.items.size()) {
            output << ',';
        }
        output << '\n';
    }
    output << "  ],\n  \"blocking_reasons\": [";
    for (std::size_t index = 0U; index < report.blocking_reasons.size(); ++index) {
        if (index != 0U) {
            output << ", ";
        }
        output << '"' << escape_json(report.blocking_reasons[index]) << '"';
    }
    output << "]\n}\n";
    return output.str();
}

}  // namespace rlf::solstice
