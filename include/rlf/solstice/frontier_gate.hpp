#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rlf::solstice {

struct FrontierBenchmarkTarget final {
    std::string benchmark;
    std::string capability;
    double minimum_score{};
    std::size_t minimum_examples{};
    std::string reference_date;
    std::string reference_source;
};

struct FrontierBenchmarkEvidence final {
    std::string benchmark;
    std::string capability;
    double score{};
    std::size_t examples{};
    bool external_dataset{};
    bool contamination_audited{};
    bool independent_harness{};
    bool evidence_artifact_present{};
    std::string evidence_path;
    std::string dataset_version;
    std::string harness_name;
    std::string harness_version;
    std::string evaluator;
    std::string evidence_sha256;
    bool evidence_sha256_verified{};
    std::string contamination_report_path;
    std::string contamination_report_sha256;
    bool contamination_report_present{};
    bool contamination_sha256_verified{};
    std::string model_checkpoint_sha256;
    std::string training_manifest_sha256;
    bool model_identity_verified{};
};

struct FrontierGateItem final {
    FrontierBenchmarkTarget target;
    bool present{};
    bool passed{};
    double best_score{};
    std::size_t examples{};
    std::string reason;
};

struct FrontierGateReport final {
    std::vector<FrontierGateItem> items;
    bool all_targets_passed{};
    bool broad_frontier_parity_proven{};
    std::size_t external_examples{};
    std::size_t minimum_external_examples{};
    std::size_t passed_targets{};
    std::vector<std::string> blocking_reasons;
};

[[nodiscard]] std::vector<FrontierBenchmarkTarget> leading_system_targets_2026();
[[nodiscard]] FrontierGateReport evaluate_frontier_gate(
    std::span<const FrontierBenchmarkEvidence> evidence,
    std::span<const FrontierBenchmarkTarget> targets
);
[[nodiscard]] std::string frontier_gate_json(const FrontierGateReport& report);

}  // namespace rlf::solstice
