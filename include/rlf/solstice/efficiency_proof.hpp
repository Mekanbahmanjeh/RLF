#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace rlf::solstice {

struct EfficiencyProofConfig final {
    std::size_t schema_instances{10'001U};
    std::size_t routing_vectors{524'288U};
    std::size_t routing_dimensions{16U};
    std::size_t routing_queries{256U};
    std::size_t routing_trials{3U};
    std::size_t continual_classes{32U};
    std::size_t continual_tasks{12U};
    double target_efficiency_ratio{10'000.0};
    double minimum_accuracy{0.999};
};

struct EfficiencyMetric final {
    std::string name;
    std::string scope;
    std::string unit;
    double baseline_cost{};
    double rlf_cost{};
    double ratio{};
    double baseline_accuracy{};
    double rlf_accuracy{};
    double required_ratio{};
    double required_accuracy{};
    bool passed{};
    std::string qualification;
};

struct CapabilityProof final {
    std::string name;
    double score{};
    double required_score{};
    bool passed{};
    std::string qualification;
};

struct EfficiencyProofReport final {
    std::string suite;
    std::vector<EfficiencyMetric> efficiency_metrics;
    std::vector<CapabilityProof> capability_proofs;
    bool narrow_ten_thousand_x_proven{};
    bool general_learning_efficiency_proven{};
    bool frontier_parity_proven{};
    bool all_internal_proofs_passed{};
    std::string claim_boundary;
};

[[nodiscard]] EfficiencyProofReport run_efficiency_proofs(
    const EfficiencyProofConfig& config = {}
);

void write_efficiency_proof_json(
    std::ostream& output,
    const EfficiencyProofReport& report
);

}  // namespace rlf::solstice
