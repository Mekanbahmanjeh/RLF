#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace rlf::experiments {

struct OperatorCompositionConfig final {
    std::uint64_t seed{0x524C464FULL};
    std::size_t dimension{48U};
    std::size_t context_dimensions{8U};
    std::size_t training_examples{32U};
    std::size_t evaluation_examples{64U};
    double noise_radians{0.08};
};

struct OperatorCompositionCaseResult final {
    std::string name;
    std::string category;
    double operator_extension_accuracy{};
    double phase_offset_accuracy{};
    double nearest_neighbor_accuracy{};
    double supervised_operator_accuracy{};
    double oracle_accuracy{};
};

struct OperatorCompositionResult final {
    std::uint64_t seed{};
    std::size_t dimension{};
    std::size_t context_dimensions{};
    std::size_t training_examples{};
    std::size_t evaluation_examples{};
    std::vector<OperatorCompositionCaseResult> cases;
    double operator_familiar_accuracy{};
    double operator_unseen_composition_accuracy{};
    double phase_offset_familiar_accuracy{};
    double phase_offset_unseen_composition_accuracy{};
    double nearest_neighbor_familiar_accuracy{};
    double nearest_neighbor_unseen_composition_accuracy{};
    double supervised_familiar_accuracy{};
    double supervised_unseen_composition_accuracy{};
    double oracle_unseen_composition_accuracy{};
    double noisy_input_accuracy{};
    double ambiguous_context_accuracy{};
    double conflicting_mode_accuracy{};
    std::size_t operator_mode_count{};
    std::size_t operator_selected_family_count{};
    std::size_t operator_bytes{};
    double training_seconds{};
    double inference_seconds{};
    std::uint64_t deterministic_run_hash{};
    std::string scientific_decision;
};

[[nodiscard]] OperatorCompositionResult run_operator_composition(
    const OperatorCompositionConfig& config
);
void write_operator_composition_json(
    std::ostream& output,
    const OperatorCompositionResult& result
);

}  // namespace rlf::experiments
