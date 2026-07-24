#pragma once

#include "rlf/core/phase_vector.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace rlf::core {

enum class StopReason {
    none,
    converged,
    confidence_threshold,
    cycle_limit,
    explicit_halt,
    no_active_modes,
};

[[nodiscard]] std::string_view to_string(StopReason reason) noexcept;

struct SettlingConfig final {
    std::size_t candidate_count{256U};
    std::size_t active_count{64U};
    std::size_t maximum_cycles{64U};
    std::size_t minimum_cycles{1U};
    double minimum_resonance{0.0};
    double convergence_tolerance_radians{1.0e-5};
    std::optional<double> confidence_threshold{};
    double input_weight{0.25};
    double previous_state_weight{0.25};
    double proposal_weight_scale{1.0};
    double utility_weight{0.1};
};

struct ModeParticipation final {
    std::size_t mode_index;
    std::uint64_t mode_id;
    double resonance;
    double normalized_contribution;
    PhaseVector proposal_state;
};

struct SettlingCycleTrace final {
    std::size_t cycle_number{};
    std::vector<std::uint64_t> retrieved_mode_ids;
    std::vector<double> resonance_scores;
    std::vector<std::uint64_t> active_mode_ids;
    std::vector<double> proposal_weights;
    double state_change_radians{};
    double coherence{};
    double confidence{};
    StopReason stopping_reason{StopReason::none};
};

struct ExecutionTrace final {
    std::vector<SettlingCycleTrace> cycles;
    StopReason stopping_reason{StopReason::none};
};

using HaltCondition = std::function<bool(const SettlingCycleTrace&)>;

class SettlingPolicy {
public:
    virtual ~SettlingPolicy() = default;

    [[nodiscard]] virtual PhaseVector next_state(
        const PhaseVector& input,
        const PhaseVector& previous_state,
        std::span<const PhaseVector> proposals,
        std::span<const double> proposal_weights,
        const SettlingConfig& config
    ) const = 0;
};

class StableCircularSettlingPolicy final : public SettlingPolicy {
public:
    [[nodiscard]] PhaseVector next_state(
        const PhaseVector& input,
        const PhaseVector& previous_state,
        std::span<const PhaseVector> proposals,
        std::span<const double> proposal_weights,
        const SettlingConfig& config
    ) const override;
};

struct SettleResult final {
    PhaseVector state;
    StopReason stopping_reason;
    std::size_t cycles;
    double coherence;
    double confidence;
    std::vector<ModeParticipation> participants;
    std::optional<ExecutionTrace> trace;
};

}  // namespace rlf::core
