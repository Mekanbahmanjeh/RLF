#include "rlf/core/settling.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace rlf::core {

std::string_view to_string(const StopReason reason) noexcept {
    switch (reason) {
        case StopReason::none:
            return "none";
        case StopReason::converged:
            return "converged";
        case StopReason::confidence_threshold:
            return "confidence_threshold";
        case StopReason::cycle_limit:
            return "cycle_limit";
        case StopReason::explicit_halt:
            return "explicit_halt";
        case StopReason::no_active_modes:
            return "no_active_modes";
    }
    return "unknown";
}

PhaseVector StableCircularSettlingPolicy::next_state(
    const PhaseVector& input,
    const PhaseVector& previous_state,
    const std::span<const PhaseVector> proposals,
    const std::span<const double> proposal_weights,
    const SettlingConfig& config
) const {
    if (proposals.size() != proposal_weights.size()) {
        throw std::invalid_argument(
            "settling requires one weight per proposal"
        );
    }
    if (input.size() != previous_state.size()) {
        throw std::invalid_argument(
            "settling input and previous state dimensions must match"
        );
    }

    std::vector<PhaseVector> contributors;
    std::vector<float> weights;
    contributors.reserve(proposals.size() + 2U);
    weights.reserve(proposals.size() + 2U);

    if (config.input_weight > 0.0) {
        contributors.push_back(input);
        weights.push_back(static_cast<float>(config.input_weight));
    }
    if (config.previous_state_weight > 0.0) {
        contributors.push_back(previous_state);
        weights.push_back(static_cast<float>(config.previous_state_weight));
    }

    for (std::size_t proposal_index = 0U;
         proposal_index < proposals.size();
         ++proposal_index) {
        if (proposals[proposal_index].size() != input.size()) {
            throw std::invalid_argument(
                "settling proposal dimensions must match the input"
            );
        }
        const double scaled_weight =
            proposal_weights[proposal_index] * config.proposal_weight_scale;
        if (!std::isfinite(scaled_weight) || scaled_weight < 0.0) {
            throw std::invalid_argument(
                "settling proposal weights must be finite and non-negative"
            );
        }
        if (scaled_weight > 0.0) {
            contributors.push_back(proposals[proposal_index]);
            weights.push_back(static_cast<float>(scaled_weight));
        }
    }

    if (contributors.empty()) {
        throw std::invalid_argument(
            "settling policy requires a positive contributor weight"
        );
    }
    return PhaseVector::weighted_circular_average(contributors, weights);
}

}  // namespace rlf::core
