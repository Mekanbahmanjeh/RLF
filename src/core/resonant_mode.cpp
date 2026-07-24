#include "rlf/core/resonant_mode.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace rlf::core {

ResonantMode::ResonantMode(
    const std::uint64_t mode_id,
    PhaseVector context,
    PhaseVector phase_transformation,
    const float mode_selectivity,
    const float mode_confidence,
    const float mode_utility,
    const std::uint64_t mode_creation_step
)
    : id(mode_id),
      context_key(std::move(context)),
      transformation(std::move(phase_transformation)),
      selectivity(mode_selectivity),
      confidence(mode_confidence),
      utility(mode_utility),
      creation_step(mode_creation_step),
      last_used_step(mode_creation_step) {
    if (id == 0ULL) {
        throw std::invalid_argument("resonant mode IDs must be non-zero");
    }
    if (context_key.size() != transformation.size()) {
        throw std::invalid_argument(
            "resonant mode key and transformation dimensions must match"
        );
    }
    if (!std::isfinite(selectivity) || selectivity <= 0.0F) {
        throw std::invalid_argument(
            "resonant mode selectivity must be finite and positive"
        );
    }
    if (!std::isfinite(confidence) ||
        confidence < 0.0F ||
        confidence > 1.0F) {
        throw std::invalid_argument(
            "resonant mode confidence must be in [0, 1]"
        );
    }
    if (!std::isfinite(utility)) {
        throw std::invalid_argument("resonant mode utility must be finite");
    }
}

double ResonantMode::resonance(const PhaseVector& state) const {
    if (!enabled) {
        return 0.0;
    }
    const double base_resonance = context_key.similarity(state);
    return std::clamp(
        std::pow(base_resonance, static_cast<double>(selectivity)),
        0.0,
        1.0
    );
}

PhaseVector ResonantMode::propose(const PhaseVector& state) const {
    return state.composed(transformation);
}

}  // namespace rlf::core
