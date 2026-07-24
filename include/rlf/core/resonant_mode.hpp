#pragma once

#include "rlf/core/phase_vector.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rlf::core {

struct CorrectionSummary final {
    PhaseVector context;
    PhaseVector desired_transformation;
    double proposal_quality;
    bool improved_prediction;
    std::uint64_t step;
};

struct ResonantMode final {
    std::uint64_t id;
    PhaseVector context_key;
    PhaseVector transformation;
    float selectivity{1.0F};
    float confidence{0.25F};
    float utility{0.0F};
    std::uint64_t activation_count{0ULL};
    std::uint64_t successful_update_count{0ULL};
    std::uint64_t unsuccessful_update_count{0ULL};
    std::uint64_t creation_step{0ULL};
    std::uint64_t last_used_step{0ULL};
    bool enabled{true};
    std::vector<CorrectionSummary> recent_corrections;

    ResonantMode(
        std::uint64_t mode_id,
        PhaseVector context,
        PhaseVector phase_transformation,
        float mode_selectivity = 1.0F,
        float mode_confidence = 0.25F,
        float mode_utility = 0.0F,
        std::uint64_t mode_creation_step = 0ULL
    );

    [[nodiscard]] double resonance(const PhaseVector& state) const;
    [[nodiscard]] PhaseVector propose(const PhaseVector& state) const;
};

}  // namespace rlf::core
