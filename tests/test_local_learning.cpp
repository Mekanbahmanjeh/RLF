#include "test_framework.hpp"

#include "rlf/learning/local_learning.hpp"

#include <vector>

namespace {

[[nodiscard]] std::vector<rlf::learning::ModeEvidence> evidence() {
    return {
        {
            .mode_index = 0U,
            .mode_id = 20ULL,
            .resonance = 0.8,
            .normalized_contribution = 0.75,
            .baseline_quality = 0.1,
            .proposal_quality = 0.9,
            .prediction_quality = 0.8,
            .utility = 0.2,
            .improved_prediction = true,
        },
        {
            .mode_index = 1U,
            .mode_id = 10ULL,
            .resonance = 0.4,
            .normalized_contribution = 0.25,
            .baseline_quality = 0.1,
            .proposal_quality = 0.2,
            .prediction_quality = 0.8,
            .utility = -0.1,
            .improved_prediction = true,
        },
    };
}

}  // namespace

RLF_TEST_CASE("normalized responsibility sums to one") {
    const rlf::learning::NormalizedResponsibilityUpdate strategy;
    const std::vector<rlf::learning::ModeEvidence> mode_evidence = evidence();
    const std::vector<double> responsibilities =
        strategy.responsibilities(mode_evidence, {});

    RLF_CHECK(responsibilities.size() == 2U);
    RLF_CHECK_NEAR(
        responsibilities[0U] + responsibilities[1U],
        1.0,
        1.0e-12
    );
    RLF_CHECK(responsibilities[0U] > responsibilities[1U]);
}

RLF_TEST_CASE("winner-only responsibility updates exactly one mode") {
    const rlf::learning::WinnerOnlyUpdate strategy;
    const std::vector<rlf::learning::ModeEvidence> mode_evidence = evidence();
    const std::vector<double> responsibilities =
        strategy.responsibilities(mode_evidence, {});

    RLF_CHECK(responsibilities[0U] == 1.0);
    RLF_CHECK(responsibilities[1U] == 0.0);
}

RLF_TEST_CASE("resonance-weighted responsibility remains bounded") {
    const rlf::learning::ResonanceWeightedUpdate strategy;
    const std::vector<rlf::learning::ModeEvidence> mode_evidence = evidence();
    const std::vector<double> responsibilities =
        strategy.responsibilities(mode_evidence, {});

    for (const double responsibility : responsibilities) {
        RLF_CHECK(responsibility >= 0.0);
        RLF_CHECK(responsibility <= 1.0);
    }
    RLF_CHECK(responsibilities[0U] > responsibilities[1U]);
}
