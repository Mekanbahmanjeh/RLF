#include "rlf/learning/local_learning.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace rlf::learning {
namespace {

void validate_config(const LocalLearningConfig& config) {
    const auto valid_rate = [](const double rate) {
        return std::isfinite(rate) && rate >= 0.0 && rate <= 1.0;
    };
    if (!valid_rate(config.transformation_learning_rate) ||
        !valid_rate(config.context_learning_rate) ||
        !valid_rate(config.confidence_learning_rate) ||
        !valid_rate(config.utility_learning_rate)) {
        throw std::invalid_argument("local learning rates must be in [0, 1]");
    }
    if (!std::isfinite(config.utility_influence) ||
        config.utility_influence < 0.0) {
        throw std::invalid_argument(
            "utility influence must be finite and non-negative"
        );
    }
    if (!std::isfinite(config.harmful_update_multiplier) ||
        config.harmful_update_multiplier <= 0.0) {
        throw std::invalid_argument(
            "harmful update multiplier must be finite and positive"
        );
    }
}

[[nodiscard]] double evidence_score(
    const ModeEvidence& item,
    const LocalLearningConfig& config
) {
    const double resonance = std::clamp(item.resonance, 0.0, 1.0);
    const double contribution = std::clamp(
        item.normalized_contribution,
        0.0,
        1.0
    );
    const double proposal_quality = std::clamp(
        item.proposal_quality,
        0.0,
        1.0
    );
    const double prediction_quality = std::clamp(
        item.prediction_quality,
        0.0,
        1.0
    );
    const double quality_factor =
        (0.25 + (0.75 * proposal_quality)) *
        (0.25 + (0.75 * prediction_quality));
    const double utility_factor = config.use_utility
        ? std::clamp(
              1.0 + (config.utility_influence * item.utility),
              0.1,
              2.0
          )
        : 1.0;
    const double outcome_factor = item.improved_prediction
        ? 1.0
        : config.harmful_update_multiplier;
    return std::max(
        0.0,
        resonance * contribution * quality_factor * utility_factor *
            outcome_factor
    );
}

[[nodiscard]] std::vector<double> scores(
    const std::span<const ModeEvidence> evidence,
    const LocalLearningConfig& config
) {
    validate_config(config);
    std::vector<double> result;
    result.reserve(evidence.size());
    for (const ModeEvidence& item : evidence) {
        result.push_back(evidence_score(item, config));
    }
    return result;
}

}  // namespace

std::string_view ResonanceWeightedUpdate::name() const noexcept {
    return "resonance_weighted";
}

std::vector<double> ResonanceWeightedUpdate::responsibilities(
    const std::span<const ModeEvidence> evidence,
    const LocalLearningConfig& config
) const {
    std::vector<double> result = scores(evidence, config);
    for (double& responsibility : result) {
        responsibility = std::clamp(responsibility, 0.0, 1.0);
    }
    return result;
}

std::string_view WinnerOnlyUpdate::name() const noexcept {
    return "winner_only";
}

std::vector<double> WinnerOnlyUpdate::responsibilities(
    const std::span<const ModeEvidence> evidence,
    const LocalLearningConfig& config
) const {
    const std::vector<double> evidence_scores = scores(evidence, config);
    std::vector<double> result(evidence.size(), 0.0);
    if (evidence_scores.empty()) {
        return result;
    }

    std::size_t winner_index = 0U;
    for (std::size_t index = 1U;
         index < evidence_scores.size();
         ++index) {
        if (evidence_scores[index] > evidence_scores[winner_index] ||
            (evidence_scores[index] == evidence_scores[winner_index] &&
             evidence[index].mode_id < evidence[winner_index].mode_id)) {
            winner_index = index;
        }
    }
    if (evidence_scores[winner_index] > 0.0) {
        result[winner_index] = 1.0;
    }
    return result;
}

std::string_view NormalizedResponsibilityUpdate::name() const noexcept {
    return "normalized_responsibility";
}

std::vector<double> NormalizedResponsibilityUpdate::responsibilities(
    const std::span<const ModeEvidence> evidence,
    const LocalLearningConfig& config
) const {
    std::vector<double> result = scores(evidence, config);
    double total = 0.0;
    for (const double responsibility : result) {
        total += responsibility;
    }
    if (total <= 0.0) {
        std::fill(result.begin(), result.end(), 0.0);
        return result;
    }
    for (double& responsibility : result) {
        responsibility /= total;
    }
    return result;
}

}  // namespace rlf::learning
