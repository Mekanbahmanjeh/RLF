#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace rlf::learning {

struct ModeEvidence final {
    std::size_t mode_index;
    std::uint64_t mode_id;
    double resonance;
    double normalized_contribution;
    double baseline_quality;
    double proposal_quality;
    double prediction_quality;
    double utility;
    bool improved_prediction;
};

struct LocalLearningConfig final {
    double transformation_learning_rate{0.2};
    double context_learning_rate{0.02};
    double confidence_learning_rate{0.05};
    double utility_learning_rate{0.05};
    double utility_influence{0.25};
    double harmful_update_multiplier{1.25};
    bool use_utility{true};
};

class LocalUpdateStrategy {
public:
    virtual ~LocalUpdateStrategy() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual std::vector<double> responsibilities(
        std::span<const ModeEvidence> evidence,
        const LocalLearningConfig& config
    ) const = 0;
};

class ResonanceWeightedUpdate final : public LocalUpdateStrategy {
public:
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] std::vector<double> responsibilities(
        std::span<const ModeEvidence> evidence,
        const LocalLearningConfig& config
    ) const override;
};

class WinnerOnlyUpdate final : public LocalUpdateStrategy {
public:
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] std::vector<double> responsibilities(
        std::span<const ModeEvidence> evidence,
        const LocalLearningConfig& config
    ) const override;
};

class NormalizedResponsibilityUpdate final : public LocalUpdateStrategy {
public:
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] std::vector<double> responsibilities(
        std::span<const ModeEvidence> evidence,
        const LocalLearningConfig& config
    ) const override;
};

}  // namespace rlf::learning
