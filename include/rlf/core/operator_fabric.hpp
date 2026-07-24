#pragma once

#include "rlf/core/transformation_operator.hpp"

#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>
#include <vector>

namespace rlf::core {

struct OperatorFabricConfig final {
    std::size_t dimension{64U};
    std::size_t context_dimensions{16U};
    std::size_t history_capacity{64U};
    std::size_t candidate_count{32U};
    double creation_resonance_threshold{0.8};
    double context_learning_rate{0.05};
    double complexity_penalty{0.002};
};

struct OperatorMode final {
    std::uint64_t id;
    PhaseVector context_key;
    OperatorFamily family;
    TransformationOperator transformation;
    double confidence{0.0};
    double training_error{std::numbers::pi_v<double>};
    std::uint64_t activation_count{0ULL};
    std::uint64_t update_count{0ULL};
    std::vector<OperatorTrainingExample> recent_examples;

    [[nodiscard]] double resonance(std::span<const float> context) const;
    [[nodiscard]] PhaseVector propose(const PhaseVector& input) const;
};

struct OperatorPrediction final {
    PhaseVector state;
    std::uint64_t mode_id;
    OperatorFamily family;
    double resonance;
    double confidence;
};

class OperatorFabric final {
public:
    explicit OperatorFabric(OperatorFabricConfig config);

    [[nodiscard]] const OperatorFabricConfig& config() const noexcept;
    [[nodiscard]] std::span<const OperatorMode> modes() const noexcept;
    void learn(const PhaseVector& input, const PhaseVector& target);
    [[nodiscard]] OperatorPrediction predict(
        const PhaseVector& input
    ) const;

private:
    [[nodiscard]] std::vector<std::size_t> retrieve_bank(
        const PhaseVector& input
    ) const;
    void create_bank(const PhaseVector& input);

    OperatorFabricConfig config_;
    std::vector<OperatorMode> modes_;
    std::uint64_t next_mode_id_{1ULL};
};

}  // namespace rlf::core
