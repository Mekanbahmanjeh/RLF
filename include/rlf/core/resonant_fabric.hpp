#pragma once

#include "rlf/core/resonant_mode.hpp"
#include "rlf/core/settling.hpp"
#include "rlf/learning/local_learning.hpp"
#include "rlf/learning/structural_learning.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace rlf::retrieval {
class ModeRetriever;
}

namespace rlf::core {

struct FabricConfig final {
    std::size_t dimension{1'024U};
    std::size_t maximum_modes{1'000'000U};
    SettlingConfig settling{};
    learning::StructuralLearningConfig structural_learning{};
};

struct LearningResult final {
    SettleResult prediction;
    PhaseVector desired_transformation;
    double prediction_similarity;
    double prediction_error;
    std::vector<learning::ModeEvidence> evidence;
    std::vector<double> responsibilities;
    std::vector<std::uint64_t> updated_mode_ids;
    std::vector<learning::StructuralEvent> structural_events;
};

class ResonantFabric final {
public:
    explicit ResonantFabric(FabricConfig config);
    ResonantFabric(
        FabricConfig config,
        std::unique_ptr<retrieval::ModeRetriever> retriever,
        std::unique_ptr<SettlingPolicy> settling_policy,
        std::unique_ptr<learning::LocalUpdateStrategy> update_strategy
    );
    ~ResonantFabric();

    ResonantFabric(ResonantFabric&&) noexcept;
    ResonantFabric& operator=(ResonantFabric&&) noexcept;
    ResonantFabric(const ResonantFabric&) = delete;
    ResonantFabric& operator=(const ResonantFabric&) = delete;

    [[nodiscard]] const FabricConfig& config() const noexcept;
    [[nodiscard]] std::span<const ResonantMode> modes() const noexcept;
    [[nodiscard]] std::uint64_t training_step() const noexcept;
    [[nodiscard]] std::string_view update_strategy_name() const noexcept;
    [[nodiscard]] const learning::StructuralStatistics&
    structural_statistics() const noexcept;
    [[nodiscard]] std::span<const learning::StructuralEvent>
    structural_events() const noexcept;

    void add_mode(ResonantMode mode);
    void set_update_strategy(
        std::unique_ptr<learning::LocalUpdateStrategy> update_strategy
    );
    void maintain_structure();

    [[nodiscard]] SettleResult settle(
        const PhaseVector& input,
        bool capture_trace = false,
        const HaltCondition& halt_condition = {}
    );
    [[nodiscard]] LearningResult learn(
        const PhaseVector& input,
        const PhaseVector& target,
        const learning::LocalLearningConfig& learning_config = {},
        bool capture_trace = false
    );

private:
    FabricConfig config_;
    std::vector<ResonantMode> modes_;
    std::unique_ptr<retrieval::ModeRetriever> retriever_;
    std::unique_ptr<SettlingPolicy> settling_policy_;
    std::unique_ptr<learning::LocalUpdateStrategy> update_strategy_;
    learning::StructuralLearner structural_learner_;
    std::uint64_t training_step_{0ULL};
    std::uint64_t execution_step_{0ULL};
    std::uint64_t next_mode_id_{1ULL};
};

}  // namespace rlf::core
