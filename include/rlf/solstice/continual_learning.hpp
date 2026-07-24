#pragma once

#include "rlf/solstice/sparse_router.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rlf::solstice {

struct ContinualLearningConfig final {
    std::size_t feature_dimensions{64U};
    std::size_t maximum_prototypes{1'000'000U};
    std::size_t replay_capacity{250'000U};
    std::size_t consolidation_interval{1'024U};
    std::size_t replay_batch_size{256U};
    double base_learning_rate{0.15};
    double stability_strength{4.0};
    double novelty_threshold{0.72};
    double contrastive_margin{0.15};
    SparseRouterConfig router;
};

struct ContinualPrototype final {
    std::uint64_t id{};
    std::string task;
    std::string label;
    std::vector<float> centroid;
    std::uint64_t support{1U};
    double importance{};
    double plasticity{1.0};
    std::uint64_t last_update_step{};
};

struct ReplayExperience final {
    std::uint64_t id{};
    std::string task;
    std::string label;
    std::vector<float> features;
    double priority{1.0};
    std::uint64_t seen_step{};
};

struct ContinualPrediction final {
    std::string label;
    double confidence{};
    double novelty{1.0};
    std::uint64_t prototype_id{};
    std::uint64_t candidates_examined{};
    std::uint64_t exhaustive_candidates{};
};

struct ContinualLearningStats final {
    std::size_t prototypes{};
    std::size_t replay_experiences{};
    std::uint64_t learning_steps{};
    std::uint64_t consolidations{};
    double average_importance{};
    double average_plasticity{};
};

struct ContinualLearningSnapshot final {
    ContinualLearningConfig config;
    std::uint64_t next_prototype_id{1U};
    std::uint64_t next_experience_id{1U};
    std::uint64_t step{};
    std::uint64_t consolidations{};
    std::vector<ContinualPrototype> prototypes;
    std::vector<ReplayExperience> replay;
};

class ContinualLearningFabric final {
public:
    explicit ContinualLearningFabric(ContinualLearningConfig config = {});

    ContinualPrediction learn(
        std::string_view task,
        std::string_view label,
        std::span<const float> features,
        double sample_weight = 1.0
    );
    [[nodiscard]] ContinualPrediction predict(
        std::string_view task,
        std::span<const float> features
    ) const;
    void consolidate();

    [[nodiscard]] std::span<const ContinualPrototype> prototypes() const noexcept;
    [[nodiscard]] std::span<const ReplayExperience> replay() const noexcept;
    [[nodiscard]] const ContinualLearningConfig& config() const noexcept;
    [[nodiscard]] ContinualLearningStats stats() const noexcept;
    [[nodiscard]] std::uint64_t deterministic_hash() const noexcept;

    [[nodiscard]] ContinualLearningSnapshot snapshot() const;
    [[nodiscard]] static ContinualLearningFabric from_snapshot(
        ContinualLearningSnapshot snapshot
    );

private:
    [[nodiscard]] ContinualPrediction predict_internal(
        std::string_view task,
        std::span<const float> features,
        bool allow_empty
    ) const;
    void update_router() const;
    void add_replay(
        std::string_view task,
        std::string_view label,
        std::span<const float> features,
        double priority
    );
    void apply_update(
        ContinualPrototype& prototype,
        std::span<const float> features,
        double sample_weight,
        bool consolidation
    );

    ContinualLearningConfig config_;
    std::uint64_t next_prototype_id_{1U};
    std::uint64_t next_experience_id_{1U};
    std::uint64_t step_{};
    std::uint64_t consolidations_{};
    std::vector<ContinualPrototype> prototypes_;
    std::vector<ReplayExperience> replay_;
    mutable SparseRoutingIndex router_;
    mutable bool router_dirty_{true};
};

}  // namespace rlf::solstice
