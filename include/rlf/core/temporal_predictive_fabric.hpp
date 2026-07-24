#pragma once

#include "rlf/core/phase_vector.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rlf::core {

struct TemporalFabricConfig final {
    std::size_t dimension{48U};
    std::size_t maximum_prototypes{65'536U};
    std::size_t maximum_contexts{1'000'000U};
    std::size_t maximum_context_order{12U};
    std::size_t minimum_context_support{2U};
    std::size_t maximum_options{4'096U};
    std::size_t minimum_option_length{3U};
    std::size_t maximum_option_length{12U};
    std::size_t minimum_option_support{4U};
    std::size_t option_prefix_minimum{2U};
    std::size_t maximum_prediction_outcomes{16U};
    double prototype_merge_distance{0.08};
    double recent_decay{0.995};
    double recent_weight{0.65};
    double smoothing{0.10};
    double minimum_option_confidence{0.70};
    double minimum_option_gain{0.05};
    double surprise_slow_rate{0.01};
    double surprise_fast_rate{0.20};
    double change_threshold{0.75};
    std::size_t change_cooldown{32U};
};

struct TemporalPrototype final {
    std::uint64_t id{};
    PhaseVector key{std::vector<float>{0.0F}};
    std::uint64_t support{1ULL};
    std::uint64_t creation_step{};
    std::uint64_t last_used_step{};
};

struct TemporalOutcomeCount final {
    std::uint64_t prototype_id{};
    std::uint64_t total_count{};
    double recent_count{};
    std::uint64_t last_update_step{};
};

struct TemporalContext final {
    std::uint64_t id{};
    std::vector<std::uint64_t> history;
    std::uint64_t support{};
    std::uint64_t creation_step{};
    std::uint64_t last_used_step{};
    std::vector<TemporalOutcomeCount> outcomes;
};

struct TemporalOption final {
    std::uint64_t id{};
    std::vector<std::uint64_t> sequence;
    std::uint64_t support{};
    double confidence{};
    double predictive_gain{};
    double compression_gain{};
    std::uint64_t creation_step{};
    std::uint64_t last_used_step{};
};

struct TemporalPredictionOutcome final {
    std::uint64_t prototype_id{};
    double probability{};
};

struct TemporalPrediction final {
    std::vector<TemporalPredictionOutcome> outcomes;
    std::size_t context_order{};
    std::uint64_t context_id{};
    bool used_option{false};
    std::uint64_t option_id{};
    double uncertainty{1.0};
    std::size_t context_comparisons{};
    std::size_t option_comparisons{};
};

struct TemporalForecast final {
    std::vector<std::uint64_t> prototype_ids;
    std::size_t decision_operations{};
    std::size_t option_uses{};
    std::size_t predicted_tokens{};
};

struct TemporalStepResult final {
    std::uint64_t prototype_id{};
    bool prototype_created{false};
    bool prediction_available{false};
    double actual_probability{};
    double surprise{};
    bool change_detected{false};
    TemporalPrediction prediction;
};

struct TemporalFabricStats final {
    std::uint64_t observations_seen{};
    std::uint64_t sequences_seen{};
    std::uint64_t prototypes_created{};
    std::uint64_t prototypes_merged{};
    std::uint64_t contexts_created{};
    std::uint64_t contexts_updated{};
    std::uint64_t option_candidates{};
    std::uint64_t options_created{};
    std::uint64_t prediction_queries{};
    std::uint64_t context_comparisons{};
    std::uint64_t option_comparisons{};
    std::uint64_t change_points_detected{};
};

struct TemporalFabricSnapshot final {
    TemporalFabricConfig config;
    std::uint64_t seed{};
    std::uint64_t training_step{};
    std::uint64_t next_prototype_id{};
    std::uint64_t next_context_id{};
    std::uint64_t next_option_id{};
    std::vector<TemporalPrototype> prototypes;
    std::vector<TemporalContext> contexts;
    std::vector<TemporalOption> options;
    std::vector<std::uint64_t> recent_history;
    double slow_surprise{};
    double fast_surprise{};
    bool surprise_initialized{false};
    std::uint64_t last_change_step{};
    TemporalFabricStats stats;
};

class TemporalPredictiveFabric final {
public:
    explicit TemporalPredictiveFabric(
        TemporalFabricConfig config,
        std::uint64_t seed
    );

    [[nodiscard]] const TemporalFabricConfig& config() const noexcept;
    [[nodiscard]] std::uint64_t seed() const noexcept;
    [[nodiscard]] std::uint64_t training_step() const noexcept;
    [[nodiscard]] std::span<const TemporalPrototype> prototypes() const noexcept;
    [[nodiscard]] std::span<const TemporalContext> contexts() const noexcept;
    [[nodiscard]] std::span<const TemporalOption> options() const noexcept;
    [[nodiscard]] const TemporalFabricStats& stats() const noexcept;

    void reset_sequence();
    [[nodiscard]] TemporalStepResult observe(const PhaseVector& observation);
    void observe_sequence(std::span<const PhaseVector> sequence);
    void discover_options();

    [[nodiscard]] std::optional<std::uint64_t> match_prototype(
        const PhaseVector& observation,
        std::size_t* comparisons = nullptr
    ) const;
    [[nodiscard]] TemporalPrediction predict_next(
        std::span<const std::uint64_t> history,
        bool use_options = true
    ) const;
    [[nodiscard]] TemporalPrediction predict_next(
        std::span<const PhaseVector> history,
        bool use_options = true
    ) const;
    [[nodiscard]] TemporalForecast forecast(
        std::span<const std::uint64_t> history,
        std::size_t horizon,
        bool use_options = true
    ) const;

    [[nodiscard]] const TemporalPrototype& prototype_by_id(
        std::uint64_t id
    ) const;
    [[nodiscard]] std::size_t estimated_storage_bytes() const noexcept;
    [[nodiscard]] std::uint64_t deterministic_hash() const noexcept;

    [[nodiscard]] TemporalFabricSnapshot snapshot() const;
    [[nodiscard]] static TemporalPredictiveFabric from_snapshot(
        TemporalFabricSnapshot snapshot
    );

private:
    struct SequenceHash final {
        [[nodiscard]] std::size_t operator()(
            const std::vector<std::uint64_t>& value
        ) const noexcept;
    };

    [[nodiscard]] std::pair<std::uint64_t, bool> encode_or_create(
        const PhaseVector& observation
    );
    [[nodiscard]] std::optional<std::size_t> prototype_index(
        std::uint64_t id
    ) const noexcept;
    [[nodiscard]] std::optional<std::size_t> context_index(
        std::span<const std::uint64_t> history
    ) const;
    [[nodiscard]] std::size_t get_or_create_context(
        std::span<const std::uint64_t> history
    );
    void update_context(
        std::span<const std::uint64_t> history,
        std::uint64_t next_id
    );
    [[nodiscard]] double effective_recent(
        const TemporalOutcomeCount& outcome
    ) const noexcept;
    [[nodiscard]] std::optional<std::pair<const TemporalOption*, std::size_t>>
    best_option_continuation(
        std::span<const std::uint64_t> history
    ) const;
    [[nodiscard]] bool update_surprise(double surprise);
    void rebuild_indices();
    void validate_snapshot() const;

    TemporalFabricConfig config_;
    std::uint64_t seed_{};
    std::uint64_t training_step_{};
    std::uint64_t next_prototype_id_{1ULL};
    std::uint64_t next_context_id_{1ULL};
    std::uint64_t next_option_id_{1ULL};
    std::vector<TemporalPrototype> prototypes_;
    std::vector<TemporalContext> contexts_;
    std::vector<TemporalOption> options_;
    std::vector<std::uint64_t> recent_history_;
    std::vector<std::vector<std::uint64_t>> mining_sequences_;
    std::unordered_map<std::uint64_t, std::size_t> prototype_index_by_id_;
    std::unordered_map<std::vector<std::uint64_t>, std::size_t, SequenceHash>
        context_index_by_history_;
    double slow_surprise_{};
    double fast_surprise_{};
    bool surprise_initialized_{false};
    std::uint64_t last_change_step_{};
    mutable TemporalFabricStats stats_;
};

[[nodiscard]] std::string_view rlf4_architecture_name() noexcept;

}  // namespace rlf::core
