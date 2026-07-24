#pragma once

#include "rlf/core/phase_vector.hpp"
#include "rlf/core/resonant_mode.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rlf::learning {

enum class StructuralEventType {
    created,
    split,
    merged,
    pruned,
};

[[nodiscard]] std::string_view to_string(
    StructuralEventType event_type
) noexcept;

struct StructuralEvent final {
    StructuralEventType type;
    std::uint64_t step;
    std::uint64_t primary_mode_id;
    std::vector<std::uint64_t> related_mode_ids;
    std::string reason;
    double metric;
};

struct StructuralStatistics final {
    std::uint64_t modes_created{0ULL};
    std::uint64_t modes_split{0ULL};
    std::uint64_t modes_merged{0ULL};
    std::uint64_t modes_pruned{0ULL};
};

struct StructuralLearningConfig final {
    bool enabled{false};
    bool enable_creation{true};
    bool enable_splitting{true};
    bool enable_merging{true};
    bool enable_pruning{true};
    double creation_minimum_resonance{0.15};
    double creation_prediction_error_threshold{0.5};
    float creation_confidence{0.1F};
    float creation_utility{0.0F};
    float creation_selectivity{1.0F};

    std::size_t correction_history_capacity{32U};
    std::size_t split_minimum_samples{12U};
    std::size_t split_minimum_cluster_size{4U};
    std::size_t split_kmeans_iterations{8U};
    double split_minimum_transformation_separation_radians{0.5};
    double split_minimum_context_separation_radians{0.25};
    double split_minimum_validation_gain_radians{0.1};
    double split_context_distance_weight{0.25};
    float split_child_confidence_scale{0.75F};

    double merge_maximum_key_error_radians{0.05};
    double merge_maximum_transformation_error_radians{0.05};
    double merge_maximum_history_dispersion_radians{0.1};

    std::uint64_t pruning_minimum_age_steps{1'000ULL};
    std::uint64_t pruning_maximum_inactive_steps{10'000ULL};
    std::uint64_t pruning_minimum_activation_count{2ULL};
    double pruning_maximum_utility{0.0};
    double pruning_harmful_update_ratio{0.8};
    std::uint64_t pruning_disabled_grace_steps{32ULL};
    std::size_t minimum_retained_modes{1U};
    std::size_t memory_budget_bytes{0U};
};

class StructuralLearner final {
public:
    explicit StructuralLearner(StructuralLearningConfig config);

    [[nodiscard]] const StructuralLearningConfig& config() const noexcept;
    [[nodiscard]] const StructuralStatistics& statistics() const noexcept;
    [[nodiscard]] std::span<const StructuralEvent> events() const noexcept;

    void record_correction(
        core::ResonantMode& mode,
        const core::PhaseVector& context,
        const core::PhaseVector& desired_transformation,
        double proposal_quality,
        bool improved_prediction,
        std::uint64_t step
    ) const;

    [[nodiscard]] bool consider_creation(
        std::vector<core::ResonantMode>& modes,
        std::uint64_t& next_mode_id,
        std::size_t maximum_modes,
        const core::PhaseVector& input,
        const core::PhaseVector& desired_transformation,
        double best_resonance,
        double prediction_error,
        std::uint64_t step
    );

    void maintain(
        std::vector<core::ResonantMode>& modes,
        std::uint64_t& next_mode_id,
        std::size_t maximum_modes,
        std::uint64_t step
    );

    [[nodiscard]] static std::size_t estimate_mode_bytes(
        const core::ResonantMode& mode
    ) noexcept;

private:
    StructuralLearningConfig config_;
    StructuralStatistics statistics_;
    std::vector<StructuralEvent> events_;

    void consider_splits(
        std::vector<core::ResonantMode>& modes,
        std::uint64_t& next_mode_id,
        std::size_t maximum_modes,
        std::uint64_t step
    );
    void consider_merges(
        std::vector<core::ResonantMode>& modes,
        std::uint64_t step
    );
    void consider_pruning(
        std::vector<core::ResonantMode>& modes,
        std::uint64_t step
    );
};

}  // namespace rlf::learning
