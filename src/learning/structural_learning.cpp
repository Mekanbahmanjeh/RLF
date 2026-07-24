#include "rlf/learning/structural_learning.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rlf::learning {
namespace {

struct ClusterModel final {
    std::vector<std::size_t> first_indices;
    std::vector<std::size_t> second_indices;
    core::PhaseVector first_context;
    core::PhaseVector second_context;
    core::PhaseVector first_transformation;
    core::PhaseVector second_transformation;
    double validation_gain;
};

void validate_config(const StructuralLearningConfig& config) {
    const auto finite_unit = [](const double value) {
        return std::isfinite(value) && value >= 0.0 && value <= 1.0;
    };
    const auto finite_non_negative = [](const double value) {
        return std::isfinite(value) && value >= 0.0;
    };
    if (!finite_unit(config.creation_minimum_resonance) ||
        !finite_unit(config.creation_prediction_error_threshold) ||
        !std::isfinite(config.creation_confidence) ||
        config.creation_confidence < 0.0F ||
        config.creation_confidence > 1.0F ||
        !std::isfinite(config.creation_utility) ||
        !std::isfinite(config.creation_selectivity) ||
        config.creation_selectivity <= 0.0F) {
        throw std::invalid_argument(
            "invalid structural mode-creation configuration"
        );
    }
    if (config.correction_history_capacity == 0U ||
        config.split_minimum_samples < 2U ||
        config.split_minimum_cluster_size == 0U ||
        config.split_kmeans_iterations == 0U ||
        config.split_minimum_samples >
            config.correction_history_capacity ||
        (2U * config.split_minimum_cluster_size) >
            config.correction_history_capacity) {
        throw std::invalid_argument(
            "invalid structural split sample configuration"
        );
    }
    if (!finite_non_negative(
            config.split_minimum_transformation_separation_radians
        ) ||
        !finite_non_negative(
            config.split_minimum_context_separation_radians
        ) ||
        !finite_non_negative(
            config.split_minimum_validation_gain_radians
        ) ||
        !finite_non_negative(config.split_context_distance_weight) ||
        !std::isfinite(config.split_child_confidence_scale) ||
        config.split_child_confidence_scale <= 0.0F ||
        config.split_child_confidence_scale > 1.0F ||
        !finite_non_negative(config.merge_maximum_key_error_radians) ||
        !finite_non_negative(
            config.merge_maximum_transformation_error_radians
        ) ||
        !finite_non_negative(
            config.merge_maximum_history_dispersion_radians
        ) ||
        !std::isfinite(config.pruning_maximum_utility) ||
        !finite_unit(config.pruning_harmful_update_ratio)) {
        throw std::invalid_argument(
            "invalid structural split, merge, or prune thresholds"
        );
    }
}

[[nodiscard]] core::PhaseVector average_vectors(
    const std::vector<core::CorrectionSummary>& samples,
    const std::vector<std::size_t>& indices,
    const bool use_context
) {
    std::vector<core::PhaseVector> vectors;
    std::vector<float> weights(indices.size(), 1.0F);
    vectors.reserve(indices.size());
    for (const std::size_t index : indices) {
        vectors.push_back(
            use_context
                ? samples[index].context
                : samples[index].desired_transformation
        );
    }
    return core::PhaseVector::weighted_circular_average(vectors, weights);
}

[[nodiscard]] double assignment_distance(
    const core::CorrectionSummary& sample,
    const core::PhaseVector& context_center,
    const core::PhaseVector& transformation_center,
    const StructuralLearningConfig& config
) {
    return sample.desired_transformation.mean_angular_error(
               transformation_center
           ) +
        (config.split_context_distance_weight *
         sample.context.mean_angular_error(context_center));
}

[[nodiscard]] std::vector<std::size_t> all_indices(
    const std::size_t count
) {
    std::vector<std::size_t> indices(count);
    for (std::size_t index = 0U; index < count; ++index) {
        indices[index] = index;
    }
    return indices;
}

[[nodiscard]] double mean_transformation_error(
    const std::vector<core::CorrectionSummary>& samples,
    const std::vector<std::size_t>& indices,
    const core::PhaseVector& center
) {
    double total = 0.0;
    for (const std::size_t index : indices) {
        total += samples[index].desired_transformation.mean_angular_error(
            center
        );
    }
    return total / static_cast<double>(indices.size());
}

[[nodiscard]] ClusterModel cluster_corrections(
    const std::vector<core::CorrectionSummary>& samples,
    const StructuralLearningConfig& config
) {
    const std::size_t first_seed = 0U;
    std::size_t second_seed = 1U;
    double maximum_distance = -1.0;
    for (std::size_t index = 1U; index < samples.size(); ++index) {
        const double distance =
            samples[first_seed].desired_transformation.mean_angular_error(
                samples[index].desired_transformation
            );
        if (distance > maximum_distance) {
            maximum_distance = distance;
            second_seed = index;
        }
    }

    core::PhaseVector first_context = samples[first_seed].context;
    core::PhaseVector second_context = samples[second_seed].context;
    core::PhaseVector first_transformation =
        samples[first_seed].desired_transformation;
    core::PhaseVector second_transformation =
        samples[second_seed].desired_transformation;
    std::vector<std::size_t> first_indices;
    std::vector<std::size_t> second_indices;

    for (std::size_t iteration = 0U;
         iteration < config.split_kmeans_iterations;
         ++iteration) {
        first_indices.clear();
        second_indices.clear();
        for (std::size_t index = 0U; index < samples.size(); ++index) {
            const double first_distance = assignment_distance(
                samples[index],
                first_context,
                first_transformation,
                config
            );
            const double second_distance = assignment_distance(
                samples[index],
                second_context,
                second_transformation,
                config
            );
            if (first_distance <= second_distance) {
                first_indices.push_back(index);
            } else {
                second_indices.push_back(index);
            }
        }
        if (first_indices.empty() || second_indices.empty()) {
            break;
        }

        first_context = average_vectors(samples, first_indices, true);
        second_context = average_vectors(samples, second_indices, true);
        first_transformation =
            average_vectors(samples, first_indices, false);
        second_transformation =
            average_vectors(samples, second_indices, false);
    }

    double validation_gain = -std::numeric_limits<double>::infinity();
    if (!first_indices.empty() && !second_indices.empty()) {
        const std::vector<std::size_t> every_index =
            all_indices(samples.size());
        const core::PhaseVector parent_center =
            average_vectors(samples, every_index, false);
        const double parent_error = mean_transformation_error(
            samples,
            every_index,
            parent_center
        );
        const double child_error =
            ((mean_transformation_error(
                  samples,
                  first_indices,
                  first_transformation
              ) *
              static_cast<double>(first_indices.size())) +
             (mean_transformation_error(
                  samples,
                  second_indices,
                  second_transformation
              ) *
              static_cast<double>(second_indices.size()))) /
            static_cast<double>(samples.size());
        validation_gain = parent_error - child_error;
    }

    return {
        .first_indices = std::move(first_indices),
        .second_indices = std::move(second_indices),
        .first_context = std::move(first_context),
        .second_context = std::move(second_context),
        .first_transformation = std::move(first_transformation),
        .second_transformation = std::move(second_transformation),
        .validation_gain = validation_gain,
    };
}

[[nodiscard]] std::vector<core::CorrectionSummary> selected_samples(
    const std::vector<core::CorrectionSummary>& samples,
    const std::vector<std::size_t>& indices
) {
    std::vector<core::CorrectionSummary> selected;
    selected.reserve(indices.size());
    for (const std::size_t index : indices) {
        selected.push_back(samples[index]);
    }
    return selected;
}

[[nodiscard]] core::PhaseVector weighted_average(
    const core::PhaseVector& first,
    const core::PhaseVector& second,
    const double first_weight,
    const double second_weight
) {
    const std::vector<core::PhaseVector> vectors{first, second};
    const std::vector<float> weights{
        static_cast<float>(first_weight),
        static_cast<float>(second_weight),
    };
    return core::PhaseVector::weighted_circular_average(vectors, weights);
}

[[nodiscard]] bool histories_are_mergeable(
    const core::ResonantMode& first,
    const core::ResonantMode& second,
    const StructuralLearningConfig& config
) {
    std::vector<core::CorrectionSummary> combined = first.recent_corrections;
    combined.insert(
        combined.end(),
        second.recent_corrections.begin(),
        second.recent_corrections.end()
    );
    if (combined.empty()) {
        return true;
    }
    const std::vector<std::size_t> indices = all_indices(combined.size());
    const core::PhaseVector center =
        average_vectors(combined, indices, false);
    return mean_transformation_error(combined, indices, center) <=
        config.merge_maximum_history_dispersion_radians;
}

[[nodiscard]] double harmful_ratio(const core::ResonantMode& mode) {
    const std::uint64_t total_updates =
        mode.successful_update_count + mode.unsuccessful_update_count;
    if (total_updates == 0ULL) {
        return 0.0;
    }
    return static_cast<double>(mode.unsuccessful_update_count) /
        static_cast<double>(total_updates);
}

}  // namespace

std::string_view to_string(const StructuralEventType event_type) noexcept {
    switch (event_type) {
        case StructuralEventType::created:
            return "created";
        case StructuralEventType::split:
            return "split";
        case StructuralEventType::merged:
            return "merged";
        case StructuralEventType::pruned:
            return "pruned";
    }
    return "unknown";
}

StructuralLearner::StructuralLearner(StructuralLearningConfig config)
    : config_(std::move(config)) {
    validate_config(config_);
}

const StructuralLearningConfig& StructuralLearner::config() const noexcept {
    return config_;
}

const StructuralStatistics& StructuralLearner::statistics() const noexcept {
    return statistics_;
}

std::span<const StructuralEvent> StructuralLearner::events() const noexcept {
    return events_;
}

void StructuralLearner::record_correction(
    core::ResonantMode& mode,
    const core::PhaseVector& context,
    const core::PhaseVector& desired_transformation,
    const double proposal_quality,
    const bool improved_prediction,
    const std::uint64_t step
) const {
    if (!config_.enabled) {
        return;
    }
    if (mode.recent_corrections.size() >=
        config_.correction_history_capacity) {
        mode.recent_corrections.erase(mode.recent_corrections.begin());
    }
    mode.recent_corrections.push_back({
        .context = context,
        .desired_transformation = desired_transformation,
        .proposal_quality = proposal_quality,
        .improved_prediction = improved_prediction,
        .step = step,
    });
}

bool StructuralLearner::consider_creation(
    std::vector<core::ResonantMode>& modes,
    std::uint64_t& next_mode_id,
    const std::size_t maximum_modes,
    const core::PhaseVector& input,
    const core::PhaseVector& desired_transformation,
    const double best_resonance,
    const double prediction_error,
    const std::uint64_t step
) {
    if (!config_.enabled) {
        return false;
    }
    if (!config_.enable_creation) {
        return false;
    }
    const bool low_resonance =
        best_resonance < config_.creation_minimum_resonance;
    const bool high_error =
        prediction_error > config_.creation_prediction_error_threshold;
    if (!low_resonance && !high_error) {
        return false;
    }
    if (modes.size() >= maximum_modes) {
        return false;
    }
    if (next_mode_id == 0ULL) {
        throw std::overflow_error("resonant mode ID space exhausted");
    }

    const std::uint64_t created_id = next_mode_id;
    ++next_mode_id;
    modes.emplace_back(
        created_id,
        input,
        desired_transformation,
        config_.creation_selectivity,
        config_.creation_confidence,
        config_.creation_utility,
        step
    );
    ++statistics_.modes_created;
    events_.push_back({
        .type = StructuralEventType::created,
        .step = step,
        .primary_mode_id = created_id,
        .related_mode_ids = {},
        .reason = low_resonance
            ? "resonance_below_creation_threshold"
            : "prediction_error_above_creation_threshold",
        .metric = low_resonance ? best_resonance : prediction_error,
    });
    return true;
}

void StructuralLearner::maintain(
    std::vector<core::ResonantMode>& modes,
    std::uint64_t& next_mode_id,
    const std::size_t maximum_modes,
    const std::uint64_t step
) {
    if (!config_.enabled) {
        return;
    }
    if (config_.enable_splitting) {
        consider_splits(modes, next_mode_id, maximum_modes, step);
    }
    if (config_.enable_merging) {
        consider_merges(modes, step);
    }
    if (config_.enable_pruning) {
        consider_pruning(modes, step);
    }
}

std::size_t StructuralLearner::estimate_mode_bytes(
    const core::ResonantMode& mode
) noexcept {
    const std::size_t phase_bytes =
        (mode.context_key.size() + mode.transformation.size()) *
        sizeof(core::PhaseVector::Angle);
    std::size_t correction_bytes = 0U;
    for (const core::CorrectionSummary& correction :
         mode.recent_corrections) {
        correction_bytes +=
            (correction.context.size() +
             correction.desired_transformation.size()) *
            sizeof(core::PhaseVector::Angle);
        correction_bytes += sizeof(core::CorrectionSummary);
    }
    return sizeof(core::ResonantMode) + phase_bytes + correction_bytes;
}

void StructuralLearner::consider_splits(
    std::vector<core::ResonantMode>& modes,
    std::uint64_t& next_mode_id,
    const std::size_t maximum_modes,
    const std::uint64_t step
) {
    const std::size_t original_mode_count = modes.size();
    for (std::size_t mode_index = 0U;
         mode_index < original_mode_count;
         ++mode_index) {
        core::ResonantMode& parent = modes[mode_index];
        if (!parent.enabled ||
            parent.recent_corrections.size() <
                config_.split_minimum_samples ||
            modes.size() + 2U > maximum_modes) {
            continue;
        }

        const ClusterModel clusters = cluster_corrections(
            parent.recent_corrections,
            config_
        );
        if (clusters.first_indices.size() <
                config_.split_minimum_cluster_size ||
            clusters.second_indices.size() <
                config_.split_minimum_cluster_size) {
            continue;
        }

        const double transformation_separation =
            clusters.first_transformation.mean_angular_error(
                clusters.second_transformation
            );
        const double context_separation =
            clusters.first_context.mean_angular_error(
                clusters.second_context
            );
        if (transformation_separation <
                config_.split_minimum_transformation_separation_radians ||
            context_separation <
                config_.split_minimum_context_separation_radians ||
            clusters.validation_gain <
                config_.split_minimum_validation_gain_radians) {
            continue;
        }
        if (next_mode_id >
            std::numeric_limits<std::uint64_t>::max() - 1ULL) {
            throw std::overflow_error("resonant mode ID space exhausted");
        }

        const std::uint64_t first_child_id = next_mode_id;
        const std::uint64_t second_child_id = next_mode_id + 1ULL;
        next_mode_id += 2ULL;
        const std::uint64_t parent_id = parent.id;
        const float child_confidence = std::clamp(
            parent.confidence *
                config_.split_child_confidence_scale,
            config_.creation_confidence,
            1.0F
        );

        core::ResonantMode first_child(
            first_child_id,
            clusters.first_context,
            clusters.first_transformation,
            parent.selectivity,
            child_confidence,
            parent.utility,
            step
        );
        core::ResonantMode second_child(
            second_child_id,
            clusters.second_context,
            clusters.second_transformation,
            parent.selectivity,
            child_confidence,
            parent.utility,
            step
        );
        first_child.recent_corrections = selected_samples(
            parent.recent_corrections,
            clusters.first_indices
        );
        second_child.recent_corrections = selected_samples(
            parent.recent_corrections,
            clusters.second_indices
        );

        parent.enabled = false;
        parent.last_used_step = step;
        modes.push_back(std::move(first_child));
        modes.push_back(std::move(second_child));
        ++statistics_.modes_split;
        statistics_.modes_created += 2ULL;
        events_.push_back({
            .type = StructuralEventType::split,
            .step = step,
            .primary_mode_id = parent_id,
            .related_mode_ids = {first_child_id, second_child_id},
            .reason = "validated_two_cluster_correction_disagreement",
            .metric = clusters.validation_gain,
        });
    }
}

void StructuralLearner::consider_merges(
    std::vector<core::ResonantMode>& modes,
    const std::uint64_t step
) {
    for (std::size_t first_index = 0U;
         first_index < modes.size();
         ++first_index) {
        core::ResonantMode& first = modes[first_index];
        if (!first.enabled) {
            continue;
        }
        for (std::size_t second_index = first_index + 1U;
             second_index < modes.size();
             ++second_index) {
            core::ResonantMode& second = modes[second_index];
            if (!second.enabled ||
                first.context_key.mean_angular_error(second.context_key) >
                    config_.merge_maximum_key_error_radians ||
                first.transformation.mean_angular_error(
                    second.transformation
                ) >
                    config_.merge_maximum_transformation_error_radians ||
                !histories_are_mergeable(first, second, config_)) {
                continue;
            }

            core::ResonantMode* retained = &first;
            core::ResonantMode* absorbed = &second;
            if (second.id < first.id) {
                retained = &second;
                absorbed = &first;
            }
            const double retained_weight =
                static_cast<double>(retained->activation_count + 1ULL);
            const double absorbed_weight =
                static_cast<double>(absorbed->activation_count + 1ULL);
            retained->context_key = weighted_average(
                retained->context_key,
                absorbed->context_key,
                retained_weight,
                absorbed_weight
            );
            retained->transformation = weighted_average(
                retained->transformation,
                absorbed->transformation,
                retained_weight,
                absorbed_weight
            );
            retained->confidence = static_cast<float>(
                ((retained_weight *
                  static_cast<double>(retained->confidence)) +
                 (absorbed_weight *
                  static_cast<double>(absorbed->confidence))) /
                (retained_weight + absorbed_weight)
            );
            retained->utility = static_cast<float>(
                ((retained_weight *
                  static_cast<double>(retained->utility)) +
                 (absorbed_weight *
                  static_cast<double>(absorbed->utility))) /
                (retained_weight + absorbed_weight)
            );
            retained->activation_count += absorbed->activation_count;
            retained->successful_update_count +=
                absorbed->successful_update_count;
            retained->unsuccessful_update_count +=
                absorbed->unsuccessful_update_count;
            retained->last_used_step = std::max(
                retained->last_used_step,
                absorbed->last_used_step
            );
            retained->recent_corrections.insert(
                retained->recent_corrections.end(),
                absorbed->recent_corrections.begin(),
                absorbed->recent_corrections.end()
            );
            if (retained->recent_corrections.size() >
                config_.correction_history_capacity) {
                const std::size_t excess =
                    retained->recent_corrections.size() -
                    config_.correction_history_capacity;
                retained->recent_corrections.erase(
                    retained->recent_corrections.begin(),
                    retained->recent_corrections.begin() +
                        static_cast<std::ptrdiff_t>(excess)
                );
            }
            absorbed->enabled = false;
            absorbed->last_used_step = step;
            ++statistics_.modes_merged;
            events_.push_back({
                .type = StructuralEventType::merged,
                .step = step,
                .primary_mode_id = retained->id,
                .related_mode_ids = {absorbed->id},
                .reason = "key_transformation_and_history_compatible",
                .metric = retained->transformation.mean_angular_error(
                    absorbed->transformation
                ),
            });
            break;
        }
    }
}

void StructuralLearner::consider_pruning(
    std::vector<core::ResonantMode>& modes,
    const std::uint64_t step
) {
    std::size_t enabled_count = static_cast<std::size_t>(std::count_if(
        modes.begin(),
        modes.end(),
        [](const core::ResonantMode& mode) {
            return mode.enabled;
        }
    ));

    std::size_t total_bytes = 0U;
    for (const core::ResonantMode& mode : modes) {
        total_bytes += estimate_mode_bytes(mode);
    }
    const bool memory_exceeded =
        config_.memory_budget_bytes != 0U &&
        total_bytes > config_.memory_budget_bytes;

    for (core::ResonantMode& mode : modes) {
        if (!mode.enabled) {
            continue;
        }
        if (enabled_count <= config_.minimum_retained_modes) {
            break;
        }

        const std::uint64_t age = step >= mode.creation_step
            ? step - mode.creation_step
            : 0ULL;
        const std::uint64_t inactivity = step >= mode.last_used_step
            ? step - mode.last_used_step
            : 0ULL;
        const bool old_enough =
            age >= config_.pruning_minimum_age_steps;
        const bool low_use =
            mode.activation_count <
            config_.pruning_minimum_activation_count;
        const bool low_utility =
            static_cast<double>(mode.utility) <=
            config_.pruning_maximum_utility;
        const bool inactive =
            inactivity >= config_.pruning_maximum_inactive_steps;
        const bool harmful =
            harmful_ratio(mode) >=
            config_.pruning_harmful_update_ratio;
        if (!memory_exceeded &&
            !(old_enough &&
              ((low_use && low_utility) || inactive || harmful))) {
            continue;
        }

        mode.enabled = false;
        mode.last_used_step = step;
        --enabled_count;
        ++statistics_.modes_pruned;
        events_.push_back({
            .type = StructuralEventType::pruned,
            .step = step,
            .primary_mode_id = mode.id,
            .related_mode_ids = {},
            .reason = memory_exceeded
                ? "memory_budget_exceeded"
                : (harmful
                       ? "repeated_harmful_contribution"
                       : (inactive
                              ? "inactive_mode"
                              : "old_low_use_low_utility_mode")),
            .metric = harmful ? harmful_ratio(mode)
                              : static_cast<double>(inactivity),
        });
    }

    modes.erase(
        std::remove_if(
            modes.begin(),
            modes.end(),
            [this, step](const core::ResonantMode& mode) {
                if (mode.enabled) {
                    return false;
                }
                const std::uint64_t disabled_age =
                    step >= mode.last_used_step
                    ? step - mode.last_used_step
                    : 0ULL;
                return disabled_age >=
                    config_.pruning_disabled_grace_steps;
            }
        ),
        modes.end()
    );
}

}  // namespace rlf::learning
