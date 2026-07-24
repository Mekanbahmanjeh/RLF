#include "rlf/core/resonant_fabric.hpp"

#include "rlf/retrieval/mode_retriever.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rlf::core {
namespace {

void validate_settling_config(const SettlingConfig& config) {
    if (config.candidate_count == 0U ||
        config.active_count == 0U ||
        config.maximum_cycles == 0U ||
        config.minimum_cycles == 0U ||
        config.minimum_cycles > config.maximum_cycles) {
        throw std::invalid_argument(
            "settling counts and cycle bounds must be positive and consistent"
        );
    }
    if (config.active_count > config.candidate_count) {
        throw std::invalid_argument(
            "active mode count cannot exceed candidate count"
        );
    }
    const auto valid_non_negative = [](const double value) {
        return std::isfinite(value) && value >= 0.0;
    };
    if (!valid_non_negative(config.minimum_resonance) ||
        config.minimum_resonance > 1.0 ||
        !valid_non_negative(config.convergence_tolerance_radians) ||
        !valid_non_negative(config.input_weight) ||
        !valid_non_negative(config.previous_state_weight) ||
        !valid_non_negative(config.proposal_weight_scale) ||
        !valid_non_negative(config.utility_weight)) {
        throw std::invalid_argument(
            "settling coefficients must be finite and non-negative"
        );
    }
    if (config.confidence_threshold.has_value() &&
        (!std::isfinite(*config.confidence_threshold) ||
         *config.confidence_threshold < 0.0 ||
         *config.confidence_threshold > 1.0)) {
        throw std::invalid_argument(
            "settling confidence threshold must be in [0, 1]"
        );
    }
    if (config.input_weight == 0.0 &&
        config.previous_state_weight == 0.0 &&
        config.proposal_weight_scale == 0.0) {
        throw std::invalid_argument(
            "settling requires at least one non-zero contributor coefficient"
        );
    }
}

[[nodiscard]] PhaseVector circular_interpolate(
    const PhaseVector& current,
    const PhaseVector& target,
    const double amount
) {
    const double clamped_amount = std::clamp(amount, 0.0, 1.0);
    const std::vector<PhaseVector> vectors{current, target};
    const std::vector<float> weights{
        static_cast<float>(1.0 - clamped_amount),
        static_cast<float>(clamped_amount),
    };
    if (clamped_amount == 0.0) {
        return current;
    }
    if (clamped_amount == 1.0) {
        return target;
    }
    return PhaseVector::weighted_circular_average(vectors, weights);
}

[[nodiscard]] double proposal_weight(
    const ResonantMode& mode,
    const double resonance,
    const SettlingConfig& config
) {
    const double utility_factor = std::clamp(
        1.0 + (config.utility_weight * static_cast<double>(mode.utility)),
        0.05,
        2.0
    );
    return resonance *
        std::max(static_cast<double>(mode.confidence), 1.0e-6) *
        utility_factor;
}

[[nodiscard]] double calculate_coherence(
    const PhaseVector& next_state,
    const PhaseVector& input,
    const PhaseVector& previous_state,
    const std::span<const PhaseVector> proposals,
    const std::span<const double> proposal_weights,
    const SettlingConfig& config
) {
    double weighted_similarity = 0.0;
    double total_weight = 0.0;
    if (config.input_weight > 0.0) {
        weighted_similarity +=
            config.input_weight * next_state.similarity(input);
        total_weight += config.input_weight;
    }
    if (config.previous_state_weight > 0.0) {
        weighted_similarity +=
            config.previous_state_weight *
            next_state.similarity(previous_state);
        total_weight += config.previous_state_weight;
    }
    for (std::size_t proposal_index = 0U;
         proposal_index < proposals.size();
         ++proposal_index) {
        const double weight =
            proposal_weights[proposal_index] * config.proposal_weight_scale;
        weighted_similarity +=
            weight * next_state.similarity(proposals[proposal_index]);
        total_weight += weight;
    }
    return total_weight > 0.0
        ? std::clamp(weighted_similarity / total_weight, 0.0, 1.0)
        : 0.0;
}

}  // namespace

ResonantFabric::ResonantFabric(FabricConfig config)
    : ResonantFabric(
          std::move(config),
          std::make_unique<retrieval::ExactModeRetriever>(),
          std::make_unique<StableCircularSettlingPolicy>(),
          std::make_unique<learning::NormalizedResponsibilityUpdate>()
      ) {}

ResonantFabric::ResonantFabric(
    FabricConfig config,
    std::unique_ptr<retrieval::ModeRetriever> retriever,
    std::unique_ptr<SettlingPolicy> settling_policy,
    std::unique_ptr<learning::LocalUpdateStrategy> update_strategy
)
    : config_(std::move(config)),
      retriever_(std::move(retriever)),
      settling_policy_(std::move(settling_policy)),
      update_strategy_(std::move(update_strategy)),
      structural_learner_(config_.structural_learning) {
    if (config_.dimension == 0U || config_.maximum_modes == 0U) {
        throw std::invalid_argument(
            "fabric dimension and maximum mode count must be positive"
        );
    }
    validate_settling_config(config_.settling);
    if (!retriever_ || !settling_policy_ || !update_strategy_) {
        throw std::invalid_argument(
            "fabric retrieval, settling, and learning strategies are required"
        );
    }
}

ResonantFabric::~ResonantFabric() = default;
ResonantFabric::ResonantFabric(ResonantFabric&&) noexcept = default;
ResonantFabric& ResonantFabric::operator=(ResonantFabric&&) noexcept = default;

const FabricConfig& ResonantFabric::config() const noexcept {
    return config_;
}

std::span<const ResonantMode> ResonantFabric::modes() const noexcept {
    return modes_;
}

std::uint64_t ResonantFabric::training_step() const noexcept {
    return training_step_;
}

std::string_view ResonantFabric::update_strategy_name() const noexcept {
    return update_strategy_->name();
}

const learning::StructuralStatistics&
ResonantFabric::structural_statistics() const noexcept {
    return structural_learner_.statistics();
}

std::span<const learning::StructuralEvent>
ResonantFabric::structural_events() const noexcept {
    return structural_learner_.events();
}

void ResonantFabric::add_mode(ResonantMode mode) {
    if (modes_.size() >= config_.maximum_modes) {
        throw std::runtime_error("fabric maximum mode count reached");
    }
    if (mode.context_key.size() != config_.dimension ||
        mode.transformation.size() != config_.dimension) {
        throw std::invalid_argument(
            "mode dimensions must match the fabric dimension"
        );
    }
    const auto duplicate = std::find_if(
        modes_.begin(),
        modes_.end(),
        [&mode](const ResonantMode& existing_mode) {
            return existing_mode.id == mode.id;
        }
    );
    if (duplicate != modes_.end()) {
        throw std::invalid_argument("resonant mode IDs must be unique");
    }
    if (mode.id >= next_mode_id_) {
        if (mode.id == std::numeric_limits<std::uint64_t>::max()) {
            next_mode_id_ = 0ULL;
        } else {
            next_mode_id_ = mode.id + 1ULL;
        }
    }
    modes_.push_back(std::move(mode));
}

void ResonantFabric::set_update_strategy(
    std::unique_ptr<learning::LocalUpdateStrategy> update_strategy
) {
    if (!update_strategy) {
        throw std::invalid_argument("local update strategy must not be null");
    }
    update_strategy_ = std::move(update_strategy);
}

void ResonantFabric::maintain_structure() {
    structural_learner_.maintain(
        modes_,
        next_mode_id_,
        config_.maximum_modes,
        training_step_
    );
}

SettleResult ResonantFabric::settle(
    const PhaseVector& input,
    const bool capture_trace,
    const HaltCondition& halt_condition
) {
    if (input.size() != config_.dimension) {
        throw std::invalid_argument(
            "settling input dimension must match the fabric"
        );
    }

    ++execution_step_;
    PhaseVector previous_state = input;
    std::vector<ModeParticipation> final_participants;
    std::optional<ExecutionTrace> execution_trace;
    if (capture_trace) {
        execution_trace.emplace();
        execution_trace->cycles.reserve(config_.settling.maximum_cycles);
    }

    double final_coherence = 0.0;
    double final_confidence = 0.0;
    StopReason stopping_reason = StopReason::cycle_limit;
    std::size_t completed_cycles = 0U;

    for (std::size_t cycle_index = 0U;
         cycle_index < config_.settling.maximum_cycles;
         ++cycle_index) {
        const std::vector<retrieval::RetrievedMode> retrieved =
            retriever_->retrieve(
                previous_state,
                modes_,
                config_.settling.candidate_count
            );

        SettlingCycleTrace cycle_trace;
        cycle_trace.cycle_number = cycle_index;
        cycle_trace.retrieved_mode_ids.reserve(retrieved.size());
        cycle_trace.resonance_scores.reserve(retrieved.size());
        for (const retrieval::RetrievedMode& candidate : retrieved) {
            cycle_trace.retrieved_mode_ids.push_back(candidate.mode_id);
            cycle_trace.resonance_scores.push_back(candidate.resonance);
        }

        std::vector<PhaseVector> proposals;
        std::vector<double> weights;
        std::vector<ModeParticipation> participants;
        proposals.reserve(config_.settling.active_count);
        weights.reserve(config_.settling.active_count);
        participants.reserve(config_.settling.active_count);

        for (const retrieval::RetrievedMode& candidate : retrieved) {
            if (participants.size() >= config_.settling.active_count) {
                break;
            }
            if (candidate.resonance < config_.settling.minimum_resonance) {
                continue;
            }

            ResonantMode& mode = modes_.at(candidate.mode_index);
            const double weight = proposal_weight(
                mode,
                candidate.resonance,
                config_.settling
            );
            if (weight <= 0.0) {
                continue;
            }

            PhaseVector proposal = mode.propose(previous_state);
            proposals.push_back(proposal);
            weights.push_back(weight);
            participants.push_back({
                .mode_index = candidate.mode_index,
                .mode_id = candidate.mode_id,
                .resonance = candidate.resonance,
                .normalized_contribution = 0.0,
                .proposal_state = std::move(proposal),
            });
            cycle_trace.active_mode_ids.push_back(candidate.mode_id);
            cycle_trace.proposal_weights.push_back(weight);
            ++mode.activation_count;
            mode.last_used_step = execution_step_;
        }

        if (participants.empty()) {
            stopping_reason = StopReason::no_active_modes;
            completed_cycles = cycle_index + 1U;
            cycle_trace.stopping_reason = stopping_reason;
            if (capture_trace) {
                execution_trace->cycles.push_back(std::move(cycle_trace));
            }
            break;
        }

        double total_proposal_weight = 0.0;
        for (const double weight : weights) {
            total_proposal_weight += weight;
        }
        for (std::size_t participant_index = 0U;
             participant_index < participants.size();
             ++participant_index) {
            participants[participant_index].normalized_contribution =
                weights[participant_index] / total_proposal_weight;
        }

        const PhaseVector next_state = settling_policy_->next_state(
            input,
            previous_state,
            proposals,
            weights,
            config_.settling
        );
        const double state_change =
            next_state.mean_angular_error(previous_state);
        final_coherence = calculate_coherence(
            next_state,
            input,
            previous_state,
            proposals,
            weights,
            config_.settling
        );
        final_confidence = 0.0;
        for (const ModeParticipation& participant : participants) {
            const ResonantMode& mode = modes_.at(participant.mode_index);
            final_confidence +=
                participant.normalized_contribution *
                participant.resonance *
                static_cast<double>(mode.confidence);
        }
        final_confidence = std::clamp(final_confidence, 0.0, 1.0);

        cycle_trace.state_change_radians = state_change;
        cycle_trace.coherence = final_coherence;
        cycle_trace.confidence = final_confidence;
        completed_cycles = cycle_index + 1U;
        final_participants = std::move(participants);

        if (halt_condition && halt_condition(cycle_trace)) {
            stopping_reason = StopReason::explicit_halt;
        } else if (
            config_.settling.confidence_threshold.has_value() &&
            final_confidence >= *config_.settling.confidence_threshold) {
            stopping_reason = StopReason::confidence_threshold;
        } else if (
            completed_cycles >= config_.settling.minimum_cycles &&
            state_change <=
                config_.settling.convergence_tolerance_radians) {
            stopping_reason = StopReason::converged;
        } else if (completed_cycles >= config_.settling.maximum_cycles) {
            stopping_reason = StopReason::cycle_limit;
        } else {
            stopping_reason = StopReason::none;
        }

        cycle_trace.stopping_reason = stopping_reason;
        if (capture_trace) {
            execution_trace->cycles.push_back(std::move(cycle_trace));
        }
        previous_state = next_state;

        if (stopping_reason != StopReason::none) {
            break;
        }
    }

    if (execution_trace.has_value()) {
        execution_trace->stopping_reason = stopping_reason;
    }
    return SettleResult{
        .state = std::move(previous_state),
        .stopping_reason = stopping_reason,
        .cycles = completed_cycles,
        .coherence = final_coherence,
        .confidence = final_confidence,
        .participants = std::move(final_participants),
        .trace = std::move(execution_trace),
    };
}

LearningResult ResonantFabric::learn(
    const PhaseVector& input,
    const PhaseVector& target,
    const learning::LocalLearningConfig& learning_config,
    const bool capture_trace
) {
    if (target.size() != config_.dimension) {
        throw std::invalid_argument(
            "learning target dimension must match the fabric"
        );
    }

    SettleResult prediction = settle(input, capture_trace);
    const PhaseVector desired_transformation =
        PhaseVector::phase_difference(input, target);
    const double prediction_similarity =
        prediction.state.similarity(target);
    const double baseline_quality = input.similarity(target);

    std::vector<learning::ModeEvidence> evidence;
    evidence.reserve(prediction.participants.size());
    for (const ModeParticipation& participant : prediction.participants) {
        const ResonantMode& mode = modes_.at(participant.mode_index);
        const double proposal_quality =
            participant.proposal_state.similarity(target);
        evidence.push_back({
            .mode_index = participant.mode_index,
            .mode_id = participant.mode_id,
            .resonance = participant.resonance,
            .normalized_contribution =
                participant.normalized_contribution,
            .baseline_quality = baseline_quality,
            .proposal_quality = proposal_quality,
            .prediction_quality = prediction_similarity,
            .utility = static_cast<double>(mode.utility),
            .improved_prediction =
                proposal_quality > baseline_quality + 1.0e-12,
        });
    }

    std::vector<double> responsibilities =
        update_strategy_->responsibilities(evidence, learning_config);
    if (responsibilities.size() != evidence.size()) {
        throw std::runtime_error(
            "local update strategy returned an invalid responsibility count"
        );
    }

    std::vector<std::uint64_t> updated_mode_ids;
    updated_mode_ids.reserve(evidence.size());
    for (std::size_t evidence_index = 0U;
         evidence_index < evidence.size();
         ++evidence_index) {
        const learning::ModeEvidence& item = evidence[evidence_index];
        const double responsibility = responsibilities[evidence_index];
        if (!std::isfinite(responsibility) || responsibility < 0.0) {
            throw std::runtime_error(
                "local update responsibility must be finite and non-negative"
            );
        }
        if (responsibility == 0.0) {
            continue;
        }

        ResonantMode& mode = modes_.at(item.mode_index);
        const double transformation_amount =
            learning_config.transformation_learning_rate * responsibility;
        const double context_amount =
            learning_config.context_learning_rate * responsibility;
        mode.transformation = circular_interpolate(
            mode.transformation,
            desired_transformation,
            transformation_amount
        );
        mode.context_key = circular_interpolate(
            mode.context_key,
            input,
            context_amount
        );

        const double confidence_target = item.proposal_quality;
        mode.confidence = static_cast<float>(std::clamp(
            static_cast<double>(mode.confidence) +
                (learning_config.confidence_learning_rate *
                 responsibility *
                 (confidence_target -
                  static_cast<double>(mode.confidence))),
            0.0,
            1.0
        ));
        const double signed_quality_delta =
            item.proposal_quality - item.baseline_quality;
        mode.utility = static_cast<float>(
            static_cast<double>(mode.utility) +
            (learning_config.utility_learning_rate *
             responsibility *
             (signed_quality_delta -
              static_cast<double>(mode.utility)))
        );

        if (item.improved_prediction) {
            ++mode.successful_update_count;
        } else {
            ++mode.unsuccessful_update_count;
        }
        if (config_.structural_learning.enable_splitting) {
            structural_learner_.record_correction(
                mode,
                input,
                desired_transformation,
                item.proposal_quality,
                item.improved_prediction,
                training_step_ + 1ULL
            );
        }
        updated_mode_ids.push_back(mode.id);
    }

    double best_resonance = 0.0;
    for (const ResonantMode& mode : modes_) {
        if (mode.enabled) {
            best_resonance = std::max(
                best_resonance,
                mode.resonance(input)
            );
        }
    }
    const std::size_t previous_event_count =
        structural_learner_.events().size();
    ++training_step_;
    static_cast<void>(structural_learner_.consider_creation(
        modes_,
        next_mode_id_,
        config_.maximum_modes,
        input,
        desired_transformation,
        best_resonance,
        1.0 - prediction_similarity,
        training_step_
    ));
    maintain_structure();

    std::vector<learning::StructuralEvent> new_structural_events;
    const std::span<const learning::StructuralEvent> all_events =
        structural_learner_.events();
    new_structural_events.reserve(
        all_events.size() - previous_event_count
    );
    for (std::size_t event_index = previous_event_count;
         event_index < all_events.size();
         ++event_index) {
        new_structural_events.push_back(all_events[event_index]);
    }
    return LearningResult{
        .prediction = std::move(prediction),
        .desired_transformation = desired_transformation,
        .prediction_similarity = prediction_similarity,
        .prediction_error = 1.0 - prediction_similarity,
        .evidence = std::move(evidence),
        .responsibilities = std::move(responsibilities),
        .updated_mode_ids = std::move(updated_mode_ids),
        .structural_events = std::move(new_structural_events),
    };
}

}  // namespace rlf::core
