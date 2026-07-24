#include "rlf/core/operator_fabric.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rlf::core {
namespace {

constexpr OperatorFamily candidate_families[]{
    OperatorFamily::phase_shift,
    OperatorFamily::coordinate_permutation,
    OperatorFamily::conjugation,
    OperatorFamily::permutation_then_phase_shift,
    OperatorFamily::conjugation_then_phase_shift,
    OperatorFamily::explicit_sequence,
};

[[nodiscard]] double context_similarity(
    const std::span<const float> left,
    const std::span<const float> right
) {
    if (left.empty() || left.size() != right.size()) {
        throw std::invalid_argument(
            "operator context similarity requires equal non-empty inputs"
        );
    }
    double real = 0.0;
    double imaginary = 0.0;
    for (std::size_t index = 0U; index < left.size(); ++index) {
        const double difference =
            static_cast<double>(left[index]) -
            static_cast<double>(right[index]);
        real += std::cos(difference);
        imaginary += std::sin(difference);
    }
    const double dimension = static_cast<double>(left.size());
    return std::clamp(
        ((real * real) + (imaginary * imaginary)) /
            (dimension * dimension),
        0.0,
        1.0
    );
}

[[nodiscard]] double family_complexity(
    const OperatorFamily family
) noexcept {
    switch (family) {
    case OperatorFamily::phase_shift:
    case OperatorFamily::coordinate_permutation:
    case OperatorFamily::conjugation:
        return 1.0;
    case OperatorFamily::permutation_then_phase_shift:
    case OperatorFamily::conjugation_then_phase_shift:
        return 2.0;
    case OperatorFamily::explicit_sequence:
        return 3.0;
    }
    return 3.0;
}

[[nodiscard]] PhaseVector interpolate_context(
    const PhaseVector& current,
    const PhaseVector& input,
    const std::size_t context_dimensions,
    const double amount
) {
    std::vector<float> angles(
        current.angles().begin(),
        current.angles().end()
    );
    for (std::size_t index = 0U;
         index < context_dimensions;
         ++index) {
        const std::vector<PhaseVector> values{
            PhaseVector(std::vector<float>{current[index]}),
            PhaseVector(std::vector<float>{input[index]}),
        };
        const std::vector<float> weights{
            static_cast<float>(1.0 - amount),
            static_cast<float>(amount),
        };
        angles[index] =
            PhaseVector::weighted_circular_average(values, weights)[0U];
    }
    return PhaseVector(std::move(angles));
}

}  // namespace

double OperatorMode::resonance(
    const std::span<const float> context
) const {
    return context_similarity(
        context_key.angles().first(context.size()),
        context
    );
}

PhaseVector OperatorMode::propose(const PhaseVector& input) const {
    return transformation.apply(input);
}

OperatorFabric::OperatorFabric(OperatorFabricConfig config)
    : config_(std::move(config)) {
    if (config_.dimension == 0U ||
        config_.context_dimensions == 0U ||
        config_.context_dimensions >= config_.dimension ||
        config_.history_capacity == 0U ||
        config_.candidate_count == 0U ||
        !std::isfinite(config_.creation_resonance_threshold) ||
        config_.creation_resonance_threshold < 0.0 ||
        config_.creation_resonance_threshold > 1.0 ||
        !std::isfinite(config_.context_learning_rate) ||
        config_.context_learning_rate < 0.0 ||
        config_.context_learning_rate > 1.0 ||
        !std::isfinite(config_.complexity_penalty) ||
        config_.complexity_penalty < 0.0) {
        throw std::invalid_argument(
            "invalid operator fabric configuration"
        );
    }
}

const OperatorFabricConfig& OperatorFabric::config() const noexcept {
    return config_;
}

std::span<const OperatorMode> OperatorFabric::modes() const noexcept {
    return modes_;
}

std::vector<std::size_t> OperatorFabric::retrieve_bank(
    const PhaseVector& input
) const {
    if (input.size() != config_.dimension) {
        throw std::invalid_argument(
            "operator fabric input dimension mismatch"
        );
    }
    struct Candidate final {
        std::size_t index;
        double resonance;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(modes_.size());
    const std::span<const float> context =
        input.angles().first(config_.context_dimensions);
    for (std::size_t index = 0U; index < modes_.size(); ++index) {
        candidates.push_back({
            .index = index,
            .resonance = modes_[index].resonance(context),
        });
    }
    std::sort(
        candidates.begin(),
        candidates.end(),
        [this](const Candidate& left, const Candidate& right) {
            const OperatorMode& left_mode = modes_[left.index];
            const OperatorMode& right_mode = modes_[right.index];
            if (left.resonance != right.resonance) {
                return left.resonance > right.resonance;
            }
            return left_mode.id < right_mode.id;
        }
    );
    if (candidates.empty()) {
        return {};
    }
    const double strongest = candidates.front().resonance;
    std::vector<std::size_t> bank;
    for (const Candidate& candidate : candidates) {
        if (bank.size() >= config_.candidate_count ||
            candidate.resonance + 1.0e-12 < strongest) {
            break;
        }
        bank.push_back(candidate.index);
    }
    return bank;
}

void OperatorFabric::create_bank(const PhaseVector& input) {
    for (const OperatorFamily family : candidate_families) {
        modes_.push_back({
            .id = next_mode_id_++,
            .context_key = input,
            .family = family,
            .transformation =
                TransformationOperator::identity(config_.dimension),
            .confidence = 0.0,
            .training_error = std::numbers::pi_v<double>,
            .activation_count = 0ULL,
            .update_count = 0ULL,
            .recent_examples = {},
        });
    }
}

void OperatorFabric::learn(
    const PhaseVector& input,
    const PhaseVector& target
) {
    if (input.size() != config_.dimension ||
        target.size() != config_.dimension) {
        throw std::invalid_argument(
            "operator fabric learning dimension mismatch"
        );
    }
    std::vector<std::size_t> bank = retrieve_bank(input);
    if (bank.empty() ||
        modes_[bank.front()].resonance(
            input.angles().first(config_.context_dimensions)
        ) < config_.creation_resonance_threshold) {
        create_bank(input);
        bank = retrieve_bank(input);
    }
    for (const std::size_t index : bank) {
        OperatorMode& mode = modes_[index];
        if (mode.recent_examples.size() >= config_.history_capacity) {
            mode.recent_examples.erase(mode.recent_examples.begin());
        }
        mode.recent_examples.push_back({
            .input = input,
            .target = target,
        });
        mode.transformation = fit_operator(
            mode.family,
            mode.recent_examples,
            config_.context_dimensions
        );
        mode.training_error = operator_mean_error(
            mode.transformation,
            mode.recent_examples
        );
        mode.confidence = std::clamp(
            1.0 -
                (mode.training_error / std::numbers::pi_v<double>) -
                (config_.complexity_penalty *
                 family_complexity(mode.family)),
            0.0,
            1.0
        );
        mode.context_key = interpolate_context(
            mode.context_key,
            input,
            config_.context_dimensions,
            config_.context_learning_rate
        );
        ++mode.update_count;
    }
}

OperatorPrediction OperatorFabric::predict(
    const PhaseVector& input
) const {
    const std::vector<std::size_t> candidates = retrieve_bank(input);
    if (candidates.empty()) {
        throw std::runtime_error(
            "operator fabric has no learned modes"
        );
    }
    const std::span<const float> context =
        input.angles().first(config_.context_dimensions);
    std::size_t best_index = candidates.front();
    double best_resonance = modes_[best_index].resonance(context);
    double best_score =
        best_resonance * modes_[best_index].confidence;
    for (const std::size_t index : candidates) {
        const double resonance = modes_[index].resonance(context);
        const double score = resonance * modes_[index].confidence;
        if (score > best_score + 1.0e-12 ||
            (std::abs(score - best_score) <= 1.0e-12 &&
             modes_[index].id < modes_[best_index].id)) {
            best_index = index;
            best_resonance = resonance;
            best_score = score;
        }
    }
    const OperatorMode& best = modes_[best_index];
    return {
        .state = best.propose(input),
        .mode_id = best.id,
        .family = best.family,
        .resonance = best_resonance,
        .confidence = best.confidence,
    };
}

}  // namespace rlf::core
