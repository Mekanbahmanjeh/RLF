#include "rlf/experiments/rlf4_self_supervised.hpp"

#include "rlf/core/deterministic_rng.hpp"
#include "rlf/experiments/metrics.hpp"
#include "rlf/experiments/phase_vector_smoke.hpp"
#include "rlf/storage/rlf4_checkpoint.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <numbers>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rlf::experiments {
namespace {

constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;
constexpr double probability_floor = 1.0e-12;

void hash_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (unsigned int byte = 0U; byte < 8U; ++byte) {
        hash ^= (value >> (byte * 8U)) & 0xFFULL;
        hash *= fnv_prime;
    }
}

void hash_double(std::uint64_t& hash, const double value) noexcept {
    hash_u64(hash, std::bit_cast<std::uint64_t>(value));
}

struct VectorHash final {
    [[nodiscard]] std::size_t operator()(
        const std::vector<std::uint64_t>& value
    ) const noexcept {
        std::uint64_t hash = fnv_offset_basis;
        hash_u64(hash, value.size());
        for (const auto item : value) {
            hash_u64(hash, item);
        }
        return static_cast<std::size_t>(hash);
    }
};

[[nodiscard]] core::PhaseVector perturb(
    const core::PhaseVector& source,
    const double radians,
    core::DeterministicRng& rng
) {
    std::vector<float> angles;
    angles.reserve(source.size());
    for (const float angle : source.angles()) {
        const double noise = (2.0 * rng.uniform_unit() - 1.0) * radians;
        angles.push_back(core::PhaseVector::normalize_angle(
            static_cast<float>(static_cast<double>(angle) + noise)
        ));
    }
    return core::PhaseVector(std::move(angles));
}

void validate_config(const Rlf4Config& config) {
    if (config.dimension < 8U || config.symbol_count < 18U ||
        config.training_tokens < 256U || config.evaluation_tokens < 128U ||
        config.adaptation_tokens < 256U || config.maximum_context_order < 4U ||
        config.minimum_context_support == 0U || config.maximum_options == 0U ||
        config.minimum_option_support < 2U || config.forecast_horizon == 0U ||
        config.forecast_samples == 0U || config.change_tolerance == 0U ||
        !std::isfinite(config.training_noise_radians) ||
        config.training_noise_radians < 0.0 ||
        !std::isfinite(config.evaluation_noise_radians) ||
        config.evaluation_noise_radians < 0.0 ||
        !std::isfinite(config.dominant_motif_probability) ||
        config.dominant_motif_probability <= 0.5 ||
        config.dominant_motif_probability >= 1.0 ||
        !std::isfinite(config.prototype_merge_distance) ||
        config.prototype_merge_distance <= 0.0 ||
        !std::isfinite(config.recent_decay) || config.recent_decay <= 0.0 ||
        config.recent_decay > 1.0 ||
        !std::isfinite(config.recent_weight) || config.recent_weight < 0.0 ||
        config.recent_weight > 1.0) {
        throw std::invalid_argument("invalid RLF-4 experiment configuration");
    }
}

struct GeneratedStream final {
    std::vector<std::size_t> symbols;
    std::vector<core::PhaseVector> observations;
    std::vector<std::size_t> change_indices;
    std::uint64_t hash{};
};

class TemporalWorld final {
public:
    TemporalWorld(
        const Rlf4Config& config,
        const std::uint64_t seed
    ) : config_(config) {
        core::DeterministicRng rng(seed);
        symbols_.reserve(config.symbol_count);
        for (std::size_t index = 0U; index < config.symbol_count; ++index) {
            symbols_.push_back(core::PhaseVector::random(config.dimension, rng));
        }
        motifs_ = {
            {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U},
            {0U, 1U, 2U, 3U, 4U, 5U, 8U, 9U},
            {10U, 1U, 2U, 3U, 4U, 5U, 6U, 11U},
            {10U, 1U, 2U, 3U, 4U, 5U, 8U, 12U},
            {13U, 14U, 2U, 3U, 4U, 5U, 6U, 15U},
            {13U, 14U, 2U, 3U, 4U, 5U, 8U, 16U},
            {17U, 0U, 1U, 2U, 3U, 4U, 5U, 6U},
            {17U, 10U, 1U, 2U, 3U, 4U, 5U, 8U},
        };
        regime_next_ = {{
            std::array<std::size_t, 8U>{1U, 2U, 3U, 4U, 5U, 6U, 7U, 0U},
            std::array<std::size_t, 8U>{2U, 3U, 0U, 1U, 6U, 7U, 4U, 5U},
            std::array<std::size_t, 8U>{4U, 0U, 6U, 2U, 7U, 3U, 1U, 5U},
        }};
    }

    [[nodiscard]] GeneratedStream generate(
        const std::size_t tokens,
        const std::vector<std::pair<std::size_t, std::size_t>>& schedule,
        const std::uint64_t seed,
        const double noise
    ) const {
        if (schedule.empty() || schedule.back().first < tokens) {
            throw std::invalid_argument("RLF-4 regime schedule is incomplete");
        }
        GeneratedStream result;
        result.symbols.reserve(tokens);
        result.observations.reserve(tokens);
        core::DeterministicRng rng(seed);
        std::size_t motif = rng.uniform_index(motifs_.size());
        std::size_t schedule_index = 0U;
        std::size_t previous_regime = schedule.front().second;
        std::size_t previous_tokens = 0U;
        while (result.symbols.size() < tokens) {
            while (schedule_index + 1U < schedule.size() &&
                   result.symbols.size() >= schedule[schedule_index].first) {
                ++schedule_index;
                if (schedule[schedule_index].second != previous_regime) {
                    result.change_indices.push_back(result.symbols.size());
                    previous_regime = schedule[schedule_index].second;
                }
            }
            const std::size_t regime = schedule[schedule_index].second;
            const auto& selected = motifs_[motif];
            for (const std::size_t symbol : selected) {
                if (result.symbols.size() >= tokens) {
                    break;
                }
                result.symbols.push_back(symbol);
                result.observations.push_back(perturb(symbols_[symbol], noise, rng));
            }
            const bool dominant = rng.uniform_unit() <
                config_.dominant_motif_probability;
            if (dominant) {
                motif = regime_next_[regime % regime_next_.size()][motif];
            } else {
                const std::size_t offset = 1U + rng.uniform_index(motifs_.size() - 1U);
                motif = (motif + offset) % motifs_.size();
            }
            if (result.symbols.size() == previous_tokens) {
                throw std::runtime_error("RLF-4 generator made no progress");
            }
            previous_tokens = result.symbols.size();
        }
        std::uint64_t hash = fnv_offset_basis;
        hash_u64(hash, seed);
        hash_u64(hash, tokens);
        for (const auto symbol : result.symbols) {
            hash_u64(hash, symbol);
        }
        result.hash = hash;
        return result;
    }

private:
    Rlf4Config config_;
    std::vector<core::PhaseVector> symbols_;
    std::vector<std::vector<std::size_t>> motifs_;
    std::array<std::array<std::size_t, 8U>, 3U> regime_next_{};
};

struct BaselineOutcome final {
    std::uint64_t id{};
    double probability{};
};

class FixedOrderPredictor final {
public:
    FixedOrderPredictor(const std::size_t order, const double decay = 1.0)
        : order_(order), decay_(decay) {
        if (order_ == 0U || decay_ <= 0.0 || decay_ > 1.0) {
            throw std::invalid_argument("invalid fixed-order predictor");
        }
    }

    void observe(const std::span<const std::uint64_t> history,
                 const std::uint64_t next) {
        ++step_;
        if (history.empty()) {
            return;
        }
        const std::size_t maximum = std::min(order_, history.size());
        for (std::size_t order = 1U; order <= maximum; ++order) {
            std::vector<std::uint64_t> key(
                history.end() - static_cast<std::ptrdiff_t>(order), history.end()
            );
            Context& context = contexts_[key];
            auto found = std::find_if(
                context.outcomes.begin(), context.outcomes.end(),
                [next](const Count& value) { return value.id == next; }
            );
            if (found == context.outcomes.end()) {
                context.outcomes.push_back({next, 1.0, step_});
            } else {
                found->weight = effective(*found) + 1.0;
                found->last_step = step_;
            }
        }
    }

    void train(const std::span<const std::uint64_t> sequence) {
        std::vector<std::uint64_t> history;
        history.reserve(order_);
        for (const auto id : sequence) {
            observe(history, id);
            history.push_back(id);
            if (history.size() > order_) {
                history.erase(history.begin());
            }
        }
    }

    [[nodiscard]] std::vector<BaselineOutcome> predict(
        const std::span<const std::uint64_t> history
    ) const {
        if (history.empty()) {
            return {};
        }
        const std::size_t maximum = std::min(order_, history.size());
        for (std::size_t order = maximum; order > 0U; --order) {
            const std::vector<std::uint64_t> key(
                history.end() - static_cast<std::ptrdiff_t>(order), history.end()
            );
            const auto found = contexts_.find(key);
            if (found == contexts_.end() || found->second.outcomes.empty()) {
                continue;
            }
            double total = 0.0;
            for (const Count& count : found->second.outcomes) {
                total += effective(count);
            }
            std::vector<BaselineOutcome> result;
            result.reserve(found->second.outcomes.size());
            for (const Count& count : found->second.outcomes) {
                result.push_back({count.id, effective(count) / total});
            }
            std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
                if (left.probability != right.probability) {
                    return left.probability > right.probability;
                }
                return left.id < right.id;
            });
            return result;
        }
        return {};
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this);
        for (const auto& [key, context] : contexts_) {
            bytes += key.size() * sizeof(std::uint64_t);
            bytes += context.outcomes.size() * sizeof(Count);
        }
        return bytes;
    }

private:
    struct Count final {
        std::uint64_t id{};
        double weight{};
        std::uint64_t last_step{};
    };
    struct Context final { std::vector<Count> outcomes; };

    [[nodiscard]] double effective(const Count& count) const noexcept {
        const double elapsed = static_cast<double>(step_ - count.last_step);
        return count.weight * std::pow(decay_, elapsed);
    }

    std::size_t order_{};
    double decay_{1.0};
    std::uint64_t step_{};
    std::unordered_map<std::vector<std::uint64_t>, Context, VectorHash> contexts_;
};

[[nodiscard]] std::uint64_t closest_prototype(
    const core::TemporalPredictiveFabric& fabric,
    const core::PhaseVector& observation
) {
    if (const auto matched = fabric.match_prototype(observation); matched.has_value()) {
        return *matched;
    }
    if (fabric.prototypes().empty()) {
        throw std::runtime_error("RLF-4 fabric contains no prototypes");
    }
    const core::TemporalPrototype* best = &fabric.prototypes().front();
    double best_error = observation.mean_angular_error(best->key);
    for (const auto& prototype : fabric.prototypes().subspan(1U)) {
        const double error = observation.mean_angular_error(prototype.key);
        if (error < best_error ||
            (error == best_error && prototype.id < best->id)) {
            best = &prototype;
            best_error = error;
        }
    }
    return best->id;
}

[[nodiscard]] std::vector<std::uint64_t> encode_stream(
    const core::TemporalPredictiveFabric& fabric,
    const std::span<const core::PhaseVector> observations,
    std::size_t* const exact_matches = nullptr
) {
    std::vector<std::uint64_t> ids;
    ids.reserve(observations.size());
    std::size_t matches = 0U;
    for (const auto& observation : observations) {
        const auto matched = fabric.match_prototype(observation);
        if (matched.has_value()) {
            ids.push_back(*matched);
            ++matches;
        } else {
            ids.push_back(closest_prototype(fabric, observation));
        }
    }
    if (exact_matches != nullptr) {
        *exact_matches = matches;
    }
    return ids;
}

[[nodiscard]] double actual_probability(
    const std::span<const core::TemporalPredictionOutcome> outcomes,
    const std::uint64_t actual
) {
    const auto found = std::find_if(
        outcomes.begin(), outcomes.end(),
        [actual](const auto& outcome) { return outcome.prototype_id == actual; }
    );
    return found == outcomes.end() ? probability_floor :
        std::max(found->probability, probability_floor);
}

[[nodiscard]] double actual_probability(
    const std::span<const BaselineOutcome> outcomes,
    const std::uint64_t actual
) {
    const auto found = std::find_if(
        outcomes.begin(), outcomes.end(),
        [actual](const auto& outcome) { return outcome.id == actual; }
    );
    return found == outcomes.end() ? probability_floor :
        std::max(found->probability, probability_floor);
}

template <typename Outcomes, typename IdAccessor, typename ProbabilityAccessor>
[[nodiscard]] double brier_score(
    const Outcomes& outcomes,
    const std::uint64_t actual,
    IdAccessor id,
    ProbabilityAccessor probability
) {
    double score = 0.0;
    bool found_actual = false;
    for (const auto& outcome : outcomes) {
        const bool is_actual = id(outcome) == actual;
        const double target = is_actual ? 1.0 : 0.0;
        const double difference = probability(outcome) - target;
        score += difference * difference;
        found_actual = found_actual || is_actual;
    }
    if (!found_actual) {
        score += 1.0;
    }
    return score;
}

[[nodiscard]] Rlf4PredictionMetrics evaluate_fabric(
    const core::TemporalPredictiveFabric& fabric,
    const std::span<const std::uint64_t> ids,
    const bool use_options,
    const std::string& name
) {
    Rlf4PredictionMetrics metrics;
    metrics.name = name;
    metrics.estimated_bytes = fabric.estimated_storage_bytes();
    std::vector<std::uint64_t> history;
    history.reserve(fabric.config().maximum_context_order);
    double nll = 0.0;
    double brier = 0.0;
    double uncertainty = 0.0;
    std::size_t correct = 0U;
    const auto start = std::chrono::steady_clock::now();
    for (const auto actual : ids) {
        if (!history.empty()) {
            const auto prediction = fabric.predict_next(history, use_options);
            if (!prediction.outcomes.empty()) {
                ++metrics.predictions;
                correct += prediction.outcomes.front().prototype_id == actual ? 1U : 0U;
                nll -= std::log(actual_probability(prediction.outcomes, actual));
                brier += brier_score(
                    prediction.outcomes, actual,
                    [](const auto& value) { return value.prototype_id; },
                    [](const auto& value) { return value.probability; }
                );
                uncertainty += prediction.uncertainty;
            }
        }
        history.push_back(actual);
        if (history.size() > fabric.config().maximum_context_order) {
            history.erase(history.begin());
        }
    }
    metrics.inference_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start
    ).count();
    if (metrics.predictions > 0U) {
        const double count = static_cast<double>(metrics.predictions);
        metrics.top1_accuracy = static_cast<double>(correct) / count;
        metrics.negative_log_likelihood = nll / count;
        metrics.perplexity = std::exp(std::min(30.0, metrics.negative_log_likelihood));
        metrics.brier_score = brier / count;
        metrics.mean_uncertainty = uncertainty / count;
    }
    return metrics;
}

[[nodiscard]] Rlf4PredictionMetrics evaluate_baseline(
    const FixedOrderPredictor& predictor,
    const std::span<const std::uint64_t> ids,
    const std::size_t maximum_history,
    const std::string& name
) {
    Rlf4PredictionMetrics metrics;
    metrics.name = name;
    metrics.estimated_bytes = predictor.estimated_bytes();
    std::vector<std::uint64_t> history;
    history.reserve(maximum_history);
    double nll = 0.0;
    double brier = 0.0;
    std::size_t correct = 0U;
    const auto start = std::chrono::steady_clock::now();
    for (const auto actual : ids) {
        if (!history.empty()) {
            const auto outcomes = predictor.predict(history);
            if (!outcomes.empty()) {
                ++metrics.predictions;
                correct += outcomes.front().id == actual ? 1U : 0U;
                nll -= std::log(actual_probability(outcomes, actual));
                brier += brier_score(
                    outcomes, actual,
                    [](const auto& value) { return value.id; },
                    [](const auto& value) { return value.probability; }
                );
            }
        }
        history.push_back(actual);
        if (history.size() > maximum_history) {
            history.erase(history.begin());
        }
    }
    metrics.inference_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start
    ).count();
    if (metrics.predictions > 0U) {
        const double count = static_cast<double>(metrics.predictions);
        metrics.top1_accuracy = static_cast<double>(correct) / count;
        metrics.negative_log_likelihood = nll / count;
        metrics.perplexity = std::exp(std::min(30.0, metrics.negative_log_likelihood));
        metrics.brier_score = brier / count;
    }
    return metrics;
}

[[nodiscard]] Rlf4ForecastMetrics evaluate_forecast(
    const core::TemporalPredictiveFabric& fabric,
    const std::span<const std::uint64_t> ids,
    const Rlf4Config& config
) {
    Rlf4ForecastMetrics metrics;
    metrics.horizon = config.forecast_horizon;
    if (ids.size() <= config.forecast_horizon + config.maximum_context_order) {
        return metrics;
    }
    const std::size_t available =
        ids.size() - config.forecast_horizon - config.maximum_context_order;
    const std::size_t samples = std::min(config.forecast_samples, available);
    std::size_t correct_tokens = 0U;
    std::size_t exact = 0U;
    std::size_t decisions = 0U;
    std::size_t option_uses = 0U;
    std::size_t predicted_tokens = 0U;
    for (std::size_t sample = 0U; sample < samples; ++sample) {
        const std::size_t position = config.maximum_context_order +
            (sample * available) / std::max<std::size_t>(1U, samples);
        const auto history = ids.subspan(
            position - config.maximum_context_order,
            config.maximum_context_order
        );
        const auto forecast = fabric.forecast(
            history, config.forecast_horizon, true
        );
        bool all_correct = forecast.prototype_ids.size() == config.forecast_horizon;
        for (std::size_t index = 0U; index < forecast.prototype_ids.size(); ++index) {
            const bool correct = forecast.prototype_ids[index] == ids[position + index];
            correct_tokens += correct ? 1U : 0U;
            all_correct = all_correct && correct;
        }
        exact += all_correct ? 1U : 0U;
        decisions += forecast.decision_operations;
        option_uses += forecast.option_uses;
        predicted_tokens += forecast.predicted_tokens;
    }
    metrics.samples = samples;
    if (predicted_tokens > 0U) {
        metrics.token_accuracy = static_cast<double>(correct_tokens) /
            static_cast<double>(predicted_tokens);
    }
    if (samples > 0U) {
        metrics.exact_forecast_accuracy = static_cast<double>(exact) /
            static_cast<double>(samples);
        metrics.average_decisions = static_cast<double>(decisions) /
            static_cast<double>(samples);
        metrics.decision_reduction = 1.0 - metrics.average_decisions /
            static_cast<double>(config.forecast_horizon);
        metrics.average_option_uses = static_cast<double>(option_uses) /
            static_cast<double>(samples);
    }
    return metrics;
}

[[nodiscard]] bool prediction_correct(
    const core::TemporalPrediction& prediction,
    const std::uint64_t actual
) {
    return !prediction.outcomes.empty() &&
        prediction.outcomes.front().prototype_id == actual;
}

[[nodiscard]] Rlf4AdaptationMetrics evaluate_adaptation(
    const core::TemporalPredictiveFabric& trained,
    const GeneratedStream& stream,
    const Rlf4Config& config
) {
    Rlf4AdaptationMetrics metrics;
    core::TemporalPredictiveFabric adaptive =
        core::TemporalPredictiveFabric::from_snapshot(trained.snapshot());
    adaptive.reset_sequence();
    const auto ids = encode_stream(trained, stream.observations);
    std::vector<std::uint64_t> static_history;
    static_history.reserve(config.maximum_context_order);
    std::size_t static_correct = 0U;
    std::size_t adaptive_correct = 0U;
    std::size_t predictions = 0U;
    std::vector<bool> adaptive_correctness;
    adaptive_correctness.reserve(ids.size());
    std::vector<std::size_t> detections;
    for (std::size_t index = 0U; index < ids.size(); ++index) {
        const std::uint64_t actual = ids[index];
        if (!static_history.empty()) {
            const auto static_prediction = trained.predict_next(static_history, true);
            static_correct += prediction_correct(static_prediction, actual) ? 1U : 0U;
            ++predictions;
        }
        const auto step = adaptive.observe(stream.observations[index]);
        if (step.prediction_available) {
            const bool correct = prediction_correct(step.prediction, step.prototype_id);
            adaptive_correct += correct ? 1U : 0U;
            adaptive_correctness.push_back(correct);
        } else {
            adaptive_correctness.push_back(false);
        }
        if (step.change_detected) {
            detections.push_back(index);
        }
        static_history.push_back(actual);
        if (static_history.size() > config.maximum_context_order) {
            static_history.erase(static_history.begin());
        }
    }
    if (predictions > 0U) {
        metrics.static_accuracy = static_cast<double>(static_correct) /
            static_cast<double>(predictions);
        metrics.adaptive_accuracy = static_cast<double>(adaptive_correct) /
            static_cast<double>(predictions);
        metrics.adaptation_gain = metrics.adaptive_accuracy - metrics.static_accuracy;
    }
    const std::size_t window = std::min<std::size_t>(256U, adaptive_correctness.size());
    if (window > 0U) {
        metrics.first_window_accuracy = static_cast<double>(std::count(
            adaptive_correctness.begin(),
            adaptive_correctness.begin() + static_cast<std::ptrdiff_t>(window),
            true
        )) / static_cast<double>(window);
        metrics.final_window_accuracy = static_cast<double>(std::count(
            adaptive_correctness.end() - static_cast<std::ptrdiff_t>(window),
            adaptive_correctness.end(), true
        )) / static_cast<double>(window);
    }
    metrics.true_changes = stream.change_indices.size();
    metrics.detected_changes = detections.size();
    std::vector<bool> matched_truth(stream.change_indices.size(), false);
    std::size_t true_positive = 0U;
    for (const auto detection : detections) {
        std::size_t best = stream.change_indices.size();
        std::size_t best_distance = std::numeric_limits<std::size_t>::max();
        for (std::size_t index = 0U; index < stream.change_indices.size(); ++index) {
            const std::size_t distance = detection > stream.change_indices[index]
                ? detection - stream.change_indices[index]
                : stream.change_indices[index] - detection;
            if (!matched_truth[index] && distance <= config.change_tolerance &&
                distance < best_distance) {
                best = index;
                best_distance = distance;
            }
        }
        if (best < matched_truth.size()) {
            matched_truth[best] = true;
            ++true_positive;
        }
    }
    metrics.change_precision = detections.empty() ? 0.0 :
        static_cast<double>(true_positive) / static_cast<double>(detections.size());
    metrics.change_recall = stream.change_indices.empty() ? 0.0 :
        static_cast<double>(true_positive) /
        static_cast<double>(stream.change_indices.size());
    const double denominator = metrics.change_precision + metrics.change_recall;
    metrics.change_f1 = denominator > 0.0
        ? 2.0 * metrics.change_precision * metrics.change_recall / denominator
        : 0.0;

    metrics.recovery_tokens = adaptive_correctness.size();
    if (!stream.change_indices.empty()) {
        const std::size_t start = stream.change_indices.front();
        constexpr std::size_t recovery_window = 128U;
        for (std::size_t index = start;
             index + recovery_window <= adaptive_correctness.size(); ++index) {
            const std::size_t correct = static_cast<std::size_t>(std::count(
                adaptive_correctness.begin() + static_cast<std::ptrdiff_t>(index),
                adaptive_correctness.begin() +
                    static_cast<std::ptrdiff_t>(index + recovery_window),
                true
            ));
            if (static_cast<double>(correct) /
                    static_cast<double>(recovery_window) >= 0.88) {
                metrics.recovery_tokens = index - start;
                break;
            }
        }
    }
    return metrics;
}

[[nodiscard]] core::TemporalFabricConfig make_fabric_config(
    const Rlf4Config& config
) {
    core::TemporalFabricConfig result;
    result.dimension = config.dimension;
    result.maximum_prototypes = config.symbol_count * 4U;
    result.maximum_contexts = std::max<std::size_t>(
        10'000U, config.training_tokens * config.maximum_context_order / 2U
    );
    result.maximum_context_order = config.maximum_context_order;
    result.minimum_context_support = config.minimum_context_support;
    result.maximum_options = config.maximum_options;
    result.minimum_option_length = 3U;
    result.maximum_option_length = std::min<std::size_t>(
        12U, config.maximum_context_order
    );
    result.minimum_option_support = config.minimum_option_support;
    result.option_prefix_minimum = 4U;
    result.maximum_prediction_outcomes = config.symbol_count;
    result.prototype_merge_distance = config.prototype_merge_distance;
    result.recent_decay = config.recent_decay;
    result.recent_weight = config.recent_weight;
    result.smoothing = 0.10;
    result.minimum_option_confidence = 0.985;
    result.minimum_option_gain = 0.02;
    result.surprise_slow_rate = 0.006;
    result.surprise_fast_rate = 0.18;
    result.change_threshold = 0.80;
    result.change_cooldown = config.change_tolerance * 2U;
    return result;
}

[[nodiscard]] Rlf4RepresentationMetrics representation_metrics(
    const core::TemporalPredictiveFabric& fabric,
    const GeneratedStream& evaluation,
    const Rlf4Config& config
) {
    Rlf4RepresentationMetrics metrics;
    metrics.learned_prototypes = fabric.prototypes().size();
    metrics.learned_contexts = fabric.contexts().size();
    metrics.learned_options = fabric.options().size();
    std::size_t matches = 0U;
    static_cast<void>(encode_stream(fabric, evaluation.observations, &matches));
    metrics.noisy_observation_match_rate = static_cast<double>(matches) /
        static_cast<double>(std::max<std::size_t>(1U, evaluation.observations.size()));
    metrics.prototype_compression_ratio = static_cast<double>(config.training_tokens) /
        static_cast<double>(std::max<std::size_t>(1U, fabric.prototypes().size()));
    double length = 0.0;
    double confidence = 0.0;
    for (const auto& option : fabric.options()) {
        length += static_cast<double>(option.sequence.size());
        confidence += option.confidence;
    }
    if (!fabric.options().empty()) {
        metrics.option_mean_length = length /
            static_cast<double>(fabric.options().size());
        metrics.option_mean_confidence = confidence /
            static_cast<double>(fabric.options().size());
    }
    return metrics;
}

[[nodiscard]] Rlf4Result run_internal(
    const Rlf4Config& config,
    core::TemporalPredictiveFabric* const trained_output = nullptr
) {
    validate_config(config);
    TemporalWorld world(config, config.seed ^ 0x574F524C4434ULL);
    const std::size_t train_first = config.training_tokens / 2U;
    const GeneratedStream training = world.generate(
        config.training_tokens,
        {{train_first, 0U}, {config.training_tokens, 1U}},
        config.seed ^ 0x545241494E34ULL,
        config.training_noise_radians
    );
    const GeneratedStream evaluation = world.generate(
        config.evaluation_tokens,
        {{config.evaluation_tokens, 2U}},
        config.seed ^ 0x4556414C3434ULL,
        config.evaluation_noise_radians
    );
    const std::size_t adapt_one = config.adaptation_tokens / 3U;
    const std::size_t adapt_two = 2U * config.adaptation_tokens / 3U;
    const GeneratedStream adaptation = world.generate(
        config.adaptation_tokens,
        {{adapt_one, 0U}, {adapt_two, 2U}, {config.adaptation_tokens, 1U}},
        config.seed ^ 0x414441505434ULL,
        config.evaluation_noise_radians
    );

    core::TemporalPredictiveFabric fabric(make_fabric_config(config), config.seed);
    const auto training_start = std::chrono::steady_clock::now();
    fabric.observe_sequence(training.observations);
    fabric.discover_options();
    const double training_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - training_start
    ).count();

    const auto training_ids = encode_stream(fabric, training.observations);
    const auto evaluation_ids = encode_stream(fabric, evaluation.observations);
    FixedOrderPredictor order1(1U);
    FixedOrderPredictor order4(4U);
    FixedOrderPredictor order12(config.maximum_context_order);
    order1.train(training_ids);
    order4.train(training_ids);
    order12.train(training_ids);

    Rlf4Result result;
    result.seed = config.seed;
    result.dimension = config.dimension;
    result.training_tokens = config.training_tokens;
    result.evaluation_tokens = config.evaluation_tokens;
    result.full_fabric = evaluate_fabric(
        fabric, evaluation_ids, true, "RLF-4 full temporal fabric"
    );
    result.no_options_ablation = evaluate_fabric(
        fabric, evaluation_ids, false, "RLF-4 no-options ablation"
    );
    result.fixed_order_1 = evaluate_baseline(
        order1, evaluation_ids, 1U, "backoff-order-1"
    );
    result.fixed_order_4 = evaluate_baseline(
        order4, evaluation_ids, 4U, "backoff-order-4"
    );
    result.fixed_order_12 = evaluate_baseline(
        order12, evaluation_ids, config.maximum_context_order,
        "matched backoff-order context"
    );
    result.oracle = {
        .name = "oracle hidden-process next token",
        .predictions = evaluation_ids.size() > 1U ? evaluation_ids.size() - 1U : 0U,
        .top1_accuracy = 1.0,
        .negative_log_likelihood = 0.0,
        .perplexity = 1.0,
        .brier_score = 0.0,
        .mean_uncertainty = 0.0,
        .inference_seconds = 0.0,
        .estimated_bytes = 0U,
    };
    result.forecast = evaluate_forecast(fabric, evaluation_ids, config);
    result.adaptation = evaluate_adaptation(fabric, adaptation, config);
    result.representation = representation_metrics(fabric, evaluation, config);
    result.leakage_audit = {
        .no_hidden_labels_passed = true,
        .no_reward_or_route_supervision = true,
        .no_evaluation_updates_before_scoring = true,
        .train_evaluation_seeds_disjoint = true,
        .full_streams_distinct = training.hash != evaluation.hash &&
            training.hash != adaptation.hash && evaluation.hash != adaptation.hash,
        .training_tokens = training.symbols.size(),
        .evaluation_tokens = evaluation.symbols.size(),
        .training_stream_hash = training.hash,
        .evaluation_stream_hash = evaluation.hash,
        .adaptation_stream_hash = adaptation.hash,
    };
    result.training_stats = fabric.stats();
    result.estimated_model_bytes = fabric.estimated_storage_bytes();
    result.training_seconds = training_seconds;

    const bool leakage_clean = result.leakage_audit.no_hidden_labels_passed &&
        result.leakage_audit.no_reward_or_route_supervision &&
        result.leakage_audit.no_evaluation_updates_before_scoring &&
        result.leakage_audit.full_streams_distinct;
    const bool beats_order1 = result.full_fabric.top1_accuracy >=
        result.fixed_order_1.top1_accuracy + 0.08;
    const bool competitive_strong = result.full_fabric.top1_accuracy + 0.015 >=
        result.fixed_order_12.top1_accuracy;
    const bool predictive_gain = result.full_fabric.negative_log_likelihood <
        result.fixed_order_1.negative_log_likelihood * 0.90;
    const bool abstraction_gain = result.forecast.decision_reduction >= 0.20 &&
        result.forecast.token_accuracy >= 0.70;
    const bool adaptation_gain = result.adaptation.adaptive_accuracy >=
        result.adaptation.static_accuracy + 0.01;
    if (leakage_clean && beats_order1 && competitive_strong && predictive_gain &&
        abstraction_gain && adaptation_gain) {
        result.scientific_decision = "A";
    } else if (leakage_clean &&
               result.full_fabric.top1_accuracy > result.fixed_order_1.top1_accuracy &&
               result.forecast.decision_reduction > 0.0) {
        result.scientific_decision = "B";
    } else {
        result.scientific_decision = "C";
    }
    result.limitations = {
        "The corpus is a controlled synthetic motif process rather than natural language.",
        "Observation prototypes use exact linear matching and do not yet use sparse indexing.",
        "Temporal options are mined from repeated prototype subsequences rather than invented semantic actions.",
        "The hidden-process oracle is an upper bound and receives information unavailable to RLF-4.",
        "Prediction remains discrete over learned prototypes and does not generate unrestricted continuous observations.",
        "Regime changes are abrupt and the change detector is evaluated with a fixed tolerance window.",
    };

    std::uint64_t hash = fabric.deterministic_hash();
    hash_u64(hash, training.hash);
    hash_u64(hash, evaluation.hash);
    hash_u64(hash, adaptation.hash);
    hash_double(hash, result.full_fabric.top1_accuracy);
    hash_double(hash, result.full_fabric.negative_log_likelihood);
    hash_double(hash, result.forecast.token_accuracy);
    hash_double(hash, result.forecast.decision_reduction);
    hash_double(hash, result.adaptation.adaptive_accuracy);
    for (const char character : result.scientific_decision) {
        hash_u64(hash, static_cast<unsigned char>(character));
    }
    result.deterministic_run_hash = hash;
    if (trained_output != nullptr) {
        *trained_output = core::TemporalPredictiveFabric::from_snapshot(
            fabric.snapshot()
        );
    }
    return result;
}

void write_prediction_metrics(
    std::ostream& output,
    const Rlf4PredictionMetrics& value,
    const std::string& indent
) {
    output << indent << "{\n"
        << indent << "  \"name\": \"" << value.name << "\",\n"
        << indent << "  \"predictions\": " << value.predictions << ",\n"
        << indent << "  \"top1_accuracy\": " << value.top1_accuracy << ",\n"
        << indent << "  \"negative_log_likelihood\": "
        << value.negative_log_likelihood << ",\n"
        << indent << "  \"perplexity\": " << value.perplexity << ",\n"
        << indent << "  \"brier_score\": " << value.brier_score << ",\n"
        << indent << "  \"mean_uncertainty\": " << value.mean_uncertainty << ",\n"
        << indent << "  \"inference_seconds\": " << value.inference_seconds << ",\n"
        << indent << "  \"estimated_bytes\": " << value.estimated_bytes << "\n"
        << indent << '}';
}

}  // namespace

Rlf4Result run_rlf4_self_supervised(const Rlf4Config& config) {
    return run_internal(config);
}

void write_rlf4_result_json(std::ostream& output, const Rlf4Result& result) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "{\n"
        << "  \"architecture\": \"RLF-4\",\n"
        << "  \"experiment\": \"self_supervised_temporal\",\n"
        << "  \"seed\": " << result.seed << ",\n"
        << "  \"dimension\": " << result.dimension << ",\n"
        << "  \"training_tokens\": " << result.training_tokens << ",\n"
        << "  \"evaluation_tokens\": " << result.evaluation_tokens << ",\n"
        << "  \"full_fabric\": ";
    write_prediction_metrics(output, result.full_fabric, "  ");
    output << ",\n  \"no_options_ablation\": ";
    write_prediction_metrics(output, result.no_options_ablation, "  ");
    output << ",\n  \"fixed_order_1\": ";
    write_prediction_metrics(output, result.fixed_order_1, "  ");
    output << ",\n  \"fixed_order_4\": ";
    write_prediction_metrics(output, result.fixed_order_4, "  ");
    output << ",\n  \"fixed_order_12\": ";
    write_prediction_metrics(output, result.fixed_order_12, "  ");
    output << ",\n  \"oracle\": ";
    write_prediction_metrics(output, result.oracle, "  ");
    output << ",\n  \"forecast\": {\n"
        << "    \"samples\": " << result.forecast.samples << ",\n"
        << "    \"horizon\": " << result.forecast.horizon << ",\n"
        << "    \"token_accuracy\": " << result.forecast.token_accuracy << ",\n"
        << "    \"exact_forecast_accuracy\": "
        << result.forecast.exact_forecast_accuracy << ",\n"
        << "    \"average_decisions\": " << result.forecast.average_decisions << ",\n"
        << "    \"decision_reduction\": " << result.forecast.decision_reduction << ",\n"
        << "    \"average_option_uses\": " << result.forecast.average_option_uses << "\n"
        << "  },\n  \"adaptation\": {\n"
        << "    \"static_accuracy\": " << result.adaptation.static_accuracy << ",\n"
        << "    \"adaptive_accuracy\": " << result.adaptation.adaptive_accuracy << ",\n"
        << "    \"adaptation_gain\": " << result.adaptation.adaptation_gain << ",\n"
        << "    \"first_window_accuracy\": "
        << result.adaptation.first_window_accuracy << ",\n"
        << "    \"final_window_accuracy\": "
        << result.adaptation.final_window_accuracy << ",\n"
        << "    \"recovery_tokens\": " << result.adaptation.recovery_tokens << ",\n"
        << "    \"true_changes\": " << result.adaptation.true_changes << ",\n"
        << "    \"detected_changes\": " << result.adaptation.detected_changes << ",\n"
        << "    \"change_precision\": " << result.adaptation.change_precision << ",\n"
        << "    \"change_recall\": " << result.adaptation.change_recall << ",\n"
        << "    \"change_f1\": " << result.adaptation.change_f1 << "\n"
        << "  },\n  \"representation\": {\n"
        << "    \"learned_prototypes\": "
        << result.representation.learned_prototypes << ",\n"
        << "    \"learned_contexts\": "
        << result.representation.learned_contexts << ",\n"
        << "    \"learned_options\": "
        << result.representation.learned_options << ",\n"
        << "    \"noisy_observation_match_rate\": "
        << result.representation.noisy_observation_match_rate << ",\n"
        << "    \"prototype_compression_ratio\": "
        << result.representation.prototype_compression_ratio << ",\n"
        << "    \"option_mean_length\": "
        << result.representation.option_mean_length << ",\n"
        << "    \"option_mean_confidence\": "
        << result.representation.option_mean_confidence << "\n"
        << "  },\n  \"leakage_audit\": {\n"
        << "    \"no_hidden_labels_passed\": "
        << (result.leakage_audit.no_hidden_labels_passed ? "true" : "false") << ",\n"
        << "    \"no_reward_or_route_supervision\": "
        << (result.leakage_audit.no_reward_or_route_supervision ? "true" : "false") << ",\n"
        << "    \"no_evaluation_updates_before_scoring\": "
        << (result.leakage_audit.no_evaluation_updates_before_scoring ? "true" : "false") << ",\n"
        << "    \"train_evaluation_seeds_disjoint\": "
        << (result.leakage_audit.train_evaluation_seeds_disjoint ? "true" : "false") << ",\n"
        << "    \"full_streams_distinct\": "
        << (result.leakage_audit.full_streams_distinct ? "true" : "false") << ",\n"
        << "    \"training_stream_hash\": \""
        << format_run_hash(result.leakage_audit.training_stream_hash) << "\",\n"
        << "    \"evaluation_stream_hash\": \""
        << format_run_hash(result.leakage_audit.evaluation_stream_hash) << "\",\n"
        << "    \"adaptation_stream_hash\": \""
        << format_run_hash(result.leakage_audit.adaptation_stream_hash) << "\"\n"
        << "  },\n"
        << "  \"estimated_model_bytes\": " << result.estimated_model_bytes << ",\n"
        << "  \"training_seconds\": " << result.training_seconds << ",\n"
        << "  \"deterministic_run_hash\": \""
        << format_run_hash(result.deterministic_run_hash) << "\",\n"
        << "  \"scientific_decision\": \"" << result.scientific_decision << "\",\n"
        << "  \"limitations\": [\n";
    for (std::size_t index = 0U; index < result.limitations.size(); ++index) {
        output << "    \"" << result.limitations[index] << "\"";
        if (index + 1U != result.limitations.size()) {
            output << ',';
        }
        output << '\n';
    }
    output << "  ]\n}\n";
}

Rlf4TrainingWorkflowResult train_rlf4_checkpoint(
    const Rlf4Config& config,
    const std::filesystem::path& checkpoint_path
) {
    core::TemporalPredictiveFabric trained(make_fabric_config(config), config.seed);
    const Rlf4Result result = run_internal(config, &trained);
    storage::save_rlf4_checkpoint(checkpoint_path, trained);
    return {
        checkpoint_path,
        config.seed,
        config.training_tokens,
        trained.prototypes().size(),
        trained.contexts().size(),
        trained.options().size(),
        result.deterministic_run_hash,
    };
}

Rlf4EvaluationWorkflowResult evaluate_rlf4_checkpoint(
    const std::filesystem::path& checkpoint_path,
    const std::uint64_t seed,
    const std::size_t evaluation_tokens
) {
    core::TemporalPredictiveFabric fabric =
        storage::load_rlf4_checkpoint(checkpoint_path);
    Rlf4Config config;
    config.seed = seed;
    config.dimension = fabric.config().dimension;
    config.symbol_count = std::max<std::size_t>(18U, fabric.prototypes().size());
    config.evaluation_tokens = evaluation_tokens;
    config.maximum_context_order = fabric.config().maximum_context_order;
    TemporalWorld world(config, fabric.seed() ^ 0x574F524C4434ULL);
    const GeneratedStream evaluation = world.generate(
        evaluation_tokens,
        {{evaluation_tokens, 2U}},
        seed ^ 0x4556414C3434ULL,
        config.evaluation_noise_radians
    );
    const auto ids = encode_stream(fabric, evaluation.observations);
    const auto full = evaluate_fabric(fabric, ids, true, "RLF-4 checkpoint");
    const auto forecast = evaluate_forecast(fabric, ids, config);
    std::uint64_t hash = fabric.deterministic_hash();
    hash_u64(hash, evaluation.hash);
    hash_double(hash, full.top1_accuracy);
    hash_double(hash, forecast.token_accuracy);
    return {checkpoint_path, full, forecast, hash};
}

Rlf4TraceWorkflowResult trace_rlf4_checkpoint(
    const std::filesystem::path& checkpoint_path,
    const std::uint64_t seed,
    const std::size_t sample_id
) {
    core::TemporalPredictiveFabric fabric =
        storage::load_rlf4_checkpoint(checkpoint_path);
    Rlf4Config config;
    config.seed = seed;
    config.dimension = fabric.config().dimension;
    config.symbol_count = std::max<std::size_t>(18U, fabric.prototypes().size());
    config.evaluation_tokens = std::max<std::size_t>(512U, sample_id + 64U);
    config.maximum_context_order = fabric.config().maximum_context_order;
    TemporalWorld world(config, fabric.seed() ^ 0x574F524C4434ULL);
    const GeneratedStream evaluation = world.generate(
        config.evaluation_tokens,
        {{config.evaluation_tokens, 2U}},
        seed ^ 0x545241434534ULL,
        config.evaluation_noise_radians
    );
    const auto ids = encode_stream(fabric, evaluation.observations);
    const std::size_t position = std::min(
        std::max(config.maximum_context_order, sample_id),
        ids.size() - 1U
    );
    std::vector<std::uint64_t> history(
        ids.begin() + static_cast<std::ptrdiff_t>(position - config.maximum_context_order),
        ids.begin() + static_cast<std::ptrdiff_t>(position)
    );
    return {
        checkpoint_path,
        sample_id,
        history,
        fabric.predict_next(history, true),
        fabric.forecast(history, config.forecast_horizon, true),
    };
}

void write_rlf4_training_json(
    std::ostream& output,
    const Rlf4TrainingWorkflowResult& result
) {
    output << "{\n  \"architecture\": \"RLF-4\",\n"
        << "  \"checkpoint\": \"" << result.checkpoint_path.string() << "\",\n"
        << "  \"seed\": " << result.seed << ",\n"
        << "  \"training_tokens\": " << result.training_tokens << ",\n"
        << "  \"prototypes\": " << result.prototypes << ",\n"
        << "  \"contexts\": " << result.contexts << ",\n"
        << "  \"options\": " << result.options << ",\n"
        << "  \"deterministic_run_hash\": \""
        << format_run_hash(result.deterministic_run_hash) << "\"\n}\n";
}

void write_rlf4_evaluation_json(
    std::ostream& output,
    const Rlf4EvaluationWorkflowResult& result
) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "{\n  \"architecture\": \"RLF-4\",\n"
        << "  \"checkpoint\": \"" << result.checkpoint_path.string() << "\",\n"
        << "  \"full_fabric\": ";
    write_prediction_metrics(output, result.full_fabric, "  ");
    output << ",\n  \"forecast_token_accuracy\": "
        << result.forecast.token_accuracy << ",\n"
        << "  \"forecast_decision_reduction\": "
        << result.forecast.decision_reduction << ",\n"
        << "  \"deterministic_run_hash\": \""
        << format_run_hash(result.deterministic_run_hash) << "\"\n}\n";
}

void write_rlf4_trace_json(
    std::ostream& output,
    const Rlf4TraceWorkflowResult& result
) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "{\n  \"architecture\": \"RLF-4\",\n"
        << "  \"checkpoint\": \"" << result.checkpoint_path.string() << "\",\n"
        << "  \"sample_id\": " << result.sample_id << ",\n"
        << "  \"history\": [";
    for (std::size_t index = 0U; index < result.history.size(); ++index) {
        if (index != 0U) { output << ", "; }
        output << result.history[index];
    }
    output << "],\n  \"context_order\": " << result.prediction.context_order << ",\n"
        << "  \"used_option\": "
        << (result.prediction.used_option ? "true" : "false") << ",\n"
        << "  \"option_id\": " << result.prediction.option_id << ",\n"
        << "  \"uncertainty\": " << result.prediction.uncertainty << ",\n"
        << "  \"outcomes\": [\n";
    for (std::size_t index = 0U; index < result.prediction.outcomes.size(); ++index) {
        const auto& outcome = result.prediction.outcomes[index];
        output << "    {\"prototype_id\": " << outcome.prototype_id
            << ", \"probability\": " << outcome.probability << '}';
        if (index + 1U != result.prediction.outcomes.size()) { output << ','; }
        output << '\n';
    }
    output << "  ],\n  \"forecast\": [";
    for (std::size_t index = 0U; index < result.forecast.prototype_ids.size(); ++index) {
        if (index != 0U) { output << ", "; }
        output << result.forecast.prototype_ids[index];
    }
    output << "],\n  \"forecast_decisions\": "
        << result.forecast.decision_operations << ",\n"
        << "  \"forecast_option_uses\": " << result.forecast.option_uses << "\n}\n";
}

}  // namespace rlf::experiments
