#include "rlf/core/temporal_predictive_fabric.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rlf::core {
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

[[nodiscard]] double normalized_distance(
    const PhaseVector& left,
    const PhaseVector& right
) {
    return left.mean_angular_error(right) / std::numbers::pi_v<double>;
}

void validate_config(const TemporalFabricConfig& config) {
    if (config.dimension == 0U || config.maximum_prototypes == 0U ||
        config.maximum_contexts == 0U || config.maximum_context_order == 0U ||
        config.minimum_context_support == 0U || config.maximum_options == 0U ||
        config.minimum_option_length < 2U ||
        config.maximum_option_length < config.minimum_option_length ||
        config.minimum_option_support < 2U ||
        config.option_prefix_minimum == 0U ||
        config.maximum_prediction_outcomes == 0U ||
        !std::isfinite(config.prototype_merge_distance) ||
        config.prototype_merge_distance < 0.0 ||
        !std::isfinite(config.recent_decay) || config.recent_decay <= 0.0 ||
        config.recent_decay > 1.0 || !std::isfinite(config.recent_weight) ||
        config.recent_weight < 0.0 || config.recent_weight > 1.0 ||
        !std::isfinite(config.smoothing) || config.smoothing < 0.0 ||
        !std::isfinite(config.minimum_option_confidence) ||
        config.minimum_option_confidence < 0.0 ||
        config.minimum_option_confidence > 1.0 ||
        !std::isfinite(config.minimum_option_gain) ||
        config.minimum_option_gain < 0.0 ||
        !std::isfinite(config.surprise_slow_rate) ||
        config.surprise_slow_rate <= 0.0 || config.surprise_slow_rate > 1.0 ||
        !std::isfinite(config.surprise_fast_rate) ||
        config.surprise_fast_rate <= 0.0 || config.surprise_fast_rate > 1.0 ||
        config.surprise_fast_rate <= config.surprise_slow_rate ||
        !std::isfinite(config.change_threshold) || config.change_threshold < 0.0) {
        throw std::invalid_argument("invalid RLF-4 temporal fabric configuration");
    }
}

[[nodiscard]] bool vector_less(
    const std::vector<std::uint64_t>& left,
    const std::vector<std::uint64_t>& right
) noexcept {
    return std::lexicographical_compare(
        left.begin(), left.end(), right.begin(), right.end()
    );
}

}  // namespace

std::size_t TemporalPredictiveFabric::SequenceHash::operator()(
    const std::vector<std::uint64_t>& value
) const noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, static_cast<std::uint64_t>(value.size()));
    for (const std::uint64_t item : value) {
        hash_u64(hash, item);
    }
    return static_cast<std::size_t>(hash);
}

TemporalPredictiveFabric::TemporalPredictiveFabric(
    TemporalFabricConfig config,
    const std::uint64_t seed
)
    : config_(std::move(config)), seed_(seed) {
    validate_config(config_);
    prototypes_.reserve(std::min<std::size_t>(config_.maximum_prototypes, 4096U));
    contexts_.reserve(std::min<std::size_t>(config_.maximum_contexts, 65'536U));
    options_.reserve(std::min<std::size_t>(config_.maximum_options, 4096U));
    recent_history_.reserve(config_.maximum_context_order);
}

const TemporalFabricConfig& TemporalPredictiveFabric::config() const noexcept {
    return config_;
}

std::uint64_t TemporalPredictiveFabric::seed() const noexcept {
    return seed_;
}

std::uint64_t TemporalPredictiveFabric::training_step() const noexcept {
    return training_step_;
}

std::span<const TemporalPrototype> TemporalPredictiveFabric::prototypes() const noexcept {
    return prototypes_;
}

std::span<const TemporalContext> TemporalPredictiveFabric::contexts() const noexcept {
    return contexts_;
}

std::span<const TemporalOption> TemporalPredictiveFabric::options() const noexcept {
    return options_;
}

const TemporalFabricStats& TemporalPredictiveFabric::stats() const noexcept {
    return stats_;
}

void TemporalPredictiveFabric::reset_sequence() {
    recent_history_.clear();
    mining_sequences_.emplace_back();
    ++stats_.sequences_seen;
}

std::optional<std::size_t> TemporalPredictiveFabric::prototype_index(
    const std::uint64_t id
) const noexcept {
    const auto found = prototype_index_by_id_.find(id);
    if (found == prototype_index_by_id_.end()) {
        return std::nullopt;
    }
    return found->second;
}

std::optional<std::uint64_t> TemporalPredictiveFabric::match_prototype(
    const PhaseVector& observation,
    std::size_t* const comparisons
) const {
    if (observation.size() != config_.dimension) {
        throw std::invalid_argument("RLF-4 observation dimension mismatch");
    }
    if (comparisons != nullptr) {
        *comparisons = prototypes_.size();
    }
    if (prototypes_.empty()) {
        return std::nullopt;
    }
    std::size_t best_index = 0U;
    double best_distance = normalized_distance(observation, prototypes_.front().key);
    for (std::size_t index = 1U; index < prototypes_.size(); ++index) {
        const double distance = normalized_distance(observation, prototypes_[index].key);
        if (distance < best_distance ||
            (distance == best_distance &&
             prototypes_[index].id < prototypes_[best_index].id)) {
            best_distance = distance;
            best_index = index;
        }
    }
    if (best_distance > config_.prototype_merge_distance) {
        return std::nullopt;
    }
    return prototypes_[best_index].id;
}

std::pair<std::uint64_t, bool> TemporalPredictiveFabric::encode_or_create(
    const PhaseVector& observation
) {
    if (observation.size() != config_.dimension) {
        throw std::invalid_argument("RLF-4 observation dimension mismatch");
    }
    std::size_t comparisons = 0U;
    const auto matched = match_prototype(observation, &comparisons);
    stats_.context_comparisons += comparisons;
    if (matched.has_value()) {
        const std::size_t index = *prototype_index(*matched);
        TemporalPrototype& prototype = prototypes_[index];
        const float old_weight = static_cast<float>(prototype.support);
        const std::array<PhaseVector, 2U> values{prototype.key, observation};
        const std::array<float, 2U> weights{old_weight, 1.0F};
        prototype.key = PhaseVector::weighted_circular_average(values, weights);
        ++prototype.support;
        prototype.last_used_step = training_step_;
        ++stats_.prototypes_merged;
        return {*matched, false};
    }
    if (prototypes_.size() >= config_.maximum_prototypes) {
        if (prototypes_.empty()) {
            throw std::runtime_error("RLF-4 prototype capacity is zero");
        }
        std::size_t best_index = 0U;
        double best_distance = normalized_distance(observation, prototypes_.front().key);
        for (std::size_t index = 1U; index < prototypes_.size(); ++index) {
            const double distance = normalized_distance(observation, prototypes_[index].key);
            if (distance < best_distance) {
                best_distance = distance;
                best_index = index;
            }
        }
        TemporalPrototype& prototype = prototypes_[best_index];
        ++prototype.support;
        prototype.last_used_step = training_step_;
        ++stats_.prototypes_merged;
        return {prototype.id, false};
    }
    const std::uint64_t id = next_prototype_id_++;
    prototypes_.push_back({id, observation, 1ULL, training_step_, training_step_});
    prototype_index_by_id_[id] = prototypes_.size() - 1U;
    ++stats_.prototypes_created;
    return {id, true};
}

std::optional<std::size_t> TemporalPredictiveFabric::context_index(
    const std::span<const std::uint64_t> history
) const {
    const std::vector<std::uint64_t> key(history.begin(), history.end());
    const auto found = context_index_by_history_.find(key);
    if (found == context_index_by_history_.end()) {
        return std::nullopt;
    }
    return found->second;
}

std::size_t TemporalPredictiveFabric::get_or_create_context(
    const std::span<const std::uint64_t> history
) {
    if (history.empty() || history.size() > config_.maximum_context_order) {
        throw std::invalid_argument("invalid RLF-4 temporal context length");
    }
    if (const auto found = context_index(history); found.has_value()) {
        return *found;
    }
    if (contexts_.size() >= config_.maximum_contexts) {
        throw std::runtime_error("RLF-4 temporal context capacity exceeded");
    }
    TemporalContext context;
    context.id = next_context_id_++;
    context.history.assign(history.begin(), history.end());
    context.creation_step = training_step_;
    context.last_used_step = training_step_;
    contexts_.push_back(std::move(context));
    context_index_by_history_[contexts_.back().history] = contexts_.size() - 1U;
    ++stats_.contexts_created;
    return contexts_.size() - 1U;
}

void TemporalPredictiveFabric::update_context(
    const std::span<const std::uint64_t> history,
    const std::uint64_t next_id
) {
    const std::size_t index = get_or_create_context(history);
    TemporalContext& context = contexts_[index];
    ++context.support;
    context.last_used_step = training_step_;
    auto outcome = std::find_if(
        context.outcomes.begin(),
        context.outcomes.end(),
        [next_id](const TemporalOutcomeCount& value) {
            return value.prototype_id == next_id;
        }
    );
    if (outcome == context.outcomes.end()) {
        context.outcomes.push_back({next_id, 1ULL, 1.0, training_step_});
    } else {
        outcome->recent_count = effective_recent(*outcome) + 1.0;
        outcome->last_update_step = training_step_;
        ++outcome->total_count;
    }
    ++stats_.contexts_updated;
}

double TemporalPredictiveFabric::effective_recent(
    const TemporalOutcomeCount& outcome
) const noexcept {
    if (training_step_ <= outcome.last_update_step) {
        return outcome.recent_count;
    }
    const auto elapsed = static_cast<double>(training_step_ - outcome.last_update_step);
    return outcome.recent_count * std::pow(config_.recent_decay, elapsed);
}

std::optional<std::pair<const TemporalOption*, std::size_t>>
TemporalPredictiveFabric::best_option_continuation(
    const std::span<const std::uint64_t> history
) const {
    const TemporalOption* best = nullptr;
    const std::size_t prefix = config_.option_prefix_minimum;
    if (history.size() < prefix) {
        return std::nullopt;
    }
    const auto suffix = history.subspan(history.size() - prefix, prefix);
    for (const TemporalOption& option : options_) {
        ++stats_.option_comparisons;
        if (option.sequence.size() <= prefix ||
            option.confidence < config_.minimum_option_confidence ||
            option.predictive_gain < config_.minimum_option_gain) {
            continue;
        }
        if (!std::equal(suffix.begin(), suffix.end(), option.sequence.begin())) {
            continue;
        }
        if (best == nullptr || option.confidence > best->confidence ||
            (option.confidence == best->confidence &&
             (option.sequence.size() > best->sequence.size() ||
              (option.sequence.size() == best->sequence.size() &&
               option.id < best->id)))) {
            best = &option;
        }
    }
    if (best == nullptr) {
        return std::nullopt;
    }
    return std::pair{best, prefix};
}

TemporalPrediction TemporalPredictiveFabric::predict_next(
    const std::span<const std::uint64_t> history,
    const bool use_options
) const {
    TemporalPrediction prediction;
    ++stats_.prediction_queries;
    std::unordered_map<std::uint64_t, double> weights;
    const std::size_t max_order = std::min(
        history.size(), config_.maximum_context_order
    );
    for (std::size_t order = max_order; order > 0U; --order) {
        ++prediction.context_comparisons;
        ++stats_.context_comparisons;
        const auto suffix = history.subspan(history.size() - order, order);
        const auto found = context_index(suffix);
        if (!found.has_value()) {
            continue;
        }
        const TemporalContext& context = contexts_[*found];
        if (context.support < config_.minimum_context_support ||
            context.outcomes.empty()) {
            continue;
        }
        if (prediction.context_order == 0U) {
            prediction.context_order = order;
            prediction.context_id = context.id;
        }
        double context_total = 0.0;
        std::vector<std::pair<std::uint64_t, double>> local;
        local.reserve(context.outcomes.size());
        for (const TemporalOutcomeCount& outcome : context.outcomes) {
            const double total = static_cast<double>(outcome.total_count);
            const double recent = effective_recent(outcome);
            const double blended =
                (1.0 - config_.recent_weight) * total +
                config_.recent_weight * recent + config_.smoothing;
            context_total += blended;
            local.emplace_back(outcome.prototype_id, blended);
        }
        const double order_factor = 0.25 + 0.75 *
            static_cast<double>(order) / static_cast<double>(max_order);
        const double support_factor = std::min(
            1.0,
            std::log1p(static_cast<double>(context.support)) / std::log(32.0)
        );
        const double context_weight = order_factor *
            std::max(0.15, support_factor);
        for (const auto& [id, value] : local) {
            weights[id] += context_weight * value / context_total;
        }
    }

    if (use_options) {
        const std::uint64_t before = stats_.option_comparisons;
        const auto option = best_option_continuation(history);
        prediction.option_comparisons = static_cast<std::size_t>(
            stats_.option_comparisons - before
        );
        if (option.has_value()) {
            const TemporalOption& selected = *option->first;
            const std::size_t prefix = option->second;
            const std::uint64_t next_id = selected.sequence[prefix];
            const double option_weight = 0.75 * selected.confidence;
            weights[next_id] += option_weight;
            prediction.used_option = true;
            prediction.option_id = selected.id;
        }
    }

    if (weights.empty()) {
        return prediction;
    }
    double total_weight = 0.0;
    for (const auto& [id, weight] : weights) {
        static_cast<void>(id);
        total_weight += weight;
    }
    prediction.outcomes.reserve(weights.size());
    for (const auto& [id, weight] : weights) {
        prediction.outcomes.push_back({id, weight / total_weight});
    }
    std::sort(
        prediction.outcomes.begin(), prediction.outcomes.end(),
        [](const TemporalPredictionOutcome& left,
           const TemporalPredictionOutcome& right) {
            if (left.probability != right.probability) {
                return left.probability > right.probability;
            }
            return left.prototype_id < right.prototype_id;
        }
    );
    if (prediction.outcomes.size() > config_.maximum_prediction_outcomes) {
        prediction.outcomes.resize(config_.maximum_prediction_outcomes);
        double retained = 0.0;
        for (const auto& outcome : prediction.outcomes) {
            retained += outcome.probability;
        }
        for (auto& outcome : prediction.outcomes) {
            outcome.probability /= retained;
        }
    }
    double entropy = 0.0;
    for (const auto& outcome : prediction.outcomes) {
        if (outcome.probability > 0.0) {
            entropy -= outcome.probability * std::log(outcome.probability);
        }
    }
    const double maximum_entropy = prediction.outcomes.size() > 1U
        ? std::log(static_cast<double>(prediction.outcomes.size()))
        : 1.0;
    prediction.uncertainty = prediction.outcomes.size() > 1U
        ? std::clamp(entropy / maximum_entropy, 0.0, 1.0)
        : 0.0;
    return prediction;
}

TemporalPrediction TemporalPredictiveFabric::predict_next(
    const std::span<const PhaseVector> history,
    const bool use_options
) const {
    std::vector<std::uint64_t> encoded;
    encoded.reserve(history.size());
    for (const PhaseVector& observation : history) {
        const auto id = match_prototype(observation);
        if (!id.has_value()) {
            return {};
        }
        encoded.push_back(*id);
    }
    return predict_next(encoded, use_options);
}

bool TemporalPredictiveFabric::update_surprise(const double surprise) {
    if (!std::isfinite(surprise)) {
        return false;
    }
    if (!surprise_initialized_) {
        slow_surprise_ = surprise;
        fast_surprise_ = surprise;
        surprise_initialized_ = true;
        return false;
    }
    slow_surprise_ =
        (1.0 - config_.surprise_slow_rate) * slow_surprise_ +
        config_.surprise_slow_rate * surprise;
    fast_surprise_ =
        (1.0 - config_.surprise_fast_rate) * fast_surprise_ +
        config_.surprise_fast_rate * surprise;
    const bool cooldown_complete = training_step_ >=
        last_change_step_ + config_.change_cooldown;
    const bool changed = cooldown_complete &&
        fast_surprise_ - slow_surprise_ > config_.change_threshold;
    if (changed) {
        last_change_step_ = training_step_;
        ++stats_.change_points_detected;
    }
    return changed;
}

TemporalStepResult TemporalPredictiveFabric::observe(
    const PhaseVector& observation
) {
    ++training_step_;
    ++stats_.observations_seen;
    TemporalStepResult result;
    const auto [id, created] = encode_or_create(observation);
    result.prototype_id = id;
    result.prototype_created = created;
    if (!recent_history_.empty()) {
        result.prediction = predict_next(recent_history_, true);
        result.prediction_available = !result.prediction.outcomes.empty();
        if (result.prediction_available) {
            const auto found = std::find_if(
                result.prediction.outcomes.begin(),
                result.prediction.outcomes.end(),
                [id](const TemporalPredictionOutcome& outcome) {
                    return outcome.prototype_id == id;
                }
            );
            result.actual_probability = found == result.prediction.outcomes.end()
                ? probability_floor
                : std::max(found->probability, probability_floor);
            result.surprise = -std::log(result.actual_probability);
            result.change_detected = update_surprise(result.surprise);
        }
        const std::size_t max_order = std::min(
            recent_history_.size(), config_.maximum_context_order
        );
        for (std::size_t order = 1U; order <= max_order; ++order) {
            const auto suffix = std::span<const std::uint64_t>(recent_history_)
                .subspan(recent_history_.size() - order, order);
            update_context(suffix, id);
        }
    }
    recent_history_.push_back(id);
    if (recent_history_.size() > config_.maximum_context_order) {
        recent_history_.erase(recent_history_.begin());
    }
    if (mining_sequences_.empty()) {
        mining_sequences_.emplace_back();
        ++stats_.sequences_seen;
    }
    mining_sequences_.back().push_back(id);
    return result;
}

void TemporalPredictiveFabric::observe_sequence(
    const std::span<const PhaseVector> sequence
) {
    reset_sequence();
    for (const PhaseVector& observation : sequence) {
        static_cast<void>(observe(observation));
    }
}

void TemporalPredictiveFabric::discover_options() {
    struct Candidate final {
        std::vector<std::uint64_t> sequence;
        std::uint64_t support{};
        std::uint64_t prefix_support{};
        double confidence{};
        double gain{};
        double compression{};
    };
    std::unordered_map<std::vector<std::uint64_t>, std::uint64_t, SequenceHash>
        counts;
    std::unordered_map<std::vector<std::uint64_t>, std::uint64_t, SequenceHash>
        prefix_counts;
    for (const auto& sequence : mining_sequences_) {
        for (std::size_t start = 0U; start < sequence.size(); ++start) {
            if (start + config_.option_prefix_minimum <= sequence.size()) {
                std::vector<std::uint64_t> prefix(
                    sequence.begin() + static_cast<std::ptrdiff_t>(start),
                    sequence.begin() + static_cast<std::ptrdiff_t>(
                        start + config_.option_prefix_minimum)
                );
                ++prefix_counts[prefix];
            }
            const std::size_t maximum_length = std::min(
                config_.maximum_option_length, sequence.size() - start
            );
            for (std::size_t length = config_.minimum_option_length;
                 length <= maximum_length; ++length) {
                std::vector<std::uint64_t> candidate(
                    sequence.begin() + static_cast<std::ptrdiff_t>(start),
                    sequence.begin() + static_cast<std::ptrdiff_t>(start + length)
                );
                ++counts[candidate];
            }
        }
    }
    stats_.option_candidates += counts.size();
    std::vector<Candidate> candidates;
    candidates.reserve(counts.size());
    for (const auto& [sequence, support] : counts) {
        if (support < config_.minimum_option_support) {
            continue;
        }
        if (sequence.size() <= config_.option_prefix_minimum) {
            continue;
        }
        std::vector<std::uint64_t> prefix(
            sequence.begin(),
            sequence.begin() + static_cast<std::ptrdiff_t>(
                config_.option_prefix_minimum)
        );
        const auto prefix_found = prefix_counts.find(prefix);
        const std::uint64_t prefix_support = prefix_found == prefix_counts.end()
            ? support
            : prefix_found->second;
        const double confidence = static_cast<double>(support) /
            static_cast<double>(std::max<std::uint64_t>(1ULL, prefix_support));
        const std::size_t continuation_length =
            sequence.size() - config_.option_prefix_minimum;
        const double uniform_probability = std::pow(
            1.0 / static_cast<double>(std::max<std::size_t>(2U, prototypes_.size())),
            static_cast<double>(continuation_length)
        );
        const double gain = std::max(0.0, confidence - uniform_probability);
        const double compression =
            static_cast<double>(support * continuation_length) -
            static_cast<double>(sequence.size());
        if (confidence >= config_.minimum_option_confidence &&
            gain >= config_.minimum_option_gain && compression > 0.0) {
            candidates.push_back({
                sequence, support, prefix_support, confidence, gain, compression
            });
        }
    }
    std::sort(
        candidates.begin(), candidates.end(),
        [](const Candidate& left, const Candidate& right) {
            if (left.compression != right.compression) {
                return left.compression > right.compression;
            }
            if (left.gain != right.gain) {
                return left.gain > right.gain;
            }
            if (left.sequence.size() != right.sequence.size()) {
                return left.sequence.size() > right.sequence.size();
            }
            return vector_less(left.sequence, right.sequence);
        }
    );
    options_.clear();
    next_option_id_ = 1ULL;
    std::unordered_set<std::vector<std::uint64_t>, SequenceHash> accepted;
    for (const Candidate& candidate : candidates) {
        if (options_.size() >= config_.maximum_options) {
            break;
        }
        if (!accepted.insert(candidate.sequence).second) {
            continue;
        }
        options_.push_back({
            next_option_id_++, candidate.sequence, candidate.support,
            candidate.confidence, candidate.gain, candidate.compression,
            training_step_, training_step_
        });
    }
    stats_.options_created += options_.size();
}

TemporalForecast TemporalPredictiveFabric::forecast(
    const std::span<const std::uint64_t> history,
    const std::size_t horizon,
    const bool use_options
) const {
    TemporalForecast result;
    std::vector<std::uint64_t> working(history.begin(), history.end());
    result.prototype_ids.reserve(horizon);
    while (result.prototype_ids.size() < horizon) {
        if (use_options) {
            const auto option = best_option_continuation(working);
            if (option.has_value()) {
                const TemporalOption& selected = *option->first;
                const std::size_t prefix = option->second;
                const std::size_t remaining = selected.sequence.size() - prefix;
                const std::size_t count = std::min(
                    remaining, horizon - result.prototype_ids.size()
                );
                for (std::size_t index = 0U; index < count; ++index) {
                    const std::uint64_t id = selected.sequence[prefix + index];
                    result.prototype_ids.push_back(id);
                    working.push_back(id);
                }
                ++result.decision_operations;
                ++result.option_uses;
                continue;
            }
        }
        const TemporalPrediction prediction = predict_next(working, false);
        if (prediction.outcomes.empty()) {
            break;
        }
        const std::uint64_t id = prediction.outcomes.front().prototype_id;
        result.prototype_ids.push_back(id);
        working.push_back(id);
        ++result.decision_operations;
    }
    result.predicted_tokens = result.prototype_ids.size();
    return result;
}

const TemporalPrototype& TemporalPredictiveFabric::prototype_by_id(
    const std::uint64_t id
) const {
    const auto index = prototype_index(id);
    if (!index.has_value()) {
        throw std::out_of_range("unknown RLF-4 prototype id");
    }
    return prototypes_[*index];
}

std::size_t TemporalPredictiveFabric::estimated_storage_bytes() const noexcept {
    std::size_t bytes = sizeof(*this);
    for (const auto& prototype : prototypes_) {
        bytes += sizeof(prototype) + prototype.key.size() * sizeof(float);
    }
    for (const auto& context : contexts_) {
        bytes += sizeof(context) + context.history.size() * sizeof(std::uint64_t) +
            context.outcomes.size() * sizeof(TemporalOutcomeCount);
    }
    for (const auto& option : options_) {
        bytes += sizeof(option) + option.sequence.size() * sizeof(std::uint64_t);
    }
    bytes += recent_history_.size() * sizeof(std::uint64_t);
    return bytes;
}

std::uint64_t TemporalPredictiveFabric::deterministic_hash() const noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, seed_);
    hash_u64(hash, training_step_);
    hash_u64(hash, prototypes_.size());
    for (const auto& prototype : prototypes_) {
        hash_u64(hash, prototype.id);
        hash_u64(hash, prototype.support);
        for (const float angle : prototype.key.angles()) {
            hash_u64(hash, std::bit_cast<std::uint32_t>(angle));
        }
    }
    hash_u64(hash, contexts_.size());
    for (const auto& context : contexts_) {
        hash_u64(hash, context.id);
        for (const auto id : context.history) {
            hash_u64(hash, id);
        }
        for (const auto& outcome : context.outcomes) {
            hash_u64(hash, outcome.prototype_id);
            hash_u64(hash, outcome.total_count);
            hash_double(hash, outcome.recent_count);
        }
    }
    hash_u64(hash, options_.size());
    for (const auto& option : options_) {
        hash_u64(hash, option.id);
        for (const auto id : option.sequence) {
            hash_u64(hash, id);
        }
        hash_u64(hash, option.support);
        hash_double(hash, option.confidence);
    }
    return hash;
}

TemporalFabricSnapshot TemporalPredictiveFabric::snapshot() const {
    return {
        .config = config_,
        .seed = seed_,
        .training_step = training_step_,
        .next_prototype_id = next_prototype_id_,
        .next_context_id = next_context_id_,
        .next_option_id = next_option_id_,
        .prototypes = prototypes_,
        .contexts = contexts_,
        .options = options_,
        .recent_history = recent_history_,
        .slow_surprise = slow_surprise_,
        .fast_surprise = fast_surprise_,
        .surprise_initialized = surprise_initialized_,
        .last_change_step = last_change_step_,
        .stats = stats_,
    };
}

TemporalPredictiveFabric TemporalPredictiveFabric::from_snapshot(
    TemporalFabricSnapshot snapshot
) {
    TemporalPredictiveFabric fabric(snapshot.config, snapshot.seed);
    fabric.training_step_ = snapshot.training_step;
    fabric.next_prototype_id_ = snapshot.next_prototype_id;
    fabric.next_context_id_ = snapshot.next_context_id;
    fabric.next_option_id_ = snapshot.next_option_id;
    fabric.prototypes_ = std::move(snapshot.prototypes);
    fabric.contexts_ = std::move(snapshot.contexts);
    fabric.options_ = std::move(snapshot.options);
    fabric.recent_history_ = std::move(snapshot.recent_history);
    fabric.slow_surprise_ = snapshot.slow_surprise;
    fabric.fast_surprise_ = snapshot.fast_surprise;
    fabric.surprise_initialized_ = snapshot.surprise_initialized;
    fabric.last_change_step_ = snapshot.last_change_step;
    fabric.stats_ = snapshot.stats;
    fabric.rebuild_indices();
    fabric.validate_snapshot();
    return fabric;
}

void TemporalPredictiveFabric::rebuild_indices() {
    prototype_index_by_id_.clear();
    for (std::size_t index = 0U; index < prototypes_.size(); ++index) {
        prototype_index_by_id_[prototypes_[index].id] = index;
    }
    context_index_by_history_.clear();
    for (std::size_t index = 0U; index < contexts_.size(); ++index) {
        context_index_by_history_[contexts_[index].history] = index;
    }
}

void TemporalPredictiveFabric::validate_snapshot() const {
    validate_config(config_);
    if (prototypes_.size() > config_.maximum_prototypes ||
        contexts_.size() > config_.maximum_contexts ||
        options_.size() > config_.maximum_options ||
        recent_history_.size() > config_.maximum_context_order) {
        throw std::runtime_error("RLF-4 snapshot exceeds configured limits");
    }
    std::unordered_set<std::uint64_t> prototype_ids;
    for (const auto& prototype : prototypes_) {
        if (prototype.id == 0ULL || prototype.key.size() != config_.dimension ||
            !prototype_ids.insert(prototype.id).second) {
            throw std::runtime_error("invalid RLF-4 prototype snapshot");
        }
    }
    std::unordered_set<std::uint64_t> context_ids;
    std::unordered_set<std::vector<std::uint64_t>, SequenceHash> context_histories;
    for (const auto& context : contexts_) {
        if (context.id == 0ULL || context.history.empty() ||
            context.history.size() > config_.maximum_context_order ||
            context.support == 0ULL || !context_ids.insert(context.id).second ||
            !context_histories.insert(context.history).second) {
            throw std::runtime_error("invalid RLF-4 context snapshot");
        }
        for (const auto id : context.history) {
            if (!prototype_ids.contains(id)) {
                throw std::runtime_error("RLF-4 context references unknown prototype");
            }
        }
        std::unordered_set<std::uint64_t> outcome_ids;
        for (const auto& outcome : context.outcomes) {
            if (!prototype_ids.contains(outcome.prototype_id) ||
                !outcome_ids.insert(outcome.prototype_id).second ||
                outcome.total_count == 0ULL ||
                !std::isfinite(outcome.recent_count) || outcome.recent_count < 0.0) {
                throw std::runtime_error("invalid RLF-4 outcome snapshot");
            }
        }
    }
    std::unordered_set<std::uint64_t> option_ids;
    std::unordered_set<std::vector<std::uint64_t>, SequenceHash> option_sequences;
    for (const auto& option : options_) {
        if (option.id == 0ULL || option.support == 0ULL ||
            option.sequence.size() < config_.minimum_option_length ||
            option.sequence.size() > config_.maximum_option_length ||
            !option_ids.insert(option.id).second ||
            !option_sequences.insert(option.sequence).second ||
            !std::isfinite(option.confidence) || option.confidence < 0.0 ||
            option.confidence > 1.0 ||
            !std::isfinite(option.predictive_gain) ||
            !std::isfinite(option.compression_gain)) {
            throw std::runtime_error("invalid RLF-4 option snapshot");
        }
        for (const auto id : option.sequence) {
            if (!prototype_ids.contains(id)) {
                throw std::runtime_error("RLF-4 option references unknown prototype");
            }
        }
    }
    for (const auto id : recent_history_) {
        if (!prototype_ids.contains(id)) {
            throw std::runtime_error("RLF-4 history references unknown prototype");
        }
    }
    const auto maximum_id = [](const auto& values) {
        std::uint64_t result = 0ULL;
        for (const auto& value : values) {
            result = std::max(result, value.id);
        }
        return result;
    };
    if (next_prototype_id_ <= maximum_id(prototypes_) ||
        next_context_id_ <= maximum_id(contexts_) ||
        next_option_id_ <= maximum_id(options_)) {
        throw std::runtime_error("RLF-4 snapshot contains invalid next ID");
    }
}

std::string_view rlf4_architecture_name() noexcept {
    return "RLF-4 self-supervised temporal predictive fabric";
}

}  // namespace rlf::core
