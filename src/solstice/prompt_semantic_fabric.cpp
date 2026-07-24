#include "rlf/solstice/prompt_semantic_fabric.hpp"

#include "rlf/core/deterministic_rng.hpp"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace rlf::solstice {
namespace {

constexpr std::uint64_t fnv_offset = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;

[[nodiscard]] std::uint64_t text_hash(const std::string_view text) noexcept {
    std::uint64_t hash = fnv_offset;
    for (const char character : text) {
        hash ^= static_cast<std::uint64_t>(
            static_cast<unsigned char>(character)
        );
        hash *= fnv_prime;
    }
    return hash;
}

void hash_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (unsigned int shift = 0U; shift < 64U; shift += 8U) {
        hash ^= (value >> shift) & 0xFFULL;
        hash *= fnv_prime;
    }
}

void validate_config(const PromptSemanticConfig& config) {
    if (config.phase_dimension == 0U || config.maximum_words == 0U ||
        config.maximum_word_bytes == 0U ||
        config.maximum_words_per_record == 0U || config.context_window == 0U ||
        config.bucket_bits == 0U || config.bucket_bits > 16U ||
        config.bucket_bits > config.phase_dimension ||
        config.maximum_expansions_per_word == 0U || config.minimum_support == 0U ||
        !std::isfinite(config.learning_rate) || config.learning_rate <= 0.0 ||
        config.learning_rate > 1.0) {
        throw std::invalid_argument("invalid prompt-semantic configuration");
    }
}

}  // namespace

PromptSemanticFabric::PromptSemanticFabric(PromptSemanticConfig config)
    : config_(std::move(config)) {
    validate_config(config_);
    modes_.reserve(std::min<std::size_t>(config_.maximum_words, 65'536U));
}

std::vector<std::string> PromptSemanticFabric::tokenize(
    const std::string_view text
) const {
    std::vector<std::string> words;
    std::string current;
    const auto flush = [&]() {
        if (!current.empty()) {
            if (words.size() >= config_.maximum_words_per_record) {
                throw std::length_error(
                    "prompt-semantic record exceeds maximum word count"
                );
            }
            words.push_back(std::move(current));
            current.clear();
        }
    };
    for (const char character : text) {
        const auto byte = static_cast<unsigned char>(character);
        if (std::isalnum(byte) != 0 || byte >= 0x80U) {
            if (current.size() < config_.maximum_word_bytes) {
                current.push_back(static_cast<char>(
                    byte < 0x80U
                        ? static_cast<unsigned char>(std::tolower(byte))
                        : byte
                ));
            }
        } else {
            flush();
        }
    }
    flush();
    return words;
}

core::PhaseVector PromptSemanticFabric::carrier(
    const std::string_view word
) const {
    core::DeterministicRng random(config_.seed ^ text_hash(word));
    return core::PhaseVector::random(config_.phase_dimension, random);
}

std::uint16_t PromptSemanticFabric::signature(
    const core::PhaseVector& value
) const {
    std::uint16_t result = 0U;
    for (std::size_t index = 0U; index < config_.bucket_bits; ++index) {
        result = static_cast<std::uint16_t>(result << 1U);
        const double cosine = std::cos(static_cast<double>(value[index]));
        result = static_cast<std::uint16_t>(
            result | static_cast<std::uint16_t>(cosine >= 0.0 ? 1U : 0U)
        );
    }
    return result;
}

void PromptSemanticFabric::train_record(const std::string_view text) {
    const auto words = tokenize(text);
    ++stats_.records_seen;
    stats_.words_seen += static_cast<std::uint64_t>(words.size());
    if (words.empty()) {
        return;
    }
    const std::size_t dimension = config_.phase_dimension;
    if (words.size() == std::numeric_limits<std::size_t>::max() ||
        words.size() + 1U >
            std::numeric_limits<std::size_t>::max() / dimension ||
        words.size() * dimension >
            std::numeric_limits<std::size_t>::max() / 2U) {
        throw std::overflow_error("prompt-semantic record workspace overflow");
    }
    std::vector<float> token_cartesian(words.size() * dimension * 2U);
    for (std::size_t token = 0U; token < words.size(); ++token) {
        const auto found = word_index_.find(words[token]);
        const std::size_t output_offset = token * dimension * 2U;
        if (found != word_index_.end()) {
            const std::size_t cache_offset = found->second * dimension * 2U;
            std::copy_n(
                carrier_cartesian_.begin() +
                    static_cast<std::ptrdiff_t>(cache_offset),
                dimension * 2U,
                token_cartesian.begin() +
                    static_cast<std::ptrdiff_t>(output_offset)
            );
            continue;
        }
        const core::PhaseVector value = carrier(words[token]);
        for (std::size_t index = 0U; index < dimension; ++index) {
            const double angle = static_cast<double>(value[index]);
            token_cartesian[output_offset + index * 2U] =
                static_cast<float>(std::cos(angle));
            token_cartesian[output_offset + index * 2U + 1U] =
                static_cast<float>(std::sin(angle));
        }
    }

    const std::size_t prefix_values = (words.size() + 1U) * dimension;
    std::vector<double> prefix_cosine(prefix_values, 0.0);
    std::vector<double> prefix_sine(prefix_values, 0.0);
    for (std::size_t token = 0U; token < words.size(); ++token) {
        for (std::size_t index = 0U; index < dimension; ++index) {
            const std::size_t previous = token * dimension + index;
            const std::size_t next = (token + 1U) * dimension + index;
            const std::size_t carrier_offset =
                token * dimension * 2U + index * 2U;
            prefix_cosine[next] = prefix_cosine[previous] +
                static_cast<double>(token_cartesian[carrier_offset]);
            prefix_sine[next] = prefix_sine[previous] +
                static_cast<double>(token_cartesian[carrier_offset + 1U]);
        }
    }
    for (std::size_t center = 0U; center < words.size(); ++center) {
        const std::size_t begin = center > config_.context_window
            ? center - config_.context_window
            : 0U;
        const std::size_t end = std::min(
            words.size(), center + config_.context_window + 1U
        );
        const std::size_t context_count = end - begin - 1U;
        if (context_count == 0U) {
            continue;
        }
        std::vector<float> desired_angles(dimension);
        for (std::size_t index = 0U; index < dimension; ++index) {
            const std::size_t begin_offset = begin * dimension + index;
            const std::size_t end_offset = end * dimension + index;
            const std::size_t center_offset =
                center * dimension * 2U + index * 2U;
            const double cosine = prefix_cosine[end_offset] -
                prefix_cosine[begin_offset] -
                static_cast<double>(token_cartesian[center_offset]);
            const double sine = prefix_sine[end_offset] -
                prefix_sine[begin_offset] -
                static_cast<double>(token_cartesian[center_offset + 1U]);
            desired_angles[index] = std::hypot(cosine, sine) <=
                    1.0e-6 * static_cast<double>(context_count)
                ? 0.0F
                : static_cast<float>(std::atan2(sine, cosine));
        }
        core::PhaseVector desired(std::move(desired_angles));
        const auto found = word_index_.find(words[center]);
        if (found == word_index_.end()) {
            if (modes_.size() >= config_.maximum_words) {
                ++stats_.capacity_skips;
                continue;
            }
            const std::size_t index = modes_.size();
            modes_.push_back({
                .id = next_mode_id_++,
                .word = words[center],
                .context_prototype = std::move(desired),
                .support = 1U,
            });
            word_index_.emplace(modes_.back().word, index);
            const std::size_t carrier_begin = center * dimension * 2U;
            carrier_cartesian_.insert(
                carrier_cartesian_.end(),
                token_cartesian.begin() +
                    static_cast<std::ptrdiff_t>(carrier_begin),
                token_cartesian.begin() + static_cast<std::ptrdiff_t>(
                    carrier_begin + dimension * 2U
                )
            );
            ++stats_.modes_created;
        } else {
            auto& mode = modes_[found->second];
            std::vector<float> updated(dimension);
            for (std::size_t index = 0U; index < dimension; ++index) {
                const double current =
                    static_cast<double>(mode.context_prototype[index]);
                const double target = static_cast<double>(desired[index]);
                const double cosine =
                    (1.0 - config_.learning_rate) * std::cos(current) +
                    config_.learning_rate * std::cos(target);
                const double sine =
                    (1.0 - config_.learning_rate) * std::sin(current) +
                    config_.learning_rate * std::sin(target);
                updated[index] = std::hypot(cosine, sine) <= 1.0e-6
                    ? mode.context_prototype[index]
                    : static_cast<float>(std::atan2(sine, cosine));
            }
            mode.context_prototype = core::PhaseVector(std::move(updated));
            if (mode.support != std::numeric_limits<std::uint64_t>::max()) {
                ++mode.support;
            }
            ++stats_.modes_updated;
        }
    }
    buckets_dirty_ = true;
}

void PromptSemanticFabric::ensure_buckets() const {
    if (!buckets_dirty_) {
        return;
    }
    buckets_.clear();
    for (std::size_t index = 0U; index < modes_.size(); ++index) {
        if (modes_[index].support >= config_.minimum_support) {
            buckets_[signature(modes_[index].context_prototype)].push_back(index);
        }
    }
    buckets_dirty_ = false;
}

std::vector<std::string> PromptSemanticFabric::similar_words(
    const std::string_view word,
    const std::size_t maximum_results
) const {
    if (maximum_results == 0U) {
        return {};
    }
    const auto found = word_index_.find(std::string(word));
    if (found == word_index_.end() ||
        modes_[found->second].support < config_.minimum_support) {
        return {};
    }
    ensure_buckets();
    ++stats_.semantic_queries;
    const auto bucket = buckets_.find(signature(
        modes_[found->second].context_prototype
    ));
    if (bucket == buckets_.end()) {
        return {};
    }
    std::vector<std::pair<double, std::size_t>> scored;
    scored.reserve(bucket->second.size());
    for (const auto index : bucket->second) {
        if (index == found->second) {
            continue;
        }
        ++stats_.bucket_candidates_scored;
        scored.emplace_back(
            modes_[found->second].context_prototype.similarity(
                modes_[index].context_prototype
            ),
            index
        );
    }
    std::sort(scored.begin(), scored.end(), [&](const auto& left, const auto& right) {
        if (left.first != right.first) return left.first > right.first;
        return modes_[left.second].id < modes_[right.second].id;
    });
    if (scored.size() > maximum_results) {
        scored.resize(maximum_results);
    }
    std::vector<std::string> result;
    result.reserve(scored.size());
    for (const auto& [score, index] : scored) {
        static_cast<void>(score);
        result.push_back(modes_[index].word);
    }
    return result;
}

std::vector<std::uint64_t> PromptSemanticFabric::semantic_concept_hashes(
    const std::string_view prompt,
    const std::size_t maximum_concepts
) const {
    std::vector<std::uint64_t> result;
    if (maximum_concepts == 0U) return result;
    const auto words = tokenize(prompt);
    for (const auto& word : words) {
        if (result.size() >= maximum_concepts) break;
        const auto aliases = similar_words(
            word, config_.maximum_expansions_per_word
        );
        for (const auto& alias : aliases) {
            result.push_back(text_hash("word:" + alias));
            if (result.size() >= maximum_concepts) break;
        }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    if (result.size() > maximum_concepts) result.resize(maximum_concepts);
    return result;
}

const PromptSemanticConfig& PromptSemanticFabric::config() const noexcept {
    return config_;
}

std::span<const PromptSemanticMode> PromptSemanticFabric::modes() const noexcept {
    return modes_;
}

PromptSemanticStats PromptSemanticFabric::stats() const noexcept {
    return stats_;
}

std::size_t PromptSemanticFabric::estimated_bytes() const noexcept {
    std::size_t bytes = sizeof(*this);
    bytes += carrier_cartesian_.capacity() * sizeof(float);
    for (const auto& mode : modes_) {
        bytes += sizeof(mode) + mode.word.capacity() +
            mode.context_prototype.size() * sizeof(float);
    }
    for (const auto& [key, values] : buckets_) {
        static_cast<void>(key);
        bytes += values.capacity() * sizeof(std::size_t);
    }
    return bytes;
}

std::uint64_t PromptSemanticFabric::deterministic_hash() const noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, static_cast<std::uint64_t>(config_.phase_dimension));
    hash_u64(hash, static_cast<std::uint64_t>(config_.maximum_words));
    hash_u64(hash, static_cast<std::uint64_t>(config_.maximum_word_bytes));
    hash_u64(hash, static_cast<std::uint64_t>(config_.maximum_words_per_record));
    hash_u64(hash, static_cast<std::uint64_t>(config_.context_window));
    hash_u64(hash, static_cast<std::uint64_t>(config_.bucket_bits));
    hash_u64(
        hash,
        static_cast<std::uint64_t>(config_.maximum_expansions_per_word)
    );
    hash_u64(hash, config_.minimum_support);
    hash_u64(hash, std::bit_cast<std::uint64_t>(config_.learning_rate));
    hash_u64(hash, config_.seed);
    hash_u64(hash, next_mode_id_);
    hash_u64(hash, stats_.records_seen);
    hash_u64(hash, stats_.words_seen);
    for (const auto& mode : modes_) {
        hash_u64(hash, mode.id);
        hash_u64(hash, text_hash(mode.word));
        hash_u64(hash, mode.support);
        for (const auto angle : mode.context_prototype.angles()) {
            hash_u64(hash, static_cast<std::uint64_t>(
                std::bit_cast<std::uint32_t>(angle)
            ));
        }
    }
    return hash;
}

PromptSemanticSnapshot PromptSemanticFabric::snapshot() const {
    return {config_, next_mode_id_, modes_, stats_};
}

void PromptSemanticFabric::rebuild_indices() {
    word_index_.clear();
    carrier_cartesian_.clear();
    if (modes_.size() > std::numeric_limits<std::size_t>::max() /
            config_.phase_dimension / 2U) {
        throw std::invalid_argument(
            "prompt-semantic carrier cache size overflow"
        );
    }
    carrier_cartesian_.reserve(
        modes_.size() * config_.phase_dimension * 2U
    );
    for (std::size_t index = 0U; index < modes_.size(); ++index) {
        if (!word_index_.emplace(modes_[index].word, index).second) {
            throw std::invalid_argument("duplicate prompt-semantic word");
        }
        const core::PhaseVector value = carrier(modes_[index].word);
        for (const float angle_value : value.angles()) {
            const double angle = static_cast<double>(angle_value);
            carrier_cartesian_.push_back(static_cast<float>(std::cos(angle)));
            carrier_cartesian_.push_back(static_cast<float>(std::sin(angle)));
        }
    }
    buckets_dirty_ = true;
    ensure_buckets();
}

PromptSemanticFabric PromptSemanticFabric::from_snapshot(
    PromptSemanticSnapshot snapshot
) {
    validate_config(snapshot.config);
    if (snapshot.modes.size() > snapshot.config.maximum_words ||
        snapshot.next_mode_id == 0U) {
        throw std::invalid_argument("invalid prompt-semantic snapshot bounds");
    }
    std::unordered_set<std::uint64_t> ids;
    for (const auto& mode : snapshot.modes) {
        if (mode.id == 0U || mode.id >= snapshot.next_mode_id ||
            mode.word.empty() ||
            mode.word.size() > snapshot.config.maximum_word_bytes ||
            mode.context_prototype.size() != snapshot.config.phase_dimension ||
            mode.support == 0U || !ids.insert(mode.id).second) {
            throw std::invalid_argument("invalid prompt-semantic mode");
        }
    }
    PromptSemanticFabric fabric(snapshot.config);
    fabric.next_mode_id_ = snapshot.next_mode_id;
    fabric.modes_ = std::move(snapshot.modes);
    fabric.stats_ = snapshot.stats;
    fabric.rebuild_indices();
    return fabric;
}

}  // namespace rlf::solstice
