#include "rlf/solstice/language_fabric.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace rlf::solstice {
namespace {

constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;
constexpr std::size_t outcome_index_threshold = 32U;

[[nodiscard]] bool indexed_outcome_updates_from_environment() {
    const char* const value = std::getenv("RLF_LANGUAGE_OUTCOME_POLICY");
    if (value == nullptr || std::string_view(value).empty() ||
        std::string_view(value) == "indexed") {
        return true;
    }
    if (std::string_view(value) == "linear") {
        return false;
    }
    throw std::invalid_argument(
        "RLF_LANGUAGE_OUTCOME_POLICY must be indexed or linear"
    );
}

[[nodiscard]] bool fused_dialogue_encoding_from_environment() {
    const char* const value = std::getenv("RLF_LANGUAGE_DIALOGUE_ENCODING_POLICY");
    if (value == nullptr || std::string_view(value).empty() ||
        std::string_view(value) == "fused") {
        return true;
    }
    if (std::string_view(value) == "redundant") {
        return false;
    }
    throw std::invalid_argument(
        "RLF_LANGUAGE_DIALOGUE_ENCODING_POLICY must be fused or redundant"
    );
}

void hash_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (unsigned int index = 0U; index < 8U; ++index) {
        hash ^= (value >> (index * 8U)) & 0xFFULL;
        hash *= fnv_prime;
    }
}

void hash_double(std::uint64_t& hash, const double value) noexcept {
    hash_u64(hash, std::bit_cast<std::uint64_t>(value));
}

[[nodiscard]] double clamp_probability(const double value) noexcept {
    return std::clamp(value, 0.0, 1.0);
}

[[nodiscard]] std::uint64_t next_random(std::uint64_t& state) noexcept {
    state ^= state >> 12U;
    state ^= state << 25U;
    state ^= state >> 27U;
    return state * 2'685'821'657'736'338'717ULL;
}

[[nodiscard]] double random_unit(std::uint64_t& state) noexcept {
    constexpr double denominator =
        static_cast<double>(std::numeric_limits<std::uint64_t>::max());
    return static_cast<double>(next_random(state)) / denominator;
}

[[nodiscard]] std::vector<TokenId> unique_content_tokens(
    const std::span<const TokenId> tokens
) {
    std::vector<TokenId> unique;
    unique.reserve(tokens.size());
    for (const TokenId token : tokens) {
        if (token < 256U) {
            const unsigned char byte = static_cast<unsigned char>(token);
            if (byte <= 32U) {
                continue;
            }
        }
        unique.push_back(token);
    }
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    return unique;
}

[[nodiscard]] double jaccard_similarity(
    const std::span<const TokenId> left,
    const std::span<const TokenId> right
) {
    const std::vector<TokenId> left_unique = unique_content_tokens(left);
    const std::vector<TokenId> right_unique = unique_content_tokens(right);
    if (left_unique.empty() || right_unique.empty()) {
        return 0.0;
    }
    std::size_t left_index = 0U;
    std::size_t right_index = 0U;
    std::size_t intersection = 0U;
    std::size_t union_size = 0U;
    while (left_index < left_unique.size() || right_index < right_unique.size()) {
        if (right_index >= right_unique.size() ||
            (left_index < left_unique.size() &&
             left_unique[left_index] < right_unique[right_index])) {
            ++left_index;
            ++union_size;
        } else if (left_index >= left_unique.size() ||
                   right_unique[right_index] < left_unique[left_index]) {
            ++right_index;
            ++union_size;
        } else {
            ++left_index;
            ++right_index;
            ++intersection;
            ++union_size;
        }
    }
    return union_size == 0U
        ? 0.0
        : static_cast<double>(intersection) / static_cast<double>(union_size);
}

[[nodiscard]] bool valid_configuration(const HierarchicalLanguageConfig& config) {
    if (config.context_orders.empty() || config.context_orders.front() != 0U ||
        !std::is_sorted(config.context_orders.begin(), config.context_orders.end()) ||
        std::adjacent_find(config.context_orders.begin(), config.context_orders.end()) !=
            config.context_orders.end() ||
        config.maximum_contexts == 0U || config.maximum_episodes == 0U ||
        config.maximum_episode_cue_tokens == 0U ||
        config.maximum_episode_response_tokens == 0U ||
        config.maximum_generation_tokens == 0U ||
        config.prediction_candidate_limit == 0U ||
        !std::isfinite(config.smoothing) || config.smoothing < 0.0 ||
        !std::isfinite(config.long_context_weight) ||
        config.long_context_weight < 0.0 ||
        !std::isfinite(config.episode_conditioning_weight) ||
        config.episode_conditioning_weight < 0.0 ||
        !std::isfinite(config.repetition_penalty) ||
        config.repetition_penalty < 1.0) {
        return false;
    }
    return true;
}

}  // namespace

HierarchicalLanguageFabric::HierarchicalLanguageFabric(
    HierarchicalLanguageConfig config
) : config_(std::move(config)),
    indexed_outcome_updates_(indexed_outcome_updates_from_environment()),
    fused_dialogue_encoding_(fused_dialogue_encoding_from_environment()),
    bounded_capacity_replacement_(
        config_.maximum_contexts >= 1'000'000'000U ||
        config_.maximum_episodes >= 250'000'000U
    ) {
    if (!valid_configuration(config_)) {
        throw std::invalid_argument("invalid Solstice hierarchical language configuration");
    }
    contexts_.reserve(std::min<std::size_t>(config_.maximum_contexts, 65'536U));
    episodes_.reserve(std::min<std::size_t>(config_.maximum_episodes, 8'192U));
}

std::size_t HierarchicalLanguageFabric::ContextKeyHash::operator()(
    const ContextKey& key
) const noexcept {
    std::uint64_t hash = fnv_offset_basis;
    for (const TokenId token : key.history) {
        hash_u64(hash, token);
    }
    hash_u64(hash, key.history.size());
    return static_cast<std::size_t>(hash);
}

void HierarchicalLanguageFabric::train_corpus(
    const SolsticeTokenizer& tokenizer,
    const std::string_view corpus
) {
    const TokenId bos = tokenizer.special_id("<bos>");
    const TokenId eos = tokenizer.special_id("<eos>");
    std::size_t line_start = 0U;
    while (line_start <= corpus.size()) {
        const std::size_t newline = corpus.find('\n', line_start);
        const std::size_t line_end = newline == std::string_view::npos
            ? corpus.size()
            : newline;
        std::string_view line = corpus.substr(line_start, line_end - line_start);
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1U);
        }
        if (!line.empty()) {
            std::vector<TokenId> sequence;
            const std::vector<TokenId> encoded = tokenizer.encode(line);
            sequence.reserve(encoded.size() + 2U);
            sequence.push_back(bos);
            sequence.insert(sequence.end(), encoded.begin(), encoded.end());
            sequence.push_back(eos);
            train_token_sequence(sequence);
        }
        if (newline == std::string_view::npos) {
            break;
        }
        line_start = newline + 1U;
    }
}

std::vector<TokenId> HierarchicalLanguageFabric::dialogue_prefix(
    const SolsticeTokenizer& tokenizer,
    const std::string_view prompt,
    const std::string_view grounding
) const {
    std::vector<TokenId> prefix;
    prefix.push_back(tokenizer.special_id("<bos>"));
    if (!grounding.empty()) {
        prefix.push_back(tokenizer.special_id("<image>"));
        const std::vector<TokenId> grounding_tokens = tokenizer.encode(grounding);
        prefix.insert(prefix.end(), grounding_tokens.begin(), grounding_tokens.end());
        prefix.push_back(tokenizer.special_id("<image_end>"));
    }
    prefix.push_back(tokenizer.special_id("<user>"));
    const std::vector<TokenId> prompt_tokens = tokenizer.encode(prompt);
    prefix.insert(prefix.end(), prompt_tokens.begin(), prompt_tokens.end());
    prefix.push_back(tokenizer.special_id("<assistant>"));
    return prefix;
}

void HierarchicalLanguageFabric::train_dialogue(
    const SolsticeTokenizer& tokenizer,
    const std::string_view prompt,
    const std::string_view response,
    const std::string_view grounding
) {
    ++training_operation_stats_.dialogue_training_calls;
    std::vector<TokenId> prompt_tokens;
    std::vector<TokenId> grounding_tokens;
    std::vector<TokenId> prefix;
    if (fused_dialogue_encoding_) {
        prompt_tokens = tokenizer.encode(prompt);
        ++training_operation_stats_.dialogue_tokenizer_encode_calls;
        if (!grounding.empty()) {
            grounding_tokens = tokenizer.encode(grounding);
            ++training_operation_stats_.dialogue_tokenizer_encode_calls;
        }
        prefix.reserve(
            prompt_tokens.size() + grounding_tokens.size() + 5U
        );
        prefix.push_back(tokenizer.special_id("<bos>"));
        if (!grounding.empty()) {
            prefix.push_back(tokenizer.special_id("<image>"));
            prefix.insert(
                prefix.end(), grounding_tokens.begin(), grounding_tokens.end()
            );
            prefix.push_back(tokenizer.special_id("<image_end>"));
        }
        prefix.push_back(tokenizer.special_id("<user>"));
        prefix.insert(prefix.end(), prompt_tokens.begin(), prompt_tokens.end());
        prefix.push_back(tokenizer.special_id("<assistant>"));
        training_operation_stats_.redundant_dialogue_encode_calls_avoided += 2U;
    } else {
        prefix = dialogue_prefix(tokenizer, prompt, grounding);
        training_operation_stats_.dialogue_tokenizer_encode_calls +=
            grounding.empty() ? 1U : 2U;
    }
    std::vector<TokenId> response_tokens = tokenizer.encode(response);
    ++training_operation_stats_.dialogue_tokenizer_encode_calls;
    if (response_tokens.size() > config_.maximum_episode_response_tokens) {
        response_tokens.resize(config_.maximum_episode_response_tokens);
    }
    std::vector<TokenId> sequence = prefix;
    sequence.insert(sequence.end(), response_tokens.begin(), response_tokens.end());
    sequence.push_back(tokenizer.special_id("<eos>"));
    train_token_sequence(sequence);

    std::vector<TokenId> cue;
    if (!fused_dialogue_encoding_) {
        prompt_tokens = tokenizer.encode(prompt);
        grounding_tokens = tokenizer.encode(grounding);
        training_operation_stats_.dialogue_tokenizer_encode_calls += 2U;
    }
    cue.reserve(prompt_tokens.size() + grounding_tokens.size());
    cue.insert(cue.end(), grounding_tokens.begin(), grounding_tokens.end());
    cue.insert(cue.end(), prompt_tokens.begin(), prompt_tokens.end());
    if (cue.size() > config_.maximum_episode_cue_tokens) {
        cue.erase(cue.begin(), cue.end() -
            static_cast<std::ptrdiff_t>(config_.maximum_episode_cue_tokens));
    }

    LanguageEpisode* duplicate = nullptr;
    const std::vector<TokenId> unique_cue = unique_content_tokens(cue);
    if (!unique_cue.empty()) {
        const auto posting = episode_postings_.find(unique_cue.front());
        if (posting != episode_postings_.end()) {
            for (const std::size_t index : posting->second) {
                LanguageEpisode& candidate = episodes_[index];
                if (candidate.cue == cue && candidate.response == response_tokens) {
                    duplicate = &candidate;
                    break;
                }
            }
        }
    } else {
        const auto found = std::find_if(
            episodes_.begin(), episodes_.end(),
            [&cue, &response_tokens](const LanguageEpisode& episode) {
                return episode.cue == cue && episode.response == response_tokens;
            }
        );
        if (found != episodes_.end()) {
            duplicate = &*found;
        }
    }
    if (duplicate != nullptr) {
        ++duplicate->support;
        ++training_operation_stats_.episode_duplicate_updates;
    } else {
        ++training_operation_stats_.episode_insert_attempts;
        if (episodes_.size() >= config_.maximum_episodes) {
            if (!bounded_capacity_replacement_) {
                ++training_operation_stats_.episode_capacity_skips;
                return;
            }
            const std::size_t replacement = episode_replacement_index(
                cue, response_tokens
            );
            unindex_episode(replacement);
            episodes_[replacement] = LanguageEpisode{
                next_episode_id_++,
                std::move(cue),
                std::move(response_tokens),
                1U,
            };
            index_episode(replacement);
            ++training_operation_stats_.episode_replacements;
            return;
        }
        episodes_.push_back(LanguageEpisode{
            next_episode_id_++,
            std::move(cue),
            std::move(response_tokens),
            1U,
        });
        index_episode(episodes_.size() - 1U);
        ++training_operation_stats_.episode_inserts;
    }
}

void HierarchicalLanguageFabric::train_token_sequence(
    const std::span<const TokenId> sequence
) {
    if (sequence.size() < 2U) {
        return;
    }
    for (std::size_t position = 1U; position < sequence.size(); ++position) {
        const TokenId target = sequence[position];
        for (const std::size_t order : config_.context_orders) {
            if (order > position) {
                break;
            }
            ContextKey key;
            if (order != 0U) {
                const auto begin = sequence.begin() +
                    static_cast<std::ptrdiff_t>(position - order);
                const auto end = sequence.begin() +
                    static_cast<std::ptrdiff_t>(position);
                key.history.assign(begin, end);
            }
            auto found = context_index_.find(key);
            if (found == context_index_.end()) {
                ++training_operation_stats_.context_insert_attempts;
                if (contexts_.size() >= config_.maximum_contexts) {
                    if (!bounded_capacity_replacement_) {
                        ++training_operation_stats_.context_capacity_skips;
                        continue;
                    }
                    const std::size_t replacement =
                        context_replacement_index(key);
                    ContextKey replaced_key{contexts_[replacement].history};
                    if (context_index_.erase(replaced_key) != 1U) {
                        throw std::logic_error(
                            "failed to remove replaced language context"
                        );
                    }
                    outcome_indices_.erase(replacement);
                    contexts_[replacement] = PredictiveContext{
                        next_context_id_++, key.history, 0U, {},
                    };
                    const auto inserted =
                        context_index_.emplace(std::move(key), replacement);
                    if (!inserted.second) {
                        throw std::logic_error(
                            "failed to index replacement language context"
                        );
                    }
                    ++training_operation_stats_.context_replacements;
                    found = inserted.first;
                } else {
                    const std::size_t index = contexts_.size();
                    contexts_.push_back(PredictiveContext{
                        next_context_id_++, key.history, 0U, {},
                    });
                    const auto inserted =
                        context_index_.emplace(std::move(key), index);
                    if (!inserted.second) {
                        throw std::logic_error(
                            "failed to index Solstice predictive context"
                        );
                    }
                    ++training_operation_stats_.context_inserts;
                    found = inserted.first;
                }
            }
            PredictiveContext& context = contexts_[found->second];
            ++context.support;
            ++training_operation_stats_.outcome_update_lookups;
            std::size_t outcome_position = context.outcomes.size();
            auto indexed_context = outcome_indices_.find(found->second);
            if (indexed_outcome_updates_ && indexed_context != outcome_indices_.end()) {
                ++training_operation_stats_.indexed_outcome_lookups;
                const auto indexed_outcome = std::lower_bound(
                    indexed_context->second.begin(), indexed_context->second.end(),
                    target,
                    [](const auto& value, const TokenId token) {
                        return value.first < token;
                    }
                );
                if (indexed_outcome != indexed_context->second.end() &&
                    indexed_outcome->first == target) {
                    outcome_position = indexed_outcome->second;
                }
            } else {
                for (std::size_t index = 0U; index < context.outcomes.size(); ++index) {
                    ++training_operation_stats_.linear_outcome_comparisons;
                    if (context.outcomes[index].token == target) {
                        outcome_position = index;
                        break;
                    }
                }
            }
            if (outcome_position == context.outcomes.size()) {
                context.outcomes.push_back(TokenOutcome{target, 1U});
                outcome_position = context.outcomes.size() - 1U;
                if (indexed_outcome_updates_) {
                    if (indexed_context != outcome_indices_.end()) {
                        const auto insertion = std::lower_bound(
                            indexed_context->second.begin(),
                            indexed_context->second.end(),
                            target,
                            [](const auto& value, const TokenId token) {
                                return value.first < token;
                            }
                        );
                        indexed_context->second.insert(
                            insertion, {target, outcome_position}
                        );
                        ++training_operation_stats_.outcome_index_incremental_inserts;
                    } else if (context.outcomes.size() == outcome_index_threshold) {
                        auto& index = outcome_indices_[found->second];
                        index.reserve(context.outcomes.size());
                        for (std::size_t position_index = 0U;
                             position_index < context.outcomes.size();
                             ++position_index) {
                            index.emplace_back(
                                context.outcomes[position_index].token,
                                position_index
                            );
                        }
                        std::sort(index.begin(), index.end());
                        ++training_operation_stats_.outcome_index_builds;
                        training_operation_stats_.outcome_index_entries_built +=
                            index.size();
                    }
                }
            } else {
                ++context.outcomes[outcome_position].count;
            }
        }
        ++tokens_seen_;
    }
}

HierarchicalPrediction HierarchicalLanguageFabric::predict_next(
    const std::span<const TokenId> history
) const {
    const std::vector<EpisodeMatch> no_matches;
    return predict_conditioned(history, no_matches, 0U);
}

std::vector<HierarchicalLanguageFabric::EpisodeMatch>
HierarchicalLanguageFabric::match_episodes(
    const std::span<const TokenId> cue,
    const std::size_t limit
) const {
    std::vector<EpisodeMatch> matches;
    if (limit == 0U || episodes_.empty()) {
        return matches;
    }

    const std::vector<TokenId> cue_tokens = unique_content_tokens(cue);
    if (cue_tokens.empty()) {
        return matches;
    }

    std::unordered_map<std::size_t, std::size_t> overlap_counts;
    const std::size_t candidate_budget = std::max<std::size_t>(limit * 4'096U, 8'192U);
    for (const TokenId token : cue_tokens) {
        const auto posting = episode_postings_.find(token);
        if (posting == episode_postings_.end()) {
            continue;
        }
        for (const std::size_t index : posting->second) {
            ++overlap_counts[index];
        }
    }

    std::vector<std::pair<std::size_t, std::size_t>> candidates;
    candidates.reserve(overlap_counts.size());
    for (const auto& [index, overlap] : overlap_counts) {
        candidates.emplace_back(index, overlap);
    }
    std::sort(
        candidates.begin(), candidates.end(),
        [](const auto& left, const auto& right) {
            if (left.second != right.second) {
                return left.second > right.second;
            }
            return left.first < right.first;
        }
    );
    if (candidates.size() > candidate_budget) {
        candidates.resize(candidate_budget);
    }

    const double document_count = static_cast<double>(episodes_.size() + 1U);
    for (const auto& [index, overlap] : candidates) {
        static_cast<void>(overlap);
        const LanguageEpisode& episode = episodes_[index];
        const std::vector<TokenId> episode_tokens = unique_content_tokens(episode.cue);
        std::size_t left = 0U;
        std::size_t right = 0U;
        double intersection = 0.0;
        double union_weight = 0.0;
        while (left < cue_tokens.size() || right < episode_tokens.size()) {
            TokenId token{};
            bool in_left = false;
            bool in_right = false;
            if (right >= episode_tokens.size() ||
                (left < cue_tokens.size() && cue_tokens[left] < episode_tokens[right])) {
                token = cue_tokens[left++];
                in_left = true;
            } else if (left >= cue_tokens.size() ||
                       episode_tokens[right] < cue_tokens[left]) {
                token = episode_tokens[right++];
                in_right = true;
            } else {
                token = cue_tokens[left++];
                ++right;
                in_left = true;
                in_right = true;
            }
            const auto frequency = episode_document_frequency_.find(token);
            const double df = frequency == episode_document_frequency_.end()
                ? 0.0
                : static_cast<double>(frequency->second);
            const double idf = std::log((document_count + 1.0) / (df + 1.0)) + 1.0;
            union_weight += idf;
            if (in_left && in_right) {
                intersection += idf;
            }
        }
        const double similarity = union_weight > 0.0
            ? intersection / union_weight
            : jaccard_similarity(cue, episode.cue);
        if (similarity > 0.0) {
            matches.push_back(EpisodeMatch{&episode, similarity});
        }
    }

    std::sort(
        matches.begin(), matches.end(),
        [](const EpisodeMatch& left, const EpisodeMatch& right) {
            if (left.similarity != right.similarity) {
                return left.similarity > right.similarity;
            }
            return left.episode->id < right.episode->id;
        }
    );
    if (matches.size() > limit) {
        matches.resize(limit);
    }
    return matches;
}

HierarchicalPrediction HierarchicalLanguageFabric::predict_conditioned(
    const std::span<const TokenId> history,
    const std::span<const EpisodeMatch> matches,
    const std::size_t response_position
) const {
    std::unordered_map<TokenId, double> scores;
    std::size_t deepest_order = 0U;
    for (const std::size_t order : config_.context_orders) {
        if (order > history.size()) {
            break;
        }
        ContextKey key;
        if (order != 0U) {
            key.history.assign(
                history.end() - static_cast<std::ptrdiff_t>(order),
                history.end()
            );
        }
        const auto found = context_index_.find(key);
        if (found == context_index_.end()) {
            continue;
        }
        const PredictiveContext& context = contexts_[found->second];
        deepest_order = std::max(deepest_order, order);
        const double order_weight = 1.0 +
            config_.long_context_weight * std::log2(static_cast<double>(order) + 1.0);
        const double support_weight = std::log1p(static_cast<double>(context.support));
        const double denominator = static_cast<double>(context.support) +
            config_.smoothing * static_cast<double>(context.outcomes.size());
        for (const TokenOutcome& outcome : context.outcomes) {
            const double probability = denominator <= 0.0
                ? 0.0
                : (static_cast<double>(outcome.count) + config_.smoothing) /
                    denominator;
            scores[outcome.token] += order_weight * support_weight * probability;
        }
    }

    for (const EpisodeMatch& match : matches) {
        if (response_position >= match.episode->response.size()) {
            continue;
        }
        const double support = std::log1p(static_cast<double>(match.episode->support));
        scores[match.episode->response[response_position]] +=
            config_.episode_conditioning_weight * match.similarity * support;
    }

    HierarchicalPrediction prediction;
    prediction.deepest_context_order = deepest_order;
    if (scores.empty()) {
        return prediction;
    }
    double total = 0.0;
    for (const auto& [token, score] : scores) {
        static_cast<void>(token);
        total += std::max(0.0, score);
    }
    prediction.candidates.reserve(scores.size());
    for (const auto& [token, score] : scores) {
        prediction.candidates.push_back(TokenCandidate{
            token,
            total > 0.0 ? std::max(0.0, score) / total : 0.0,
            score,
        });
    }
    std::sort(
        prediction.candidates.begin(), prediction.candidates.end(),
        [](const TokenCandidate& left, const TokenCandidate& right) {
            if (left.score != right.score) {
                return left.score > right.score;
            }
            return left.token < right.token;
        }
    );
    if (prediction.candidates.size() > config_.prediction_candidate_limit) {
        prediction.candidates.resize(config_.prediction_candidate_limit);
    }
    prediction.uncertainty = prediction.candidates.empty()
        ? 1.0
        : clamp_probability(1.0 - prediction.candidates.front().probability);
    return prediction;
}

LanguageResponse HierarchicalLanguageFabric::generate_response(
    const SolsticeTokenizer& tokenizer,
    const std::string_view prompt,
    const std::string_view grounding,
    GenerationSettings settings
) const {
    settings.maximum_tokens = std::min(
        settings.maximum_tokens,
        config_.maximum_generation_tokens
    );
    if (settings.maximum_tokens == 0U || settings.top_k == 0U ||
        !std::isfinite(settings.temperature) || settings.temperature <= 0.0) {
        throw std::invalid_argument("invalid Solstice generation settings");
    }
    std::vector<TokenId> history = dialogue_prefix(tokenizer, prompt, grounding);
    std::vector<TokenId> cue = tokenizer.encode(grounding);
    const std::vector<TokenId> prompt_tokens = tokenizer.encode(prompt);
    cue.insert(cue.end(), prompt_tokens.begin(), prompt_tokens.end());
    const std::vector<EpisodeMatch> matches = match_episodes(cue, 8U);

    LanguageResponse response;
    response.episode_similarity = matches.empty() ? 0.0 : matches.front().similarity;
    
    // Direct Episode Attractor Routing: if a high-confidence match exists (similarity >= 0.55),
    // output the exact matched episode response directly, avoiding auto-regressive n-gram fallback blending.
    if (!matches.empty() && matches.front().similarity >= 0.55 && !matches.front().episode->response.empty()) {
        response.generated_tokens = matches.front().episode->response;
        response.text = tokenizer.decode(response.generated_tokens);
        response.uncertainty = clamp_probability(1.0 - matches.front().similarity);
        return response;
    }

    std::uint64_t random_state = settings.seed == 0U
        ? 0x9E37'79B9'7F4A'7C15ULL
        : settings.seed;
    const TokenId eos = tokenizer.special_id("<eos>");
    double uncertainty_sum = 0.0;


    for (std::size_t position = 0U; position < settings.maximum_tokens; ++position) {
        HierarchicalPrediction prediction = predict_conditioned(
            history, matches, position
        );
        if (prediction.candidates.empty()) {
            break;
        }
        uncertainty_sum += prediction.uncertainty;

        std::vector<TokenCandidate> candidates;
        candidates.reserve(std::min(settings.top_k, prediction.candidates.size()));
        for (const TokenCandidate& candidate : prediction.candidates) {
            if (candidate.token != eos && tokenizer.is_special(candidate.token)) {
                continue;
            }
            double score = std::max(candidate.score, 1.0e-12);
            const std::size_t recent_count = static_cast<std::size_t>(std::count(
                history.end() - static_cast<std::ptrdiff_t>(
                    std::min<std::size_t>(history.size(), 32U)
                ),
                history.end(),
                candidate.token
            ));
            if (recent_count != 0U) {
                score /= std::pow(
                    config_.repetition_penalty,
                    static_cast<double>(recent_count)
                );
            }
            candidates.push_back(TokenCandidate{candidate.token, 0.0, score});
            if (candidates.size() >= settings.top_k) {
                break;
            }
        }
        if (candidates.empty()) {
            break;
        }

        TokenId selected = candidates.front().token;
        if (!settings.deterministic && candidates.size() > 1U) {
            double total = 0.0;
            for (TokenCandidate& candidate : candidates) {
                candidate.probability = std::pow(
                    std::max(candidate.score, 1.0e-12),
                    1.0 / settings.temperature
                );
                total += candidate.probability;
            }
            double draw = random_unit(random_state) * total;
            for (const TokenCandidate& candidate : candidates) {
                if (draw <= candidate.probability) {
                    selected = candidate.token;
                    break;
                }
                draw -= candidate.probability;
            }
        }
        if (selected == eos) {
            break;
        }
        response.generated_tokens.push_back(selected);
        history.push_back(selected);
    }
    response.text = tokenizer.decode(response.generated_tokens);
    response.uncertainty = response.generated_tokens.empty()
        ? 1.0
        : clamp_probability(
            uncertainty_sum / static_cast<double>(response.generated_tokens.size())
        );
    return response;
}

std::string HierarchicalLanguageFabric::generate_continuation(
    const SolsticeTokenizer& tokenizer,
    const std::string_view prompt,
    GenerationSettings settings
) const {
    settings.maximum_tokens = std::min(
        settings.maximum_tokens,
        config_.maximum_generation_tokens
    );
    std::vector<TokenId> history = tokenizer.encode(prompt);
    std::vector<TokenId> output;
    const TokenId eos = tokenizer.special_id("<eos>");
    std::uint64_t random_state = settings.seed;
    for (std::size_t index = 0U; index < settings.maximum_tokens; ++index) {
        const HierarchicalPrediction prediction = predict_next(history);
        if (prediction.candidates.empty()) {
            break;
        }
        const std::size_t limit = std::min(settings.top_k, prediction.candidates.size());
        TokenId selected = prediction.candidates.front().token;
        if (!settings.deterministic && limit > 1U) {
            double total = 0.0;
            std::vector<double> weights(limit, 0.0);
            for (std::size_t candidate_index = 0U;
                 candidate_index < limit; ++candidate_index) {
                weights[candidate_index] = std::pow(
                    std::max(prediction.candidates[candidate_index].score, 1.0e-12),
                    1.0 / settings.temperature
                );
                total += weights[candidate_index];
            }
            double draw = random_unit(random_state) * total;
            for (std::size_t candidate_index = 0U;
                 candidate_index < limit; ++candidate_index) {
                if (draw <= weights[candidate_index]) {
                    selected = prediction.candidates[candidate_index].token;
                    break;
                }
                draw -= weights[candidate_index];
            }
        }
        if (selected == eos) {
            break;
        }
        if (!tokenizer.is_special(selected)) {
            output.push_back(selected);
        }
        history.push_back(selected);
    }
    return tokenizer.decode(output);
}

std::span<const PredictiveContext> HierarchicalLanguageFabric::contexts() const noexcept {
    return contexts_;
}

std::span<const LanguageEpisode> HierarchicalLanguageFabric::episodes() const noexcept {
    return episodes_;
}

const HierarchicalLanguageConfig& HierarchicalLanguageFabric::config() const noexcept {
    return config_;
}

std::uint64_t HierarchicalLanguageFabric::tokens_seen() const noexcept {
    return tokens_seen_;
}

void HierarchicalLanguageFabric::index_episode(const std::size_t episode_index) {
    const std::vector<TokenId> tokens = unique_content_tokens(episodes_[episode_index].cue);
    for (const TokenId token : tokens) {
        episode_postings_[token].push_back(episode_index);
        ++episode_document_frequency_[token];
    }
}

void HierarchicalLanguageFabric::unindex_episode(
    const std::size_t episode_index
) {
    const std::vector<TokenId> tokens =
        unique_content_tokens(episodes_[episode_index].cue);
    for (const TokenId token : tokens) {
        const auto posting = episode_postings_.find(token);
        if (posting == episode_postings_.end()) {
            throw std::logic_error("missing episode posting during replacement");
        }
        auto& indices = posting->second;
        const auto found = std::find(indices.begin(), indices.end(), episode_index);
        if (found == indices.end()) {
            throw std::logic_error("missing episode index during replacement");
        }
        indices.erase(found);
        if (indices.empty()) {
            episode_postings_.erase(posting);
        }
        const auto frequency = episode_document_frequency_.find(token);
        if (frequency == episode_document_frequency_.end() ||
            frequency->second == 0U) {
            throw std::logic_error("invalid episode frequency during replacement");
        }
        --frequency->second;
        if (frequency->second == 0U) {
            episode_document_frequency_.erase(frequency);
        }
    }
}

std::size_t HierarchicalLanguageFabric::context_replacement_index(
    const ContextKey& key
) const {
    if (contexts_.empty()) {
        throw std::logic_error("cannot replace an empty context table");
    }
    std::uint64_t state = ContextKeyHash{}(key);
    std::size_t selected = contexts_.size();
    constexpr std::size_t candidate_count = 8U;
    for (std::size_t candidate = 0U; candidate < candidate_count; ++candidate) {
        const std::size_t index = static_cast<std::size_t>(
            next_random(state) % contexts_.size()
        );
        if (contexts_[index].history.empty()) {
            continue;
        }
        if (selected == contexts_.size() ||
            contexts_[index].support < contexts_[selected].support ||
            (contexts_[index].support == contexts_[selected].support &&
             contexts_[index].id < contexts_[selected].id)) {
            selected = index;
        }
    }
    if (selected != contexts_.size()) {
        return selected;
    }
    const auto found = std::find_if(
        contexts_.begin(), contexts_.end(),
        [](const PredictiveContext& context) {
            return !context.history.empty();
        }
    );
    if (found == contexts_.end()) {
        throw std::runtime_error(
            "language context capacity contains no replaceable entry"
        );
    }
    return static_cast<std::size_t>(
        std::distance(contexts_.begin(), found)
    );
}

std::size_t HierarchicalLanguageFabric::episode_replacement_index(
    const std::span<const TokenId> cue,
    const std::span<const TokenId> response
) const {
    if (episodes_.empty()) {
        throw std::logic_error("cannot replace an empty episode table");
    }
    std::uint64_t state = fnv_offset_basis;
    for (const TokenId token : cue) hash_u64(state, token);
    hash_u64(state, cue.size());
    for (const TokenId token : response) hash_u64(state, token);
    hash_u64(state, response.size());
    std::size_t selected = 0U;
    constexpr std::size_t candidate_count = 8U;
    for (std::size_t candidate = 0U; candidate < candidate_count; ++candidate) {
        const std::size_t index = static_cast<std::size_t>(
            next_random(state) % episodes_.size()
        );
        if (episodes_[index].support < episodes_[selected].support ||
            (episodes_[index].support == episodes_[selected].support &&
             episodes_[index].id < episodes_[selected].id)) {
            selected = index;
        }
    }
    return selected;
}

void HierarchicalLanguageFabric::rebuild_index() {
    context_index_.clear();
    context_index_.reserve(contexts_.size());
    for (std::size_t index = 0U; index < contexts_.size(); ++index) {
        const bool inserted = context_index_.emplace(
            ContextKey{contexts_[index].history}, index
        ).second;
        if (!inserted) {
            throw std::invalid_argument("duplicate Solstice predictive context in snapshot");
        }
    }
    episode_postings_.clear();
    episode_document_frequency_.clear();
    for (std::size_t index = 0U; index < episodes_.size(); ++index) {
        index_episode(index);
    }
    rebuild_outcome_indexes();
}

void HierarchicalLanguageFabric::rebuild_outcome_indexes() {
    outcome_indices_.clear();
    if (!indexed_outcome_updates_) {
        return;
    }
    for (std::size_t context_index = 0U;
         context_index < contexts_.size();
         ++context_index) {
        const auto& outcomes = contexts_[context_index].outcomes;
        if (outcomes.size() < outcome_index_threshold) {
            continue;
        }
        auto& index = outcome_indices_[context_index];
        index.reserve(outcomes.size());
        for (std::size_t outcome_index = 0U;
             outcome_index < outcomes.size();
             ++outcome_index) {
            index.emplace_back(outcomes[outcome_index].token, outcome_index);
        }
        std::sort(index.begin(), index.end());
        ++training_operation_stats_.outcome_index_builds;
        training_operation_stats_.outcome_index_entries_built += index.size();
    }
}

void HierarchicalLanguageFabric::set_indexed_outcome_updates(const bool enabled) {
    indexed_outcome_updates_ = enabled;
    rebuild_outcome_indexes();
}

void HierarchicalLanguageFabric::set_fused_dialogue_encoding(
    const bool enabled
) {
    fused_dialogue_encoding_ = enabled;
}

void HierarchicalLanguageFabric::set_bounded_capacity_replacement(
    const bool enabled
) {
    bounded_capacity_replacement_ = enabled;
}

LanguageTrainingOperationStats
HierarchicalLanguageFabric::training_operation_stats() const noexcept {
    return training_operation_stats_;
}

HierarchicalLanguageSnapshot HierarchicalLanguageFabric::snapshot() const {
    return HierarchicalLanguageSnapshot{
        config_, next_context_id_, next_episode_id_, tokens_seen_, contexts_, episodes_,
    };
}

HierarchicalLanguageFabric HierarchicalLanguageFabric::from_snapshot(
    HierarchicalLanguageSnapshot snapshot
) {
    HierarchicalLanguageFabric fabric(snapshot.config);
    if (snapshot.contexts.size() > snapshot.config.maximum_contexts ||
        snapshot.episodes.size() > snapshot.config.maximum_episodes ||
        snapshot.next_context_id == 0U || snapshot.next_episode_id == 0U) {
        throw std::invalid_argument("invalid Solstice language snapshot dimensions");
    }
    for (const PredictiveContext& context : snapshot.contexts) {
        if (context.id == 0U || context.history.size() > snapshot.config.context_orders.back() ||
            context.outcomes.empty()) {
            throw std::invalid_argument("invalid Solstice predictive context snapshot");
        }
    }
    for (const LanguageEpisode& episode : snapshot.episodes) {
        if (episode.id == 0U || episode.cue.size() > snapshot.config.maximum_episode_cue_tokens ||
            episode.response.size() > snapshot.config.maximum_episode_response_tokens) {
            throw std::invalid_argument("invalid Solstice language episode snapshot");
        }
    }
    fabric.next_context_id_ = snapshot.next_context_id;
    fabric.next_episode_id_ = snapshot.next_episode_id;
    fabric.tokens_seen_ = snapshot.tokens_seen;
    fabric.contexts_ = std::move(snapshot.contexts);
    fabric.episodes_ = std::move(snapshot.episodes);
    fabric.rebuild_index();
    return fabric;
}

std::uint64_t HierarchicalLanguageFabric::deterministic_hash() const noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, tokens_seen_);
    for (const std::size_t order : config_.context_orders) {
        hash_u64(hash, order);
    }
    hash_double(hash, config_.smoothing);
    std::vector<const PredictiveContext*> ordered_contexts;
    ordered_contexts.reserve(contexts_.size());
    for (const PredictiveContext& context : contexts_) {
        ordered_contexts.push_back(&context);
    }
    std::sort(
        ordered_contexts.begin(), ordered_contexts.end(),
        [](const PredictiveContext* left, const PredictiveContext* right) {
            return left->id < right->id;
        }
    );
    for (const PredictiveContext* context : ordered_contexts) {
        hash_u64(hash, context->id);
        hash_u64(hash, context->support);
        for (const TokenId token : context->history) {
            hash_u64(hash, token);
        }
        std::vector<TokenOutcome> outcomes = context->outcomes;
        std::sort(
            outcomes.begin(), outcomes.end(),
            [](const TokenOutcome& left, const TokenOutcome& right) {
                return left.token < right.token;
            }
        );
        for (const TokenOutcome& outcome : outcomes) {
            hash_u64(hash, outcome.token);
            hash_u64(hash, outcome.count);
        }
    }
    for (const LanguageEpisode& episode : episodes_) {
        hash_u64(hash, episode.id);
        hash_u64(hash, episode.support);
        for (const TokenId token : episode.cue) {
            hash_u64(hash, token);
        }
        hash_u64(hash, 0xFFFF'FFFFULL);
        for (const TokenId token : episode.response) {
            hash_u64(hash, token);
        }
    }
    return hash;
}

}  // namespace rlf::solstice
