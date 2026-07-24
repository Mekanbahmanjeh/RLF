#pragma once

#include "rlf/solstice/tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rlf::solstice {

struct HierarchicalLanguageConfig final {
    std::vector<std::size_t> context_orders{0U, 1U, 2U, 4U, 8U, 16U, 32U, 64U};
    std::size_t maximum_contexts{1'500'000U};
    std::size_t maximum_episodes{100'000U};
    std::size_t maximum_episode_cue_tokens{256U};
    std::size_t maximum_episode_response_tokens{512U};
    std::size_t maximum_generation_tokens{256U};
    std::size_t prediction_candidate_limit{128U};
    double smoothing{0.05};
    double long_context_weight{0.30};
    double episode_conditioning_weight{8.0};
    double repetition_penalty{1.15};
};

struct TokenOutcome final {
    TokenId token{};
    std::uint64_t count{};
};

struct PredictiveContext final {
    std::uint64_t id{};
    std::vector<TokenId> history;
    std::uint64_t support{};
    std::vector<TokenOutcome> outcomes;
};

struct LanguageEpisode final {
    std::uint64_t id{};
    std::vector<TokenId> cue;
    std::vector<TokenId> response;
    std::uint64_t support{1U};
};

struct TokenCandidate final {
    TokenId token{};
    double probability{};
    double score{};
};

struct HierarchicalPrediction final {
    std::vector<TokenCandidate> candidates;
    std::size_t deepest_context_order{};
    double uncertainty{1.0};
};

struct GenerationSettings final {
    std::size_t maximum_tokens{128U};
    std::size_t top_k{16U};
    double temperature{0.8};
    bool deterministic{true};
    std::uint64_t seed{0x534F4C5354494345ULL};
};

struct LanguageResponse final {
    std::string text;
    std::vector<TokenId> generated_tokens;
    double episode_similarity{};
    double uncertainty{1.0};
};

struct HierarchicalLanguageSnapshot final {
    HierarchicalLanguageConfig config;
    std::uint64_t next_context_id{1U};
    std::uint64_t next_episode_id{1U};
    std::uint64_t tokens_seen{};
    std::vector<PredictiveContext> contexts;
    std::vector<LanguageEpisode> episodes;
};

struct LanguageTrainingOperationStats final {
    std::uint64_t dialogue_training_calls{};
    std::uint64_t dialogue_tokenizer_encode_calls{};
    std::uint64_t redundant_dialogue_encode_calls_avoided{};
    std::uint64_t episode_duplicate_updates{};
    std::uint64_t episode_insert_attempts{};
    std::uint64_t episode_inserts{};
    std::uint64_t episode_replacements{};
    std::uint64_t episode_capacity_skips{};
    std::uint64_t context_insert_attempts{};
    std::uint64_t context_inserts{};
    std::uint64_t context_replacements{};
    std::uint64_t context_capacity_skips{};
    std::uint64_t outcome_update_lookups{};
    std::uint64_t linear_outcome_comparisons{};
    std::uint64_t indexed_outcome_lookups{};
    std::uint64_t outcome_index_builds{};
    std::uint64_t outcome_index_entries_built{};
    std::uint64_t outcome_index_incremental_inserts{};
};

class HierarchicalLanguageFabric final {
public:
    explicit HierarchicalLanguageFabric(HierarchicalLanguageConfig config = {});

    void train_corpus(const SolsticeTokenizer& tokenizer, std::string_view corpus);
    void train_dialogue(
        const SolsticeTokenizer& tokenizer,
        std::string_view prompt,
        std::string_view response,
        std::string_view grounding = {}
    );
    void train_token_sequence(std::span<const TokenId> sequence);

    [[nodiscard]] HierarchicalPrediction predict_next(
        std::span<const TokenId> history
    ) const;
    [[nodiscard]] LanguageResponse generate_response(
        const SolsticeTokenizer& tokenizer,
        std::string_view prompt,
        std::string_view grounding,
        GenerationSettings settings = {}
    ) const;
    [[nodiscard]] std::string generate_continuation(
        const SolsticeTokenizer& tokenizer,
        std::string_view prompt,
        GenerationSettings settings = {}
    ) const;

    [[nodiscard]] std::span<const PredictiveContext> contexts() const noexcept;
    [[nodiscard]] std::span<const LanguageEpisode> episodes() const noexcept;
    [[nodiscard]] const HierarchicalLanguageConfig& config() const noexcept;
    [[nodiscard]] std::uint64_t tokens_seen() const noexcept;
    [[nodiscard]] std::uint64_t deterministic_hash() const noexcept;
    void set_indexed_outcome_updates(bool enabled);
    void set_fused_dialogue_encoding(bool enabled);
    void set_bounded_capacity_replacement(bool enabled);
    [[nodiscard]] LanguageTrainingOperationStats training_operation_stats() const noexcept;

    [[nodiscard]] HierarchicalLanguageSnapshot snapshot() const;
    [[nodiscard]] static HierarchicalLanguageFabric from_snapshot(
        HierarchicalLanguageSnapshot snapshot
    );

private:
    struct ContextKey final {
        std::vector<TokenId> history;
        [[nodiscard]] bool operator==(const ContextKey&) const noexcept = default;
    };

    struct ContextKeyHash final {
        [[nodiscard]] std::size_t operator()(const ContextKey& key) const noexcept;
    };

    struct EpisodeMatch final {
        const LanguageEpisode* episode{};
        double similarity{};
    };

    [[nodiscard]] std::vector<TokenId> dialogue_prefix(
        const SolsticeTokenizer& tokenizer,
        std::string_view prompt,
        std::string_view grounding
    ) const;
    [[nodiscard]] std::vector<EpisodeMatch> match_episodes(
        std::span<const TokenId> cue,
        std::size_t limit
    ) const;
    [[nodiscard]] HierarchicalPrediction predict_conditioned(
        std::span<const TokenId> history,
        std::span<const EpisodeMatch> matches,
        std::size_t response_position
    ) const;
    void rebuild_index();
    void rebuild_outcome_indexes();
    void index_episode(std::size_t episode_index);
    void unindex_episode(std::size_t episode_index);
    [[nodiscard]] std::size_t context_replacement_index(
        const ContextKey& key
    ) const;
    [[nodiscard]] std::size_t episode_replacement_index(
        std::span<const TokenId> cue,
        std::span<const TokenId> response
    ) const;

    HierarchicalLanguageConfig config_;
    std::uint64_t next_context_id_{1U};
    std::uint64_t next_episode_id_{1U};
    std::uint64_t tokens_seen_{};
    std::vector<PredictiveContext> contexts_;
    std::vector<LanguageEpisode> episodes_;
    std::unordered_map<ContextKey, std::size_t, ContextKeyHash> context_index_;
    std::unordered_map<TokenId, std::vector<std::size_t>> episode_postings_;
    std::unordered_map<TokenId, std::size_t> episode_document_frequency_;
    std::unordered_map<
        std::size_t,
        std::vector<std::pair<TokenId, std::size_t>>
    > outcome_indices_;
    bool indexed_outcome_updates_{true};
    bool fused_dialogue_encoding_{true};
    bool bounded_capacity_replacement_{};
    LanguageTrainingOperationStats training_operation_stats_;
};

}  // namespace rlf::solstice
