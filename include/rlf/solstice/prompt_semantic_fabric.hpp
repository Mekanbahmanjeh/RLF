#pragma once

#include "rlf/core/phase_vector.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rlf::solstice {

struct PromptSemanticConfig final {
    std::size_t phase_dimension{64U};
    std::size_t maximum_words{2'000'000U};
    std::size_t maximum_word_bytes{64U};
    std::size_t maximum_words_per_record{4'096U};
    std::size_t context_window{4U};
    std::size_t bucket_bits{14U};
    std::size_t maximum_expansions_per_word{8U};
    std::uint64_t minimum_support{2U};
    double learning_rate{0.10};
    std::uint64_t seed{0x50524F4D50545345ULL};
};

struct PromptSemanticMode final {
    std::uint64_t id{};
    std::string word;
    core::PhaseVector context_prototype{core::PhaseVector::zeros(1U)};
    std::uint64_t support{};
};

struct PromptSemanticStats final {
    std::uint64_t records_seen{};
    std::uint64_t words_seen{};
    std::uint64_t modes_created{};
    std::uint64_t modes_updated{};
    std::uint64_t capacity_skips{};
    std::uint64_t semantic_queries{};
    std::uint64_t bucket_candidates_scored{};
};

struct PromptSemanticSnapshot final {
    PromptSemanticConfig config;
    std::uint64_t next_mode_id{1U};
    std::vector<PromptSemanticMode> modes;
    PromptSemanticStats stats;
};

// Non-neural distributional language component. Each word stores a locally
// updated phase prototype of its neighboring word carriers. Similar contexts
// share sparse phase-signature buckets used to expand image prompts.
class PromptSemanticFabric final {
public:
    explicit PromptSemanticFabric(PromptSemanticConfig config = {});

    void train_record(std::string_view text);
    [[nodiscard]] std::vector<std::uint64_t> semantic_concept_hashes(
        std::string_view prompt,
        std::size_t maximum_concepts
    ) const;
    [[nodiscard]] std::vector<std::string> similar_words(
        std::string_view word,
        std::size_t maximum_results
    ) const;

    [[nodiscard]] const PromptSemanticConfig& config() const noexcept;
    [[nodiscard]] std::span<const PromptSemanticMode> modes() const noexcept;
    [[nodiscard]] PromptSemanticStats stats() const noexcept;
    [[nodiscard]] std::size_t estimated_bytes() const noexcept;
    [[nodiscard]] std::uint64_t deterministic_hash() const noexcept;
    [[nodiscard]] PromptSemanticSnapshot snapshot() const;
    [[nodiscard]] static PromptSemanticFabric from_snapshot(
        PromptSemanticSnapshot snapshot
    );

private:
    PromptSemanticConfig config_;
    std::uint64_t next_mode_id_{1U};
    std::vector<PromptSemanticMode> modes_;
    // Deterministically derived, non-persistent cosine/sine carrier cache:
    // [mode][dimension][cos,sin]. Rebuilt from words after checkpoint load.
    std::vector<float> carrier_cartesian_;
    std::unordered_map<std::string, std::size_t> word_index_;
    mutable std::unordered_map<std::uint16_t, std::vector<std::size_t>> buckets_;
    mutable bool buckets_dirty_{true};
    mutable PromptSemanticStats stats_;

    [[nodiscard]] std::vector<std::string> tokenize(std::string_view text) const;
    [[nodiscard]] core::PhaseVector carrier(std::string_view word) const;
    [[nodiscard]] std::uint16_t signature(const core::PhaseVector& value) const;
    void ensure_buckets() const;
    void rebuild_indices();
};

}  // namespace rlf::solstice
