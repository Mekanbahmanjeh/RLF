#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rlf::solstice {

using TokenId = std::uint32_t;

struct TokenizerConfig final {
    std::size_t maximum_vocabulary{8'192U};
    std::size_t maximum_merges{2'048U};
    std::size_t minimum_pair_support{2U};
    std::size_t maximum_piece_bytes{64U};
    std::size_t maximum_training_bytes{64U * 1024U * 1024U};
};

struct TokenPiece final {
    TokenId id{};
    std::string bytes;
    bool special{};
};

struct TokenMerge final {
    TokenId left{};
    TokenId right{};
    TokenId result{};
    std::uint64_t support{};
};

struct TokenizerSnapshot final {
    TokenizerConfig config;
    std::vector<TokenPiece> pieces;
    std::vector<TokenMerge> merges;
};

class SolsticeTokenizer final {
public:
    explicit SolsticeTokenizer(TokenizerConfig config = {});

    void train(std::string_view corpus);

    [[nodiscard]] std::vector<TokenId> encode(std::string_view text) const;
    [[nodiscard]] std::string decode(
        std::span<const TokenId> tokens,
        bool include_special_tokens = false
    ) const;

    [[nodiscard]] TokenId special_id(std::string_view name) const;
    [[nodiscard]] bool is_special(TokenId id) const noexcept;
    [[nodiscard]] const TokenPiece& piece(TokenId id) const;
    [[nodiscard]] std::size_t vocabulary_size() const noexcept;
    [[nodiscard]] const TokenizerConfig& config() const noexcept;
    [[nodiscard]] std::span<const TokenPiece> pieces() const noexcept;
    [[nodiscard]] std::span<const TokenMerge> merges() const noexcept;

    [[nodiscard]] TokenizerSnapshot snapshot() const;
    [[nodiscard]] static SolsticeTokenizer from_snapshot(TokenizerSnapshot snapshot);
    [[nodiscard]] std::uint64_t deterministic_hash() const noexcept;

private:
    void initialize_base_vocabulary();
    void rebuild_indices();
    [[nodiscard]] std::vector<TokenId> apply_merges(
        std::vector<TokenId> tokens
    ) const;

    TokenizerConfig config_;
    std::vector<TokenPiece> pieces_;
    std::vector<TokenMerge> merges_;
    std::unordered_map<std::string, TokenId> special_ids_;
    std::unordered_map<std::string, TokenId> piece_ids_;
};

}  // namespace rlf::solstice
