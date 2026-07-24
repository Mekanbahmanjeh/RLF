#include "rlf/solstice/tokenizer.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>
#include <utility>

namespace rlf::solstice {
namespace {

constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;
constexpr std::size_t byte_vocabulary_size = 256U;

constexpr std::string_view special_names[] = {
    "<pad>",
    "<bos>",
    "<eos>",
    "<user>",
    "<assistant>",
    "<image>",
    "<image_end>",
    "<tool_call>",
    "<tool_result>",
    "<tool_error>",
};

[[nodiscard]] std::uint64_t pair_key(
    const TokenId left,
    const TokenId right
) noexcept {
    return (static_cast<std::uint64_t>(left) << 32U) |
        static_cast<std::uint64_t>(right);
}

[[nodiscard]] TokenId key_left(const std::uint64_t key) noexcept {
    return static_cast<TokenId>(key >> 32U);
}

[[nodiscard]] TokenId key_right(const std::uint64_t key) noexcept {
    return static_cast<TokenId>(key & 0xFFFF'FFFFULL);
}

void hash_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (unsigned int index = 0U; index < 8U; ++index) {
        hash ^= (value >> (index * 8U)) & 0xFFULL;
        hash *= fnv_prime;
    }
}

void hash_string(std::uint64_t& hash, const std::string_view value) noexcept {
    hash_u64(hash, value.size());
    for (const char character : value) {
        hash ^= static_cast<unsigned char>(character);
        hash *= fnv_prime;
    }
}

}  // namespace

SolsticeTokenizer::SolsticeTokenizer(TokenizerConfig config)
    : config_(config) {
    if (config_.maximum_vocabulary <
            byte_vocabulary_size + std::size(special_names) ||
        config_.maximum_vocabulary >
            static_cast<std::size_t>(std::numeric_limits<TokenId>::max()) ||
        config_.maximum_piece_bytes == 0U ||
        config_.maximum_training_bytes == 0U) {
        throw std::invalid_argument("invalid Solstice tokenizer configuration");
    }
    initialize_base_vocabulary();
}

void SolsticeTokenizer::initialize_base_vocabulary() {
    pieces_.clear();
    merges_.clear();
    pieces_.reserve(config_.maximum_vocabulary);
    for (std::size_t value = 0U; value < byte_vocabulary_size; ++value) {
        TokenPiece piece_value;
        piece_value.id = static_cast<TokenId>(value);
        piece_value.bytes.push_back(static_cast<char>(value));
        pieces_.push_back(std::move(piece_value));
    }
    for (const std::string_view special : special_names) {
        const auto id = static_cast<TokenId>(pieces_.size());
        pieces_.push_back(TokenPiece{id, std::string(special), true});
    }
    rebuild_indices();
}

void SolsticeTokenizer::rebuild_indices() {
    special_ids_.clear();
    piece_ids_.clear();
    for (const TokenPiece& token_piece : pieces_) {
        if (token_piece.special) {
            special_ids_.emplace(token_piece.bytes, token_piece.id);
        } else {
            piece_ids_.emplace(token_piece.bytes, token_piece.id);
        }
    }
}

void SolsticeTokenizer::train(const std::string_view corpus) {
    if (corpus.size() > config_.maximum_training_bytes) {
        throw std::invalid_argument("Solstice tokenizer training corpus exceeds configured limit");
    }
    std::vector<TokenId> sequence;
    sequence.reserve(corpus.size());
    for (const char raw_character : corpus) {
        sequence.push_back(static_cast<unsigned char>(raw_character));
    }
    sequence = apply_merges(std::move(sequence));

    const std::size_t available_merges = std::min(
        config_.maximum_merges - std::min(config_.maximum_merges, merges_.size()),
        config_.maximum_vocabulary - pieces_.size()
    );
    for (std::size_t iteration = 0U;
         iteration < available_merges && sequence.size() >= 2U;
         ++iteration) {
        std::unordered_map<std::uint64_t, std::uint64_t> counts;
        counts.reserve(sequence.size());
        for (std::size_t index = 1U; index < sequence.size(); ++index) {
            ++counts[pair_key(sequence[index - 1U], sequence[index])];
        }

        std::uint64_t best_key = 0U;
        std::uint64_t best_count = 0U;
        for (const auto& [key, count] : counts) {
            if (count > best_count || (count == best_count && key < best_key)) {
                best_key = key;
                best_count = count;
            }
        }
        if (best_count < config_.minimum_pair_support) {
            break;
        }

        const TokenId left = key_left(best_key);
        const TokenId right = key_right(best_key);
        if (static_cast<std::size_t>(left) >= pieces_.size() ||
            static_cast<std::size_t>(right) >= pieces_.size()) {
            throw std::logic_error("Solstice tokenizer pair references unknown token");
        }
        const std::string combined = pieces_[left].bytes + pieces_[right].bytes;
        if (combined.size() > config_.maximum_piece_bytes) {
            counts.erase(best_key);
            if (counts.empty()) {
                break;
            }
            // Avoid repeatedly selecting an oversized pair by replacing it once
            // with a count below the acceptance threshold.
            bool found_acceptable = false;
            for (const auto& [candidate_key, candidate_count] : counts) {
                const TokenId candidate_left = key_left(candidate_key);
                const TokenId candidate_right = key_right(candidate_key);
                const std::string candidate =
                    pieces_[candidate_left].bytes + pieces_[candidate_right].bytes;
                if (candidate.size() <= config_.maximum_piece_bytes &&
                    candidate_count >= config_.minimum_pair_support &&
                    (!found_acceptable || candidate_count > best_count ||
                     (candidate_count == best_count && candidate_key < best_key))) {
                    best_key = candidate_key;
                    best_count = candidate_count;
                    found_acceptable = true;
                }
            }
            if (!found_acceptable) {
                break;
            }
        }

        const TokenId selected_left = key_left(best_key);
        const TokenId selected_right = key_right(best_key);
        const std::string selected_piece =
            pieces_[selected_left].bytes + pieces_[selected_right].bytes;
        const auto existing = piece_ids_.find(selected_piece);
        TokenId result{};
        if (existing != piece_ids_.end()) {
            result = existing->second;
        } else {
            result = static_cast<TokenId>(pieces_.size());
            pieces_.push_back(TokenPiece{result, selected_piece, false});
            piece_ids_.emplace(selected_piece, result);
        }
        merges_.push_back(TokenMerge{
            selected_left,
            selected_right,
            result,
            best_count,
        });

        std::vector<TokenId> merged;
        merged.reserve(sequence.size());
        for (std::size_t index = 0U; index < sequence.size();) {
            if (index + 1U < sequence.size() &&
                sequence[index] == selected_left &&
                sequence[index + 1U] == selected_right) {
                merged.push_back(result);
                index += 2U;
            } else {
                merged.push_back(sequence[index]);
                ++index;
            }
        }
        sequence = std::move(merged);
    }
}

std::vector<TokenId> SolsticeTokenizer::apply_merges(
    std::vector<TokenId> tokens
) const {
    for (const TokenMerge& merge : merges_) {
        if (tokens.size() < 2U) {
            break;
        }
        std::vector<TokenId> result;
        result.reserve(tokens.size());
        for (std::size_t index = 0U; index < tokens.size();) {
            if (index + 1U < tokens.size() &&
                tokens[index] == merge.left &&
                tokens[index + 1U] == merge.right) {
                result.push_back(merge.result);
                index += 2U;
            } else {
                result.push_back(tokens[index]);
                ++index;
            }
        }
        tokens = std::move(result);
    }
    return tokens;
}

std::vector<TokenId> SolsticeTokenizer::encode(const std::string_view text) const {
    std::vector<TokenId> tokens;
    tokens.reserve(text.size());
    for (const char raw_character : text) {
        tokens.push_back(static_cast<unsigned char>(raw_character));
    }
    return apply_merges(std::move(tokens));
}

std::string SolsticeTokenizer::decode(
    const std::span<const TokenId> tokens,
    const bool include_special_tokens
) const {
    std::string result;
    for (const TokenId token : tokens) {
        const TokenPiece& token_piece = piece(token);
        if (!token_piece.special || include_special_tokens) {
            result += token_piece.bytes;
        }
    }
    return result;
}

TokenId SolsticeTokenizer::special_id(const std::string_view name) const {
    const auto found = special_ids_.find(std::string(name));
    if (found == special_ids_.end()) {
        throw std::out_of_range("unknown Solstice special token");
    }
    return found->second;
}

bool SolsticeTokenizer::is_special(const TokenId id) const noexcept {
    return static_cast<std::size_t>(id) < pieces_.size() && pieces_[id].special;
}

const TokenPiece& SolsticeTokenizer::piece(const TokenId id) const {
    if (static_cast<std::size_t>(id) >= pieces_.size()) {
        throw std::out_of_range("unknown Solstice token ID");
    }
    return pieces_[id];
}

std::size_t SolsticeTokenizer::vocabulary_size() const noexcept {
    return pieces_.size();
}

const TokenizerConfig& SolsticeTokenizer::config() const noexcept {
    return config_;
}

std::span<const TokenPiece> SolsticeTokenizer::pieces() const noexcept {
    return pieces_;
}

std::span<const TokenMerge> SolsticeTokenizer::merges() const noexcept {
    return merges_;
}

TokenizerSnapshot SolsticeTokenizer::snapshot() const {
    return TokenizerSnapshot{config_, pieces_, merges_};
}

SolsticeTokenizer SolsticeTokenizer::from_snapshot(TokenizerSnapshot snapshot) {
    SolsticeTokenizer tokenizer(snapshot.config);
    if (snapshot.pieces.size() < byte_vocabulary_size + std::size(special_names) ||
        snapshot.pieces.size() > snapshot.config.maximum_vocabulary ||
        snapshot.merges.size() > snapshot.config.maximum_merges) {
        throw std::invalid_argument("invalid Solstice tokenizer snapshot dimensions");
    }
    for (std::size_t index = 0U; index < snapshot.pieces.size(); ++index) {
        if (snapshot.pieces[index].id != static_cast<TokenId>(index) ||
            snapshot.pieces[index].bytes.empty()) {
            throw std::invalid_argument("invalid Solstice tokenizer snapshot token table");
        }
    }
    for (const TokenMerge& merge : snapshot.merges) {
        if (static_cast<std::size_t>(merge.left) >= snapshot.pieces.size() ||
            static_cast<std::size_t>(merge.right) >= snapshot.pieces.size() ||
            static_cast<std::size_t>(merge.result) >= snapshot.pieces.size()) {
            throw std::invalid_argument("invalid Solstice tokenizer snapshot merge");
        }
    }
    tokenizer.pieces_ = std::move(snapshot.pieces);
    tokenizer.merges_ = std::move(snapshot.merges);
    tokenizer.rebuild_indices();
    for (const std::string_view special : special_names) {
        if (!tokenizer.special_ids_.contains(std::string(special))) {
            throw std::invalid_argument("Solstice tokenizer snapshot is missing a special token");
        }
    }
    return tokenizer;
}

std::uint64_t SolsticeTokenizer::deterministic_hash() const noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, config_.maximum_vocabulary);
    hash_u64(hash, config_.maximum_merges);
    hash_u64(hash, config_.minimum_pair_support);
    hash_u64(hash, config_.maximum_piece_bytes);
    for (const TokenPiece& token_piece : pieces_) {
        hash_u64(hash, token_piece.id);
        hash_u64(hash, token_piece.special ? 1U : 0U);
        hash_string(hash, token_piece.bytes);
    }
    for (const TokenMerge& merge : merges_) {
        hash_u64(hash, merge.left);
        hash_u64(hash, merge.right);
        hash_u64(hash, merge.result);
        hash_u64(hash, merge.support);
    }
    return hash;
}

}  // namespace rlf::solstice
