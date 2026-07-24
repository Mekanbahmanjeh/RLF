#include "rlf/core/language_fabric.hpp"

#include "rlf/core/deterministic_rng.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rlf::core {
namespace {

constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;
constexpr double probability_floor = 1.0e-12;
constexpr std::size_t base_byte_lexemes = 256U;

void hash_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (unsigned int byte = 0U; byte < 8U; ++byte) {
        hash ^= (value >> (byte * 8U)) & 0xFFULL;
        hash *= fnv_prime;
    }
}

void hash_string(std::uint64_t& hash, const std::string_view value) noexcept {
    hash_u64(hash, value.size());
    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
        hash ^= character;
        hash *= fnv_prime;
    }
}

void hash_double(std::uint64_t& hash, const double value) noexcept {
    hash_u64(hash, std::bit_cast<std::uint64_t>(value));
}

[[nodiscard]] std::uint64_t string_hash(const std::string_view value) noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_string(hash, value);
    return hash;
}

[[nodiscard]] bool is_word_byte(const unsigned char character) noexcept {
    return std::isalnum(character) != 0 || character == '\'' || character == '-';
}

[[nodiscard]] bool token_is_content(const std::string_view bytes) noexcept {
    return std::any_of(bytes.begin(), bytes.end(), [](const unsigned char character) {
        return std::isalnum(character) != 0;
    });
}

[[nodiscard]] std::uint8_t surface_form(const std::string_view bytes) noexcept {
    std::string lowered;
    lowered.reserve(bytes.size());
    for (const char raw_character : bytes) {
        const auto character = static_cast<unsigned char>(raw_character);
        lowered.push_back(static_cast<char>(std::tolower(character)));
    }
    if (lowered.size() >= 3U && lowered.ends_with("ing")) {
        return 3U;
    }
    if (lowered.size() >= 2U && lowered.ends_with("ed")) {
        return 2U;
    }
    if (lowered.size() >= 2U && lowered.ends_with('s')) {
        return 1U;
    }
    return 0U;
}

void validate_config(const LanguageFabricConfig& config) {
    if (config.phase_dimension == 0U ||
        config.maximum_lexemes < base_byte_lexemes ||
        config.maximum_lexemes > std::numeric_limits<std::uint32_t>::max() ||
        config.maximum_merges > config.maximum_lexemes - base_byte_lexemes ||
        config.minimum_pair_support < 2U || config.maximum_contexts == 0U ||
        config.maximum_context_order == 0U ||
        config.minimum_context_support == 0U ||
        config.maximum_constructions == 0U ||
        config.minimum_construction_support == 0U ||
        config.maximum_generation_tokens == 0U ||
        config.maximum_semantic_values == 0U ||
        config.maximum_surfaces_per_concept == 0U ||
        !std::isfinite(config.smoothing) || config.smoothing <= 0.0 ||
        !std::isfinite(config.minimum_lexical_score) ||
        config.minimum_lexical_score < 0.0 ||
        !std::isfinite(config.construction_support_weight) ||
        config.construction_support_weight < 0.0 ||
        !std::isfinite(config.literal_match_weight) ||
        config.literal_match_weight < 0.0 ||
        !std::isfinite(config.slot_match_weight) ||
        config.slot_match_weight <= 0.0) {
        throw std::invalid_argument("invalid RLF-5 language-fabric configuration");
    }
}

[[nodiscard]] std::string construction_signature(
    const LanguageAct act,
    const std::span<const LanguagePatternItem> pattern
) {
    std::string signature;
    signature.reserve(16U + pattern.size() * 32U);
    signature += std::to_string(static_cast<unsigned int>(act));
    signature.push_back('|');
    for (const auto& item : pattern) {
        signature += std::to_string(static_cast<unsigned int>(item.kind));
        signature.push_back(':');
        if (item.kind == LanguagePatternKind::literal) {
            signature += std::to_string(item.token_id);
        } else {
            signature += std::to_string(static_cast<unsigned int>(item.role));
            signature.push_back('/');
            signature += std::to_string(static_cast<unsigned int>(item.surface_form));
        }
        signature.push_back(',');
    }
    return signature;
}

[[nodiscard]] std::size_t role_index(const LanguageRole role) noexcept {
    return static_cast<std::size_t>(role);
}

[[nodiscard]] std::size_t act_index(const LanguageAct act) noexcept {
    return static_cast<std::size_t>(act);
}

constexpr std::array<LanguageRole, 6U> all_roles{
    LanguageRole::predicate,
    LanguageRole::agent,
    LanguageRole::patient,
    LanguageRole::agent_attribute,
    LanguageRole::patient_attribute,
    LanguageRole::location,
};

[[nodiscard]] bool frame_matches_query(
    const LanguageFrame& fact,
    const LanguageFrame& query
) noexcept {
    if (!query.predicate.empty() && fact.predicate != query.predicate) {
        return false;
    }
    if (!query.agent.empty() && fact.agent != query.agent) {
        return false;
    }
    if (!query.patient.empty() && fact.patient != query.patient) {
        return false;
    }
    if (!query.agent_attribute.empty() &&
        fact.agent_attribute != query.agent_attribute) {
        return false;
    }
    if (!query.patient_attribute.empty() &&
        fact.patient_attribute != query.patient_attribute) {
        return false;
    }
    if (!query.location.empty() && fact.location != query.location) {
        return false;
    }
    return true;
}

[[nodiscard]] std::uint64_t pair_key(
    const std::uint64_t left,
    const std::uint64_t right
) noexcept {
    return (left << 32U) | right;
}

[[nodiscard]] std::pair<std::uint64_t, std::uint64_t> unpack_pair(
    const std::uint64_t key
) noexcept {
    return {key >> 32U, key & 0xFFFF'FFFFULL};
}

}  // namespace

std::size_t LanguageFabric::VectorHash::operator()(
    const std::vector<std::uint64_t>& value
) const noexcept {
    std::uint64_t hash = fnv_offset_basis;
    for (const auto item : value) {
        hash_u64(hash, item);
    }
    return static_cast<std::size_t>(hash);
}

std::size_t LanguageFabric::ConceptKeyHash::operator()(
    const ConceptKey& value
) const noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, static_cast<std::uint64_t>(value.role));
    hash_string(hash, value.value);
    return static_cast<std::size_t>(hash);
}

LanguageFabric::LanguageFabric(
    LanguageFabricConfig config,
    const std::uint64_t seed
) : config_(std::move(config)), seed_(seed) {
    validate_config(config_);
    lexemes_.reserve(config_.maximum_lexemes);
    for (std::size_t byte = 0U; byte < base_byte_lexemes; ++byte) {
        std::string bytes(1U, static_cast<char>(byte));
        const std::uint64_t id = next_lexeme_id_++;
        lexemes_.push_back({id, bytes, 0ULL, lexeme_phase(bytes)});
    }
    rebuild_indices();
}

const LanguageFabricConfig& LanguageFabric::config() const noexcept {
    return config_;
}

std::uint64_t LanguageFabric::seed() const noexcept {
    return seed_;
}

std::span<const LanguageLexeme> LanguageFabric::lexemes() const noexcept {
    return lexemes_;
}

std::span<const LanguageMerge> LanguageFabric::merges() const noexcept {
    return merges_;
}

std::span<const LanguageContext> LanguageFabric::contexts() const noexcept {
    return contexts_;
}

std::span<const LanguageConcept> LanguageFabric::concepts() const noexcept {
    return concepts_;
}

std::span<const LanguageConstruction> LanguageFabric::constructions() const noexcept {
    return constructions_;
}

const LanguageFabricStats& LanguageFabric::stats() const noexcept {
    return stats_;
}

PhaseVector LanguageFabric::lexeme_phase(const std::string_view bytes) const {
    const std::uint64_t derived = seed_ ^ string_hash(bytes) ^
        0x4C414E4750484153ULL;
    DeterministicRng rng(derived);
    return PhaseVector::random(config_.phase_dimension, rng);
}

bool LanguageFabric::merge_allowed(
    const LanguageLexeme& left,
    const LanguageLexeme& right
) const noexcept {
    if (left.bytes.empty() || right.bytes.empty()) {
        return false;
    }
    return std::all_of(left.bytes.begin(), left.bytes.end(), [](const unsigned char c) {
               return is_word_byte(c);
           }) &&
        std::all_of(right.bytes.begin(), right.bytes.end(), [](const unsigned char c) {
            return is_word_byte(c);
        });
}

std::vector<std::uint64_t> LanguageFabric::base_encode(
    const std::string_view text
) const {
    std::vector<std::uint64_t> result;
    result.reserve(text.size());
    for (const char raw_character : text) {
        const auto character = static_cast<unsigned char>(raw_character);
        result.push_back(static_cast<std::uint64_t>(character) + 1ULL);
    }
    return result;
}

std::vector<std::uint64_t> LanguageFabric::apply_merges(
    const std::span<const std::uint64_t> base_tokens
) const {
    std::vector<std::uint64_t> sequence(base_tokens.begin(), base_tokens.end());
    for (const LanguageMerge& merge : merges_) {
        std::vector<std::uint64_t> next;
        next.reserve(sequence.size());
        for (std::size_t index = 0U; index < sequence.size();) {
            if (index + 1U < sequence.size() &&
                sequence[index] == merge.left_id &&
                sequence[index + 1U] == merge.right_id) {
                next.push_back(merge.result_id);
                index += 2U;
            } else {
                next.push_back(sequence[index]);
                ++index;
            }
        }
        sequence.swap(next);
    }
    return sequence;
}

void LanguageFabric::learn_lexicon(const std::string_view raw_corpus) {
    if (raw_corpus.empty()) {
        throw std::invalid_argument("RLF-5 raw corpus must not be empty");
    }
    if (!contexts_.empty() || !concepts_.empty() || !constructions_.empty()) {
        throw std::logic_error(
            "RLF-5 lexicon must be learned before language or semantic training"
        );
    }

    lexemes_.resize(base_byte_lexemes);
    merges_.clear();
    next_lexeme_id_ = base_byte_lexemes + 1ULL;
    rebuild_indices();
    std::vector<std::uint64_t> sequence = base_encode(raw_corpus);

    for (std::size_t merge_index = 0U;
         merge_index < config_.maximum_merges &&
         lexemes_.size() < config_.maximum_lexemes;
         ++merge_index) {
        std::unordered_map<std::uint64_t, std::uint64_t> counts;
        counts.reserve(sequence.size() / 2U + 1U);
        for (std::size_t index = 0U; index + 1U < sequence.size(); ++index) {
            const auto left_found = lexeme_index_by_id_.find(sequence[index]);
            const auto right_found = lexeme_index_by_id_.find(sequence[index + 1U]);
            if (left_found == lexeme_index_by_id_.end() ||
                right_found == lexeme_index_by_id_.end()) {
                throw std::logic_error("RLF-5 lexicon index is inconsistent");
            }
            const auto& left = lexemes_[left_found->second];
            const auto& right = lexemes_[right_found->second];
            if (!merge_allowed(left, right)) {
                continue;
            }
            ++counts[pair_key(left.id, right.id)];
        }

        std::uint64_t best_key = 0ULL;
        std::uint64_t best_count = 0ULL;
        std::int64_t best_gain = std::numeric_limits<std::int64_t>::min();
        std::string best_bytes;
        for (const auto& [key, count] : counts) {
            if (count < config_.minimum_pair_support) {
                continue;
            }
            const auto [left_id, right_id] = unpack_pair(key);
            const auto& left = lexeme_by_id(left_id);
            const auto& right = lexeme_by_id(right_id);
            std::string combined = left.bytes + right.bytes;
            if (lexeme_id_by_bytes_.contains(combined)) {
                continue;
            }
            const auto dictionary_cost = static_cast<std::int64_t>(
                2U + combined.size() / 4U
            );
            const auto gain = static_cast<std::int64_t>(count) - dictionary_cost;
            if (gain <= 0) {
                continue;
            }
            if (gain > best_gain ||
                (gain == best_gain &&
                 (count > best_count ||
                  (count == best_count &&
                   (best_bytes.empty() || combined < best_bytes))))) {
                best_key = key;
                best_count = count;
                best_gain = gain;
                best_bytes = std::move(combined);
            }
        }
        if (best_count == 0ULL) {
            break;
        }

        const auto [left_id, right_id] = unpack_pair(best_key);
        const std::uint64_t result_id = next_lexeme_id_++;
        lexemes_.push_back({
            result_id,
            best_bytes,
            best_count,
            lexeme_phase(best_bytes),
        });
        lexeme_index_by_id_[result_id] = lexemes_.size() - 1U;
        lexeme_id_by_bytes_[best_bytes] = result_id;
        merges_.push_back({left_id, right_id, result_id, best_count, best_gain});

        std::vector<std::uint64_t> next;
        next.reserve(sequence.size());
        for (std::size_t index = 0U; index < sequence.size();) {
            if (index + 1U < sequence.size() &&
                sequence[index] == left_id && sequence[index + 1U] == right_id) {
                next.push_back(result_id);
                index += 2U;
            } else {
                next.push_back(sequence[index]);
                ++index;
            }
        }
        sequence.swap(next);
    }

    for (auto& lexeme : lexemes_) {
        lexeme.support = 0ULL;
    }
    for (const auto id : sequence) {
        const auto found = lexeme_index_by_id_.find(id);
        if (found == lexeme_index_by_id_.end()) {
            throw std::logic_error("RLF-5 learned sequence references unknown lexeme");
        }
        ++lexemes_[found->second].support;
    }
    stats_.raw_bytes_seen += raw_corpus.size();
    stats_.lexicon_merges = merges_.size();
}

std::vector<std::uint64_t> LanguageFabric::encode(
    const std::string_view text
) const {
    return apply_merges(base_encode(text));
}

std::string LanguageFabric::decode(
    const std::span<const std::uint64_t> tokens
) const {
    std::string result;
    for (const auto id : tokens) {
        result += lexeme_by_id(id).bytes;
    }
    return result;
}

std::optional<std::size_t> LanguageFabric::context_index(
    const std::span<const std::uint64_t> history
) const {
    const std::vector<std::uint64_t> key(history.begin(), history.end());
    const auto found = context_index_by_history_.find(key);
    if (found == context_index_by_history_.end()) {
        return std::nullopt;
    }
    return found->second;
}

std::size_t LanguageFabric::get_or_create_context(
    const std::span<const std::uint64_t> history
) {
    if (history.size() > config_.maximum_context_order) {
        throw std::invalid_argument("RLF-5 language context is too long");
    }
    if (const auto found = context_index(history); found.has_value()) {
        return *found;
    }
    if (contexts_.size() >= config_.maximum_contexts) {
        throw std::runtime_error("RLF-5 language context capacity exceeded");
    }
    LanguageContext context;
    context.id = next_context_id_++;
    context.history.assign(history.begin(), history.end());
    contexts_.push_back(std::move(context));
    context_index_by_history_[contexts_.back().history] = contexts_.size() - 1U;
    ++stats_.contexts_created;
    return contexts_.size() - 1U;
}

void LanguageFabric::update_context(
    const std::span<const std::uint64_t> history,
    const std::uint64_t next_id
) {
    const std::size_t index = get_or_create_context(history);
    auto& context = contexts_[index];
    ++context.support;
    auto found = std::find_if(
        context.outcomes.begin(), context.outcomes.end(),
        [next_id](const LanguageOutcomeCount& outcome) {
            return outcome.token_id == next_id;
        }
    );
    if (found == context.outcomes.end()) {
        context.outcomes.push_back({next_id, 1ULL});
    } else {
        ++found->count;
    }
    ++stats_.contexts_updated;
}

void LanguageFabric::train_language_model(const std::string_view raw_corpus) {
    if (raw_corpus.empty()) {
        throw std::invalid_argument("RLF-5 language-model corpus must not be empty");
    }
    contexts_.clear();
    context_index_by_history_.clear();
    next_context_id_ = 1ULL;
    const auto tokens = encode(raw_corpus);
    std::vector<std::uint64_t> history;
    history.reserve(config_.maximum_context_order);
    for (const auto token : tokens) {
        update_context({}, token);
        const std::size_t maximum = std::min(
            history.size(), config_.maximum_context_order
        );
        for (std::size_t order = 1U; order <= maximum; ++order) {
            update_context(
                std::span<const std::uint64_t>(history).subspan(
                    history.size() - order, order
                ),
                token
            );
        }
        ++stats_.language_tokens_seen;
        const auto& bytes = lexeme_by_id(token).bytes;
        if (bytes.find('\n') != std::string::npos) {
            history.clear();
        } else {
            history.push_back(token);
            if (history.size() > config_.maximum_context_order) {
                history.erase(history.begin());
            }
        }
    }
}

LanguagePrediction LanguageFabric::predict_next(
    const std::span<const std::uint64_t> history
) const {
    LanguagePrediction prediction;
    const std::size_t maximum = std::min(
        history.size(), config_.maximum_context_order
    );
    const LanguageContext* selected = nullptr;
    for (std::size_t order = maximum; order > 0U; --order) {
        const auto suffix = history.subspan(history.size() - order, order);
        const auto found = context_index(suffix);
        if (!found.has_value()) {
            continue;
        }
        const auto& context = contexts_[*found];
        if (context.support < config_.minimum_context_support ||
            context.outcomes.empty()) {
            continue;
        }
        selected = &context;
        prediction.context_order = order;
        prediction.context_id = context.id;
        break;
    }
    if (selected == nullptr) {
        const auto found = context_index({});
        if (found.has_value() && !contexts_[*found].outcomes.empty()) {
            selected = &contexts_[*found];
            prediction.context_id = selected->id;
        }
    }
    if (selected == nullptr) {
        return prediction;
    }

    double total = 0.0;
    for (const auto& outcome : selected->outcomes) {
        total += static_cast<double>(outcome.count) + config_.smoothing;
    }
    prediction.outcomes.reserve(selected->outcomes.size());
    for (const auto& outcome : selected->outcomes) {
        prediction.outcomes.push_back({
            outcome.token_id,
            (static_cast<double>(outcome.count) + config_.smoothing) / total,
        });
    }
    std::sort(
        prediction.outcomes.begin(), prediction.outcomes.end(),
        [](const auto& left, const auto& right) {
            if (left.probability != right.probability) {
                return left.probability > right.probability;
            }
            return left.token_id < right.token_id;
        }
    );
    double entropy = 0.0;
    for (const auto& outcome : prediction.outcomes) {
        if (outcome.probability > 0.0) {
            entropy -= outcome.probability * std::log(outcome.probability);
        }
    }
    const double maximum_entropy = prediction.outcomes.size() > 1U
        ? std::log(static_cast<double>(prediction.outcomes.size()))
        : 0.0;
    prediction.uncertainty = maximum_entropy > 0.0
        ? std::clamp(entropy / maximum_entropy, 0.0, 1.0)
        : 0.0;
    return prediction;
}

std::string LanguageFabric::generate(
    const std::string_view prompt,
    const std::size_t maximum_tokens,
    const bool stop_at_newline
) const {
    if (maximum_tokens == 0U || maximum_tokens > config_.maximum_generation_tokens) {
        throw std::invalid_argument("invalid RLF-5 generation-token budget");
    }
    std::vector<std::uint64_t> tokens = encode(prompt);
    std::vector<std::uint64_t> history;
    const std::size_t copy = std::min(tokens.size(), config_.maximum_context_order);
    history.assign(tokens.end() - static_cast<std::ptrdiff_t>(copy), tokens.end());
    for (std::size_t step = 0U; step < maximum_tokens; ++step) {
        const auto prediction = predict_next(history);
        if (prediction.outcomes.empty()) {
            break;
        }
        const auto next = prediction.outcomes.front().token_id;
        tokens.push_back(next);
        const auto& bytes = lexeme_by_id(next).bytes;
        if (stop_at_newline && bytes.find('\n') != std::string::npos) {
            break;
        }
        history.push_back(next);
        if (history.size() > config_.maximum_context_order) {
            history.erase(history.begin());
        }
    }
    return decode(tokens);
}

double LanguageFabric::sequence_nll(const std::string_view text) const {
    const auto tokens = encode(text);
    if (tokens.empty()) {
        return 0.0;
    }
    std::vector<std::uint64_t> history;
    history.reserve(config_.maximum_context_order);
    double nll = 0.0;
    for (const auto token : tokens) {
        const auto prediction = predict_next(history);
        double probability = probability_floor;
        const auto found = std::find_if(
            prediction.outcomes.begin(), prediction.outcomes.end(),
            [token](const LanguagePredictionOutcome& outcome) {
                return outcome.token_id == token;
            }
        );
        if (found != prediction.outcomes.end()) {
            probability = std::max(found->probability, probability_floor);
        }
        nll -= std::log(probability);
        const auto& bytes = lexeme_by_id(token).bytes;
        if (bytes.find('\n') != std::string::npos) {
            history.clear();
        } else {
            history.push_back(token);
            if (history.size() > config_.maximum_context_order) {
                history.erase(history.begin());
            }
        }
    }
    return nll / static_cast<double>(tokens.size());
}

std::vector<std::pair<LanguageRole, std::string_view>>
LanguageFabric::frame_values(const LanguageFrame& frame) {
    std::vector<std::pair<LanguageRole, std::string_view>> result;
    result.reserve(all_roles.size());
    for (const auto role : all_roles) {
        const auto value = frame_value(frame, role);
        if (!value.empty()) {
            result.emplace_back(role, value);
        }
    }
    return result;
}

std::string_view LanguageFabric::frame_value(
    const LanguageFrame& frame,
    const LanguageRole role
) {
    switch (role) {
        case LanguageRole::predicate:
            return frame.predicate;
        case LanguageRole::agent:
            return frame.agent;
        case LanguageRole::patient:
            return frame.patient;
        case LanguageRole::agent_attribute:
            return frame.agent_attribute;
        case LanguageRole::patient_attribute:
            return frame.patient_attribute;
        case LanguageRole::location:
            return frame.location;
    }
    return {};
}

void LanguageFabric::set_frame_value(
    LanguageFrame& frame,
    const LanguageRole role,
    std::string value
) {
    switch (role) {
        case LanguageRole::predicate:
            frame.predicate = std::move(value);
            break;
        case LanguageRole::agent:
            frame.agent = std::move(value);
            break;
        case LanguageRole::patient:
            frame.patient = std::move(value);
            break;
        case LanguageRole::agent_attribute:
            frame.agent_attribute = std::move(value);
            break;
        case LanguageRole::patient_attribute:
            frame.patient_attribute = std::move(value);
            break;
        case LanguageRole::location:
            frame.location = std::move(value);
            break;
    }
}

const LanguageConcept* LanguageFabric::find_concept(
    const LanguageRole role,
    const std::string_view value
) const {
    const auto found = concept_index_.find({role, std::string(value)});
    if (found == concept_index_.end()) {
        return nullptr;
    }
    return &concepts_[found->second];
}

double LanguageFabric::lexical_score(
    const LanguageRole role,
    const std::uint64_t token_id,
    const std::string_view value
) const {
    const auto* selected = find_concept(role, value);
    if (selected == nullptr) {
        return 0.0;
    }
    const auto found = std::find_if(
        selected->surfaces.begin(), selected->surfaces.end(),
        [token_id](const LanguageSurfaceCount& surface) {
            return surface.token_id == token_id;
        }
    );
    if (found != selected->surfaces.end()) {
        return found->association;
    }
    const auto& lexeme = lexeme_by_id(token_id);
    if (!token_is_content(lexeme.bytes)) {
        return 0.0;
    }
    return 0.10 * lexeme.key.similarity(selected->key);
}

std::optional<std::pair<std::string, double>> LanguageFabric::best_value(
    const LanguageRole role,
    const std::uint64_t token_id
) const {
    const LanguageConcept* best = nullptr;
    double best_score = 0.0;
    for (const auto& candidate : concepts_) {
        if (candidate.role != role) {
            continue;
        }
        const double score = lexical_score(role, token_id, candidate.value);
        if (score > best_score ||
            (score == best_score && best != nullptr &&
             candidate.value < best->value)) {
            best = &candidate;
            best_score = score;
        }
    }
    if (best == nullptr || best_score < config_.minimum_lexical_score) {
        return std::nullopt;
    }
    return std::pair{best->value, best_score};
}

std::uint64_t LanguageFabric::best_surface(
    const LanguageRole role,
    const std::string_view value,
    const std::uint8_t requested_form
) const {
    const auto* selected = find_concept(role, value);
    if (selected == nullptr || selected->surfaces.empty()) {
        return 0ULL;
    }
    const LanguageSurfaceCount* best = nullptr;
    bool best_form_match = false;
    for (const auto& surface : selected->surfaces) {
        const auto& lexeme = lexeme_by_id(surface.token_id);
        const bool form_match = role != LanguageRole::predicate ||
            surface_form(lexeme.bytes) == requested_form;
        if (best == nullptr || (form_match && !best_form_match) ||
            (form_match == best_form_match &&
             (surface.association > best->association ||
              (surface.association == best->association &&
               (surface.count > best->count ||
                (surface.count == best->count &&
                 surface.token_id < best->token_id)))))) {
            best = &surface;
            best_form_match = form_match;
        }
    }
    return best == nullptr ? 0ULL : best->token_id;
}

void LanguageFabric::train_semantics(
    const std::span<const LanguageSupervisedExample> examples
) {
    if (examples.empty()) {
        throw std::invalid_argument("RLF-5 semantic examples must not be empty");
    }
    concepts_.clear();
    constructions_.clear();
    concept_index_.clear();
    next_concept_id_ = 1ULL;
    next_construction_id_ = 1ULL;

    struct TokenizedExample final {
        std::vector<std::uint64_t> tokens;
        LanguageFrame frame;
    };
    std::vector<TokenizedExample> tokenized;
    tokenized.reserve(examples.size());
    std::unordered_map<std::uint64_t, std::uint64_t> token_totals;
    std::unordered_map<ConceptKey, std::uint64_t, ConceptKeyHash> value_support;
    std::unordered_map<ConceptKey,
        std::unordered_map<std::uint64_t, std::uint64_t>, ConceptKeyHash> cooccurrence;

    for (const auto& example : examples) {
        auto tokens = encode(example.text);
        if (tokens.empty()) {
            continue;
        }
        std::unordered_set<std::uint64_t> unique_content_tokens;
        for (const auto token : tokens) {
            if (token_is_content(lexeme_by_id(token).bytes)) {
                unique_content_tokens.insert(token);
            }
        }
        for (const auto token : unique_content_tokens) {
            ++token_totals[token];
        }
        for (const auto& [role, value] : frame_values(example.frame)) {
            ConceptKey key{role, std::string(value)};
            ++value_support[key];
            for (const auto token : unique_content_tokens) {
                ++cooccurrence[key][token];
            }
        }
        tokenized.push_back({std::move(tokens), example.frame});
    }
    if (value_support.size() > config_.maximum_semantic_values) {
        throw std::runtime_error("RLF-5 semantic value capacity exceeded");
    }

    std::vector<ConceptKey> keys;
    keys.reserve(value_support.size());
    for (const auto& [key, support] : value_support) {
        static_cast<void>(support);
        keys.push_back(key);
    }
    std::sort(keys.begin(), keys.end(), [](const ConceptKey& left, const ConceptKey& right) {
        if (left.role != right.role) {
            return role_index(left.role) < role_index(right.role);
        }
        return left.value < right.value;
    });

    for (const auto& key : keys) {
        std::vector<LanguageSurfaceCount> surfaces;
        const double value_total = static_cast<double>(value_support[key]);
        for (const auto& [token, count] : cooccurrence[key]) {
            const double token_total = static_cast<double>(token_totals[token]);
            const double joint = static_cast<double>(count);
            const double precision = token_total > 0.0 ? joint / token_total : 0.0;
            const double recall = value_total > 0.0 ? joint / value_total : 0.0;
            // Semantic surfaces must be discriminative, not merely frequent.
            // Precision suppresses generic words that co-occur with every role,
            // while sqrt(recall) still rewards aliases observed repeatedly.
            const double association = precision > 0.0 && recall > 0.0
                ? precision * std::sqrt(recall)
                : 0.0;
            if (association >= config_.minimum_lexical_score * 0.5) {
                surfaces.push_back({token, count, association});
            }
        }
        std::sort(surfaces.begin(), surfaces.end(), [](const auto& left, const auto& right) {
            if (left.association != right.association) {
                return left.association > right.association;
            }
            if (left.count != right.count) {
                return left.count > right.count;
            }
            return left.token_id < right.token_id;
        });
        if (surfaces.size() > config_.maximum_surfaces_per_concept) {
            surfaces.resize(config_.maximum_surfaces_per_concept);
        }
        if (surfaces.empty()) {
            continue;
        }
        std::vector<PhaseVector> vectors;
        std::vector<float> weights;
        vectors.reserve(surfaces.size());
        weights.reserve(surfaces.size());
        for (const auto& surface : surfaces) {
            vectors.push_back(lexeme_by_id(surface.token_id).key);
            weights.push_back(static_cast<float>(
                std::max(surface.association * static_cast<double>(surface.count), 1.0e-4)
            ));
        }
        LanguageConcept concept_value;
        concept_value.id = next_concept_id_++;
        concept_value.role = key.role;
        concept_value.value = key.value;
        concept_value.key = PhaseVector::weighted_circular_average(vectors, weights);
        concept_value.support = value_support[key];
        concept_value.surfaces = std::move(surfaces);
        concept_index_[key] = concepts_.size();
        concepts_.push_back(std::move(concept_value));
        ++stats_.concepts_created;
    }

    struct Assignment final {
        double score{};
        LanguageRole role{LanguageRole::predicate};
        std::size_t position{};
    };
    std::unordered_map<std::string, std::size_t> construction_by_signature;
    std::array<std::uint64_t, 7U> act_totals{};
    for (const auto& example : tokenized) {
        ++act_totals[act_index(example.frame.act)];
        std::vector<Assignment> candidates;
        for (const auto& [role, value] : frame_values(example.frame)) {
            for (std::size_t position = 0U; position < example.tokens.size(); ++position) {
                const double score = lexical_score(role, example.tokens[position], value);
                if (score >= config_.minimum_lexical_score) {
                    candidates.push_back({score, role, position});
                }
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
            if (left.score != right.score) {
                return left.score > right.score;
            }
            if (left.role != right.role) {
                return role_index(left.role) < role_index(right.role);
            }
            return left.position < right.position;
        });
        std::array<bool, all_roles.size()> assigned_roles{};
        std::vector<std::optional<LanguageRole>> assigned_positions(example.tokens.size());
        for (const auto& candidate : candidates) {
            const auto role = role_index(candidate.role);
            if (assigned_roles[role] || assigned_positions[candidate.position].has_value()) {
                continue;
            }
            assigned_roles[role] = true;
            assigned_positions[candidate.position] = candidate.role;
        }

        std::vector<LanguagePatternItem> pattern;
        pattern.reserve(example.tokens.size());
        for (std::size_t position = 0U; position < example.tokens.size(); ++position) {
            if (assigned_positions[position].has_value()) {
                pattern.push_back({
                    LanguagePatternKind::slot,
                    0ULL,
                    *assigned_positions[position],
                    *assigned_positions[position] == LanguageRole::predicate
                        ? surface_form(lexeme_by_id(example.tokens[position]).bytes)
                        : static_cast<std::uint8_t>(0U),
                });
            } else {
                pattern.push_back({
                    LanguagePatternKind::literal,
                    example.tokens[position],
                    LanguageRole::predicate,
                    0U,
                });
            }
        }
        const std::string signature = construction_signature(example.frame.act, pattern);
        const auto found = construction_by_signature.find(signature);
        if (found == construction_by_signature.end()) {
            if (constructions_.size() >= config_.maximum_constructions) {
                continue;
            }
            LanguageConstruction construction;
            construction.id = next_construction_id_++;
            construction.act = example.frame.act;
            construction.pattern = std::move(pattern);
            construction.support = 1ULL;
            constructions_.push_back(std::move(construction));
            construction_by_signature[signature] = constructions_.size() - 1U;
        } else {
            ++constructions_[found->second].support;
        }
    }

    constructions_.erase(
        std::remove_if(
            constructions_.begin(), constructions_.end(),
            [this](const LanguageConstruction& construction) {
                return construction.support < config_.minimum_construction_support;
            }
        ),
        constructions_.end()
    );
    for (auto& construction : constructions_) {
        const auto total = act_totals[act_index(construction.act)];
        construction.confidence = total > 0ULL
            ? static_cast<double>(construction.support) / static_cast<double>(total)
            : 0.0;
    }
    std::sort(constructions_.begin(), constructions_.end(), [](const auto& left, const auto& right) {
        if (left.act != right.act) {
            return act_index(left.act) < act_index(right.act);
        }
        if (left.support != right.support) {
            return left.support > right.support;
        }
        return left.id < right.id;
    });
    stats_.constructions_created += constructions_.size();
    stats_.supervised_examples_seen += tokenized.size();
}

LanguageParse LanguageFabric::parse(const std::string_view text) const {
    LanguageParse result;
    result.tokens = encode(text);
    ++const_cast<LanguageFabricStats&>(stats_).parse_queries;
    double best_score = -std::numeric_limits<double>::infinity();
    double second_score = -std::numeric_limits<double>::infinity();
    for (const auto& construction : constructions_) {
        if (construction.pattern.size() != result.tokens.size()) {
            continue;
        }
        LanguageFrame frame;
        frame.act = construction.act;
        double score = config_.construction_support_weight *
            std::log1p(static_cast<double>(construction.support));
        bool matched = true;
        for (std::size_t position = 0U; position < result.tokens.size(); ++position) {
            const auto& item = construction.pattern[position];
            const auto token = result.tokens[position];
            if (item.kind == LanguagePatternKind::literal) {
                if (token != item.token_id) {
                    matched = false;
                    break;
                }
                score += config_.literal_match_weight;
            } else {
                const auto value = best_value(item.role, token);
                if (!value.has_value()) {
                    matched = false;
                    break;
                }
                set_frame_value(frame, item.role, value->first);
                score += config_.slot_match_weight * value->second;
            }
        }
        if (!matched) {
            continue;
        }
        if (score > best_score ||
            (score == best_score && construction.id < result.construction_id)) {
            second_score = best_score;
            best_score = score;
            result.success = true;
            result.frame = std::move(frame);
            result.construction_id = construction.id;
            result.score = score;
        } else if (score > second_score) {
            second_score = score;
        }
    }
    if (result.success) {
        if (!std::isfinite(second_score)) {
            result.uncertainty = 0.0;
        } else {
            const double margin = best_score - second_score;
            result.uncertainty = std::exp(-std::max(0.0, margin));
        }
    }
    return result;
}

LanguageGeneration LanguageFabric::generate_frame(const LanguageFrame& frame) const {
    LanguageGeneration result;
    ++const_cast<LanguageFabricStats&>(stats_).generation_queries;
    const LanguageConstruction* best = nullptr;
    for (const auto& construction : constructions_) {
        if (construction.act != frame.act) {
            continue;
        }
        bool usable = true;
        for (const auto& item : construction.pattern) {
            if (item.kind == LanguagePatternKind::slot &&
                frame_value(frame, item.role).empty()) {
                usable = false;
                break;
            }
        }
        if (!usable) {
            continue;
        }
        if (best == nullptr || construction.support > best->support ||
            (construction.support == best->support &&
             construction.id < best->id)) {
            best = &construction;
        }
    }
    if (best == nullptr) {
        return result;
    }
    result.tokens.reserve(best->pattern.size());
    for (const auto& item : best->pattern) {
        if (item.kind == LanguagePatternKind::literal) {
            result.tokens.push_back(item.token_id);
        } else {
            const auto token = best_surface(
                item.role, frame_value(frame, item.role), item.surface_form
            );
            if (token == 0ULL) {
                return {};
            }
            result.tokens.push_back(token);
        }
    }
    result.success = true;
    result.construction_id = best->id;
    result.text = decode(result.tokens);
    return result;
}

LanguageFrame LanguageFabric::answer_frame_for(
    const LanguageFrame& fact,
    const LanguageFrame& query
) const {
    LanguageFrame answer_frame;
    switch (query.act) {
        case LanguageAct::query_agent:
            answer_frame.act = LanguageAct::answer_agent;
            answer_frame.agent = fact.agent;
            answer_frame.agent_attribute = fact.agent_attribute;
            break;
        case LanguageAct::query_patient:
            answer_frame.act = LanguageAct::answer_patient;
            answer_frame.patient = fact.patient;
            answer_frame.patient_attribute = fact.patient_attribute;
            break;
        case LanguageAct::query_location:
            answer_frame.act = LanguageAct::answer_location;
            answer_frame.location = fact.location;
            break;
        default:
            break;
    }
    return answer_frame;
}

LanguageAnswer LanguageFabric::answer(
    const std::span<const std::string> context_sentences,
    const std::string_view question
) const {
    LanguageAnswer result;
    ++const_cast<LanguageFabricStats&>(stats_).answer_queries;
    const auto query = parse(question);
    if (!query.success ||
        (query.frame.act != LanguageAct::query_agent &&
         query.frame.act != LanguageAct::query_patient &&
         query.frame.act != LanguageAct::query_location)) {
        return result;
    }
    result.query = query.frame;
    for (const auto& sentence : context_sentences) {
        const auto parsed = parse(sentence);
        if (!parsed.success || parsed.frame.act != LanguageAct::statement) {
            continue;
        }
        if (!frame_matches_query(parsed.frame, query.frame)) {
            continue;
        }
        result.matched_fact = parsed.frame;
        result.answer_frame = answer_frame_for(parsed.frame, query.frame);
        const auto generated = generate_frame(result.answer_frame);
        if (generated.success) {
            result.success = true;
            result.text = generated.text;
            return result;
        }
    }
    return result;
}

std::optional<std::uint64_t> LanguageFabric::lexeme_id(
    const std::string_view bytes
) const {
    const auto found = lexeme_id_by_bytes_.find(std::string(bytes));
    if (found == lexeme_id_by_bytes_.end()) {
        return std::nullopt;
    }
    return found->second;
}

const LanguageLexeme& LanguageFabric::lexeme_by_id(
    const std::uint64_t id
) const {
    const auto found = lexeme_index_by_id_.find(id);
    if (found == lexeme_index_by_id_.end()) {
        throw std::out_of_range("unknown RLF-5 lexeme ID");
    }
    return lexemes_[found->second];
}

std::size_t LanguageFabric::estimated_storage_bytes() const noexcept {
    std::size_t bytes = sizeof(*this);
    for (const auto& lexeme : lexemes_) {
        bytes += sizeof(lexeme) + lexeme.bytes.size() +
            lexeme.key.size() * sizeof(float);
    }
    bytes += merges_.size() * sizeof(LanguageMerge);
    for (const auto& context : contexts_) {
        bytes += sizeof(context) + context.history.size() * sizeof(std::uint64_t) +
            context.outcomes.size() * sizeof(LanguageOutcomeCount);
    }
    for (const auto& concept_value : concepts_) {
        bytes += sizeof(concept_value) + concept_value.value.size() +
            concept_value.key.size() * sizeof(float) +
            concept_value.surfaces.size() * sizeof(LanguageSurfaceCount);
    }
    for (const auto& construction : constructions_) {
        bytes += sizeof(construction) +
            construction.pattern.size() * sizeof(LanguagePatternItem);
    }
    return bytes;
}

std::uint64_t LanguageFabric::deterministic_hash() const noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, seed_);
    hash_u64(hash, config_.phase_dimension);
    for (const auto& lexeme : lexemes_) {
        hash_u64(hash, lexeme.id);
        hash_string(hash, lexeme.bytes);
        hash_u64(hash, lexeme.support);
        for (const auto angle : lexeme.key.angles()) {
            hash_u64(hash, std::bit_cast<std::uint32_t>(angle));
        }
    }
    for (const auto& merge : merges_) {
        hash_u64(hash, merge.left_id);
        hash_u64(hash, merge.right_id);
        hash_u64(hash, merge.result_id);
        hash_u64(hash, merge.support);
        hash_u64(hash, static_cast<std::uint64_t>(merge.description_gain));
    }
    for (const auto& context : contexts_) {
        hash_u64(hash, context.id);
        for (const auto id : context.history) {
            hash_u64(hash, id);
        }
        hash_u64(hash, context.support);
        for (const auto& outcome : context.outcomes) {
            hash_u64(hash, outcome.token_id);
            hash_u64(hash, outcome.count);
        }
    }
    for (const auto& concept_value : concepts_) {
        hash_u64(hash, concept_value.id);
        hash_u64(hash, static_cast<std::uint64_t>(concept_value.role));
        hash_string(hash, concept_value.value);
        hash_u64(hash, concept_value.support);
        for (const auto& surface : concept_value.surfaces) {
            hash_u64(hash, surface.token_id);
            hash_u64(hash, surface.count);
            hash_double(hash, surface.association);
        }
    }
    for (const auto& construction : constructions_) {
        hash_u64(hash, construction.id);
        hash_u64(hash, static_cast<std::uint64_t>(construction.act));
        hash_u64(hash, construction.support);
        hash_double(hash, construction.confidence);
        for (const auto& item : construction.pattern) {
            hash_u64(hash, static_cast<std::uint64_t>(item.kind));
            hash_u64(hash, item.token_id);
            hash_u64(hash, static_cast<std::uint64_t>(item.role));
            hash_u64(hash, item.surface_form);
        }
    }
    return hash;
}

LanguageFabricSnapshot LanguageFabric::snapshot() const {
    return {
        config_, seed_, next_lexeme_id_, next_context_id_, next_concept_id_,
        next_construction_id_, lexemes_, merges_, contexts_, concepts_,
        constructions_, stats_,
    };
}

LanguageFabric LanguageFabric::from_snapshot(LanguageFabricSnapshot snapshot) {
    LanguageFabric fabric(snapshot.config, snapshot.seed);
    fabric.next_lexeme_id_ = snapshot.next_lexeme_id;
    fabric.next_context_id_ = snapshot.next_context_id;
    fabric.next_concept_id_ = snapshot.next_concept_id;
    fabric.next_construction_id_ = snapshot.next_construction_id;
    fabric.lexemes_ = std::move(snapshot.lexemes);
    fabric.merges_ = std::move(snapshot.merges);
    fabric.contexts_ = std::move(snapshot.contexts);
    fabric.concepts_ = std::move(snapshot.concepts);
    fabric.constructions_ = std::move(snapshot.constructions);
    fabric.stats_ = snapshot.stats;
    fabric.rebuild_indices();
    fabric.validate_snapshot();
    return fabric;
}

void LanguageFabric::rebuild_indices() {
    lexeme_index_by_id_.clear();
    lexeme_id_by_bytes_.clear();
    for (std::size_t index = 0U; index < lexemes_.size(); ++index) {
        lexeme_index_by_id_[lexemes_[index].id] = index;
        lexeme_id_by_bytes_[lexemes_[index].bytes] = lexemes_[index].id;
    }
    context_index_by_history_.clear();
    for (std::size_t index = 0U; index < contexts_.size(); ++index) {
        context_index_by_history_[contexts_[index].history] = index;
    }
    concept_index_.clear();
    for (std::size_t index = 0U; index < concepts_.size(); ++index) {
        concept_index_[{concepts_[index].role, concepts_[index].value}] = index;
    }
}

void LanguageFabric::validate_snapshot() const {
    validate_config(config_);
    if (lexemes_.size() < base_byte_lexemes ||
        lexemes_.size() > config_.maximum_lexemes ||
        merges_.size() > config_.maximum_merges ||
        contexts_.size() > config_.maximum_contexts ||
        constructions_.size() > config_.maximum_constructions ||
        concepts_.size() > config_.maximum_semantic_values) {
        throw std::runtime_error("RLF-5 snapshot exceeds configured bounds");
    }
    std::unordered_set<std::uint64_t> lexeme_ids;
    std::unordered_set<std::string> lexeme_bytes;
    for (const auto& lexeme : lexemes_) {
        if (lexeme.id == 0ULL || lexeme.bytes.empty() ||
            lexeme.key.size() != config_.phase_dimension ||
            !lexeme_ids.insert(lexeme.id).second ||
            !lexeme_bytes.insert(lexeme.bytes).second) {
            throw std::runtime_error("invalid RLF-5 lexeme snapshot");
        }
    }
    for (const auto& merge : merges_) {
        if (!lexeme_ids.contains(merge.left_id) ||
            !lexeme_ids.contains(merge.right_id) ||
            !lexeme_ids.contains(merge.result_id)) {
            throw std::runtime_error("RLF-5 merge references unknown lexeme");
        }
    }
    std::unordered_set<std::uint64_t> context_ids;
    std::unordered_set<std::vector<std::uint64_t>, VectorHash> context_histories;
    for (const auto& context : contexts_) {
        if (context.id == 0ULL ||
            context.history.size() > config_.maximum_context_order ||
            !context_ids.insert(context.id).second ||
            !context_histories.insert(context.history).second) {
            throw std::runtime_error("invalid RLF-5 context snapshot");
        }
        for (const auto token : context.history) {
            if (!lexeme_ids.contains(token)) {
                throw std::runtime_error("RLF-5 context references unknown lexeme");
            }
        }
        std::unordered_set<std::uint64_t> outcome_tokens;
        for (const auto& outcome : context.outcomes) {
            if (!lexeme_ids.contains(outcome.token_id) || outcome.count == 0ULL ||
                !outcome_tokens.insert(outcome.token_id).second) {
                throw std::runtime_error("invalid RLF-5 context outcome");
            }
        }
    }
    std::unordered_set<std::uint64_t> concept_ids;
    std::unordered_set<std::string> concept_keys;
    for (const auto& concept_value : concepts_) {
        const std::string key = std::to_string(
            static_cast<unsigned int>(concept_value.role)
        ) + "|" + concept_value.value;
        if (concept_value.id == 0ULL || concept_value.value.empty() ||
            concept_value.key.size() != config_.phase_dimension ||
            !concept_ids.insert(concept_value.id).second ||
            !concept_keys.insert(key).second) {
            throw std::runtime_error("invalid RLF-5 concept snapshot");
        }
        std::unordered_set<std::uint64_t> surface_tokens;
        for (const auto& surface : concept_value.surfaces) {
            if (!lexeme_ids.contains(surface.token_id) || surface.count == 0ULL ||
                !std::isfinite(surface.association) ||
                surface.association < 0.0 ||
                !surface_tokens.insert(surface.token_id).second) {
                throw std::runtime_error("invalid RLF-5 concept surface");
            }
        }
    }
    std::unordered_set<std::uint64_t> construction_ids;
    for (const auto& construction : constructions_) {
        if (construction.id == 0ULL || construction.pattern.empty() ||
            !construction_ids.insert(construction.id).second ||
            !std::isfinite(construction.confidence) ||
            construction.confidence < 0.0 || construction.confidence > 1.0) {
            throw std::runtime_error("invalid RLF-5 construction snapshot");
        }
        for (const auto& item : construction.pattern) {
            if (item.kind == LanguagePatternKind::literal &&
                !lexeme_ids.contains(item.token_id)) {
                throw std::runtime_error(
                    "RLF-5 construction references unknown literal"
                );
            }
        }
    }
    const auto maximum_id = [](const auto& values) {
        std::uint64_t result = 0ULL;
        for (const auto& value : values) {
            result = std::max(result, value.id);
        }
        return result;
    };
    if (next_lexeme_id_ <= maximum_id(lexemes_) ||
        next_context_id_ <= maximum_id(contexts_) ||
        next_concept_id_ <= maximum_id(concepts_) ||
        next_construction_id_ <= maximum_id(constructions_)) {
        throw std::runtime_error("invalid RLF-5 next stable ID");
    }
}

std::string_view rlf5_architecture_name() noexcept {
    return "Compositional Language Fabric";
}

std::string_view language_act_name(const LanguageAct act) noexcept {
    switch (act) {
        case LanguageAct::statement:
            return "statement";
        case LanguageAct::query_agent:
            return "query_agent";
        case LanguageAct::query_patient:
            return "query_patient";
        case LanguageAct::query_location:
            return "query_location";
        case LanguageAct::answer_agent:
            return "answer_agent";
        case LanguageAct::answer_patient:
            return "answer_patient";
        case LanguageAct::answer_location:
            return "answer_location";
    }
    return "unknown";
}

std::string_view language_role_name(const LanguageRole role) noexcept {
    switch (role) {
        case LanguageRole::predicate:
            return "predicate";
        case LanguageRole::agent:
            return "agent";
        case LanguageRole::patient:
            return "patient";
        case LanguageRole::agent_attribute:
            return "agent_attribute";
        case LanguageRole::patient_attribute:
            return "patient_attribute";
        case LanguageRole::location:
            return "location";
    }
    return "unknown";
}

}  // namespace rlf::core
