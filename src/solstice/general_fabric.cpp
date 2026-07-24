#include "rlf/solstice/general_fabric.hpp"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace rlf::solstice {
namespace {

constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;

[[nodiscard]] bool indexed_preference_duplicates_from_environment() {
    const char* const value = std::getenv("RLF_PREFERENCE_DUPLICATE_POLICY");
    if (value == nullptr || std::string_view(value).empty() ||
        std::string_view(value) == "indexed") {
        return true;
    }
    if (std::string_view(value) == "linear") {
        return false;
    }
    throw std::invalid_argument(
        "RLF_PREFERENCE_DUPLICATE_POLICY must be indexed or linear"
    );
}

[[nodiscard]] bool indexed_active_learning_duplicates_from_environment() {
    const char* const value = std::getenv("RLF_ACTIVE_LEARNING_DUPLICATE_POLICY");
    if (value == nullptr || std::string_view(value).empty() ||
        std::string_view(value) == "indexed") {
        return true;
    }
    if (std::string_view(value) == "linear") {
        return false;
    }
    throw std::invalid_argument(
        "RLF_ACTIVE_LEARNING_DUPLICATE_POLICY must be indexed or linear"
    );
}

[[nodiscard]] bool indexed_instruction_duplicates_from_environment() {
    const char* const value = std::getenv("RLF_INSTRUCTION_DUPLICATE_POLICY");
    if (value == nullptr || std::string_view(value).empty() ||
        std::string_view(value) == "indexed") {
        return true;
    }
    if (std::string_view(value) == "retrieval") {
        return false;
    }
    throw std::invalid_argument(
        "RLF_INSTRUCTION_DUPLICATE_POLICY must be indexed or retrieval"
    );
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
        const unsigned char byte = static_cast<unsigned char>(character);
        hash ^= byte;
        hash *= fnv_prime;
    }
}

void hash_double(std::uint64_t& hash, const double value) noexcept {
    hash_u64(hash, std::bit_cast<std::uint64_t>(value));
}

[[nodiscard]] std::uint64_t hash_text(const std::string_view value) noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_string(hash, value);
    return hash;
}

[[nodiscard]] std::uint64_t preference_key(
    const std::string_view prompt,
    const std::string_view chosen,
    const std::string_view rejected
) noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_string(hash, prompt);
    hash_string(hash, chosen);
    hash_string(hash, rejected);
    return hash;
}

[[nodiscard]] std::uint64_t active_learning_key(
    const std::string_view prompt,
    const std::string_view grounding
) noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_string(hash, prompt);
    hash_string(hash, grounding);
    return hash;
}

[[nodiscard]] std::uint64_t instruction_key(
    const std::string_view task,
    const std::string_view domain,
    const std::string_view prompt,
    const std::string_view rationale,
    const std::string_view response
) noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_string(hash, task);
    hash_string(hash, domain);
    hash_string(hash, prompt);
    hash_string(hash, rationale);
    hash_string(hash, response);
    return hash;
}

[[nodiscard]] std::vector<std::string> words(const std::string_view text) {
    std::vector<std::string> result;
    std::string current;
    for (const char character : text) {
        const unsigned char byte = static_cast<unsigned char>(character);
        if (std::isalnum(byte) != 0 || byte >= 128U || byte == '_') {
            current.push_back(static_cast<char>(std::tolower(byte)));
        } else if (!current.empty()) {
            result.push_back(std::move(current));
            current.clear();
        }
    }
    if (!current.empty()) {
        result.push_back(std::move(current));
    }
    return result;
}

[[nodiscard]] double clamp01(const double value) noexcept {
    return std::clamp(value, 0.0, 1.0);
}

[[nodiscard]] bool valid_config(const GeneralFabricConfig& config) noexcept {
    return config.maximum_demonstrations > 0U &&
        config.maximum_preferences > 0U &&
        config.maximum_active_learning_items > 0U &&
        config.maximum_concepts_per_item > 0U &&
        config.maximum_retrieval_candidates > 0U &&
        config.maximum_retrieved_demonstrations > 0U &&
        config.maximum_context_characters > 0U &&
        config.deliberation_candidates > 0U &&
        std::isfinite(config.minimum_retrieval_similarity) &&
        config.minimum_retrieval_similarity >= 0.0 &&
        config.minimum_retrieval_similarity <= 1.0 &&
        std::isfinite(config.exact_task_bonus) && config.exact_task_bonus >= 0.0 &&
        std::isfinite(config.exact_domain_bonus) && config.exact_domain_bonus >= 0.0 &&
        std::isfinite(config.preference_weight) && config.preference_weight >= 0.0 &&
        std::isfinite(config.direct_recall_threshold) &&
        config.direct_recall_threshold >= 0.0 && config.direct_recall_threshold <= 1.0 &&
        std::isfinite(config.active_learning_uncertainty) &&
        config.active_learning_uncertainty >= 0.0 &&
        config.active_learning_uncertainty <= 1.0;
}

void append_bounded(
    std::string& output,
    const std::string_view value,
    const std::size_t limit
) {
    if (output.size() >= limit || value.empty()) {
        return;
    }
    const std::size_t remaining = limit - output.size();
    output.append(value.substr(0U, remaining));
}

[[nodiscard]] double lexical_diversity(const std::string_view text) {
    const std::vector<std::string> tokens = words(text);
    if (tokens.empty()) {
        return 0.0;
    }
    std::unordered_set<std::string> unique(tokens.begin(), tokens.end());
    return static_cast<double>(unique.size()) / static_cast<double>(tokens.size());
}

}  // namespace

GeneralInstructionFabric::GeneralInstructionFabric(GeneralFabricConfig config)
    : config_(std::move(config)),
      indexed_preference_duplicates_(
          indexed_preference_duplicates_from_environment()
      ),
      indexed_active_learning_duplicates_(
          indexed_active_learning_duplicates_from_environment()
      ),
      indexed_instruction_duplicates_(
          indexed_instruction_duplicates_from_environment()
      ) {
    if (!valid_config(config_)) {
        throw std::invalid_argument("invalid Solstice general instruction configuration");
    }
    demonstrations_.reserve(
        std::min<std::size_t>(config_.maximum_demonstrations, 65'536U)
    );
    preferences_.reserve(
        std::min<std::size_t>(config_.maximum_preferences, 16'384U)
    );
    active_learning_items_.reserve(
        std::min<std::size_t>(config_.maximum_active_learning_items, 8'192U)
    );
}

std::string GeneralInstructionFabric::normalize_label(const std::string_view value) {
    std::string normalized;
    normalized.reserve(value.size());
    bool pending_separator = false;
    for (const char character : value) {
        const unsigned char byte = static_cast<unsigned char>(character);
        if (std::isalnum(byte) != 0 || byte >= 128U) {
            if (pending_separator && !normalized.empty()) {
                normalized.push_back('_');
            }
            normalized.push_back(static_cast<char>(std::tolower(byte)));
            pending_separator = false;
        } else {
            pending_separator = true;
        }
    }
    return normalized;
}

std::vector<std::uint64_t> GeneralInstructionFabric::concept_hashes(
    const std::string_view text,
    const std::size_t maximum_concepts
) {
    const std::vector<std::string> tokens = words(text);
    std::vector<std::uint64_t> hashes;
    hashes.reserve(std::min(tokens.size() * 2U, maximum_concepts));
    for (std::size_t index = 0U; index < tokens.size(); ++index) {
        const std::string& token = tokens[index];
        if (token.size() >= 2U) {
            hashes.push_back(hash_text(token));
        }
        if (index != 0U) {
            std::string pair = tokens[index - 1U];
            pair.push_back(' ');
            pair += token;
            hashes.push_back(hash_text(pair));
        }
        if (hashes.size() >= maximum_concepts) {
            break;
        }
    }
    std::sort(hashes.begin(), hashes.end());
    hashes.erase(std::unique(hashes.begin(), hashes.end()), hashes.end());
    if (hashes.size() > maximum_concepts) {
        hashes.resize(maximum_concepts);
    }
    return hashes;
}

SemanticSignature GeneralInstructionFabric::make_signature(const std::string_view text) {
    SemanticSignature signature;
    const std::vector<std::uint64_t> concepts = concept_hashes(text, 512U);
    for (const std::uint64_t feature_hash : concepts) {
        std::uint64_t state = feature_hash;
        for (unsigned int probe = 0U; probe < 4U; ++probe) {
            state ^= state >> 12U;
            state ^= state << 25U;
            state ^= state >> 27U;
            state *= 2'685'821'657'736'338'717ULL;
            const std::size_t bit = static_cast<std::size_t>(state & 255ULL);
            signature.words[bit / 64U] |= 1ULL << (bit % 64U);
        }
    }
    return signature;
}

double GeneralInstructionFabric::signature_similarity(
    const SemanticSignature& left,
    const SemanticSignature& right
) noexcept {
    std::uint32_t intersection = 0U;
    std::uint32_t union_count = 0U;
    for (std::size_t index = 0U; index < left.words.size(); ++index) {
        intersection += static_cast<std::uint32_t>(
            std::popcount(left.words[index] & right.words[index])
        );
        union_count += static_cast<std::uint32_t>(
            std::popcount(left.words[index] | right.words[index])
        );
    }
    return union_count == 0U
        ? 0.0
        : static_cast<double>(intersection) / static_cast<double>(union_count);
}

std::array<std::uint64_t, 8U> GeneralInstructionFabric::band_keys(
    const SemanticSignature& signature
) noexcept {
    std::array<std::uint64_t, 8U> keys{};
    for (std::size_t word_index = 0U; word_index < signature.words.size(); ++word_index) {
        const std::uint64_t value = signature.words[word_index];
        const std::size_t low_band = word_index * 2U;
        const std::size_t high_band = low_band + 1U;
        keys[low_band] = (static_cast<std::uint64_t>(low_band) << 32U) |
            (value & 0xFFFF'FFFFULL);
        keys[high_band] = (static_cast<std::uint64_t>(high_band) << 32U) |
            (value >> 32U);
    }
    return keys;
}

void GeneralInstructionFabric::index_demonstration(const std::size_t index) {
    const auto keys = band_keys(demonstrations_[index].signature);
    for (const std::uint64_t key : keys) {
        band_postings_[key].push_back(index);
    }
}

void GeneralInstructionFabric::rebuild_index() {
    band_postings_.clear();
    for (std::size_t index = 0U; index < demonstrations_.size(); ++index) {
        index_demonstration(index);
    }
    rebuild_preference_index();
    rebuild_active_learning_index();
    rebuild_instruction_index();
}

void GeneralInstructionFabric::rebuild_preference_index() {
    preference_postings_.clear();
    if (!indexed_preference_duplicates_) {
        return;
    }
    preference_postings_.reserve(preferences_.size());
    for (std::size_t index = 0U; index < preferences_.size(); ++index) {
        const PreferenceExample& preference = preferences_[index];
        preference_postings_[preference_key(
            preference.prompt, preference.chosen, preference.rejected
        )].push_back(index);
        ++training_operation_stats_.preference_index_entries_built;
    }
    ++training_operation_stats_.preference_index_rebuilds;
}

void GeneralInstructionFabric::rebuild_active_learning_index() {
    active_learning_postings_.clear();
    if (!indexed_active_learning_duplicates_) {
        return;
    }
    active_learning_postings_.reserve(active_learning_items_.size());
    for (std::size_t index = 0U; index < active_learning_items_.size(); ++index) {
        const ActiveLearningItem& item = active_learning_items_[index];
        active_learning_postings_[active_learning_key(
            item.prompt, item.grounding
        )].push_back(index);
        ++training_operation_stats_.active_learning_index_entries_built;
    }
    ++training_operation_stats_.active_learning_index_rebuilds;
}

void GeneralInstructionFabric::rebuild_instruction_index() {
    instruction_keys_.clear();
    if (!indexed_instruction_duplicates_) {
        return;
    }
    instruction_keys_.reserve(demonstrations_.size());
    for (const InstructionDemonstration& demonstration : demonstrations_) {
        instruction_keys_.insert(instruction_key(
            demonstration.task,
            demonstration.domain,
            demonstration.prompt,
            demonstration.rationale,
            demonstration.response
        ));
        ++training_operation_stats_.instruction_index_entries_built;
    }
    ++training_operation_stats_.instruction_index_rebuilds;
}

std::uint64_t GeneralInstructionFabric::train_instruction(
    const std::string_view task,
    const std::string_view domain,
    const std::string_view prompt,
    const std::string_view rationale,
    const std::string_view response,
    const double quality
) {
    if (prompt.empty() || response.empty() || !std::isfinite(quality) || quality <= 0.0) {
        throw std::invalid_argument("invalid general instruction demonstration");
    }
    const std::string normalized_task = normalize_label(task.empty() ? "general" : task);
    const std::string normalized_domain = normalize_label(domain.empty() ? "general" : domain);
    std::string signature_text;
    signature_text.reserve(task.size() + domain.size() + prompt.size() + 2U);
    signature_text.append(task);
    signature_text.push_back(' ');
    signature_text.append(domain);
    signature_text.push_back(' ');
    signature_text.append(prompt);
    const SemanticSignature signature = make_signature(signature_text);

    const std::uint64_t exact_key = instruction_key(
        normalized_task, normalized_domain, prompt, rationale, response
    );
    ++training_operation_stats_.instruction_duplicate_prefilter_lookups;
    const bool run_reference_retrieval = !indexed_instruction_duplicates_ ||
        instruction_keys_.contains(exact_key);
    if (run_reference_retrieval) {
        ++training_operation_stats_.instruction_duplicate_retrievals;
        const auto matches = retrieve(
            normalized_task, normalized_domain, prompt, {}, 8U
        );
        for (const GeneralRetrievalMatch& match : matches) {
            InstructionDemonstration& existing =
                demonstrations_[match.demonstration_index];
            if (existing.task == normalized_task &&
                existing.domain == normalized_domain &&
                existing.prompt == prompt && existing.rationale == rationale &&
                existing.response == response) {
                ++existing.support;
                existing.quality = std::max(existing.quality, quality);
                return existing.id;
            }
        }
    } else {
        ++training_operation_stats_.instruction_duplicate_retrievals_avoided;
    }

    if (demonstrations_.size() >= config_.maximum_demonstrations) {
        throw std::runtime_error("Solstice general demonstration capacity exceeded");
    }
    const std::uint64_t id = next_demonstration_id_++;
    demonstrations_.push_back(InstructionDemonstration{
        id,
        normalized_task,
        normalized_domain,
        std::string(prompt),
        std::string(rationale),
        std::string(response),
        signature,
        1U,
        quality,
    });
    index_demonstration(demonstrations_.size() - 1U);
    if (indexed_instruction_duplicates_) {
        instruction_keys_.insert(exact_key);
        ++training_operation_stats_.instruction_index_incremental_inserts;
    }
    return id;
}

std::uint64_t GeneralInstructionFabric::train_preference(
    const std::string_view prompt,
    const std::string_view chosen,
    const std::string_view rejected,
    const std::string_view feedback,
    const double weight
) {
    if (prompt.empty() || chosen.empty() || rejected.empty() || chosen == rejected ||
        !std::isfinite(weight) || weight <= 0.0) {
        throw std::invalid_argument("invalid Solstice preference example");
    }
    ++training_operation_stats_.preference_duplicate_lookups;
    PreferenceExample* duplicate = nullptr;
    const std::uint64_t key = preference_key(prompt, chosen, rejected);
    if (indexed_preference_duplicates_) {
        const auto posting = preference_postings_.find(key);
        if (posting != preference_postings_.end()) {
            for (const std::size_t index : posting->second) {
                ++training_operation_stats_.indexed_preference_candidates;
                PreferenceExample& preference = preferences_[index];
                if (preference.prompt == prompt && preference.chosen == chosen &&
                    preference.rejected == rejected) {
                    duplicate = &preference;
                    break;
                }
            }
        }
    } else {
        for (PreferenceExample& preference : preferences_) {
            ++training_operation_stats_.linear_preference_comparisons;
            if (preference.prompt == prompt && preference.chosen == chosen &&
                preference.rejected == rejected) {
                duplicate = &preference;
                break;
            }
        }
    }
    if (duplicate != nullptr) {
            PreferenceExample& preference = *duplicate;
            preference.weight += weight;
            if (preference.feedback.empty() && !feedback.empty()) {
                preference.feedback = feedback;
            }
            return preference.id;
    }
    if (preferences_.size() >= config_.maximum_preferences) {
        throw std::runtime_error("Solstice preference capacity exceeded");
    }
    const std::uint64_t id = next_preference_id_++;
    preferences_.push_back(PreferenceExample{
        id,
        std::string(prompt),
        std::string(chosen),
        std::string(rejected),
        std::string(feedback),
        make_signature(prompt),
        make_signature(chosen),
        make_signature(rejected),
        weight,
    });
    if (indexed_preference_duplicates_) {
        preference_postings_[key].push_back(preferences_.size() - 1U);
        ++training_operation_stats_.preference_index_incremental_inserts;
    }
    return id;
}

void GeneralInstructionFabric::set_indexed_preference_duplicates(
    const bool enabled
) {
    indexed_preference_duplicates_ = enabled;
    rebuild_preference_index();
}

void GeneralInstructionFabric::set_indexed_active_learning_duplicates(
    const bool enabled
) {
    indexed_active_learning_duplicates_ = enabled;
    rebuild_active_learning_index();
}

void GeneralInstructionFabric::set_indexed_instruction_duplicates(
    const bool enabled
) {
    indexed_instruction_duplicates_ = enabled;
    rebuild_instruction_index();
}

GeneralTrainingOperationStats
GeneralInstructionFabric::training_operation_stats() const noexcept {
    return training_operation_stats_;
}

std::uint64_t GeneralInstructionFabric::observe_uncertain(
    const std::string_view prompt,
    const std::string_view grounding,
    const double uncertainty
) {
    if (prompt.empty() || !std::isfinite(uncertainty) || uncertainty < 0.0 ||
        uncertainty > 1.0) {
        throw std::invalid_argument("invalid active-learning observation");
    }
    if (uncertainty < config_.active_learning_uncertainty) {
        return 0U;
    }
    ++training_operation_stats_.active_learning_duplicate_lookups;
    ActiveLearningItem* duplicate = nullptr;
    const std::uint64_t key = active_learning_key(prompt, grounding);
    if (indexed_active_learning_duplicates_) {
        const auto posting = active_learning_postings_.find(key);
        if (posting != active_learning_postings_.end()) {
            for (const std::size_t index : posting->second) {
                ++training_operation_stats_.indexed_active_learning_candidates;
                ActiveLearningItem& item = active_learning_items_[index];
                if (item.prompt == prompt && item.grounding == grounding) {
                    duplicate = &item;
                    break;
                }
            }
        }
    } else {
        for (ActiveLearningItem& item : active_learning_items_) {
            ++training_operation_stats_.linear_active_learning_comparisons;
            if (item.prompt == prompt && item.grounding == grounding) {
                duplicate = &item;
                break;
            }
        }
    }
    if (duplicate != nullptr) {
        ++duplicate->observations;
        duplicate->uncertainty = std::max(duplicate->uncertainty, uncertainty);
        return duplicate->id;
    }
    if (active_learning_items_.size() >= config_.maximum_active_learning_items) {
        const auto lowest = std::min_element(
            active_learning_items_.begin(), active_learning_items_.end(),
            [](const ActiveLearningItem& left, const ActiveLearningItem& right) {
                if (left.uncertainty != right.uncertainty) {
                    return left.uncertainty < right.uncertainty;
                }
                return left.observations < right.observations;
            }
        );
        if (lowest != active_learning_items_.end() && lowest->uncertainty < uncertainty) {
            const std::size_t index = static_cast<std::size_t>(
                std::distance(active_learning_items_.begin(), lowest)
            );
            if (indexed_active_learning_duplicates_) {
                const std::uint64_t old_key = active_learning_key(
                    lowest->prompt, lowest->grounding
                );
                auto posting = active_learning_postings_.find(old_key);
                if (posting != active_learning_postings_.end()) {
                    std::erase(posting->second, index);
                    if (posting->second.empty()) {
                        active_learning_postings_.erase(posting);
                    }
                }
            }
            *lowest = ActiveLearningItem{
                next_active_learning_id_++, std::string(prompt), std::string(grounding),
                uncertainty, 1U,
            };
            if (indexed_active_learning_duplicates_) {
                active_learning_postings_[key].push_back(index);
                ++training_operation_stats_.active_learning_index_incremental_inserts;
            }
            return lowest->id;
        }
        return 0U;
    }
    const std::uint64_t id = next_active_learning_id_++;
    active_learning_items_.push_back(ActiveLearningItem{
        id, std::string(prompt), std::string(grounding), uncertainty, 1U,
    });
    if (indexed_active_learning_duplicates_) {
        active_learning_postings_[key].push_back(active_learning_items_.size() - 1U);
        ++training_operation_stats_.active_learning_index_incremental_inserts;
    }
    return id;
}

std::vector<GeneralRetrievalMatch> GeneralInstructionFabric::retrieve(
    const std::string_view task,
    const std::string_view domain,
    const std::string_view prompt,
    const std::string_view grounding,
    std::size_t maximum_results
) const {
    if (demonstrations_.empty() || prompt.empty()) {
        return {};
    }
    if (maximum_results == 0U) {
        maximum_results = config_.maximum_retrieved_demonstrations;
    }
    maximum_results = std::min(maximum_results, config_.maximum_retrieved_demonstrations);
    const std::string normalized_task = normalize_label(task.empty() ? "general" : task);
    const std::string normalized_domain = normalize_label(domain.empty() ? "general" : domain);
    std::string query_text;
    query_text.reserve(task.size() + domain.size() + prompt.size() + grounding.size() + 3U);
    query_text.append(task);
    query_text.push_back(' ');
    query_text.append(domain);
    query_text.push_back(' ');
    query_text.append(prompt);
    query_text.push_back(' ');
    query_text.append(grounding);
    const SemanticSignature signature = make_signature(query_text);

    std::unordered_map<std::size_t, std::uint8_t> votes;
    const auto keys = band_keys(signature);
    for (const std::uint64_t key : keys) {
        const auto posting = band_postings_.find(key);
        if (posting == band_postings_.end()) {
            continue;
        }
        for (const std::size_t index : posting->second) {
            auto [position, inserted] = votes.emplace(
                index,
                std::uint8_t{1U}
            );
            if (!inserted && position->second < std::numeric_limits<std::uint8_t>::max()) {
                ++position->second;
            }
        }
    }

    std::vector<std::pair<std::size_t, std::uint8_t>> candidate_votes;
    candidate_votes.reserve(votes.size());
    for (const auto& [index, count] : votes) {
        candidate_votes.emplace_back(index, count);
    }
    std::sort(
        candidate_votes.begin(), candidate_votes.end(),
        [](const auto& left, const auto& right) {
            if (left.second != right.second) {
                return left.second > right.second;
            }
            return left.first < right.first;
        }
    );
    if (candidate_votes.empty()) {
        const std::size_t fallback = std::min(
            demonstrations_.size(), config_.maximum_retrieval_candidates
        );
        candidate_votes.reserve(fallback);
        for (std::size_t index = 0U; index < fallback; ++index) {
            candidate_votes.emplace_back(index, std::uint8_t{0U});
        }
    } else if (candidate_votes.size() > config_.maximum_retrieval_candidates) {
        candidate_votes.resize(config_.maximum_retrieval_candidates);
    }

    std::vector<GeneralRetrievalMatch> matches;
    matches.reserve(std::min(maximum_results * 4U, candidate_votes.size()));
    for (const auto& [index, votes_for_candidate] : candidate_votes) {
        static_cast<void>(votes_for_candidate);
        const InstructionDemonstration& demonstration = demonstrations_[index];
        const double semantic_score = signature_similarity(
            signature, demonstration.signature
        );
        if (semantic_score < config_.minimum_retrieval_similarity &&
            demonstration.prompt != prompt) {
            continue;
        }
        double score = semantic_score;
        if (demonstration.task == normalized_task) {
            score += config_.exact_task_bonus;
        }
        if (demonstration.domain == normalized_domain) {
            score += config_.exact_domain_bonus;
        }
        score *= std::clamp(demonstration.quality, 0.1, 4.0);
        score *= 1.0 + 0.04 * std::log1p(static_cast<double>(demonstration.support));
        if (score >= config_.minimum_retrieval_similarity) {
            matches.push_back(GeneralRetrievalMatch{demonstration.id, index, score});
        }
    }
    std::sort(
        matches.begin(), matches.end(),
        [](const GeneralRetrievalMatch& left, const GeneralRetrievalMatch& right) {
            if (left.score != right.score) {
                return left.score > right.score;
            }
            return left.demonstration_id < right.demonstration_id;
        }
    );
    if (matches.size() > maximum_results) {
        matches.resize(maximum_results);
    }
    return matches;
}

DeliberationContext GeneralInstructionFabric::build_context(
    const std::string_view task,
    const std::string_view domain,
    const std::string_view prompt,
    const std::string_view grounding,
    const std::string_view knowledge
) const {
    DeliberationContext result;
    result.task = normalize_label(task.empty() ? "general" : task);
    result.domain = normalize_label(domain.empty() ? "general" : domain);
    const std::vector<GeneralRetrievalMatch> matches = retrieve(
        result.task, result.domain, prompt, grounding
    );
    if (!matches.empty()) {
        const InstructionDemonstration& top =
            demonstrations_[matches.front().demonstration_index];
        std::string query_text;
        query_text.reserve(task.size() + domain.size() + prompt.size() + grounding.size() + 3U);
        query_text.append(task);
        query_text.push_back(' ');
        query_text.append(domain);
        query_text.push_back(' ');
        query_text.append(prompt);
        query_text.push_back(' ');
        query_text.append(grounding);
        const double semantic_confidence = signature_similarity(
            make_signature(query_text), top.signature
        );
        result.confidence = clamp01(semantic_confidence);
        const bool exact_transfer_domain =
            top.task == result.task && top.domain == result.domain;
        const bool strong_structural_match =
            !matches.empty() && matches.front().score >= 0.55;
        if (top.prompt == prompt ||
            (exact_transfer_domain &&
             (semantic_confidence >= config_.direct_recall_threshold ||
              strong_structural_match))) {
            result.direct_response = top.response;
        }
    }

    std::ostringstream prefix;
    prefix << "Task: " << result.task << "\nDomain: " << result.domain << '\n';
    append_bounded(result.context, prefix.str(), config_.maximum_context_characters);
    if (!grounding.empty()) {
        append_bounded(result.context, "Perceptual grounding:\n", config_.maximum_context_characters);
        append_bounded(result.context, grounding, config_.maximum_context_characters);
        append_bounded(result.context, "\n", config_.maximum_context_characters);
    }
    if (!knowledge.empty()) {
        append_bounded(result.context, "Retrieved knowledge:\n", config_.maximum_context_characters);
        append_bounded(result.context, knowledge, config_.maximum_context_characters);
        append_bounded(result.context, "\n", config_.maximum_context_characters);
    }
    if (!matches.empty()) {
        append_bounded(
            result.context,
            "Relevant solved examples (use the transferable pattern, not unsupported details):\n",
            config_.maximum_context_characters
        );
    }
    for (std::size_t rank = 0U; rank < matches.size(); ++rank) {
        const GeneralRetrievalMatch& match = matches[rank];
        const InstructionDemonstration& demonstration =
            demonstrations_[match.demonstration_index];
        result.demonstration_ids.push_back(demonstration.id);
        std::ostringstream header;
        header << "Example " << (rank + 1U) << " [score=" << match.score
               << ", task=" << demonstration.task
               << ", domain=" << demonstration.domain << "]\n";
        append_bounded(result.context, header.str(), config_.maximum_context_characters);
        append_bounded(result.context, "Problem: ", config_.maximum_context_characters);
        append_bounded(result.context, demonstration.prompt, config_.maximum_context_characters);
        append_bounded(result.context, "\n", config_.maximum_context_characters);
        if (!demonstration.rationale.empty()) {
            append_bounded(result.context, "Solution sketch: ", config_.maximum_context_characters);
            append_bounded(result.context, demonstration.rationale, config_.maximum_context_characters);
            append_bounded(result.context, "\n", config_.maximum_context_characters);
            if (result.plan_hint.empty()) {
                result.plan_hint = demonstration.rationale;
            }
        }
        append_bounded(result.context, "Answer pattern: ", config_.maximum_context_characters);
        append_bounded(result.context, demonstration.response, config_.maximum_context_characters);
        append_bounded(result.context, "\n", config_.maximum_context_characters);
        if (result.context.size() >= config_.maximum_context_characters) {
            break;
        }
    }
    append_bounded(
        result.context,
        "Answer the current request using only justified knowledge. State uncertainty or abstain when evidence is insufficient.\n",
        config_.maximum_context_characters
    );
    return result;
}

double GeneralInstructionFabric::score_response(
    const std::string_view prompt,
    const std::string_view response,
    const std::span<const GeneralRetrievalMatch> matches
) const {
    if (response.empty()) {
        return -std::numeric_limits<double>::infinity();
    }
    const SemanticSignature response_signature = make_signature(response);
    double score = 0.15 + 0.25 * lexical_diversity(response);
    const std::size_t match_limit = std::min<std::size_t>(matches.size(), 8U);
    for (std::size_t index = 0U; index < match_limit; ++index) {
        const GeneralRetrievalMatch& match = matches[index];
        const InstructionDemonstration& demonstration =
            demonstrations_[match.demonstration_index];
        const double response_match = signature_similarity(
            response_signature, make_signature(demonstration.response)
        );
        score += response_match * match.score * demonstration.quality /
            static_cast<double>(index + 1U);
    }

    const SemanticSignature prompt_signature = make_signature(prompt);
    for (const PreferenceExample& preference : preferences_) {
        const double prompt_match = signature_similarity(
            prompt_signature, preference.prompt_signature
        );
        if (prompt_match < 0.20) {
            continue;
        }
        const double chosen_match = signature_similarity(
            response_signature, preference.chosen_signature
        );
        const double rejected_match = signature_similarity(
            response_signature, preference.rejected_signature
        );
        score += config_.preference_weight * preference.weight * prompt_match *
            (chosen_match - rejected_match);
    }

    const std::vector<std::string> response_words = words(response);
    if (response_words.size() > 8U) {
        const double diversity = lexical_diversity(response);
        if (diversity < 0.35) {
            score -= (0.35 - diversity) * 4.0;
        }
    }
    return score;
}

std::span<const InstructionDemonstration>
GeneralInstructionFabric::demonstrations() const noexcept {
    return demonstrations_;
}

std::span<const PreferenceExample>
GeneralInstructionFabric::preferences() const noexcept {
    return preferences_;
}

std::span<const ActiveLearningItem>
GeneralInstructionFabric::active_learning_items() const noexcept {
    return active_learning_items_;
}

const GeneralFabricConfig& GeneralInstructionFabric::config() const noexcept {
    return config_;
}

std::uint64_t GeneralInstructionFabric::deterministic_hash() const noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, next_demonstration_id_);
    hash_u64(hash, next_preference_id_);
    hash_u64(hash, next_active_learning_id_);
    hash_u64(hash, demonstrations_.size());
    for (const InstructionDemonstration& demonstration : demonstrations_) {
        hash_u64(hash, demonstration.id);
        hash_string(hash, demonstration.task);
        hash_string(hash, demonstration.domain);
        hash_string(hash, demonstration.prompt);
        hash_string(hash, demonstration.rationale);
        hash_string(hash, demonstration.response);
        for (const std::uint64_t word : demonstration.signature.words) {
            hash_u64(hash, word);
        }
        hash_u64(hash, demonstration.support);
        hash_double(hash, demonstration.quality);
    }
    hash_u64(hash, preferences_.size());
    for (const PreferenceExample& preference : preferences_) {
        hash_u64(hash, preference.id);
        hash_string(hash, preference.prompt);
        hash_string(hash, preference.chosen);
        hash_string(hash, preference.rejected);
        hash_string(hash, preference.feedback);
        hash_double(hash, preference.weight);
    }
    hash_u64(hash, active_learning_items_.size());
    for (const ActiveLearningItem& item : active_learning_items_) {
        hash_u64(hash, item.id);
        hash_string(hash, item.prompt);
        hash_string(hash, item.grounding);
        hash_double(hash, item.uncertainty);
        hash_u64(hash, item.observations);
    }
    return hash;
}

GeneralFabricSnapshot GeneralInstructionFabric::snapshot() const {
    return GeneralFabricSnapshot{
        config_, next_demonstration_id_, next_preference_id_,
        next_active_learning_id_, demonstrations_, preferences_,
        active_learning_items_,
    };
}

GeneralInstructionFabric GeneralInstructionFabric::from_snapshot(
    GeneralFabricSnapshot snapshot
) {
    GeneralInstructionFabric fabric(snapshot.config);
    fabric.next_demonstration_id_ = snapshot.next_demonstration_id;
    fabric.next_preference_id_ = snapshot.next_preference_id;
    fabric.next_active_learning_id_ = snapshot.next_active_learning_id;
    fabric.demonstrations_ = std::move(snapshot.demonstrations);
    fabric.preferences_ = std::move(snapshot.preferences);
    fabric.active_learning_items_ = std::move(snapshot.active_learning_items);
    fabric.rebuild_index();
    return fabric;
}

}  // namespace rlf::solstice
