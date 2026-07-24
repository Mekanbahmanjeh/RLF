#include "rlf/frontier/knowledge_fabric.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cctype>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace rlf::frontier {
namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

void hash_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (unsigned int index = 0U; index < 8U; ++index) {
        hash ^= (value >> (index * 8U)) & 0xFFULL;
        hash *= fnv_prime;
    }
}

void hash_string(std::uint64_t& hash, const std::string_view value) noexcept {
    hash_u64(hash, static_cast<std::uint64_t>(value.size()));
    for (const char character : value) {
        hash ^= static_cast<unsigned char>(character);
        hash *= fnv_prime;
    }
}

void hash_double(std::uint64_t& hash, const double value) noexcept {
    hash_u64(hash, std::bit_cast<std::uint64_t>(value));
}

[[nodiscard]] bool finite_probability(const double value) noexcept {
    return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

[[nodiscard]] std::uint64_t approximate_record_bytes(
    const KnowledgeRecord& record
) noexcept {
    return static_cast<std::uint64_t>(sizeof(KnowledgeRecord)) +
        static_cast<std::uint64_t>(record.subject.size()) +
        static_cast<std::uint64_t>(record.predicate.size()) +
        static_cast<std::uint64_t>(record.object.size()) +
        static_cast<std::uint64_t>(record.source.size());
}

[[nodiscard]] std::uint64_t approximate_mode_bytes(
    const ModeRecord& mode
) noexcept {
    return static_cast<std::uint64_t>(sizeof(ModeRecord)) +
        static_cast<std::uint64_t>(mode.label.size()) +
        static_cast<std::uint64_t>(mode.prototype.size() * sizeof(float)) +
        static_cast<std::uint64_t>(
            (mode.child_ids.size() + mode.linked_modes.size()) *
            sizeof(std::uint64_t)
        );
}

}  // namespace

std::string_view to_string(const KnowledgeKind kind) noexcept {
    switch (kind) {
    case KnowledgeKind::claim: return "claim";
    case KnowledgeKind::observed_fact: return "observed_fact";
    case KnowledgeKind::verified_fact: return "verified_fact";
    case KnowledgeKind::inferred_belief: return "inferred_belief";
    case KnowledgeKind::prediction: return "prediction";
    case KnowledgeKind::procedure: return "procedure";
    case KnowledgeKind::concept_record: return "concept";
    }
    return "unknown";
}

std::string_view to_string(const MemoryTier tier) noexcept {
    switch (tier) {
    case MemoryTier::active: return "active";
    case MemoryTier::hot: return "hot";
    case MemoryTier::warm: return "warm";
    case MemoryTier::cold: return "cold";
    }
    return "unknown";
}

std::string_view to_string(const Modality modality) noexcept {
    switch (modality) {
    case Modality::text: return "text";
    case Modality::structured: return "structured";
    case Modality::image: return "image";
    case Modality::video: return "video";
    case Modality::audio: return "audio";
    case Modality::sensor: return "sensor";
    case Modality::action: return "action";
    }
    return "unknown";
}

KnowledgeFabric::KnowledgeFabric(const std::uint64_t seed) : seed_(seed) {}

std::uint64_t KnowledgeFabric::seed() const noexcept { return seed_; }
std::uint64_t KnowledgeFabric::step() const noexcept { return step_; }
void KnowledgeFabric::set_step(const std::uint64_t step) noexcept { step_ = step; }

void KnowledgeFabric::reserve_records(
    const std::size_t expected_records,
    const std::size_t expected_terms_per_record
) {
    exact_index_.reserve(expected_records);
    if (expected_terms_per_record != 0U &&
        expected_records <= std::numeric_limits<std::size_t>::max() / expected_terms_per_record) {
        term_index_.reserve(expected_records * expected_terms_per_record);
    } else {
        term_index_.reserve(expected_records);
    }
}

void KnowledgeFabric::validate_record(const KnowledgeRecord& record) {
    if (record.subject.empty() || record.predicate.empty() || record.object.empty()) {
        throw std::invalid_argument("knowledge subject, predicate, and object must be non-empty");
    }
    if (!finite_probability(record.confidence)) {
        throw std::invalid_argument("knowledge confidence must be in [0,1]");
    }
    if (!std::isfinite(record.utility)) {
        throw std::invalid_argument("knowledge utility must be finite");
    }
    if (record.valid_until != 0U && record.valid_until < record.valid_from) {
        throw std::invalid_argument("knowledge validity interval is reversed");
    }
}

void KnowledgeFabric::validate_mode(const ModeRecord& mode) {
    if (mode.label.empty() || mode.prototype.empty()) {
        throw std::invalid_argument("mode label and prototype must be non-empty");
    }
    if (!finite_probability(mode.confidence) || !std::isfinite(mode.utility)) {
        throw std::invalid_argument("mode confidence or utility is invalid");
    }
    if (std::any_of(mode.prototype.begin(), mode.prototype.end(), [](const float value) {
            return !std::isfinite(static_cast<double>(value));
        })) {
        throw std::invalid_argument("mode prototype contains non-finite values");
    }
}

std::uint64_t KnowledgeFabric::token_key(const std::string_view token) noexcept {
    std::uint64_t hash = fnv_offset;
    for (const char character : token) {
        hash ^= static_cast<unsigned char>(character);
        hash *= fnv_prime;
    }
    return hash;
}

std::uint64_t KnowledgeFabric::exact_key(
    const std::string_view subject,
    const std::string_view predicate
) noexcept {
    std::uint64_t hash = fnv_offset;
    for (const char character : subject) {
        hash ^= static_cast<unsigned char>(character);
        hash *= fnv_prime;
    }
    hash ^= 0x1FU;
    hash *= fnv_prime;
    for (const char character : predicate) {
        hash ^= static_cast<unsigned char>(character);
        hash *= fnv_prime;
    }
    return hash;
}

std::vector<std::string> KnowledgeFabric::tokenize(const std::string_view value) {
    std::vector<std::string> tokens;
    std::string current;
    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
        if (std::isalnum(character) != 0 || character == static_cast<unsigned char>('_')) {
            current.push_back(static_cast<char>(std::tolower(character)));
        } else if (!current.empty()) {
            tokens.push_back(std::move(current));
            current.clear();
        }
    }
    if (!current.empty()) tokens.push_back(std::move(current));
    std::sort(tokens.begin(), tokens.end());
    tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
    return tokens;
}

void KnowledgeFabric::index_record(const KnowledgeRecord& record) {
    exact_index_.emplace(exact_key(record.subject, record.predicate), record.stable_id);
    const std::string combined = record.subject + " " + record.predicate + " " + record.object;
    for (const auto& token : tokenize(combined)) {
        term_index_.emplace(token_key(token), record.stable_id);
    }
}

void KnowledgeFabric::unindex_record(const KnowledgeRecord& record) {
    const auto erase_id = [id = record.stable_id](auto& index, const std::uint64_t key) {
        const auto range = index.equal_range(key);
        for (auto iterator = range.first; iterator != range.second;) {
            if (iterator->second == id) iterator = index.erase(iterator);
            else ++iterator;
        }
    };
    erase_id(exact_index_, exact_key(record.subject, record.predicate));
    const std::string combined = record.subject + " " + record.predicate + " " + record.object;
    for (const auto& token : tokenize(combined)) {
        erase_id(term_index_, token_key(token));
    }
}

void KnowledgeFabric::detect_contradictions(KnowledgeRecord& record) {
    const auto range = exact_index_.equal_range(exact_key(record.subject, record.predicate));
    for (auto iterator = range.first; iterator != range.second; ++iterator) {
        auto existing = records_.find(iterator->second);
        if (existing == records_.end() || existing->second.invalidated || existing->second.stale) continue;
        if (existing->second.object != record.object) {
            ++existing->second.contradiction_count;
            ++record.contradiction_count;
        }
    }
}

std::uint64_t KnowledgeFabric::insert(KnowledgeRecord record) {
    validate_record(record);
    if (record.stable_id == 0U) record.stable_id = next_record_id_++;
    if (records_.contains(record.stable_id)) {
        throw std::invalid_argument("duplicate knowledge record ID");
    }
    next_record_id_ = std::max(next_record_id_, record.stable_id + 1U);
    if (record.creation_step == 0U) record.creation_step = step_;
    if (record.last_used_step == 0U) record.last_used_step = record.creation_step;
    detect_contradictions(record);
    const std::uint64_t id = record.stable_id;
    records_.emplace(id, std::move(record));
    index_record(records_.at(id));
    return id;
}

bool KnowledgeFabric::update(
    const std::uint64_t stable_id,
    const KnowledgeRecord& replacement
) {
    auto iterator = records_.find(stable_id);
    if (iterator == records_.end()) return false;
    KnowledgeRecord updated = replacement;
    updated.stable_id = stable_id;
    updated.version = iterator->second.version + 1U;
    validate_record(updated);
    unindex_record(iterator->second);
    iterator->second = std::move(updated);
    detect_contradictions(iterator->second);
    index_record(iterator->second);
    return true;
}

bool KnowledgeFabric::invalidate(const std::uint64_t stable_id) {
    if (auto* record = find(stable_id); record != nullptr) {
        record->invalidated = true;
        record->tier = MemoryTier::cold;
        return true;
    }
    return false;
}

bool KnowledgeFabric::mark_stale(const std::uint64_t stable_id, const bool stale) {
    if (auto* record = find(stable_id); record != nullptr) {
        record->stale = stale;
        if (stale) record->tier = MemoryTier::cold;
        return true;
    }
    return false;
}

bool KnowledgeFabric::verify(const std::uint64_t stable_id, const double confidence) {
    if (!finite_probability(confidence)) {
        throw std::invalid_argument("verification confidence must be in [0,1]");
    }
    if (auto* record = find(stable_id); record != nullptr) {
        record->verified = true;
        record->kind = KnowledgeKind::verified_fact;
        record->confidence = confidence;
        record->stale = false;
        record->last_used_step = step_;
        record->tier = MemoryTier::hot;
        return true;
    }
    return false;
}

std::vector<KnowledgeHit> KnowledgeFabric::query(const KnowledgeQuery& query_value) {
    if (query_value.maximum_results == 0U) return {};
    std::vector<std::uint64_t> candidates;
    std::vector<std::string> required_terms;
    bool exact = false;
    if (!query_value.subject.empty() || !query_value.predicate.empty()) {
        const auto range = exact_index_.equal_range(exact_key(query_value.subject, query_value.predicate));
        for (auto iterator = range.first; iterator != range.second; ++iterator) {
            candidates.push_back(iterator->second);
        }
        exact = !candidates.empty();
    }
    if (candidates.empty()) {
        required_terms = query_value.terms;
        if (required_terms.empty()) {
            const std::string combined = query_value.subject + " " + query_value.predicate;
            required_terms = tokenize(combined);
        } else {
            std::vector<std::string> normalized_terms;
            for (const auto& raw_term : required_terms) {
                const auto normalized = tokenize(raw_term);
                normalized_terms.insert(normalized_terms.end(), normalized.begin(), normalized.end());
            }
            std::sort(normalized_terms.begin(), normalized_terms.end());
            normalized_terms.erase(
                std::unique(normalized_terms.begin(), normalized_terms.end()),
                normalized_terms.end()
            );
            required_terms = std::move(normalized_terms);
        }
        bool first = true;
        for (const auto& term : required_terms) {
            const auto range = term_index_.equal_range(token_key(term));
            if (range.first == range.second) {
                candidates.clear();
                first = false;
                break;
            }
            std::vector<std::uint64_t> posting;
            for (auto iterator = range.first; iterator != range.second; ++iterator) {
                posting.push_back(iterator->second);
            }
            std::sort(posting.begin(), posting.end());
            posting.erase(std::unique(posting.begin(), posting.end()), posting.end());
            if (first) {
                candidates = std::move(posting);
                first = false;
            } else {
                std::sort(candidates.begin(), candidates.end());
                candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
                std::vector<std::uint64_t> intersection;
                intersection.reserve(std::min(candidates.size(), posting.size()));
                std::set_intersection(
                    candidates.begin(), candidates.end(),
                    posting.begin(), posting.end(),
                    std::back_inserter(intersection)
                );
                candidates = std::move(intersection);
            }
            if (candidates.empty()) break;
        }
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    last_candidates_examined_ = candidates.size();
    total_candidates_examined_ += candidates.size();
    std::vector<KnowledgeHit> hits;
    hits.reserve(candidates.size());
    for (const std::uint64_t id : candidates) {
        auto iterator = records_.find(id);
        if (iterator == records_.end()) continue;
        KnowledgeRecord& record = iterator->second;
        if (exact && (record.subject != query_value.subject || record.predicate != query_value.predicate)) {
            continue;
        }
        if (!exact && !required_terms.empty()) {
            const std::string combined = record.subject + " " + record.predicate + " " + record.object;
            const auto record_terms = tokenize(combined);
            if (!std::includes(
                    record_terms.begin(), record_terms.end(),
                    required_terms.begin(), required_terms.end())) {
                continue;
            }
        }
        if ((!query_value.include_stale && record.stale) ||
            (!query_value.include_invalidated && record.invalidated)) continue;
        double score = record.confidence;
        if (record.verified) score += 0.25;
        if (record.contradiction_count != 0U) {
            score -= std::min(0.25, 0.025 * static_cast<double>(record.contradiction_count));
        }
        score += std::clamp(record.utility, -1.0, 1.0) * 0.05;
        hits.push_back({.stable_id = id, .score = score, .exact = exact});
    }
    std::sort(hits.begin(), hits.end(), [](const KnowledgeHit& left, const KnowledgeHit& right) {
        return std::tie(left.score, left.stable_id) > std::tie(right.score, right.stable_id);
    });
    if (hits.size() > query_value.maximum_results) hits.resize(query_value.maximum_results);
    for (const auto& hit : hits) {
        KnowledgeRecord& record = records_.at(hit.stable_id);
        ++record.use_count;
        record.last_used_step = step_;
        if (record.tier == MemoryTier::cold) record.tier = MemoryTier::warm;
    }
    return hits;
}

const KnowledgeRecord* KnowledgeFabric::find(const std::uint64_t stable_id) const {
    const auto iterator = records_.find(stable_id);
    return iterator == records_.end() ? nullptr : &iterator->second;
}

KnowledgeRecord* KnowledgeFabric::find(const std::uint64_t stable_id) {
    const auto iterator = records_.find(stable_id);
    return iterator == records_.end() ? nullptr : &iterator->second;
}

double KnowledgeFabric::cosine_similarity(
    const std::span<const float> left,
    const std::span<const float> right
) {
    if (left.size() != right.size() || left.empty()) return 0.0;
    double dot = 0.0;
    double left_norm = 0.0;
    double right_norm = 0.0;
    for (std::size_t index = 0U; index < left.size(); ++index) {
        const double left_value = static_cast<double>(left[index]);
        const double right_value = static_cast<double>(right[index]);
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
    }
    if (left_norm <= std::numeric_limits<double>::epsilon() ||
        right_norm <= std::numeric_limits<double>::epsilon()) return 0.0;
    return std::clamp(dot / std::sqrt(left_norm * right_norm), -1.0, 1.0);
}

bool KnowledgeFabric::would_create_cycle(
    const std::uint64_t child_id,
    const std::uint64_t parent_id
) const {
    std::uint64_t current = parent_id;
    std::set<std::uint64_t> visited;
    while (true) {
        if (current == child_id) return true;
        if (!visited.insert(current).second) return true;
        const auto iterator = modes_.find(current);
        if (iterator == modes_.end() || !iterator->second.parent_id.has_value()) return false;
        current = *iterator->second.parent_id;
    }
}

std::uint64_t KnowledgeFabric::learn_mode(
    const Modality modality,
    std::string label,
    const std::span<const float> prototype,
    const std::optional<std::uint64_t> parent_id,
    const double confidence
) {
    if (prototype.empty()) throw std::invalid_argument("mode prototype must not be empty");
    if (!finite_probability(confidence)) throw std::invalid_argument("mode confidence must be in [0,1]");
    for (auto& [id, mode] : modes_) {
        if (mode.enabled && mode.modality == modality && mode.label == label) {
            if (mode.prototype.size() != prototype.size()) {
                throw std::invalid_argument("mode prototype dimension changed");
            }
            const double old_support = static_cast<double>(mode.support);
            const double denominator = old_support + 1.0;
            for (std::size_t index = 0U; index < prototype.size(); ++index) {
                const double updated =
                    (static_cast<double>(mode.prototype[index]) * old_support +
                     static_cast<double>(prototype[index])) / denominator;
                mode.prototype[index] = static_cast<float>(updated);
            }
            ++mode.support;
            mode.confidence = std::clamp(
                (mode.confidence * old_support + confidence) / denominator,
                0.0,
                1.0
            );
            mode.last_used_step = step_;
            return id;
        }
    }
    ModeRecord mode;
    mode.stable_id = next_mode_id_++;
    mode.modality = modality;
    mode.label = std::move(label);
    mode.prototype.assign(prototype.begin(), prototype.end());
    mode.parent_id = parent_id;
    mode.confidence = confidence;
    mode.support = 1U;
    mode.creation_step = step_;
    mode.last_used_step = step_;
    validate_mode(mode);
    if (parent_id.has_value()) {
        auto parent = modes_.find(*parent_id);
        if (parent == modes_.end()) throw std::invalid_argument("mode parent does not exist");
        if (would_create_cycle(mode.stable_id, *parent_id)) {
            throw std::invalid_argument("mode hierarchy would become cyclic");
        }
        parent->second.child_ids.insert(mode.stable_id);
    }
    const std::uint64_t id = mode.stable_id;
    modes_.emplace(id, std::move(mode));
    return id;
}

bool KnowledgeFabric::link_modes(const std::uint64_t left, const std::uint64_t right) {
    if (left == right) return false;
    auto left_iterator = modes_.find(left);
    auto right_iterator = modes_.find(right);
    if (left_iterator == modes_.end() || right_iterator == modes_.end()) return false;
    left_iterator->second.linked_modes.insert(right);
    right_iterator->second.linked_modes.insert(left);
    return true;
}

std::vector<ModeHit> KnowledgeFabric::retrieve_modes(
    const Modality modality,
    const std::span<const float> query_value,
    const std::size_t maximum_results
) const {
    if (maximum_results == 0U || query_value.empty()) return {};
    std::vector<ModeHit> hits;
    for (const auto& [id, mode] : modes_) {
        if (!mode.enabled || mode.modality != modality || mode.prototype.size() != query_value.size()) continue;
        const double similarity = cosine_similarity(mode.prototype, query_value);
        hits.push_back({
            .stable_id = id,
            .score = similarity * (0.75 + 0.25 * mode.confidence),
        });
    }
    std::sort(hits.begin(), hits.end(), [](const ModeHit& left, const ModeHit& right) {
        return std::tie(left.score, left.stable_id) > std::tie(right.score, right.stable_id);
    });
    if (hits.size() > maximum_results) hits.resize(maximum_results);
    return hits;
}

const ModeRecord* KnowledgeFabric::find_mode(const std::uint64_t stable_id) const {
    const auto iterator = modes_.find(stable_id);
    return iterator == modes_.end() ? nullptr : &iterator->second;
}

ModeRecord* KnowledgeFabric::find_mode(const std::uint64_t stable_id) {
    const auto iterator = modes_.find(stable_id);
    return iterator == modes_.end() ? nullptr : &iterator->second;
}

void KnowledgeFabric::consolidate(
    const std::size_t hot_limit,
    const std::size_t active_limit
) {
    std::vector<KnowledgeRecord*> ranked;
    ranked.reserve(records_.size());
    for (auto& [id, record] : records_) {
        static_cast<void>(id);
        ranked.push_back(&record);
    }
    std::sort(ranked.begin(), ranked.end(), [](const KnowledgeRecord* left, const KnowledgeRecord* right) {
        return std::tie(left->use_count, left->verified, left->confidence, left->last_used_step) >
            std::tie(right->use_count, right->verified, right->confidence, right->last_used_step);
    });
    for (std::size_t index = 0U; index < ranked.size(); ++index) {
        if (ranked[index]->invalidated || ranked[index]->stale) {
            ranked[index]->tier = MemoryTier::cold;
        } else if (index < active_limit) {
            ranked[index]->tier = MemoryTier::active;
        } else if (index < active_limit + hot_limit) {
            ranked[index]->tier = MemoryTier::hot;
        } else if (ranked[index]->use_count == 0U) {
            ranked[index]->tier = MemoryTier::cold;
        } else {
            ranked[index]->tier = MemoryTier::warm;
        }
    }
}

void KnowledgeFabric::prune(
    const std::size_t maximum_records,
    const double minimum_utility
) {
    if (!std::isfinite(minimum_utility)) throw std::invalid_argument("minimum utility must be finite");
    if (records_.size() <= maximum_records) return;
    std::vector<std::uint64_t> candidates;
    for (const auto& [id, record] : records_) {
        if (record.invalidated || record.stale ||
            (record.utility < minimum_utility && !record.verified)) {
            candidates.push_back(id);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [this](const std::uint64_t left, const std::uint64_t right) {
        const auto& left_record = records_.at(left);
        const auto& right_record = records_.at(right);
        return std::tie(left_record.utility, left_record.use_count, left_record.last_used_step, left) <
            std::tie(right_record.utility, right_record.use_count, right_record.last_used_step, right);
    });
    for (const std::uint64_t id : candidates) {
        if (records_.size() <= maximum_records) break;
        auto iterator = records_.find(id);
        if (iterator == records_.end()) continue;
        unindex_record(iterator->second);
        records_.erase(iterator);
    }
}

const std::map<std::uint64_t, KnowledgeRecord>& KnowledgeFabric::records() const noexcept {
    return records_;
}

const std::map<std::uint64_t, ModeRecord>& KnowledgeFabric::modes() const noexcept {
    return modes_;
}

void KnowledgeFabric::import_record(KnowledgeRecord record) {
    if (record.stable_id == 0U) throw std::invalid_argument("imported record ID must be non-zero");
    insert(std::move(record));
}

void KnowledgeFabric::import_mode(ModeRecord mode) {
    validate_mode(mode);
    if (mode.stable_id == 0U || modes_.contains(mode.stable_id)) {
        throw std::invalid_argument("invalid or duplicate imported mode ID");
    }
    if (mode.parent_id.has_value() && !modes_.contains(*mode.parent_id)) {
        throw std::invalid_argument("imported mode parent must precede child");
    }
    next_mode_id_ = std::max(next_mode_id_, mode.stable_id + 1U);
    const std::uint64_t id = mode.stable_id;
    modes_.emplace(id, std::move(mode));
    if (modes_.at(id).parent_id.has_value()) {
        modes_.at(*modes_.at(id).parent_id).child_ids.insert(id);
    }
}

KnowledgeStatistics KnowledgeFabric::statistics() const {
    KnowledgeStatistics result;
    result.records = records_.size();
    result.modes = modes_.size();
    result.index_keys = exact_index_.size() + term_index_.size();
    result.last_candidates_examined = last_candidates_examined_;
    result.total_candidates_examined = total_candidates_examined_;
    for (const auto& [id, record] : records_) {
        static_cast<void>(id);
        switch (record.tier) {
        case MemoryTier::active: ++result.active_records; break;
        case MemoryTier::hot: ++result.hot_records; break;
        case MemoryTier::warm: ++result.warm_records; break;
        case MemoryTier::cold: ++result.cold_records; break;
        }
        result.contradictions += record.contradiction_count;
        if (record.stale) ++result.stale_records;
        if (record.invalidated) ++result.invalidated_records;
        result.approximate_persistent_bytes += approximate_record_bytes(record);
    }
    for (const auto& [id, mode] : modes_) {
        static_cast<void>(id);
        result.approximate_persistent_bytes += approximate_mode_bytes(mode);
    }
    return result;
}

std::uint64_t KnowledgeFabric::deterministic_hash() const noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, seed_);
    hash_u64(hash, step_);
    for (const auto& [id, record] : records_) {
        hash_u64(hash, id);
        hash_u64(hash, static_cast<std::uint64_t>(record.kind));
        hash_string(hash, record.subject);
        hash_string(hash, record.predicate);
        hash_string(hash, record.object);
        hash_string(hash, record.source);
        hash_double(hash, record.confidence);
        hash_double(hash, record.utility);
        hash_u64(hash, record.version);
        hash_u64(hash, record.use_count);
        hash_u64(hash, record.contradiction_count);
        hash_u64(hash, static_cast<std::uint64_t>(record.tier));
        hash_u64(hash, record.verified ? 1U : 0U);
        hash_u64(hash, record.stale ? 1U : 0U);
        hash_u64(hash, record.invalidated ? 1U : 0U);
    }
    for (const auto& [id, mode] : modes_) {
        hash_u64(hash, id);
        hash_u64(hash, static_cast<std::uint64_t>(mode.modality));
        hash_string(hash, mode.label);
        for (const float value : mode.prototype) {
            hash_u64(hash, std::bit_cast<std::uint32_t>(value));
        }
        hash_u64(hash, mode.parent_id.value_or(0U));
        hash_double(hash, mode.confidence);
        hash_double(hash, mode.utility);
        hash_u64(hash, mode.support);
        for (const auto child : mode.child_ids) hash_u64(hash, child);
        for (const auto linked : mode.linked_modes) hash_u64(hash, linked);
    }
    return hash;
}

}  // namespace rlf::frontier
