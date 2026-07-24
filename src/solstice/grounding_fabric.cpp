#include "rlf/solstice/grounding_fabric.hpp"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace rlf::solstice {
namespace {

constexpr std::uint64_t fnv_offset = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;

[[nodiscard]] bool persistent_link_index_from_environment() {
    const char* const value = std::getenv("RLF_GROUNDING_INDEX_POLICY");
    if (value == nullptr || std::string_view(value).empty() ||
        std::string_view(value) == "persistent") {
        return true;
    }
    if (std::string_view(value) == "rebuild") {
        return false;
    }
    throw std::invalid_argument(
        "RLF_GROUNDING_INDEX_POLICY must be persistent or rebuild"
    );
}

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

}  // namespace

CrossModalGroundingFabric::CrossModalGroundingFabric(GroundingConfig config)
    : config_(std::move(config)),
      persistent_link_index_(persistent_link_index_from_environment()) {
    if (config_.maximum_links == 0U || config_.maximum_concepts == 0U ||
        config_.maximum_results == 0U ||
        !std::isfinite(config_.smoothing) || config_.smoothing <= 0.0 ||
        !std::isfinite(config_.minimum_score) ||
        config_.minimum_score < 0.0 || config_.minimum_score > 1.0 ||
        !std::isfinite(config_.negative_weight) ||
        config_.negative_weight < 0.0) {
        throw std::invalid_argument("invalid grounding configuration");
    }
}

std::string CrossModalGroundingFabric::normalize_concept(
    const std::string_view concept_name
) {
    std::string result;
    result.reserve(concept_name.size());
    bool pending_space = false;
    for (const char raw : concept_name) {
        const unsigned char character = static_cast<unsigned char>(raw);
        if (std::isalnum(character) != 0 || character == '_') {
            if (pending_space && !result.empty()) {
                result.push_back(' ');
            }
            pending_space = false;
            result.push_back(static_cast<char>(std::tolower(character)));
        } else {
            pending_space = !result.empty();
        }
    }
    return result;
}

double CrossModalGroundingFabric::link_score(
    const GroundingLink& link
) const noexcept {
    const double positive = static_cast<double>(link.positive_count) +
        config_.smoothing;
    const double negative = config_.negative_weight *
        static_cast<double>(link.negative_count) + config_.smoothing;
    const double log_odds = std::log(positive / negative);
    const double support = std::log1p(
        static_cast<double>(link.positive_count + link.negative_count)
    );
    return std::clamp(
        1.0 / (1.0 + std::exp(-(log_odds + 0.15 * support))),
        0.0,
        1.0
    );
}

void CrossModalGroundingFabric::observe(
    const std::span<const std::uint64_t> visual_mode_ids,
    const std::span<const std::string> positive_concepts,
    const std::span<const std::string> negative_concepts
) {
    if (visual_mode_ids.empty() || positive_concepts.empty()) {
        return;
    }
    std::set<std::uint64_t> modes(
        visual_mode_ids.begin(), visual_mode_ids.end()
    );
    std::set<std::string> positives;
    std::set<std::string> negatives;
    for (const std::string& concept_name : positive_concepts) {
        const std::string normalized = normalize_concept(concept_name);
        if (!normalized.empty()) {
            positives.insert(normalized);
        }
    }
    for (const std::string& concept_name : negative_concepts) {
        const std::string normalized = normalize_concept(concept_name);
        if (!normalized.empty() && !positives.contains(normalized)) {
            negatives.insert(normalized);
        }
    }
    if (positives.empty()) {
        return;
    }
    std::map<std::pair<std::uint64_t, std::string>, std::size_t> rebuilt_index;
    if (!persistent_link_index_) {
        for (std::size_t link_index = 0U; link_index < links_.size(); ++link_index) {
            rebuilt_index[{links_[link_index].visual_mode_id,
                           links_[link_index].concept_name}] = link_index;
        }
        operation_stats_.full_lookup_entries_rebuilt += links_.size();
    }
    std::vector<std::size_t> touched_links;
    touched_links.reserve(modes.size() * (positives.size() + negatives.size()));
    const bool sparse_confidence_update =
        persistent_link_index_ && confidence_cache_valid_;
    confidence_cache_valid_ = false;
    const auto find_link = [this, &rebuilt_index](
        const std::uint64_t mode,
        const std::string& concept_name
    ) -> std::size_t {
        ++operation_stats_.link_lookups;
        if (!persistent_link_index_) {
            const auto iterator = rebuilt_index.find({mode, concept_name});
            return iterator == rebuilt_index.end()
                ? links_.size()
                : iterator->second;
        }
        const auto posting = link_indices_by_mode_.find(mode);
        if (posting == link_indices_by_mode_.end()) {
            return links_.size();
        }
        for (auto iterator = posting->second.rbegin();
             iterator != posting->second.rend();
             ++iterator) {
            ++operation_stats_.indexed_link_candidates_examined;
            const std::size_t link_index = *iterator;
            if (links_[link_index].concept_name == concept_name) {
                return link_index;
            }
        }
        return links_.size();
    };
    const auto insert_link = [this, &rebuilt_index](const std::size_t link_index) {
        const GroundingLink& link = links_[link_index];
        if (!persistent_link_index_) {
            rebuilt_index.emplace(
                std::make_pair(link.visual_mode_id, link.concept_name),
                link_index
            );
            return;
        }
        link_indices_by_mode_[link.visual_mode_id].push_back(link_index);
        link_indices_by_concept_[link.concept_name].push_back(link_index);
        indexed_link_count_ = links_.size();
        operation_stats_.incremental_posting_inserts += 2U;
    };
    for (const std::uint64_t mode : modes) {
        for (const std::string& concept_name : positives) {
            std::size_t link_index = find_link(mode, concept_name);
            if (link_index == links_.size()) {
                if (links_.size() >= config_.maximum_links) {
                    throw std::runtime_error("grounding link capacity exceeded");
                }
                GroundingLink link;
                link.visual_mode_id = mode;
                link.concept_name = concept_name;
                link.positive_count = 1U;
                links_.push_back(std::move(link));
                link_index = links_.size() - 1U;
                insert_link(link_index);
            } else {
                ++links_[link_index].positive_count;
            }
            touched_links.push_back(link_index);
        }
        for (const std::string& concept_name : negatives) {
            std::size_t link_index = find_link(mode, concept_name);
            if (link_index == links_.size()) {
                if (links_.size() >= config_.maximum_links) {
                    throw std::runtime_error("grounding link capacity exceeded");
                }
                GroundingLink link;
                link.visual_mode_id = mode;
                link.concept_name = concept_name;
                link.negative_count = 1U;
                links_.push_back(std::move(link));
                link_index = links_.size() - 1U;
                insert_link(link_index);
            } else {
                ++links_[link_index].negative_count;
            }
            touched_links.push_back(link_index);
        }
    }
    if (sparse_confidence_update) {
        for (const std::size_t link_index : touched_links) {
            links_[link_index].confidence = link_score(links_[link_index]);
        }
        operation_stats_.confidence_recomputations += touched_links.size();
    } else {
        for (GroundingLink& link : links_) {
            link.confidence = link_score(link);
        }
        operation_stats_.confidence_recomputations += links_.size();
        operation_stats_.full_confidence_sweep_entries += links_.size();
    }
    confidence_cache_valid_ = true;
    ++observations_;
    if (!persistent_link_index_) {
        rebuild_index();
    }
}

std::vector<GroundingHit> CrossModalGroundingFabric::concepts_for_mode(
    const std::uint64_t visual_mode_id,
    const std::size_t maximum_results
) const {
    const std::size_t limit = maximum_results == 0U
        ? config_.maximum_results
        : maximum_results;
    std::vector<GroundingHit> hits;
    const auto collect = [this, visual_mode_id, &hits](const std::size_t index) {
        const GroundingLink& link = links_[index];
        if (link.visual_mode_id != visual_mode_id) {
            return;
        }
        const double score = link_score(link);
        if (score < config_.minimum_score) {
            return;
        }
        hits.push_back({
            link.visual_mode_id,
            link.concept_name,
            score,
            link.positive_count + link.negative_count,
        });
    };
    if (persistent_link_index_) {
        const auto posting = link_indices_by_mode_.find(visual_mode_id);
        if (posting != link_indices_by_mode_.end()) {
            operation_stats_.mode_query_indexed_candidates += posting->second.size();
            for (const std::size_t index : posting->second) {
                collect(index);
            }
        }
    } else {
        operation_stats_.mode_query_full_scan_entries += links_.size();
        for (std::size_t index = 0U; index < links_.size(); ++index) {
            collect(index);
        }
    }
    std::sort(
        hits.begin(), hits.end(),
        [](const GroundingHit& left, const GroundingHit& right) {
            if (left.score != right.score) {
                return left.score > right.score;
            }
            if (left.support != right.support) {
                return left.support > right.support;
            }
            return left.concept_name < right.concept_name;
        }
    );
    if (hits.size() > limit) {
        hits.resize(limit);
    }
    return hits;
}

std::vector<GroundingHit> CrossModalGroundingFabric::modes_for_concept(
    const std::string_view concept_name,
    const std::size_t maximum_results
) const {
    const std::string normalized = normalize_concept(concept_name);
    const std::size_t limit = maximum_results == 0U
        ? config_.maximum_results
        : maximum_results;
    std::vector<GroundingHit> hits;
    const auto collect = [this, &normalized, &hits](const std::size_t index) {
        const GroundingLink& link = links_[index];
        if (link.concept_name != normalized) {
            return;
        }
        const double score = link_score(link);
        if (score < config_.minimum_score) {
            return;
        }
        hits.push_back({
            link.visual_mode_id,
            link.concept_name,
            score,
            link.positive_count + link.negative_count,
        });
    };
    if (persistent_link_index_) {
        const auto posting = link_indices_by_concept_.find(normalized);
        if (posting != link_indices_by_concept_.end()) {
            operation_stats_.concept_query_indexed_candidates += posting->second.size();
            for (const std::size_t index : posting->second) {
                collect(index);
            }
        }
    } else {
        operation_stats_.concept_query_full_scan_entries += links_.size();
        for (std::size_t index = 0U; index < links_.size(); ++index) {
            collect(index);
        }
    }
    std::sort(
        hits.begin(), hits.end(),
        [](const GroundingHit& left, const GroundingHit& right) {
            if (left.score != right.score) {
                return left.score > right.score;
            }
            return left.visual_mode_id < right.visual_mode_id;
        }
    );
    if (hits.size() > limit) {
        hits.resize(limit);
    }
    return hits;
}

std::vector<GroundingHit> CrossModalGroundingFabric::compose_concepts(
    const std::span<const std::string> concepts,
    const std::size_t maximum_results
) const {
    const std::size_t limit = maximum_results == 0U
        ? config_.maximum_results
        : maximum_results;
    if (concepts.empty()) {
        return {};
    }
    std::map<std::uint64_t, std::pair<double, std::size_t>> scores;
    std::vector<std::string> normalized_concepts;
    for (const std::string& concept_name : concepts) {
        const std::string normalized = normalize_concept(concept_name);
        if (!normalized.empty()) {
            normalized_concepts.push_back(normalized);
        }
    }
    if (normalized_concepts.empty()) {
        return {};
    }
    for (const std::string& concept_name : normalized_concepts) {
        for (const GroundingHit& hit : modes_for_concept(
                 concept_name, config_.maximum_links)) {
            auto& entry = scores[hit.visual_mode_id];
            entry.first += std::log(std::max(hit.score, 1.0e-9));
            ++entry.second;
        }
    }
    std::vector<GroundingHit> hits;
    for (const auto& [mode, aggregate] : scores) {
        const double coverage = static_cast<double>(aggregate.second) /
            static_cast<double>(normalized_concepts.size());
        const double geometric = std::exp(
            aggregate.first /
            static_cast<double>(std::max<std::size_t>(aggregate.second, 1U))
        );
        hits.push_back({
            mode,
            normalized_concepts.front(),
            geometric * coverage,
            aggregate.second,
        });
    }
    std::sort(
        hits.begin(), hits.end(),
        [](const GroundingHit& left, const GroundingHit& right) {
            if (left.score != right.score) {
                return left.score > right.score;
            }
            return left.visual_mode_id < right.visual_mode_id;
        }
    );
    if (hits.size() > limit) {
        hits.resize(limit);
    }
    return hits;
}

std::span<const GroundingLink>
CrossModalGroundingFabric::links() const noexcept {
    return links_;
}

const GroundingConfig&
CrossModalGroundingFabric::config() const noexcept {
    return config_;
}

std::uint64_t CrossModalGroundingFabric::observations() const noexcept {
    return observations_;
}

std::uint64_t CrossModalGroundingFabric::deterministic_hash() const noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, observations_);
    for (const GroundingLink& link : links_) {
        hash_u64(hash, link.visual_mode_id);
        hash_string(hash, link.concept_name);
        hash_u64(hash, link.positive_count);
        hash_u64(hash, link.negative_count);
        hash_u64(hash, std::bit_cast<std::uint64_t>(link.confidence));
    }
    return hash;
}

void CrossModalGroundingFabric::set_persistent_link_index(const bool enabled) {
    persistent_link_index_ = enabled;
    if (enabled && indexed_link_count_ != links_.size()) {
        rebuild_index();
    }
}

GroundingOperationStats CrossModalGroundingFabric::operation_stats() const noexcept {
    return operation_stats_;
}

GroundingSnapshot CrossModalGroundingFabric::snapshot() const {
    return {config_, observations_, links_};
}

CrossModalGroundingFabric CrossModalGroundingFabric::from_snapshot(
    GroundingSnapshot snapshot
) {
    CrossModalGroundingFabric fabric(snapshot.config);
    fabric.observations_ = snapshot.observations;
    fabric.links_ = std::move(snapshot.links);
    fabric.rebuild_index();
    fabric.confidence_cache_valid_ = std::all_of(
        fabric.links_.begin(), fabric.links_.end(),
        [&fabric](const GroundingLink& link) {
            return link.confidence == fabric.link_score(link);
        }
    );
    return fabric;
}

void CrossModalGroundingFabric::rebuild_index() {
    link_indices_by_mode_.clear();
    link_indices_by_concept_.clear();
    order_by_mode_.resize(links_.size());
    order_by_concept_.resize(links_.size());
    for (std::size_t index = 0U; index < links_.size(); ++index) {
        order_by_mode_[index] = index;
        order_by_concept_[index] = index;
        link_indices_by_mode_[links_[index].visual_mode_id].push_back(index);
        link_indices_by_concept_[links_[index].concept_name].push_back(index);
    }
    std::sort(
        order_by_mode_.begin(), order_by_mode_.end(),
        [this](const std::size_t left, const std::size_t right) {
            const GroundingLink& a = links_[left];
            const GroundingLink& b = links_[right];
            if (a.visual_mode_id != b.visual_mode_id) {
                return a.visual_mode_id < b.visual_mode_id;
            }
            return a.concept_name < b.concept_name;
        }
    );
    std::sort(
        order_by_concept_.begin(), order_by_concept_.end(),
        [this](const std::size_t left, const std::size_t right) {
            const GroundingLink& a = links_[left];
            const GroundingLink& b = links_[right];
            if (a.concept_name != b.concept_name) {
                return a.concept_name < b.concept_name;
            }
            return a.visual_mode_id < b.visual_mode_id;
        }
    );
    indexed_link_count_ = links_.size();
    operation_stats_.derived_sort_entries += 2U * links_.size();
}

}  // namespace rlf::solstice
