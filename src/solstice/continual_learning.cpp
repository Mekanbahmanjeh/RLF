#include "rlf/solstice/continual_learning.hpp"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace rlf::solstice {
namespace {

constexpr std::uint64_t fnv_offset = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;

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

[[nodiscard]] std::string normalize(const std::string_view value) {
    std::string result;
    result.reserve(value.size());
    for (const char raw : value) {
        const unsigned char character = static_cast<unsigned char>(raw);
        if (std::isspace(character) != 0) {
            if (!result.empty() && result.back() != ' ') {
                result.push_back(' ');
            }
        } else {
            result.push_back(static_cast<char>(std::tolower(character)));
        }
    }
    while (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }
    return result;
}

[[nodiscard]] double cosine_similarity(
    const std::span<const float> left,
    const std::span<const float> right
) noexcept {
    if (left.size() != right.size() || left.empty()) {
        return 0.0;
    }
    double dot = 0.0;
    double left_norm = 0.0;
    double right_norm = 0.0;
    for (std::size_t index = 0U; index < left.size(); ++index) {
        const double a = left[index];
        const double b = right[index];
        dot += a * b;
        left_norm += a * a;
        right_norm += b * b;
    }
    const double denominator = std::sqrt(left_norm * right_norm);
    if (denominator <= std::numeric_limits<double>::epsilon()) {
        return 0.0;
    }
    return std::clamp((dot / denominator + 1.0) * 0.5, 0.0, 1.0);
}

void validate_features(
    const std::span<const float> features,
    const std::size_t expected
) {
    if (features.size() != expected) {
        throw std::invalid_argument("continual feature dimension mismatch");
    }
    for (const float value : features) {
        if (!std::isfinite(static_cast<double>(value))) {
            throw std::invalid_argument("continual features contain non-finite value");
        }
    }
}

}  // namespace

ContinualLearningFabric::ContinualLearningFabric(ContinualLearningConfig config)
    : config_(std::move(config)), router_(config_.router) {
    if (config_.feature_dimensions == 0U ||
        config_.maximum_prototypes == 0U ||
        config_.replay_capacity == 0U ||
        config_.consolidation_interval == 0U ||
        config_.replay_batch_size == 0U ||
        !std::isfinite(config_.base_learning_rate) ||
        config_.base_learning_rate <= 0.0 ||
        config_.base_learning_rate > 1.0 ||
        !std::isfinite(config_.stability_strength) ||
        config_.stability_strength < 0.0 ||
        !std::isfinite(config_.novelty_threshold) ||
        config_.novelty_threshold < 0.0 ||
        config_.novelty_threshold > 1.0) {
        throw std::invalid_argument("invalid continual learning configuration");
    }
}

void ContinualLearningFabric::update_router() const {
    if (!router_dirty_) {
        return;
    }
    std::vector<float> matrix;
    matrix.reserve(prototypes_.size() * config_.feature_dimensions);
    for (const ContinualPrototype& prototype : prototypes_) {
        matrix.insert(
            matrix.end(), prototype.centroid.begin(), prototype.centroid.end()
        );
    }
    router_.rebuild(
        matrix,
        prototypes_.size(),
        config_.feature_dimensions
    );
    router_dirty_ = false;
}

ContinualPrediction ContinualLearningFabric::predict_internal(
    const std::string_view task,
    const std::span<const float> features,
    const bool allow_empty
) const {
    validate_features(features, config_.feature_dimensions);
    const std::string normalized_task = normalize(task);
    if (prototypes_.empty()) {
        if (allow_empty) {
            return {};
        }
        throw std::runtime_error("continual learning fabric has no prototypes");
    }
    update_router();
    const SparseRouteResult route = router_.route(features);
    std::vector<std::size_t> candidates = route.candidate_indices;
    const auto task_matches = [this, &normalized_task](const std::size_t index) {
        return normalized_task.empty() || prototypes_[index].task == normalized_task;
    };
    if (std::none_of(candidates.begin(), candidates.end(), task_matches)) {
        candidates.clear();
        for (std::size_t index = 0U; index < prototypes_.size(); ++index) {
            if (task_matches(index)) {
                candidates.push_back(index);
            }
        }
    }
    if (candidates.empty()) {
        if (allow_empty) {
            return {};
        }
        throw std::runtime_error("continual learning task has no prototypes");
    }
    std::size_t best_index = candidates.front();
    double best_similarity = -1.0;
    std::size_t examined = 0U;
    for (const std::size_t index : candidates) {
        if (!task_matches(index)) {
            continue;
        }
        ++examined;
        const double similarity = cosine_similarity(
            prototypes_[index].centroid, features
        );
        if (similarity > best_similarity ||
            (similarity == best_similarity &&
             prototypes_[index].id < prototypes_[best_index].id)) {
            best_similarity = similarity;
            best_index = index;
        }
    }
    if (examined == 0U) {
        return {};
    }
    return {
        prototypes_[best_index].label,
        std::max(0.0, best_similarity),
        1.0 - std::max(0.0, best_similarity),
        prototypes_[best_index].id,
        examined,
        static_cast<std::uint64_t>(prototypes_.size()),
    };
}

ContinualPrediction ContinualLearningFabric::predict(
    const std::string_view task,
    const std::span<const float> features
) const {
    return predict_internal(task, features, false);
}

void ContinualLearningFabric::apply_update(
    ContinualPrototype& prototype,
    const std::span<const float> features,
    const double sample_weight,
    const bool consolidation
) {
    const double stability = 1.0 +
        config_.stability_strength * prototype.importance;
    double learning_rate = config_.base_learning_rate * sample_weight /
        stability;
    if (consolidation) {
        learning_rate *= 0.25;
    }
    learning_rate = std::clamp(learning_rate, 0.0001, 1.0);
    for (std::size_t index = 0U; index < prototype.centroid.size(); ++index) {
        prototype.centroid[index] = static_cast<float>(
            static_cast<double>(prototype.centroid[index]) +
            learning_rate *
                (static_cast<double>(features[index]) -
                 static_cast<double>(prototype.centroid[index]))
        );
    }
    ++prototype.support;
    prototype.importance = std::clamp(
        prototype.importance +
            (1.0 - prototype.importance) /
                static_cast<double>(prototype.support + 1U),
        0.0,
        1.0
    );
    prototype.plasticity = 1.0 / stability;
    prototype.last_update_step = step_;
    router_dirty_ = true;
}

void ContinualLearningFabric::add_replay(
    const std::string_view task,
    const std::string_view label,
    const std::span<const float> features,
    const double priority
) {
    ReplayExperience experience;
    experience.id = next_experience_id_++;
    experience.task = std::string(task);
    experience.label = std::string(label);
    experience.features.assign(features.begin(), features.end());
    experience.priority = std::clamp(priority, 0.0, 10.0);
    experience.seen_step = step_;
    if (replay_.size() < config_.replay_capacity) {
        replay_.push_back(std::move(experience));
        return;
    }
    std::map<std::pair<std::string, std::string>, std::size_t> counts;
    for (const ReplayExperience& item : replay_) {
        ++counts[{item.task, item.label}];
    }
    const auto new_key = std::make_pair(experience.task, experience.label);
    std::size_t replace_index = replay_.size();
    double weakest_score = std::numeric_limits<double>::infinity();
    for (std::size_t index = 0U; index < replay_.size(); ++index) {
        const ReplayExperience& current = replay_[index];
        const auto current_key = std::make_pair(current.task, current.label);
        const double age = static_cast<double>(step_ - current.seen_step + 1U);
        const double overrepresentation = static_cast<double>(counts[current_key]);
        const double score = current.priority / age -
            0.001 * overrepresentation;
        if (score < weakest_score &&
            (counts[current_key] > counts[new_key] ||
             experience.priority > current.priority)) {
            weakest_score = score;
            replace_index = index;
        }
    }
    if (replace_index < replay_.size()) {
        replay_[replace_index] = std::move(experience);
    }
}

ContinualPrediction ContinualLearningFabric::learn(
    const std::string_view task,
    const std::string_view label,
    const std::span<const float> features,
    const double sample_weight
) {
    validate_features(features, config_.feature_dimensions);
    if (!std::isfinite(sample_weight) || sample_weight <= 0.0) {
        throw std::invalid_argument("sample weight must be positive and finite");
    }
    const std::string normalized_task = normalize(task);
    const std::string normalized_label = normalize(label);
    if (normalized_task.empty() || normalized_label.empty()) {
        throw std::invalid_argument("task and label must be non-empty");
    }
    ++step_;
    const ContinualPrediction before = predict_internal(
        normalized_task, features, true
    );
    std::size_t best_same_index = prototypes_.size();
    double best_same_similarity = -1.0;
    for (std::size_t index = 0U; index < prototypes_.size(); ++index) {
        const ContinualPrototype& prototype = prototypes_[index];
        if (prototype.task != normalized_task ||
            prototype.label != normalized_label) {
            continue;
        }
        const double similarity = cosine_similarity(
            prototype.centroid, features
        );
        if (similarity > best_same_similarity) {
            best_same_similarity = similarity;
            best_same_index = index;
        }
    }
    if (best_same_index == prototypes_.size() ||
        best_same_similarity < config_.novelty_threshold) {
        if (prototypes_.size() >= config_.maximum_prototypes) {
            throw std::runtime_error("continual prototype capacity exceeded");
        }
        ContinualPrototype prototype;
        prototype.id = next_prototype_id_++;
        prototype.task = normalized_task;
        prototype.label = normalized_label;
        prototype.centroid.assign(features.begin(), features.end());
        prototype.importance = 0.05;
        prototype.last_update_step = step_;
        prototypes_.push_back(std::move(prototype));
        router_dirty_ = true;
    } else {
        apply_update(
            prototypes_[best_same_index], features, sample_weight, false
        );
    }
    if (!before.label.empty() && before.label != normalized_label &&
        before.prototype_id != 0U && config_.contrastive_margin > 0.0) {
        const auto competitor = std::find_if(
            prototypes_.begin(), prototypes_.end(),
            [&before](const ContinualPrototype& prototype) {
                return prototype.id == before.prototype_id;
            }
        );
        if (competitor != prototypes_.end()) {
            const double rate = std::clamp(
                config_.base_learning_rate * config_.contrastive_margin *
                    competitor->plasticity,
                0.0,
                0.25
            );
            for (std::size_t index = 0U;
                 index < competitor->centroid.size();
                 ++index) {
                competitor->centroid[index] = static_cast<float>(
                    static_cast<double>(competitor->centroid[index]) -
                    rate *
                        (static_cast<double>(features[index]) -
                         static_cast<double>(competitor->centroid[index]))
                );
            }
            competitor->last_update_step = step_;
            router_dirty_ = true;
        }
    }
    const double error_priority = before.label.empty()
        ? 1.0
        : (before.label == normalized_label
            ? 1.0 - before.confidence
            : 1.0 + before.confidence);
    add_replay(
        normalized_task,
        normalized_label,
        features,
        error_priority
    );
    if (step_ % config_.consolidation_interval == 0U) {
        consolidate();
    }
    return predict_internal(normalized_task, features, false);
}

void ContinualLearningFabric::consolidate() {
    if (replay_.empty() || prototypes_.empty()) {
        return;
    }
    std::vector<std::size_t> order(replay_.size());
    std::iota(order.begin(), order.end(), 0U);
    std::sort(
        order.begin(), order.end(),
        [this](const std::size_t left, const std::size_t right) {
            const ReplayExperience& a = replay_[left];
            const ReplayExperience& b = replay_[right];
            if (a.priority != b.priority) {
                return a.priority > b.priority;
            }
            return a.id < b.id;
        }
    );
    const std::size_t count = std::min(
        config_.replay_batch_size, order.size()
    );
    for (std::size_t rank = 0U; rank < count; ++rank) {
        const ReplayExperience& experience = replay_[order[rank]];
        std::size_t best_index = prototypes_.size();
        double best_similarity = -1.0;
        for (std::size_t index = 0U; index < prototypes_.size(); ++index) {
            const ContinualPrototype& prototype = prototypes_[index];
            if (prototype.task != experience.task ||
                prototype.label != experience.label) {
                continue;
            }
            const double similarity = cosine_similarity(
                prototype.centroid, experience.features
            );
            if (similarity > best_similarity) {
                best_similarity = similarity;
                best_index = index;
            }
        }
        if (best_index < prototypes_.size()) {
            apply_update(
                prototypes_[best_index],
                experience.features,
                1.0 + experience.priority,
                true
            );
        }
    }
    for (ContinualPrototype& prototype : prototypes_) {
        prototype.importance = std::clamp(
            prototype.importance * 0.999 + 0.001,
            0.0,
            1.0
        );
        prototype.plasticity = 1.0 /
            (1.0 + config_.stability_strength * prototype.importance);
    }
    ++consolidations_;
}

std::span<const ContinualPrototype>
ContinualLearningFabric::prototypes() const noexcept {
    return prototypes_;
}

std::span<const ReplayExperience>
ContinualLearningFabric::replay() const noexcept {
    return replay_;
}

const ContinualLearningConfig&
ContinualLearningFabric::config() const noexcept {
    return config_;
}

ContinualLearningStats ContinualLearningFabric::stats() const noexcept {
    double importance = 0.0;
    double plasticity = 0.0;
    for (const ContinualPrototype& prototype : prototypes_) {
        importance += prototype.importance;
        plasticity += prototype.plasticity;
    }
    const double denominator = prototypes_.empty()
        ? 1.0
        : static_cast<double>(prototypes_.size());
    return {
        prototypes_.size(),
        replay_.size(),
        step_,
        consolidations_,
        importance / denominator,
        plasticity / denominator,
    };
}

std::uint64_t ContinualLearningFabric::deterministic_hash() const noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, next_prototype_id_);
    hash_u64(hash, next_experience_id_);
    hash_u64(hash, step_);
    for (const ContinualPrototype& prototype : prototypes_) {
        hash_u64(hash, prototype.id);
        hash_string(hash, prototype.task);
        hash_string(hash, prototype.label);
        for (const float value : prototype.centroid) {
            hash_u64(hash, std::bit_cast<std::uint32_t>(value));
        }
        hash_u64(hash, prototype.support);
        hash_u64(hash, std::bit_cast<std::uint64_t>(prototype.importance));
    }
    return hash;
}

ContinualLearningSnapshot ContinualLearningFabric::snapshot() const {
    return {
        config_, next_prototype_id_, next_experience_id_, step_,
        consolidations_, prototypes_, replay_,
    };
}

ContinualLearningFabric ContinualLearningFabric::from_snapshot(
    ContinualLearningSnapshot snapshot
) {
    ContinualLearningFabric fabric(snapshot.config);
    fabric.next_prototype_id_ = snapshot.next_prototype_id;
    fabric.next_experience_id_ = snapshot.next_experience_id;
    fabric.step_ = snapshot.step;
    fabric.consolidations_ = snapshot.consolidations;
    fabric.prototypes_ = std::move(snapshot.prototypes);
    fabric.replay_ = std::move(snapshot.replay);
    fabric.router_dirty_ = true;
    return fabric;
}

}  // namespace rlf::solstice
