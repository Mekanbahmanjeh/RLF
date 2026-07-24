#include "rlf/baselines/prototype_classifier.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rlf::baselines {

FixedPrototypeClassifier::FixedPrototypeClassifier(
    const std::size_t dimension
)
    : dimension_(dimension) {
    if (dimension_ == 0U) {
        throw std::invalid_argument(
            "prototype-classifier dimension must be positive"
        );
    }
}

void FixedPrototypeClassifier::observe(
    const std::uint64_t label,
    const core::PhaseVector& sample
) {
    if (sample.size() != dimension_) {
        throw std::invalid_argument(
            "prototype sample dimension must match the classifier"
        );
    }
    const auto found = classes_.find(label);
    if (found == classes_.end()) {
        classes_.emplace(
            label,
            ClassSamples{
                .samples = {sample},
                .prototype = sample,
            }
        );
        return;
    }
    ClassSamples& class_samples = found->second;
    class_samples.samples.push_back(sample);
    std::vector<float> weights(
        class_samples.samples.size(),
        1.0F
    );
    class_samples.prototype =
        core::PhaseVector::weighted_circular_average(
            class_samples.samples,
            weights
        );
}

std::optional<PrototypePrediction>
FixedPrototypeClassifier::predict(
    const core::PhaseVector& sample
) const {
    if (sample.size() != dimension_) {
        throw std::invalid_argument(
            "prototype query dimension must match the classifier"
        );
    }
    if (classes_.empty()) {
        return std::nullopt;
    }
    auto found = classes_.begin();
    std::uint64_t best_label = found->first;
    double best_similarity =
        sample.similarity(found->second.prototype);
    ++found;
    for (; found != classes_.end(); ++found) {
        const double similarity =
            sample.similarity(found->second.prototype);
        if (similarity > best_similarity ||
            (similarity == best_similarity &&
             found->first < best_label)) {
            best_label = found->first;
            best_similarity = similarity;
        }
    }
    return PrototypePrediction{
        .label = best_label,
        .similarity = best_similarity,
    };
}

std::size_t FixedPrototypeClassifier::classes() const noexcept {
    return classes_.size();
}

std::size_t FixedPrototypeClassifier::bytes_stored() const noexcept {
    std::size_t bytes = sizeof(*this);
    for (const auto& [label, class_samples] : classes_) {
        static_cast<void>(label);
        bytes += sizeof(ClassSamples) + sizeof(std::uint64_t);
        bytes += class_samples.prototype.size() *
            sizeof(core::PhaseVector::Angle);
        for (const core::PhaseVector& sample :
             class_samples.samples) {
            bytes += sizeof(core::PhaseVector);
            bytes += sample.size() *
                sizeof(core::PhaseVector::Angle);
        }
    }
    return bytes;
}

}  // namespace rlf::baselines
