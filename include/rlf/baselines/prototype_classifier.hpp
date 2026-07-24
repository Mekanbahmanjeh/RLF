#pragma once

#include "rlf/core/phase_vector.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

namespace rlf::baselines {

struct PrototypePrediction final {
    std::uint64_t label;
    double similarity;
};

class FixedPrototypeClassifier final {
public:
    explicit FixedPrototypeClassifier(std::size_t dimension);

    void observe(
        std::uint64_t label,
        const core::PhaseVector& sample
    );
    [[nodiscard]] std::optional<PrototypePrediction> predict(
        const core::PhaseVector& sample
    ) const;
    [[nodiscard]] std::size_t classes() const noexcept;
    [[nodiscard]] std::size_t bytes_stored() const noexcept;

private:
    struct ClassSamples final {
        std::vector<core::PhaseVector> samples;
        core::PhaseVector prototype;
    };

    std::size_t dimension_;
    std::map<std::uint64_t, ClassSamples> classes_;
};

}  // namespace rlf::baselines
