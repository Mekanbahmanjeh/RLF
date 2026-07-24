#include "rlf/baselines/transition_table.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace rlf::baselines {

void TransitionTablePredictor::observe(
    const std::uint64_t current,
    const std::uint64_t next
) {
    ++counts_[current][next];
    ++observations_;
}

std::optional<std::uint64_t> TransitionTablePredictor::predict(
    const std::uint64_t current
) const {
    const auto context = counts_.find(current);
    if (context == counts_.end() || context->second.empty()) {
        return std::nullopt;
    }
    std::uint64_t best_next = context->second.begin()->first;
    std::uint64_t best_count = context->second.begin()->second;
    for (const auto& [next, count] : context->second) {
        if (count > best_count ||
            (count == best_count && next < best_next)) {
            best_next = next;
            best_count = count;
        }
    }
    return best_next;
}

std::vector<std::pair<std::uint64_t, std::uint64_t>>
TransitionTablePredictor::counts(
    const std::uint64_t current
) const {
    const auto context = counts_.find(current);
    if (context == counts_.end()) {
        return {};
    }
    return {context->second.begin(), context->second.end()};
}

std::size_t TransitionTablePredictor::contexts() const noexcept {
    return counts_.size();
}

std::size_t TransitionTablePredictor::transitions() const noexcept {
    std::size_t result = 0U;
    for (const auto& [current, next_counts] : counts_) {
        static_cast<void>(current);
        result += next_counts.size();
    }
    return result;
}

std::size_t TransitionTablePredictor::bytes_stored() const noexcept {
    constexpr std::size_t approximate_map_node_overhead =
        sizeof(void*) * 4U;
    return sizeof(*this) +
        (contexts() *
         (sizeof(std::uint64_t) +
          sizeof(std::map<std::uint64_t, std::uint64_t>) +
          approximate_map_node_overhead)) +
        (transitions() *
         ((2U * sizeof(std::uint64_t)) +
          approximate_map_node_overhead));
}

}  // namespace rlf::baselines
