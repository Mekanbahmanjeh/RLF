#include "rlf/retrieval/contiguous_mode_index.hpp"

#include "rlf/backend/compute_backend.hpp"
#include "rlf/core/phase_vector.hpp"
#include "rlf/core/resonant_mode.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace rlf::retrieval {
namespace {

[[nodiscard]] bool strongest_first(
    const RetrievedMode& left,
    const RetrievedMode& right
) {
    if (left.resonance != right.resonance) {
        return left.resonance > right.resonance;
    }
    if (left.mode_id != right.mode_id) {
        return left.mode_id < right.mode_id;
    }
    return left.mode_index < right.mode_index;
}

void retain_top(
    std::vector<RetrievedMode>& candidates,
    const std::size_t count
) {
    if (candidates.size() > count) {
        std::partial_sort(
            candidates.begin(),
            candidates.begin() + static_cast<std::ptrdiff_t>(count),
            candidates.end(),
            strongest_first
        );
        candidates.resize(count);
    } else {
        std::sort(candidates.begin(), candidates.end(), strongest_first);
    }
}

}  // namespace

ContiguousModeIndex::ContiguousModeIndex(
    const std::span<const core::ResonantMode> modes,
    std::shared_ptr<const backend::ComputeBackend> backend
)
    : dimension_(modes.empty() ? 0U : modes.front().context_key.size()),
      backend_(std::move(backend)) {
    if (modes.empty()) {
        throw std::invalid_argument(
            "contiguous mode index requires at least one mode"
        );
    }
    if (!backend_ || !backend_->available()) {
        throw std::invalid_argument(
            "contiguous mode index requires an available backend"
        );
    }
    context_angles_.reserve(modes.size() * dimension_);
    mode_ids_.reserve(modes.size());
    source_indices_.reserve(modes.size());
    selectivities_.reserve(modes.size());
    for (std::size_t index = 0U; index < modes.size(); ++index) {
        const core::ResonantMode& mode = modes[index];
        if (mode.context_key.size() != dimension_) {
            throw std::invalid_argument(
                "contiguous mode index dimensions must match"
            );
        }
        if (!mode.enabled) {
            continue;
        }
        context_angles_.insert(
            context_angles_.end(),
            mode.context_key.angles().begin(),
            mode.context_key.angles().end()
        );
        mode_ids_.push_back(mode.id);
        source_indices_.push_back(index);
        selectivities_.push_back(mode.selectivity);
    }
    if (mode_ids_.empty()) {
        throw std::invalid_argument(
            "contiguous mode index requires an enabled mode"
        );
    }
}

std::size_t ContiguousModeIndex::dimension() const noexcept {
    return dimension_;
}

std::size_t ContiguousModeIndex::size() const noexcept {
    return mode_ids_.size();
}

std::size_t ContiguousModeIndex::bytes_stored() const noexcept {
    return sizeof(*this) +
        (context_angles_.capacity() * sizeof(float)) +
        (mode_ids_.capacity() * sizeof(std::uint64_t)) +
        (source_indices_.capacity() * sizeof(std::size_t)) +
        (selectivities_.capacity() * sizeof(float));
}

std::vector<RetrievedMode> ContiguousModeIndex::retrieve(
    const core::PhaseVector& state,
    const std::size_t candidate_count,
    const std::size_t thread_count
) const {
    if (state.size() != dimension_) {
        throw std::invalid_argument(
            "contiguous retrieval dimension mismatch"
        );
    }
    if (thread_count == 0U) {
        throw std::invalid_argument(
            "contiguous retrieval thread count must be positive"
        );
    }
    if (candidate_count == 0U) {
        return {};
    }
    const std::size_t worker_count =
        std::min(thread_count, size());
    std::vector<std::vector<RetrievedMode>> local(worker_count);
    std::vector<std::jthread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker_index = 0U;
         worker_index < worker_count;
         ++worker_index) {
        const std::size_t begin =
            (worker_index * size()) / worker_count;
        const std::size_t end =
            ((worker_index + 1U) * size()) / worker_count;
        workers.emplace_back([&, worker_index, begin, end] {
            std::vector<RetrievedMode>& output = local[worker_index];
            output.reserve(std::min(candidate_count, end - begin));
            for (std::size_t compact_index = begin;
                 compact_index < end;
                 ++compact_index) {
                const std::span<const float> context(
                    context_angles_.data() +
                        (compact_index * dimension_),
                    dimension_
                );
                const double base = backend_->similarity_angles(
                    context,
                    state.angles()
                );
                output.push_back({
                    .mode_index = source_indices_[compact_index],
                    .mode_id = mode_ids_[compact_index],
                    .resonance = std::clamp(
                        std::pow(
                            base,
                            static_cast<double>(
                                selectivities_[compact_index]
                            )
                        ),
                        0.0,
                        1.0
                    ),
                });
            }
            retain_top(output, candidate_count);
        });
    }
    workers.clear();
    std::vector<RetrievedMode> result;
    result.reserve(candidate_count * worker_count);
    for (std::vector<RetrievedMode>& output : local) {
        result.insert(
            result.end(),
            std::make_move_iterator(output.begin()),
            std::make_move_iterator(output.end())
        );
    }
    retain_top(result, candidate_count);
    return result;
}

}  // namespace rlf::retrieval
