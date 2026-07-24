#include "rlf/retrieval/mode_retriever.hpp"

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

[[nodiscard]] double mode_resonance(
    const core::PhaseVector& state,
    const core::ResonantMode& mode,
    const backend::ComputeBackend& backend
) {
    const double base = backend.similarity(mode.context_key, state);
    return std::clamp(
        std::pow(base, static_cast<double>(mode.selectivity)),
        0.0,
        1.0
    );
}

void retain_top_candidates(
    std::vector<RetrievedMode>& candidates,
    const std::size_t candidate_count
) {
    if (candidates.size() > candidate_count) {
        std::partial_sort(
            candidates.begin(),
            candidates.begin() +
                static_cast<std::ptrdiff_t>(candidate_count),
            candidates.end(),
            strongest_first
        );
        candidates.resize(candidate_count);
    } else {
        std::sort(
            candidates.begin(),
            candidates.end(),
            strongest_first
        );
    }
}

}  // namespace

ExactModeRetriever::ExactModeRetriever()
    : ExactModeRetriever(
          backend::make_backend(backend::BackendKind::scalar_cpu)
      ) {}

ExactModeRetriever::ExactModeRetriever(
    std::shared_ptr<const backend::ComputeBackend> backend
)
    : backend_(std::move(backend)) {
    if (!backend_ || !backend_->available()) {
        throw std::invalid_argument(
            "exact retrieval requires an available compute backend"
        );
    }
}

std::vector<RetrievedMode> ExactModeRetriever::retrieve(
    const core::PhaseVector& state,
    const std::span<const core::ResonantMode> modes,
    const std::size_t candidate_count
) const {
    if (candidate_count == 0U) {
        return {};
    }

    std::vector<RetrievedMode> candidates;
    candidates.reserve(std::min(candidate_count, modes.size()));
    for (std::size_t mode_index = 0U;
         mode_index < modes.size();
         ++mode_index) {
        const core::ResonantMode& mode = modes[mode_index];
        if (!mode.enabled) {
            continue;
        }
        candidates.push_back({
            .mode_index = mode_index,
            .mode_id = mode.id,
            .resonance =
                mode_resonance(state, mode, *backend_),
        });
    }
    retain_top_candidates(candidates, candidate_count);
    return candidates;
}

ParallelExactModeRetriever::ParallelExactModeRetriever(
    const std::size_t thread_count,
    std::shared_ptr<const backend::ComputeBackend> backend
)
    : thread_count_(thread_count),
      backend_(std::move(backend)) {
    if (thread_count_ == 0U) {
        throw std::invalid_argument(
            "parallel retrieval requires at least one thread"
        );
    }
    if (!backend_ || !backend_->available()) {
        throw std::invalid_argument(
            "parallel retrieval requires an available compute backend"
        );
    }
}

std::vector<RetrievedMode> ParallelExactModeRetriever::retrieve(
    const core::PhaseVector& state,
    const std::span<const core::ResonantMode> modes,
    const std::size_t candidate_count
) const {
    if (candidate_count == 0U || modes.empty()) {
        return {};
    }
    const std::size_t worker_count =
        std::min(thread_count_, modes.size());
    std::vector<std::vector<RetrievedMode>> local_candidates(
        worker_count
    );
    std::vector<std::jthread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker_index = 0U;
         worker_index < worker_count;
         ++worker_index) {
        const std::size_t begin =
            (worker_index * modes.size()) / worker_count;
        const std::size_t end =
            ((worker_index + 1U) * modes.size()) / worker_count;
        workers.emplace_back(
            [&, worker_index, begin, end] {
                std::vector<RetrievedMode>& local =
                    local_candidates[worker_index];
                local.reserve(end - begin);
                for (std::size_t mode_index = begin;
                     mode_index < end;
                     ++mode_index) {
                    const core::ResonantMode& mode =
                        modes[mode_index];
                    if (!mode.enabled) {
                        continue;
                    }
                    local.push_back({
                        .mode_index = mode_index,
                        .mode_id = mode.id,
                        .resonance = mode_resonance(
                            state,
                            mode,
                            *backend_
                        ),
                    });
                }
                retain_top_candidates(local, candidate_count);
            }
        );
    }
    workers.clear();

    std::vector<RetrievedMode> candidates;
    candidates.reserve(candidate_count * worker_count);
    for (std::vector<RetrievedMode>& local : local_candidates) {
        candidates.insert(
            candidates.end(),
            std::make_move_iterator(local.begin()),
            std::make_move_iterator(local.end())
        );
    }
    retain_top_candidates(candidates, candidate_count);
    return candidates;
}

std::size_t ParallelExactModeRetriever::thread_count() const noexcept {
    return thread_count_;
}

}  // namespace rlf::retrieval
