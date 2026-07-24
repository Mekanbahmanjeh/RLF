#pragma once

#include "rlf/retrieval/mode_retriever.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rlf::backend {
class ComputeBackend;
}

namespace rlf::core {
class PhaseVector;
struct ResonantMode;
}

namespace rlf::retrieval {

class ContiguousModeIndex final {
public:
    ContiguousModeIndex(
        std::span<const core::ResonantMode> modes,
        std::shared_ptr<const backend::ComputeBackend> backend
    );

    [[nodiscard]] std::size_t dimension() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t bytes_stored() const noexcept;
    [[nodiscard]] std::vector<RetrievedMode> retrieve(
        const core::PhaseVector& state,
        std::size_t candidate_count,
        std::size_t thread_count = 1U
    ) const;

private:
    std::size_t dimension_;
    std::vector<float> context_angles_;
    std::vector<std::uint64_t> mode_ids_;
    std::vector<std::size_t> source_indices_;
    std::vector<float> selectivities_;
    std::shared_ptr<const backend::ComputeBackend> backend_;
};

}  // namespace rlf::retrieval
