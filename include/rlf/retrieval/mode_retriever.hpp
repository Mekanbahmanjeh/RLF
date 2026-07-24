#pragma once

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
}  // namespace rlf::core

namespace rlf::retrieval {

struct RetrievedMode final {
    std::size_t mode_index;
    std::uint64_t mode_id;
    double resonance;
};

class ModeRetriever {
public:
    virtual ~ModeRetriever() = default;

    [[nodiscard]] virtual std::vector<RetrievedMode> retrieve(
        const core::PhaseVector& state,
        std::span<const core::ResonantMode> modes,
        std::size_t candidate_count
    ) const = 0;
};

class ExactModeRetriever final : public ModeRetriever {
public:
    ExactModeRetriever();
    explicit ExactModeRetriever(
        std::shared_ptr<const backend::ComputeBackend> backend
    );

    [[nodiscard]] std::vector<RetrievedMode> retrieve(
        const core::PhaseVector& state,
        std::span<const core::ResonantMode> modes,
        std::size_t candidate_count
    ) const override;

private:
    std::shared_ptr<const backend::ComputeBackend> backend_;
};

class ParallelExactModeRetriever final : public ModeRetriever {
public:
    explicit ParallelExactModeRetriever(
        std::size_t thread_count,
        std::shared_ptr<const backend::ComputeBackend> backend
    );

    [[nodiscard]] std::vector<RetrievedMode> retrieve(
        const core::PhaseVector& state,
        std::span<const core::ResonantMode> modes,
        std::size_t candidate_count
    ) const override;

    [[nodiscard]] std::size_t thread_count() const noexcept;

private:
    std::size_t thread_count_;
    std::shared_ptr<const backend::ComputeBackend> backend_;
};

}  // namespace rlf::retrieval
