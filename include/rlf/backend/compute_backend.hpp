#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string_view>

namespace rlf::core {
class PhaseVector;
}

namespace rlf::backend {

enum class BackendKind {
    scalar_cpu,
    optimized_cpu,
    future_cuda,
};

[[nodiscard]] std::string_view to_string(BackendKind kind) noexcept;

class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;

    [[nodiscard]] virtual BackendKind kind() const noexcept = 0;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual bool available() const noexcept = 0;
    [[nodiscard]] virtual double similarity_angles(
        std::span<const float> left,
        std::span<const float> right
    ) const = 0;
    [[nodiscard]] double similarity(
        const core::PhaseVector& left,
        const core::PhaseVector& right
    ) const;
};

class ScalarCpuBackend final : public ComputeBackend {
public:
    [[nodiscard]] BackendKind kind() const noexcept override;
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] bool available() const noexcept override;
    [[nodiscard]] double similarity_angles(
        std::span<const float> left,
        std::span<const float> right
    ) const override;
};

class OptimizedCpuBackend final : public ComputeBackend {
public:
    static constexpr std::size_t lookup_table_size = 65'536U;

    [[nodiscard]] BackendKind kind() const noexcept override;
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] bool available() const noexcept override;
    [[nodiscard]] double similarity_angles(
        std::span<const float> left,
        std::span<const float> right
    ) const override;
};

class FutureCudaBackend final : public ComputeBackend {
public:
    [[nodiscard]] BackendKind kind() const noexcept override;
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] bool available() const noexcept override;
    [[nodiscard]] double similarity_angles(
        std::span<const float> left,
        std::span<const float> right
    ) const override;
};

[[nodiscard]] std::shared_ptr<const ComputeBackend> make_backend(
    BackendKind kind
);

}  // namespace rlf::backend
