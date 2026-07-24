#include "rlf/backend/compute_backend.hpp"

#include "rlf/core/phase_vector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>

namespace rlf::backend {
namespace {

constexpr double tau = 2.0 * std::numbers::pi_v<double>;

struct TrigonometricTable final {
    std::array<double, OptimizedCpuBackend::lookup_table_size> cosine{};
    std::array<double, OptimizedCpuBackend::lookup_table_size> sine{};

    TrigonometricTable() {
        for (std::size_t index = 0U;
             index < OptimizedCpuBackend::lookup_table_size;
             ++index) {
            const double angle =
                tau * static_cast<double>(index) /
                static_cast<double>(
                    OptimizedCpuBackend::lookup_table_size
                );
            cosine[index] = std::cos(angle);
            sine[index] = std::sin(angle);
        }
    }
};

[[nodiscard]] const TrigonometricTable& trigonometric_table() {
    static const TrigonometricTable table;
    return table;
}

struct TrigonometricValue final {
    double cosine;
    double sine;
};

[[nodiscard]] TrigonometricValue lookup_trigonometric(
    const TrigonometricTable& table,
    const double angle
) noexcept {
    double normalized = angle;
    if (normalized < 0.0) {
        normalized += tau;
    } else if (normalized >= tau) {
        normalized -= tau;
    }
    const double scaled =
        normalized *
        static_cast<double>(OptimizedCpuBackend::lookup_table_size) /
        tau;
    const auto lower = static_cast<std::size_t>(scaled);
    const std::size_t upper =
        lower + 1U == OptimizedCpuBackend::lookup_table_size
        ? 0U
        : lower + 1U;
    const double fraction = scaled - static_cast<double>(lower);
    return {
        .cosine =
            table.cosine[lower] +
            ((table.cosine[upper] - table.cosine[lower]) *
             fraction),
        .sine =
            table.sine[lower] +
            ((table.sine[upper] - table.sine[lower]) *
             fraction),
    };
}

void validate_dimensions(
    const std::span<const float> left,
    const std::span<const float> right
) {
    if (left.size() != right.size() || left.empty()) {
        throw std::invalid_argument(
            "backend similarity requires equal non-empty dimensions"
        );
    }
}

}  // namespace

double ComputeBackend::similarity(
    const core::PhaseVector& left,
    const core::PhaseVector& right
) const {
    return similarity_angles(left.angles(), right.angles());
}

std::string_view to_string(const BackendKind kind) noexcept {
    switch (kind) {
    case BackendKind::scalar_cpu:
        return "scalar_cpu";
    case BackendKind::optimized_cpu:
        return "optimized_cpu";
    case BackendKind::future_cuda:
        return "future_cuda";
    }
    return "unknown";
}

BackendKind ScalarCpuBackend::kind() const noexcept {
    return BackendKind::scalar_cpu;
}

std::string_view ScalarCpuBackend::name() const noexcept {
    return to_string(kind());
}

bool ScalarCpuBackend::available() const noexcept {
    return true;
}

double ScalarCpuBackend::similarity_angles(
    const std::span<const float> left,
    const std::span<const float> right
) const {
    validate_dimensions(left, right);
    double real_alignment = 0.0;
    double imaginary_alignment = 0.0;
    for (std::size_t index = 0U; index < left.size(); ++index) {
        const double difference =
            static_cast<double>(left[index]) -
            static_cast<double>(right[index]);
        real_alignment += std::cos(difference);
        imaginary_alignment += std::sin(difference);
    }
    const double dimension = static_cast<double>(left.size());
    return std::clamp(
        ((real_alignment * real_alignment) +
         (imaginary_alignment * imaginary_alignment)) /
            (dimension * dimension),
        0.0,
        1.0
    );
}

BackendKind OptimizedCpuBackend::kind() const noexcept {
    return BackendKind::optimized_cpu;
}

std::string_view OptimizedCpuBackend::name() const noexcept {
    return to_string(kind());
}

bool OptimizedCpuBackend::available() const noexcept {
    return true;
}

double OptimizedCpuBackend::similarity_angles(
    const std::span<const float> left,
    const std::span<const float> right
) const {
    validate_dimensions(left, right);
    const TrigonometricTable& table = trigonometric_table();
    double real_alignment = 0.0;
    double imaginary_alignment = 0.0;
    std::size_t index = 0U;
    for (; index + 3U < left.size(); index += 4U) {
        const TrigonometricValue first = lookup_trigonometric(
            table,
            static_cast<double>(left[index]) -
            static_cast<double>(right[index])
        );
        const TrigonometricValue second = lookup_trigonometric(
            table,
            static_cast<double>(left[index + 1U]) -
            static_cast<double>(right[index + 1U])
        );
        const TrigonometricValue third = lookup_trigonometric(
            table,
            static_cast<double>(left[index + 2U]) -
            static_cast<double>(right[index + 2U])
        );
        const TrigonometricValue fourth = lookup_trigonometric(
            table,
            static_cast<double>(left[index + 3U]) -
            static_cast<double>(right[index + 3U])
        );
        real_alignment +=
            first.cosine + second.cosine +
            third.cosine + fourth.cosine;
        imaginary_alignment +=
            first.sine + second.sine +
            third.sine + fourth.sine;
    }
    for (; index < left.size(); ++index) {
        const TrigonometricValue value = lookup_trigonometric(
            table,
            static_cast<double>(left[index]) -
            static_cast<double>(right[index])
        );
        real_alignment += value.cosine;
        imaginary_alignment += value.sine;
    }
    const double dimension = static_cast<double>(left.size());
    return std::clamp(
        ((real_alignment * real_alignment) +
         (imaginary_alignment * imaginary_alignment)) /
            (dimension * dimension),
        0.0,
        1.0
    );
}

BackendKind FutureCudaBackend::kind() const noexcept {
    return BackendKind::future_cuda;
}

std::string_view FutureCudaBackend::name() const noexcept {
    return to_string(kind());
}

bool FutureCudaBackend::available() const noexcept {
    return false;
}

double FutureCudaBackend::similarity_angles(
    const std::span<const float>,
    const std::span<const float>
) const {
    throw std::runtime_error(
        "future CUDA backend is intentionally unavailable"
    );
}

std::shared_ptr<const ComputeBackend> make_backend(
    const BackendKind kind
) {
    switch (kind) {
    case BackendKind::scalar_cpu:
        return std::make_shared<ScalarCpuBackend>();
    case BackendKind::optimized_cpu:
        return std::make_shared<OptimizedCpuBackend>();
    case BackendKind::future_cuda:
        return std::make_shared<FutureCudaBackend>();
    }
    throw std::invalid_argument("unknown compute backend kind");
}

}  // namespace rlf::backend
