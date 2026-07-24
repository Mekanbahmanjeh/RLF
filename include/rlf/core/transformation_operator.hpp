#pragma once

#include "rlf/core/phase_vector.hpp"

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace rlf::core {

enum class OperatorPrimitiveKind {
    phase_shift,
    coordinate_permutation,
    conjugation,
};

struct OperatorPrimitive final {
    OperatorPrimitiveKind kind;
    PhaseVector phase_shift;
    std::vector<std::size_t> permutation;
    std::size_t begin_index{0U};

    [[nodiscard]] static OperatorPrimitive shift(
        PhaseVector value,
        std::size_t begin_index = 0U
    );
    [[nodiscard]] static OperatorPrimitive permute(
        std::vector<std::size_t> value,
        std::size_t begin_index = 0U
    );
    [[nodiscard]] static OperatorPrimitive conjugate(
        std::size_t dimension,
        std::size_t begin_index = 0U
    );
};

class TransformationOperator final {
public:
    explicit TransformationOperator(
        std::size_t dimension,
        std::vector<OperatorPrimitive> primitives = {}
    );

    [[nodiscard]] static TransformationOperator identity(
        std::size_t dimension
    );
    [[nodiscard]] std::size_t dimension() const noexcept;
    [[nodiscard]] std::span<const OperatorPrimitive> primitives() const noexcept;
    [[nodiscard]] PhaseVector apply(const PhaseVector& input) const;
    [[nodiscard]] TransformationOperator then(
        const TransformationOperator& next
    ) const;
    [[nodiscard]] TransformationOperator inverse() const;

private:
    std::size_t dimension_;
    std::vector<OperatorPrimitive> primitives_;
};

enum class OperatorFamily {
    phase_shift,
    coordinate_permutation,
    conjugation,
    permutation_then_phase_shift,
    conjugation_then_phase_shift,
    explicit_sequence,
};

[[nodiscard]] std::string_view to_string(OperatorFamily family) noexcept;

struct OperatorTrainingExample final {
    PhaseVector input;
    PhaseVector target;
};

[[nodiscard]] TransformationOperator fit_operator(
    OperatorFamily family,
    std::span<const OperatorTrainingExample> examples,
    std::size_t begin_index
);

[[nodiscard]] double operator_mean_error(
    const TransformationOperator& transformation,
    std::span<const OperatorTrainingExample> examples
);

}  // namespace rlf::core
