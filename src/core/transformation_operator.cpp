#include "rlf/core/transformation_operator.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace rlf::core {
namespace {

void validate_begin_index(
    const std::size_t begin_index,
    const std::size_t dimension
) {
    if (begin_index > dimension) {
        throw std::invalid_argument(
            "operator begin index exceeds its dimension"
        );
    }
}

void validate_permutation(
    const std::span<const std::size_t> permutation,
    const std::size_t dimension,
    const std::size_t begin_index
) {
    if (permutation.size() != dimension) {
        throw std::invalid_argument(
            "operator permutation dimension mismatch"
        );
    }
    std::vector<bool> seen(dimension, false);
    for (std::size_t output_index = 0U;
         output_index < dimension;
         ++output_index) {
        const std::size_t input_index = permutation[output_index];
        if (input_index >= dimension || seen[input_index]) {
            throw std::invalid_argument(
                "operator permutation must be bijective"
            );
        }
        if (output_index < begin_index && input_index != output_index) {
            throw std::invalid_argument(
                "operator permutation must preserve the context prefix"
            );
        }
        if (output_index >= begin_index && input_index < begin_index) {
            throw std::invalid_argument(
                "operator permutation cannot mix context and payload"
            );
        }
        seen[input_index] = true;
    }
}

[[nodiscard]] double signed_angle_error(
    const float input,
    const float target,
    const bool conjugated,
    const float shift
) {
    const float source = conjugated
        ? PhaseVector::normalize_angle(-input)
        : input;
    const float predicted = PhaseVector::normalize_angle(source + shift);
    const float difference = PhaseVector::normalize_angle(target - predicted);
    const double wrapped = std::min(
        static_cast<double>(difference),
        (2.0 * std::numbers::pi_v<double>) -
            static_cast<double>(difference)
    );
    return wrapped;
}

[[nodiscard]] float circular_mean_shift(
    const std::span<const OperatorTrainingExample> examples,
    const std::size_t input_index,
    const std::size_t output_index,
    const bool conjugated
) {
    double cosine_total = 0.0;
    double sine_total = 0.0;
    for (const OperatorTrainingExample& example : examples) {
        const double source = conjugated
            ? -static_cast<double>(example.input[input_index])
            : static_cast<double>(example.input[input_index]);
        const double difference =
            static_cast<double>(example.target[output_index]) - source;
        cosine_total += std::cos(difference);
        sine_total += std::sin(difference);
    }
    return PhaseVector::normalize_angle(
        static_cast<float>(std::atan2(sine_total, cosine_total))
    );
}

[[nodiscard]] double coordinate_fit_error(
    const std::span<const OperatorTrainingExample> examples,
    const std::size_t input_index,
    const std::size_t output_index,
    const bool conjugated,
    const bool allow_shift,
    float& fitted_shift
) {
    fitted_shift = allow_shift
        ? circular_mean_shift(
              examples,
              input_index,
              output_index,
              conjugated
          )
        : 0.0F;
    double total = 0.0;
    for (const OperatorTrainingExample& example : examples) {
        total += signed_angle_error(
            example.input[input_index],
            example.target[output_index],
            conjugated,
            fitted_shift
        );
    }
    return total / static_cast<double>(examples.size());
}

struct MappingFit final {
    std::vector<std::size_t> permutation;
    PhaseVector shift;
    double error;
};

[[nodiscard]] MappingFit fit_mapping(
    const std::span<const OperatorTrainingExample> examples,
    const std::size_t begin_index,
    const bool conjugated,
    const bool allow_shift
) {
    const std::size_t dimension = examples.front().input.size();
    std::vector<std::size_t> permutation(dimension);
    for (std::size_t index = 0U; index < dimension; ++index) {
        permutation[index] = index;
    }
    std::vector<float> shifts(dimension, 0.0F);

    using Candidate =
        std::tuple<double, std::size_t, std::size_t, float>;
    std::vector<Candidate> candidates;
    const std::size_t payload_size = dimension - begin_index;
    candidates.reserve(payload_size * payload_size);
    for (std::size_t output_index = begin_index;
         output_index < dimension;
         ++output_index) {
        for (std::size_t input_index = begin_index;
             input_index < dimension;
             ++input_index) {
            float shift = 0.0F;
            const double error = coordinate_fit_error(
                examples,
                input_index,
                output_index,
                conjugated,
                allow_shift,
                shift
            );
            candidates.emplace_back(
                error,
                output_index,
                input_index,
                shift
            );
        }
    }
    std::sort(candidates.begin(), candidates.end());
    std::vector<bool> output_assigned(dimension, false);
    std::vector<bool> input_assigned(dimension, false);
    double total_error = 0.0;
    for (const auto& [error, output_index, input_index, shift] :
         candidates) {
        if (output_assigned[output_index] ||
            input_assigned[input_index]) {
            continue;
        }
        permutation[output_index] = input_index;
        shifts[output_index] = shift;
        output_assigned[output_index] = true;
        input_assigned[input_index] = true;
        total_error += error;
    }
    return {
        .permutation = std::move(permutation),
        .shift = PhaseVector(std::move(shifts)),
        .error = payload_size == 0U
            ? 0.0
            : total_error / static_cast<double>(payload_size),
    };
}

void validate_examples(
    const std::span<const OperatorTrainingExample> examples,
    const std::size_t begin_index
) {
    if (examples.empty()) {
        throw std::invalid_argument(
            "operator fitting requires at least one example"
        );
    }
    const std::size_t dimension = examples.front().input.size();
    validate_begin_index(begin_index, dimension);
    for (const OperatorTrainingExample& example : examples) {
        if (example.input.size() != dimension ||
            example.target.size() != dimension) {
            throw std::invalid_argument(
                "operator fitting requires equal example dimensions"
            );
        }
    }
}

}  // namespace

OperatorPrimitive OperatorPrimitive::shift(
    PhaseVector value,
    const std::size_t begin_index
) {
    validate_begin_index(begin_index, value.size());
    return {
        .kind = OperatorPrimitiveKind::phase_shift,
        .phase_shift = std::move(value),
        .permutation = {},
        .begin_index = begin_index,
    };
}

OperatorPrimitive OperatorPrimitive::permute(
    std::vector<std::size_t> value,
    const std::size_t begin_index
) {
    validate_begin_index(begin_index, value.size());
    validate_permutation(value, value.size(), begin_index);
    return {
        .kind = OperatorPrimitiveKind::coordinate_permutation,
        .phase_shift = PhaseVector::zeros(value.size()),
        .permutation = std::move(value),
        .begin_index = begin_index,
    };
}

OperatorPrimitive OperatorPrimitive::conjugate(
    const std::size_t dimension,
    const std::size_t begin_index
) {
    validate_begin_index(begin_index, dimension);
    return {
        .kind = OperatorPrimitiveKind::conjugation,
        .phase_shift = PhaseVector::zeros(dimension),
        .permutation = {},
        .begin_index = begin_index,
    };
}

TransformationOperator::TransformationOperator(
    const std::size_t dimension,
    std::vector<OperatorPrimitive> primitives
)
    : dimension_(dimension),
      primitives_(std::move(primitives)) {
    if (dimension_ == 0U) {
        throw std::invalid_argument(
            "transformation operators require a positive dimension"
        );
    }
    for (const OperatorPrimitive& primitive : primitives_) {
        if (primitive.phase_shift.size() != dimension_) {
            throw std::invalid_argument(
                "operator primitive dimension mismatch"
            );
        }
        validate_begin_index(primitive.begin_index, dimension_);
        if (primitive.kind ==
            OperatorPrimitiveKind::coordinate_permutation) {
            validate_permutation(
                primitive.permutation,
                dimension_,
                primitive.begin_index
            );
        }
    }
}

TransformationOperator TransformationOperator::identity(
    const std::size_t dimension
) {
    return TransformationOperator(dimension);
}

std::size_t TransformationOperator::dimension() const noexcept {
    return dimension_;
}

std::span<const OperatorPrimitive>
TransformationOperator::primitives() const noexcept {
    return primitives_;
}

PhaseVector TransformationOperator::apply(const PhaseVector& input) const {
    if (input.size() != dimension_) {
        throw std::invalid_argument(
            "operator input dimension mismatch"
        );
    }
    PhaseVector state = input;
    for (const OperatorPrimitive& primitive : primitives_) {
        switch (primitive.kind) {
        case OperatorPrimitiveKind::phase_shift: {
            std::vector<float> angles(state.angles().begin(), state.angles().end());
            for (std::size_t index = primitive.begin_index;
                 index < dimension_;
                 ++index) {
                angles[index] = PhaseVector::normalize_angle(
                    angles[index] + primitive.phase_shift[index]
                );
            }
            state = PhaseVector(std::move(angles));
            break;
        }
        case OperatorPrimitiveKind::coordinate_permutation:
            state = state.permuted(primitive.permutation);
            break;
        case OperatorPrimitiveKind::conjugation: {
            std::vector<float> angles(state.angles().begin(), state.angles().end());
            for (std::size_t index = primitive.begin_index;
                 index < dimension_;
                 ++index) {
                angles[index] = PhaseVector::normalize_angle(-angles[index]);
            }
            state = PhaseVector(std::move(angles));
            break;
        }
        }
    }
    return state;
}

TransformationOperator TransformationOperator::then(
    const TransformationOperator& next
) const {
    if (dimension_ != next.dimension_) {
        throw std::invalid_argument(
            "operator composition requires equal dimensions"
        );
    }
    std::vector<OperatorPrimitive> combined = primitives_;
    combined.insert(
        combined.end(),
        next.primitives_.begin(),
        next.primitives_.end()
    );
    return TransformationOperator(dimension_, std::move(combined));
}

TransformationOperator TransformationOperator::inverse() const {
    std::vector<OperatorPrimitive> inverted;
    inverted.reserve(primitives_.size());
    for (auto iterator = primitives_.rbegin();
         iterator != primitives_.rend();
         ++iterator) {
        const OperatorPrimitive& primitive = *iterator;
        switch (primitive.kind) {
        case OperatorPrimitiveKind::phase_shift: {
            std::vector<float> angles(
                primitive.phase_shift.angles().begin(),
                primitive.phase_shift.angles().end()
            );
            for (std::size_t index = primitive.begin_index;
                 index < angles.size();
                 ++index) {
                angles[index] = PhaseVector::normalize_angle(-angles[index]);
            }
            inverted.push_back(OperatorPrimitive::shift(
                PhaseVector(std::move(angles)),
                primitive.begin_index
            ));
            break;
        }
        case OperatorPrimitiveKind::coordinate_permutation: {
            std::vector<std::size_t> inverse_permutation(dimension_);
            for (std::size_t output_index = 0U;
                 output_index < dimension_;
                 ++output_index) {
                inverse_permutation[primitive.permutation[output_index]] =
                    output_index;
            }
            inverted.push_back(OperatorPrimitive::permute(
                std::move(inverse_permutation),
                primitive.begin_index
            ));
            break;
        }
        case OperatorPrimitiveKind::conjugation:
            inverted.push_back(OperatorPrimitive::conjugate(
                dimension_,
                primitive.begin_index
            ));
            break;
        }
    }
    return TransformationOperator(dimension_, std::move(inverted));
}


std::string_view to_string(const OperatorFamily family) noexcept {
    switch (family) {
    case OperatorFamily::phase_shift:
        return "phase_shift";
    case OperatorFamily::coordinate_permutation:
        return "coordinate_permutation";
    case OperatorFamily::conjugation:
        return "conjugation";
    case OperatorFamily::permutation_then_phase_shift:
        return "permutation_then_phase_shift";
    case OperatorFamily::conjugation_then_phase_shift:
        return "conjugation_then_phase_shift";
    case OperatorFamily::explicit_sequence:
        return "explicit_sequence";
    }
    return "unknown";
}

TransformationOperator fit_operator(
    const OperatorFamily family,
    const std::span<const OperatorTrainingExample> examples,
    const std::size_t begin_index
) {
    validate_examples(examples, begin_index);
    const std::size_t dimension = examples.front().input.size();
    switch (family) {
    case OperatorFamily::phase_shift: {
        const MappingFit fit =
            fit_mapping(examples, begin_index, false, true);
        std::vector<float> identity_shift(
            fit.shift.angles().begin(),
            fit.shift.angles().end()
        );
        for (std::size_t index = begin_index; index < dimension; ++index) {
            identity_shift[index] = circular_mean_shift(
                examples,
                index,
                index,
                false
            );
        }
        return TransformationOperator(
            dimension,
            {OperatorPrimitive::shift(
                PhaseVector(std::move(identity_shift)),
                begin_index
            )}
        );
    }
    case OperatorFamily::coordinate_permutation: {
        const MappingFit fit =
            fit_mapping(examples, begin_index, false, false);
        return TransformationOperator(
            dimension,
            {OperatorPrimitive::permute(
                fit.permutation,
                begin_index
            )}
        );
    }
    case OperatorFamily::conjugation:
        return TransformationOperator(
            dimension,
            {OperatorPrimitive::conjugate(dimension, begin_index)}
        );
    case OperatorFamily::permutation_then_phase_shift: {
        const MappingFit fit =
            fit_mapping(examples, begin_index, false, true);
        return TransformationOperator(
            dimension,
            {
                OperatorPrimitive::permute(
                    fit.permutation,
                    begin_index
                ),
                OperatorPrimitive::shift(fit.shift, begin_index),
            }
        );
    }
    case OperatorFamily::conjugation_then_phase_shift: {
        std::vector<float> shifts(dimension, 0.0F);
        for (std::size_t index = begin_index; index < dimension; ++index) {
            shifts[index] = circular_mean_shift(
                examples,
                index,
                index,
                true
            );
        }
        return TransformationOperator(
            dimension,
            {
                OperatorPrimitive::conjugate(dimension, begin_index),
                OperatorPrimitive::shift(
                    PhaseVector(std::move(shifts)),
                    begin_index
                ),
            }
        );
    }
    case OperatorFamily::explicit_sequence: {
        const MappingFit direct =
            fit_mapping(examples, begin_index, false, true);
        const MappingFit conjugated =
            fit_mapping(examples, begin_index, true, true);
        if (conjugated.error < direct.error) {
            return TransformationOperator(
                dimension,
                {
                    OperatorPrimitive::conjugate(dimension, begin_index),
                    OperatorPrimitive::permute(
                        conjugated.permutation,
                        begin_index
                    ),
                    OperatorPrimitive::shift(
                        conjugated.shift,
                        begin_index
                    ),
                }
            );
        }
        return TransformationOperator(
            dimension,
            {
                OperatorPrimitive::permute(
                    direct.permutation,
                    begin_index
                ),
                OperatorPrimitive::shift(direct.shift, begin_index),
            }
        );
    }
    }
    throw std::logic_error("unknown operator family");
}

double operator_mean_error(
    const TransformationOperator& transformation,
    const std::span<const OperatorTrainingExample> examples
) {
    if (examples.empty()) {
        throw std::invalid_argument(
            "operator error requires at least one example"
        );
    }
    double total = 0.0;
    for (const OperatorTrainingExample& example : examples) {
        total += transformation.apply(example.input)
            .mean_angular_error(example.target);
    }
    return total / static_cast<double>(examples.size());
}

}  // namespace rlf::core
