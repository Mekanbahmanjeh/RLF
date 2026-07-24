#include "rlf/core/phase_vector.hpp"

#include "rlf/core/deterministic_rng.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <istream>
#include <limits>
#include <numbers>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rlf::core {
namespace {

constexpr double tau = 2.0 * std::numbers::pi_v<double>;
constexpr float tau_float = 2.0F * std::numbers::pi_v<float>;
constexpr double circular_degeneracy_tolerance = 1.0e-6;
constexpr std::uint32_t phase_vector_format_version = 1U;
constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;
constexpr std::array<char, 8> phase_vector_magic{
    'R', 'L', 'F', 'P', 'V', '0', '0', '1'
};

[[nodiscard]] float normalize_angle_double(const double angle) {
    if (!std::isfinite(angle)) {
        throw std::invalid_argument("phase angles must be finite");
    }

    double normalized_angle = std::fmod(angle, tau);
    if (normalized_angle < 0.0) {
        normalized_angle += tau;
    }

    float result = static_cast<float>(normalized_angle);
    if (result >= tau_float) {
        result = 0.0F;
    }
    return result;
}

void require_same_dimension(
    const PhaseVector& left,
    const PhaseVector& right,
    const char* operation
) {
    if (left.size() != right.size()) {
        throw std::invalid_argument(
            std::string(operation) + " requires equal phase-vector dimensions"
        );
    }
    if (left.empty()) {
        throw std::invalid_argument(
            std::string(operation) + " requires non-empty phase vectors"
        );
    }
}

void update_checksum(std::uint64_t& checksum, const std::uint8_t byte) noexcept {
    checksum ^= static_cast<std::uint64_t>(byte);
    checksum *= fnv_prime;
}

void write_raw_byte(std::ostream& output, const std::uint8_t byte) {
    output.put(static_cast<char>(byte));
    if (!output) {
        throw std::runtime_error("failed to write phase-vector data");
    }
}

void write_u32(
    std::ostream& output,
    const std::uint32_t value,
    std::uint64_t* checksum
) {
    for (unsigned int byte_index = 0U; byte_index < 4U; ++byte_index) {
        const auto byte = static_cast<std::uint8_t>(
            (value >> (byte_index * 8U)) & 0xFFU
        );
        write_raw_byte(output, byte);
        if (checksum != nullptr) {
            update_checksum(*checksum, byte);
        }
    }
}

void write_u64(
    std::ostream& output,
    const std::uint64_t value,
    std::uint64_t* checksum
) {
    for (unsigned int byte_index = 0U; byte_index < 8U; ++byte_index) {
        const auto byte = static_cast<std::uint8_t>(
            (value >> (byte_index * 8U)) & 0xFFULL
        );
        write_raw_byte(output, byte);
        if (checksum != nullptr) {
            update_checksum(*checksum, byte);
        }
    }
}

[[nodiscard]] std::uint8_t read_raw_byte(std::istream& input) {
    const int value = input.get();
    if (value == std::char_traits<char>::eof()) {
        throw std::runtime_error("truncated phase-vector data");
    }
    return static_cast<std::uint8_t>(value);
}

[[nodiscard]] std::uint32_t read_u32(
    std::istream& input,
    std::uint64_t* checksum
) {
    std::uint32_t value = 0U;
    for (unsigned int byte_index = 0U; byte_index < 4U; ++byte_index) {
        const std::uint8_t byte = read_raw_byte(input);
        value |= static_cast<std::uint32_t>(byte) << (byte_index * 8U);
        if (checksum != nullptr) {
            update_checksum(*checksum, byte);
        }
    }
    return value;
}

[[nodiscard]] std::uint64_t read_u64(
    std::istream& input,
    std::uint64_t* checksum
) {
    std::uint64_t value = 0ULL;
    for (unsigned int byte_index = 0U; byte_index < 8U; ++byte_index) {
        const std::uint8_t byte = read_raw_byte(input);
        value |= static_cast<std::uint64_t>(byte) << (byte_index * 8U);
        if (checksum != nullptr) {
            update_checksum(*checksum, byte);
        }
    }
    return value;
}

}  // namespace

PhaseVector::PhaseVector(std::vector<Angle> angles)
    : angles_(std::move(angles)) {
    if (angles_.empty()) {
        throw std::invalid_argument("phase vectors must have a positive dimension");
    }
    for (Angle& angle : angles_) {
        angle = normalize_angle(angle);
    }
}

PhaseVector PhaseVector::zeros(const std::size_t dimension) {
    if (dimension == 0U) {
        throw std::invalid_argument("phase vectors must have a positive dimension");
    }
    return PhaseVector(std::vector<Angle>(dimension, 0.0F));
}

PhaseVector PhaseVector::random(
    const std::size_t dimension,
    DeterministicRng& random_number_generator
) {
    if (dimension == 0U) {
        throw std::invalid_argument("phase vectors must have a positive dimension");
    }

    std::vector<Angle> angles(dimension);
    for (Angle& angle : angles) {
        angle = random_number_generator.uniform_angle();
    }
    return PhaseVector(std::move(angles));
}

PhaseVector PhaseVector::from_complex(
    const std::span<const std::complex<float>> values
) {
    if (values.empty()) {
        throw std::invalid_argument("complex phase input must not be empty");
    }

    std::vector<Angle> angles;
    angles.reserve(values.size());
    for (const std::complex<float>& value : values) {
        const double real_part = static_cast<double>(value.real());
        const double imaginary_part = static_cast<double>(value.imag());
        if (!std::isfinite(real_part) || !std::isfinite(imaginary_part)) {
            throw std::invalid_argument("complex phase input must be finite");
        }
        if (std::hypot(real_part, imaginary_part) <=
            static_cast<double>(std::numeric_limits<float>::epsilon())) {
            throw std::invalid_argument("zero-magnitude complex values have no phase");
        }
        angles.push_back(normalize_angle_double(std::atan2(imaginary_part, real_part)));
    }
    return PhaseVector(std::move(angles));
}

PhaseVector PhaseVector::phase_difference(
    const PhaseVector& from,
    const PhaseVector& to
) {
    require_same_dimension(from, to, "phase difference");
    std::vector<Angle> differences(from.size());
    for (std::size_t dimension_index = 0U;
         dimension_index < from.size();
         ++dimension_index) {
        differences[dimension_index] = normalize_angle_double(
            static_cast<double>(to[dimension_index]) -
            static_cast<double>(from[dimension_index])
        );
    }
    return PhaseVector(std::move(differences));
}

PhaseVector PhaseVector::weighted_circular_average(
    const std::span<const PhaseVector> vectors,
    const std::span<const float> weights
) {
    if (vectors.empty()) {
        throw std::invalid_argument("circular average requires at least one vector");
    }
    if (vectors.size() != weights.size()) {
        throw std::invalid_argument("circular average requires one weight per vector");
    }

    const std::size_t dimension = vectors.front().size();
    double total_weight = 0.0;
    std::size_t fallback_index = 0U;
    bool fallback_selected = false;

    for (std::size_t vector_index = 0U;
         vector_index < vectors.size();
         ++vector_index) {
        if (vectors[vector_index].size() != dimension) {
            throw std::invalid_argument(
                "circular average requires equal phase-vector dimensions"
            );
        }
        const double weight = static_cast<double>(weights[vector_index]);
        if (!std::isfinite(weight) || weight < 0.0) {
            throw std::invalid_argument(
                "circular-average weights must be finite and non-negative"
            );
        }
        total_weight += weight;
        if (!fallback_selected && weight > 0.0) {
            fallback_index = vector_index;
            fallback_selected = true;
        }
    }

    if (!fallback_selected || !std::isfinite(total_weight) || total_weight <= 0.0) {
        throw std::invalid_argument(
            "circular average requires a positive finite total weight"
        );
    }

    std::vector<Angle> result(dimension);
    for (std::size_t dimension_index = 0U;
         dimension_index < dimension;
         ++dimension_index) {
        double weighted_cosine = 0.0;
        double weighted_sine = 0.0;
        for (std::size_t vector_index = 0U;
             vector_index < vectors.size();
             ++vector_index) {
            const double angle =
                static_cast<double>(vectors[vector_index][dimension_index]);
            const double weight = static_cast<double>(weights[vector_index]);
            weighted_cosine += weight * std::cos(angle);
            weighted_sine += weight * std::sin(angle);
        }

        if (std::hypot(weighted_cosine, weighted_sine) <=
            circular_degeneracy_tolerance * total_weight) {
            result[dimension_index] =
                vectors[fallback_index][dimension_index];
        } else {
            result[dimension_index] = normalize_angle_double(
                std::atan2(weighted_sine, weighted_cosine)
            );
        }
    }

    return PhaseVector(std::move(result));
}

PhaseVector PhaseVector::deserialize(
    std::istream& input,
    const std::size_t max_dimension
) {
    if (max_dimension == 0U) {
        throw std::invalid_argument("maximum serialized dimension must be positive");
    }

    for (const char expected_byte : phase_vector_magic) {
        const auto actual_byte = static_cast<char>(read_raw_byte(input));
        if (actual_byte != expected_byte) {
            throw std::runtime_error("invalid phase-vector serialization magic");
        }
    }

    std::uint64_t checksum = fnv_offset_basis;
    const std::uint32_t version = read_u32(input, &checksum);
    if (version != phase_vector_format_version) {
        throw std::runtime_error("unsupported phase-vector serialization version");
    }

    const std::uint64_t serialized_dimension = read_u64(input, &checksum);
    if (serialized_dimension == 0ULL ||
        serialized_dimension > static_cast<std::uint64_t>(max_dimension) ||
        serialized_dimension >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("invalid phase-vector serialized dimension");
    }

    const auto dimension = static_cast<std::size_t>(serialized_dimension);
    std::vector<Angle> angles;
    angles.reserve(dimension);
    for (std::size_t dimension_index = 0U;
         dimension_index < dimension;
         ++dimension_index) {
        const std::uint32_t angle_bits = read_u32(input, &checksum);
        const float angle = std::bit_cast<float>(angle_bits);
        if (!std::isfinite(angle) || angle < 0.0F || angle >= tau_float) {
            throw std::runtime_error("invalid serialized phase angle");
        }
        angles.push_back(angle);
    }

    const std::uint64_t stored_checksum = read_u64(input, nullptr);
    if (stored_checksum != checksum) {
        throw std::runtime_error("phase-vector checksum mismatch");
    }

    return PhaseVector(std::move(angles));
}

std::size_t PhaseVector::size() const noexcept {
    return angles_.size();
}

bool PhaseVector::empty() const noexcept {
    return angles_.empty();
}

std::span<const PhaseVector::Angle> PhaseVector::angles() const noexcept {
    return angles_;
}

PhaseVector::Angle PhaseVector::operator[](const std::size_t index) const {
    return angles_.at(index);
}

PhaseVector PhaseVector::normalized() const {
    std::vector<Angle> normalized_angles = angles_;
    for (Angle& angle : normalized_angles) {
        angle = normalize_angle(angle);
    }
    return PhaseVector(std::move(normalized_angles));
}

PhaseVector PhaseVector::conjugated() const {
    std::vector<Angle> conjugated_angles(size());
    for (std::size_t dimension_index = 0U;
         dimension_index < size();
         ++dimension_index) {
        conjugated_angles[dimension_index] = normalize_angle_double(
            -static_cast<double>(angles_[dimension_index])
        );
    }
    return PhaseVector(std::move(conjugated_angles));
}

PhaseVector PhaseVector::composed(const PhaseVector& other) const {
    require_same_dimension(*this, other, "phase composition");
    std::vector<Angle> composed_angles(size());
    for (std::size_t dimension_index = 0U;
         dimension_index < size();
         ++dimension_index) {
        composed_angles[dimension_index] = normalize_angle_double(
            static_cast<double>(angles_[dimension_index]) +
            static_cast<double>(other[dimension_index])
        );
    }
    return PhaseVector(std::move(composed_angles));
}

PhaseVector PhaseVector::permuted(
    const std::span<const std::size_t> permutation
) const {
    if (permutation.size() != size()) {
        throw std::invalid_argument(
            "phase permutation must match the vector dimension"
        );
    }

    std::vector<bool> seen(size(), false);
    std::vector<Angle> permuted_angles(size());
    for (std::size_t output_index = 0U;
         output_index < size();
         ++output_index) {
        const std::size_t input_index = permutation[output_index];
        if (input_index >= size() || seen[input_index]) {
            throw std::invalid_argument(
                "phase permutation must be a bijection"
            );
        }
        seen[input_index] = true;
        permuted_angles[output_index] = angles_[input_index];
    }
    return PhaseVector(std::move(permuted_angles));
}

std::vector<std::complex<float>> PhaseVector::to_complex() const {
    std::vector<std::complex<float>> values;
    values.reserve(size());
    for (const Angle angle : angles_) {
        const double promoted_angle = static_cast<double>(angle);
        values.emplace_back(
            static_cast<float>(std::cos(promoted_angle)),
            static_cast<float>(std::sin(promoted_angle))
        );
    }
    return values;
}

double PhaseVector::similarity(const PhaseVector& other) const {
    require_same_dimension(*this, other, "phase similarity");

    double real_alignment = 0.0;
    double imaginary_alignment = 0.0;
    for (std::size_t dimension_index = 0U;
         dimension_index < size();
         ++dimension_index) {
        const double difference =
            static_cast<double>(angles_[dimension_index]) -
            static_cast<double>(other[dimension_index]);
        real_alignment += std::cos(difference);
        imaginary_alignment += std::sin(difference);
    }

    const double dimension = static_cast<double>(size());
    const double similarity_value =
        ((real_alignment * real_alignment) +
         (imaginary_alignment * imaginary_alignment)) /
        (dimension * dimension);
    return std::clamp(similarity_value, 0.0, 1.0);
}

double PhaseVector::mean_angular_error(const PhaseVector& other) const {
    require_same_dimension(*this, other, "angular error");

    double total_error = 0.0;
    for (std::size_t dimension_index = 0U;
         dimension_index < size();
         ++dimension_index) {
        const double difference = std::remainder(
            static_cast<double>(angles_[dimension_index]) -
                static_cast<double>(other[dimension_index]),
            tau
        );
        total_error += std::abs(difference);
    }
    return total_error / static_cast<double>(size());
}

void PhaseVector::serialize(std::ostream& output) const {
    if (empty()) {
        throw std::logic_error("cannot serialize an empty phase vector");
    }

    for (const char magic_byte : phase_vector_magic) {
        output.put(magic_byte);
        if (!output) {
            throw std::runtime_error("failed to write phase-vector magic");
        }
    }

    std::uint64_t checksum = fnv_offset_basis;
    write_u32(output, phase_vector_format_version, &checksum);
    write_u64(output, static_cast<std::uint64_t>(size()), &checksum);
    for (const Angle angle : angles_) {
        write_u32(output, std::bit_cast<std::uint32_t>(angle), &checksum);
    }
    write_u64(output, checksum, nullptr);
}

PhaseVector::Angle PhaseVector::normalize_angle(const Angle angle) {
    return normalize_angle_double(static_cast<double>(angle));
}

}  // namespace rlf::core
