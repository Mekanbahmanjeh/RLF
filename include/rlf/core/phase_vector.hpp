#pragma once

#include <complex>
#include <cstddef>
#include <iosfwd>
#include <span>
#include <vector>

namespace rlf::core {

class DeterministicRng;

class PhaseVector final {
public:
    using Angle = float;

    static constexpr std::size_t default_max_serialized_dimension = 16'777'216;

    explicit PhaseVector(std::vector<Angle> angles);

    [[nodiscard]] static PhaseVector zeros(std::size_t dimension);
    [[nodiscard]] static PhaseVector random(
        std::size_t dimension,
        DeterministicRng& random_number_generator
    );
    [[nodiscard]] static PhaseVector from_complex(
        std::span<const std::complex<float>> values
    );
    [[nodiscard]] static PhaseVector phase_difference(
        const PhaseVector& from,
        const PhaseVector& to
    );
    [[nodiscard]] static PhaseVector weighted_circular_average(
        std::span<const PhaseVector> vectors,
        std::span<const float> weights
    );
    [[nodiscard]] static PhaseVector deserialize(
        std::istream& input,
        std::size_t max_dimension = default_max_serialized_dimension
    );

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] std::span<const Angle> angles() const noexcept;
    [[nodiscard]] Angle operator[](std::size_t index) const;

    [[nodiscard]] PhaseVector normalized() const;
    [[nodiscard]] PhaseVector conjugated() const;
    [[nodiscard]] PhaseVector composed(const PhaseVector& other) const;
    [[nodiscard]] PhaseVector permuted(std::span<const std::size_t> permutation) const;
    [[nodiscard]] std::vector<std::complex<float>> to_complex() const;

    [[nodiscard]] double similarity(const PhaseVector& other) const;
    [[nodiscard]] double mean_angular_error(const PhaseVector& other) const;

    void serialize(std::ostream& output) const;

    [[nodiscard]] static Angle normalize_angle(Angle angle);

private:
    std::vector<Angle> angles_;
};

}  // namespace rlf::core
