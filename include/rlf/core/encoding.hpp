#pragma once

#include "rlf/core/phase_vector.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rlf::core {

struct DecodedSymbol final {
    std::string symbol;
    double similarity;
};

class SymbolEncoder final {
public:
    SymbolEncoder(
        std::size_t dimension,
        std::uint64_t seed,
        std::vector<std::string> symbols
    );

    [[nodiscard]] std::size_t dimension() const noexcept;
    [[nodiscard]] std::uint64_t seed() const noexcept;
    [[nodiscard]] std::span<const std::string> symbols() const noexcept;
    [[nodiscard]] const PhaseVector& encode(std::string_view symbol) const;
    [[nodiscard]] DecodedSymbol decode(
        const PhaseVector& value
    ) const;

private:
    std::size_t dimension_;
    std::uint64_t seed_;
    std::vector<std::string> symbols_;
    std::vector<PhaseVector> codebook_;
    std::unordered_map<std::string, std::size_t> symbol_indices_;
};

class IntegerEncoder final {
public:
    IntegerEncoder(
        std::size_t dimension,
        std::uint64_t seed,
        std::int64_t minimum,
        std::int64_t maximum
    );

    [[nodiscard]] std::size_t dimension() const noexcept;
    [[nodiscard]] std::int64_t minimum() const noexcept;
    [[nodiscard]] std::int64_t maximum() const noexcept;
    [[nodiscard]] const PhaseVector& encode(std::int64_t value) const;
    [[nodiscard]] std::int64_t decode(const PhaseVector& value) const;

private:
    std::size_t dimension_;
    std::uint64_t seed_;
    std::int64_t minimum_;
    std::int64_t maximum_;
    std::vector<PhaseVector> codebook_;
};

class PermutationFamily final {
public:
    PermutationFamily(
        std::size_t dimension,
        std::uint64_t seed,
        std::size_t count
    );

    [[nodiscard]] std::size_t dimension() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::span<const std::size_t> permutation(
        std::size_t index
    ) const;
    [[nodiscard]] std::span<const std::size_t> inverse_permutation(
        std::size_t index
    ) const;
    [[nodiscard]] PhaseVector apply(
        const PhaseVector& value,
        std::size_t index
    ) const;
    [[nodiscard]] PhaseVector invert(
        const PhaseVector& value,
        std::size_t index
    ) const;

private:
    std::size_t dimension_;
    std::vector<std::vector<std::size_t>> permutations_;
    std::vector<std::vector<std::size_t>> inverse_permutations_;
};

[[nodiscard]] PhaseVector bind(
    const PhaseVector& role,
    const PhaseVector& value
);
[[nodiscard]] PhaseVector unbind(
    const PhaseVector& bound,
    const PhaseVector& role
);
[[nodiscard]] PhaseVector bundle(
    std::span<const PhaseVector> values
);
[[nodiscard]] PhaseVector bundle(
    std::span<const PhaseVector> values,
    std::span<const float> weights
);
[[nodiscard]] PhaseVector encode_ordered_sequence(
    std::span<const PhaseVector> elements,
    const PermutationFamily& positions
);
[[nodiscard]] PhaseVector recover_sequence_element(
    const PhaseVector& encoded_sequence,
    const PermutationFamily& positions,
    std::size_t position
);

}  // namespace rlf::core
