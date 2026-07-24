#include "rlf/core/encoding.hpp"

#include "rlf/core/deterministic_rng.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rlf::core {
namespace {

[[nodiscard]] std::size_t checked_integer_count(
    const std::int64_t minimum,
    const std::int64_t maximum
) {
    if (minimum > maximum) {
        throw std::invalid_argument(
            "integer encoder minimum must not exceed maximum"
        );
    }
    const std::uint64_t span =
        static_cast<std::uint64_t>(maximum) -
        static_cast<std::uint64_t>(minimum);
    if (span == std::numeric_limits<std::uint64_t>::max()) {
        throw std::invalid_argument(
            "integer encoder range is too large"
        );
    }
    const std::uint64_t count = span + 1ULL;
    if (count >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max()
        )) {
        throw std::invalid_argument(
            "integer encoder range exceeds addressable storage"
        );
    }
    return static_cast<std::size_t>(count);
}

[[nodiscard]] std::size_t integer_index(
    const std::int64_t value,
    const std::int64_t minimum,
    const std::int64_t maximum
) {
    if (value < minimum || value > maximum) {
        throw std::out_of_range(
            "integer value is outside the encoder range"
        );
    }
    const std::uint64_t offset =
        static_cast<std::uint64_t>(value) -
        static_cast<std::uint64_t>(minimum);
    return static_cast<std::size_t>(offset);
}

}  // namespace

SymbolEncoder::SymbolEncoder(
    const std::size_t dimension,
    const std::uint64_t seed,
    std::vector<std::string> symbols
)
    : dimension_(dimension),
      seed_(seed),
      symbols_(std::move(symbols)) {
    if (dimension_ == 0U || symbols_.empty()) {
        throw std::invalid_argument(
            "symbol encoder dimension and symbol set must be non-empty"
        );
    }

    DeterministicRng random_number_generator(seed_);
    codebook_.reserve(symbols_.size());
    symbol_indices_.reserve(symbols_.size());
    for (std::size_t symbol_index = 0U;
         symbol_index < symbols_.size();
         ++symbol_index) {
        if (symbols_[symbol_index].empty()) {
            throw std::invalid_argument(
                "symbol encoder symbols must not be empty"
            );
        }
        if (!symbol_indices_.emplace(
                symbols_[symbol_index],
                symbol_index
            ).second) {
            throw std::invalid_argument(
                "symbol encoder symbols must be unique"
            );
        }
        codebook_.push_back(
            PhaseVector::random(
                dimension_,
                random_number_generator
            )
        );
    }
}

std::size_t SymbolEncoder::dimension() const noexcept {
    return dimension_;
}

std::uint64_t SymbolEncoder::seed() const noexcept {
    return seed_;
}

std::span<const std::string> SymbolEncoder::symbols() const noexcept {
    return symbols_;
}

const PhaseVector& SymbolEncoder::encode(
    const std::string_view symbol
) const {
    const auto found = symbol_indices_.find(std::string(symbol));
    if (found == symbol_indices_.end()) {
        throw std::out_of_range("unknown symbol");
    }
    return codebook_[found->second];
}

DecodedSymbol SymbolEncoder::decode(
    const PhaseVector& value
) const {
    if (value.size() != dimension_) {
        throw std::invalid_argument(
            "symbol decode dimension must match the encoder"
        );
    }
    std::size_t best_index = 0U;
    double best_similarity = -1.0;
    for (std::size_t index = 0U;
         index < codebook_.size();
         ++index) {
        const double similarity = value.similarity(codebook_[index]);
        if (similarity > best_similarity ||
            (similarity == best_similarity &&
             symbols_[index] < symbols_[best_index])) {
            best_similarity = similarity;
            best_index = index;
        }
    }
    return {
        .symbol = symbols_[best_index],
        .similarity = best_similarity,
    };
}

IntegerEncoder::IntegerEncoder(
    const std::size_t dimension,
    const std::uint64_t seed,
    const std::int64_t minimum,
    const std::int64_t maximum
)
    : dimension_(dimension),
      seed_(seed),
      minimum_(minimum),
      maximum_(maximum) {
    if (dimension_ == 0U) {
        throw std::invalid_argument(
            "integer encoder dimension must be positive"
        );
    }
    const std::size_t count = checked_integer_count(
        minimum_,
        maximum_
    );
    DeterministicRng random_number_generator(seed_);
    codebook_.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        codebook_.push_back(
            PhaseVector::random(
                dimension_,
                random_number_generator
            )
        );
    }
}

std::size_t IntegerEncoder::dimension() const noexcept {
    return dimension_;
}

std::int64_t IntegerEncoder::minimum() const noexcept {
    return minimum_;
}

std::int64_t IntegerEncoder::maximum() const noexcept {
    return maximum_;
}

const PhaseVector& IntegerEncoder::encode(
    const std::int64_t value
) const {
    return codebook_[integer_index(value, minimum_, maximum_)];
}

std::int64_t IntegerEncoder::decode(
    const PhaseVector& value
) const {
    if (value.size() != dimension_) {
        throw std::invalid_argument(
            "integer decode dimension must match the encoder"
        );
    }
    std::size_t best_index = 0U;
    double best_similarity = -1.0;
    for (std::size_t index = 0U;
         index < codebook_.size();
         ++index) {
        const double similarity = value.similarity(codebook_[index]);
        if (similarity > best_similarity) {
            best_similarity = similarity;
            best_index = index;
        }
    }
    if (best_index >
        static_cast<std::size_t>(
            std::numeric_limits<std::int64_t>::max()
        )) {
        throw std::overflow_error(
            "decoded integer index exceeds signed range"
        );
    }
    return minimum_ + static_cast<std::int64_t>(best_index);
}

PermutationFamily::PermutationFamily(
    const std::size_t dimension,
    const std::uint64_t seed,
    const std::size_t count
)
    : dimension_(dimension) {
    if (dimension_ == 0U || count == 0U) {
        throw std::invalid_argument(
            "permutation family dimension and count must be positive"
        );
    }
    DeterministicRng random_number_generator(seed);
    permutations_.reserve(count);
    inverse_permutations_.reserve(count);
    for (std::size_t permutation_index = 0U;
         permutation_index < count;
         ++permutation_index) {
        std::vector<std::size_t> permutation(dimension_);
        for (std::size_t index = 0U; index < dimension_; ++index) {
            permutation[index] = index;
        }
        for (std::size_t remaining = dimension_;
             remaining > 1U;
             --remaining) {
            const std::size_t selected =
                random_number_generator.uniform_index(remaining);
            std::swap(
                permutation[remaining - 1U],
                permutation[selected]
            );
        }

        std::vector<std::size_t> inverse(dimension_);
        for (std::size_t output_index = 0U;
             output_index < dimension_;
             ++output_index) {
            inverse[permutation[output_index]] = output_index;
        }
        permutations_.push_back(std::move(permutation));
        inverse_permutations_.push_back(std::move(inverse));
    }
}

std::size_t PermutationFamily::dimension() const noexcept {
    return dimension_;
}

std::size_t PermutationFamily::size() const noexcept {
    return permutations_.size();
}

std::span<const std::size_t> PermutationFamily::permutation(
    const std::size_t index
) const {
    return permutations_.at(index);
}

std::span<const std::size_t> PermutationFamily::inverse_permutation(
    const std::size_t index
) const {
    return inverse_permutations_.at(index);
}

PhaseVector PermutationFamily::apply(
    const PhaseVector& value,
    const std::size_t index
) const {
    if (value.size() != dimension_) {
        throw std::invalid_argument(
            "permutation input dimension must match the family"
        );
    }
    return value.permuted(permutation(index));
}

PhaseVector PermutationFamily::invert(
    const PhaseVector& value,
    const std::size_t index
) const {
    if (value.size() != dimension_) {
        throw std::invalid_argument(
            "inverse permutation dimension must match the family"
        );
    }
    return value.permuted(inverse_permutation(index));
}

PhaseVector bind(
    const PhaseVector& role,
    const PhaseVector& value
) {
    return role.composed(value);
}

PhaseVector unbind(
    const PhaseVector& bound,
    const PhaseVector& role
) {
    return bound.composed(role.conjugated());
}

PhaseVector bundle(const std::span<const PhaseVector> values) {
    std::vector<float> weights(values.size(), 1.0F);
    return bundle(values, weights);
}

PhaseVector bundle(
    const std::span<const PhaseVector> values,
    const std::span<const float> weights
) {
    return PhaseVector::weighted_circular_average(values, weights);
}

PhaseVector encode_ordered_sequence(
    const std::span<const PhaseVector> elements,
    const PermutationFamily& positions
) {
    if (elements.empty()) {
        throw std::invalid_argument(
            "ordered sequence must contain at least one element"
        );
    }
    if (elements.size() > positions.size()) {
        throw std::invalid_argument(
            "ordered sequence exceeds available position permutations"
        );
    }
    std::vector<PhaseVector> positioned;
    positioned.reserve(elements.size());
    for (std::size_t index = 0U; index < elements.size(); ++index) {
        positioned.push_back(positions.apply(elements[index], index));
    }
    return bundle(positioned);
}

PhaseVector recover_sequence_element(
    const PhaseVector& encoded_sequence,
    const PermutationFamily& positions,
    const std::size_t position
) {
    return positions.invert(encoded_sequence, position);
}

}  // namespace rlf::core
