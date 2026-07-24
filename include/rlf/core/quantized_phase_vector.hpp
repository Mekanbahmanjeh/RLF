#pragma once

#include "rlf/core/phase_vector.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <span>
#include <string_view>
#include <vector>

namespace rlf::core {

enum class PhaseEncoding {
    float32,
    uint8_phase,
    uint16_phase,
    float16_compatible,
    int8_residual,
};

[[nodiscard]] std::string_view to_string(PhaseEncoding encoding) noexcept;

class QuantizedPhaseVector final {
public:
    [[nodiscard]] static QuantizedPhaseVector encode(
        const PhaseVector& value,
        PhaseEncoding encoding
    );
    [[nodiscard]] static QuantizedPhaseVector deserialize(
        std::istream& input,
        std::size_t maximum_dimension =
            PhaseVector::default_max_serialized_dimension
    );

    [[nodiscard]] PhaseEncoding encoding() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t bytes_stored() const noexcept;
    [[nodiscard]] PhaseVector decode() const;
    [[nodiscard]] double mean_angular_error(
        const PhaseVector& reference
    ) const;
    void serialize(std::ostream& output) const;

private:
    QuantizedPhaseVector(
        PhaseEncoding encoding,
        std::size_t dimension,
        float residual_base,
        std::vector<std::uint8_t> storage
    );

    PhaseEncoding encoding_;
    std::size_t dimension_;
    float residual_base_;
    std::vector<std::uint8_t> storage_;
};

}  // namespace rlf::core
