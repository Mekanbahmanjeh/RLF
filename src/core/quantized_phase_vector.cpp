#include "rlf/core/quantized_phase_vector.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <istream>
#include <limits>
#include <numbers>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace rlf::core {
namespace {

constexpr double tau = 2.0 * std::numbers::pi_v<double>;
constexpr std::array<char, 8> magic{
    'R', 'L', 'F', 'Q', 'P', '0', '0', '1'
};
constexpr std::uint32_t format_version = 1U;

void append_u16(
    std::vector<std::uint8_t>& storage,
    const std::uint16_t value
) {
    storage.push_back(static_cast<std::uint8_t>(value & 0xFFU));
    storage.push_back(
        static_cast<std::uint8_t>((value >> 8U) & 0xFFU)
    );
}

[[nodiscard]] std::uint16_t read_u16(
    const std::span<const std::uint8_t> storage,
    const std::size_t index
) {
    return static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(storage[index]) |
        (static_cast<std::uint16_t>(storage[index + 1U]) << 8U)
    );
}

void append_u32(
    std::vector<std::uint8_t>& storage,
    const std::uint32_t value
) {
    for (unsigned int byte = 0U; byte < 4U; ++byte) {
        storage.push_back(static_cast<std::uint8_t>(
            (value >> (byte * 8U)) & 0xFFU
        ));
    }
}

[[nodiscard]] std::uint32_t read_u32(
    const std::span<const std::uint8_t> storage,
    const std::size_t index
) {
    std::uint32_t result = 0U;
    for (unsigned int byte = 0U; byte < 4U; ++byte) {
        result |=
            static_cast<std::uint32_t>(storage[index + byte]) <<
            (byte * 8U);
    }
    return result;
}

[[nodiscard]] std::uint16_t float_to_half(const float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t sign = (bits >> 16U) & 0x8000U;
    const std::uint32_t exponent = (bits >> 23U) & 0xFFU;
    const std::uint32_t mantissa = bits & 0x7FFFFFU;
    if (exponent == 0xFFU) {
        return static_cast<std::uint16_t>(
            sign | 0x7C00U | (mantissa == 0U ? 0U : 0x0200U)
        );
    }
    const int adjusted_exponent =
        static_cast<int>(exponent) - 127 + 15;
    if (adjusted_exponent <= 0) {
        if (adjusted_exponent < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        const std::uint32_t normalized = mantissa | 0x800000U;
        const unsigned int shift =
            static_cast<unsigned int>(14 - adjusted_exponent);
        const std::uint32_t rounded =
            (normalized + (1U << (shift - 1U))) >> shift;
        return static_cast<std::uint16_t>(sign | rounded);
    }
    if (adjusted_exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7C00U);
    }
    const std::uint32_t rounded_mantissa = mantissa + 0x1000U;
    std::uint32_t half_exponent =
        static_cast<std::uint32_t>(adjusted_exponent);
    std::uint32_t half_mantissa = rounded_mantissa >> 13U;
    if ((half_mantissa & 0x0400U) != 0U) {
        half_mantissa = 0U;
        ++half_exponent;
        if (half_exponent >= 31U) {
            return static_cast<std::uint16_t>(sign | 0x7C00U);
        }
    }
    return static_cast<std::uint16_t>(
        sign | (half_exponent << 10U) |
        (half_mantissa & 0x03FFU)
    );
}

[[nodiscard]] float half_to_float(const std::uint16_t half) {
    const std::uint32_t sign =
        static_cast<std::uint32_t>(half & 0x8000U) << 16U;
    const std::uint32_t exponent = (half >> 10U) & 0x1FU;
    const std::uint32_t mantissa = half & 0x03FFU;
    std::uint32_t bits = 0U;
    if (exponent == 0U) {
        if (mantissa == 0U) {
            bits = sign;
        } else {
            std::uint32_t normalized = mantissa;
            int shift = 0;
            while ((normalized & 0x0400U) == 0U) {
                normalized <<= 1U;
                ++shift;
            }
            normalized &= 0x03FFU;
            const std::uint32_t float_exponent =
                static_cast<std::uint32_t>(127 - 15 - shift + 1);
            bits = sign | (float_exponent << 23U) |
                (normalized << 13U);
        }
    } else if (exponent == 0x1FU) {
        bits = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        const std::uint32_t float_exponent =
            exponent + static_cast<std::uint32_t>(127 - 15);
        bits = sign | (float_exponent << 23U) |
            (mantissa << 13U);
    }
    return std::bit_cast<float>(bits);
}

[[nodiscard]] float circular_mean(const PhaseVector& value) {
    double real = 0.0;
    double imaginary = 0.0;
    for (const float angle : value.angles()) {
        real += std::cos(static_cast<double>(angle));
        imaginary += std::sin(static_cast<double>(angle));
    }
    return PhaseVector::normalize_angle(
        static_cast<float>(std::atan2(imaginary, real))
    );
}

void write_u32(std::ostream& output, const std::uint32_t value) {
    for (unsigned int byte = 0U; byte < 4U; ++byte) {
        output.put(static_cast<char>(
            (value >> (byte * 8U)) & 0xFFU
        ));
    }
}

[[nodiscard]] std::uint32_t read_u32(std::istream& input) {
    std::uint32_t value = 0U;
    for (unsigned int byte = 0U; byte < 4U; ++byte) {
        const int next = input.get();
        if (next == std::char_traits<char>::eof()) {
            throw std::runtime_error(
                "truncated quantized phase vector"
            );
        }
        value |= static_cast<std::uint32_t>(
            static_cast<std::uint8_t>(next)
        ) << (byte * 8U);
    }
    return value;
}

}  // namespace

std::string_view to_string(const PhaseEncoding encoding) noexcept {
    switch (encoding) {
    case PhaseEncoding::float32:
        return "float32";
    case PhaseEncoding::uint8_phase:
        return "uint8_phase";
    case PhaseEncoding::uint16_phase:
        return "uint16_phase";
    case PhaseEncoding::float16_compatible:
        return "float16_compatible";
    case PhaseEncoding::int8_residual:
        return "int8_residual";
    }
    return "unknown";
}

QuantizedPhaseVector::QuantizedPhaseVector(
    const PhaseEncoding encoding,
    const std::size_t dimension,
    const float residual_base,
    std::vector<std::uint8_t> storage
)
    : encoding_(encoding),
      dimension_(dimension),
      residual_base_(residual_base),
      storage_(std::move(storage)) {
    if (dimension_ == 0U) {
        throw std::invalid_argument(
            "quantized phase vector dimension must be positive"
        );
    }
}

QuantizedPhaseVector QuantizedPhaseVector::encode(
    const PhaseVector& value,
    const PhaseEncoding encoding
) {
    std::vector<std::uint8_t> storage;
    float residual_base = 0.0F;
    switch (encoding) {
    case PhaseEncoding::float32:
        storage.reserve(value.size() * sizeof(float));
        for (const float angle : value.angles()) {
            append_u32(
                storage,
                std::bit_cast<std::uint32_t>(angle)
            );
        }
        break;
    case PhaseEncoding::uint8_phase:
        storage.reserve(value.size());
        for (const float angle : value.angles()) {
            const double scaled =
                static_cast<double>(angle) * 256.0 / tau;
            storage.push_back(static_cast<std::uint8_t>(
                static_cast<std::size_t>(scaled + 0.5) % 256U
            ));
        }
        break;
    case PhaseEncoding::uint16_phase:
        storage.reserve(value.size() * sizeof(std::uint16_t));
        for (const float angle : value.angles()) {
            const double scaled =
                static_cast<double>(angle) * 65'536.0 / tau;
            append_u16(
                storage,
                static_cast<std::uint16_t>(
                    static_cast<std::size_t>(scaled + 0.5) %
                    65'536U
                )
            );
        }
        break;
    case PhaseEncoding::float16_compatible:
        storage.reserve(value.size() * sizeof(std::uint16_t));
        for (const float angle : value.angles()) {
            append_u16(storage, float_to_half(angle));
        }
        break;
    case PhaseEncoding::int8_residual:
        residual_base = circular_mean(value);
        storage.reserve(value.size());
        for (const float angle : value.angles()) {
            const double difference = std::remainder(
                static_cast<double>(angle) -
                    static_cast<double>(residual_base),
                tau
            );
            const double scaled = std::clamp(
                difference / std::numbers::pi_v<double>,
                -1.0,
                1.0
            ) * 127.0;
            const auto quantized = static_cast<std::int8_t>(
                std::lround(scaled)
            );
            storage.push_back(
                std::bit_cast<std::uint8_t>(quantized)
            );
        }
        break;
    }
    return QuantizedPhaseVector(
        encoding,
        value.size(),
        residual_base,
        std::move(storage)
    );
}

PhaseEncoding QuantizedPhaseVector::encoding() const noexcept {
    return encoding_;
}

std::size_t QuantizedPhaseVector::size() const noexcept {
    return dimension_;
}

std::size_t QuantizedPhaseVector::bytes_stored() const noexcept {
    return storage_.size() +
        (encoding_ == PhaseEncoding::int8_residual
             ? sizeof(residual_base_)
             : 0U);
}

PhaseVector QuantizedPhaseVector::decode() const {
    std::vector<float> angles;
    angles.reserve(dimension_);
    switch (encoding_) {
    case PhaseEncoding::float32:
        if (storage_.size() != dimension_ * sizeof(float)) {
            throw std::runtime_error(
                "invalid float32 quantized storage size"
            );
        }
        for (std::size_t index = 0U;
             index < storage_.size();
             index += sizeof(float)) {
            angles.push_back(
                std::bit_cast<float>(read_u32(storage_, index))
            );
        }
        break;
    case PhaseEncoding::uint8_phase:
        if (storage_.size() != dimension_) {
            throw std::runtime_error(
                "invalid uint8 phase storage size"
            );
        }
        for (const std::uint8_t code : storage_) {
            angles.push_back(static_cast<float>(
                tau * static_cast<double>(code) / 256.0
            ));
        }
        break;
    case PhaseEncoding::uint16_phase:
        if (storage_.size() !=
            dimension_ * sizeof(std::uint16_t)) {
            throw std::runtime_error(
                "invalid uint16 phase storage size"
            );
        }
        for (std::size_t index = 0U;
             index < storage_.size();
             index += sizeof(std::uint16_t)) {
            angles.push_back(static_cast<float>(
                tau *
                static_cast<double>(read_u16(storage_, index)) /
                65'536.0
            ));
        }
        break;
    case PhaseEncoding::float16_compatible:
        if (storage_.size() !=
            dimension_ * sizeof(std::uint16_t)) {
            throw std::runtime_error(
                "invalid float16 storage size"
            );
        }
        for (std::size_t index = 0U;
             index < storage_.size();
             index += sizeof(std::uint16_t)) {
            angles.push_back(
                half_to_float(read_u16(storage_, index))
            );
        }
        break;
    case PhaseEncoding::int8_residual:
        if (storage_.size() != dimension_) {
            throw std::runtime_error(
                "invalid residual storage size"
            );
        }
        for (const std::uint8_t code : storage_) {
            const std::int8_t residual =
                std::bit_cast<std::int8_t>(code);
            angles.push_back(PhaseVector::normalize_angle(
                residual_base_ +
                static_cast<float>(
                    static_cast<double>(residual) *
                    std::numbers::pi_v<double> / 127.0
                )
            ));
        }
        break;
    }
    return PhaseVector(std::move(angles));
}

double QuantizedPhaseVector::mean_angular_error(
    const PhaseVector& reference
) const {
    return decode().mean_angular_error(reference);
}

void QuantizedPhaseVector::serialize(std::ostream& output) const {
    for (const char value : magic) {
        output.put(value);
    }
    write_u32(output, format_version);
    write_u32(output, static_cast<std::uint32_t>(encoding_));
    if (dimension_ >
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()
        )) {
        throw std::runtime_error(
            "quantized phase dimension exceeds format"
        );
    }
    write_u32(output, static_cast<std::uint32_t>(dimension_));
    write_u32(
        output,
        std::bit_cast<std::uint32_t>(residual_base_)
    );
    if (storage_.size() >
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()
        )) {
        throw std::runtime_error(
            "quantized phase payload exceeds format"
        );
    }
    write_u32(output, static_cast<std::uint32_t>(storage_.size()));
    output.write(
        reinterpret_cast<const char*>(storage_.data()),
        static_cast<std::streamsize>(storage_.size())
    );
    if (!output) {
        throw std::runtime_error(
            "failed to serialize quantized phase vector"
        );
    }
}

QuantizedPhaseVector QuantizedPhaseVector::deserialize(
    std::istream& input,
    const std::size_t maximum_dimension
) {
    for (const char expected : magic) {
        const int actual = input.get();
        if (actual == std::char_traits<char>::eof() ||
            static_cast<char>(actual) != expected) {
            throw std::runtime_error(
                "invalid quantized phase vector magic"
            );
        }
    }
    if (read_u32(input) != format_version) {
        throw std::runtime_error(
            "unsupported quantized phase vector version"
        );
    }
    const std::uint32_t encoded = read_u32(input);
    if (encoded > static_cast<std::uint32_t>(
            PhaseEncoding::int8_residual
        )) {
        throw std::runtime_error(
            "invalid quantized phase encoding"
        );
    }
    const std::size_t dimension = read_u32(input);
    if (dimension == 0U || dimension > maximum_dimension) {
        throw std::runtime_error(
            "invalid quantized phase dimension"
        );
    }
    const float residual_base =
        std::bit_cast<float>(read_u32(input));
    const std::size_t payload_size = read_u32(input);
    const std::size_t maximum_payload =
        dimension * sizeof(float);
    if (payload_size == 0U || payload_size > maximum_payload) {
        throw std::runtime_error(
            "invalid quantized phase payload size"
        );
    }
    std::vector<std::uint8_t> storage(payload_size);
    input.read(
        reinterpret_cast<char*>(storage.data()),
        static_cast<std::streamsize>(storage.size())
    );
    if (!input) {
        throw std::runtime_error(
            "truncated quantized phase vector payload"
        );
    }
    return QuantizedPhaseVector(
        static_cast<PhaseEncoding>(encoded),
        dimension,
        residual_base,
        std::move(storage)
    );
}

}  // namespace rlf::core
