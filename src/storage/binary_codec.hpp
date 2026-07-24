#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace rlf::storage::detail {

constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;

[[nodiscard]] inline std::uint64_t checksum(
    const std::span<const std::uint8_t> bytes
) noexcept {
    std::uint64_t value = fnv_offset_basis;
    for (const std::uint8_t byte : bytes) {
        value ^= static_cast<std::uint64_t>(byte);
        value *= fnv_prime;
    }
    return value;
}

class BufferWriter final {
public:
    void write_u8(const std::uint8_t value) {
        bytes_.push_back(value);
    }

    void write_bool(const bool value) {
        write_u8(value ? 1U : 0U);
    }

    void write_u32(const std::uint32_t value) {
        for (unsigned int byte_index = 0U; byte_index < 4U; ++byte_index) {
            write_u8(static_cast<std::uint8_t>(
                (value >> (byte_index * 8U)) & 0xFFU
            ));
        }
    }

    void write_u64(const std::uint64_t value) {
        for (unsigned int byte_index = 0U; byte_index < 8U; ++byte_index) {
            write_u8(static_cast<std::uint8_t>(
                (value >> (byte_index * 8U)) & 0xFFULL
            ));
        }
    }

    void write_float(const float value) {
        write_u32(std::bit_cast<std::uint32_t>(value));
    }

    void write_double(const double value) {
        write_u64(std::bit_cast<std::uint64_t>(value));
    }

    void write_bytes(const std::span<const std::uint8_t> bytes) {
        bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
    }

    void write_string(const std::string_view value) {
        write_u64(static_cast<std::uint64_t>(value.size()));
        for (const char character : value) {
            write_u8(static_cast<std::uint8_t>(character));
        }
    }

    [[nodiscard]] const std::vector<std::uint8_t>& bytes() const noexcept {
        return bytes_;
    }

    [[nodiscard]] std::vector<std::uint8_t> take() noexcept {
        return std::move(bytes_);
    }

private:
    std::vector<std::uint8_t> bytes_;
};

class BufferReader final {
public:
    explicit BufferReader(const std::span<const std::uint8_t> bytes) noexcept
        : bytes_(bytes) {}

    [[nodiscard]] std::size_t remaining() const noexcept {
        return bytes_.size() - position_;
    }

    [[nodiscard]] bool empty() const noexcept {
        return position_ == bytes_.size();
    }

    [[nodiscard]] std::uint8_t read_u8() {
        require(1U);
        return bytes_[position_++];
    }

    [[nodiscard]] bool read_bool() {
        const std::uint8_t value = read_u8();
        if (value > 1U) {
            throw std::runtime_error("invalid serialized boolean");
        }
        return value == 1U;
    }

    [[nodiscard]] std::uint32_t read_u32() {
        require(4U);
        std::uint32_t value = 0U;
        for (unsigned int byte_index = 0U; byte_index < 4U; ++byte_index) {
            value |= static_cast<std::uint32_t>(read_u8())
                << (byte_index * 8U);
        }
        return value;
    }

    [[nodiscard]] std::uint64_t read_u64() {
        require(8U);
        std::uint64_t value = 0ULL;
        for (unsigned int byte_index = 0U; byte_index < 8U; ++byte_index) {
            value |= static_cast<std::uint64_t>(read_u8())
                << (byte_index * 8U);
        }
        return value;
    }

    [[nodiscard]] float read_float() {
        return std::bit_cast<float>(read_u32());
    }

    [[nodiscard]] double read_double() {
        return std::bit_cast<double>(read_u64());
    }

    [[nodiscard]] std::vector<std::uint8_t> read_bytes(
        const std::size_t count
    ) {
        require(count);
        const auto first = bytes_.begin() +
            static_cast<std::ptrdiff_t>(position_);
        const auto last = first + static_cast<std::ptrdiff_t>(count);
        std::vector<std::uint8_t> result(first, last);
        position_ += count;
        return result;
    }

    [[nodiscard]] std::string read_string(
        const std::size_t maximum_size
    ) {
        const std::uint64_t serialized_size = read_u64();
        if (serialized_size > static_cast<std::uint64_t>(maximum_size) ||
            serialized_size >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max()
                )) {
            throw std::runtime_error("serialized string exceeds size limit");
        }
        const auto size = static_cast<std::size_t>(serialized_size);
        require(size);
        std::string result;
        result.reserve(size);
        for (std::size_t index = 0U; index < size; ++index) {
            result.push_back(static_cast<char>(read_u8()));
        }
        return result;
    }

private:
    void require(const std::size_t count) const {
        if (count > remaining()) {
            throw std::runtime_error("truncated binary data");
        }
    }

    std::span<const std::uint8_t> bytes_;
    std::size_t position_{0U};
};

[[nodiscard]] inline std::vector<std::uint8_t> read_file(
    const std::filesystem::path& path,
    const std::size_t maximum_bytes
) {
    std::error_code size_error;
    const std::uintmax_t file_size =
        std::filesystem::file_size(path, size_error);
    if (size_error) {
        throw std::runtime_error(
            "unable to inspect file: " + path.string()
        );
    }
    if (file_size > static_cast<std::uintmax_t>(maximum_bytes) ||
        file_size >
            static_cast<std::uintmax_t>(
                std::numeric_limits<std::size_t>::max()
            )) {
        throw std::runtime_error("file exceeds configured size limit");
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("unable to open file: " + path.string());
    }
    std::vector<std::uint8_t> bytes(
        static_cast<std::size_t>(file_size)
    );
    if (!bytes.empty()) {
        input.read(
            reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size())
        );
    }
    if (!input || input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("failed to read complete file");
    }
    return bytes;
}

inline void write_file_transactionally(
    const std::filesystem::path& path,
    const std::span<const std::uint8_t> bytes
) {
    if (path.empty()) {
        throw std::invalid_argument("output path must not be empty");
    }
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::filesystem::path temporary_path = path;
    temporary_path += ".tmp";
    {
        std::ofstream output(
            temporary_path,
            std::ios::binary | std::ios::trunc
        );
        if (!output) {
            throw std::runtime_error(
                "unable to open temporary file: " +
                temporary_path.string()
            );
        }
        if (!bytes.empty()) {
            output.write(
                reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size())
            );
        }
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "failed to flush temporary file: " +
                temporary_path.string()
            );
        }
    }

    std::error_code rename_error;
    std::filesystem::rename(temporary_path, path, rename_error);
    if (rename_error) {
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        rename_error.clear();
        std::filesystem::rename(temporary_path, path, rename_error);
    }
    if (rename_error) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary_path, cleanup_error);
        throw std::runtime_error(
            "failed to publish file: " + rename_error.message()
        );
    }
}

template <typename Integer>
[[nodiscard]] inline std::size_t checked_size(
    const Integer value,
    const std::size_t maximum,
    const std::string_view label
) {
    static_assert(std::is_integral_v<Integer>);
    if constexpr (std::is_signed_v<Integer>) {
        if (value < 0) {
            throw std::runtime_error(
                std::string(label) + " is negative"
            );
        }
    }
    using Unsigned = std::make_unsigned_t<Integer>;
    const auto unsigned_value = static_cast<Unsigned>(value);
    if (unsigned_value > static_cast<Unsigned>(maximum) ||
        unsigned_value >
            static_cast<Unsigned>(
                std::numeric_limits<std::size_t>::max()
            )) {
        throw std::runtime_error(
            std::string(label) + " exceeds configured size limit"
        );
    }
    return static_cast<std::size_t>(unsigned_value);
}

}  // namespace rlf::storage::detail
