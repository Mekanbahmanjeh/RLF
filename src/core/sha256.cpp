#include "rlf/core/sha256.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace rlf::core {
namespace {

constexpr std::array<std::uint32_t, 64U> round_constants{
    0x428A2F98U, 0x71374491U, 0xB5C0FBCFU, 0xE9B5DBA5U,
    0x3956C25BU, 0x59F111F1U, 0x923F82A4U, 0xAB1C5ED5U,
    0xD807AA98U, 0x12835B01U, 0x243185BEU, 0x550C7DC3U,
    0x72BE5D74U, 0x80DEB1FEU, 0x9BDC06A7U, 0xC19BF174U,
    0xE49B69C1U, 0xEFBE4786U, 0x0FC19DC6U, 0x240CA1CCU,
    0x2DE92C6FU, 0x4A7484AAU, 0x5CB0A9DCU, 0x76F988DAU,
    0x983E5152U, 0xA831C66DU, 0xB00327C8U, 0xBF597FC7U,
    0xC6E00BF3U, 0xD5A79147U, 0x06CA6351U, 0x14292967U,
    0x27B70A85U, 0x2E1B2138U, 0x4D2C6DFCU, 0x53380D13U,
    0x650A7354U, 0x766A0ABBU, 0x81C2C92EU, 0x92722C85U,
    0xA2BFE8A1U, 0xA81A664BU, 0xC24B8B70U, 0xC76C51A3U,
    0xD192E819U, 0xD6990624U, 0xF40E3585U, 0x106AA070U,
    0x19A4C116U, 0x1E376C08U, 0x2748774CU, 0x34B0BCB5U,
    0x391C0CB3U, 0x4ED8AA4AU, 0x5B9CCA4FU, 0x682E6FF3U,
    0x748F82EEU, 0x78A5636FU, 0x84C87814U, 0x8CC70208U,
    0x90BEFFFAU, 0xA4506CEBU, 0xBEF9A3F7U, 0xC67178F2U,
};

class Sha256State final {
public:
    void update(const std::span<const std::uint8_t> bytes) noexcept {
        for (const std::uint8_t byte : bytes) {
            block_[block_size_++] = byte;
            ++total_bytes_;
            if (block_size_ == block_.size()) {
                transform(block_);
                block_size_ = 0U;
            }
        }
    }

    [[nodiscard]] Sha256Digest finish() noexcept {
        const std::uint64_t bit_length = total_bytes_ * 8ULL;
        block_[block_size_++] = 0x80U;
        if (block_size_ > 56U) {
            while (block_size_ < block_.size()) block_[block_size_++] = 0U;
            transform(block_);
            block_size_ = 0U;
        }
        while (block_size_ < 56U) block_[block_size_++] = 0U;
        for (std::size_t index = 0U; index < 8U; ++index) {
            const unsigned int shift = static_cast<unsigned int>((7U - index) * 8U);
            block_[block_size_++] = static_cast<std::uint8_t>(bit_length >> shift);
        }
        transform(block_);
        Sha256Digest digest{};
        for (std::size_t word = 0U; word < state_.size(); ++word) {
            for (std::size_t byte = 0U; byte < 4U; ++byte) {
                const unsigned int shift = static_cast<unsigned int>((3U - byte) * 8U);
                digest[word * 4U + byte] = static_cast<std::uint8_t>(state_[word] >> shift);
            }
        }
        return digest;
    }

private:
    void transform(const std::array<std::uint8_t, 64U>& block) noexcept {
        std::array<std::uint32_t, 64U> words{};
        for (std::size_t index = 0U; index < 16U; ++index) {
            const std::size_t offset = index * 4U;
            words[index] = (static_cast<std::uint32_t>(block[offset]) << 24U) |
                (static_cast<std::uint32_t>(block[offset + 1U]) << 16U) |
                (static_cast<std::uint32_t>(block[offset + 2U]) << 8U) |
                static_cast<std::uint32_t>(block[offset + 3U]);
        }
        for (std::size_t index = 16U; index < words.size(); ++index) {
            const std::uint32_t s0 = std::rotr(words[index - 15U], 7) ^
                std::rotr(words[index - 15U], 18) ^ (words[index - 15U] >> 3U);
            const std::uint32_t s1 = std::rotr(words[index - 2U], 17) ^
                std::rotr(words[index - 2U], 19) ^ (words[index - 2U] >> 10U);
            words[index] = words[index - 16U] + s0 + words[index - 7U] + s1;
        }
        std::uint32_t a = state_[0U];
        std::uint32_t b = state_[1U];
        std::uint32_t c = state_[2U];
        std::uint32_t d = state_[3U];
        std::uint32_t e = state_[4U];
        std::uint32_t f = state_[5U];
        std::uint32_t g = state_[6U];
        std::uint32_t h = state_[7U];
        for (std::size_t index = 0U; index < words.size(); ++index) {
            const std::uint32_t upper_sigma_one = std::rotr(e, 6) ^
                std::rotr(e, 11) ^ std::rotr(e, 25);
            const std::uint32_t choose = (e & f) ^ ((~e) & g);
            const std::uint32_t temporary_one = h + upper_sigma_one + choose +
                round_constants[index] + words[index];
            const std::uint32_t upper_sigma_zero = std::rotr(a, 2) ^
                std::rotr(a, 13) ^ std::rotr(a, 22);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temporary_two = upper_sigma_zero + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary_one;
            d = c;
            c = b;
            b = a;
            a = temporary_one + temporary_two;
        }
        state_[0U] += a;
        state_[1U] += b;
        state_[2U] += c;
        state_[3U] += d;
        state_[4U] += e;
        state_[5U] += f;
        state_[6U] += g;
        state_[7U] += h;
    }

    std::array<std::uint32_t, 8U> state_{
        0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U, 0xA54FF53AU,
        0x510E527FU, 0x9B05688CU, 0x1F83D9ABU, 0x5BE0CD19U,
    };
    std::array<std::uint8_t, 64U> block_{};
    std::size_t block_size_{};
    std::uint64_t total_bytes_{};
};

}  // namespace

Sha256Digest sha256(const std::span<const std::uint8_t> bytes) noexcept {
    Sha256State state;
    state.update(bytes);
    return state.finish();
}

Sha256Digest sha256(const std::string_view text) noexcept {
    return sha256(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(text.data()), text.size()
    ));
}

Sha256Digest sha256_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("unable to open file for SHA-256: " + path.string());
    Sha256State state;
    std::array<char, 1U << 16U> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) {
            state.update(std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(buffer.data()),
                static_cast<std::size_t>(count)
            ));
        }
    }
    if (!input.eof()) throw std::runtime_error("failed while hashing file: " + path.string());
    return state.finish();
}

Sha256Digest sha256_file_range(
    const std::filesystem::path& path,
    const std::uint64_t offset,
    const std::uint64_t length
) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("unable to open file for ranged SHA-256: " + path.string());
    }
    const std::streampos end = input.tellg();
    if (end < 0) {
        throw std::runtime_error("unable to determine file size for ranged SHA-256");
    }
    const auto file_size = static_cast<std::uint64_t>(end);
    if (offset > file_size || length > file_size - offset ||
        offset > static_cast<std::uint64_t>(
            std::numeric_limits<std::streamoff>::max()
        )) {
        throw std::runtime_error("ranged SHA-256 request exceeds file bounds");
    }
    input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!input) {
        throw std::runtime_error("unable to seek for ranged SHA-256");
    }
    Sha256State state;
    std::array<char, 1U << 16U> buffer{};
    std::uint64_t remaining = length;
    while (remaining != 0U) {
        const std::size_t requested = static_cast<std::size_t>(std::min<std::uint64_t>(
            remaining,
            buffer.size()
        ));
        input.read(buffer.data(), static_cast<std::streamsize>(requested));
        if (input.gcount() != static_cast<std::streamsize>(requested)) {
            throw std::runtime_error("truncated file during ranged SHA-256");
        }
        state.update(std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(buffer.data()),
            requested
        ));
        remaining -= requested;
    }
    return state.finish();
}

std::string sha256_hex(const Sha256Digest& digest) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const std::uint8_t byte : digest) {
        output << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return output.str();
}

bool is_sha256_hex(const std::string_view value) noexcept {
    if (value.size() != 64U) return false;
    for (const char character : value) {
        const bool digit = character >= '0' && character <= '9';
        const bool lower = character >= 'a' && character <= 'f';
        const bool upper = character >= 'A' && character <= 'F';
        if (!digit && !lower && !upper) return false;
    }
    return true;
}

}  // namespace rlf::core
