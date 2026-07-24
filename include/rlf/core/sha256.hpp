#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>

namespace rlf::core {

using Sha256Digest = std::array<std::uint8_t, 32U>;

[[nodiscard]] Sha256Digest sha256(std::span<const std::uint8_t> bytes) noexcept;
[[nodiscard]] Sha256Digest sha256(std::string_view text) noexcept;
[[nodiscard]] Sha256Digest sha256_file(const std::filesystem::path& path);
[[nodiscard]] Sha256Digest sha256_file_range(
    const std::filesystem::path& path,
    std::uint64_t offset,
    std::uint64_t length
);
[[nodiscard]] std::string sha256_hex(const Sha256Digest& digest);
[[nodiscard]] bool is_sha256_hex(std::string_view value) noexcept;

}  // namespace rlf::core
