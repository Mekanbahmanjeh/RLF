#pragma once

#include "rlf/solstice/vision_fabric.hpp"

#include <cstddef>
#include <cstdint>
#include <array>
#include <filesystem>
#include <string>

namespace rlf::solstice {

struct ImageGenerationDataAuditOptions final {
    std::size_t maximum_records{100'000U};
    unsigned int near_duplicate_hamming_distance{3U};
};

struct ImageGenerationDataAuditReport final {
    std::string pair_manifest_sha256;
    std::string evaluation_manifest_sha256;
    std::string license_policy_sha256;
    std::size_t records_audited{};
    std::size_t evaluation_records_audited{};
    std::size_t exact_duplicates{};
    std::size_t near_duplicates{};
    std::size_t perceptual_duplicates{};
    std::size_t overlap_records{};
    std::size_t unresolved_license_records{};
    std::size_t disallowed_license_records{};

    [[nodiscard]] bool passed() const noexcept;
};

[[nodiscard]] std::uint64_t image_difference_hash(const ImageData& image);
[[nodiscard]] std::uint64_t image_average_hash(const ImageData& image);
[[nodiscard]] unsigned int image_hash_hamming(
    std::uint64_t left,
    std::uint64_t right
) noexcept;
[[nodiscard]] std::array<std::uint8_t, 3U> image_mean_rgb(
    const ImageData& image
);
[[nodiscard]] unsigned int image_mean_rgb_distance(
    const std::array<std::uint8_t, 3U>& left,
    const std::array<std::uint8_t, 3U>& right
) noexcept;

// Audits the immutable eight-column image pair manifests, computes exact and
// perceptual fingerprints from decoded pixels, and writes the six reports
// consumed by verify_imagegen_data_audits.sh. No neural model is involved.
[[nodiscard]] ImageGenerationDataAuditReport audit_image_generation_data(
    const std::filesystem::path& pair_manifest,
    const std::filesystem::path& evaluation_manifest,
    const std::filesystem::path& license_policy,
    const std::filesystem::path& output_directory,
    ImageGenerationDataAuditOptions options = {}
);

}  // namespace rlf::solstice
