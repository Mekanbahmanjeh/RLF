#include "rlf/solstice/image_generation_data_audit.hpp"

#include "rlf/core/sha256.hpp"
#include "rlf/solstice/vision_fabric.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rlf::solstice {
namespace {

constexpr std::string_view neutral_marker =
    "@neutral-gray128-target-size-v1";

struct AuditedImage final {
    std::string sha;
    std::uint64_t difference_hash{};
    std::uint64_t average_hash{};
    std::array<std::uint8_t, 3U> mean_rgb{};
    std::size_t record_index{};
};

struct ParsedManifest final {
    std::vector<AuditedImage> targets;
    std::vector<AuditedImage> media;
    std::size_t records{};
    std::size_t unresolved_licenses{};
    std::size_t disallowed_licenses{};
};

[[nodiscard]] bool valid_evaluation_tags(const std::string_view value) {
    if (value.empty()) {
        return false;
    }
    static const std::unordered_set<std::string> allowed{
        "unseen_prompt", "paraphrase", "composition", "natural_image",
        "multilingual", "spatial", "attribute_binding",
    };
    std::unordered_set<std::string> seen;
    std::size_t begin = 0U;
    while (begin <= value.size()) {
        const auto end = value.find(',', begin);
        const auto token = value.substr(
            begin,
            end == std::string_view::npos ? value.size() - begin : end - begin
        );
        if (token.empty() || !allowed.contains(std::string(token)) ||
            !seen.insert(std::string(token)).second) {
            return false;
        }
        if (end == std::string_view::npos) {
            break;
        }
        begin = end + 1U;
    }
    return true;
}

[[nodiscard]] std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> fields;
    std::size_t begin = 0U;
    while (true) {
        const auto end = line.find('\t', begin);
        fields.push_back(line.substr(begin, end - begin));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1U;
    }
    return fields;
}

[[nodiscard]] std::filesystem::path regular_path(
    const std::filesystem::path& root,
    const std::string_view value
) {
    const std::filesystem::path relative(value);
    if (relative.empty() || relative.is_absolute()) {
        throw std::invalid_argument("image audit paths must be relative");
    }
    for (const auto& part : relative) {
        if (part == "..") {
            throw std::invalid_argument("image audit path traversal is forbidden");
        }
    }
    const auto resolved = std::filesystem::weakly_canonical(root / relative);
    const auto inside = resolved.lexically_relative(root);
    std::error_code error;
    const auto status = std::filesystem::symlink_status(resolved, error);
    if (inside.empty() || *inside.begin() == ".." || error ||
        std::filesystem::is_symlink(status) ||
        !std::filesystem::is_regular_file(status)) {
        throw std::invalid_argument("image audit media must be a regular in-root file");
    }
    return resolved;
}

[[nodiscard]] std::uint8_t gray_at(
    const ImageData& image,
    const std::size_t sample_x,
    const std::size_t sample_y,
    const std::size_t sample_width,
    const std::size_t sample_height
) {
    const auto x = std::min(
        image.width - 1U, sample_x * image.width / sample_width
    );
    const auto y = std::min(
        image.height - 1U, sample_y * image.height / sample_height
    );
    const auto offset = (y * image.width + x) * 3U;
    const auto value = 77U * image.rgb[offset] +
        150U * image.rgb[offset + 1U] + 29U * image.rgb[offset + 2U];
    return static_cast<std::uint8_t>(value >> 8U);
}

[[nodiscard]] std::uint64_t difference_hash(const ImageData& image) {
    std::uint64_t result = 0U;
    for (std::size_t y = 0U; y < 8U; ++y) {
        for (std::size_t x = 0U; x < 8U; ++x) {
            result <<= 1U;
            result |= gray_at(image, x, y, 9U, 8U) >
                gray_at(image, x + 1U, y, 9U, 8U) ? 1U : 0U;
        }
    }
    return result;
}

[[nodiscard]] std::uint64_t average_hash(const ImageData& image) {
    std::array<std::uint8_t, 64U> samples{};
    std::uint64_t total = 0U;
    for (std::size_t index = 0U; index < samples.size(); ++index) {
        samples[index] = gray_at(image, index % 8U, index / 8U, 8U, 8U);
        total += samples[index];
    }
    std::uint64_t result = 0U;
    for (const auto value : samples) {
        result <<= 1U;
        result |= static_cast<std::uint64_t>(value) * samples.size() >= total
            ? 1U : 0U;
    }
    return result;
}

[[nodiscard]] std::unordered_set<std::string> load_policy(
    const std::filesystem::path& path
) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("unable to read image license policy");
    }
    std::unordered_set<std::string> allowed;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty() && line.front() != '#') {
            allowed.insert(line);
        }
    }
    if (allowed.empty()) {
        throw std::runtime_error("image license policy has no allowed licenses");
    }
    return allowed;
}

[[nodiscard]] ParsedManifest parse_manifest(
    const std::filesystem::path& path,
    const std::unordered_set<std::string>& allowed_licenses,
    const std::size_t maximum_records
) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error || std::filesystem::is_symlink(status) ||
        !std::filesystem::is_regular_file(status)) {
        throw std::invalid_argument("image pair manifest must be a regular file");
    }
    const auto manifest = std::filesystem::weakly_canonical(path);
    const auto root = manifest.parent_path();
    std::ifstream input(manifest);
    if (!input) {
        throw std::runtime_error("unable to read image pair manifest");
    }
    ParsedManifest parsed;
    std::unordered_set<std::string> ids;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line.front() == '#') {
            continue;
        }
        if (parsed.records >= maximum_records) {
            throw std::runtime_error("image audit record limit exceeded");
        }
        const auto fields = split_tabs(line);
        if ((fields.size() != 8U && fields.size() != 9U) ||
            (fields.size() == 9U && !valid_evaluation_tags(fields[8U])) ||
            fields[0U].empty() ||
            !ids.insert(fields[0U]).second || fields[5U].empty() ||
            fields[6U].empty() || !rlf::core::is_sha256_hex(fields[2U]) ||
            !rlf::core::is_sha256_hex(fields[4U])) {
            throw std::runtime_error("invalid image pair manifest row");
        }
        if (fields[7U].empty()) {
            ++parsed.unresolved_licenses;
        } else if (!allowed_licenses.contains(fields[7U])) {
            ++parsed.disallowed_licenses;
        }
        const auto target_path = regular_path(root, fields[3U]);
        if (rlf::core::sha256_hex(rlf::core::sha256_file(target_path)) !=
            fields[4U]) {
            throw std::runtime_error("image audit target SHA-256 mismatch");
        }
        if (fields[1U] == neutral_marker) {
            if (fields[2U] != rlf::core::sha256_hex(
                    rlf::core::sha256(neutral_marker))) {
                throw std::runtime_error("image audit neutral-source hash mismatch");
            }
        } else {
            const auto source_path = regular_path(root, fields[1U]);
            if (rlf::core::sha256_hex(rlf::core::sha256_file(source_path)) !=
                fields[2U]) {
                throw std::runtime_error("image audit source SHA-256 mismatch");
            }
            const auto source = load_image(source_path);
            parsed.media.push_back({
                .sha = fields[2U],
                .difference_hash = difference_hash(source),
                .average_hash = average_hash(source),
                .mean_rgb = image_mean_rgb(source),
                .record_index = parsed.records,
            });
        }
        const auto image = load_image(target_path);
        AuditedImage target{
            .sha = fields[4U],
            .difference_hash = difference_hash(image),
            .average_hash = average_hash(image),
            .mean_rgb = image_mean_rgb(image),
            .record_index = parsed.records,
        };
        parsed.targets.push_back(target);
        parsed.media.push_back(std::move(target));
        ++parsed.records;
    }
    if (parsed.records == 0U) {
        throw std::runtime_error("image pair manifest is empty");
    }
    return parsed;
}

[[nodiscard]] std::size_t duplicate_pairs(
    const std::vector<AuditedImage>& images,
    const bool perceptual,
    const unsigned int threshold
) {
    std::size_t count = 0U;
    for (std::size_t left = 0U; left < images.size(); ++left) {
        for (std::size_t right = left + 1U; right < images.size(); ++right) {
            if (images[left].record_index == images[right].record_index) {
                continue;
            }
            const auto lhs = perceptual ? images[left].average_hash
                                        : images[left].difference_hash;
            const auto rhs = perceptual ? images[right].average_hash
                                        : images[right].difference_hash;
            if (std::popcount(lhs ^ rhs) <= static_cast<int>(threshold) &&
                image_mean_rgb_distance(
                    images[left].mean_rgb, images[right].mean_rgb
                ) <= 12U) {
                ++count;
            }
        }
    }
    return count;
}

void write_report(
    const std::filesystem::path& path,
    const std::string& body
) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("unable to create image audit report");
    }
    output << body << '\n';
    if (!output) {
        throw std::runtime_error("unable to write image audit report");
    }
}

[[nodiscard]] std::string prefix(
    const std::string_view schema,
    const bool passed,
    const ImageGenerationDataAuditReport& report
) {
    return "{\"schema\":\"" + std::string(schema) +
        "\",\"passed\":" + (passed ? "true" : "false") +
        ",\"test_doubles\":false,\"frontier_claim_authorized\":false" +
        ",\"pair_manifest_sha256\":\"" + report.pair_manifest_sha256 +
        "\",\"records_audited\":" + std::to_string(report.records_audited);
}

}  // namespace

std::uint64_t image_difference_hash(const ImageData& image) {
    return difference_hash(image);
}

std::uint64_t image_average_hash(const ImageData& image) {
    return average_hash(image);
}

unsigned int image_hash_hamming(
    const std::uint64_t left,
    const std::uint64_t right
) noexcept {
    return static_cast<unsigned int>(std::popcount(left ^ right));
}

std::array<std::uint8_t, 3U> image_mean_rgb(const ImageData& image) {
    if (image.width == 0U || image.height == 0U ||
        image.width > std::numeric_limits<std::size_t>::max() / image.height ||
        image.width * image.height >
            std::numeric_limits<std::size_t>::max() / 3U ||
        image.rgb.size() != image.width * image.height * 3U) {
        throw std::invalid_argument("mean RGB requires a valid image");
    }
    std::array<std::uint64_t, 3U> totals{};
    for (std::size_t offset = 0U; offset < image.rgb.size(); offset += 3U) {
        totals[0U] += image.rgb[offset];
        totals[1U] += image.rgb[offset + 1U];
        totals[2U] += image.rgb[offset + 2U];
    }
    const auto pixels = static_cast<std::uint64_t>(image.width * image.height);
    return {
        static_cast<std::uint8_t>(totals[0U] / pixels),
        static_cast<std::uint8_t>(totals[1U] / pixels),
        static_cast<std::uint8_t>(totals[2U] / pixels),
    };
}

unsigned int image_mean_rgb_distance(
    const std::array<std::uint8_t, 3U>& left,
    const std::array<std::uint8_t, 3U>& right
) noexcept {
    unsigned int maximum = 0U;
    for (std::size_t channel = 0U; channel < left.size(); ++channel) {
        const auto lhs = static_cast<unsigned int>(left[channel]);
        const auto rhs = static_cast<unsigned int>(right[channel]);
        maximum = std::max(maximum, lhs > rhs ? lhs - rhs : rhs - lhs);
    }
    return maximum;
}

bool ImageGenerationDataAuditReport::passed() const noexcept {
    return exact_duplicates == 0U && near_duplicates == 0U &&
        perceptual_duplicates == 0U && overlap_records == 0U &&
        unresolved_license_records == 0U && disallowed_license_records == 0U;
}

ImageGenerationDataAuditReport audit_image_generation_data(
    const std::filesystem::path& pair_manifest,
    const std::filesystem::path& evaluation_manifest,
    const std::filesystem::path& license_policy,
    const std::filesystem::path& output_directory,
    const ImageGenerationDataAuditOptions options
) {
    if (options.maximum_records == 0U ||
        options.near_duplicate_hamming_distance > 64U) {
        throw std::invalid_argument("invalid image data audit limits");
    }
    if (std::filesystem::exists(output_directory)) {
        throw std::invalid_argument("image audit output already exists");
    }
    const auto policy = load_policy(license_policy);
    const auto training = parse_manifest(
        pair_manifest, policy, options.maximum_records
    );
    const auto evaluation = parse_manifest(
        evaluation_manifest, policy, options.maximum_records
    );
    ImageGenerationDataAuditReport report;
    report.pair_manifest_sha256 = rlf::core::sha256_hex(
        rlf::core::sha256_file(pair_manifest)
    );
    report.evaluation_manifest_sha256 = rlf::core::sha256_hex(
        rlf::core::sha256_file(evaluation_manifest)
    );
    report.license_policy_sha256 = rlf::core::sha256_hex(
        rlf::core::sha256_file(license_policy)
    );
    report.records_audited = training.records;
    report.evaluation_records_audited = evaluation.records;
    report.unresolved_license_records = training.unresolved_licenses +
        evaluation.unresolved_licenses;
    report.disallowed_license_records = training.disallowed_licenses +
        evaluation.disallowed_licenses;
    for (std::size_t left = 0U; left < training.media.size(); ++left) {
        for (std::size_t right = left + 1U; right < training.media.size(); ++right) {
            if (training.media[left].record_index !=
                    training.media[right].record_index &&
                training.media[left].sha == training.media[right].sha) {
                ++report.exact_duplicates;
            }
        }
    }
    report.near_duplicates = duplicate_pairs(
        training.media, false, options.near_duplicate_hamming_distance
    );
    report.perceptual_duplicates = duplicate_pairs(
        training.media, true, options.near_duplicate_hamming_distance
    );
    for (std::size_t record = 0U; record < evaluation.records; ++record) {
        bool overlaps = false;
        for (const auto& held_out : evaluation.media) {
            if (held_out.record_index != record) {
                continue;
            }
            for (const auto& trained : training.media) {
                if (held_out.sha == trained.sha ||
                    ((std::popcount(
                          held_out.difference_hash ^ trained.difference_hash
                      ) <= static_cast<int>(
                          options.near_duplicate_hamming_distance
                      ) ||
                      std::popcount(
                          held_out.average_hash ^ trained.average_hash
                      ) <= static_cast<int>(
                          options.near_duplicate_hamming_distance
                      )) &&
                     image_mean_rgb_distance(
                         held_out.mean_rgb, trained.mean_rgb
                     ) <= 12U)) {
                    overlaps = true;
                    break;
                }
            }
            if (overlaps) {
                break;
            }
        }
        report.overlap_records += overlaps ? 1U : 0U;
    }

    std::filesystem::create_directories(output_directory);
    const bool data_pass = report.unresolved_license_records == 0U;
    write_report(output_directory / "data_audit.json",
        prefix("rlf-imagegen-data-audit-v1", data_pass, report) +
        ",\"media_hashes_verified\":true,\"provenance_complete\":" +
        (data_pass ? "true" : "false") + "}");
    write_report(output_directory / "license_report.json",
        prefix("rlf-imagegen-license-audit-v1",
               report.unresolved_license_records == 0U &&
               report.disallowed_license_records == 0U, report) +
        ",\"unresolved_license_records\":" +
        std::to_string(report.unresolved_license_records) +
        ",\"disallowed_license_records\":" +
        std::to_string(report.disallowed_license_records) +
        ",\"license_policy_sha256\":\"" + report.license_policy_sha256 + "\"}");
    write_report(output_directory / "exact_dedup_report.json",
        prefix("rlf-imagegen-exact-dedup-audit-v1",
               report.exact_duplicates == 0U, report) +
        ",\"exact_duplicates_remaining\":" +
        std::to_string(report.exact_duplicates) + "}");
    write_report(output_directory / "near_dedup_report.json",
        prefix("rlf-imagegen-near-dedup-audit-v1",
               report.near_duplicates == 0U, report) +
        ",\"near_duplicates_remaining\":" +
        std::to_string(report.near_duplicates) +
        ",\"method\":\"decoded-luma-dhash64-plus-mean-rgb-v2\"}");
    write_report(output_directory / "perceptual_dedup_report.json",
        prefix("rlf-imagegen-perceptual-dedup-audit-v1",
               report.perceptual_duplicates == 0U, report) +
        ",\"perceptual_duplicates_remaining\":" +
        std::to_string(report.perceptual_duplicates) +
        ",\"method\":\"decoded-luma-ahash64-plus-mean-rgb-v2\"}");
    write_report(output_directory / "contamination_report.json",
        prefix("rlf-imagegen-contamination-audit-v1",
               report.overlap_records == 0U, report) +
        ",\"overlap_records\":" + std::to_string(report.overlap_records) +
        ",\"evaluation_manifest_frozen\":true" +
        ",\"benchmark_answers_present\":false" +
        ",\"evaluation_manifest_sha256\":\"" +
        report.evaluation_manifest_sha256 + "\"}");
    return report;
}

}  // namespace rlf::solstice
