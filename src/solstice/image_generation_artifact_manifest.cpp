#include "rlf/solstice/image_generation_artifact_manifest.hpp"

#include "rlf/core/sha256.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace rlf::solstice {
namespace {

constexpr std::string_view manifest_header =
    "RLF_IMAGEGEN_ARTIFACT_MANIFEST\t1";
constexpr std::string_view sidecar_header =
    "RLF_IMAGEGEN_MANIFEST_SHA256\t1";

[[nodiscard]] bool is_lowercase_sha256(const std::string_view value) noexcept {
    if (!rlf::core::is_sha256_hex(value)) return false;
    return std::ranges::none_of(value, [](const char character) {
        return character >= 'A' && character <= 'F';
    });
}

[[nodiscard]] bool is_safe_text(const std::string_view value) noexcept {
    if (value.empty()) return false;
    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
        if (character < 0x20U || character == 0x7FU) return false;
    }
    return true;
}

[[nodiscard]] std::vector<std::string_view> split_exact(
    const std::string_view value,
    const char separator
) {
    std::vector<std::string_view> fields;
    std::size_t begin = 0U;
    while (true) {
        const std::size_t end = value.find(separator, begin);
        if (end == std::string_view::npos) {
            fields.push_back(value.substr(begin));
            return fields;
        }
        fields.push_back(value.substr(begin, end - begin));
        begin = end + 1U;
    }
}

[[nodiscard]] std::vector<std::string_view> canonical_lines(
    const std::string_view text,
    const std::string_view label
) {
    if (text.empty() || text.back() != '\n' || text.find('\r') != std::string_view::npos ||
        text.find('\0') != std::string_view::npos) {
        throw std::runtime_error(
            std::string(label) + " must use canonical LF-terminated text"
        );
    }
    std::vector<std::string_view> lines;
    std::size_t begin = 0U;
    while (begin < text.size()) {
        const std::size_t end = text.find('\n', begin);
        if (end == std::string_view::npos || end == begin) {
            throw std::runtime_error(std::string(label) + " contains an empty line");
        }
        lines.push_back(text.substr(begin, end - begin));
        begin = end + 1U;
    }
    return lines;
}

[[nodiscard]] std::uint64_t parse_canonical_u64(
    const std::string_view text,
    const std::string_view label
) {
    if (text.empty()) {
        throw std::runtime_error(std::string(label) + " is empty");
    }
    std::uint64_t value = 0U;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
        std::to_string(value) != text) {
        throw std::runtime_error(std::string(label) + " is not canonical decimal");
    }
    return value;
}

[[nodiscard]] std::string read_bounded_regular_file(
    const std::filesystem::path& path,
    const std::uint64_t maximum_bytes,
    const std::string_view label,
    const bool require_nonempty
) {
    std::error_code error;
    const std::filesystem::file_status status = std::filesystem::symlink_status(path, error);
    if (error) {
        throw std::runtime_error(
            "unable to inspect " + std::string(label) + ": " + error.message()
        );
    }
    if (std::filesystem::is_symlink(status)) {
        throw std::runtime_error(std::string(label) + " must not be a symlink");
    }
    if (!std::filesystem::is_regular_file(status)) {
        throw std::runtime_error(std::string(label) + " must be a regular file");
    }
    const std::uintmax_t raw_size = std::filesystem::file_size(path, error);
    if (error || raw_size > maximum_bytes ||
        raw_size > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(std::string(label) + " exceeds its size limit");
    }
    if (require_nonempty && raw_size == 0U) {
        throw std::runtime_error(std::string(label) + " must not be empty");
    }
    const auto size = static_cast<std::size_t>(raw_size);
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("unable to open " + std::string(label));
    }
    std::string contents(size, '\0');
    if (size != 0U) {
        input.read(contents.data(), static_cast<std::streamsize>(size));
        if (input.gcount() != static_cast<std::streamsize>(size)) {
            throw std::runtime_error("truncated " + std::string(label));
        }
    }
    if (input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error(std::string(label) + " changed while being read");
    }
    const std::filesystem::file_status final_status =
        std::filesystem::symlink_status(path, error);
    if (error || std::filesystem::is_symlink(final_status) ||
        !std::filesystem::is_regular_file(final_status) ||
        std::filesystem::file_size(path, error) != raw_size || error) {
        throw std::runtime_error(std::string(label) + " changed while being read");
    }
    return contents;
}

void validate_limits(const ImageGenerationArtifactManifestLimits& limits) {
    if (limits.maximum_manifest_bytes == 0U || limits.maximum_sidecar_bytes == 0U ||
        limits.maximum_relative_path_bytes == 0U ||
        limits.maximum_artifact_bytes == 0U ||
        limits.maximum_total_artifact_bytes < limits.maximum_artifact_bytes) {
        throw std::invalid_argument("invalid image-generation artifact manifest limits");
    }
}

[[nodiscard]] std::filesystem::path validate_relative_path(
    const std::string_view text,
    const std::size_t maximum_bytes
) {
    if (text.size() > maximum_bytes || !is_safe_text(text) ||
        text.find('\\') != std::string_view::npos ||
        text.find(':') != std::string_view::npos) {
        throw std::runtime_error("artifact relative path is empty, unsafe, or too long");
    }
    const std::filesystem::path path{std::string(text)};
    if (path.is_absolute() || path.has_root_name() || path.has_root_directory() ||
        path.empty() || path.filename().empty() || path.lexically_normal() != path) {
        throw std::runtime_error("artifact path must be canonical and relative");
    }
    for (const std::filesystem::path& component : path) {
        if (component.empty() || component == "." || component == "..") {
            throw std::runtime_error("artifact path contains a forbidden component");
        }
    }
    return path;
}

void reject_symlink_components(
    const std::filesystem::path& base,
    const std::filesystem::path& relative_path
) {
    std::filesystem::path current = base;
    std::size_t index = 0U;
    const std::size_t components = static_cast<std::size_t>(
        std::distance(relative_path.begin(), relative_path.end())
    );
    for (const std::filesystem::path& component : relative_path) {
        current /= component;
        std::error_code error;
        const auto status = std::filesystem::symlink_status(current, error);
        if (error) {
            throw std::runtime_error(
                "unable to inspect artifact path component: " + error.message()
            );
        }
        if (std::filesystem::is_symlink(status)) {
            throw std::runtime_error("artifact path traverses a symlink");
        }
        ++index;
        if (index < components && !std::filesystem::is_directory(status)) {
            throw std::runtime_error("artifact parent component is not a directory");
        }
    }
}

[[nodiscard]] std::filesystem::path canonical_existing_path(
    const std::filesystem::path& path,
    const std::string_view label
) {
    std::error_code error;
    const auto result = std::filesystem::canonical(path, error);
    if (error) {
        throw std::runtime_error(
            "unable to canonicalize " + std::string(label) + ": " + error.message()
        );
    }
    return result;
}

[[nodiscard]] bool paths_are_equivalent(
    const std::filesystem::path& left,
    const std::filesystem::path& right
) {
    std::error_code error;
    const bool equivalent = std::filesystem::equivalent(left, right, error);
    if (error) {
        throw std::runtime_error("unable to compare artifact file identities");
    }
    return equivalent;
}

void verify_sidecar(
    const std::filesystem::path& path,
    const std::string_view manifest_sha256,
    const ImageGenerationArtifactManifestLimits& limits
) {
    const std::string contents = read_bounded_regular_file(
        path,
        limits.maximum_sidecar_bytes,
        "image-generation manifest SHA-256 sidecar",
        true
    );
    const auto lines = canonical_lines(contents, "manifest SHA-256 sidecar");
    if (lines.size() != 2U || lines[0U] != sidecar_header) {
        throw std::runtime_error("invalid image-generation manifest sidecar schema");
    }
    const auto fields = split_exact(lines[1U], '\t');
    if (fields.size() != 2U || fields[0U] != "sha256" ||
        !is_lowercase_sha256(fields[1U]) || fields[1U] != manifest_sha256) {
        throw std::runtime_error("image-generation manifest self-hash mismatch");
    }
}

struct ParsedArtifact final {
    std::string kind;
    std::filesystem::path relative_path;
    std::uint64_t bytes{};
    std::string sha256;
};

[[nodiscard]] std::vector<ParsedArtifact> parse_manifest(
    const std::string_view contents,
    const ImageGenerationArtifactManifestLimits& limits
) {
    const auto lines = canonical_lines(contents, "image-generation artifact manifest");
    if (lines.size() != required_image_generation_artifact_kinds.size() + 1U ||
        lines[0U] != manifest_header) {
        throw std::runtime_error("invalid image-generation artifact manifest schema");
    }
    std::set<std::string, std::less<>> kinds;
    std::set<std::filesystem::path> paths;
    std::vector<ParsedArtifact> artifacts;
    artifacts.reserve(required_image_generation_artifact_kinds.size());
    for (std::size_t index = 0U;
         index < required_image_generation_artifact_kinds.size();
         ++index) {
        const auto fields = split_exact(lines[index + 1U], '\t');
        if (fields.size() != 4U) {
            throw std::runtime_error("artifact manifest row must contain four fields");
        }
        if (!kinds.insert(std::string(fields[0U])).second) {
            throw std::runtime_error("duplicate artifact kind in manifest");
        }
        if (fields[0U] != required_image_generation_artifact_kinds[index]) {
            throw std::runtime_error("missing, extra, or out-of-order artifact kind");
        }
        ParsedArtifact artifact;
        artifact.kind = std::string(fields[0U]);
        artifact.relative_path = validate_relative_path(
            fields[1U], limits.maximum_relative_path_bytes
        );
        if (!paths.insert(artifact.relative_path).second) {
            throw std::runtime_error("duplicate artifact path in manifest");
        }
        artifact.bytes = parse_canonical_u64(fields[2U], "artifact byte count");
        if (artifact.bytes == 0U || artifact.bytes > limits.maximum_artifact_bytes) {
            throw std::runtime_error("artifact byte count is empty or exceeds limit");
        }
        if (!is_lowercase_sha256(fields[3U])) {
            throw std::runtime_error("artifact SHA-256 must be lowercase hexadecimal");
        }
        artifact.sha256 = std::string(fields[3U]);
        artifacts.push_back(std::move(artifact));
    }
    return artifacts;
}

}  // namespace

std::filesystem::path image_generation_manifest_sidecar_path(
    const std::filesystem::path& manifest_path
) {
    std::filesystem::path sidecar = manifest_path;
    sidecar += ".sha256";
    return sidecar;
}

ImageGenerationArtifactManifestReport verify_image_generation_artifact_manifest(
    const std::filesystem::path& manifest_path,
    const ImageGenerationArtifactManifestLimits limits
) {
    ImageGenerationArtifactManifestReport report;
    report.manifest_path = manifest_path;
    report.sidecar_path = image_generation_manifest_sidecar_path(manifest_path);
    try {
        validate_limits(limits);
        const std::string contents = read_bounded_regular_file(
            manifest_path,
            limits.maximum_manifest_bytes,
            "image-generation artifact manifest",
            true
        );
        report.manifest_sha256 = rlf::core::sha256_hex(rlf::core::sha256(contents));
        verify_sidecar(report.sidecar_path, report.manifest_sha256, limits);
        if (rlf::core::sha256_hex(rlf::core::sha256_file(manifest_path)) !=
            report.manifest_sha256) {
            throw std::runtime_error("artifact manifest changed while being verified");
        }
        report.manifest_self_hash_verified = true;

        const std::vector<ParsedArtifact> parsed = parse_manifest(contents, limits);
        report.format_version = image_generation_artifact_manifest_version;
        report.schema_valid = true;
        report.artifact_set_complete =
            parsed.size() == required_image_generation_artifact_kinds.size();

        const std::filesystem::path manifest_canonical = canonical_existing_path(
            manifest_path, "artifact manifest"
        );
        const std::filesystem::path sidecar_canonical = canonical_existing_path(
            report.sidecar_path, "artifact manifest sidecar"
        );
        const std::filesystem::path base = manifest_canonical.parent_path();
        std::set<std::filesystem::path> canonical_artifacts;
        std::vector<std::filesystem::path> verified_artifact_paths;
        verified_artifact_paths.reserve(parsed.size());
        std::uint64_t total_bytes = 0U;
        for (const ParsedArtifact& parsed_artifact : parsed) {
            ImageGenerationArtifactVerification artifact;
            artifact.kind = parsed_artifact.kind;
            artifact.relative_path = parsed_artifact.relative_path;
            artifact.declared_bytes = parsed_artifact.bytes;
            artifact.declared_sha256 = parsed_artifact.sha256;

            reject_symlink_components(base, parsed_artifact.relative_path);
            const std::filesystem::path full_path = base / parsed_artifact.relative_path;
            const std::filesystem::path canonical_path = canonical_existing_path(
                full_path, "artifact"
            );
            const std::filesystem::path relative_from_base =
                canonical_path.lexically_relative(base);
            if (relative_from_base.empty() || relative_from_base.is_absolute() ||
                *relative_from_base.begin() == ".." || canonical_path == manifest_canonical ||
                canonical_path == sidecar_canonical ||
                !canonical_artifacts.insert(canonical_path).second) {
                throw std::runtime_error("artifact path escapes or aliases bundle metadata");
            }
            if (paths_are_equivalent(canonical_path, manifest_canonical) ||
                paths_are_equivalent(canonical_path, sidecar_canonical) ||
                std::ranges::any_of(
                    verified_artifact_paths,
                    [&canonical_path](const std::filesystem::path& prior) {
                        return paths_are_equivalent(canonical_path, prior);
                    }
                )) {
                throw std::runtime_error("artifact paths alias the same physical file");
            }
            verified_artifact_paths.push_back(canonical_path);

            std::error_code error;
            const auto status = std::filesystem::symlink_status(full_path, error);
            if (error || std::filesystem::is_symlink(status) ||
                !std::filesystem::is_regular_file(status)) {
                throw std::runtime_error("artifact is missing, a symlink, or not regular");
            }
            artifact.regular_file_verified = true;
            const std::uintmax_t raw_size = std::filesystem::file_size(full_path, error);
            if (error || raw_size > std::numeric_limits<std::uint64_t>::max()) {
                throw std::runtime_error("unable to determine artifact byte size");
            }
            artifact.observed_bytes = static_cast<std::uint64_t>(raw_size);
            artifact.size_verified = artifact.observed_bytes == artifact.declared_bytes;
            if (!artifact.size_verified) {
                throw std::runtime_error("artifact declared byte size mismatch");
            }
            if (artifact.observed_bytes > limits.maximum_total_artifact_bytes - total_bytes) {
                throw std::runtime_error("artifact bundle exceeds total byte limit");
            }
            total_bytes += artifact.observed_bytes;
            artifact.observed_sha256 = rlf::core::sha256_hex(
                rlf::core::sha256_file(full_path)
            );
            artifact.sha256_verified =
                artifact.observed_sha256 == artifact.declared_sha256;
            if (!artifact.sha256_verified) {
                throw std::runtime_error("artifact SHA-256 mismatch");
            }
            const auto final_status = std::filesystem::symlink_status(full_path, error);
            const std::uintmax_t final_size = std::filesystem::file_size(full_path, error);
            if (error || std::filesystem::is_symlink(final_status) ||
                !std::filesystem::is_regular_file(final_status) ||
                final_size != raw_size) {
                throw std::runtime_error("artifact changed while being verified");
            }
            if (artifact.kind == "checkpoint") {
                const auto summary = inspect_image_generation_checkpoint(
                    full_path, limits.checkpoint_limits
                );
                if (summary.file_bytes != artifact.observed_bytes ||
                    summary.file_sha256 != artifact.observed_sha256) {
                    throw std::runtime_error("checkpoint inspection did not cross-bind artifact");
                }
                report.checkpoint_summary = summary;
                report.checkpoint_verified = true;
            }
            report.artifacts.push_back(std::move(artifact));
        }
        report.artifact_integrity_verified = std::ranges::all_of(
            report.artifacts,
            [](const ImageGenerationArtifactVerification& artifact) {
                return artifact.regular_file_verified && artifact.size_verified &&
                    artifact.sha256_verified;
            }
        );
        report.bundle_integrity_verified = report.schema_valid &&
            report.manifest_self_hash_verified && report.artifact_set_complete &&
            report.artifact_integrity_verified && report.checkpoint_verified;
    } catch (const std::exception& error) {
        report.failures.emplace_back(error.what());
    }

    // SHA-256 and a self-hash sidecar are integrity checks, not signatures or
    // independent evaluations. Never elevate this local report into a SOTA claim.
    report.origin_authenticated = false;
    report.state_of_art_claim_proven = false;
    if (!report.bundle_integrity_verified && report.failures.empty()) {
        report.failures.emplace_back("image-generation artifact bundle is incomplete");
    }
    return report;
}

}  // namespace rlf::solstice
