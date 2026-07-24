#pragma once

#include "rlf/solstice/image_generation_checkpoint.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rlf::solstice {

inline constexpr std::uint32_t image_generation_artifact_manifest_version = 1U;
inline constexpr std::array<std::string_view, 16U>
    required_image_generation_artifact_kinds{
        "checkpoint",
        "ledger",
        "source_manifest",
        "data_audit",
        "license_report",
        "exact_dedup_report",
        "near_dedup_report",
        "perceptual_dedup_report",
        "contamination_report",
        "training_telemetry",
        "resource_summary",
        "raw_gpu_trace",
        "environment",
        "checkpoint_inspection",
        "readiness_report",
        "resume_equivalence_report",
    };

// Canonical manifest schema (UTF-8/ASCII fields, LF line endings, final LF):
//
//   RLF_IMAGEGEN_ARTIFACT_MANIFEST<TAB>1
//   <kind><TAB><relative-path><TAB><decimal-bytes><TAB><lowercase-sha256>
//   ... exactly one row for each required kind, in the order above ...
//
// Paths must be distinct, canonical relative paths beneath the manifest's
// directory. Absolute paths, '.', '..', empty components, symlinks, directory
// artifacts, and aliases to the manifest or sidecar are rejected. Every
// artifact must be nonempty. The adjacent sidecar is named by appending
// ".sha256" to the manifest filename and has this exact canonical schema:
//
//   RLF_IMAGEGEN_MANIFEST_SHA256<TAB>1
//   sha256<TAB><lowercase-sha256-of-the-complete-manifest-file>
//
// The sidecar detects accidental or uncoordinated manifest mutation. It is not
// a signature and therefore does not authenticate the bundle's origin.

struct ImageGenerationArtifactManifestLimits final {
    std::uint64_t maximum_manifest_bytes{64ULL * 1024ULL};
    std::uint64_t maximum_sidecar_bytes{256ULL};
    std::size_t maximum_relative_path_bytes{1'024U};
    std::uint64_t maximum_artifact_bytes{
        513ULL * 1024ULL * 1024ULL * 1024ULL
    };
    std::uint64_t maximum_total_artifact_bytes{
        1'024ULL * 1024ULL * 1024ULL * 1024ULL
    };
    ImageGenerationCheckpointLimits checkpoint_limits{};
};

struct ImageGenerationArtifactVerification final {
    std::string kind;
    std::filesystem::path relative_path;
    std::uint64_t declared_bytes{};
    std::uint64_t observed_bytes{};
    std::string declared_sha256;
    std::string observed_sha256;
    bool regular_file_verified{};
    bool size_verified{};
    bool sha256_verified{};
};

struct ImageGenerationArtifactManifestReport final {
    std::uint32_t format_version{};
    std::filesystem::path manifest_path;
    std::filesystem::path sidecar_path;
    std::string manifest_sha256;
    bool schema_valid{};
    bool manifest_self_hash_verified{};
    bool artifact_set_complete{};
    bool artifact_integrity_verified{};
    bool checkpoint_verified{};
    bool bundle_integrity_verified{};

    // A hash sidecar supplies integrity, not identity or independent external
    // attestation. These values deliberately remain false in this verifier.
    bool origin_authenticated{};
    bool state_of_art_claim_proven{};

    std::vector<ImageGenerationArtifactVerification> artifacts;
    std::optional<ImageGenerationCheckpointSummary> checkpoint_summary;
    std::vector<std::string> failures;
};

[[nodiscard]] std::filesystem::path image_generation_manifest_sidecar_path(
    const std::filesystem::path& manifest_path
);

// Opens and hashes every referenced artifact. Malformed or incomplete bundles
// return a report with failures and all dependent gates false; caller-provided
// booleans are neither accepted nor trusted.
[[nodiscard]] ImageGenerationArtifactManifestReport
verify_image_generation_artifact_manifest(
    const std::filesystem::path& manifest_path,
    ImageGenerationArtifactManifestLimits limits = {}
);

}  // namespace rlf::solstice
