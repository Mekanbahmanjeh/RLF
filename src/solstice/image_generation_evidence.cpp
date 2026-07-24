#include "rlf/solstice/image_generation_evidence.hpp"

#include "rlf/core/sha256.hpp"

#include <algorithm>
#include <cmath>
#include <string_view>

namespace rlf::solstice {
namespace {

[[nodiscard]] bool contains_case_sensitive(
    const std::string_view text,
    const std::string_view needle
) noexcept {
    return text.find(needle) != std::string_view::npos;
}

void require(
    const bool condition,
    const std::string_view failure,
    std::vector<std::string>& failures
) {
    if (!condition) {
        failures.emplace_back(failure);
    }
}

[[nodiscard]] bool hash_present(const std::string_view value) noexcept {
    return rlf::core::is_sha256_hex(value);
}

}  // namespace

A100ImageGenerationContract evaluate_a100_image_generation_contract(
    const A100ImageGenerationEvidence& evidence
) {
    A100ImageGenerationContract report;
    report.evidence = evidence;
    report.profile_bound = evidence.profile == ImageGenerationProfile::a100_80g &&
        image_generation_profile_config_matches(evidence.profile, evidence.checkpoint_config);
    require(report.profile_bound, "checkpoint is not bound to imagegen-a100-80g", report.failures);
    report.architecture_eligible =
        evidence.architecture == ImageGenerationArchitecture::resonant_fabric;
    require(
        report.architecture_eligible,
        "patch-quilt baseline is not eligible as the authoritative RLF image generator",
        report.failures
    );

    const bool finite_times = std::isfinite(evidence.training_wall_seconds) &&
        std::isfinite(evidence.gpu_active_seconds);
    const bool device_is_a100_80g = contains_case_sensitive(evidence.device_name, "A100") &&
        (contains_case_sensitive(evidence.device_name, "80GB") ||
         contains_case_sensitive(evidence.device_name, "80 GB"));
    const bool compute_capability_ok = evidence.compute_capability_major == 8U &&
        evidence.compute_capability_minor == 0U;
    constexpr std::uint64_t minimum_total_vram = 76ULL * 1024ULL * 1024ULL * 1024ULL;
    report.physical_a100_evidence_complete = device_is_a100_80g &&
        !evidence.device_uuid.empty() && evidence.device_count == 1U &&
        evidence.mig_disabled && evidence.backend == "cuda-persistent" &&
        evidence.cuda_kernel_launches > 0U && evidence.cuda_device_bytes > 0U &&
        evidence.cuda_training_operations > 0U &&
        evidence.cuda_generation_operations > 0U &&
        evidence.cpu_fallback_operations == 0U && compute_capability_ok &&
        evidence.total_vram_bytes >= minimum_total_vram && finite_times &&
        evidence.training_wall_seconds > 0.0 && evidence.gpu_active_seconds > 0.0 &&
        evidence.gpu_active_seconds <= evidence.training_wall_seconds &&
        hash_present(evidence.raw_gpu_trace_sha256);
    require(
        report.physical_a100_evidence_complete,
        "complete single-A100-80GB CUDA identity, timing, and raw trace are required",
        report.failures
    );

    const std::uint64_t peak_limit = estimate_image_generation_capacity(
        ImageGenerationProfile::a100_80g
    ).peak_vram_limit_bytes;
    report.resource_limit_passed = evidence.peak_vram_bytes > 0U &&
        evidence.peak_vram_bytes <= peak_limit;
    require(
        report.resource_limit_passed,
        "measured peak VRAM must be nonzero and at most 76 GiB",
        report.failures
    );

    report.model_evidence_complete = evidence.source_images > 0U &&
        evidence.tile_prototypes > 0U && evidence.source_capacity_rejections == 0U &&
        evidence.tile_capacity_rejections == 0U &&
        evidence.string_budget_rejections == 0U &&
        evidence.posting_budget_rejections == 0U && evidence.checkpoint_verified &&
        evidence.resume_reproduced && hash_present(evidence.checkpoint_sha256);
    require(
        report.model_evidence_complete,
        "verified resumable checkpoint with learned state and zero capacity rejection is required",
        report.failures
    );

    report.data_controls_complete = evidence.provenance_verified &&
        evidence.license_audit_passed && evidence.exact_dedup_passed &&
        evidence.near_dedup_passed && evidence.perceptual_dedup_passed &&
        evidence.contamination_audit_passed &&
        hash_present(evidence.training_manifest_sha256);
    require(
        report.data_controls_complete,
        "provenance, licensing, deduplication, contamination, and manifest evidence are required",
        report.failures
    );

    constexpr std::uint64_t minimum_external_prompts = 30'000U;
    constexpr std::uint64_t minimum_human_judgments = 10'000U;
    constexpr std::size_t minimum_benchmark_families = 5U;
    report.external_quality_evidence_complete =
        evidence.evaluated_prompts >= minimum_external_prompts &&
        evidence.successful_generations == evidence.evaluated_prompts &&
        evidence.human_pairwise_judgments >= minimum_human_judgments &&
        evidence.external_benchmark_families >= minimum_benchmark_families &&
        evidence.evaluator_independent && evidence.matched_current_best_baselines &&
        evidence.current_best_protocol_frozen_before_run &&
        evidence.candidate_not_worse_than_current_best_on_all_required_metrics &&
        evidence.external_leaderboard_rank == 1U &&
        hash_present(evidence.raw_generation_artifacts_sha256) &&
        hash_present(evidence.raw_external_evaluation_sha256);
    require(
        report.external_quality_evidence_complete,
        "rank-one independent external quality evidence with full raw artifacts is required",
        report.failures
    );

    // Caller-populated booleans and hash-shaped strings are structural inputs,
    // not authenticated evidence. Keep this false until a verifier recomputes
    // and cross-binds the physical, data, checkpoint, and evaluation artifacts.
    report.authenticated_artifact_verification_complete = false;
    require(
        report.authenticated_artifact_verification_complete,
        "verifier-issued artifact authentication is required; caller attestations and test doubles are ineligible",
        report.failures
    );
    report.state_of_art_image_generation_proven = report.profile_bound &&
        report.architecture_eligible &&
        report.physical_a100_evidence_complete && report.resource_limit_passed &&
        report.model_evidence_complete && report.data_controls_complete &&
        report.external_quality_evidence_complete &&
        report.authenticated_artifact_verification_complete &&
        evidence.independent_reproduction;
    require(
        evidence.independent_reproduction,
        "independent physical reproduction is required for the state-of-art claim",
        report.failures
    );
    return report;
}

}  // namespace rlf::solstice
