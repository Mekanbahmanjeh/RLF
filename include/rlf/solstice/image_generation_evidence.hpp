#pragma once

#include "rlf/solstice/image_generation_fabric.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rlf::solstice {

struct A100ImageGenerationEvidence final {
    bool test_doubles{true};
    ImageGenerationArchitecture architecture{
        ImageGenerationArchitecture::patch_quilt_baseline
    };
    ImageGenerationProfile profile{ImageGenerationProfile::a100_80g};
    ImageGenerationConfig checkpoint_config{};
    std::string device_name;
    std::string device_uuid;
    std::size_t device_count{};
    bool mig_disabled{};
    std::string backend;
    std::uint64_t cuda_kernel_launches{};
    std::uint64_t cuda_device_bytes{};
    std::uint64_t cuda_training_operations{};
    std::uint64_t cuda_generation_operations{};
    std::uint64_t cpu_fallback_operations{};
    unsigned int compute_capability_major{};
    unsigned int compute_capability_minor{};
    std::uint64_t total_vram_bytes{};
    std::uint64_t peak_vram_bytes{};
    double training_wall_seconds{};
    double gpu_active_seconds{};
    std::uint64_t source_images{};
    std::uint64_t tile_prototypes{};
    std::uint64_t source_capacity_rejections{};
    std::uint64_t tile_capacity_rejections{};
    std::uint64_t string_budget_rejections{};
    std::uint64_t posting_budget_rejections{};
    bool checkpoint_verified{};
    bool resume_reproduced{};
    bool provenance_verified{};
    bool license_audit_passed{};
    bool exact_dedup_passed{};
    bool near_dedup_passed{};
    bool perceptual_dedup_passed{};
    bool contamination_audit_passed{};
    std::string training_manifest_sha256;
    std::string checkpoint_sha256;
    std::string raw_gpu_trace_sha256;
    std::string raw_generation_artifacts_sha256;
    std::string raw_external_evaluation_sha256;
    std::uint64_t evaluated_prompts{};
    std::uint64_t successful_generations{};
    std::uint64_t human_pairwise_judgments{};
    std::size_t external_benchmark_families{};
    bool evaluator_independent{};
    bool matched_current_best_baselines{};
    bool current_best_protocol_frozen_before_run{};
    bool candidate_not_worse_than_current_best_on_all_required_metrics{};
    std::size_t external_leaderboard_rank{};
    bool independent_reproduction{};
};

struct A100ImageGenerationContract final {
    A100ImageGenerationEvidence evidence;
    bool profile_bound{};
    bool architecture_eligible{};
    bool physical_a100_evidence_complete{};
    bool resource_limit_passed{};
    bool model_evidence_complete{};
    bool data_controls_complete{};
    bool external_quality_evidence_complete{};
    bool authenticated_artifact_verification_complete{};
    bool state_of_art_image_generation_proven{};
    std::vector<std::string> failures;
};

// This structural evaluator deliberately cannot authenticate artifacts. A future
// verifier must open, hash, and cross-bind every raw artifact before a separate
// verifier-issued proof can make the final claim eligible.
[[nodiscard]] A100ImageGenerationContract evaluate_a100_image_generation_contract(
    const A100ImageGenerationEvidence& evidence
);

}  // namespace rlf::solstice
