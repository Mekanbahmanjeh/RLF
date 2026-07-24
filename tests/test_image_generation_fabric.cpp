#include "test_framework.hpp"

#include "rlf/solstice/image_generation_fabric.hpp"
#include "rlf/solstice/image_generation_evidence.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

[[nodiscard]] rlf::solstice::ImageData solid_image(
    const std::size_t width,
    const std::size_t height,
    const std::uint8_t red,
    const std::uint8_t green,
    const std::uint8_t blue
) {
    rlf::solstice::ImageData image;
    image.width = width;
    image.height = height;
    image.rgb.resize(width * height * 3U);
    for (std::size_t offset = 0U; offset < image.rgb.size(); offset += 3U) {
        image.rgb[offset] = red;
        image.rgb[offset + 1U] = green;
        image.rgb[offset + 2U] = blue;
    }
    return image;
}

[[nodiscard]] rlf::solstice::ImageGenerationConfig tiny_config() {
    rlf::solstice::ImageGenerationConfig config;
    config.tile_size = 8U;
    config.coordinate_bins = 4U;
    config.maximum_source_images = 16U;
    config.maximum_tile_prototypes = 64U;
    config.maximum_candidates_per_cell = 16U;
    config.default_output_width = 8U;
    config.default_output_height = 8U;
    config.maximum_output_side = 64U;
    config.maximum_output_pixels = 4'096U;
    return config;
}

}  // namespace

RLF_TEST_CASE("Patch-quilt baseline retrieves text-conditioned tiles deterministically") {
    rlf::solstice::PatchQuiltBaseline fabric(tiny_config());
    fabric.train(solid_image(8U, 8U, 255U, 0U, 0U), "red square");
    fabric.train(solid_image(8U, 8U, 0U, 0U, 255U), "blue square");

    const rlf::solstice::ImageGenerationRequest request{"red square", 8U, 8U, 17U};
    const auto first = fabric.generate(request);
    const auto second = fabric.generate(request);
    RLF_CHECK(first.image.rgb == second.image.rgb);
    RLF_CHECK(first.selected_prototype_ids == second.selected_prototype_ids);
    RLF_CHECK(first.deterministic_hash == second.deterministic_hash);
    RLF_CHECK(first.mean_semantic_score == 1.0);
    for (std::size_t offset = 0U; offset < first.image.rgb.size(); offset += 3U) {
        RLF_CHECK(first.image.rgb[offset] == 255U);
        RLF_CHECK(first.image.rgb[offset + 1U] == 0U);
        RLF_CHECK(first.image.rgb[offset + 2U] == 0U);
    }
    const auto stats = fabric.operation_stats();
    RLF_CHECK(stats.training_calls == 2U);
    RLF_CHECK(stats.source_images_inserted == 2U);
    RLF_CHECK(stats.tile_prototypes_inserted == 2U);
    RLF_CHECK(stats.generation_calls == 2U);
    RLF_CHECK(stats.candidates_scored <= 2U * tiny_config().maximum_candidates_per_cell);
}

RLF_TEST_CASE("Patch-quilt baseline reconstructs stored spatial composition") {
    auto config = tiny_config();
    config.coordinate_bins = 2U;
    config.default_output_width = 16U;
    rlf::solstice::ImageData split = solid_image(16U, 8U, 0U, 0U, 0U);
    for (std::size_t y = 0U; y < split.height; ++y) {
        for (std::size_t x = 0U; x < split.width; ++x) {
            const std::size_t offset = (y * split.width + x) * 3U;
            split.rgb[offset] = x < 8U ? 255U : 0U;
            split.rgb[offset + 1U] = x < 8U ? 0U : 255U;
        }
    }
    rlf::solstice::PatchQuiltBaseline fabric(config);
    fabric.train(split, "red green split composition");
    const auto result = fabric.generate({"red green split composition", 16U, 8U, 9U});
    RLF_CHECK(result.image.rgb == split.rgb);
    RLF_CHECK(result.selected_prototype_ids.size() == 2U);
    RLF_CHECK(result.fallback_cells == 0U);
}

RLF_TEST_CASE("Patch-quilt baseline capacity rejects before changing state") {
    auto config = tiny_config();
    config.maximum_source_images = 1U;
    config.maximum_tile_prototypes = 1U;
    rlf::solstice::PatchQuiltBaseline fabric(config);
    fabric.train(solid_image(8U, 8U, 1U, 2U, 3U), "first");
    const std::uint64_t before = fabric.deterministic_hash();
    bool rejected = false;
    try {
        fabric.train(solid_image(8U, 8U, 4U, 5U, 6U), "second");
    } catch (const std::length_error&) {
        rejected = true;
    }
    RLF_CHECK(rejected);
    RLF_CHECK(fabric.deterministic_hash() == before);
    RLF_CHECK(fabric.operation_stats().source_capacity_rejections == 1U);
}

RLF_TEST_CASE("Patch-quilt baseline bounds source dimensions and caption storage") {
    auto config = tiny_config();
    config.maximum_source_side = 8U;
    config.maximum_source_pixels = 64U;
    config.maximum_caption_bytes = 4U;
    rlf::solstice::PatchQuiltBaseline fabric(config);
    const std::uint64_t empty_hash = fabric.deterministic_hash();
    bool caption_rejected = false;
    try {
        fabric.train(solid_image(8U, 8U, 1U, 2U, 3U), "oversized");
    } catch (const std::length_error&) {
        caption_rejected = true;
    }
    RLF_CHECK(caption_rejected);
    RLF_CHECK(fabric.deterministic_hash() == empty_hash);
    bool image_rejected = false;
    try {
        fabric.train(solid_image(9U, 8U, 1U, 2U, 3U), "fit");
    } catch (const std::length_error&) {
        image_rejected = true;
    }
    RLF_CHECK(image_rejected);
    RLF_CHECK(fabric.deterministic_hash() == empty_hash);
}

RLF_TEST_CASE("Patch-quilt total caption budget rejection is transactional") {
    auto config = tiny_config();
    config.maximum_caption_bytes = 5U;
    config.maximum_total_caption_bytes = 5U;
    rlf::solstice::PatchQuiltBaseline fabric(config);
    fabric.train(solid_image(8U, 8U, 1U, 2U, 3U), "abc");
    const std::uint64_t before_hash = fabric.deterministic_hash();
    const std::size_t before_sources = fabric.sources().size();
    const std::size_t before_tiles = fabric.tiles().size();
    const std::uint64_t before_images_seen = fabric.images_seen();

    RLF_CHECK_THROWS_AS(
        fabric.train(solid_image(8U, 8U, 4U, 5U, 6U), "def"),
        std::length_error
    );

    RLF_CHECK(fabric.deterministic_hash() == before_hash);
    RLF_CHECK(fabric.sources().size() == before_sources);
    RLF_CHECK(fabric.tiles().size() == before_tiles);
    RLF_CHECK(fabric.images_seen() == before_images_seen);
    RLF_CHECK(fabric.operation_stats().string_budget_rejections == 1U);
    RLF_CHECK(fabric.operation_stats().posting_budget_rejections == 0U);
}

RLF_TEST_CASE("Patch-quilt total concept budget rejection is transactional") {
    auto config = tiny_config();
    config.maximum_caption_bytes = 64U;
    config.maximum_total_caption_bytes = 64U;
    config.maximum_total_concept_bytes = 4U;
    rlf::solstice::PatchQuiltBaseline fabric(config);
    fabric.train(solid_image(8U, 8U, 1U, 2U, 3U), "aa");
    const std::uint64_t before_hash = fabric.deterministic_hash();
    const std::size_t before_sources = fabric.sources().size();
    const std::size_t before_tiles = fabric.tiles().size();
    const std::uint64_t before_images_seen = fabric.images_seen();

    RLF_CHECK_THROWS_AS(
        fabric.train(solid_image(8U, 8U, 4U, 5U, 6U), "bbb"),
        std::length_error
    );

    RLF_CHECK(fabric.deterministic_hash() == before_hash);
    RLF_CHECK(fabric.sources().size() == before_sources);
    RLF_CHECK(fabric.tiles().size() == before_tiles);
    RLF_CHECK(fabric.images_seen() == before_images_seen);
    RLF_CHECK(fabric.operation_stats().string_budget_rejections == 1U);
    RLF_CHECK(fabric.operation_stats().posting_budget_rejections == 0U);
}

RLF_TEST_CASE("Patch-quilt posting budget rejection is transactional") {
    auto config = tiny_config();
    config.maximum_posting_entries = 2U;
    rlf::solstice::PatchQuiltBaseline fabric(config);
    fabric.train(solid_image(8U, 8U, 1U, 2U, 3U), "first");
    const std::uint64_t before_hash = fabric.deterministic_hash();
    const std::size_t before_sources = fabric.sources().size();
    const std::size_t before_tiles = fabric.tiles().size();
    const std::uint64_t before_images_seen = fabric.images_seen();

    RLF_CHECK_THROWS_AS(
        fabric.train(solid_image(8U, 8U, 4U, 5U, 6U), "second"),
        std::length_error
    );

    RLF_CHECK(fabric.deterministic_hash() == before_hash);
    RLF_CHECK(fabric.sources().size() == before_sources);
    RLF_CHECK(fabric.tiles().size() == before_tiles);
    RLF_CHECK(fabric.images_seen() == before_images_seen);
    RLF_CHECK(fabric.operation_stats().string_budget_rejections == 0U);
    RLF_CHECK(fabric.operation_stats().posting_budget_rejections == 1U);
}

RLF_TEST_CASE("Patch-quilt oversized bucket sampling reaches later insertions") {
    auto config = tiny_config();
    config.maximum_candidates_per_cell = 4U;
    rlf::solstice::PatchQuiltBaseline fabric(config);
    for (std::uint8_t index = 0U; index < 12U; ++index) {
        fabric.train(
            solid_image(8U, 8U, index, static_cast<std::uint8_t>(index + 1U), 0U),
            "shared"
        );
    }

    bool selected_beyond_first_candidate_window = false;
    for (std::uint64_t seed = 0U; seed < 64U; ++seed) {
        const auto first = fabric.generate({"shared", 8U, 8U, seed});
        const auto repeated = fabric.generate({"shared", 8U, 8U, seed});
        RLF_CHECK(first.selected_prototype_ids == repeated.selected_prototype_ids);
        RLF_CHECK(first.deterministic_hash == repeated.deterministic_hash);
        RLF_CHECK(first.selected_prototype_ids.size() == 1U);
        if (first.selected_prototype_ids.front() >
            config.maximum_candidates_per_cell) {
            selected_beyond_first_candidate_window = true;
        }
    }
    RLF_CHECK(selected_beyond_first_candidate_window);
    RLF_CHECK(
        fabric.operation_stats().candidates_scored <=
        128U * config.maximum_candidates_per_cell
    );
}

RLF_TEST_CASE("Patch-quilt per-concept quota preserves later concept candidates") {
    auto config = tiny_config();
    config.maximum_candidates_per_cell = 4U;
    rlf::solstice::PatchQuiltBaseline fabric(config);
    for (std::uint8_t index = 0U; index < 12U; ++index) {
        fabric.train(solid_image(8U, 8U, index, 0U, 0U), "alpha");
    }
    fabric.train(solid_image(8U, 8U, 0U, 255U, 0U), "alpha beta");
    fabric.train(solid_image(8U, 8U, 0U, 0U, 255U), "alpha beta");

    const auto result = fabric.generate({"alpha beta", 8U, 8U, 31U});
    RLF_CHECK(result.selected_prototype_ids.size() == 1U);
    RLF_CHECK(result.selected_prototype_ids.front() > 12U);
    RLF_CHECK(result.mean_semantic_score == 1.0);
    RLF_CHECK(fabric.operation_stats().candidates_scored <=
        config.maximum_candidates_per_cell);
}

RLF_TEST_CASE("Patch-quilt concurrent immutable generation preserves exact results") {
    rlf::solstice::PatchQuiltBaseline fabric(tiny_config());
    fabric.train(solid_image(8U, 8U, 23U, 45U, 67U), "thread safe sample");
    const rlf::solstice::ImageGenerationRequest request{
        "thread safe sample",
        8U,
        8U,
        81U,
    };
    const auto expected = fabric.generate(request);
    constexpr std::size_t thread_count = 8U;
    constexpr std::size_t generations_per_thread = 32U;
    std::vector<std::uint64_t> hashes(thread_count * generations_per_thread);
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (std::size_t thread_index = 0U; thread_index < thread_count; ++thread_index) {
        threads.emplace_back([&, thread_index]() {
            for (std::size_t generation = 0U;
                 generation < generations_per_thread;
                 ++generation) {
                hashes[thread_index * generations_per_thread + generation] =
                    fabric.generate(request).deterministic_hash;
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    for (const std::uint64_t hash : hashes) {
        RLF_CHECK(hash == expected.deterministic_hash);
    }
    RLF_CHECK(
        fabric.operation_stats().generation_calls ==
        1U + thread_count * generations_per_thread
    );
}

RLF_TEST_CASE("Patch-quilt rejects arithmetic-unsafe profiles and projections") {
    auto string_config = tiny_config();
    string_config.maximum_caption_bytes = 6U;
    string_config.maximum_total_caption_bytes = 5U;
    RLF_CHECK_THROWS_AS(
        rlf::solstice::PatchQuiltBaseline(string_config),
        std::invalid_argument
    );

    auto coordinate_config = tiny_config();
    coordinate_config.maximum_source_side =
        std::numeric_limits<std::size_t>::max();
    RLF_CHECK_THROWS_AS(
        rlf::solstice::PatchQuiltBaseline(coordinate_config),
        std::invalid_argument
    );

    auto projection_config = tiny_config();
    projection_config.maximum_tile_prototypes =
        std::numeric_limits<std::size_t>::max();
    RLF_CHECK_THROWS_AS(
        rlf::solstice::project_patch_quilt_scale(
            projection_config,
            8U,
            8U
        ),
        std::overflow_error
    );
}

RLF_TEST_CASE("Patch-quilt snapshots rebuild derived retrieval exactly") {
    rlf::solstice::PatchQuiltBaseline original(tiny_config());
    original.train(solid_image(8U, 8U, 12U, 34U, 56U), "muted sample");
    const auto expected = original.generate({"muted sample", 8U, 8U, 77U});
    auto snapshot = original.snapshot();
    auto restored = rlf::solstice::PatchQuiltBaseline::from_snapshot(snapshot);
    const auto actual = restored.generate({"muted sample", 8U, 8U, 77U});
    RLF_CHECK(restored.deterministic_hash() == original.deterministic_hash());
    RLF_CHECK(actual.image.rgb == expected.image.rgb);
    RLF_CHECK(actual.deterministic_hash == expected.deterministic_hash);

    auto ownership = original.snapshot();
    ownership.tiles.front().source_id = 999U;
    bool ownership_rejected = false;
    try {
        static_cast<void>(
            rlf::solstice::PatchQuiltBaseline::from_snapshot(std::move(ownership))
        );
    } catch (const std::invalid_argument&) {
        ownership_rejected = true;
    }
    RLF_CHECK(ownership_rejected);

    auto noncanonical = original.snapshot();
    ++noncanonical.next_tile_id;
    bool noncanonical_rejected = false;
    try {
        static_cast<void>(
            rlf::solstice::PatchQuiltBaseline::from_snapshot(std::move(noncanonical))
        );
    } catch (const std::invalid_argument&) {
        noncanonical_rejected = true;
    }
    RLF_CHECK(noncanonical_rejected);

    snapshot.tiles.front().rgb.pop_back();
    bool rejected = false;
    try {
        static_cast<void>(
            rlf::solstice::PatchQuiltBaseline::from_snapshot(std::move(snapshot))
        );
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    RLF_CHECK(rejected);
}

RLF_TEST_CASE("A100 patch-quilt capacity profile is isolated and fail-closed") {
    constexpr std::uint64_t gib = 1024ULL * 1024ULL * 1024ULL;
    const auto profile = rlf::solstice::ImageGenerationProfile::a100_80g;
    const auto config = rlf::solstice::make_image_generation_profile_config(profile);
    const auto capacity = rlf::solstice::estimate_image_generation_capacity(profile);
    RLF_CHECK(rlf::solstice::to_string(profile) == "imagegen-a100-80g");
    RLF_CHECK(rlf::solstice::parse_image_generation_profile("a100-80g") == profile);
    RLF_CHECK(config.maximum_tile_prototypes == 48'000'000U);
    RLF_CHECK(config.default_output_width == 1'024U);
    RLF_CHECK(capacity.gpu_working_set_bytes == 72ULL * gib);
    RLF_CHECK(capacity.peak_vram_limit_bytes == 76ULL * gib);
    const auto projection_1024 = rlf::solstice::project_patch_quilt_scale(
        config,
        1'024U,
        1'024U
    );
    RLF_CHECK(projection_1024.tiles_per_image == 4'096U);
    RLF_CHECK(projection_1024.maximum_images_by_tile_capacity == 11'718U);
    RLF_CHECK(projection_1024.maximum_images_by_source_capacity == 4'000'000U);
    RLF_CHECK(projection_1024.maximum_simultaneous_images == 11'718U);
    const auto projection_4096 = rlf::solstice::project_patch_quilt_scale(
        config,
        4'096U,
        4'096U
    );
    RLF_CHECK(projection_4096.tiles_per_image == 65'536U);
    RLF_CHECK(projection_4096.maximum_images_by_tile_capacity == 732U);
    RLF_CHECK(projection_4096.maximum_images_by_source_capacity == 4'000'000U);
    RLF_CHECK(projection_4096.maximum_simultaneous_images == 732U);
    RLF_CHECK(rlf::solstice::image_generation_profile_config_matches(profile, config));
    auto tampered = config;
    --tampered.maximum_tile_prototypes;
    RLF_CHECK(!rlf::solstice::image_generation_profile_config_matches(
        profile,
        tampered
    ));
}

RLF_TEST_CASE("V100 image-generation profile preserves logical capacity within 30 GiB") {
    constexpr std::uint64_t gib = 1024ULL * 1024ULL * 1024ULL;
    const auto a100 = rlf::solstice::ImageGenerationProfile::a100_80g;
    const auto v100 = rlf::solstice::ImageGenerationProfile::v100_32g;
    const auto a100_config =
        rlf::solstice::make_image_generation_profile_config(a100);
    const auto v100_config =
        rlf::solstice::make_image_generation_profile_config(v100);
    const auto capacity = rlf::solstice::estimate_image_generation_capacity(v100);

    RLF_CHECK(rlf::solstice::parse_image_generation_profile(
        "imagegen-v100-32g"
    ) == v100);
    RLF_CHECK(rlf::solstice::to_string(v100) == "imagegen-v100-32g");
    RLF_CHECK(v100_config.maximum_source_images ==
              a100_config.maximum_source_images);
    RLF_CHECK(v100_config.maximum_tile_prototypes ==
              a100_config.maximum_tile_prototypes);
    RLF_CHECK(v100_config.maximum_candidates_per_cell ==
              a100_config.maximum_candidates_per_cell);
    RLF_CHECK(v100_config.default_output_width == a100_config.default_output_width);
    RLF_CHECK(capacity.gpu_working_set_bytes == 30ULL * gib);
    RLF_CHECK(capacity.peak_vram_limit_bytes == 30ULL * gib);
    RLF_CHECK(capacity.host_ram_recommended_bytes == 512ULL * gib);
    RLF_CHECK(capacity.checkpoint_ceiling_bytes == 512ULL * gib);
    RLF_CHECK(rlf::solstice::image_generation_profile_config_matches(
        v100,
        v100_config
    ));
}

RLF_TEST_CASE("A100 image-generation state-of-art gate requires external physical evidence") {
    const auto missing = rlf::solstice::evaluate_a100_image_generation_contract({});
    RLF_CHECK(!missing.state_of_art_image_generation_proven);
    RLF_CHECK(!missing.failures.empty());

    rlf::solstice::A100ImageGenerationEvidence evidence;
    evidence.checkpoint_config = rlf::solstice::make_image_generation_profile_config(
        rlf::solstice::ImageGenerationProfile::a100_80g
    );
    evidence.device_name = "NVIDIA A100-SXM4-80GB";
    evidence.device_uuid = "GPU-TEST-A100-0001";
    evidence.device_count = 1U;
    evidence.mig_disabled = true;
    evidence.backend = "cuda-persistent";
    evidence.compute_capability_major = 8U;
    evidence.total_vram_bytes = 80ULL * 1024ULL * 1024ULL * 1024ULL;
    evidence.peak_vram_bytes = 76ULL * 1024ULL * 1024ULL * 1024ULL;
    evidence.training_wall_seconds = 1'000.0;
    evidence.gpu_active_seconds = 900.0;
    evidence.source_images = 1'000'000U;
    evidence.tile_prototypes = 12'000'000U;
    evidence.checkpoint_verified = true;
    evidence.resume_reproduced = true;
    evidence.provenance_verified = true;
    evidence.license_audit_passed = true;
    evidence.exact_dedup_passed = true;
    evidence.near_dedup_passed = true;
    evidence.perceptual_dedup_passed = true;
    evidence.contamination_audit_passed = true;
    const std::string hash(64U, 'a');
    evidence.training_manifest_sha256 = hash;
    evidence.checkpoint_sha256 = hash;
    evidence.raw_gpu_trace_sha256 = hash;
    evidence.raw_generation_artifacts_sha256 = hash;
    evidence.raw_external_evaluation_sha256 = hash;
    evidence.evaluated_prompts = 30'000U;
    evidence.successful_generations = 30'000U;
    evidence.human_pairwise_judgments = 10'000U;
    evidence.external_benchmark_families = 5U;
    evidence.evaluator_independent = true;
    evidence.matched_current_best_baselines = true;
    evidence.current_best_protocol_frozen_before_run = true;
    evidence.candidate_not_worse_than_current_best_on_all_required_metrics = true;
    evidence.external_leaderboard_rank = 1U;
    evidence.independent_reproduction = true;
    evidence.test_doubles = false;
    evidence.architecture = rlf::solstice::ImageGenerationArchitecture::resonant_fabric;
    evidence.cuda_kernel_launches = 1U;
    evidence.cuda_device_bytes = 1U;
    evidence.cuda_training_operations = 1U;
    evidence.cuda_generation_operations = 1U;
    const auto structurally_complete =
        rlf::solstice::evaluate_a100_image_generation_contract(evidence);
    RLF_CHECK(structurally_complete.external_quality_evidence_complete);
    RLF_CHECK(structurally_complete.architecture_eligible);
    RLF_CHECK(structurally_complete.physical_a100_evidence_complete);
    RLF_CHECK(structurally_complete.resource_limit_passed);
    RLF_CHECK(structurally_complete.model_evidence_complete);
    RLF_CHECK(structurally_complete.data_controls_complete);
    RLF_CHECK(!structurally_complete.authenticated_artifact_verification_complete);
    RLF_CHECK(!structurally_complete.state_of_art_image_generation_proven);
    RLF_CHECK(!structurally_complete.failures.empty());

    evidence.peak_vram_bytes += 1U;
    const auto over_limit = rlf::solstice::evaluate_a100_image_generation_contract(evidence);
    RLF_CHECK(!over_limit.resource_limit_passed);
    RLF_CHECK(!over_limit.state_of_art_image_generation_proven);
}
