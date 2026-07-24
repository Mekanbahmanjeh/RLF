#include "test_framework.hpp"

#include "rlf/core/sha256.hpp"
#include "rlf/solstice/image_generation_checkpoint.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

[[nodiscard]] rlf::solstice::ImageData checkpoint_image(
    const std::uint8_t red,
    const std::uint8_t green,
    const std::uint8_t blue
) {
    rlf::solstice::ImageData image;
    image.width = 16U;
    image.height = 16U;
    image.rgb.resize(16U * 16U * 3U);
    for (std::size_t offset = 0U; offset < image.rgb.size(); offset += 3U) {
        image.rgb[offset] = red;
        image.rgb[offset + 1U] = green;
        image.rgb[offset + 2U] = blue;
    }
    return image;
}

[[nodiscard]] rlf::solstice::ImageGenerationCheckpointState checkpoint_state() {
    rlf::solstice::ImageGenerationCheckpointState state;
    state.profile = rlf::solstice::ImageGenerationProfile::reference;
    state.master_seed = 123U;
    state.training_step = 1U;
    state.fabric.train(checkpoint_image(10U, 20U, 30U), "muted reference tile");
    state.cumulative_operations = state.fabric.operation_stats();
    const std::string a_hash(64U, 'a');
    const std::string b_hash(64U, 'b');
    state.completed_shards.push_back({
        "fixture-shard",
        a_hash,
        b_hash,
        "local:test-fixture",
        "CC0-1.0",
        1U,
        768U,
    });
    return state;
}

void overwrite_byte(
    const std::filesystem::path& path,
    const std::streamoff offset,
    const char value
) {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!file) throw std::runtime_error("unable to open corrupt-checkpoint fixture");
    file.seekp(offset, std::ios::beg);
    file.put(value);
    file.flush();
}

}  // namespace

RLF_TEST_CASE("Patch-quilt checkpoint is deterministic and restores generation exactly") {
    const auto directory = std::filesystem::temp_directory_path() /
        "rlf_patch_quilt_checkpoint_roundtrip";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const auto first_path = directory / "first.rlfimg";
    const auto second_path = directory / "second.rlfimg";
    const auto state = checkpoint_state();
    const auto before = state.fabric.generate({"muted reference tile", 16U, 16U, 7U});
    rlf::solstice::save_image_generation_checkpoint(first_path, state);
    const auto loaded = rlf::solstice::load_image_generation_checkpoint(first_path);
    const auto after = loaded.fabric.generate({"muted reference tile", 16U, 16U, 7U});
    RLF_CHECK(loaded.profile == state.profile);
    RLF_CHECK(loaded.architecture ==
              rlf::solstice::ImageGenerationArchitecture::patch_quilt_baseline);
    RLF_CHECK(loaded.master_seed == state.master_seed);
    RLF_CHECK(loaded.training_step == state.training_step);
    RLF_CHECK(loaded.fabric.deterministic_hash() == state.fabric.deterministic_hash());
    RLF_CHECK(after.image.rgb == before.image.rgb);
    RLF_CHECK(after.deterministic_hash == before.deterministic_hash);
    rlf::solstice::save_image_generation_checkpoint(second_path, loaded);
    RLF_CHECK(rlf::core::sha256_file(first_path) ==
              rlf::core::sha256_file(second_path));
    const auto summary = rlf::solstice::inspect_image_generation_checkpoint(first_path);
    RLF_CHECK(summary.format_version == 4U);
    RLF_CHECK(summary.source_images == 1U);
    RLF_CHECK(summary.tile_prototypes == 1U);
    RLF_CHECK(summary.completed_shards == 1U);
    RLF_CHECK(summary.deterministic_model_hash == state.fabric.deterministic_hash());
    RLF_CHECK(rlf::core::is_sha256_hex(summary.file_sha256));
    RLF_CHECK(rlf::core::is_sha256_hex(summary.payload_sha256));
    std::filesystem::remove_all(directory);
}

RLF_TEST_CASE("Patch-quilt checkpoint rejects corruption truncation and trailing bytes") {
    const auto directory = std::filesystem::temp_directory_path() /
        "rlf_patch_quilt_checkpoint_corruption";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const auto valid = directory / "valid.rlfimg";
    rlf::solstice::save_image_generation_checkpoint(valid, checkpoint_state());
    const auto expect_rejected = [](const std::filesystem::path& path) {
        bool rejected = false;
        try {
            static_cast<void>(rlf::solstice::load_image_generation_checkpoint(path));
        } catch (const std::runtime_error&) {
            rejected = true;
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        RLF_CHECK(rejected);
    };

    const auto payload_corrupt = directory / "payload-corrupt.rlfimg";
    std::filesystem::copy_file(valid, payload_corrupt);
    overwrite_byte(payload_corrupt, 80, static_cast<char>(0x7F));
    expect_rejected(payload_corrupt);

    const auto bad_version = directory / "bad-version.rlfimg";
    std::filesystem::copy_file(valid, bad_version);
    overwrite_byte(bad_version, 8, static_cast<char>(5));
    expect_rejected(bad_version);

    const auto bad_flags = directory / "bad-flags.rlfimg";
    std::filesystem::copy_file(valid, bad_flags);
    overwrite_byte(bad_flags, 56, static_cast<char>(1));
    expect_rejected(bad_flags);

    const auto truncated = directory / "truncated.rlfimg";
    std::filesystem::copy_file(valid, truncated);
    std::filesystem::resize_file(
        truncated,
        std::filesystem::file_size(truncated) - 1U
    );
    expect_rejected(truncated);

    const auto trailing = directory / "trailing.rlfimg";
    std::filesystem::copy_file(valid, trailing);
    {
        std::ofstream output(trailing, std::ios::binary | std::ios::app);
        output.put('x');
    }
    expect_rejected(trailing);

    auto limits = rlf::solstice::ImageGenerationCheckpointLimits{};
    limits.maximum_tile_prototypes = 0U;
    bool limit_rejected = false;
    try {
        static_cast<void>(rlf::solstice::load_image_generation_checkpoint(valid, limits));
    } catch (const std::runtime_error&) {
        limit_rejected = true;
    }
    RLF_CHECK(limit_rejected);
    std::filesystem::remove_all(directory);
}

RLF_TEST_CASE("Patch-quilt checkpoint resume equals uninterrupted shard training") {
    const auto directory = std::filesystem::temp_directory_path() /
        "rlf_patch_quilt_checkpoint_resume";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const auto path = directory / "resume.rlfimg";
    rlf::solstice::ImageGenerationCheckpointState interrupted;
    interrupted.fabric.train(checkpoint_image(255U, 0U, 0U), "red tile");
    interrupted.training_step = 1U;
    rlf::solstice::save_image_generation_checkpoint(path, interrupted);
    auto resumed = rlf::solstice::load_image_generation_checkpoint(path);
    resumed.fabric.train(checkpoint_image(0U, 0U, 255U), "blue tile");
    resumed.training_step = 2U;

    rlf::solstice::PatchQuiltBaseline uninterrupted;
    uninterrupted.train(checkpoint_image(255U, 0U, 0U), "red tile");
    uninterrupted.train(checkpoint_image(0U, 0U, 255U), "blue tile");
    RLF_CHECK(resumed.fabric.deterministic_hash() == uninterrupted.deterministic_hash());
    std::filesystem::remove_all(directory);
}

RLF_TEST_CASE("Patch-quilt checkpoint refuses profile mismatch and directory target") {
    auto state = checkpoint_state();
    state.profile = rlf::solstice::ImageGenerationProfile::a100_80g;
    const auto directory = std::filesystem::temp_directory_path() /
        "rlf_patch_quilt_checkpoint_target";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    bool mismatch_rejected = false;
    try {
        rlf::solstice::save_image_generation_checkpoint(
            directory / "mismatch.rlfimg",
            state
        );
    } catch (const std::invalid_argument&) {
        mismatch_rejected = true;
    }
    RLF_CHECK(mismatch_rejected);

    state.profile = rlf::solstice::ImageGenerationProfile::reference;
    bool directory_rejected = false;
    try {
        rlf::solstice::save_image_generation_checkpoint(directory, state);
    } catch (const std::invalid_argument&) {
        directory_rejected = true;
    }
    RLF_CHECK(directory_rejected);
    std::filesystem::remove_all(directory);
}

RLF_TEST_CASE("Resonant image checkpoint restores and resumes exact local learning") {
    const auto directory = std::filesystem::temp_directory_path() /
        "rlf_resonant_image_checkpoint_roundtrip";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const auto path = directory / "resonant.rlfimg";

    rlf::solstice::ImageGenerationCheckpointState state;
    state.profile = rlf::solstice::ImageGenerationProfile::reference;
    state.architecture =
        rlf::solstice::ImageGenerationArchitecture::resonant_fabric;
    const auto black = checkpoint_image(0U, 0U, 0U);
    const auto red = checkpoint_image(32U, 0U, 0U);
    const auto blue = checkpoint_image(0U, 0U, 32U);
    for (std::size_t repeat = 0U; repeat < 4U; ++repeat) {
        state.resonant_fabric.train_prompt_language_record(
            "bright crimson pigment on canvas"
        );
        state.resonant_fabric.train_prompt_language_record(
            "bright scarlet pigment on canvas"
        );
    }
    static_cast<void>(state.resonant_fabric.train({black, red, "red-shift"}));
    state.training_step = 1U;
    const std::uint64_t before_hash =
        state.resonant_fabric.deterministic_hash();

    rlf::solstice::save_image_generation_checkpoint(path, state);
    auto loaded = rlf::solstice::load_image_generation_checkpoint(path);
    RLF_CHECK(loaded.architecture ==
              rlf::solstice::ImageGenerationArchitecture::resonant_fabric);
    RLF_CHECK(loaded.resonant_fabric.deterministic_hash() == before_hash);
    RLF_CHECK(
        loaded.resonant_fabric.prompt_semantics().stats().records_seen == 8U
    );
    const auto generated = loaded.resonant_fabric.generate({
        black,
        {"red-shift"},
        false,
    });
    RLF_CHECK(generated.image.rgb == red.rgb);

    static_cast<void>(state.resonant_fabric.train({black, blue, "blue-shift"}));
    static_cast<void>(loaded.resonant_fabric.train({black, blue, "blue-shift"}));
    RLF_CHECK(loaded.resonant_fabric.deterministic_hash() ==
              state.resonant_fabric.deterministic_hash());
    const auto summary = rlf::solstice::inspect_image_generation_checkpoint(path);
    RLF_CHECK(summary.format_version == 4U);
    RLF_CHECK(summary.learned_modes == state.resonant_fabric.modes().size() / 2U);
    RLF_CHECK(summary.source_images == 0U);
    RLF_CHECK(summary.tile_prototypes == 0U);
    RLF_CHECK(summary.deterministic_model_hash == before_hash);
    std::filesystem::remove_all(directory);
}
