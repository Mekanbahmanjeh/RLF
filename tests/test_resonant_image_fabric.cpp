#include "test_framework.hpp"

#include "rlf/solstice/resonant_image_fabric.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {

using rlf::solstice::ImageData;

[[nodiscard]] ImageData solid_image(
    const std::size_t width,
    const std::size_t height,
    const std::uint8_t red,
    const std::uint8_t green,
    const std::uint8_t blue
) {
    ImageData image{
        .width = width,
        .height = height,
        .rgb = std::vector<std::uint8_t>(width * height * 3U),
    };
    for (std::size_t pixel = 0U; pixel < width * height; ++pixel) {
        image.rgb[pixel * 3U] = red;
        image.rgb[pixel * 3U + 1U] = green;
        image.rgb[pixel * 3U + 2U] = blue;
    }
    return image;
}

void set_pixel(
    ImageData& image,
    const std::size_t x,
    const std::size_t y,
    const std::uint8_t red,
    const std::uint8_t green,
    const std::uint8_t blue
) {
    const std::size_t offset = (y * image.width + x) * 3U;
    image.rgb.at(offset) = red;
    image.rgb.at(offset + 1U) = green;
    image.rgb.at(offset + 2U) = blue;
}

[[nodiscard]] rlf::solstice::ResonantImageConfig tiny_config() {
    return {
        .patch_size = 1U,
        .phase_redundancy = 4U,
        .coordinate_bins = 2U,
        .maximum_modes = 128U,
        .maximum_concept_bytes = 64U,
        .maximum_image_side = 16U,
        .maximum_image_pixels = 256U,
        .candidate_count = 4U,
        .active_count = 1U,
        .maximum_settling_cycles = 16U,
        .maximum_trace_entries = 4'096U,
        .minimum_resonance = 0.5,
        .convergence_tolerance_radians = 1.0e-5,
        .settling_relaxation = 0.75,
        .transformation_learning_rate = 0.25,
        .context_learning_rate = 0.05,
        .confidence_learning_rate = 0.05,
        .seed = 77ULL,
    };
}

struct CompositionFixture final {
    ImageData base;
    ImageData red_left;
    ImageData green_top;
    ImageData composed;
};

[[nodiscard]] CompositionFixture composition_fixture() {
    CompositionFixture fixture{
        .base = solid_image(2U, 2U, 0U, 0U, 0U),
        .red_left = solid_image(2U, 2U, 0U, 0U, 0U),
        .green_top = solid_image(2U, 2U, 0U, 0U, 0U),
        .composed = solid_image(2U, 2U, 0U, 0U, 0U),
    };
    set_pixel(fixture.red_left, 0U, 0U, 64U, 0U, 0U);
    set_pixel(fixture.red_left, 0U, 1U, 64U, 0U, 0U);
    set_pixel(fixture.green_top, 0U, 0U, 0U, 96U, 0U);
    set_pixel(fixture.green_top, 1U, 0U, 0U, 96U, 0U);
    set_pixel(fixture.composed, 0U, 0U, 64U, 96U, 0U);
    set_pixel(fixture.composed, 1U, 0U, 0U, 96U, 0U);
    set_pixel(fixture.composed, 0U, 1U, 64U, 0U, 0U);
    return fixture;
}

void train_individual_operators(
    rlf::solstice::ResonantImageFabric& fabric,
    const CompositionFixture& fixture
) {
    static_cast<void>(fabric.train({
        .source = fixture.base,
        .target = fixture.red_left,
        .semantic_label = "red-left",
    }));
    static_cast<void>(fabric.train({
        .source = fixture.base,
        .target = fixture.green_top,
        .semantic_label = "green-top",
    }));
}

}  // namespace

RLF_TEST_CASE("RLF image patch phase codec is lossless for RGB bytes") {
    rlf::solstice::ResonantImageConfig config = tiny_config();
    config.patch_size = 2U;
    config.coordinate_bins = 1U;
    rlf::solstice::ResonantImageFabric fabric(config);
    ImageData source{
        .width = 2U,
        .height = 2U,
        .rgb = {
            0U, 1U, 255U,
            17U, 63U, 129U,
            254U, 128U, 2U,
            31U, 97U, 203U,
        },
    };

    const rlf::core::PhaseVector encoded = fabric.encode_patch(source, 0U, 0U);
    const ImageData decoded = fabric.decode_patch(encoded);

    RLF_CHECK(encoded.size() == 2U * 2U * 3U * config.phase_redundancy);
    RLF_CHECK(decoded.width == source.width);
    RLF_CHECK(decoded.height == source.height);
    RLF_CHECK(decoded.rgb == source.rgb);
}

RLF_TEST_CASE("V100 resonant image profile preserves A100 learned capacity") {
    const auto a100 = rlf::solstice::make_resonant_image_profile_config(
        rlf::solstice::ImageGenerationProfile::a100_80g
    );
    const auto v100 = rlf::solstice::make_resonant_image_profile_config(
        rlf::solstice::ImageGenerationProfile::v100_32g
    );
    RLF_CHECK(v100.maximum_modes == a100.maximum_modes);
    RLF_CHECK(v100.maximum_image_pixels == a100.maximum_image_pixels);
    RLF_CHECK(v100.patch_size == a100.patch_size);
    RLF_CHECK(v100.candidate_count == a100.candidate_count);
    RLF_CHECK(rlf::solstice::resonant_image_profile_config_matches(
        rlf::solstice::ImageGenerationProfile::v100_32g,
        v100
    ));
    auto reduced = v100;
    --reduced.maximum_modes;
    RLF_CHECK(!rlf::solstice::resonant_image_profile_config_matches(
        rlf::solstice::ImageGenerationProfile::v100_32g,
        reduced
    ));
}

RLF_TEST_CASE("RLF image CUDA backend fails closed when unavailable") {
    rlf::solstice::ResonantImageFabric fabric(tiny_config());
    auto probe = rlf::frontier::make_frontier_backend(
        rlf::frontier::FrontierBackendKind::cuda
    );
    if (probe->capabilities().available) {
        fabric.set_backend(rlf::frontier::FrontierBackendKind::cuda);
        RLF_CHECK(fabric.backend_kind() ==
                  rlf::frontier::FrontierBackendKind::cuda);
    } else {
        RLF_CHECK_THROWS_AS(
            fabric.set_backend(rlf::frontier::FrontierBackendKind::cuda),
            std::runtime_error
        );
        RLF_CHECK(fabric.backend_kind() ==
                  rlf::frontier::FrontierBackendKind::optimized_cpu);
    }
}

RLF_TEST_CASE(
    "RLF image held-out composition combines separately learned operators"
) {
    const CompositionFixture fixture = composition_fixture();
    rlf::solstice::ResonantImageFabric fabric(tiny_config());

    // The combined request and combined target are never supplied to train().
    // Only each independent transformation is observed against the base canvas.
    train_individual_operators(fabric, fixture);
    const rlf::solstice::ResonantGeneratedImage generated = fabric.generate({
        .base_image = fixture.base,
        .transformations = {"red-left", "green-top"},
        .capture_trace = true,
    });
    const rlf::solstice::ResonantImageComparison comparison =
        fabric.compare(generated, fixture.composed);

    RLF_CHECK(generated.image.rgb == fixture.composed.rgb);
    RLF_CHECK(comparison.quality.mean_absolute_error == 0.0);
    RLF_CHECK(comparison.quality.mean_squared_error == 0.0);
    RLF_CHECK(std::isinf(comparison.quality.peak_signal_to_noise_db));
    RLF_CHECK(comparison.quality.exact_pixel_fraction == 1.0);
    RLF_CHECK(comparison.learned_modes == 8U);
    RLF_CHECK(comparison.generation_operations.sparse_bucket_lookups > 8ULL);
    RLF_CHECK(comparison.generation_operations.settling_cycles > 8ULL);
    RLF_CHECK(
        comparison.generation_operations.resonance_evaluations ==
        comparison.generation_operations.sparse_bucket_lookups
    );
    RLF_CHECK(!generated.trace.empty());
}

RLF_TEST_CASE("RLF image open-vocabulary retrieval handles unseen prompt wording") {
    const CompositionFixture fixture = composition_fixture();
    rlf::solstice::ResonantImageFabric fabric(tiny_config());
    static_cast<void>(fabric.train({
        .source = fixture.base,
        .target = fixture.red_left,
        .semantic_label = "turn the left region red",
    }));
    fabric.reset_operation_stats();

    const auto generated = fabric.generate({
        .base_image = fixture.base,
        .transformations = {"please make the left region red"},
        .capture_trace = true,
    });

    RLF_CHECK(generated.image.rgb == fixture.red_left.rgb);
    RLF_CHECK(generated.operation_delta.semantic_bucket_lookups > 0ULL);
    RLF_CHECK(generated.operation_delta.semantic_candidates_scored > 0ULL);
    RLF_CHECK(generated.operation_delta.semantic_matches > 0ULL);
    RLF_CHECK(!generated.selected_mode_ids.empty());

    auto restored = rlf::solstice::ResonantImageFabric::from_snapshot(
        fabric.snapshot()
    );
    const auto repeated = restored.generate({
        fixture.base,
        {"please make the left region red"},
        false,
    });
    RLF_CHECK(repeated.image.rgb == generated.image.rgb);
    RLF_CHECK(repeated.selected_mode_ids == generated.selected_mode_ids);
}

RLF_TEST_CASE("RLF image grounds corpus-learned prompt synonyms") {
    const CompositionFixture fixture = composition_fixture();
    rlf::solstice::ResonantImageConfig config = tiny_config();
    config.prompt_semantics.bucket_bits = 6U;
    rlf::solstice::ResonantImageFabric fabric(config);
    for (std::size_t repeat = 0U; repeat < 8U; ++repeat) {
        fabric.train_prompt_language_record(
            "bright crimson pigment on canvas"
        );
        fabric.train_prompt_language_record(
            "bright scarlet pigment on canvas"
        );
    }
    static_cast<void>(fabric.train({
        .source = fixture.base,
        .target = fixture.red_left,
        .semantic_label = "apply crimson",
    }));
    fabric.reset_operation_stats();

    const auto generated = fabric.generate({
        .base_image = fixture.base,
        .transformations = {"apply scarlet"},
        .capture_trace = false,
    });

    RLF_CHECK(generated.image.rgb == fixture.red_left.rgb);
    RLF_CHECK(generated.operation_delta.semantic_matches > 0ULL);
    auto restored = rlf::solstice::ResonantImageFabric::from_snapshot(
        fabric.snapshot()
    );
    RLF_CHECK(restored.deterministic_hash() == fabric.deterministic_hash());
    const auto repeated = restored.generate({
        fixture.base, {"apply scarlet"}, false,
    });
    RLF_CHECK(repeated.image.rgb == generated.image.rgb);
}

RLF_TEST_CASE("RLF image natural prompt parser exposes unseen operator composition") {
    const CompositionFixture fixture = composition_fixture();
    rlf::solstice::ResonantImageFabric fabric(tiny_config());
    train_individual_operators(fabric, fixture);
    const auto operations = rlf::solstice::parse_resonant_image_prompt(
        "please apply red left and then make green top"
    );
    RLF_CHECK(operations.size() == 2U);
    const auto generated = fabric.generate({
        fixture.base,
        operations,
        false,
    });
    RLF_CHECK(generated.image.rgb == fixture.composed.rgb);
    RLF_CHECK(generated.operation_delta.semantic_matches > 0ULL);
}

RLF_TEST_CASE("RLF image retrieval is spatially sparse within many modes") {
    const CompositionFixture fixture = composition_fixture();
    rlf::solstice::ResonantImageFabric fabric(tiny_config());
    train_individual_operators(fabric, fixture);
    for (std::size_t label_index = 0U; label_index < 20U; ++label_index) {
        static_cast<void>(fabric.train({
            .source = fixture.base,
            .target = fixture.base,
            .semantic_label = "identity-" + std::to_string(label_index),
        }));
    }
    RLF_CHECK(fabric.modes().size() == 88U);
    fabric.reset_operation_stats();

    const rlf::solstice::ResonantGeneratedImage generated = fabric.generate({
        .base_image = fixture.base,
        .transformations = {"red-left"},
        .capture_trace = false,
    });

    RLF_CHECK(generated.image.rgb == fixture.red_left.rgb);
    RLF_CHECK(
        generated.operation_delta.resonance_evaluations ==
        generated.operation_delta.sparse_bucket_lookups
    );
    RLF_CHECK(
        generated.operation_delta.resonance_evaluations <
        static_cast<std::uint64_t>(fabric.modes().size())
    );
}

RLF_TEST_CASE("RLF image local circular update changes a learned operator") {
    rlf::solstice::ResonantImageFabric fabric(tiny_config());
    const ImageData base = solid_image(1U, 1U, 0U, 0U, 0U);
    const ImageData first_target = solid_image(1U, 1U, 64U, 0U, 0U);
    const ImageData second_target = solid_image(1U, 1U, 96U, 0U, 0U);
    const auto first = fabric.train({base, first_target, "red-shift"});
    const std::uint64_t before_hash = fabric.deterministic_hash();
    const auto second = fabric.train({base, second_target, "red-shift"});
    const std::uint64_t after_hash = fabric.deterministic_hash();
    const auto generated = fabric.generate({base, {"red-shift"}, false});

    RLF_CHECK(first.modes_created == 1U);
    RLF_CHECK(second.modes_updated == 1U);
    RLF_CHECK(second.operation_delta.local_mode_updates == 1ULL);
    RLF_CHECK(before_hash != after_hash);
    RLF_CHECK(generated.image.rgb[0U] > 64U);
    RLF_CHECK(generated.image.rgb[0U] < 96U);
    RLF_CHECK(generated.image.rgb[1U] == 0U);
    RLF_CHECK(generated.image.rgb[2U] == 0U);
}

RLF_TEST_CASE("RLF image runs repeat deterministically") {
    const CompositionFixture fixture = composition_fixture();
    rlf::solstice::ResonantImageFabric first(tiny_config());
    rlf::solstice::ResonantImageFabric second(tiny_config());
    train_individual_operators(first, fixture);
    train_individual_operators(second, fixture);
    const std::uint64_t learned_hash = first.deterministic_hash();

    const auto first_result = first.generate({
        fixture.base,
        {"red-left", "green-top"},
        true,
    });
    const auto second_result = second.generate({
        fixture.base,
        {"red-left", "green-top"},
        true,
    });
    const auto repeated_result = first.generate({
        fixture.base,
        {"red-left", "green-top"},
        true,
    });

    RLF_CHECK(first_result.image.rgb == second_result.image.rgb);
    RLF_CHECK(
        first_result.deterministic_hash == second_result.deterministic_hash
    );
    RLF_CHECK(first.deterministic_hash() == second.deterministic_hash());
    RLF_CHECK(first.deterministic_hash() == learned_hash);
    RLF_CHECK(
        repeated_result.deterministic_hash == first_result.deterministic_hash
    );
    RLF_CHECK(
        first_result.operation_delta.resonance_evaluations ==
        second_result.operation_delta.resonance_evaluations
    );
}

RLF_TEST_CASE("RLF image mode-capacity rejection has no partial mutation") {
    rlf::solstice::ResonantImageConfig config = tiny_config();
    config.maximum_modes = 3U;
    rlf::solstice::ResonantImageFabric fabric(config);
    const CompositionFixture fixture = composition_fixture();

    RLF_CHECK_THROWS_AS(
        fabric.train({fixture.base, fixture.red_left, "red-left"}),
        std::runtime_error
    );
    RLF_CHECK(fabric.modes().empty());
    RLF_CHECK(fabric.operation_stats().training_examples == 0ULL);
    RLF_CHECK(fabric.operation_stats().modes_created == 0ULL);
}

RLF_TEST_CASE("RLF image unknown operator leaves canvas unchanged") {
    rlf::solstice::ResonantImageFabric fabric(tiny_config());
    const ImageData base = solid_image(2U, 2U, 7U, 11U, 13U);
    const auto generated = fabric.generate({base, {"unlearned"}, true});

    RLF_CHECK(generated.image.rgb == base.rgb);
    RLF_CHECK(
        generated.operation_delta.unresolved_patch_transformations == 4ULL
    );
    RLF_CHECK(generated.operation_delta.resonance_evaluations == 0ULL);
    RLF_CHECK(generated.trace.size() == 4U);
}

RLF_TEST_CASE("RLF image snapshot restores exact learned behavior and resume") {
    const CompositionFixture fixture = composition_fixture();
    rlf::solstice::ResonantImageFabric original(tiny_config());
    train_individual_operators(original, fixture);
    const std::uint64_t learned_hash = original.deterministic_hash();
    auto restored = rlf::solstice::ResonantImageFabric::from_snapshot(
        original.snapshot()
    );

    RLF_CHECK(restored.deterministic_hash() == learned_hash);
    const auto before = restored.generate({
        fixture.base,
        {"red-left", "green-top"},
        false,
    });
    RLF_CHECK(before.image.rgb == fixture.composed.rgb);

    const ImageData blue = solid_image(2U, 2U, 0U, 0U, 32U);
    static_cast<void>(original.train({fixture.base, blue, "blue"}));
    static_cast<void>(restored.train({fixture.base, blue, "blue"}));
    RLF_CHECK(restored.deterministic_hash() == original.deterministic_hash());
}

RLF_TEST_CASE("RLF image snapshot rejects duplicate cells and invalid IDs") {
    const CompositionFixture fixture = composition_fixture();
    rlf::solstice::ResonantImageFabric fabric(tiny_config());
    train_individual_operators(fabric, fixture);

    auto duplicate = fabric.snapshot();
    duplicate.modes.push_back(duplicate.modes.front());
    duplicate.modes.back().resonant_mode.id = duplicate.next_mode_id++;
    RLF_CHECK_THROWS_AS(
        rlf::solstice::ResonantImageFabric::from_snapshot(
            std::move(duplicate)
        ),
        std::invalid_argument
    );

    auto invalid_id = fabric.snapshot();
    invalid_id.next_mode_id = invalid_id.modes.back().resonant_mode.id;
    RLF_CHECK_THROWS_AS(
        rlf::solstice::ResonantImageFabric::from_snapshot(
            std::move(invalid_id)
        ),
        std::invalid_argument
    );
}
