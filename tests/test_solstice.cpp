#include "test_framework.hpp"

#include "rlf/frontier/frontier_backend.hpp"
#include "rlf/solstice/checkpoint.hpp"
#include "rlf/solstice/language_fabric.hpp"
#include "rlf/solstice/profile.hpp"
#include "rlf/solstice/solstice_model.hpp"
#include "rlf/solstice/tokenizer.hpp"
#include "rlf/solstice/tool_protocol.hpp"
#include "rlf/solstice/vision_fabric.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

[[nodiscard]] rlf::solstice::ImageData solid_image(
    const std::uint8_t red,
    const std::uint8_t green,
    const std::uint8_t blue
) {
    rlf::solstice::ImageData image;
    image.width = 32U;
    image.height = 32U;
    image.rgb.resize(image.width * image.height * 3U);
    for (std::size_t pixel = 0U; pixel < image.width * image.height; ++pixel) {
        image.rgb[pixel * 3U] = red;
        image.rgb[pixel * 3U + 1U] = green;
        image.rgb[pixel * 3U + 2U] = blue;
    }
    return image;
}

[[nodiscard]] std::filesystem::path temporary_path(const std::string& name) {
    return std::filesystem::temp_directory_path() /
        ("rlf_solstice_" + name);
}

[[nodiscard]] std::vector<rlf::solstice::ImageData> moving_square_frames(
    const bool moves_right
) {
    std::vector<rlf::solstice::ImageData> frames;
    for (std::size_t frame_index = 0U; frame_index < 4U; ++frame_index) {
        rlf::solstice::ImageData image;
        image.width = 32U;
        image.height = 24U;
        image.rgb.assign(image.width * image.height * 3U, 0U);
        const std::size_t step = moves_right ? frame_index : 3U - frame_index;
        const std::size_t start_x = 2U + step * 5U;
        for (std::size_t y = 8U; y < 14U; ++y) {
            for (std::size_t x = start_x; x < start_x + 6U; ++x) {
                const std::size_t offset = (y * image.width + x) * 3U;
                image.rgb[offset] = 230U;
                image.rgb[offset + 1U] = 40U;
                image.rgb[offset + 2U] = 20U;
            }
        }
        frames.push_back(std::move(image));
    }
    return frames;
}

}  // namespace

RLF_TEST_CASE("Solstice tokenizer preserves arbitrary UTF-8 bytes") {
    rlf::solstice::SolsticeTokenizer tokenizer;
    const std::string corpus = "hello hello world world image image tool tool";
    tokenizer.train(corpus);
    const std::string input = "hello, world! \xE2\x98\x83";
    const std::vector<rlf::solstice::TokenId> tokens = tokenizer.encode(input);
    RLF_CHECK(!tokens.empty());
    RLF_CHECK(tokenizer.decode(tokens) == input);
    RLF_CHECK(tokenizer.vocabulary_size() > 266U);
}

RLF_TEST_CASE("Solstice hierarchical language generates trained dialogue") {
    rlf::solstice::SolsticeModel model;
    model.bootstrap();
    const rlf::solstice::SolsticeResponse response = model.respond(
        "What can you do?", nullptr, nullptr,
        rlf::solstice::GenerationSettings{96U, 8U, 0.8, true, 7U}
    );
    RLF_CHECK(response.text.find("visual patterns") != std::string::npos);
    RLF_CHECK(model.language().contexts().size() > 100U);
    RLF_CHECK(model.language().episodes().size() >= 8U);
}

RLF_TEST_CASE("Indexed language outcomes preserve exact training and prediction") {
    rlf::solstice::HierarchicalLanguageConfig config;
    config.context_orders = {0U};
    config.maximum_contexts = 4U;
    config.prediction_candidate_limit = 128U;
    rlf::solstice::HierarchicalLanguageFabric linear(config);
    rlf::solstice::HierarchicalLanguageFabric indexed(config);
    linear.set_indexed_outcome_updates(false);
    indexed.set_indexed_outcome_updates(true);
    for (rlf::solstice::TokenId token = 1U; token <= 80U; ++token) {
        const std::array<rlf::solstice::TokenId, 2U> sequence{999U, token};
        linear.train_token_sequence(sequence);
        indexed.train_token_sequence(sequence);
        RLF_CHECK(linear.deterministic_hash() == indexed.deterministic_hash());
    }
    for (rlf::solstice::TokenId token = 80U; token >= 1U; --token) {
        const std::array<rlf::solstice::TokenId, 2U> sequence{999U, token};
        linear.train_token_sequence(sequence);
        indexed.train_token_sequence(sequence);
        RLF_CHECK(linear.deterministic_hash() == indexed.deterministic_hash());
    }
    const auto expected = linear.predict_next({});
    const auto actual = indexed.predict_next({});
    RLF_CHECK(expected.deepest_context_order == actual.deepest_context_order);
    RLF_CHECK(expected.uncertainty == actual.uncertainty);
    RLF_CHECK(expected.candidates.size() == actual.candidates.size());
    for (std::size_t position = 0U; position < expected.candidates.size(); ++position) {
        RLF_CHECK(expected.candidates[position].token == actual.candidates[position].token);
        RLF_CHECK(expected.candidates[position].probability ==
                  actual.candidates[position].probability);
        RLF_CHECK(expected.candidates[position].score ==
                  actual.candidates[position].score);
    }

    auto resumed = rlf::solstice::HierarchicalLanguageFabric::from_snapshot(
        indexed.snapshot()
    );
    resumed.set_indexed_outcome_updates(true);
    const std::array<rlf::solstice::TokenId, 2U> resumed_sequence{999U, 77U};
    indexed.train_token_sequence(resumed_sequence);
    resumed.train_token_sequence(resumed_sequence);
    RLF_CHECK(indexed.deterministic_hash() == resumed.deterministic_hash());

    auto duplicate_snapshot = indexed.snapshot();
    duplicate_snapshot.contexts.front().outcomes.push_back({77U, 9U});
    auto duplicate_linear =
        rlf::solstice::HierarchicalLanguageFabric::from_snapshot(
            duplicate_snapshot
        );
    auto duplicate_indexed =
        rlf::solstice::HierarchicalLanguageFabric::from_snapshot(
            std::move(duplicate_snapshot)
        );
    duplicate_linear.set_indexed_outcome_updates(false);
    duplicate_indexed.set_indexed_outcome_updates(true);
    duplicate_linear.train_token_sequence(resumed_sequence);
    duplicate_indexed.train_token_sequence(resumed_sequence);
    RLF_CHECK(duplicate_linear.deterministic_hash() ==
              duplicate_indexed.deterministic_hash());

    const auto linear_stats = linear.training_operation_stats();
    const auto indexed_stats = indexed.training_operation_stats();
    RLF_CHECK(linear_stats.outcome_update_lookups == 160U);
    RLF_CHECK(linear_stats.linear_outcome_comparisons > 0U);
    RLF_CHECK(linear_stats.indexed_outcome_lookups == 0U);
    RLF_CHECK(indexed_stats.indexed_outcome_lookups > 0U);
    RLF_CHECK(indexed_stats.outcome_index_builds > 0U);
    RLF_CHECK(indexed_stats.outcome_index_entries_built >= 32U);
    RLF_CHECK(indexed_stats.outcome_index_incremental_inserts > 0U);
}

RLF_TEST_CASE("Fused dialogue tokenization preserves exact language learning") {
    rlf::solstice::SolsticeTokenizer tokenizer;
    tokenizer.train(
        "alpha prompt beta response grounded context second dialogue"
    );
    rlf::solstice::HierarchicalLanguageFabric redundant;
    rlf::solstice::HierarchicalLanguageFabric fused;
    redundant.set_fused_dialogue_encoding(false);
    fused.set_fused_dialogue_encoding(true);
    redundant.train_dialogue(
        tokenizer, "alpha prompt", "beta response", "grounded context"
    );
    fused.train_dialogue(
        tokenizer, "alpha prompt", "beta response", "grounded context"
    );
    RLF_CHECK(redundant.deterministic_hash() == fused.deterministic_hash());
    redundant.train_dialogue(tokenizer, "second prompt", "second response");
    fused.train_dialogue(tokenizer, "second prompt", "second response");
    RLF_CHECK(redundant.deterministic_hash() == fused.deterministic_hash());
    const auto redundant_stats = redundant.training_operation_stats();
    const auto fused_stats = fused.training_operation_stats();
    RLF_CHECK(redundant_stats.dialogue_tokenizer_encode_calls == 9U);
    RLF_CHECK(fused_stats.dialogue_tokenizer_encode_calls == 5U);
    RLF_CHECK(fused_stats.redundant_dialogue_encode_calls_avoided == 4U);
}

RLF_TEST_CASE("Language training reports episode and context capacity saturation") {
    rlf::solstice::HierarchicalLanguageConfig config;
    config.context_orders = {0U, 1U};
    config.maximum_contexts = 1U;
    config.maximum_episodes = 1U;
    rlf::solstice::SolsticeTokenizer tokenizer;
    rlf::solstice::HierarchicalLanguageFabric fabric(config);
    fabric.train_dialogue(tokenizer, "first unique prompt", "first response");
    fabric.train_dialogue(tokenizer, "second unique prompt", "second response");
    const auto stats = fabric.training_operation_stats();
    RLF_CHECK(stats.dialogue_training_calls == 2U);
    RLF_CHECK(stats.episode_insert_attempts == 2U);
    RLF_CHECK(stats.episode_inserts == 1U);
    RLF_CHECK(stats.episode_capacity_skips == 1U);
    RLF_CHECK(stats.context_insert_attempts > stats.context_inserts);
    RLF_CHECK(stats.context_inserts == 1U);
    RLF_CHECK(stats.context_capacity_skips > 0U);
}

RLF_TEST_CASE("Bounded language replacement continues learning without capacity skips") {
    rlf::solstice::HierarchicalLanguageConfig config;
    config.context_orders = {0U, 1U};
    config.maximum_contexts = 3U;
    config.maximum_episodes = 1U;
    rlf::solstice::SolsticeTokenizer tokenizer;
    rlf::solstice::HierarchicalLanguageFabric first(config);
    rlf::solstice::HierarchicalLanguageFabric second(config);
    first.set_bounded_capacity_replacement(true);
    second.set_bounded_capacity_replacement(true);
    for (std::size_t index = 0U; index < 12U; ++index) {
        const std::string prompt = "unique prompt " + std::to_string(index);
        const std::string response = "unique response " + std::to_string(index);
        first.train_dialogue(tokenizer, prompt, response);
        second.train_dialogue(tokenizer, prompt, response);
    }
    const auto stats = first.training_operation_stats();
    RLF_CHECK(first.contexts().size() == config.maximum_contexts);
    RLF_CHECK(first.episodes().size() == config.maximum_episodes);
    RLF_CHECK(stats.context_replacements > 0U);
    RLF_CHECK(stats.episode_replacements > 0U);
    RLF_CHECK(stats.context_capacity_skips == 0U);
    RLF_CHECK(stats.episode_capacity_skips == 0U);
    RLF_CHECK(first.deterministic_hash() == second.deterministic_hash());
}

RLF_TEST_CASE("Solstice visual patch fabric learns caption associations") {
    rlf::solstice::VisualPatchFabric vision;
    const rlf::solstice::ImageData red = solid_image(240U, 20U, 20U);
    const rlf::solstice::ImageData blue = solid_image(20U, 20U, 240U);
    vision.train(red, "a red warning panel");
    vision.train(blue, "a blue information panel");
    const rlf::solstice::VisionAnalysis analysis = vision.analyze(red);
    RLF_CHECK(analysis.description == "a red warning panel");
    RLF_CHECK(!analysis.regions.empty());
    RLF_CHECK(analysis.confidence > 0.5);
}

RLF_TEST_CASE("Fused visual training analysis matches the separate reference path") {
    const rlf::solstice::ImageData red = solid_image(240U, 20U, 20U);
    const rlf::solstice::ImageData blue = solid_image(20U, 20U, 240U);
    rlf::solstice::VisualPatchFabric reference;
    rlf::solstice::VisualPatchFabric fused;
    reference.train(red, "a red warning panel");
    fused.train(red, "a red warning panel");

    reference.train(blue, "a blue information panel");
    const rlf::solstice::VisionAnalysis expected = reference.analyze(blue);
    const rlf::solstice::VisionAnalysis actual = fused.train_and_analyze(
        blue, "a blue information panel"
    );

    RLF_CHECK(fused.deterministic_hash() == reference.deterministic_hash());
    RLF_CHECK(actual.width == expected.width);
    RLF_CHECK(actual.height == expected.height);
    RLF_CHECK(actual.description == expected.description);
    RLF_CHECK(actual.concepts == expected.concepts);
    RLF_CHECK(actual.confidence == expected.confidence);
    RLF_CHECK(actual.nearest_example_id == expected.nearest_example_id);
    RLF_CHECK(actual.regions.size() == expected.regions.size());
    for (std::size_t index = 0U; index < actual.regions.size(); ++index) {
        const auto& left = actual.regions[index];
        const auto& right = expected.regions[index];
        RLF_CHECK(left.mode_id == right.mode_id);
        RLF_CHECK(left.x == right.x);
        RLF_CHECK(left.y == right.y);
        RLF_CHECK(left.width == right.width);
        RLF_CHECK(left.height == right.height);
        RLF_CHECK(left.patch_count == right.patch_count);
        RLF_CHECK(left.concept_name == right.concept_name);
        RLF_CHECK(left.confidence == right.confidence);
    }
}

RLF_TEST_CASE("Fused Solstice image training matches reference grounding state") {
    rlf::solstice::SolsticeModel fused;
    rlf::solstice::SolsticeModel reference;
    const rlf::solstice::ImageData image = solid_image(91U, 37U, 203U);
    fused.train_image(image, "a violet status panel");
    reference.train_image_reference(image, "a violet status panel");
    RLF_CHECK(fused.deterministic_hash() == reference.deterministic_hash());
    RLF_CHECK(
        fused.backend_operation_stats().local_update_calls ==
        reference.backend_operation_stats().local_update_calls
    );
}

RLF_TEST_CASE("Solstice video fabric learns retrieves and renders a motion prototype") {
    rlf::solstice::VideoFabricConfig config;
    config.maximum_sequences = 8U;
    config.output_width = 32U;
    config.output_height = 24U;
    rlf::solstice::VideoPrototypeFabric video(config);
    const auto right = moving_square_frames(true);
    const auto left = moving_square_frames(false);
    const std::uint64_t right_id = video.train(
        "right-clip", "a red square moves steadily right", 12.0, right
    );
    static_cast<void>(video.train(
        "left-clip", "a red square moves steadily left", 12.0, left
    ));
    const rlf::solstice::VideoGeneration generated = video.generate(
        "a red square moves steadily right", 5U
    );
    RLF_CHECK(generated.prototype_id == right_id);
    RLF_CHECK(generated.frames.size() == 5U);
    RLF_CHECK(generated.prompt_similarity > 0.99);
    RLF_CHECK(rlf::solstice::VideoPrototypeFabric::describe(generated.frames).velocity_x > 0.0);
}

RLF_TEST_CASE("Solstice format six video extension roundtrips learned prototypes") {
    const std::filesystem::path path = temporary_path("video-roundtrip.rlfsp");
    std::filesystem::remove(path);
    rlf::solstice::SolsticeConfig config;
    config.video.maximum_sequences = 8U;
    config.video.output_width = 32U;
    config.video.output_height = 24U;
    rlf::solstice::SolsticeModel model(config, 77U);
    const auto frames = moving_square_frames(true);
    static_cast<void>(model.train_video_sequence(
        "video-roundtrip", "a red block travels right", 10.0, frames
    ));
    const std::uint64_t expected_hash = model.deterministic_hash();
    rlf::solstice::save_solstice_checkpoint(path, model);
    RLF_CHECK(rlf::solstice::inspect_solstice_checkpoint(path).format_version == 6U);
    const rlf::solstice::SolsticeModel restored =
        rlf::solstice::load_solstice_checkpoint(path);
    RLF_CHECK(restored.deterministic_hash() == expected_hash);
    RLF_CHECK(restored.video().prototypes().size() == 1U);
    RLF_CHECK(restored.stats().video_frames_seen == 4U);
    RLF_CHECK(restored.generate_video("a red block travels right", 3U).frames.size() == 3U);
    std::filesystem::remove(path);
}

RLF_TEST_CASE("Solstice tool protocol validates and executes safe builtins") {
    rlf::solstice::ToolRuntime tools;
    tools.register_safe_builtins();
    const rlf::solstice::ToolCall call =
        rlf::solstice::ToolRuntime::parse_call(
            "{\"name\":\"calculator\",\"arguments\":{\"expression\":\"(4+5)*3\"}}"
        );
    const rlf::solstice::ToolResult result = tools.execute(call);
    RLF_CHECK(result.success);
    RLF_CHECK(result.output == "27");
    const std::string serialized = rlf::solstice::ToolRuntime::serialize_call(call);
    RLF_CHECK(rlf::solstice::ToolRuntime::parse_call(serialized).name == "calculator");
}

RLF_TEST_CASE("Indexed tool keyword updates preserve exact learned routes") {
    rlf::solstice::ToolRouterConfig config;
    config.maximum_tools = 4U;
    config.maximum_keywords_per_tool = 256U;
    rlf::solstice::ToolRouter linear(config);
    rlf::solstice::ToolRouter indexed(config);
    linear.set_indexed_keyword_updates(false);
    indexed.set_indexed_keyword_updates(true);
    for (std::size_t item = 0U; item < 64U; ++item) {
        const std::string request =
            "route stable unique_" + std::to_string(item);
        linear.train(request, "example_tool");
        indexed.train(request, "example_tool");
        RLF_CHECK(linear.deterministic_hash() == indexed.deterministic_hash());
    }

    auto resumed = rlf::solstice::ToolRouter::from_snapshot(indexed.snapshot());
    resumed.set_indexed_keyword_updates(true);
    indexed.train("route stable unique_63", "example_tool");
    resumed.train("route stable unique_63", "example_tool");
    RLF_CHECK(indexed.deterministic_hash() == resumed.deterministic_hash());

    const auto linear_stats = linear.training_operation_stats();
    const auto indexed_stats = indexed.training_operation_stats();
    RLF_CHECK(linear_stats.linear_keyword_comparisons > 0U);
    RLF_CHECK(linear_stats.indexed_keyword_lookups == 0U);
    RLF_CHECK(indexed_stats.linear_keyword_comparisons == 0U);
    RLF_CHECK(indexed_stats.indexed_keyword_lookups > 0U);
    RLF_CHECK(indexed_stats.keyword_capacity_skips == 0U);
    RLF_CHECK(indexed_stats.keyword_index_incremental_inserts ==
              indexed.routes().front().keywords.size());
}

RLF_TEST_CASE("Tool router reports keyword capacity saturation") {
    rlf::solstice::ToolRouterConfig config;
    config.maximum_tools = 1U;
    config.maximum_keywords_per_tool = 2U;
    rlf::solstice::ToolRouter router(config);
    router.train("alpha beta gamma", "example_tool");
    const auto stats = router.training_operation_stats();
    RLF_CHECK(stats.training_rows == 1U);
    RLF_CHECK(stats.keyword_insert_attempts == 3U);
    RLF_CHECK(stats.keyword_inserts == 2U);
    RLF_CHECK(stats.keyword_capacity_skips == 1U);
}

RLF_TEST_CASE("Solstice file tools remain inside configured sandbox") {
    const std::filesystem::path root = temporary_path("tool_root");
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    {
        std::ofstream output(root / "note.txt");
        output << "safe content";
    }
    rlf::solstice::ToolPolicy policy;
    policy.sandbox_root = root;
    policy.allow_file_reads = true;
    rlf::solstice::ToolRuntime tools(policy);
    tools.register_safe_builtins();
    const rlf::solstice::ToolResult allowed = tools.execute(
        rlf::solstice::ToolCall{"read_text_file", {{"path", "note.txt"}}}
    );
    RLF_CHECK(allowed.success);
    RLF_CHECK(allowed.output == "safe content");
    const rlf::solstice::ToolResult blocked = tools.execute(
        rlf::solstice::ToolCall{"read_text_file", {{"path", "../note.txt"}}}
    );
    RLF_CHECK(!blocked.success);
    std::filesystem::remove_all(root);
}

RLF_TEST_CASE("Solstice checkpoint roundtrip preserves behavior") {
    const std::filesystem::path path = temporary_path("roundtrip.rlfsp");
    std::filesystem::remove(path);
    rlf::solstice::SolsticeModel model;
    model.bootstrap();
    model.train_image(solid_image(230U, 30U, 30U), "a red status panel");
    model.learn_fact("ada", "parent", "bea", 0.99, "unit-test");
    const std::vector<float> continual_features(64U, 0.25F);
    model.learn_continually("checkpoint", "retained", continual_features);
    const std::uint64_t expected_hash = model.deterministic_hash();
    rlf::solstice::save_solstice_checkpoint(path, model);
    const rlf::solstice::SolsticeCheckpointSummary summary =
        rlf::solstice::inspect_solstice_checkpoint(path);
    RLF_CHECK(summary.format_version == 6U);
    RLF_CHECK(summary.stats.visual_examples == 1U);
    const rlf::solstice::SolsticeModel restored =
        rlf::solstice::load_solstice_checkpoint(path);
    RLF_CHECK(restored.deterministic_hash() == expected_hash);
    RLF_CHECK(restored.abstraction().facts().size() == 1U);
    RLF_CHECK(restored.continual().prototypes().size() == 1U);
    RLF_CHECK(!restored.grounding().links().empty());
    const rlf::solstice::SolsticeResponse response = restored.respond(
        "Who are you?", nullptr, nullptr
    );
    RLF_CHECK(response.text.find("Solstice-General-Frontier") != std::string::npos);
    std::filesystem::remove(path);
}


RLF_TEST_CASE("Solstice frontier 24G profile exposes bounded large-scale capacities") {
    const rlf::solstice::SolsticeConfig config =
        rlf::solstice::make_profile_config(
            rlf::solstice::SolsticeProfile::frontier_24g
        );
    const rlf::solstice::ProfileCapacityEstimate capacity =
        rlf::solstice::estimate_profile_capacity(
            rlf::solstice::SolsticeProfile::frontier_24g
        );
    RLF_CHECK(config.tokenizer.maximum_vocabulary == 65'536U);
    RLF_CHECK(config.language.context_orders.back() == 2'048U);
    RLF_CHECK(config.language.maximum_contexts == 20'000'000U);
    RLF_CHECK(config.vision.descriptor_dimensions == 32U);
    RLF_CHECK(config.vision.patch_sizes.size() == 4U);
    RLF_CHECK(config.vision.retrieval_candidate_batch == 8'192U);
    RLF_CHECK(capacity.gpu_working_set_bytes == 21ULL * 1024ULL * 1024ULL * 1024ULL);
}

RLF_TEST_CASE("Solstice frontier vision resizes large images and preserves source coordinates") {
    rlf::solstice::VisionConfig config;
    config.patch_sizes = {8U, 16U, 32U};
    config.descriptor_dimensions = 32U;
    config.maximum_input_side = 64U;
    config.maximum_patches = 128U;
    config.retrieval_query_batch = 16U;
    config.retrieval_candidate_batch = 16U;
    config.training_patch_batch = 16U;
    rlf::solstice::VisualPatchFabric vision(config);
    rlf::solstice::ImageData image = solid_image(220U, 40U, 40U);
    image.width = 256U;
    image.height = 128U;
    image.rgb.assign(image.width * image.height * 3U, 0U);
    for (std::size_t pixel = 0U; pixel < image.width * image.height; ++pixel) {
        image.rgb[pixel * 3U] = 220U;
        image.rgb[pixel * 3U + 1U] = 40U;
        image.rgb[pixel * 3U + 2U] = 40U;
    }
    vision.train(image, "a red panoramic panel");
    const rlf::solstice::VisionAnalysis analysis = vision.analyze(image);
    RLF_CHECK(!vision.modes().empty());
    RLF_CHECK(vision.modes().front().prototype.size() == 32U);
    RLF_CHECK(analysis.width == 256U);
    RLF_CHECK(analysis.height == 128U);
    RLF_CHECK(!analysis.regions.empty());
    for (const rlf::solstice::VisualRegion& region : analysis.regions) {
        RLF_CHECK(region.x + region.width <= analysis.width);
        RLF_CHECK(region.y + region.height <= analysis.height);
    }
}

RLF_TEST_CASE("Solstice optimized backend reuses bounded candidate caches") {
    const auto backend = rlf::frontier::make_frontier_backend(
        rlf::frontier::FrontierBackendKind::optimized_cpu
    );
    const std::vector<float> candidates{
        1.0F, 0.0F, 0.0F, 1.0F, -1.0F, 0.0F,
    };
    const std::vector<float> queries{1.0F, 0.0F, 0.0F, 1.0F};
    backend->prepare_candidate_cache(candidates, 3U, 2U, 77U);
    const std::vector<float> cached = backend->batch_cosine_cached(
        queries, 2U, 1U, 2U, 2U, 77U
    );
    const std::vector<float> direct = backend->batch_cosine(
        queries,
        2U,
        std::span<const float>(candidates.data() + 2U, 4U),
        2U,
        2U
    );
    RLF_CHECK(cached.size() == direct.size());
    for (std::size_t index = 0U; index < cached.size(); ++index) {
        RLF_CHECK_NEAR(cached[index], direct[index], 1.0e-7F);
    }
    const auto operations = backend->operation_stats();
    RLF_CHECK(operations.batch_cosine_calls == 2U);
    RLF_CHECK(operations.cached_batch_cosine_calls == 1U);
    RLF_CHECK(operations.inline_norm_cosine_calls == 2U);
    RLF_CHECK(operations.host_batch_cosine_calls == 2U);
    RLF_CHECK(operations.device_batch_cosine_calls == 0U);
    RLF_CHECK(operations.precomputed_norm_cosine_calls == 0U);
    RLF_CHECK(!operations.precomputed_cached_cosine_norms);
}

RLF_TEST_CASE("Solstice indexed cosine batching preserves independent pair results") {
    const auto backend = rlf::frontier::make_frontier_backend(
        rlf::frontier::FrontierBackendKind::optimized_cpu
    );
    const std::vector<float> queries{
        1.0F, 0.0F, 0.0F, 1.0F, -1.0F, 0.0F,
    };
    const std::vector<float> candidates{
        1.0F, 0.0F, 0.0F, -1.0F, -1.0F, 0.0F, 0.0F, 1.0F,
    };
    const std::vector<std::size_t> query_indices{0U, 0U, 2U, 1U};
    const std::vector<float> indexed = backend->batch_cosine_indexed(
        queries, 3U, candidates, query_indices, 2U
    );
    for (std::size_t pair = 0U; pair < query_indices.size(); ++pair) {
        const std::vector<float> reference = backend->batch_cosine(
            std::span<const float>(
                queries.data() + query_indices[pair] * 2U, 2U
            ),
            1U,
            std::span<const float>(candidates.data() + pair * 2U, 2U),
            1U,
            2U
        );
        RLF_CHECK(indexed[pair] == reference.front());
    }
    const auto operations = backend->operation_stats();
    RLF_CHECK(operations.indexed_batch_cosine_calls == 1U);
    RLF_CHECK(operations.indexed_cosine_pairs == 4U);
}

RLF_TEST_CASE("Solstice batched sparse reranking preserves visual learning state") {
    rlf::solstice::VisionConfig config;
    config.patch_size = 4U;
    config.patch_sizes = {4U};
    config.descriptor_dimensions = 32U;
    config.maximum_input_side = 16U;
    config.maximum_patches = 64U;
    config.maximum_modes = 128U;
    config.maximum_examples = 64U;
    config.training_patch_batch = 16U;
    config.retrieval_query_batch = 16U;
    config.retrieval_candidate_batch = 16U;
    config.sparse_routing_minimum_modes = 1U;
    config.sparse_router.signature_bits = 8U;
    config.sparse_router.maximum_candidates = 16U;
    config.sparse_router.probe_radius = 2U;
    config.mode_creation_similarity = 0.99999;
    rlf::solstice::VisualPatchFabric batched(config);
    rlf::solstice::VisualPatchFabric reference(config);
    batched.set_batched_sparse_reranking(true);
    reference.set_batched_sparse_reranking(false);

    const std::array<std::array<std::uint8_t, 3U>, 8U> colors{{
        {{230U, 20U, 20U}}, {{20U, 230U, 20U}},
        {{20U, 20U, 230U}}, {{220U, 180U, 20U}},
        {{180U, 20U, 220U}}, {{20U, 180U, 220U}},
        {{120U, 80U, 40U}}, {{40U, 120U, 80U}},
    }};
    for (std::size_t index = 0U; index < colors.size(); ++index) {
        const auto& color = colors[index];
        const auto image = solid_image(color[0U], color[1U], color[2U]);
        const std::string caption = "rerank sample " + std::to_string(index);
        batched.train(image, caption);
        reference.train(image, caption);
        RLF_CHECK(batched.deterministic_hash() == reference.deterministic_hash());
    }
    const auto batched_stats = batched.backend_operation_stats();
    const auto reference_stats = reference.backend_operation_stats();
    RLF_CHECK(batched_stats.indexed_batch_cosine_calls > 0U);
    RLF_CHECK(reference_stats.indexed_batch_cosine_calls == 0U);
    RLF_CHECK(batched_stats.batch_cosine_calls < reference_stats.batch_cosine_calls);
}

RLF_TEST_CASE("Solstice indexed concept updates preserve visual learning state") {
    rlf::solstice::VisionConfig config;
    config.patch_size = 8U;
    config.patch_sizes = {8U};
    config.descriptor_dimensions = 32U;
    config.maximum_input_side = 32U;
    config.maximum_patches = 64U;
    config.maximum_modes = 128U;
    config.maximum_examples = 64U;
    config.maximum_concepts_per_mode = 64U;
    config.training_patch_batch = 16U;
    config.retrieval_query_batch = 16U;
    config.retrieval_candidate_batch = 16U;
    config.sparse_routing_minimum_modes = 1U;
    config.sparse_router.signature_bits = 8U;
    config.sparse_router.maximum_candidates = 16U;
    config.sparse_router.probe_radius = 2U;
    config.mode_creation_similarity = 0.99999;
    rlf::solstice::VisualPatchFabric indexed(config);
    rlf::solstice::VisualPatchFabric reference(config);
    indexed.set_indexed_concept_updates(true);
    reference.set_indexed_concept_updates(false);
    std::string caption;
    for (std::size_t concept_index = 0U; concept_index < 48U; ++concept_index) {
        if (!caption.empty()) caption.push_back(' ');
        caption += "concept_" + std::to_string(concept_index);
    }
    const auto image = solid_image(120U, 80U, 40U);
    for (std::size_t repetition = 0U; repetition < 6U; ++repetition) {
        indexed.train(image, caption);
        reference.train(image, caption);
        RLF_CHECK(indexed.deterministic_hash() == reference.deterministic_hash());
    }
    const auto indexed_stats = indexed.training_operation_stats();
    const auto reference_stats = reference.training_operation_stats();
    RLF_CHECK(indexed_stats.concept_update_lookups ==
              reference_stats.concept_update_lookups);
    RLF_CHECK(indexed_stats.indexed_concept_lookups > 0U);
    RLF_CHECK(indexed_stats.linear_concept_comparisons == 0U);
    RLF_CHECK(reference_stats.indexed_concept_lookups == 0U);
    RLF_CHECK(reference_stats.linear_concept_comparisons > 0U);
}

RLF_TEST_CASE("Solstice indexed example duplicate lookup preserves visual learning state") {
    rlf::solstice::VisionConfig config;
    config.patch_size = 8U;
    config.patch_sizes = {8U};
    config.descriptor_dimensions = 32U;
    config.maximum_input_side = 32U;
    config.maximum_patches = 64U;
    config.maximum_modes = 128U;
    config.maximum_examples = 128U;
    config.maximum_concepts_per_mode = 16U;
    config.training_patch_batch = 16U;
    config.retrieval_query_batch = 16U;
    config.retrieval_candidate_batch = 16U;
    config.sparse_routing_minimum_modes = 1U;
    config.sparse_router.signature_bits = 8U;
    config.sparse_router.maximum_candidates = 16U;
    config.sparse_router.probe_radius = 2U;
    config.mode_creation_similarity = 0.99999;
    rlf::solstice::VisualPatchFabric indexed(config);
    rlf::solstice::VisualPatchFabric reference(config);
    indexed.set_indexed_example_duplicate_lookup(true);
    reference.set_indexed_example_duplicate_lookup(false);

    const std::array<std::array<std::uint8_t, 3U>, 8U> colors{{
        {{230U, 20U, 20U}}, {{20U, 230U, 20U}},
        {{20U, 20U, 230U}}, {{220U, 180U, 20U}},
        {{180U, 20U, 220U}}, {{20U, 180U, 220U}},
        {{120U, 80U, 40U}}, {{40U, 120U, 80U}},
    }};
    for (std::size_t index = 0U; index < colors.size(); ++index) {
        const auto& color = colors[index];
        const auto image = solid_image(color[0U], color[1U], color[2U]);
        const std::string caption = "unique sample " + std::to_string(index);
        indexed.train(image, caption);
        reference.train(image, caption);
        RLF_CHECK(indexed.deterministic_hash() == reference.deterministic_hash());
    }
    const auto repeated = solid_image(20U, 20U, 230U);
    indexed.train(repeated, "unique sample 2");
    reference.train(repeated, "unique sample 2");
    RLF_CHECK(indexed.deterministic_hash() == reference.deterministic_hash());

    const auto indexed_stats = indexed.training_operation_stats();
    const auto reference_stats = reference.training_operation_stats();
    RLF_CHECK(indexed_stats.example_duplicate_lookups ==
              reference_stats.example_duplicate_lookups);
    RLF_CHECK(indexed_stats.linear_example_comparisons == 0U);
    RLF_CHECK(indexed_stats.indexed_example_candidates == 1U);
    RLF_CHECK(reference_stats.indexed_example_candidates == 0U);
    RLF_CHECK(reference_stats.linear_example_comparisons >
              indexed_stats.indexed_example_candidates);
}

RLF_TEST_CASE("Solstice persistent mode ID index preserves visual learning state") {
    rlf::solstice::VisionConfig config;
    config.patch_size = 4U;
    config.patch_sizes = {4U};
    config.descriptor_dimensions = 32U;
    config.maximum_input_side = 16U;
    config.maximum_patches = 64U;
    config.maximum_modes = 128U;
    config.maximum_examples = 64U;
    config.training_patch_batch = 4U;
    config.retrieval_query_batch = 4U;
    config.retrieval_candidate_batch = 16U;
    config.sparse_routing_minimum_modes = 1U;
    config.sparse_router.signature_bits = 8U;
    config.sparse_router.maximum_candidates = 16U;
    config.sparse_router.probe_radius = 2U;
    config.mode_creation_similarity = 0.99999;
    rlf::solstice::VisualPatchFabric persistent(config);
    rlf::solstice::VisualPatchFabric reference(config);
    persistent.set_persistent_mode_id_index(true);
    reference.set_persistent_mode_id_index(false);

    const std::array<std::array<std::uint8_t, 3U>, 8U> colors{{
        {{230U, 20U, 20U}}, {{20U, 230U, 20U}},
        {{20U, 20U, 230U}}, {{220U, 180U, 20U}},
        {{180U, 20U, 220U}}, {{20U, 180U, 220U}},
        {{120U, 80U, 40U}}, {{40U, 120U, 80U}},
    }};
    for (std::size_t index = 0U; index < colors.size(); ++index) {
        const auto& color = colors[index];
        const auto image = solid_image(color[0U], color[1U], color[2U]);
        const std::string caption = "mode index sample " + std::to_string(index);
        const auto persistent_analysis = persistent.train_and_analyze(image, caption);
        const auto reference_analysis = reference.train_and_analyze(image, caption);
        RLF_CHECK(persistent.deterministic_hash() == reference.deterministic_hash());
        RLF_CHECK(persistent_analysis.description == reference_analysis.description);
        RLF_CHECK(persistent_analysis.concepts == reference_analysis.concepts);
        RLF_CHECK(persistent_analysis.confidence == reference_analysis.confidence);
        RLF_CHECK(persistent_analysis.nearest_example_id ==
                  reference_analysis.nearest_example_id);
        RLF_CHECK(persistent_analysis.regions.size() ==
                  reference_analysis.regions.size());
        for (std::size_t region = 0U;
             region < persistent_analysis.regions.size();
             ++region) {
            const auto& left = persistent_analysis.regions[region];
            const auto& right = reference_analysis.regions[region];
            RLF_CHECK(left.mode_id == right.mode_id);
            RLF_CHECK(left.x == right.x);
            RLF_CHECK(left.y == right.y);
            RLF_CHECK(left.width == right.width);
            RLF_CHECK(left.height == right.height);
            RLF_CHECK(left.patch_count == right.patch_count);
            RLF_CHECK(left.concept_name == right.concept_name);
            RLF_CHECK(left.confidence == right.confidence);
        }
    }
    const auto persistent_stats = persistent.training_operation_stats();
    const auto reference_stats = reference.training_operation_stats();
    RLF_CHECK(persistent_stats.mode_id_lookups == reference_stats.mode_id_lookups);
    RLF_CHECK(persistent_stats.mode_id_index_entries_rebuilt == 0U);
    RLF_CHECK(reference_stats.mode_id_index_entries_rebuilt > 0U);
    RLF_CHECK(persistent_stats.mode_id_index_incremental_inserts ==
              reference_stats.mode_id_index_incremental_inserts);
    RLF_CHECK(persistent_stats.region_mode_id_lookups ==
              reference_stats.region_mode_id_lookups);
    RLF_CHECK(persistent_stats.indexed_region_mode_lookups > 0U);
    RLF_CHECK(reference_stats.linear_region_mode_comparisons > 0U);

    auto resumed_persistent =
        rlf::solstice::VisualPatchFabric::from_snapshot(persistent.snapshot());
    auto resumed_reference =
        rlf::solstice::VisualPatchFabric::from_snapshot(reference.snapshot());
    resumed_persistent.set_persistent_mode_id_index(true);
    resumed_reference.set_persistent_mode_id_index(false);
    const auto resumed_image = solid_image(90U, 110U, 130U);
    const auto resumed_persistent_analysis = resumed_persistent.train_and_analyze(
        resumed_image, "resumed mode index sample"
    );
    const auto resumed_reference_analysis = resumed_reference.train_and_analyze(
        resumed_image, "resumed mode index sample"
    );
    RLF_CHECK(resumed_persistent.deterministic_hash() ==
              resumed_reference.deterministic_hash());
    RLF_CHECK(resumed_persistent_analysis.description ==
              resumed_reference_analysis.description);
    RLF_CHECK(resumed_persistent_analysis.concepts ==
              resumed_reference_analysis.concepts);
    RLF_CHECK(resumed_persistent_analysis.regions.size() ==
              resumed_reference_analysis.regions.size());
}

RLF_TEST_CASE("Solstice incremental sparse routing preserves learned visual state") {
    rlf::solstice::VisionConfig config;
    config.patch_size = 4U;
    config.patch_sizes = {4U};
    config.descriptor_dimensions = 32U;
    config.maximum_input_side = 16U;
    config.maximum_patches = 64U;
    config.maximum_modes = 128U;
    config.maximum_examples = 64U;
    config.training_patch_batch = 4U;
    config.retrieval_query_batch = 4U;
    config.retrieval_candidate_batch = 16U;
    config.sparse_routing_minimum_modes = 1U;
    config.sparse_router.signature_bits = 8U;
    config.sparse_router.maximum_candidates = 16U;
    config.sparse_router.probe_radius = 2U;
    config.mode_creation_similarity = 0.99999;
    rlf::solstice::VisualPatchFabric incremental(config);
    rlf::solstice::VisualPatchFabric reference(config);
    incremental.set_incremental_sparse_router_updates(true);
    reference.set_incremental_sparse_router_updates(false);

    const std::array<std::array<std::uint8_t, 3U>, 8U> colors{{
        {{230U, 20U, 20U}}, {{20U, 230U, 20U}},
        {{20U, 20U, 230U}}, {{220U, 180U, 20U}},
        {{180U, 20U, 220U}}, {{20U, 180U, 220U}},
        {{120U, 80U, 40U}}, {{40U, 120U, 80U}},
    }};
    for (std::size_t index = 0U; index < colors.size(); ++index) {
        const auto& color = colors[index];
        const auto image = solid_image(color[0U], color[1U], color[2U]);
        const std::string caption = "color sample " + std::to_string(index);
        incremental.train(image, caption);
        reference.train(image, caption);
        RLF_CHECK(incremental.deterministic_hash() == reference.deterministic_hash());
    }
    const auto incremental_stats = incremental.sparse_router_operation_stats();
    const auto reference_stats = reference.sparse_router_operation_stats();
    RLF_CHECK(
        incremental_stats.vectors_incrementally_updated +
        incremental_stats.vectors_appended > 0U
    );
    RLF_CHECK(reference_stats.vectors_incrementally_updated == 0U);
    RLF_CHECK(incremental_stats.vectors_rebuilt < reference_stats.vectors_rebuilt);
}
