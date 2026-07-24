#include "test_framework.hpp"

#include "rlf/experiments/rlf7_frontier.hpp"
#include "rlf/frontier/frontier_backend.hpp"
#include "rlf/frontier/frontier_trainer.hpp"
#include "rlf/frontier/knowledge_fabric.hpp"
#include "rlf/frontier/multimodal.hpp"
#include "rlf/storage/rlf7_checkpoint.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

namespace {

rlf::frontier::Image test_image(const bool right, const std::size_t shift = 0U) {
    rlf::frontier::Image image{
        .width = 32U,
        .height = 32U,
        .channels = 3U,
        .pixels = std::vector<std::uint8_t>(32U * 32U * 3U, 12U),
    };
    const std::size_t start_x = right ? 19U - shift : 3U + shift;
    for (std::size_t y = 9U; y < 21U; ++y) {
        for (std::size_t x = start_x; x < start_x + 9U; ++x) {
            const std::size_t index = (y * image.width + x) * 3U;
            image.pixels[index] = right ? 25U : 230U;
            image.pixels[index + 1U] = right ? 220U : 30U;
            image.pixels[index + 2U] = 50U;
        }
    }
    return image;
}

std::vector<rlf::frontier::Image> test_video(const bool right) {
    std::vector<rlf::frontier::Image> frames;
    for (std::size_t frame = 0U; frame < 5U; ++frame) {
        rlf::frontier::Image image{
            .width = 40U,
            .height = 32U,
            .channels = 3U,
            .pixels = std::vector<std::uint8_t>(40U * 32U * 3U, 10U),
        };
        const std::size_t start_x = right ? 3U + frame * 5U : 29U - frame * 5U;
        for (std::size_t y = 11U; y < 19U; ++y) {
            for (std::size_t x = start_x; x < start_x + 7U; ++x) {
                const std::size_t index = (y * image.width + x) * 3U;
                image.pixels[index] = 50U;
                image.pixels[index + 1U] = 170U;
                image.pixels[index + 2U] = 230U;
            }
        }
        frames.push_back(std::move(image));
    }
    return frames;
}

}  // namespace

RLF_TEST_CASE("RLF-7 knowledge correction contradiction and bounded exact retrieval") {
    rlf::frontier::KnowledgeFabric fabric(7U);
    rlf::frontier::KnowledgeRecord first;
    first.subject = "earth";
    first.predicate = "shape";
    first.object = "sphere";
    first.source = "verified";
    first.confidence = 0.99;
    first.verified = true;
    const auto first_id = fabric.insert(first);
    rlf::frontier::KnowledgeRecord second = first;
    second.stable_id = 0U;
    second.object = "flat";
    second.source = "unreliable";
    second.confidence = 0.2;
    const auto second_id = fabric.insert(second);
    const auto hits = fabric.query({
        .subject = "earth",
        .predicate = "shape",
        .terms = {},
        .maximum_results = 2U,
        .include_stale = false,
        .include_invalidated = false,
    });
    RLF_CHECK(hits.size() == 2U);
    RLF_CHECK(hits.front().stable_id == first_id);
    RLF_CHECK(fabric.statistics().last_candidates_examined == 2U);
    RLF_CHECK(fabric.invalidate(second_id));
    const auto filtered = fabric.query({
        .subject = "earth", .predicate = "shape", .terms = {}, .maximum_results = 2U,
        .include_stale = false, .include_invalidated = false,
    });
    RLF_CHECK(filtered.size() == 1U);
}

RLF_TEST_CASE("RLF-7 hierarchical and cross-modal modes are learned and linked") {
    rlf::frontier::KnowledgeFabric fabric(11U);
    const std::array<float, 3U> root_value{1.0F, 0.0F, 0.0F};
    const std::array<float, 3U> child_value{0.9F, 0.1F, 0.0F};
    const std::array<float, 3U> audio_value{0.8F, 0.2F, 0.0F};
    const auto root = fabric.learn_mode(rlf::frontier::Modality::image, "object", root_value);
    const auto child = fabric.learn_mode(rlf::frontier::Modality::image, "red_object", child_value, root);
    const auto audio = fabric.learn_mode(rlf::frontier::Modality::audio, "red_object_sound", audio_value);
    RLF_CHECK(fabric.link_modes(child, audio));
    RLF_CHECK(fabric.find_mode(root)->child_ids.contains(child));
    RLF_CHECK(fabric.find_mode(child)->linked_modes.contains(audio));
    const auto hits = fabric.retrieve_modes(rlf::frontier::Modality::image, child_value, 1U);
    RLF_CHECK(hits.size() == 1U);
    RLF_CHECK(hits.front().stable_id == child);
}

RLF_TEST_CASE("RLF-7 native image understanding localizes foreground and round trips PPM") {
    const auto image = test_image(false);
    const auto observation = rlf::frontier::ImageUnderstanding::analyze(image);
    RLF_CHECK(observation.descriptor.size() == 64U);
    RLF_CHECK(!observation.regions.empty());
    RLF_CHECK(observation.regions.front().x <= 4U);
    const auto path = std::filesystem::temp_directory_path() / "rlf7_image_test.ppm";
    rlf::frontier::ImageUnderstanding::save_ppm(path, image);
    const auto loaded = rlf::frontier::ImageUnderstanding::load_pnm(path);
    RLF_CHECK(loaded.pixels == image.pixels);
    std::filesystem::remove(path);
}

RLF_TEST_CASE("RLF-7 video understanding tracks direction and predicts frames") {
    const auto right = rlf::frontier::VideoUnderstanding::analyze(test_video(true));
    const auto left = rlf::frontier::VideoUnderstanding::analyze(test_video(false));
    RLF_CHECK(!right.tracks.empty());
    RLF_CHECK(right.tracks.front().continuity > 0.9);
    RLF_CHECK(right.descriptor[0] > 0.0F);
    RLF_CHECK(left.descriptor[0] < 0.0F);
    const auto predicted = rlf::frontier::VideoUnderstanding::predict_next_frames(right, 3U);
    RLF_CHECK(predicted.size() == 3U);
    RLF_CHECK(predicted.front().valid());
}

RLF_TEST_CASE("RLF-7 PCM16 audio understanding and prototype synthesis") {
    rlf::frontier::AudioObservation source;
    source.sample_rate = 16'000U;
    source.rms = 0.25;
    source.dominant_frequency_hz = 440.0;
    const auto path = std::filesystem::temp_directory_path() / "rlf7_audio_test.wav";
    rlf::frontier::AudioUnderstanding::synthesize_prototype_wav(path, source, 0.2);
    const auto analyzed = rlf::frontier::AudioUnderstanding::analyze_wav(path);
    RLF_CHECK(analyzed.descriptor.size() == 32U);
    RLF_CHECK(analyzed.duration_seconds > 0.19);
    RLF_CHECK(analyzed.rms > 0.05);
    std::filesystem::remove(path);
}

RLF_TEST_CASE("RLF-Frontier scalar and optimized batch backends agree") {
    const auto scalar = rlf::frontier::make_frontier_backend(rlf::frontier::FrontierBackendKind::scalar_cpu);
    const auto optimized = rlf::frontier::make_frontier_backend(rlf::frontier::FrontierBackendKind::optimized_cpu);
    const std::array<float, 8U> queries{1.0F,0.0F,0.0F,1.0F, 0.0F,1.0F,1.0F,0.0F};
    const std::array<float, 8U> candidates{1.0F,0.0F,0.0F,1.0F, 1.0F,1.0F,0.0F,0.0F};
    const auto left = scalar->batch_cosine(queries, 2U, candidates, 2U, 4U);
    const auto right = optimized->batch_cosine(queries, 2U, candidates, 2U, 4U);
    RLF_CHECK(left.size() == right.size());
    for (std::size_t index = 0U; index < left.size(); ++index) {
        RLF_CHECK_NEAR(static_cast<double>(left[index]), static_cast<double>(right[index]), 1.0e-8);
    }
    std::vector<float> large_queries(64U * 16U);
    std::vector<float> large_candidates(64U * 16U);
    for (std::size_t index = 0U; index < large_queries.size(); ++index) {
        large_queries[index] = static_cast<float>((index * 17U + 3U) % 31U) / 31.0F;
        large_candidates[index] = static_cast<float>((index * 11U + 7U) % 29U) / 29.0F;
    }
    const auto scalar_large = scalar->batch_cosine(
        large_queries, 64U, large_candidates, 64U, 16U
    );
    const auto optimized_large = optimized->batch_cosine(
        large_queries, 64U, large_candidates, 64U, 16U
    );
    RLF_CHECK(scalar_large == optimized_large);
    std::array<float, 4U> scalar_prototype{0.1F, 0.2F, 0.3F, 0.4F};
    std::array<float, 4U> optimized_prototype = scalar_prototype;
    const std::array<float, 4U> observation{0.9F, 0.7F, 0.5F, 0.3F};
    scalar->local_average_update(scalar_prototype, observation, 0.25F);
    optimized->local_average_update(optimized_prototype, observation, 0.25F);
    RLF_CHECK(scalar_prototype == optimized_prototype);
    const auto scalar_operations = scalar->operation_stats();
    const auto optimized_operations = optimized->operation_stats();
    RLF_CHECK(scalar_operations.batch_cosine_calls == 2U);
    RLF_CHECK(optimized_operations.batch_cosine_calls == 2U);
    RLF_CHECK(scalar_operations.local_update_calls == 1U);
    RLF_CHECK(scalar_operations.host_local_update_calls == 1U);
    RLF_CHECK(scalar_operations.device_local_update_calls == 0U);
    RLF_CHECK(optimized_operations.local_update_calls == 1U);
    RLF_CHECK(optimized_operations.host_to_device_bytes == 0U);
#ifndef RLF_HAS_CUDA
    const auto cuda = rlf::frontier::make_frontier_backend(rlf::frontier::FrontierBackendKind::cuda);
    RLF_CHECK(!cuda->capabilities().available);
    RLF_CHECK(cuda->operation_stats().local_update_calls == 0U);
#endif
}

RLF_TEST_CASE("RLF-Frontier trainer uses selected backend for learning and inference") {
    const auto directory = std::filesystem::temp_directory_path() / "rlf_frontier_backend_train_test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    rlf::frontier::ImageUnderstanding::save_ppm(directory / "left0.ppm", test_image(false, 0U));
    rlf::frontier::ImageUnderstanding::save_ppm(directory / "left1.ppm", test_image(false, 1U));
    rlf::frontier::ImageUnderstanding::save_ppm(directory / "right0.ppm", test_image(true, 0U));
    rlf::frontier::ImageUnderstanding::save_ppm(directory / "right1.ppm", test_image(true, 1U));
    const auto manifest_path = directory / "dataset.tsv";
    std::ofstream manifest(manifest_path);
    manifest << "train\timage\tleft\tleft0.ppm\n"
        << "train\timage\tleft\tleft1.ppm\n"
        << "train\timage\tright\tright0.ppm\n"
        << "evaluation\timage\tright\tright1.ppm\n";
    manifest.close();
    const auto loaded = rlf::frontier::ManifestLoader::load(manifest_path);
    rlf::frontier::FrontierTrainer scalar(
        rlf::frontier::FrontierModel(71U),
        rlf::frontier::FrontierBackendKind::scalar_cpu
    );
    rlf::frontier::FrontierTrainer optimized(
        rlf::frontier::FrontierModel(71U),
        rlf::frontier::FrontierBackendKind::optimized_cpu
    );
    const auto scalar_training = scalar.train(loaded);
    const auto optimized_training = optimized.train(loaded);
    const auto scalar_evaluation = scalar.evaluate(loaded);
    const auto optimized_evaluation = optimized.evaluate(loaded);
    RLF_CHECK(scalar.backend_name() == "scalar_cpu");
    RLF_CHECK(optimized.backend_name() == "optimized_cpu");
    RLF_CHECK(scalar_training.deterministic_hash == optimized_training.deterministic_hash);
    RLF_CHECK(scalar_evaluation.deterministic_hash == optimized_evaluation.deterministic_hash);
    RLF_CHECK(scalar_evaluation.image.accuracy == 1.0);
    RLF_CHECK(optimized_evaluation.image.accuracy == 1.0);
#ifndef RLF_HAS_CUDA
    RLF_CHECK_THROWS_AS(
        rlf::frontier::FrontierTrainer(
            rlf::frontier::FrontierModel(71U),
            rlf::frontier::FrontierBackendKind::cuda
        ),
        std::runtime_error
    );
#endif
    std::filesystem::remove_all(directory);
}

RLF_TEST_CASE("RLF7CKP9 and RLFFRT10 checkpoints round trip and reject corruption") {
    rlf::frontier::FrontierModel model(19U);
    rlf::frontier::KnowledgeRecord record;
    record.subject = "alpha";
    record.predicate = "is";
    record.object = "beta";
    record.source = "test";
    model.fabric.insert(record);
    const std::array<float, 3U> prototype{0.1F, 0.2F, 0.3F};
    model.fabric.learn_mode(rlf::frontier::Modality::image, "sample", prototype);
    const auto rlf7 = std::filesystem::temp_directory_path() / "rlf7_ckp9_test.rlf";
    const auto frontier = std::filesystem::temp_directory_path() / "rlf_frontier_ckp10_test.rlf";
    const auto corrupt = std::filesystem::temp_directory_path() / "rlf_frontier_corrupt.rlf";
    rlf::storage::save_rlf7_checkpoint(rlf7, model);
    rlf::storage::save_frontier_checkpoint(frontier, model, "scalar_cpu");
    const auto loaded7 = rlf::storage::load_rlf7_checkpoint(rlf7);
    const auto loaded8 = rlf::storage::load_frontier_checkpoint(frontier);
    RLF_CHECK(loaded7.fabric.deterministic_hash() == model.fabric.deterministic_hash());
    RLF_CHECK(loaded8.fabric.deterministic_hash() == model.fabric.deterministic_hash());
    RLF_CHECK(rlf::storage::inspect_frontier_checkpoint(frontier).format_version == 10U);
    RLF_CHECK(rlf::storage::inspect_frontier_checkpoint(frontier).backend == "scalar_cpu");
    RLF_CHECK_THROWS_AS(
        rlf::storage::save_frontier_checkpoint(frontier, model, "invalid_backend"),
        std::invalid_argument
    );
    std::ifstream input(frontier, std::ios::binary);
    std::vector<char> bytes{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    input.close();
    bytes.back() = static_cast<char>(bytes.back() ^ 1);
    std::ofstream output(corrupt, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    output.close();
    RLF_CHECK_THROWS_AS(rlf::storage::load_frontier_checkpoint(corrupt), std::runtime_error);
    std::filesystem::remove(rlf7);
    std::filesystem::remove(frontier);
    std::filesystem::remove(corrupt);
}

RLF_TEST_CASE("RLF-7 and Frontier experiment is deterministic and honest") {
    rlf::experiments::Rlf7FrontierConfig config;
    config.seed = 23U;
    config.knowledge_records = 128U;
    config.knowledge_queries = 32U;
    config.media_training_per_class = 3U;
    config.media_evaluation_per_class = 2U;
    config.run_agent_gate = false;
    const auto first = rlf::experiments::run_rlf7_frontier(config);
    const auto second = rlf::experiments::run_rlf7_frontier(config);
    RLF_CHECK(first.deterministic_hash == second.deterministic_hash);
    RLF_CHECK(first.knowledge_retrieval_accuracy == 1.0);
    RLF_CHECK(first.image_accuracy == 1.0);
    RLF_CHECK(first.video_accuracy == 1.0);
    RLF_CHECK(!first.frontier_claim_justified);
}

RLF_TEST_CASE("RLF-Frontier native manifest trains and evaluates multimodal files without retraining") {
    const auto directory = std::filesystem::temp_directory_path() / "rlf_frontier_manifest_test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    rlf::frontier::ImageUnderstanding::save_ppm(directory / "train_left.ppm", test_image(false, 0U));
    rlf::frontier::ImageUnderstanding::save_ppm(directory / "train_right.ppm", test_image(true, 0U));
    rlf::frontier::ImageUnderstanding::save_ppm(directory / "eval_left.ppm", test_image(false, 1U));
    rlf::frontier::ImageUnderstanding::save_ppm(directory / "eval_right.ppm", test_image(true, 1U));
    const auto train_video_right = test_video(true);
    const auto train_video_left = test_video(false);
    for (std::size_t index = 0U; index < train_video_right.size(); ++index) {
        rlf::frontier::ImageUnderstanding::save_ppm(
            directory / ("vr_" + std::to_string(index) + ".ppm"), train_video_right[index]
        );
        rlf::frontier::ImageUnderstanding::save_ppm(
            directory / ("vl_" + std::to_string(index) + ".ppm"), train_video_left[index]
        );
    }
    rlf::frontier::AudioObservation low;
    low.sample_rate = 16'000U;
    low.rms = 0.2;
    low.dominant_frequency_hz = 260.0;
    rlf::frontier::AudioObservation high = low;
    high.dominant_frequency_hz = 900.0;
    rlf::frontier::AudioUnderstanding::synthesize_prototype_wav(directory / "low.wav", low, 0.15);
    rlf::frontier::AudioUnderstanding::synthesize_prototype_wav(directory / "high.wav", high, 0.15);
    rlf::frontier::AudioObservation low_eval = low;
    low_eval.dominant_frequency_hz = 268.0;
    rlf::frontier::AudioObservation high_eval = high;
    high_eval.dominant_frequency_hz = 908.0;
    rlf::frontier::AudioUnderstanding::synthesize_prototype_wav(directory / "low_eval.wav", low_eval, 0.16);
    rlf::frontier::AudioUnderstanding::synthesize_prototype_wav(directory / "high_eval.wav", high_eval, 0.16);

    const auto join_frames = [](const std::string& prefix) {
        std::string value;
        for (std::size_t index = 0U; index < 5U; ++index) {
            if (!value.empty()) value.push_back(';');
            value += prefix + std::to_string(index) + ".ppm";
        }
        return value;
    };
    const auto manifest_path = directory / "dataset.tsv";
    std::ofstream manifest(manifest_path);
    manifest << "train\timage\tleft\ttrain_left.ppm\n"
        << "train\timage\tright\ttrain_right.ppm\n"
        << "evaluation\timage\tleft\teval_left.ppm\n"
        << "evaluation\timage\tright\teval_right.ppm\n"
        << "train\tvideo\tmove_right\t" << join_frames("vr_") << "\n"
        << "train\tvideo\tmove_left\t" << join_frames("vl_") << "\n"
        << "train\taudio\tlow\tlow.wav\n"
        << "train\taudio\thigh\thigh.wav\n"
        << "evaluation\taudio\tlow\tlow_eval.wav\n"
        << "evaluation\taudio\thigh\thigh_eval.wav\n"
        << "train\tstructured\tfact\twater|state|liquid|test\n";
    manifest.close();

    const auto loaded_manifest = rlf::frontier::ManifestLoader::load(manifest_path);
    RLF_CHECK(rlf::frontier::ManifestLoader::leakage_collisions(loaded_manifest) == 0U);
    rlf::frontier::FrontierTrainer trainer(rlf::frontier::FrontierModel(31U));
    const auto training = trainer.train(loaded_manifest);
    RLF_CHECK(training.image_examples == 2U);
    RLF_CHECK(training.video_examples == 2U);
    RLF_CHECK(training.audio_examples == 2U);
    const std::uint64_t training_examples = trainer.model().training_examples;
    const auto evaluation = trainer.evaluate(loaded_manifest);
    RLF_CHECK(evaluation.image.accuracy == 1.0);
    RLF_CHECK(evaluation.audio.accuracy == 1.0);
    RLF_CHECK(trainer.model().training_examples == training_examples);
    std::filesystem::remove_all(directory);
}

RLF_TEST_CASE("RLF-Frontier manifest leakage audit rejects identical train and evaluation content") {
    const auto directory = std::filesystem::temp_directory_path() / "rlf_frontier_leakage_test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    rlf::frontier::ImageUnderstanding::save_ppm(directory / "same.ppm", test_image(false));
    const auto manifest_path = directory / "leak.tsv";
    std::ofstream manifest(manifest_path);
    manifest << "train\timage\tleft\tsame.ppm\n"
        << "evaluation\timage\tleft\tsame.ppm\n";
    manifest.close();
    const auto loaded = rlf::frontier::ManifestLoader::load(manifest_path);
    RLF_CHECK(rlf::frontier::ManifestLoader::leakage_collisions(loaded) == 1U);
    rlf::frontier::FrontierTrainer trainer;
    RLF_CHECK_THROWS_AS(trainer.train(loaded), std::runtime_error);
    std::filesystem::remove_all(directory);
}

RLF_TEST_CASE("RLF-Frontier leakage audit rejects partially overlapping video sequences") {
    const auto directory = std::filesystem::temp_directory_path() / "rlf_frontier_video_leakage_test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const auto frames = test_video(true);
    for (std::size_t index = 0U; index < frames.size(); ++index) {
        rlf::frontier::ImageUnderstanding::save_ppm(
            directory / ("frame_" + std::to_string(index) + ".ppm"), frames[index]
        );
    }
    rlf::frontier::ImageUnderstanding::save_ppm(directory / "different.ppm", test_image(false));
    const auto manifest_path = directory / "video_leak.tsv";
    std::ofstream manifest(manifest_path);
    manifest << "train\tvideo\tmove\tframe_0.ppm;frame_1.ppm;frame_2.ppm\n"
        << "evaluation\tvideo\tmove\tframe_2.ppm;frame_3.ppm;different.ppm\n";
    manifest.close();
    const auto loaded = rlf::frontier::ManifestLoader::load(manifest_path);
    RLF_CHECK(rlf::frontier::ManifestLoader::leakage_collisions(loaded) >= 1U);
    rlf::frontier::FrontierTrainer trainer;
    RLF_CHECK_THROWS_AS(trainer.train(loaded), std::runtime_error);
    std::filesystem::remove_all(directory);
}
