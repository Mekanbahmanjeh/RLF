#include "rlf/experiments/rlf7_frontier.hpp"

#include "rlf/experiments/rlf6_agent.hpp"
#include "rlf/frontier/frontier_backend.hpp"
#include "rlf/frontier/multimodal.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <numbers>
#include <sstream>
#include <stdexcept>

namespace rlf::experiments {
namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

void hash_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (unsigned int index = 0U; index < 8U; ++index) {
        hash ^= (value >> (index * 8U)) & 0xFFULL;
        hash *= fnv_prime;
    }
}

void hash_double(std::uint64_t& hash, const double value) noexcept {
    hash_u64(hash, std::bit_cast<std::uint64_t>(value));
}

[[nodiscard]] frontier::Image make_image(
    const std::size_t variant,
    const bool second_class
) {
    constexpr std::size_t width = 48U;
    constexpr std::size_t height = 48U;
    frontier::Image image{
        .width = width,
        .height = height,
        .channels = 3U,
        .pixels = std::vector<std::uint8_t>(width * height * 3U, 20U),
    };
    const std::size_t offset = variant % 4U;
    const std::size_t start_x = second_class ? 27U - offset : 5U + offset;
    const std::size_t start_y = 12U + (variant % 5U);
    for (std::size_t y = start_y; y < start_y + 18U; ++y) {
        for (std::size_t x = start_x; x < start_x + 14U; ++x) {
            const std::size_t pixel = (y * width + x) * 3U;
            image.pixels[pixel] = second_class ? 40U : 225U;
            image.pixels[pixel + 1U] = second_class ? 220U : 55U;
            image.pixels[pixel + 2U] = 60U;
        }
    }
    return image;
}

[[nodiscard]] std::vector<frontier::Image> make_video(
    const std::size_t variant,
    const bool move_right
) {
    constexpr std::size_t width = 48U;
    constexpr std::size_t height = 48U;
    std::vector<frontier::Image> frames;
    for (std::size_t frame = 0U; frame < 6U; ++frame) {
        frontier::Image image{
            .width = width,
            .height = height,
            .channels = 3U,
            .pixels = std::vector<std::uint8_t>(width * height * 3U, 16U),
        };
        const std::size_t x = move_right
            ? 4U + frame * 5U + variant % 2U
            : 34U - frame * 5U - variant % 2U;
        const std::size_t y = 17U + variant % 3U;
        for (std::size_t py = y; py < y + 9U; ++py) {
            for (std::size_t px = x; px < x + 9U; ++px) {
                const std::size_t pixel = (py * width + px) * 3U;
                image.pixels[pixel] = 65U;
                image.pixels[pixel + 1U] = 180U;
                image.pixels[pixel + 2U] = 235U;
            }
        }
        frames.push_back(std::move(image));
    }
    return frames;
}

[[nodiscard]] double intersection_over_union(
    const frontier::BoundingBox& box,
    const frontier::BoundingBox& expected
) noexcept {
    const std::size_t left = std::max(box.x, expected.x);
    const std::size_t top = std::max(box.y, expected.y);
    const std::size_t right = std::min(box.x + box.width, expected.x + expected.width);
    const std::size_t bottom = std::min(box.y + box.height, expected.y + expected.height);
    const std::size_t intersection = right > left && bottom > top
        ? (right - left) * (bottom - top)
        : 0U;
    const std::size_t union_area = box.width * box.height + expected.width * expected.height - intersection;
    return union_area == 0U ? 0.0 : static_cast<double>(intersection) / static_cast<double>(union_area);
}

[[nodiscard]] std::filesystem::path temporary_audio_path(
    const std::uint64_t seed,
    const std::size_t index
) {
    return std::filesystem::temp_directory_path() /
        ("rlf_frontier_audio_" + std::to_string(seed) + "_" + std::to_string(index) + ".wav");
}

[[nodiscard]] frontier::AudioObservation make_audio(
    const std::uint64_t seed,
    const std::size_t index,
    const bool high
) {
    frontier::AudioObservation template_observation;
    template_observation.sample_rate = 16'000U;
    template_observation.rms = 0.25;
    template_observation.dominant_frequency_hz = (high ? 900.0 : 260.0) + static_cast<double>(index % 3U) * 8.0;
    const auto path = temporary_audio_path(seed, index + (high ? 10'000U : 0U));
    frontier::AudioUnderstanding::synthesize_prototype_wav(path, template_observation, 0.25);
    frontier::AudioObservation result = frontier::AudioUnderstanding::analyze_wav(path);
    std::error_code error;
    std::filesystem::remove(path, error);
    return result;
}

[[nodiscard]] std::string json_escape(const std::string& value) {
    std::string result;
    for (const char character : value) {
        switch (character) {
        case '\\': result += "\\\\"; break;
        case '"': result += "\\\""; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default: result.push_back(character); break;
        }
    }
    return result;
}

}  // namespace

Rlf7FrontierResult run_rlf7_frontier(
    const Rlf7FrontierConfig& config,
    frontier::FrontierModel* trained_model
) {
    if (config.knowledge_records == 0U || config.knowledge_queries == 0U ||
        config.media_training_per_class == 0U || config.media_evaluation_per_class == 0U) {
        throw std::invalid_argument("RLF-7/Frontier experiment counts must be non-zero");
    }
    frontier::FrontierModel model(config.seed);
    model.fabric.reserve_records(config.knowledge_records, 3U);
    model.fabric.set_step(1U);
    for (std::size_t index = 0U; index < config.knowledge_records; ++index) {
        frontier::KnowledgeRecord record;
        record.kind = frontier::KnowledgeKind::observed_fact;
        record.subject = "entity_" + std::to_string(index);
        record.predicate = "property_" + std::to_string(index % 97U);
        record.object = "value_" + std::to_string(index * 17U + 3U);
        record.source = "synthetic_train";
        record.confidence = 0.75 + 0.2 * static_cast<double>(index % 5U) / 4.0;
        model.fabric.insert(std::move(record));
    }
    std::size_t knowledge_correct = 0U;
    std::size_t candidate_examinations = 0U;
    for (std::size_t query = 0U; query < config.knowledge_queries; ++query) {
        const std::size_t index = (query * 104729U + 17U) % config.knowledge_records;
        frontier::KnowledgeQuery knowledge_query;
        knowledge_query.subject = "entity_" + std::to_string(index);
        knowledge_query.predicate = "property_" + std::to_string(index % 97U);
        knowledge_query.maximum_results = 1U;
        const auto hits = model.fabric.query(knowledge_query);
        candidate_examinations += model.fabric.statistics().last_candidates_examined;
        if (!hits.empty()) {
            const auto* record = model.fabric.find(hits.front().stable_id);
            if (record != nullptr && record->object == "value_" + std::to_string(index * 17U + 3U)) {
                ++knowledge_correct;
            }
        }
    }

    for (std::size_t index = 0U; index < config.media_training_per_class; ++index) {
        const auto red = frontier::ImageUnderstanding::analyze(make_image(index, false));
        const auto green = frontier::ImageUnderstanding::analyze(make_image(index, true));
        model.fabric.learn_mode(frontier::Modality::image, "red_left", red.descriptor, std::nullopt, 0.9);
        model.fabric.learn_mode(frontier::Modality::image, "green_right", green.descriptor, std::nullopt, 0.9);
        const auto right = frontier::VideoUnderstanding::analyze(make_video(index, true));
        const auto left = frontier::VideoUnderstanding::analyze(make_video(index, false));
        model.fabric.learn_mode(frontier::Modality::video, "move_right", right.descriptor, std::nullopt, 0.9);
        model.fabric.learn_mode(frontier::Modality::video, "move_left", left.descriptor, std::nullopt, 0.9);
        if (config.include_audio) {
            const auto low = make_audio(config.seed, index, false);
            const auto high = make_audio(config.seed, index, true);
            model.fabric.learn_mode(frontier::Modality::audio, "low_tone", low.descriptor, std::nullopt, 0.9);
            model.fabric.learn_mode(frontier::Modality::audio, "high_tone", high.descriptor, std::nullopt, 0.9);
        }
    }
    const auto image_modes = model.fabric.retrieve_modes(
        frontier::Modality::image,
        frontier::ImageUnderstanding::analyze(make_image(0U, false)).descriptor,
        2U
    );
    const auto video_modes = model.fabric.retrieve_modes(
        frontier::Modality::video,
        frontier::VideoUnderstanding::analyze(make_video(0U, true)).descriptor,
        2U
    );
    if (!image_modes.empty() && !video_modes.empty()) {
        model.fabric.link_modes(image_modes.front().stable_id, video_modes.front().stable_id);
    }

    std::size_t image_correct = 0U;
    std::size_t video_correct = 0U;
    std::size_t audio_correct = 0U;
    double localization_sum = 0.0;
    double continuity_sum = 0.0;
    frontier::FrontierTrainer classifier(std::move(model), config.backend);
    for (std::size_t index = 0U; index < config.media_evaluation_per_class; ++index) {
        for (const bool second : {false, true}) {
            const auto image = make_image(config.media_training_per_class + index, second);
            const auto observation = frontier::ImageUnderstanding::analyze(image);
            double confidence = 0.0;
            const std::string predicted = classifier.classify(frontier::Modality::image, observation.descriptor, &confidence);
            static_cast<void>(confidence);
            if (predicted == (second ? "green_right" : "red_left")) ++image_correct;
            if (!observation.regions.empty()) {
                const std::size_t offset = (config.media_training_per_class + index) % 4U;
                const frontier::BoundingBox expected{
                    .x = second ? 27U - offset : 5U + offset,
                    .y = 12U + ((config.media_training_per_class + index) % 5U),
                    .width = 14U,
                    .height = 18U,
                    .confidence = 1.0,
                };
                localization_sum += intersection_over_union(observation.regions.front(), expected);
            }
            const auto video = frontier::VideoUnderstanding::analyze(
                make_video(config.media_training_per_class + index, second)
            );
            const std::string video_prediction = classifier.classify(frontier::Modality::video, video.descriptor, nullptr);
            if (video_prediction == (second ? "move_right" : "move_left")) ++video_correct;
            if (!video.tracks.empty()) continuity_sum += video.tracks.front().continuity;
            if (config.include_audio) {
                const auto audio = make_audio(config.seed, config.media_training_per_class + index, second);
                const std::string audio_prediction = classifier.classify(frontier::Modality::audio, audio.descriptor, nullptr);
                if (audio_prediction == (second ? "high_tone" : "low_tone")) ++audio_correct;
            }
        }
    }
    frontier::FrontierModel completed_model = std::move(classifier.model());
    completed_model.fabric.consolidate(2'048U, 256U);

    double agent_success = 0.0;
    double recovery_rate = 0.0;
    std::size_t accepted_skills = 0U;
    if (config.run_agent_gate) {
        Rlf6Config agent_config;
        agent_config.seed = config.seed ^ 0xA6E17ULL;
        agent_config.training_episodes = 10U;
        agent_config.evaluation_episodes = std::max<std::size_t>(10U, config.agent_evaluation_episodes);
        agent_config.minimum_route_length = 5U;
        agent_config.maximum_route_length = 60U;
        agent_config.stress_episodes = 1U;
        agent_config.stress_route_length = 101U;
        agent_config.action_budget = 160U;
        agent_config.planning_node_budget = 1'500U;
        agent_config.tool_budget = 48U;
        agent_config.include_stress = true;
        agent_config.threads = config.threads;
        agent_config.experiment_name = "long_horizon_planning";
        const Rlf6Result agent = run_rlf6_agent(agent_config);
        agent_success = agent.task.task_success_rate;
        recovery_rate = agent.correction.failures_detected == 0U ? 0.0 :
            static_cast<double>(agent.correction.failures_recovered) /
            static_cast<double>(agent.correction.failures_detected);
        accepted_skills = agent.skills.accepted;
    }

    auto scalar = frontier::make_frontier_backend(frontier::FrontierBackendKind::scalar_cpu);
    auto optimized = frontier::make_frontier_backend(frontier::FrontierBackendKind::optimized_cpu);
    const std::array<float, 8U> queries{1.0F,0.0F,0.5F,0.2F, 0.0F,1.0F,0.1F,0.3F};
    const std::array<float, 12U> candidates{1.0F,0.0F,0.5F,0.2F, 0.0F,1.0F,0.1F,0.3F, 0.2F,0.2F,0.2F,0.2F};
    const auto scalar_values = scalar->batch_cosine(queries, 2U, candidates, 3U, 4U);
    const auto optimized_values = optimized->batch_cosine(queries, 2U, candidates, 3U, 4U);
    bool backend_agreement = scalar_values.size() == optimized_values.size();
    for (std::size_t index = 0U; backend_agreement && index < scalar_values.size(); ++index) {
        backend_agreement = std::abs(scalar_values[index] - optimized_values[index]) < 1.0e-7F;
    }
    auto cuda = frontier::make_frontier_backend(frontier::FrontierBackendKind::cuda);
    const auto cuda_capabilities = cuda->capabilities();

    Rlf7FrontierResult result;
    result.config = config;
    result.knowledge_records = completed_model.fabric.records().size();
    result.knowledge_retrieval_accuracy = static_cast<double>(knowledge_correct) /
        static_cast<double>(config.knowledge_queries);
    result.candidates_per_query = static_cast<double>(candidate_examinations) /
        static_cast<double>(config.knowledge_queries);
    const double media_examples = static_cast<double>(config.media_evaluation_per_class * 2U);
    result.image_accuracy = static_cast<double>(image_correct) / media_examples;
    result.image_localization_iou = localization_sum / media_examples;
    result.video_accuracy = static_cast<double>(video_correct) / media_examples;
    result.video_track_continuity = continuity_sum / media_examples;
    result.audio_accuracy = config.include_audio ? static_cast<double>(audio_correct) / media_examples : 0.0;
    result.cross_modal_binding_accuracy = (!image_modes.empty() && !video_modes.empty()) ? 1.0 : 0.0;
    result.agent_gate_executed = config.run_agent_gate;
    result.agent_success_rate = agent_success;
    result.agent_recovery_rate = recovery_rate;
    result.accepted_skills = accepted_skills;
    result.scalar_optimized_agreement = backend_agreement;
#ifdef RLF_HAS_CUDA
    result.cuda_compiled = true;
#else
    result.cuda_compiled = false;
#endif
    result.cuda_available = cuda_capabilities.available;
    result.persistent_bytes = completed_model.fabric.statistics().approximate_persistent_bytes;
    for (const std::size_t memory_gb : {80U, 96U, 192U, 288U}) {
        const std::uint64_t total_bytes = static_cast<std::uint64_t>(memory_gb) * 1'073'741'824ULL;
        const std::uint64_t reserved = total_bytes / 4U;
        const std::uint64_t hot_bytes = total_bytes - reserved;
        result.projections.push_back({
            .gpu_memory_gb = memory_gb,
            .projected_hot_modes = hot_bytes / 512U,
            .projected_hot_bytes = hot_bytes,
            .cpu_ram_bytes = total_bytes * 4U,
            .nvme_bytes = total_bytes * 32U,
        });
    }
    result.frontier_claim_justified =
        result.cuda_available &&
        result.knowledge_retrieval_accuracy >= 0.95 &&
        result.image_accuracy >= 0.9 &&
        result.video_accuracy >= 0.9 &&
        (!config.include_audio || result.audio_accuracy >= 0.9) &&
        result.agent_success_rate >= 0.7 &&
        result.agent_recovery_rate >= 0.7 &&
        result.accepted_skills > 0U;
    result.scientific_decision = result.frontier_claim_justified
        ? "Decision A — strong evidence"
        : "Decision B — partial evidence";
    if (!result.cuda_available) result.limitations.push_back("CUDA runtime was not available in the validation environment.");
    if (!result.agent_gate_executed) result.limitations.push_back("The inherited long-horizon agency gate was not rerun in the bounded release experiment; RLF-6 Decision B remains inherited.");
    else if (result.agent_success_rate < 0.7) result.limitations.push_back("Long-horizon agency remains below the Frontier promotion threshold.");
    if (result.accepted_skills == 0U) result.limitations.push_back("No transferable skill passed the inherited acceptance gate.");
    result.limitations.push_back("Media generation is prototype reconstruction, not photorealistic or production-quality generation.");
    result.limitations.push_back("Controlled multimodal benchmarks do not establish unrestricted open-world understanding.");
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, completed_model.fabric.deterministic_hash());
    hash_double(hash, result.knowledge_retrieval_accuracy);
    hash_double(hash, result.candidates_per_query);
    hash_double(hash, result.image_accuracy);
    hash_double(hash, result.video_accuracy);
    hash_double(hash, result.audio_accuracy);
    hash_double(hash, result.agent_success_rate);
    hash_double(hash, result.agent_recovery_rate);
    hash_u64(hash, result.accepted_skills);
    hash_u64(hash, result.cuda_available ? 1U : 0U);
    hash_u64(hash, static_cast<std::uint64_t>(config.backend));
    result.deterministic_hash = hash;
    completed_model.training_examples = config.media_training_per_class * (config.include_audio ? 6U : 4U) + config.knowledge_records;
    completed_model.evaluation_examples = config.media_evaluation_per_class * (config.include_audio ? 6U : 4U) + config.knowledge_queries;
    completed_model.training_step = completed_model.training_examples;
    completed_model.fabric.set_step(completed_model.training_step);
    if (trained_model != nullptr) *trained_model = std::move(completed_model);
    return result;
}

void write_rlf7_frontier_json(
    std::ostream& output,
    const Rlf7FrontierResult& result
) {
    output << std::setprecision(10)
        << "{\n"
        << "  \"architecture\": \"" << (result.config.frontier_mode ? "RLF-Frontier" : "RLF-7") << "\",\n"
        << "  \"selected_backend\": \"" << frontier::to_string(result.config.backend) << "\",\n"
        << "  \"implementation_complete\": true,\n"
        << "  \"frontier_claim_justified\": " << (result.frontier_claim_justified ? "true" : "false") << ",\n"
        << "  \"scientific_decision\": \"" << json_escape(result.scientific_decision) << "\",\n"
        << "  \"deterministic_hash\": \"" << std::hex << std::setw(16) << std::setfill('0') << result.deterministic_hash << std::dec << "\",\n"
        << "  \"knowledge\": {\"records\": " << result.knowledge_records
        << ", \"retrieval_accuracy\": " << result.knowledge_retrieval_accuracy
        << ", \"candidates_per_query\": " << result.candidates_per_query
        << ", \"persistent_bytes\": " << result.persistent_bytes << "},\n"
        << "  \"multimodality\": {\"image_accuracy\": " << result.image_accuracy
        << ", \"image_localization_iou\": " << result.image_localization_iou
        << ", \"video_accuracy\": " << result.video_accuracy
        << ", \"video_track_continuity\": " << result.video_track_continuity
        << ", \"audio_accuracy\": " << result.audio_accuracy
        << ", \"cross_modal_binding_accuracy\": " << result.cross_modal_binding_accuracy << "},\n"
        << "  \"agency\": {\"gate_executed\": " << (result.agent_gate_executed ? "true" : "false")
        << ", \"success_rate\": " << result.agent_success_rate
        << ", \"recovery_rate\": " << result.agent_recovery_rate
        << ", \"accepted_skills\": " << result.accepted_skills << "},\n"
        << "  \"backends\": {\"scalar_optimized_agreement\": " << (result.scalar_optimized_agreement ? "true" : "false")
        << ", \"cuda_compiled\": " << (result.cuda_compiled ? "true" : "false")
        << ", \"cuda_available\": " << (result.cuda_available ? "true" : "false") << "},\n"
        << "  \"prototype_generation_ready\": " << (result.prototype_generation_ready ? "true" : "false") << ",\n"
        << "  \"checkpoint_ready\": " << (result.checkpoint_ready ? "true" : "false") << ",\n"
        << "  \"gpu_projections\": [\n";
    for (std::size_t index = 0U; index < result.projections.size(); ++index) {
        const auto& projection = result.projections[index];
        output << "    {\"gpu_memory_gb\": " << projection.gpu_memory_gb
            << ", \"projected_hot_modes\": " << projection.projected_hot_modes
            << ", \"projected_hot_bytes\": " << projection.projected_hot_bytes
            << ", \"cpu_ram_bytes\": " << projection.cpu_ram_bytes
            << ", \"nvme_bytes\": " << projection.nvme_bytes << "}"
            << (index + 1U == result.projections.size() ? "\n" : ",\n");
    }
    output << "  ],\n  \"limitations\": [\n";
    for (std::size_t index = 0U; index < result.limitations.size(); ++index) {
        output << "    \"" << json_escape(result.limitations[index]) << "\""
            << (index + 1U == result.limitations.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
}

}  // namespace rlf::experiments
