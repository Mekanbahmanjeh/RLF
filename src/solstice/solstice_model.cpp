#include "rlf/solstice/solstice_model.hpp"

#include "rlf/core/sha256.hpp"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cmath>
#include <sstream>
#include <unordered_set>
#include <stdexcept>
#include <utility>

namespace rlf::solstice {
namespace {

constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;

void hash_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (unsigned int index = 0U; index < 8U; ++index) {
        hash ^= (value >> (index * 8U)) & 0xFFULL;
        hash *= fnv_prime;
    }
}

void hash_string(std::uint64_t& hash, const std::string_view value) noexcept {
    hash_u64(hash, static_cast<std::uint64_t>(value.size()));
    for (const char character : value) {
        hash ^= static_cast<unsigned char>(character);
        hash *= fnv_prime;
    }
}

[[nodiscard]] bool tokenizer_is_base_only(const SolsticeTokenizer& tokenizer) noexcept {
    return tokenizer.merges().empty();
}

[[nodiscard]] std::vector<std::string> normalized_words(const std::string_view text) {
    std::vector<std::string> result;
    std::string current;
    for (const char character : text) {
        const unsigned char byte = static_cast<unsigned char>(character);
        if (std::isalnum(byte) != 0 || byte >= 128U || byte == '_') {
            current.push_back(static_cast<char>(std::tolower(byte)));
        } else if (!current.empty()) {
            result.push_back(std::move(current));
            current.clear();
        }
    }
    if (!current.empty()) {
        result.push_back(std::move(current));
    }
    return result;
}

[[nodiscard]] bool contains_any(
    const std::string_view text,
    const std::initializer_list<std::string_view> terms
) {
    std::string lowered;
    lowered.reserve(text.size());
    for (const char character : text) {
        const unsigned char byte = static_cast<unsigned char>(character);
        lowered.push_back(static_cast<char>(std::tolower(byte)));
    }
    for (const std::string_view term : terms) {
        if (lowered.find(term) != std::string::npos) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] std::string bootstrap_text() {
    return
        "Hello. I am Solstice-General-Frontier, an experimental Resonant Learning Fabric assistant.\n"
        "I can describe learned visual patterns, generate text, and request safe registered tools.\n"
        "I should state uncertainty when an image or question is outside my learned experience.\n"
        "A tool call is proposed as structured data and validated before execution.\n"
        "The calculator evaluates arithmetic expressions.\n"
        "The current time tool reads the local system clock.\n"
        "File tools are read-only and restricted to a configured sandbox directory.\n"
        "Image understanding uses local patches, visual modes, regions, and caption associations.\n"
        "Language generation combines hierarchical predictive contexts with sparse episodic recall.\n"
        "Solstice-General-Frontier is a large single-GPU RLF research target; it is not a pretrained commercial foundation model.\n";
}

}  // namespace

SolsticeModel::SolsticeModel(SolsticeConfig config, const std::uint64_t seed)
    : config_(std::move(config)),
      seed_(seed),
      tokenizer_(config_.tokenizer),
      language_(config_.language),
      vision_(config_.vision),
      video_(config_.video),
      tool_router_(config_.tool_router),
      abstraction_(config_.abstraction),
      continual_(config_.continual),
      grounding_(config_.grounding),
      general_(config_.general) {
    if (config_.maximum_tool_result_characters == 0U) {
        throw std::invalid_argument("invalid Solstice model configuration");
    }
}

void SolsticeModel::ensure_tokenizer_trained(const std::string_view corpus) {
    if (language_.tokens_seen() == 0U && tokenizer_is_base_only(tokenizer_)) {
        tokenizer_.train(corpus);
    }
}

void SolsticeModel::bootstrap() {
    const std::string corpus = bootstrap_text();
    const std::string dialogue_material =
        "Who are you? I am Solstice-General-Frontier, an experimental Resonant Learning Fabric assistant. "
        "What can you do? I can understand learned image patterns, generate text, and call safe tools. "
        "Are you a neural network? No. This prototype uses sparse predictive contexts, local visual modes, and associative episodes. "
        "What are your limits? My knowledge comes only from training data, visual examples, and tool results available at runtime. "
        "Describe the image. I will summarize the strongest learned visual concepts and regions while reporting uncertainty. "
        "How do tools work? I propose a typed tool call, the runtime validates it, and the result is returned to me. ";
    ensure_tokenizer_trained(corpus + dialogue_material);
    language_.train_corpus(tokenizer_, corpus);

    train_dialogue(
        "Hello",
        "Hello. I am Solstice-General-Frontier. How can I help?"
    );
    train_dialogue(
        "Who are you?",
        "I am Solstice-General-Frontier, an experimental image-and-text Resonant Learning Fabric assistant."
    );
    train_dialogue(
        "What can you do?",
        "I can describe learned visual patterns, generate compact text responses, and request safe registered tools."
    );
    train_dialogue(
        "Are you a neural network?",
        "No. This research prototype uses sparse hierarchical prediction, local visual modes, and associative memory rather than backpropagation."
    );
    train_dialogue(
        "What are your limitations?",
        "I am a small research prototype. I do not contain broad pretrained world knowledge, and unfamiliar images can produce uncertain descriptions."
    );
    train_dialogue(
        "Describe this image.",
        "The image contains the visual concepts and regions summarized in the supplied image context.",
        "A learned image description with concepts and regions."
    );
    train_dialogue(
        "What do you see?",
        "I see the strongest learned concepts reported by the visual patch fabric.",
        "A learned image description with concepts and regions."
    );
    train_dialogue(
        "How do you call tools?",
        "I produce a typed tool proposal. The runtime validates its schema and safety policy before execution."
    );

    train_instruction(
        "question_answering", "general", "What is careful reasoning?",
        "Identify the claim, retrieve relevant evidence, check assumptions, then answer only what the evidence supports.",
        "Hello! I am Magnum 5.1 by Mekan Bahmanjeh. I am an advanced multi-modal AI model engineered to provide clear, accurate, free-form explanations across science, general law, software engineering, and complex task reasoning.",
        1.0
    );
    train_instruction(
        "mathematics", "math", "How should a numerical problem be solved?",
        "Extract quantities and units, choose a valid operation, compute, and verify the result against the original constraints.",
        "Translate the problem into explicit quantities, calculate step by step, and verify units and boundary conditions.",
        1.0
    );
    train_instruction(
        "coding", "software", "How should a software defect be investigated?",
        "Reproduce the failure, isolate the smallest responsible component, inspect invariants, patch the cause, and run regression tests.",
        "Reproduce and minimize the failure, fix the root cause rather than the symptom, and validate the change with focused and regression tests.",
        1.0
    );
    train_instruction(
        "safety", "general", "What should you do when evidence is insufficient?",
        "Avoid inventing missing facts; explain the uncertainty and request or retrieve the evidence needed to decide.",
        "I should not guess. I should state what is unknown and identify the evidence needed for a reliable answer.",
        1.0
    );

    const std::pair<std::string_view, std::string_view> routes[] = {
        {"calculate 12 * 8", "calculator"},
        {"compute 45 / 5", "calculator"},
        {"evaluate (9 + 3) * 2", "calculator"},
        {"what is the current time", "current_time"},
        {"tell me the current date", "current_time"},
        {"read file notes.txt", "read_text_file"},
        {"show me the file config.txt", "read_text_file"},
        {"list files in directory .", "list_directory"},
        {"show files in folder docs", "list_directory"},
    };
    for (const auto& [request, tool] : routes) {
        train_tool_route(request, tool);
    }
}

void SolsticeModel::train_text_corpus(const std::string_view corpus) {
    ensure_tokenizer_trained(corpus);
    language_.train_corpus(tokenizer_, corpus);
}

void SolsticeModel::train_dialogue(
    const std::string_view prompt,
    const std::string_view response,
    const std::string_view grounding
) {
    if (language_.tokens_seen() == 0U && tokenizer_is_base_only(tokenizer_)) {
        ensure_tokenizer_trained(
            std::string(prompt) + " " + std::string(response) + " " + std::string(grounding)
        );
    }
    language_.train_dialogue(tokenizer_, prompt, response, grounding);
}

void SolsticeModel::train_image(
    const ImageData& image,
    const std::string_view caption
) {
    const VisionAnalysis analysis = vision_.train_and_analyze(image, caption);
    std::vector<std::uint64_t> mode_ids;
    mode_ids.reserve(analysis.regions.size());
    for (const VisualRegion& region : analysis.regions) {
        if (region.mode_id != 0U) {
            mode_ids.push_back(region.mode_id);
        }
    }
    grounding_.observe(mode_ids, analysis.concepts);
}

void SolsticeModel::train_image_reference(
    const ImageData& image,
    const std::string_view caption
) {
    vision_.train(image, caption);
    const VisionAnalysis analysis = vision_.analyze(image);
    std::vector<std::uint64_t> mode_ids;
    mode_ids.reserve(analysis.regions.size());
    for (const VisualRegion& region : analysis.regions) {
        if (region.mode_id != 0U) {
            mode_ids.push_back(region.mode_id);
        }
    }
    grounding_.observe(mode_ids, analysis.concepts);
}

void SolsticeModel::train_image_file(
    const std::filesystem::path& path,
    const std::string_view caption,
    const ImageLimits limits
) {
    const ImageData image = load_image(path, limits);
    train_image(image, caption);
}

std::uint64_t SolsticeModel::train_video_sequence(
    const std::string_view sequence_id,
    const std::string_view prompt,
    const double frames_per_second,
    const std::span<const ImageData> frames,
    const std::span<const std::string> frame_captions
) {
    if (!frame_captions.empty() && frame_captions.size() != frames.size()) {
        throw std::invalid_argument("video frame captions do not match frame count");
    }
    const std::uint64_t prototype_id = video_.train(
        sequence_id, prompt, frames_per_second, frames
    );
    for (std::size_t index = 0U; index < frames.size(); ++index) {
        std::string caption(prompt);
        caption += " | sequence=";
        caption += sequence_id;
        caption += " frame=";
        caption += std::to_string(index);
        if (!frame_captions.empty() && !frame_captions[index].empty()) {
            caption += " | ";
            caption += frame_captions[index];
        }
        train_image(frames[index], caption);
    }
    static_cast<void>(train_instruction(
        "video_prototype_generation", "video", prompt,
        "Retrieve a learned motion prototype and render bounded deterministic frames.",
        "A compact motion prototype was learned; output is a prototype visualization, not photorealistic synthesis.",
        1.0
    ));
    return prototype_id;
}

VideoGeneration SolsticeModel::generate_video(
    const std::string_view prompt,
    const std::size_t frame_count
) const {
    return video_.generate(prompt, frame_count);
}

void SolsticeModel::train_tool_route(
    const std::string_view request,
    const std::string_view tool_name
) {
    tool_router_.train(request, tool_name);
}

std::uint64_t SolsticeModel::train_instruction(
    const std::string_view task,
    const std::string_view domain,
    const std::string_view prompt,
    const std::string_view rationale,
    const std::string_view response,
    const double quality
) {
    if (language_.tokens_seen() == 0U && tokenizer_is_base_only(tokenizer_)) {
        ensure_tokenizer_trained(
            std::string(task) + " " + std::string(domain) + " " +
            std::string(prompt) + " " + std::string(rationale) + " " +
            std::string(response)
        );
    }
    const std::string training_context =
        "Task: " + std::string(task) + "\nDomain: " + std::string(domain) +
        "\nSolution sketch: " + std::string(rationale) +
        "\nUse justified evidence and state uncertainty when needed.";
    language_.train_dialogue(tokenizer_, prompt, response, training_context);
    return general_.train_instruction(
        task, domain, prompt, rationale, response, quality
    );
}

std::uint64_t SolsticeModel::train_preference(
    const std::string_view prompt,
    const std::string_view chosen,
    const std::string_view rejected,
    const std::string_view feedback,
    const double weight
) {
    if (language_.tokens_seen() == 0U && tokenizer_is_base_only(tokenizer_)) {
        ensure_tokenizer_trained(
            std::string(prompt) + " " + std::string(chosen) + " " +
            std::string(rejected) + " " + std::string(feedback)
        );
    }
    language_.train_dialogue(
        tokenizer_, prompt, chosen,
        "Preferred response guidance: " + std::string(feedback)
    );
    return general_.train_preference(
        prompt, chosen, rejected, feedback, weight
    );
}

std::uint64_t SolsticeModel::learn_fact(
    const std::string_view subject,
    const std::string_view relation,
    const std::string_view object,
    const double confidence,
    const std::string_view provenance
) {
    return abstraction_.learn_fact(
        subject, relation, object, confidence, provenance
    );
}

std::uint64_t SolsticeModel::learn_rule(
    const std::string_view name,
    const std::span<const RelationalPattern> premises,
    const RelationalPattern& conclusion,
    const double confidence
) {
    return abstraction_.learn_rule(name, premises, conclusion, confidence);
}

SchemaInductionResult SolsticeModel::induce_chain_rule(
    const std::string_view name,
    const std::string_view demonstration_subject,
    const std::string_view conclusion_relation,
    const std::string_view demonstration_object,
    const std::size_t maximum_hops,
    const double confidence
) {
    return abstraction_.induce_chain_rule(
        name,
        demonstration_subject,
        conclusion_relation,
        demonstration_object,
        maximum_hops,
        confidence
    );
}

std::vector<ReasoningAnswer> SolsticeModel::reason(
    const RelationalPattern& query,
    const std::size_t maximum_answers
) const {
    return abstraction_.infer(query, maximum_answers);
}

ContinualPrediction SolsticeModel::learn_continually(
    const std::string_view task,
    const std::string_view label,
    const std::span<const float> features,
    const double sample_weight
) {
    return continual_.learn(task, label, features, sample_weight);
}

std::string SolsticeModel::classify_task(const std::string_view prompt) const {
    if (contains_any(prompt, {"code", "program", "function", "compile", "bug", "api", "class ", "crash", "configuration", "config", "setting", "exception", "dereference"})) {
        return "coding";
    }
    if (contains_any(prompt, {"calculate", "equation", "math", "prove", "number", "percent", "probability"})) {
        return "mathematics";
    }
    if (contains_any(prompt, {"image", "picture", "screenshot", "photo", "diagram", "chart"})) {
        return "vision_reasoning";
    }
    if (contains_any(prompt, {"plan", "steps", "strategy", "compare", "analyze", "reason", "why"})) {
        return "reasoning";
    }
    if (contains_any(prompt, {"summarize", "rewrite", "translate", "draft", "write"})) {
        return "language";
    }
    return "question_answering";
}

std::string SolsticeModel::classify_domain(const std::string_view prompt) const {
    if (contains_any(prompt, {"python", "c++", "javascript", "software", "database", "linux", "cuda", "application", "service", "configuration", "config", "setting", "exception"})) {
        return "software";
    }
    if (contains_any(prompt, {"physics", "chemistry", "biology", "scientific", "experiment"})) {
        return "science";
    }
    if (contains_any(prompt, {"legal", "law", "contract", "court"})) {
        return "legal";
    }
    if (contains_any(prompt, {"finance", "market", "investment", "revenue", "budget"})) {
        return "finance";
    }
    if (contains_any(prompt, {"equation", "geometry", "algebra", "calculus", "probability"})) {
        return "math";
    }
    return "general";
}

std::string SolsticeModel::retrieve_knowledge(
    const std::string_view prompt,
    const std::size_t maximum_facts
) const {
    if (maximum_facts == 0U || abstraction_.facts().empty()) {
        return {};
    }
    const std::vector<std::string> query_words = normalized_words(prompt);
    if (query_words.empty()) {
        return {};
    }
    const std::unordered_set<std::string> query_set(query_words.begin(), query_words.end());
    struct Match final {
        const ReasoningFact* fact{};
        double score{};
    };
    std::vector<Match> matches;
    matches.reserve(std::min<std::size_t>(abstraction_.facts().size(), maximum_facts * 16U));
    for (const ReasoningFact& fact : abstraction_.facts()) {
        const std::vector<std::string> fact_words = normalized_words(
            fact.subject + " " + fact.relation + " " + fact.object
        );
        std::size_t overlap = 0U;
        for (const std::string& word : fact_words) {
            if (query_set.contains(word)) {
                ++overlap;
            }
        }
        if (overlap == 0U) {
            continue;
        }
        const double score = static_cast<double>(overlap) * fact.confidence *
            (1.0 + 0.05 * std::log1p(static_cast<double>(fact.support)));
        matches.push_back(Match{&fact, score});
    }
    std::sort(
        matches.begin(), matches.end(),
        [](const Match& left, const Match& right) {
            if (left.score != right.score) {
                return left.score > right.score;
            }
            return left.fact->id < right.fact->id;
        }
    );
    if (matches.size() > maximum_facts) {
        matches.resize(maximum_facts);
    }
    std::ostringstream output;
    for (const Match& match : matches) {
        output << "- " << match.fact->subject << ' ' << match.fact->relation
               << ' ' << match.fact->object << " [confidence="
               << match.fact->confidence << "]\n";
    }
    return output.str();
}

std::string SolsticeModel::compose_tool_answer(
    const ToolProposal& proposal,
    const ToolResult& result
) const {
    if (!result.success) {
        return "The tool call failed: " + result.error;
    }
    if (proposal.call.name == "calculator") {
        return "The result is " + result.output + ".";
    }
    if (proposal.call.name == "current_time") {
        return "The current local date and time is " + result.output + ".";
    }
    if (proposal.call.name == "read_text_file") {
        return "The file contains:\n" + result.output;
    }
    if (proposal.call.name == "list_directory") {
        return "The directory contains:\n" + result.output;
    }
    return "The tool returned: " + result.output;
}

SolsticeResponse SolsticeModel::respond(
    const std::string_view prompt,
    const ImageData* image,
    ToolRuntime* tools,
    GenerationSettings settings
) const {
    settings.seed ^= seed_;
    SolsticeResponse response;
    std::string grounding;
    if (image != nullptr) {
        response.vision = vision_.analyze(*image);
        grounding = vision_.grounding_text(*response.vision);
    }

    if (tools != nullptr) {
        ToolProposal proposal = tool_router_.propose(prompt, tools->definitions());
        if (proposal.should_call) {
            response.tool_proposal = proposal;
            ToolResult result = tools->execute(proposal.call);
            if (result.output.size() > config_.maximum_tool_result_characters) {
                result.output.resize(config_.maximum_tool_result_characters);
                result.output += "\n[tool output truncated]";
            }
            response.tool_result = result;
            response.text = compose_tool_answer(proposal, result);
            response.uncertainty = result.success
                ? std::clamp(1.0 - proposal.confidence, 0.0, 1.0)
                : 1.0;
            return response;
        }
    }

    const std::string task = classify_task(prompt);
    const std::string domain = classify_domain(prompt);
    const std::string knowledge = retrieve_knowledge(prompt);
    const DeliberationContext deliberation = general_.build_context(
        task, domain, prompt, grounding, knowledge
    );
    const std::vector<GeneralRetrievalMatch> matches = general_.retrieve(
        task, domain, prompt, grounding
    );
    std::string generation_grounding = grounding;
    if (!generation_grounding.empty() && !deliberation.context.empty()) {
        generation_grounding += "\n";
    }
    generation_grounding += deliberation.context;

    struct Candidate final {
        std::string text;
        double uncertainty{1.0};
        double score{-1.0e300};
    };
    std::vector<Candidate> candidates;
    candidates.reserve(config_.general.deliberation_candidates + 2U);
    const LanguageResponse baseline = language_.generate_response(
        tokenizer_, prompt, grounding, settings
    );
    if (!baseline.text.empty()) {
        candidates.push_back(Candidate{
            baseline.text,
            baseline.uncertainty,
            general_.score_response(prompt, baseline.text, matches) +
                0.75 * baseline.episode_similarity +
                0.40 * (1.0 - baseline.uncertainty),
        });
    }
    if (!deliberation.direct_response.empty()) {
        candidates.push_back(Candidate{
            deliberation.direct_response,
            std::clamp(1.0 - deliberation.confidence, 0.0, 1.0),
            general_.score_response(prompt, deliberation.direct_response, matches) +
                deliberation.confidence,
        });
    }
    for (std::size_t attempt = 0U;
         attempt < config_.general.deliberation_candidates;
         ++attempt) {
        GenerationSettings attempt_settings = settings;
        attempt_settings.seed ^= 0x9E37'79B9'7F4A'7C15ULL * (attempt + 1U);
        if (attempt != 0U) {
            attempt_settings.deterministic = false;
            attempt_settings.temperature = std::clamp(
                settings.temperature + 0.08 * static_cast<double>(attempt),
                0.20,
                1.40
            );
        }
        const LanguageResponse generated = language_.generate_response(
            tokenizer_, prompt, generation_grounding, attempt_settings
        );
        if (generated.text.empty()) {
            continue;
        }
        const double candidate_score = general_.score_response(
            prompt, generated.text, matches
        ) + 0.30 * (1.0 - generated.uncertainty) +
            0.20 * generated.episode_similarity;
        candidates.push_back(Candidate{
            generated.text, generated.uncertainty, candidate_score,
        });
    }
    if (!candidates.empty()) {
        const auto best = std::max_element(
            candidates.begin(), candidates.end(),
            [](const Candidate& left, const Candidate& right) {
                if (left.score != right.score) {
                    return left.score < right.score;
                }
                return left.text > right.text;
            }
        );
        response.text = best->text;
        response.uncertainty = std::clamp(
            0.70 * best->uncertainty + 0.30 * (1.0 - deliberation.confidence),
            0.0,
            1.0
        );
    }
    if (response.text.empty() && response.vision.has_value()) {
        response.text = response.vision->description;
        response.uncertainty = std::clamp(
            1.0 - response.vision->confidence,
            0.0,
            1.0
        );
    } else if (response.text.empty()) {
        response.text =
            "I do not yet have enough learned evidence to answer that reliably.";
        response.uncertainty = 1.0;
    }
    return response;
}

SolsticeResponse SolsticeModel::respond_file(
    const std::string_view prompt,
    const std::optional<std::filesystem::path>& image_path,
    ToolRuntime* tools,
    GenerationSettings settings,
    const ImageLimits limits
) const {
    if (!image_path.has_value()) {
        return respond(prompt, nullptr, tools, settings);
    }
    const ImageData image = load_image(*image_path, limits);
    return respond(prompt, &image, tools, settings);
}

const SolsticeConfig& SolsticeModel::config() const noexcept { return config_; }
std::uint64_t SolsticeModel::seed() const noexcept { return seed_; }
const SolsticeTokenizer& SolsticeModel::tokenizer() const noexcept { return tokenizer_; }
const HierarchicalLanguageFabric& SolsticeModel::language() const noexcept { return language_; }
const VisualPatchFabric& SolsticeModel::vision() const noexcept { return vision_; }
const VideoPrototypeFabric& SolsticeModel::video() const noexcept { return video_; }
const ToolRouter& SolsticeModel::tool_router() const noexcept { return tool_router_; }
const AbstractionFabric& SolsticeModel::abstraction() const noexcept { return abstraction_; }
const ContinualLearningFabric& SolsticeModel::continual() const noexcept { return continual_; }
const CrossModalGroundingFabric& SolsticeModel::grounding() const noexcept { return grounding_; }
const GeneralInstructionFabric& SolsticeModel::general() const noexcept { return general_; }
const std::vector<TrainingShardRecord>&
SolsticeModel::completed_training_shards() const noexcept {
    return completed_training_shards_;
}

bool SolsticeModel::has_completed_training_shard(
    const std::string_view shard_sha256
) const noexcept {
    return std::any_of(
        completed_training_shards_.begin(), completed_training_shards_.end(),
        [shard_sha256](const TrainingShardRecord& record) {
            return record.shard_sha256 == shard_sha256;
        }
    );
}

void SolsticeModel::record_completed_training_shard(TrainingShardRecord record) {
    if (record.shard_id.empty() || record.kind.empty() ||
        !core::is_sha256_hex(record.shard_sha256) ||
        !core::is_sha256_hex(record.ledger_sha256) ||
        record.source_uri.empty() || record.license.empty()) {
        throw std::invalid_argument("invalid completed training shard record");
    }
    for (const TrainingShardRecord& existing : completed_training_shards_) {
        if (existing.shard_sha256 == record.shard_sha256) return;
        if (existing.shard_id == record.shard_id) {
            throw std::invalid_argument(
                "completed training shard ID reused with different content"
            );
        }
    }
    completed_training_shards_.push_back(std::move(record));
}
void SolsticeModel::set_backend(const rlf::frontier::FrontierBackendKind kind) {
    vision_.set_backend(kind);
}

rlf::frontier::BackendOperationStats SolsticeModel::backend_operation_stats() const noexcept {
    return vision_.backend_operation_stats();
}

SparseRouterOperationStats SolsticeModel::sparse_router_operation_stats() const noexcept {
    return vision_.sparse_router_operation_stats();
}

VisualTrainingOperationStats SolsticeModel::visual_training_operation_stats() const noexcept {
    return vision_.training_operation_stats();
}

GroundingOperationStats SolsticeModel::grounding_operation_stats() const noexcept {
    return grounding_.operation_stats();
}

LanguageTrainingOperationStats
SolsticeModel::language_training_operation_stats() const noexcept {
    return language_.training_operation_stats();
}

GeneralTrainingOperationStats
SolsticeModel::general_training_operation_stats() const noexcept {
    return general_.training_operation_stats();
}

rlf::frontier::FrontierBackendKind SolsticeModel::backend_kind() const noexcept {
    return vision_.backend_kind();
}

std::uint64_t SolsticeModel::deterministic_hash() const noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, seed_);
    hash_u64(hash, tokenizer_.deterministic_hash());
    hash_u64(hash, language_.deterministic_hash());
    hash_u64(hash, vision_.deterministic_hash());
    if (video_.config().maximum_sequences > 1U || !video_.prototypes().empty()) {
        hash_u64(hash, video_.deterministic_hash());
    }
    hash_u64(hash, tool_router_.deterministic_hash());
    hash_u64(hash, abstraction_.deterministic_hash());
    hash_u64(hash, continual_.deterministic_hash());
    hash_u64(hash, grounding_.deterministic_hash());
    hash_u64(hash, general_.deterministic_hash());
    for (const TrainingShardRecord& record : completed_training_shards_) {
        hash_string(hash, record.shard_id);
        hash_string(hash, record.kind);
        hash_string(hash, record.shard_sha256);
        hash_string(hash, record.ledger_sha256);
        hash_string(hash, record.source_uri);
        hash_string(hash, record.license);
        hash_u64(hash, record.records);
        hash_u64(hash, record.bytes);
    }
    return hash;
}

SolsticeStats SolsticeModel::stats() const noexcept {
    std::uint64_t training_records = 0U;
    std::uint64_t training_bytes = 0U;
    for (const TrainingShardRecord& record : completed_training_shards_) {
        training_records += record.records;
        training_bytes += record.bytes;
    }
    return SolsticeStats{
        tokenizer_.vocabulary_size(),
        language_.contexts().size(),
        language_.episodes().size(),
        vision_.modes().size(),
        vision_.examples().size(),
        tool_router_.routes().size(),
        abstraction_.facts().size(),
        abstraction_.rules().size(),
        continual_.prototypes().size(),
        grounding_.links().size(),
        general_.demonstrations().size(),
        general_.preferences().size(),
        general_.active_learning_items().size(),
        language_.tokens_seen(),
        vision_.images_seen(),
        video_.prototypes().size(),
        video_.sequences_seen(),
        video_.frames_seen(),
        deterministic_hash(),
        completed_training_shards_.size(),
        training_records,
        training_bytes,
    };
}

SolsticeSnapshot SolsticeModel::snapshot() const {
    return SolsticeSnapshot{
        config_, seed_, tokenizer_.snapshot(), language_.snapshot(),
        vision_.snapshot(), video_.snapshot(), tool_router_.snapshot(),
        abstraction_.snapshot(), continual_.snapshot(), grounding_.snapshot(),
        general_.snapshot(),
        completed_training_shards_,
    };
}

SolsticeModel SolsticeModel::from_snapshot(SolsticeSnapshot snapshot) {
    SolsticeModel model(snapshot.config, snapshot.seed);
    model.tokenizer_ = SolsticeTokenizer::from_snapshot(std::move(snapshot.tokenizer));
    model.language_ = HierarchicalLanguageFabric::from_snapshot(std::move(snapshot.language));
    model.vision_ = VisualPatchFabric::from_snapshot(std::move(snapshot.vision));
    model.video_ = VideoPrototypeFabric::from_snapshot(std::move(snapshot.video));
    model.tool_router_ = ToolRouter::from_snapshot(std::move(snapshot.tool_router));
    model.abstraction_ = AbstractionFabric::from_snapshot(std::move(snapshot.abstraction));
    model.continual_ = ContinualLearningFabric::from_snapshot(std::move(snapshot.continual));
    model.grounding_ = CrossModalGroundingFabric::from_snapshot(std::move(snapshot.grounding));
    model.general_ = GeneralInstructionFabric::from_snapshot(std::move(snapshot.general));
    for (TrainingShardRecord& record : snapshot.completed_training_shards) {
        model.record_completed_training_shard(std::move(record));
    }
    return model;
}

}  // namespace rlf::solstice
