#pragma once

#include "rlf/solstice/abstraction_fabric.hpp"
#include "rlf/solstice/continual_learning.hpp"
#include "rlf/solstice/grounding_fabric.hpp"
#include "rlf/solstice/general_fabric.hpp"
#include "rlf/solstice/language_fabric.hpp"
#include "rlf/solstice/tokenizer.hpp"
#include "rlf/solstice/tool_protocol.hpp"
#include "rlf/solstice/video_fabric.hpp"
#include "rlf/solstice/vision_fabric.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rlf::solstice {

struct SolsticeConfig final {
    TokenizerConfig tokenizer;
    HierarchicalLanguageConfig language;
    VisionConfig vision;
    VideoFabricConfig video;
    ToolRouterConfig tool_router;
    AbstractionConfig abstraction;
    ContinualLearningConfig continual;
    GroundingConfig grounding;
    GeneralFabricConfig general;
    std::size_t maximum_tool_result_characters{16'384U};
};

struct SolsticeStats final {
    std::size_t vocabulary_size{};
    std::size_t language_contexts{};
    std::size_t language_episodes{};
    std::size_t visual_modes{};
    std::size_t visual_examples{};
    std::size_t tool_routes{};
    std::size_t reasoning_facts{};
    std::size_t reasoning_rules{};
    std::size_t continual_prototypes{};
    std::size_t grounding_links{};
    std::size_t general_demonstrations{};
    std::size_t preference_examples{};
    std::size_t active_learning_items{};
    std::uint64_t language_tokens_seen{};
    std::uint64_t images_seen{};
    std::size_t video_prototypes{};
    std::uint64_t video_sequences_seen{};
    std::uint64_t video_frames_seen{};
    std::uint64_t deterministic_hash{};
    std::size_t completed_training_shards{};
    std::uint64_t audited_training_records{};
    std::uint64_t audited_training_bytes{};
};

struct TrainingShardRecord final {
    std::string shard_id;
    std::string kind;
    std::string shard_sha256;
    std::string ledger_sha256;
    std::string source_uri;
    std::string license;
    std::uint64_t records{};
    std::uint64_t bytes{};
};

struct SolsticeResponse final {
    std::string text;
    std::optional<VisionAnalysis> vision;
    std::optional<ToolProposal> tool_proposal;
    std::optional<ToolResult> tool_result;
    double uncertainty{1.0};
};

struct SolsticeSnapshot final {
    SolsticeConfig config;
    std::uint64_t seed{};
    TokenizerSnapshot tokenizer;
    HierarchicalLanguageSnapshot language;
    VisionSnapshot vision;
    VideoFabricSnapshot video;
    ToolRouterSnapshot tool_router;
    AbstractionSnapshot abstraction;
    ContinualLearningSnapshot continual;
    GroundingSnapshot grounding;
    GeneralFabricSnapshot general;
    std::vector<TrainingShardRecord> completed_training_shards;
};

class SolsticeModel final {
public:
    explicit SolsticeModel(
        SolsticeConfig config = {},
        std::uint64_t seed = 0x534F4C5354494345ULL
    );

    void bootstrap();
    void train_text_corpus(std::string_view corpus);
    void train_dialogue(
        std::string_view prompt,
        std::string_view response,
        std::string_view grounding = {}
    );
    void train_image(const ImageData& image, std::string_view caption);
    void train_image_reference(const ImageData& image, std::string_view caption);
    void train_image_file(
        const std::filesystem::path& path,
        std::string_view caption,
        ImageLimits limits = {}
    );
    std::uint64_t train_video_sequence(
        std::string_view sequence_id,
        std::string_view prompt,
        double frames_per_second,
        std::span<const ImageData> frames,
        std::span<const std::string> frame_captions = {}
    );
    [[nodiscard]] VideoGeneration generate_video(
        std::string_view prompt,
        std::size_t frame_count
    ) const;
    void train_tool_route(std::string_view request, std::string_view tool_name);
    std::uint64_t train_instruction(
        std::string_view task,
        std::string_view domain,
        std::string_view prompt,
        std::string_view rationale,
        std::string_view response,
        double quality = 1.0
    );
    std::uint64_t train_preference(
        std::string_view prompt,
        std::string_view chosen,
        std::string_view rejected,
        std::string_view feedback = {},
        double weight = 1.0
    );
    std::uint64_t learn_fact(
        std::string_view subject,
        std::string_view relation,
        std::string_view object,
        double confidence = 1.0,
        std::string_view provenance = {}
    );
    std::uint64_t learn_rule(
        std::string_view name,
        std::span<const RelationalPattern> premises,
        const RelationalPattern& conclusion,
        double confidence = 1.0
    );
    [[nodiscard]] SchemaInductionResult induce_chain_rule(
        std::string_view name,
        std::string_view demonstration_subject,
        std::string_view conclusion_relation,
        std::string_view demonstration_object,
        std::size_t maximum_hops = 4U,
        double confidence = 1.0
    );
    [[nodiscard]] std::vector<ReasoningAnswer> reason(
        const RelationalPattern& query,
        std::size_t maximum_answers = 16U
    ) const;
    ContinualPrediction learn_continually(
        std::string_view task,
        std::string_view label,
        std::span<const float> features,
        double sample_weight = 1.0
    );

    [[nodiscard]] SolsticeResponse respond(
        std::string_view prompt,
        const ImageData* image,
        ToolRuntime* tools,
        GenerationSettings settings = {}
    ) const;
    [[nodiscard]] SolsticeResponse respond_file(
        std::string_view prompt,
        const std::optional<std::filesystem::path>& image_path,
        ToolRuntime* tools,
        GenerationSettings settings = {},
        ImageLimits limits = {}
    ) const;

    [[nodiscard]] const SolsticeConfig& config() const noexcept;
    [[nodiscard]] std::uint64_t seed() const noexcept;
    [[nodiscard]] const SolsticeTokenizer& tokenizer() const noexcept;
    [[nodiscard]] const HierarchicalLanguageFabric& language() const noexcept;
    [[nodiscard]] const VisualPatchFabric& vision() const noexcept;
    [[nodiscard]] const VideoPrototypeFabric& video() const noexcept;
    [[nodiscard]] const ToolRouter& tool_router() const noexcept;
    [[nodiscard]] const AbstractionFabric& abstraction() const noexcept;
    [[nodiscard]] const ContinualLearningFabric& continual() const noexcept;
    [[nodiscard]] const CrossModalGroundingFabric& grounding() const noexcept;
    [[nodiscard]] const GeneralInstructionFabric& general() const noexcept;
    [[nodiscard]] const std::vector<TrainingShardRecord>&
        completed_training_shards() const noexcept;
    [[nodiscard]] bool has_completed_training_shard(
        std::string_view shard_sha256
    ) const noexcept;
    void record_completed_training_shard(TrainingShardRecord record);
    void set_backend(rlf::frontier::FrontierBackendKind kind);
    [[nodiscard]] rlf::frontier::BackendOperationStats backend_operation_stats() const noexcept;
    [[nodiscard]] SparseRouterOperationStats sparse_router_operation_stats() const noexcept;
    [[nodiscard]] VisualTrainingOperationStats visual_training_operation_stats() const noexcept;
    [[nodiscard]] GroundingOperationStats grounding_operation_stats() const noexcept;
    [[nodiscard]] LanguageTrainingOperationStats language_training_operation_stats() const noexcept;
    [[nodiscard]] GeneralTrainingOperationStats general_training_operation_stats() const noexcept;
    [[nodiscard]] rlf::frontier::FrontierBackendKind backend_kind() const noexcept;
    [[nodiscard]] SolsticeStats stats() const noexcept;
    [[nodiscard]] std::uint64_t deterministic_hash() const noexcept;

    [[nodiscard]] SolsticeSnapshot snapshot() const;
    [[nodiscard]] static SolsticeModel from_snapshot(SolsticeSnapshot snapshot);

private:
    void ensure_tokenizer_trained(std::string_view corpus);
    [[nodiscard]] std::string classify_task(std::string_view prompt) const;
    [[nodiscard]] std::string classify_domain(std::string_view prompt) const;
    [[nodiscard]] std::string retrieve_knowledge(
        std::string_view prompt,
        std::size_t maximum_facts = 16U
    ) const;
    [[nodiscard]] std::string compose_tool_answer(
        const ToolProposal& proposal,
        const ToolResult& result
    ) const;

    SolsticeConfig config_;
    std::uint64_t seed_{};
    SolsticeTokenizer tokenizer_;
    HierarchicalLanguageFabric language_;
    VisualPatchFabric vision_;
    VideoPrototypeFabric video_;
    ToolRouter tool_router_;
    AbstractionFabric abstraction_;
    ContinualLearningFabric continual_;
    CrossModalGroundingFabric grounding_;
    GeneralInstructionFabric general_;
    std::vector<TrainingShardRecord> completed_training_shards_;
};

}  // namespace rlf::solstice
