#pragma once

#include "rlf/frontier/frontier_backend.hpp"
#include "rlf/solstice/language_fabric.hpp"
#include "rlf/solstice/profile.hpp"
#include "rlf/solstice/solstice_model.hpp"
#include "rlf/solstice/tool_protocol.hpp"
#include "rlf/solstice/vision_fabric.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rlf::sdk {

inline constexpr std::string_view bundle_manifest_filename = "rlf-bundle.conf";

class ChatSession;
class Trainer;

enum class PipelineTask : std::uint8_t {
    text_generation = 0U,
    image_text_to_text = 1U,
    tool_use = 2U,
};

struct BundleManifest final {
    std::uint32_t format_version{1U};
    std::string name;
    std::string architecture{"solstice"};
    std::filesystem::path checkpoint;
    std::string checkpoint_sha256;
    std::optional<solstice::SolsticeProfile> profile;
    std::vector<PipelineTask> tasks;
    std::string license;
};

struct LoadOptions final {
    frontier::FrontierBackendKind backend{
        frontier::FrontierBackendKind::optimized_cpu
    };
    std::optional<solstice::SolsticeProfile> profile;
    bool verify_checkpoint_sha256{true};
    bool enforce_profile{true};
};

struct ContextCapabilities final {
    std::size_t maximum_predictive_context_tokens{};
    std::size_t maximum_episode_cue_tokens{};
    std::size_t maximum_generation_tokens{};
    std::size_t maximum_retrieval_context_characters{};
};

struct ModelInfo final {
    std::string name;
    std::string architecture;
    std::filesystem::path checkpoint;
    std::string checkpoint_sha256;
    std::uint64_t checkpoint_bytes{};
    std::optional<solstice::SolsticeProfile> profile;
    std::vector<PipelineTask> tasks;
    std::string license;
    ContextCapabilities context;
    solstice::SolsticeStats stats;
};

struct SaveOptions final {
    std::string name{"rlf-model"};
    std::filesystem::path checkpoint_filename{"model.rlfsp"};
    std::optional<solstice::SolsticeProfile> profile;
    std::vector<PipelineTask> tasks{
        PipelineTask::text_generation,
        PipelineTask::image_text_to_text,
        PipelineTask::tool_use,
    };
    std::string license;
};

class AutoModel final {
public:
    [[nodiscard]] static AutoModel from_pretrained(
        const std::filesystem::path& path,
        LoadOptions options = {}
    );
    [[nodiscard]] static AutoModel from_profile(
        solstice::SolsticeProfile profile,
        LoadOptions options = {},
        std::uint64_t seed = 0x534F4C5354494345ULL,
        bool bootstrap = true
    );

    [[nodiscard]] const solstice::SolsticeModel& model() const noexcept;
    [[nodiscard]] const ModelInfo& info() const noexcept;
    [[nodiscard]] bool supports(PipelineTask task) const noexcept;
    [[nodiscard]] std::size_t token_count(std::string_view text) const;
    void save_pretrained(
        const std::filesystem::path& directory,
        SaveOptions options = {}
    );

private:
    struct State;

    AutoModel(
        std::shared_ptr<solstice::SolsticeModel> model,
        ModelInfo info
    );
    [[nodiscard]] solstice::SolsticeModel& mutable_model() noexcept;
    void refresh_info();

    std::shared_ptr<State> state_;

    friend class Pipeline;
    friend class ChatSession;
    friend class Trainer;
};

struct PipelineOptions final {
    solstice::GenerationSettings generation;
    solstice::ImageLimits image_limits;
    solstice::ToolPolicy tool_policy;
    bool register_safe_tools{false};
};

struct PipelineRequest final {
    std::string prompt;
    std::optional<std::filesystem::path> image;
};

struct PipelineOutput final {
    std::string text;
    std::optional<solstice::VisionAnalysis> vision;
    std::optional<solstice::ToolProposal> tool_proposal;
    std::optional<solstice::ToolResult> tool_result;
    double uncertainty{1.0};
};

class Pipeline final {
public:
    Pipeline(
        PipelineTask task,
        AutoModel model,
        PipelineOptions options = {}
    );

    [[nodiscard]] PipelineOutput operator()(const PipelineRequest& request) const;
    [[nodiscard]] PipelineOutput operator()(std::string_view prompt) const;
    [[nodiscard]] PipelineTask task() const noexcept;
    [[nodiscard]] const ModelInfo& model_info() const noexcept;

private:
    PipelineTask task_;
    AutoModel model_;
    PipelineOptions options_;
    mutable std::optional<solstice::ToolRuntime> tools_;

    friend class ChatSession;
};

[[nodiscard]] Pipeline make_pipeline(
    PipelineTask task,
    const std::filesystem::path& pretrained,
    LoadOptions load_options = {},
    PipelineOptions pipeline_options = {}
);
[[nodiscard]] BundleManifest load_bundle_manifest(
    const std::filesystem::path& bundle_directory
);
void save_bundle_manifest(
    const std::filesystem::path& bundle_directory,
    const BundleManifest& manifest
);
[[nodiscard]] PipelineTask parse_pipeline_task(std::string_view name);
[[nodiscard]] std::string_view to_string(PipelineTask task) noexcept;

}  // namespace rlf::sdk
