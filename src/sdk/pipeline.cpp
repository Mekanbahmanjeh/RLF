#include "rlf/sdk/pipeline.hpp"

#include "rlf/core/sha256.hpp"
#include "rlf/solstice/checkpoint.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace rlf::sdk {
namespace {

[[nodiscard]] ContextCapabilities context_capabilities(
    const solstice::SolsticeConfig& config
) {
    return ContextCapabilities{
        config.language.context_orders.empty()
            ? 0U
            : config.language.context_orders.back(),
        config.language.maximum_episode_cue_tokens,
        config.language.maximum_generation_tokens,
        config.general.maximum_context_characters,
    };
}

[[nodiscard]] std::string trim(const std::string_view value) {
    const std::size_t begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string_view::npos) {
        return {};
    }
    const std::size_t end = value.find_last_not_of(" \t\r\n");
    return std::string(value.substr(begin, end - begin + 1U));
}

[[nodiscard]] std::vector<std::string> split_csv(const std::string_view value) {
    std::vector<std::string> fields;
    std::size_t begin = 0U;
    while (begin <= value.size()) {
        const std::size_t delimiter = value.find(',', begin);
        const std::size_t end = delimiter == std::string_view::npos
            ? value.size()
            : delimiter;
        std::string field = trim(value.substr(begin, end - begin));
        if (field.empty()) {
            throw std::runtime_error("bundle task list contains an empty item");
        }
        fields.push_back(std::move(field));
        if (delimiter == std::string_view::npos) {
            break;
        }
        begin = delimiter + 1U;
    }
    return fields;
}

[[nodiscard]] std::filesystem::path validated_checkpoint_path(
    const std::filesystem::path& bundle_directory,
    const std::filesystem::path& relative
) {
    if (relative.empty() || relative.is_absolute()) {
        throw std::runtime_error(
            "bundle checkpoint must be a non-empty relative path"
        );
    }
    const std::filesystem::path normalized = relative.lexically_normal();
    if (normalized.empty() || *normalized.begin() == "..") {
        throw std::runtime_error("bundle checkpoint escapes the bundle directory");
    }
    return bundle_directory / normalized;
}

[[nodiscard]] std::uint32_t parse_format_version(const std::string_view value) {
    std::uint32_t parsed{};
    const char* const begin = value.data();
    const char* const end = value.data() + value.size();
    const auto [position, error] = std::from_chars(begin, end, parsed);
    if (error != std::errc{} || position != end) {
        throw std::runtime_error("bundle format must be an unsigned integer");
    }
    return parsed;
}

[[nodiscard]] ModelInfo load_model_info(
    const std::filesystem::path& checkpoint,
    std::string checkpoint_sha256,
    const BundleManifest* manifest,
    const LoadOptions& options,
    solstice::SolsticeModel& model
) {
    const std::optional<solstice::SolsticeProfile> selected_profile =
        options.profile.has_value()
        ? options.profile
        : manifest != nullptr ? manifest->profile : std::nullopt;
    if (options.enforce_profile && selected_profile.has_value() &&
        !solstice::profile_config_matches(*selected_profile, model.config())) {
        throw std::runtime_error(
            "checkpoint configuration does not match the selected profile"
        );
    }

    ModelInfo info;
    info.name = manifest != nullptr
        ? manifest->name
        : checkpoint.stem().string();
    info.architecture = manifest != nullptr
        ? manifest->architecture
        : "solstice";
    info.checkpoint = checkpoint;
    info.checkpoint_sha256 = std::move(checkpoint_sha256);
    info.checkpoint_bytes = std::filesystem::file_size(checkpoint);
    info.profile = selected_profile;
    info.tasks = manifest != nullptr
        ? manifest->tasks
        : std::vector<PipelineTask>{};
    info.license = manifest != nullptr ? manifest->license : std::string{};
    info.context = context_capabilities(model.config());
    info.stats = model.stats();
    return info;
}

}  // namespace

struct AutoModel::State final {
    std::shared_ptr<solstice::SolsticeModel> model;
    ModelInfo info;
};

PipelineTask parse_pipeline_task(const std::string_view name) {
    if (name == "text-generation") {
        return PipelineTask::text_generation;
    }
    if (name == "image-text-to-text") {
        return PipelineTask::image_text_to_text;
    }
    if (name == "tool-use") {
        return PipelineTask::tool_use;
    }
    throw std::invalid_argument("unknown RLF pipeline task: " + std::string(name));
}

std::string_view to_string(const PipelineTask task) noexcept {
    switch (task) {
        case PipelineTask::text_generation:
            return "text-generation";
        case PipelineTask::image_text_to_text:
            return "image-text-to-text";
        case PipelineTask::tool_use:
            return "tool-use";
    }
    return "unknown";
}

BundleManifest load_bundle_manifest(
    const std::filesystem::path& bundle_directory
) {
    const std::filesystem::path path =
        bundle_directory / bundle_manifest_filename;
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("unable to open RLF bundle manifest");
    }

    std::unordered_map<std::string, std::string> values;
    std::string line;
    std::size_t line_number = 0U;
    while (std::getline(input, line)) {
        ++line_number;
        const std::string cleaned = trim(line);
        if (cleaned.empty() || cleaned.front() == '#') {
            continue;
        }
        const std::size_t separator = cleaned.find('=');
        if (separator == std::string::npos) {
            throw std::runtime_error(
                "invalid RLF bundle manifest line " +
                std::to_string(line_number)
            );
        }
        std::string key = trim(std::string_view(cleaned).substr(0U, separator));
        std::string value = trim(
            std::string_view(cleaned).substr(separator + 1U)
        );
        if (key.empty() || value.empty()) {
            throw std::runtime_error(
                "empty RLF bundle manifest field on line " +
                std::to_string(line_number)
            );
        }
        const bool inserted = values.emplace(std::move(key), std::move(value)).second;
        if (!inserted) {
            throw std::runtime_error("duplicate RLF bundle manifest field");
        }
    }
    if (!input.eof()) {
        throw std::runtime_error("failed while reading RLF bundle manifest");
    }

    const std::vector<std::string_view> known_keys{
        "format", "name", "architecture", "checkpoint", "checkpoint_sha256",
        "profile", "tasks", "license",
    };
    for (const auto& [key, value] : values) {
        static_cast<void>(value);
        if (std::find(known_keys.begin(), known_keys.end(), key) ==
            known_keys.end()) {
            throw std::runtime_error("unknown RLF bundle manifest field: " + key);
        }
    }

    const auto required = [&values](const std::string_view key)
        -> const std::string& {
        const auto iterator = values.find(std::string(key));
        if (iterator == values.end()) {
            throw std::runtime_error(
                "missing RLF bundle manifest field: " + std::string(key)
            );
        }
        return iterator->second;
    };

    BundleManifest manifest;
    manifest.format_version = parse_format_version(required("format"));
    if (manifest.format_version != 1U) {
        throw std::runtime_error("unsupported RLF bundle format");
    }
    manifest.name = required("name");
    manifest.architecture = required("architecture");
    if (manifest.architecture != "solstice") {
        throw std::runtime_error("unsupported RLF bundle architecture");
    }
    manifest.checkpoint = required("checkpoint");
    static_cast<void>(validated_checkpoint_path(
        bundle_directory, manifest.checkpoint
    ));
    manifest.checkpoint_sha256 = required("checkpoint_sha256");
    if (!core::is_sha256_hex(manifest.checkpoint_sha256)) {
        throw std::runtime_error("invalid bundle checkpoint SHA-256");
    }
    const auto profile = values.find("profile");
    if (profile != values.end()) {
        manifest.profile = solstice::parse_profile(profile->second);
    }
    for (const std::string& task : split_csv(required("tasks"))) {
        const PipelineTask parsed = parse_pipeline_task(task);
        if (std::find(manifest.tasks.begin(), manifest.tasks.end(), parsed) !=
            manifest.tasks.end()) {
            throw std::runtime_error("duplicate RLF bundle pipeline task");
        }
        manifest.tasks.push_back(parsed);
    }
    const auto license = values.find("license");
    if (license != values.end()) {
        manifest.license = license->second;
    }
    return manifest;
}

void save_bundle_manifest(
    const std::filesystem::path& bundle_directory,
    const BundleManifest& manifest
) {
    if (manifest.format_version != 1U || manifest.name.empty() ||
        manifest.architecture != "solstice" || manifest.tasks.empty() ||
        !core::is_sha256_hex(manifest.checkpoint_sha256)) {
        throw std::invalid_argument("invalid RLF bundle manifest");
    }
    static_cast<void>(validated_checkpoint_path(
        bundle_directory, manifest.checkpoint
    ));
    std::filesystem::create_directories(bundle_directory);
    const std::filesystem::path final_path =
        bundle_directory / bundle_manifest_filename;
    const std::filesystem::path temporary_path =
        final_path.string() + ".tmp";
    std::ofstream output(temporary_path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("unable to create RLF bundle manifest");
    }
    output << "format=" << manifest.format_version << '\n'
           << "name=" << manifest.name << '\n'
           << "architecture=" << manifest.architecture << '\n'
           << "checkpoint=" << manifest.checkpoint.generic_string() << '\n'
           << "checkpoint_sha256=" << manifest.checkpoint_sha256 << '\n';
    if (manifest.profile.has_value()) {
        output << "profile=" << solstice::to_string(*manifest.profile) << '\n';
    }
    output << "tasks=";
    for (std::size_t index = 0U; index < manifest.tasks.size(); ++index) {
        if (index != 0U) {
            output << ',';
        }
        output << to_string(manifest.tasks[index]);
    }
    output << '\n';
    if (!manifest.license.empty()) {
        output << "license=" << manifest.license << '\n';
    }
    output.close();
    if (!output) {
        throw std::runtime_error("failed while writing RLF bundle manifest");
    }
    std::error_code error;
    std::filesystem::rename(temporary_path, final_path, error);
    if (error) {
        std::filesystem::remove(temporary_path);
        throw std::runtime_error("unable to publish RLF bundle manifest");
    }
}

AutoModel::AutoModel(
    std::shared_ptr<solstice::SolsticeModel> model,
    ModelInfo info
) : state_(std::make_shared<State>(State{
    std::move(model), std::move(info),
})) {}

AutoModel AutoModel::from_pretrained(
    const std::filesystem::path& path,
    const LoadOptions options
) {
    const bool is_bundle = std::filesystem::is_directory(path);
    std::optional<BundleManifest> manifest;
    std::filesystem::path checkpoint = path;
    if (is_bundle) {
        manifest = load_bundle_manifest(path);
        checkpoint = validated_checkpoint_path(path, manifest->checkpoint);
    }
    if (!std::filesystem::is_regular_file(checkpoint)) {
        throw std::runtime_error("RLF checkpoint does not exist");
    }
    std::string checkpoint_sha256 = core::sha256_hex(
        core::sha256_file(checkpoint)
    );
    if (manifest.has_value() && options.verify_checkpoint_sha256 &&
        checkpoint_sha256 != manifest->checkpoint_sha256) {
        throw std::runtime_error("bundle checkpoint SHA-256 mismatch");
    }

    solstice::SolsticeCheckpointLimits limits;
    const std::optional<solstice::SolsticeProfile> selected_profile =
        options.profile.has_value()
        ? options.profile
        : manifest.has_value() ? manifest->profile : std::nullopt;
    if (selected_profile.has_value()) {
        limits = solstice::checkpoint_limits_for_profile(*selected_profile);
    }
    auto model = std::make_shared<solstice::SolsticeModel>(
        solstice::load_solstice_checkpoint(checkpoint, limits)
    );
    model->set_backend(options.backend);
    ModelInfo info = load_model_info(
        checkpoint,
        std::move(checkpoint_sha256),
        manifest.has_value() ? &*manifest : nullptr,
        options,
        *model
    );
    return AutoModel(std::move(model), std::move(info));
}

AutoModel AutoModel::from_profile(
    const solstice::SolsticeProfile profile,
    LoadOptions options,
    const std::uint64_t seed,
    const bool bootstrap
) {
    if (options.profile.has_value() && options.profile != profile) {
        throw std::invalid_argument(
            "load options profile conflicts with requested model profile"
        );
    }
    options.profile = profile;
    auto model = std::make_shared<solstice::SolsticeModel>(
        solstice::make_profile_config(profile),
        seed
    );
    if (bootstrap) {
        model->bootstrap();
    }
    model->set_backend(options.backend);
    ModelInfo info;
    info.name = std::string(solstice::to_string(profile));
    info.architecture = "solstice";
    info.profile = profile;
    info.context = context_capabilities(model->config());
    info.stats = model->stats();
    return AutoModel(std::move(model), std::move(info));
}

const solstice::SolsticeModel& AutoModel::model() const noexcept {
    return *state_->model;
}

const ModelInfo& AutoModel::info() const noexcept {
    return state_->info;
}

bool AutoModel::supports(const PipelineTask task) const noexcept {
    return state_->info.tasks.empty() ||
        std::find(state_->info.tasks.begin(), state_->info.tasks.end(), task) !=
            state_->info.tasks.end();
}

std::size_t AutoModel::token_count(const std::string_view text) const {
    return state_->model->tokenizer().encode(text).size();
}

solstice::SolsticeModel& AutoModel::mutable_model() noexcept {
    return *state_->model;
}

void AutoModel::refresh_info() {
    state_->info.context = context_capabilities(state_->model->config());
    state_->info.stats = state_->model->stats();
}

void AutoModel::save_pretrained(
    const std::filesystem::path& directory,
    SaveOptions options
) {
    if (options.name.empty() || options.tasks.empty()) {
        throw std::invalid_argument("invalid RLF save_pretrained options");
    }
    const std::filesystem::path checkpoint =
        validated_checkpoint_path(directory, options.checkpoint_filename);
    std::filesystem::create_directories(checkpoint.parent_path());
    solstice::save_solstice_checkpoint(checkpoint, *state_->model);

    BundleManifest manifest;
    manifest.name = std::move(options.name);
    manifest.checkpoint = std::move(options.checkpoint_filename);
    manifest.checkpoint_sha256 = core::sha256_hex(core::sha256_file(checkpoint));
    manifest.profile = options.profile.has_value()
        ? options.profile
        : state_->info.profile;
    if (manifest.profile.has_value() &&
        !solstice::profile_config_matches(
            *manifest.profile, state_->model->config()
        )) {
        throw std::invalid_argument(
            "save_pretrained profile does not match model configuration"
        );
    }
    manifest.tasks = std::move(options.tasks);
    manifest.license = std::move(options.license);
    save_bundle_manifest(directory, manifest);

    state_->info.name = manifest.name;
    state_->info.architecture = manifest.architecture;
    state_->info.checkpoint = checkpoint;
    state_->info.checkpoint_sha256 = manifest.checkpoint_sha256;
    state_->info.checkpoint_bytes = std::filesystem::file_size(checkpoint);
    state_->info.profile = manifest.profile;
    state_->info.tasks = manifest.tasks;
    state_->info.license = manifest.license;
    refresh_info();
}

Pipeline::Pipeline(
    const PipelineTask task,
    AutoModel model,
    PipelineOptions options
) : task_(task),
    model_(std::move(model)),
    options_(std::move(options)) {
    if (!model_.supports(task_)) {
        throw std::invalid_argument(
            "model bundle does not declare support for task " +
            std::string(to_string(task_))
        );
    }
    if (task_ == PipelineTask::tool_use) {
        if (!options_.register_safe_tools) {
            throw std::invalid_argument(
                "tool-use pipeline requires register_safe_tools"
            );
        }
        tools_.emplace(options_.tool_policy);
        tools_->register_safe_builtins();
    }
}

PipelineOutput Pipeline::operator()(const PipelineRequest& request) const {
    if (request.prompt.empty()) {
        throw std::invalid_argument("pipeline prompt must not be empty");
    }
    if (task_ == PipelineTask::text_generation && request.image.has_value()) {
        throw std::invalid_argument(
            "text-generation pipeline does not accept images"
        );
    }
    solstice::ToolRuntime* runtime = tools_.has_value() ? &*tools_ : nullptr;
    const solstice::SolsticeResponse response = model_.state_->model->respond_file(
        request.prompt,
        request.image,
        runtime,
        options_.generation,
        options_.image_limits
    );
    return PipelineOutput{
        response.text,
        response.vision,
        response.tool_proposal,
        response.tool_result,
        response.uncertainty,
    };
}

PipelineOutput Pipeline::operator()(const std::string_view prompt) const {
    return (*this)(PipelineRequest{std::string(prompt), std::nullopt});
}

PipelineTask Pipeline::task() const noexcept {
    return task_;
}

const ModelInfo& Pipeline::model_info() const noexcept {
    return model_.info();
}

Pipeline make_pipeline(
    const PipelineTask task,
    const std::filesystem::path& pretrained,
    LoadOptions load_options,
    PipelineOptions pipeline_options
) {
    return Pipeline(
        task,
        AutoModel::from_pretrained(pretrained, std::move(load_options)),
        std::move(pipeline_options)
    );
}

}  // namespace rlf::sdk
