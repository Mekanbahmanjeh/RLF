#include "rlf/solstice/checkpoint.hpp"
#include "rlf/solstice/data_pipeline.hpp"
#include "rlf/solstice/evaluation_runner.hpp"
#include "rlf/solstice/image_generation_checkpoint.hpp"
#include "rlf/solstice/image_generation_artifact_manifest.hpp"
#include "rlf/solstice/image_generation_data_audit.hpp"
#include "rlf/solstice/resonant_image_fabric.hpp"
#include "rlf/solstice/profile.hpp"
#include "rlf/solstice/solstice_model.hpp"
#include "rlf/solstice/tool_protocol.hpp"
#include "rlf/solstice/vision_fabric.hpp"

#include "rlf/core/sha256.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <unordered_set>
#include <vector>

#if defined(__linux__)
#include <sys/resource.h>
#endif

namespace {

struct Options final {
    std::string command;
    std::filesystem::path checkpoint{"solstice_general_h100.rlfsp"};
    std::filesystem::path input;
    std::filesystem::path manifest;
    std::filesystem::path evaluation_manifest;
    std::filesystem::path license_policy;
    std::filesystem::path ledger;
    std::filesystem::path output;
    std::filesystem::path telemetry;
    std::optional<std::filesystem::path> image;
    std::optional<std::filesystem::path> target_image;
    std::filesystem::path tool_root;
    std::string prompt;
    std::string subject;
    std::string relation;
    std::string object{"?answer"};
    std::string backend{"optimized_cpu"};
    std::string profile{"frontier-24g"};
    std::string architecture{"resonant-fabric"};
    std::vector<std::string> transformations;
    std::uint64_t seed{0x534F4C5354494345ULL};
    std::size_t maximum_tokens{128U};
    std::size_t maximum_hops{4U};
    std::size_t top_k{16U};
    std::size_t video_frames{8U};
    std::size_t image_width{};
    std::size_t image_height{};
    std::size_t maximum_audit_records{10'000'000U};
    std::size_t maximum_new_shards{};
    std::size_t maximum_token_hash_cache_entries{};
    std::size_t maximum_prompt_bytes{16U * 1024U * 1024U};
    std::uint64_t maximum_text_shard_bytes{2ULL * 1024ULL * 1024ULL * 1024ULL};
    std::uint64_t maximum_train_shard_bytes{4ULL * 1024ULL * 1024ULL * 1024ULL};
    std::uint64_t target_training_records{};
    std::uint64_t target_training_tokens{};
    unsigned int near_duplicate_hamming_distance{3U};
    double temperature{0.8};
    bool stochastic{};
    bool blank{};
    bool enforce_profile{};
    bool require_media_hashes{};
    bool audit_media_double_read{};
    bool separate_vision_analysis{};
    bool copy_tsv_fields{};
    bool disable_tools{};
    bool enable_evaluation_tools{};
    bool help{};
    bool profile_set{};
};

[[nodiscard]] std::string_view require_value(
    const int argument_count,
    char** argument_values,
    int& index,
    const std::string_view option
) {
    if (index + 1 >= argument_count) {
        throw std::invalid_argument("missing value for " + std::string(option));
    }
    ++index;
    return argument_values[index];
}

template <typename Integer>
[[nodiscard]] Integer parse_integer(
    const std::string_view value,
    const std::string_view option
) {
    Integer result{};
    const auto [end, error] = std::from_chars(
        value.data(), value.data() + value.size(), result
    );
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::invalid_argument("invalid integer for " + std::string(option));
    }
    return result;
}

[[nodiscard]] double parse_double(
    const std::string_view value,
    const std::string_view option
) {
    double result{};
    const auto [end, error] = std::from_chars(
        value.data(), value.data() + value.size(), result
    );
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::invalid_argument("invalid number for " + std::string(option));
    }
    return result;
}

[[nodiscard]] Options parse_options(
    const int argument_count,
    char** argument_values
) {
    Options options;
    if (argument_count >= 2) {
        options.command = argument_values[1];
        if (options.command == "--help" || options.command == "-h") {
            options.help = true;
            options.command = "help";
        }
    }
    for (int index = 2; index < argument_count; ++index) {
        const std::string_view argument = argument_values[index];
        if (argument == "--checkpoint") {
            options.checkpoint = require_value(
                argument_count, argument_values, index, argument
            );
        } else if (argument == "--input") {
            options.input = require_value(
                argument_count, argument_values, index, argument
            );
        } else if (argument == "--manifest") {
            options.manifest = require_value(
                argument_count, argument_values, index, argument
            );
        } else if (argument == "--evaluation-manifest") {
            options.evaluation_manifest = require_value(
                argument_count, argument_values, index, argument
            );
        } else if (argument == "--license-policy") {
            options.license_policy = require_value(
                argument_count, argument_values, index, argument
            );
        } else if (argument == "--ledger") {
            options.ledger = require_value(
                argument_count, argument_values, index, argument
            );
        } else if (argument == "--output") {
            options.output = require_value(
                argument_count, argument_values, index, argument
            );
        } else if (argument == "--telemetry") {
            options.telemetry = require_value(
                argument_count, argument_values, index, argument
            );
        } else if (argument == "--image") {
            options.image = std::filesystem::path(require_value(
                argument_count, argument_values, index, argument
            ));
        } else if (argument == "--target-image") {
            options.target_image = std::filesystem::path(require_value(
                argument_count, argument_values, index, argument
            ));
        } else if (argument == "--tool-root") {
            options.tool_root = require_value(
                argument_count, argument_values, index, argument
            );
        } else if (argument == "--prompt") {
            options.prompt = require_value(
                argument_count, argument_values, index, argument
            );
        } else if (argument == "--subject") {
            options.subject = require_value(
                argument_count, argument_values, index, argument
            );
        } else if (argument == "--relation") {
            options.relation = require_value(
                argument_count, argument_values, index, argument
            );
        } else if (argument == "--object") {
            options.object = require_value(
                argument_count, argument_values, index, argument
            );
        } else if (argument == "--backend") {
            options.backend = require_value(
                argument_count, argument_values, index, argument
            );
        } else if (argument == "--profile") {
            options.profile = require_value(
                argument_count, argument_values, index, argument
            );
            options.profile_set = true;
        } else if (argument == "--architecture") {
            options.architecture = require_value(
                argument_count, argument_values, index, argument
            );
        } else if (argument == "--transform") {
            options.transformations.emplace_back(require_value(
                argument_count, argument_values, index, argument
            ));
        } else if (argument == "--seed") {
            options.seed = parse_integer<std::uint64_t>(
                require_value(argument_count, argument_values, index, argument),
                argument
            );
        } else if (argument == "--max-tokens") {
            options.maximum_tokens = parse_integer<std::size_t>(
                require_value(argument_count, argument_values, index, argument),
                argument
            );
        } else if (argument == "--max-hops") {
            options.maximum_hops = parse_integer<std::size_t>(
                require_value(argument_count, argument_values, index, argument),
                argument
            );
        } else if (argument == "--top-k") {
            options.top_k = parse_integer<std::size_t>(
                require_value(argument_count, argument_values, index, argument),
                argument
            );
        } else if (argument == "--frames") {
            options.video_frames = parse_integer<std::size_t>(
                require_value(argument_count, argument_values, index, argument),
                argument
            );
        } else if (argument == "--width") {
            options.image_width = parse_integer<std::size_t>(
                require_value(argument_count, argument_values, index, argument),
                argument
            );
        } else if (argument == "--height") {
            options.image_height = parse_integer<std::size_t>(
                require_value(argument_count, argument_values, index, argument),
                argument
            );
        } else if (argument == "--max-audit-records") {
            options.maximum_audit_records = parse_integer<std::size_t>(
                require_value(argument_count, argument_values, index, argument),
                argument
            );
        } else if (argument == "--maximum-new-shards") {
            options.maximum_new_shards = parse_integer<std::size_t>(
                require_value(argument_count, argument_values, index, argument),
                argument
            );
        } else if (argument == "--max-token-hash-cache-entries") {
            options.maximum_token_hash_cache_entries = parse_integer<std::size_t>(
                require_value(argument_count, argument_values, index, argument),
                argument
            );
        } else if (argument == "--max-prompt-bytes") {
            options.maximum_prompt_bytes = parse_integer<std::size_t>(
                require_value(argument_count, argument_values, index, argument),
                argument
            );
        } else if (argument == "--max-text-shard-bytes") {
            options.maximum_text_shard_bytes = parse_integer<std::uint64_t>(
                require_value(argument_count, argument_values, index, argument),
                argument
            );
        } else if (argument == "--max-train-shard-bytes") {
            options.maximum_train_shard_bytes = parse_integer<std::uint64_t>(
                require_value(argument_count, argument_values, index, argument),
                argument
            );
        } else if (argument == "--target-training-records") {
            options.target_training_records = parse_integer<std::uint64_t>(
                require_value(argument_count, argument_values, index, argument),
                argument
            );
        } else if (argument == "--target-training-tokens") {
            options.target_training_tokens = parse_integer<std::uint64_t>(
                require_value(argument_count, argument_values, index, argument),
                argument
            );
        } else if (argument == "--near-duplicate-hamming") {
            options.near_duplicate_hamming_distance = parse_integer<unsigned int>(
                require_value(argument_count, argument_values, index, argument),
                argument
            );
        } else if (argument == "--temperature") {
            options.temperature = parse_double(
                require_value(argument_count, argument_values, index, argument),
                argument
            );
        } else if (argument == "--stochastic") {
            options.stochastic = true;
        } else if (argument == "--blank") {
            options.blank = true;
        } else if (argument == "--enforce-profile") {
            options.enforce_profile = true;
        } else if (argument == "--require-media-hashes") {
            options.require_media_hashes = true;
        } else if (argument == "--audit-media-double-read") {
            options.audit_media_double_read = true;
        } else if (argument == "--separate-vision-analysis") {
            options.separate_vision_analysis = true;
        } else if (argument == "--copy-tsv-fields") {
            options.copy_tsv_fields = true;
        } else if (argument == "--no-tools") {
            options.disable_tools = true;
        } else if (argument == "--enable-tools") {
            options.enable_evaluation_tools = true;
        } else if (argument == "--help" || argument == "-h") {
            options.help = true;
        } else {
            throw std::invalid_argument("unknown option: " + std::string(argument));
        }
    }
    return options;
}

void print_help(std::ostream& output) {
    output <<
        "RLF Solstice-General-Frontier - multimodal instruction learning, reasoning, and safe tools\n\n"
        "Commands:\n"
        "  bootstrap          Create a checkpoint using --profile\n"
        "  profile-info       Print capacity and profile settings\n"
        "  device-info        Inspect the selected CPU/CUDA backend\n"
        "  train-text         Train on a UTF-8 text corpus (--input)\n"
        "  train-dialogue     Train prompt/response TSV rows (--manifest)\n"
        "  train-instructions Train task/domain/prompt/rationale/response TSV rows\n"
        "  train-preferences  Train prompt/chosen/rejected preference TSV rows\n"
        "  train-vision       Train image<TAB>caption TSV rows (--manifest)\n"
        "  train-tools        Train request<TAB>tool TSV rows (--manifest)\n"
        "  train-facts        Train subject<TAB>relation<TAB>object TSV rows\n"
        "  train-rules        Train compositional rule TSV rows\n"
        "  audit-data         Verify provenance, SHA-256, deduplication, and contamination\n"
        "  train-data         Audit and resume an immutable multi-shard training ledger\n"
        "  generate-video     Render prompt-retrieved learned motion-prototype PPM frames\n"
        "  evaluate-video     Evaluate held-out video ledger splits with raw artifacts\n"
        "  evaluate-batch     Produce resumable hash-linked raw predictions\n"
        "  imagegen-bootstrap Create a profile-bound .rlfimg checkpoint\n"
        "  imagegen-train-language-ledger Audit and learn prompt semantics from text shards\n"
        "  imagegen-train-pair Learn one prompt/target or source/target mapping\n"
        "  imagegen-train-manifest Audit and atomically train one pair-manifest shard\n"
        "  imagegen-audit-pairs Compute native provenance/dedup/contamination reports\n"
        "  imagegen-evaluate-manifest Run frozen held-out image metrics and raw outputs\n"
        "  imagegen-generate  Generate from --prompt or transform an --image\n"
        "  imagegen-inspect   Inspect a .rlfimg checkpoint\n"
        "  imagegen-verify    Fully validate a .rlfimg checkpoint\n"
        "  imagegen-verify-artifacts Rehash a complete 16-file evidence bundle\n"
        "  imagegen-profile-info Print image-training capacity and claim boundary\n"
        "  reason             Query --subject --relation [--object]\n"
        "  ask                Answer one prompt (--prompt, optional --image)\n"
        "  chat               Interactive session (optional --image)\n"
        "  inspect-image      Show visual regions and concepts (--image)\n"
        "  inspect-checkpoint Show model/checkpoint statistics\n"
        "  verify-checkpoint  Validate and load the checkpoint\n"
        "  demo               Run an in-memory bootstrap demonstration\n\n"
        "Common options:\n"
        "  --checkpoint FILE  Checkpoint path (default solstice_general_h100.rlfsp)\n"
        "  --ledger FILE      Immutable 16-field training/evaluation shard ledger\n"
        "  --output PATH      Data-audit file or evaluate-batch output directory\n"
        "  --telemetry FILE   Write end-to-end train-data timing/resource JSON\n"
        "  --tool-root DIR    Enable read-only file tools below this directory\n"
        "  --backend NAME     scalar_cpu, optimized_cpu, or cuda\n"
        "  --profile NAME     general-h100, general-h200-141g-30t, general-40g,\n"
        "                     general-v100-32g,\n"
        "                     general-v100-32g-text, general-v100-32g-500m,\n"
        "                     video-v100-32g,\n"
        "                     rtx-pro-6000-96g,\n"
        "                     general-rtx-pro-6000-96g,\n"
        "                     general-rtx-pro-6000-96g-text, video-rtx-pro-6000-96g,\n"
        "                     frontier-h100, frontier-24g, or preview-6g\n"
        "                     Image commands: imagegen-v100-32g,\n"
        "                     imagegen-a100-80g, or imagegen-reference\n"
        "  --architecture NAME  resonant-fabric (authoritative) or\n"
        "                     patch-quilt-baseline (explicit baseline)\n"
        "  --target-image FILE  Target image for imagegen-train-pair\n"
        "  --evaluation-manifest FILE Frozen held-out eight-column image pair manifest\n"
        "  --license-policy FILE Allowed licenses, one exact identifier per line\n"
        "  --width/--height N  Prompt-only imagegen output dimensions\n"
        "  --transform LABEL  Learned image operation; repeat to compose\n"
        "  --subject TEXT     Reasoning-query subject\n"
        "  --relation TEXT    Reasoning-query relation\n"
        "  --object TEXT      Query object or variable (default ?answer)\n"
        "  --max-tokens N     Maximum generated tokens\n"
        "  --top-k N          Sampling candidate limit\n"
        "  --frames N         Prototype video frames to render (default 8)\n"
        "  --max-audit-records N  Bound records retained by contamination audit\n"
        "  --maximum-new-shards N  Intentional resumability drill stop (exit 4)\n"
        "  --max-token-hash-cache-entries N  Audit token cache ceiling; 0 disables\n"
        "  --max-text-shard-bytes N  Bound one text shard before checkpointing\n"
        "  --max-train-shard-bytes N  Bound shard+media bytes before checkpointing\n"
        "  --target-training-records N  Require this exact cumulative train-data count\n"
        "  --target-training-tokens N  Require this exact cumulative tokenizer-piece count\n"
        "  --near-duplicate-hamming N  SimHash distance (0-63, default 3)\n"
        "  --temperature X    Sampling temperature\n"
        "  --stochastic       Enable seeded stochastic generation\n"
        "  --blank            Start training from an empty model when no checkpoint exists\n"
        "  --enforce-profile  Reject a checkpoint or shard modality inconsistent with --profile\n"
        "  --require-media-hashes Require SHA-256 in every image/video manifest row\n"
        "  --audit-media-double-read  Audit ablation: hash and decode via separate reads\n"
        "  --separate-vision-analysis Training ablation: repeat visual feature extraction\n"
        "  --copy-tsv-fields RTX ingest ablation: allocate owned strings per TSV row\n"
        "  --max-prompt-bytes N  Per-example batch prompt limit (default 16 MiB)\n"
        "  --enable-tools     Enable safe builtins for evaluate-batch (off by default)\n"
        "  --no-tools         Disable tool routing for ask/chat\n\n"
        "Ubuntu builds use libpng/libjpeg when available and always support PPM/PGM/BMP.\n"
        "Windows builds can additionally use Windows Imaging Component.\n";
}

[[nodiscard]] std::string read_text_file(
    const std::filesystem::path& path,
    const std::uintmax_t maximum_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL
) {
    const std::uintmax_t size = std::filesystem::file_size(path);
    if (size > maximum_bytes) {
        throw std::runtime_error("input file exceeds the configured limit");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("unable to open input file: " + path.string());
    }
    std::string text(static_cast<std::size_t>(size), '\0');
    if (!text.empty()) {
        input.read(text.data(), static_cast<std::streamsize>(text.size()));
    }
    if (!input) {
        throw std::runtime_error("failed while reading input file: " + path.string());
    }
    return text;
}

[[nodiscard]] std::vector<std::string> split_tabs(const std::string_view line) {
    std::vector<std::string> fields;
    std::size_t start = 0U;
    while (start <= line.size()) {
        const std::size_t tab = line.find('\t', start);
        const std::size_t end = tab == std::string_view::npos ? line.size() : tab;
        fields.emplace_back(line.substr(start, end - start));
        if (tab == std::string_view::npos) {
            break;
        }
        start = tab + 1U;
    }
    return fields;
}

template <typename Callback>
[[nodiscard]] std::size_t for_each_tsv(
    const std::filesystem::path& path,
    const std::size_t minimum_fields,
    const std::size_t maximum_fields,
    Callback&& callback
) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("unable to open manifest: " + path.string());
    }
    std::string line;
    std::size_t line_number = 0U;
    std::size_t row_count = 0U;
    while (std::getline(input, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line.front() == '#') {
            continue;
        }
        std::vector<std::string> fields = split_tabs(line);
        if (fields.size() < minimum_fields || fields.size() > maximum_fields) {
            throw std::runtime_error(
                "invalid field count at " + path.string() + ":" +
                std::to_string(line_number)
            );
        }
        callback(fields);
        ++row_count;
    }
    if (!input.eof()) {
        throw std::runtime_error("failed while reading manifest: " + path.string());
    }
    return row_count;
}

template <typename Callback>
[[nodiscard]] std::size_t for_each_tsv_views(
    const std::filesystem::path& path,
    const std::size_t minimum_fields,
    const std::size_t maximum_fields,
    Callback&& callback
) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("unable to open manifest: " + path.string());
    }
    std::string line;
    std::vector<std::string_view> fields;
    fields.reserve(maximum_fields);
    std::size_t line_number = 0U;
    std::size_t row_count = 0U;
    while (std::getline(input, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line.front() == '#') {
            continue;
        }
        fields.clear();
        const std::string_view line_view(line);
        std::size_t start = 0U;
        while (start <= line_view.size()) {
            const std::size_t tab = line_view.find('\t', start);
            const std::size_t end = tab == std::string_view::npos
                ? line_view.size()
                : tab;
            fields.push_back(line_view.substr(start, end - start));
            if (tab == std::string_view::npos) {
                break;
            }
            start = tab + 1U;
        }
        if (fields.size() < minimum_fields || fields.size() > maximum_fields) {
            throw std::runtime_error(
                "invalid field count at " + path.string() + ":" +
                std::to_string(line_number)
            );
        }
        callback(std::span<const std::string_view>(fields));
        ++row_count;
    }
    if (!input.eof()) {
        throw std::runtime_error("failed while reading manifest: " + path.string());
    }
    return row_count;
}

template <typename Callback>
[[nodiscard]] std::size_t for_each_training_tsv(
    const std::filesystem::path& path,
    const std::size_t minimum_fields,
    const std::size_t maximum_fields,
    const bool view_fields,
    Callback&& callback
) {
    if (view_fields) {
        return for_each_tsv_views(
            path,
            minimum_fields,
            maximum_fields,
            callback
        );
    }
    return for_each_tsv(
        path,
        minimum_fields,
        maximum_fields,
        callback
    );
}


[[nodiscard]] rlf::frontier::FrontierBackendKind parse_backend(
    const std::string_view name
) {
    if (name == "scalar_cpu") {
        return rlf::frontier::FrontierBackendKind::scalar_cpu;
    }
    if (name == "optimized_cpu") {
        return rlf::frontier::FrontierBackendKind::optimized_cpu;
    }
    if (name == "cuda") {
        return rlf::frontier::FrontierBackendKind::cuda;
    }
    throw std::invalid_argument("unknown backend: " + std::string(name));
}

[[nodiscard]] rlf::solstice::SolsticeCheckpointLimits checkpoint_limits(
    const Options& options
) {
    return rlf::solstice::checkpoint_limits_for_profile(
        rlf::solstice::parse_profile(options.profile)
    );
}

[[nodiscard]] rlf::solstice::SolsticeModel load_or_create(
    const Options& options
) {
    const rlf::solstice::SolsticeProfile profile =
        rlf::solstice::parse_profile(options.profile);
    const rlf::solstice::SolsticeCheckpointLimits checkpoint_limits =
        rlf::solstice::checkpoint_limits_for_profile(profile);
    rlf::solstice::SolsticeModel model = std::filesystem::exists(options.checkpoint)
        ? rlf::solstice::load_solstice_checkpoint(
            options.checkpoint,
            checkpoint_limits
        )
        : rlf::solstice::SolsticeModel(
            rlf::solstice::make_profile_config(profile), options.seed
        );
    if (options.enforce_profile &&
        !rlf::solstice::profile_config_matches(profile, model.config())) {
        throw std::runtime_error(
            "checkpoint configuration does not match enforced profile " +
            std::string(rlf::solstice::to_string(profile))
        );
    }
    if (options.enforce_profile &&
        !rlf::solstice::profile_allows_vision(profile) &&
        model.stats().images_seen != 0U) {
        throw std::runtime_error(
            "text-only profile cannot resume a checkpoint containing image training"
        );
    }
    if (options.enforce_profile &&
        !rlf::solstice::profile_allows_video(profile) &&
        model.stats().video_sequences_seen != 0U) {
        throw std::runtime_error(
            "non-video profile cannot resume a checkpoint containing video training"
        );
    }
    model.set_backend(parse_backend(options.backend));
    if (!std::filesystem::exists(options.checkpoint) && !options.blank) {
        model.bootstrap();
    }
    return model;
}

void enforce_loaded_profile(
    const Options& options,
    const rlf::solstice::SolsticeModel& model
) {
    if (!options.enforce_profile) return;
    const auto profile = rlf::solstice::parse_profile(options.profile);
    if (!rlf::solstice::profile_config_matches(profile, model.config())) {
        throw std::runtime_error(
            "checkpoint configuration does not match enforced profile " +
            std::string(rlf::solstice::to_string(profile))
        );
    }
    if (!rlf::solstice::profile_allows_vision(profile) &&
        model.stats().images_seen != 0U) {
        throw std::runtime_error(
            "text-only profile cannot use a checkpoint containing image training"
        );
    }
    if (!rlf::solstice::profile_allows_video(profile) &&
        model.stats().video_sequences_seen != 0U) {
        throw std::runtime_error(
            "non-video profile cannot use a checkpoint containing video training"
        );
    }
}

void print_stats(const rlf::solstice::SolsticeStats& stats) {
    std::cout
        << "vocabulary=" << stats.vocabulary_size << '\n'
        << "language_contexts=" << stats.language_contexts << '\n'
        << "language_episodes=" << stats.language_episodes << '\n'
        << "language_tokens_seen=" << stats.language_tokens_seen << '\n'
        << "visual_modes=" << stats.visual_modes << '\n'
        << "visual_examples=" << stats.visual_examples << '\n'
        << "images_seen=" << stats.images_seen << '\n'
        << "video_prototypes=" << stats.video_prototypes << '\n'
        << "video_sequences_seen=" << stats.video_sequences_seen << '\n'
        << "video_frames_seen=" << stats.video_frames_seen << '\n'
        << "tool_routes=" << stats.tool_routes << '\n'
        << "reasoning_facts=" << stats.reasoning_facts << '\n'
        << "reasoning_rules=" << stats.reasoning_rules << '\n'
        << "continual_prototypes=" << stats.continual_prototypes << '\n'
        << "grounding_links=" << stats.grounding_links << '\n'
        << "general_demonstrations=" << stats.general_demonstrations << '\n'
        << "preference_examples=" << stats.preference_examples << '\n'
        << "active_learning_items=" << stats.active_learning_items << '\n'
        << "completed_training_shards=" << stats.completed_training_shards << '\n'
        << "audited_training_records=" << stats.audited_training_records << '\n'
        << "audited_training_bytes=" << stats.audited_training_bytes << '\n'
        << "deterministic_hash=0x" << std::hex << stats.deterministic_hash
        << std::dec << '\n';
}

[[nodiscard]] rlf::solstice::ToolRuntime make_tools(const Options& options) {
    rlf::solstice::ToolPolicy policy;
    if (!options.tool_root.empty()) {
        policy.sandbox_root = options.tool_root;
        policy.allow_file_reads = true;
    }
    rlf::solstice::ToolRuntime tools(policy);
    tools.register_safe_builtins();
    return tools;
}

[[nodiscard]] rlf::solstice::GenerationSettings generation_settings(
    const Options& options
) {
    return rlf::solstice::GenerationSettings{
        options.maximum_tokens,
        options.top_k,
        options.temperature,
        !options.stochastic,
        options.seed,
    };
}

void print_response(const rlf::solstice::SolsticeResponse& response) {
    if (response.vision.has_value()) {
        std::cout << "[vision confidence=" << std::fixed << std::setprecision(3)
                  << response.vision->confidence << "] "
                  << response.vision->description << '\n';
    }
    if (response.tool_proposal.has_value()) {
        std::cout << "<tool_call>\n"
                  << rlf::solstice::ToolRuntime::serialize_call(
                         response.tool_proposal->call
                     )
                  << "\n</tool_call>\n";
    }
    if (response.tool_result.has_value()) {
        std::cout << "<tool_result>\n"
                  << rlf::solstice::ToolRuntime::serialize_result(
                         *response.tool_result
                     )
                  << "\n</tool_result>\n";
    }
    std::cout << "Solstice: " << response.text << '\n'
              << "[uncertainty=" << std::fixed << std::setprecision(3)
              << response.uncertainty << "]\n";
}

int run_profile_info(const Options& options) {
    const rlf::solstice::SolsticeProfile profile =
        rlf::solstice::parse_profile(options.profile);
    const rlf::solstice::SolsticeConfig config =
        rlf::solstice::make_profile_config(profile);
    const rlf::solstice::ProfileCapacityEstimate capacity =
        rlf::solstice::estimate_profile_capacity(profile);
    std::cout
        << "profile=" << rlf::solstice::to_string(profile) << '\n'
        << "vision_training_enabled="
        << (rlf::solstice::profile_allows_vision(profile) ? "true" : "false") << '\n'
        << "video_training_enabled="
        << (rlf::solstice::profile_allows_video(profile) ? "true" : "false") << '\n'
        << "video_prototype_ceiling=" << config.video.maximum_sequences << '\n'
        << "video_training_frames_per_sequence="
        << config.video.maximum_frames_per_sequence << '\n'
        << "video_generation_frame_ceiling="
        << config.video.maximum_generation_frames << '\n'
        << "video_prototype_output=" << config.video.output_width << 'x'
        << config.video.output_height << '\n'
        << "vocabulary_ceiling=" << config.tokenizer.maximum_vocabulary << '\n'
        << "context_ceiling=" << config.language.maximum_contexts << '\n'
        << "episode_ceiling=" << config.language.maximum_episodes << '\n'
        << "maximum_context_order=" << config.language.context_orders.back() << '\n'
        << "visual_mode_ceiling=" << config.vision.maximum_modes << '\n'
        << "visual_example_ceiling=" << config.vision.maximum_examples << '\n'
        << "visual_descriptor_dimensions=" << config.vision.descriptor_dimensions << '\n'
        << "maximum_input_side=" << config.vision.maximum_input_side << '\n'
        << "maximum_patches=" << config.vision.maximum_patches << '\n'
        << "retrieval_query_batch=" << config.vision.retrieval_query_batch << '\n'
        << "retrieval_candidate_batch=" << config.vision.retrieval_candidate_batch << '\n'
        << "training_patch_batch=" << config.vision.training_patch_batch << '\n'
        << "sparse_routing_minimum_modes=" << config.vision.sparse_routing_minimum_modes << '\n'
        << "sparse_routing_candidates=" << config.vision.sparse_router.maximum_candidates << '\n'
        << "reasoning_fact_ceiling=" << config.abstraction.maximum_facts << '\n'
        << "reasoning_rule_ceiling=" << config.abstraction.maximum_rules << '\n'
        << "continual_prototype_ceiling=" << config.continual.maximum_prototypes << '\n'
        << "replay_capacity=" << config.continual.replay_capacity << '\n'
        << "grounding_link_ceiling=" << config.grounding.maximum_links << '\n'
        << "general_demonstration_ceiling=" << config.general.maximum_demonstrations << '\n'
        << "preference_example_ceiling=" << config.general.maximum_preferences << '\n'
        << "active_learning_item_ceiling="
        << config.general.maximum_active_learning_items << '\n'
        << "general_retrieval_candidates="
        << config.general.maximum_retrieval_candidates << '\n'
        << "general_retrieved_demonstrations="
        << config.general.maximum_retrieved_demonstrations << '\n'
        << "general_context_characters="
        << config.general.maximum_context_characters << '\n'
        << "deliberation_candidates=" << config.general.deliberation_candidates << '\n'
        << "gpu_working_set_gib="
        << static_cast<double>(capacity.gpu_working_set_bytes) /
            static_cast<double>(1ULL << 30U) << '\n'
        << "cpu_resident_gib="
        << static_cast<double>(capacity.cpu_resident_bytes) /
            static_cast<double>(1ULL << 30U) << '\n'
        << "checkpoint_ceiling_gib="
        << static_cast<double>(capacity.checkpoint_ceiling_bytes) /
            static_cast<double>(1ULL << 30U) << '\n';
    std::cout << "patch_sizes=";
    if (config.vision.patch_sizes.empty()) {
        std::cout << config.vision.patch_size;
    } else {
        for (const std::size_t value : config.vision.patch_sizes) {
            std::cout << value << ' ';
        }
    }
    std::cout << '\n';
    return 0;
}

int run_device_info(const Options& options) {
    const rlf::solstice::SolsticeProfile profile =
        rlf::solstice::parse_profile(options.profile);
    const rlf::solstice::ProfileCapacityEstimate capacity =
        rlf::solstice::estimate_profile_capacity(profile);
    const auto backend = rlf::frontier::make_frontier_backend(
        parse_backend(options.backend)
    );
    const rlf::frontier::BackendCapabilities capabilities =
        backend->capabilities();
    std::cout
        << "backend=" << backend->name() << '\n'
        << "available=" << (capabilities.available ? "true" : "false") << '\n'
        << "deterministic=" << (capabilities.deterministic ? "true" : "false") << '\n'
        << "supports_batch=" << (capabilities.supports_batch ? "true" : "false") << '\n'
        << "supports_local_update="
        << (capabilities.supports_local_update ? "true" : "false") << '\n'
        << "supports_candidate_cache="
        << (capabilities.supports_candidate_cache ? "true" : "false") << '\n'
        << "maximum_batch=" << capabilities.maximum_batch << '\n'
        << "device_memory_gib="
        << static_cast<double>(capabilities.device_memory_bytes) /
            static_cast<double>(1ULL << 30U) << '\n'
        << "profile=" << rlf::solstice::to_string(profile) << '\n'
        << "profile_gpu_working_set_gib="
        << static_cast<double>(capacity.gpu_working_set_bytes) /
            static_cast<double>(1ULL << 30U) << '\n';
    if (capabilities.device_memory_bytes != 0U) {
        std::cout << "profile_fits_device="
                  << (capabilities.device_memory_bytes >= capacity.gpu_working_set_bytes
                      ? "true" : "false")
                  << '\n';
    }
    return 0;
}

int run_bootstrap(const Options& options) {
    const rlf::solstice::SolsticeProfile profile =
        rlf::solstice::parse_profile(options.profile);
    rlf::solstice::SolsticeModel model(
        rlf::solstice::make_profile_config(profile), options.seed
    );
    model.set_backend(parse_backend(options.backend));
    model.bootstrap();
    rlf::solstice::save_solstice_checkpoint(options.checkpoint, model);
    std::cout << "Created " << options.checkpoint.string() << '\n';
    print_stats(model.stats());
    return 0;
}

int run_train_text(const Options& options) {
    if (options.input.empty()) {
        throw std::invalid_argument("train-text requires --input");
    }
    rlf::solstice::SolsticeModel model = load_or_create(options);
    const std::string corpus = read_text_file(options.input);
    model.train_text_corpus(corpus);
    rlf::solstice::save_solstice_checkpoint(options.checkpoint, model);
    std::cout << "Trained text corpus and saved " << options.checkpoint.string() << '\n';
    print_stats(model.stats());
    return 0;
}

int run_train_dialogue(const Options& options) {
    if (options.manifest.empty()) {
        throw std::invalid_argument("train-dialogue requires --manifest");
    }
    rlf::solstice::SolsticeModel model = load_or_create(options);
    const std::size_t row_count = for_each_tsv(
        options.manifest, 2U, 3U,
        [&model](const std::vector<std::string>& row) {
            model.train_dialogue(
                row[0U], row[1U], row.size() == 3U ? row[2U] : ""
            );
        }
    );
    rlf::solstice::save_solstice_checkpoint(options.checkpoint, model);
    std::cout << "Trained " << row_count << " dialogue rows and saved "
              << options.checkpoint.string() << '\n';
    print_stats(model.stats());
    return 0;
}

int run_train_instructions(const Options& options) {
    if (options.manifest.empty()) {
        throw std::invalid_argument("train-instructions requires --manifest");
    }
    rlf::solstice::SolsticeModel model = load_or_create(options);
    const std::size_t row_count = for_each_tsv(
        options.manifest, 5U, 6U,
        [&model](const std::vector<std::string>& row) {
            const double quality = row.size() == 6U
                ? parse_double(row[5U], "instruction quality")
                : 1.0;
            model.train_instruction(
                row[0U], row[1U], row[2U], row[3U], row[4U], quality
            );
        }
    );
    rlf::solstice::save_solstice_checkpoint(options.checkpoint, model);
    std::cout << "Trained " << row_count << " general instruction rows and saved "
              << options.checkpoint.string() << '\n';
    print_stats(model.stats());
    return 0;
}

int run_train_preferences(const Options& options) {
    if (options.manifest.empty()) {
        throw std::invalid_argument("train-preferences requires --manifest");
    }
    rlf::solstice::SolsticeModel model = load_or_create(options);
    const std::size_t row_count = for_each_tsv(
        options.manifest, 3U, 5U,
        [&model](const std::vector<std::string>& row) {
            const std::string feedback = row.size() >= 4U ? row[3U] : "";
            const double weight = row.size() == 5U
                ? parse_double(row[4U], "preference weight")
                : 1.0;
            model.train_preference(
                row[0U], row[1U], row[2U], feedback, weight
            );
        }
    );
    rlf::solstice::save_solstice_checkpoint(options.checkpoint, model);
    std::cout << "Trained " << row_count << " preference rows and saved "
              << options.checkpoint.string() << '\n';
    print_stats(model.stats());
    return 0;
}

int run_train_vision(const Options& options) {
    if (options.manifest.empty()) {
        throw std::invalid_argument("train-vision requires --manifest");
    }
    rlf::solstice::SolsticeModel model = load_or_create(options);
    const std::filesystem::path base = options.manifest.parent_path();
    const std::size_t row_count = for_each_tsv(
        options.manifest, 2U, 2U,
        [&model, &base](const std::vector<std::string>& row) {
            std::filesystem::path image_path(row[0U]);
            if (image_path.is_relative()) {
                image_path = base / image_path;
            }
            model.train_image_file(image_path, row[1U]);
        }
    );
    rlf::solstice::save_solstice_checkpoint(options.checkpoint, model);
    std::cout << "Trained " << row_count << " image-caption rows and saved "
              << options.checkpoint.string() << '\n';
    print_stats(model.stats());
    return 0;
}

int run_train_tools(const Options& options) {
    if (options.manifest.empty()) {
        throw std::invalid_argument("train-tools requires --manifest");
    }
    rlf::solstice::SolsticeModel model = load_or_create(options);
    const std::size_t row_count = for_each_tsv(
        options.manifest, 2U, 2U,
        [&model](const std::vector<std::string>& row) {
            model.train_tool_route(row[0U], row[1U]);
        }
    );
    rlf::solstice::save_solstice_checkpoint(options.checkpoint, model);
    std::cout << "Trained " << row_count << " tool-routing rows and saved "
              << options.checkpoint.string() << '\n';
    print_stats(model.stats());
    return 0;
}

int run_ask(const Options& options) {
    if (options.prompt.empty()) {
        throw std::invalid_argument("ask requires --prompt");
    }
    const auto profile = rlf::solstice::parse_profile(options.profile);
    if (options.enforce_profile && options.image.has_value() &&
        !rlf::solstice::profile_allows_vision(profile)) {
        throw std::invalid_argument("image input is disabled by the enforced profile");
    }
    rlf::solstice::SolsticeModel model =
        rlf::solstice::load_solstice_checkpoint(
            options.checkpoint,
            checkpoint_limits(options)
        );
    enforce_loaded_profile(options, model);
    model.set_backend(parse_backend(options.backend));
    rlf::solstice::ToolRuntime tools = make_tools(options);
    rlf::solstice::ToolRuntime* tool_pointer = options.disable_tools ? nullptr : &tools;
    const rlf::solstice::SolsticeResponse response = model.respond_file(
        options.prompt,
        options.image,
        tool_pointer,
        generation_settings(options)
    );
    print_response(response);
    return 0;
}

int run_chat(const Options& options) {
    const auto profile = rlf::solstice::parse_profile(options.profile);
    if (options.enforce_profile && options.image.has_value() &&
        !rlf::solstice::profile_allows_vision(profile)) {
        throw std::invalid_argument("image input is disabled by the enforced profile");
    }
    rlf::solstice::SolsticeModel model =
        rlf::solstice::load_solstice_checkpoint(
            options.checkpoint,
            checkpoint_limits(options)
        );
    enforce_loaded_profile(options, model);
    model.set_backend(parse_backend(options.backend));
    rlf::solstice::ToolRuntime tools = make_tools(options);
    rlf::solstice::ToolRuntime* tool_pointer = options.disable_tools ? nullptr : &tools;
    std::optional<std::filesystem::path> image = options.image;
    std::cout << "Solstice-General-Frontier interactive session. Commands: /image PATH, "
                 "/clear-image, /stats, /quit\n";
    std::string line;
    while (true) {
        std::cout << "You: " << std::flush;
        if (!std::getline(std::cin, line)) {
            break;
        }
        if (line == "/quit" || line == "/exit") {
            break;
        }
        if (line == "/stats") {
            print_stats(model.stats());
            continue;
        }
        if (line == "/clear-image") {
            image.reset();
            std::cout << "Image context cleared.\n";
            continue;
        }
        if (line.starts_with("/image ")) {
            if (options.enforce_profile &&
                !rlf::solstice::profile_allows_vision(profile)) {
                std::cout << "Image input is disabled by the enforced profile.\n";
                continue;
            }
            image = std::filesystem::path(line.substr(7U));
            std::cout << "Image context set to " << image->string() << "\n";
            continue;
        }
        if (line.empty()) {
            continue;
        }
        print_response(model.respond_file(
            line, image, tool_pointer, generation_settings(options)
        ));
    }
    return 0;
}

int run_inspect_image(const Options& options) {
    if (!options.image.has_value()) {
        throw std::invalid_argument("inspect-image requires --image");
    }
    if (options.enforce_profile && !rlf::solstice::profile_allows_vision(
            rlf::solstice::parse_profile(options.profile))) {
        throw std::invalid_argument("image inspection is disabled by the enforced profile");
    }
    rlf::solstice::SolsticeModel model =
        rlf::solstice::load_solstice_checkpoint(
            options.checkpoint,
            checkpoint_limits(options)
        );
    enforce_loaded_profile(options, model);
    model.set_backend(parse_backend(options.backend));
    const rlf::solstice::VisionAnalysis analysis =
        model.vision().analyze_file(*options.image);
    std::cout << "image=" << options.image->string() << '\n'
              << "size=" << analysis.width << 'x' << analysis.height << '\n'
              << "description=" << analysis.description << '\n'
              << "confidence=" << std::fixed << std::setprecision(4)
              << analysis.confidence << '\n'
              << "concepts=";
    for (const std::string& value : analysis.concepts) {
        std::cout << value << ' ';
    }
    std::cout << "\nregions=" << analysis.regions.size() << '\n';
    for (const rlf::solstice::VisualRegion& region : analysis.regions) {
        std::cout << "  mode=" << region.mode_id
                  << " box=" << region.x << ',' << region.y << ' '
                  << region.width << 'x' << region.height
                  << " patches=" << region.patch_count
                  << " label=" << region.concept_name
                  << " confidence=" << region.confidence << '\n';
    }
    return 0;
}

int run_inspect_checkpoint(const Options& options) {
    const rlf::solstice::SolsticeCheckpointSummary summary = options.enforce_profile
        ? rlf::solstice::inspect_solstice_checkpoint_for_profile(
            options.checkpoint,
            rlf::solstice::parse_profile(options.profile)
        )
        : rlf::solstice::inspect_solstice_checkpoint(
            options.checkpoint,
            checkpoint_limits(options)
        );
    std::cout << "checkpoint=" << options.checkpoint.string() << '\n'
              << "format_version=" << summary.format_version << '\n'
              << "file_bytes=" << summary.file_bytes << '\n'
              << "payload_checksum=0x" << std::hex << summary.payload_checksum
              << std::dec << '\n'
              << "seed=" << summary.seed << '\n';
    print_stats(summary.stats);
    return 0;
}


[[nodiscard]] rlf::solstice::RelationalPattern parse_relational_pattern(
    const std::string_view value
) {
    const std::size_t first = value.find(',');
    const std::size_t second = first == std::string_view::npos
        ? std::string_view::npos
        : value.find(',', first + 1U);
    if (first == std::string_view::npos || second == std::string_view::npos) {
        throw std::invalid_argument(
            "relational pattern must be subject,relation,object"
        );
    }
    return {
        std::string(value.substr(0U, first)),
        std::string(value.substr(first + 1U, second - first - 1U)),
        std::string(value.substr(second + 1U)),
    };
}

int run_train_facts(const Options& options) {
    if (options.manifest.empty()) {
        throw std::invalid_argument("train-facts requires --manifest");
    }
    rlf::solstice::SolsticeModel model = load_or_create(options);
    const std::size_t rows = for_each_tsv(
        options.manifest, 3U, 4U,
        [&model](const std::vector<std::string>& fields) {
            const double confidence = fields.size() == 4U
                ? parse_double(fields[3U], "fact confidence")
                : 1.0;
            model.learn_fact(
                fields[0U], fields[1U], fields[2U], confidence,
                "manifest"
            );
        }
    );
    rlf::solstice::save_solstice_checkpoint(options.checkpoint, model);
    std::cout << "trained_fact_rows=" << rows << '\n';
    print_stats(model.stats());
    return 0;
}

int run_train_rules(const Options& options) {
    if (options.manifest.empty()) {
        throw std::invalid_argument("train-rules requires --manifest");
    }
    rlf::solstice::SolsticeModel model = load_or_create(options);
    const std::size_t rows = for_each_tsv(
        options.manifest, 3U, 4U,
        [&model](const std::vector<std::string>& fields) {
            std::vector<rlf::solstice::RelationalPattern> premises;
            std::size_t start = 0U;
            while (start <= fields[1U].size()) {
                const std::size_t separator = fields[1U].find(';', start);
                const std::size_t end = separator == std::string::npos
                    ? fields[1U].size()
                    : separator;
                premises.push_back(parse_relational_pattern(
                    std::string_view(fields[1U]).substr(start, end - start)
                ));
                if (separator == std::string::npos) break;
                start = separator + 1U;
            }
            const double confidence = fields.size() == 4U
                ? parse_double(fields[3U], "rule confidence")
                : 1.0;
            model.learn_rule(
                fields[0U], premises,
                parse_relational_pattern(fields[2U]), confidence
            );
        }
    );
    rlf::solstice::save_solstice_checkpoint(options.checkpoint, model);
    std::cout << "trained_rule_rows=" << rows << '\n';
    print_stats(model.stats());
    return 0;
}

[[nodiscard]] rlf::solstice::DataAuditOptions audit_options(
    const Options& options
) {
    rlf::solstice::DataAuditOptions audit;
    audit.maximum_records = options.maximum_audit_records;
    audit.maximum_token_hash_cache_entries =
        options.maximum_token_hash_cache_entries;
    audit.maximum_text_shard_bytes = options.maximum_text_shard_bytes;
    audit.maximum_train_shard_bytes = options.maximum_train_shard_bytes;
    audit.near_duplicate_hamming_distance =
        options.near_duplicate_hamming_distance;
    audit.require_media_sha256 = options.require_media_hashes;
    audit.single_read_media_audit = !options.audit_media_double_read;
    return audit;
}

void emit_audit_report(
    const Options& options,
    const rlf::solstice::DataAuditReport& report
) {
    if (options.output.empty()) {
        rlf::solstice::write_data_audit_json(std::cout, report);
        return;
    }
    if (!options.output.parent_path().empty()) {
        std::filesystem::create_directories(options.output.parent_path());
    }
    std::ofstream output(options.output, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("unable to create data audit report");
    }
    rlf::solstice::write_data_audit_json(output, report);
    output.flush();
    if (!output) {
        throw std::runtime_error("failed while writing data audit report");
    }
}

int run_audit_data(const Options& options) {
    if (options.ledger.empty()) {
        throw std::invalid_argument("audit-data requires --ledger");
    }
    const rlf::solstice::DataLedger ledger =
        rlf::solstice::load_data_ledger(options.ledger);
    const rlf::solstice::DataAuditReport report =
        rlf::solstice::audit_data_ledger(ledger, audit_options(options));
    emit_audit_report(options, report);
    return report.valid ? 0 : 3;
}

struct VideoFrameRow final {
    std::size_t index{};
    std::filesystem::path path;
    std::string caption;
};

struct VideoSequenceRows final {
    std::string id;
    double frames_per_second{};
    std::string prompt;
    std::vector<VideoFrameRow> frames;
};

[[nodiscard]] std::pair<std::vector<VideoSequenceRows>, std::size_t>
load_video_sequences(const std::filesystem::path& path) {
    const std::filesystem::path base = path.parent_path();
    std::vector<VideoSequenceRows> sequences;
    std::map<std::string, std::size_t, std::less<>> sequence_indices;
    const std::size_t rows = for_each_tsv(
        path, 7U, 7U,
        [&sequences, &sequence_indices, &base](const std::vector<std::string>& row) {
            const std::size_t frame_index = parse_integer<std::size_t>(
                row[1U], "video frame index"
            );
            const double frames_per_second = parse_double(
                row[2U], "video frames per second"
            );
            if (row[0U].empty() || row[5U].empty() || frames_per_second <= 0.0) {
                throw std::invalid_argument("video sequence ID, prompt, and FPS must be valid");
            }
            auto [iterator, inserted] = sequence_indices.emplace(
                row[0U], sequences.size()
            );
            if (inserted) {
                sequences.push_back(VideoSequenceRows{
                    row[0U], frames_per_second, row[5U], {},
                });
            }
            VideoSequenceRows& sequence = sequences[iterator->second];
            if (sequence.frames_per_second != frames_per_second || sequence.prompt != row[5U]) {
                throw std::invalid_argument(
                    "video sequence rows must use one prompt and one FPS value"
                );
            }
            std::filesystem::path frame_path(row[3U]);
            if (frame_path.is_relative()) frame_path = base / frame_path;
            sequence.frames.push_back(VideoFrameRow{frame_index, frame_path, row[6U]});
        }
    );
    for (VideoSequenceRows& sequence : sequences) {
        std::sort(sequence.frames.begin(), sequence.frames.end(), [](const auto& left, const auto& right) {
            return left.index < right.index;
        });
        if (sequence.frames.size() < 2U) {
            throw std::invalid_argument(
                "each learned video sequence requires at least two ordered frames"
            );
        }
        for (std::size_t index = 0U; index < sequence.frames.size(); ++index) {
            if (sequence.frames[index].index != index) {
                throw std::invalid_argument(
                    "video frame indices must be unique and contiguous from zero"
                );
            }
        }
    }
    return {std::move(sequences), rows};
}

using SteadyClock = std::chrono::steady_clock;

[[nodiscard]] double elapsed_seconds(const SteadyClock::time_point start) {
    return std::chrono::duration<double>(SteadyClock::now() - start).count();
}

struct TrainingStageTiming final {
    double preprocessing_seconds{};
    double image_decode_seconds{};
    double update_seconds{};
};

[[nodiscard]] std::size_t train_audited_shard(
    rlf::solstice::SolsticeModel& model,
    const rlf::solstice::DataShard& shard,
    const std::filesystem::path& path,
    const std::size_t audited_records,
    TrainingStageTiming& timing,
    const bool separate_vision_analysis,
    const bool view_tsv_fields
) {
    using rlf::solstice::DataShardKind;
    const auto shard_start = SteadyClock::now();
    const double update_before = timing.update_seconds;
    const auto finish = [&timing, shard_start, update_before](const std::size_t records) {
        const double total = elapsed_seconds(shard_start);
        const double update_delta = timing.update_seconds - update_before;
        timing.preprocessing_seconds += std::max(0.0, total - update_delta);
        return records;
    };
    switch (shard.kind) {
    case DataShardKind::text: {
        const std::string text = read_text_file(path);
        const auto update_start = SteadyClock::now();
        model.train_text_corpus(text);
        timing.update_seconds += elapsed_seconds(update_start);
        return finish(audited_records);
    }
    case DataShardKind::dialogue:
        return finish(for_each_training_tsv(path, 2U, 3U, view_tsv_fields, [&model, &timing](const auto& row) {
            const auto update_start = SteadyClock::now();
            model.train_dialogue(
                row[0U],
                row[1U],
                row.size() == 3U
                    ? std::string_view(row[2U])
                    : std::string_view{}
            );
            timing.update_seconds += elapsed_seconds(update_start);
        }));
    case DataShardKind::instruction:
        return finish(for_each_training_tsv(path, 5U, 6U, view_tsv_fields, [&model, &timing](const auto& row) {
            const double quality = row.size() == 6U
                ? parse_double(row[5U], "instruction quality") : 1.0;
            const auto update_start = SteadyClock::now();
            model.train_instruction(row[0U], row[1U], row[2U], row[3U], row[4U], quality);
            timing.update_seconds += elapsed_seconds(update_start);
        }));
    case DataShardKind::preference:
        return finish(for_each_training_tsv(path, 3U, 5U, view_tsv_fields, [&model, &timing](const auto& row) {
            const std::string_view feedback = row.size() >= 4U
                ? std::string_view(row[3U])
                : std::string_view{};
            const double weight = row.size() == 5U
                ? parse_double(row[4U], "preference weight") : 1.0;
            const auto update_start = SteadyClock::now();
            model.train_preference(row[0U], row[1U], row[2U], feedback, weight);
            timing.update_seconds += elapsed_seconds(update_start);
        }));
    case DataShardKind::vision: {
        const std::filesystem::path base = path.parent_path();
        return finish(for_each_training_tsv(
            path, 2U, 3U, view_tsv_fields,
            [&model, &base, &timing, separate_vision_analysis](
                const auto& row
            ) {
                const std::string_view image_field(row[0U]);
                std::filesystem::path image_path{
                    image_field.begin(),
                    image_field.end(),
                };
                if (image_path.is_relative()) image_path = base / image_path;
                const auto decode_start = SteadyClock::now();
                const rlf::solstice::ImageData image = rlf::solstice::load_image(image_path);
                timing.image_decode_seconds += elapsed_seconds(decode_start);
                const auto update_start = SteadyClock::now();
                if (separate_vision_analysis) {
                    model.train_image_reference(image, row.back());
                } else {
                    model.train_image(image, row.back());
                }
                timing.update_seconds += elapsed_seconds(update_start);
            }
        ));
    }
    case DataShardKind::video: {
        auto [sequences, rows] = load_video_sequences(path);
        for (const VideoSequenceRows& sequence : sequences) {
            std::vector<rlf::solstice::ImageData> frames;
            std::vector<std::string> captions;
            frames.reserve(sequence.frames.size());
            captions.reserve(sequence.frames.size());
            for (const VideoFrameRow& frame : sequence.frames) {
                const auto decode_start = SteadyClock::now();
                frames.push_back(rlf::solstice::load_image(frame.path));
                timing.image_decode_seconds += elapsed_seconds(decode_start);
                captions.push_back(frame.caption);
            }
            const auto update_start = SteadyClock::now();
            static_cast<void>(model.train_video_sequence(
                sequence.id, sequence.prompt, sequence.frames_per_second,
                frames, captions
            ));
            timing.update_seconds += elapsed_seconds(update_start);
        }
        return finish(rows);
    }
    case DataShardKind::tools:
        return finish(for_each_training_tsv(path, 2U, 2U, view_tsv_fields, [&model, &timing](const auto& row) {
            const auto update_start = SteadyClock::now();
            model.train_tool_route(row[0U], row[1U]);
            timing.update_seconds += elapsed_seconds(update_start);
        }));
    case DataShardKind::facts:
        return finish(for_each_training_tsv(path, 3U, 4U, view_tsv_fields, [&model, &shard, &timing](const auto& row) {
            const double confidence = row.size() == 4U
                ? parse_double(row[3U], "fact confidence") : 1.0;
            const auto update_start = SteadyClock::now();
            model.learn_fact(row[0U], row[1U], row[2U], confidence, shard.source_uri);
            timing.update_seconds += elapsed_seconds(update_start);
        }));
    case DataShardKind::rules:
        return finish(for_each_training_tsv(path, 3U, 4U, view_tsv_fields, [&model, &timing](const auto& row) {
            std::vector<rlf::solstice::RelationalPattern> premises;
            const std::string_view premise_text(row[1U]);
            std::size_t start = 0U;
            while (start <= premise_text.size()) {
                const std::size_t separator = premise_text.find(';', start);
                const std::size_t end = separator == std::string::npos
                    ? premise_text.size() : separator;
                premises.push_back(parse_relational_pattern(
                    premise_text.substr(start, end - start)
                ));
                if (separator == std::string::npos) break;
                start = separator + 1U;
            }
            const double confidence = row.size() == 4U
                ? parse_double(row[3U], "rule confidence") : 1.0;
            const auto update_start = SteadyClock::now();
            model.learn_rule(
                row[0U], premises, parse_relational_pattern(row[2U]), confidence
            );
            timing.update_seconds += elapsed_seconds(update_start);
        }));
    }
    throw std::logic_error("unhandled audited shard kind");
}

struct TrainDataTelemetry final {
    double ledger_parse_seconds{};
    double audit_seconds{};
    double checkpoint_load_seconds{};
    TrainingStageTiming training;
    rlf::frontier::BackendOperationStats backend_operations;
    rlf::solstice::SparseRouterOperationStats sparse_router_operations;
    rlf::solstice::VisualTrainingOperationStats visual_training_operations;
    rlf::solstice::GroundingOperationStats grounding_operations;
    rlf::solstice::LanguageTrainingOperationStats language_training_operations;
    rlf::solstice::ToolRouterTrainingOperationStats tool_router_training_operations;
    rlf::solstice::GeneralTrainingOperationStats general_training_operations;
    double checkpoint_write_seconds{};
    double total_wall_seconds{};
    double total_cpu_seconds{};
    std::uint64_t bytes_read{};
    std::uint64_t bytes_written{};
    std::uint64_t language_tokens_processed{};
    std::uint64_t images_processed{};
    std::uint64_t video_frames_processed{};
    std::optional<std::uint64_t> peak_ram_bytes;
    std::size_t audited_records{};
    std::size_t trained_records{};
    std::size_t trained_shards{};
    std::size_t resumed_shards{};
    std::uint64_t target_training_records{};
    std::uint64_t cumulative_records_before{};
    std::uint64_t cumulative_records_after{};
    std::uint64_t target_training_tokens{};
    std::uint64_t cumulative_tokens_before{};
    std::uint64_t cumulative_tokens_after{};
    bool record_target_reached{true};
    bool token_target_reached{true};
    bool training_target_reached{true};
    bool intentional_shard_stop{};
    bool separate_vision_analysis{};
    std::string tsv_field_policy{"copied"};
};

[[nodiscard]] std::optional<std::uint64_t> peak_resident_bytes() noexcept {
#if defined(__linux__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0) {
        return std::nullopt;
    }
    constexpr std::uint64_t bytes_per_kibibyte = 1'024U;
    const auto peak_kibibytes = static_cast<std::uint64_t>(usage.ru_maxrss);
    if (peak_kibibytes >
        std::numeric_limits<std::uint64_t>::max() / bytes_per_kibibyte) {
        return std::nullopt;
    }
    return peak_kibibytes * bytes_per_kibibyte;
#else
    return std::nullopt;
#endif
}

void write_train_data_telemetry(
    const Options& options,
    const rlf::solstice::DataLedger& ledger,
    const TrainDataTelemetry& telemetry
) {
    if (options.telemetry.empty()) return;
    if (!options.telemetry.parent_path().empty()) {
        std::filesystem::create_directories(options.telemetry.parent_path());
    }
    std::ofstream output(options.telemetry, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("unable to create train-data telemetry");
    const double records_per_second = telemetry.total_wall_seconds > 0.0
        ? static_cast<double>(telemetry.trained_records) / telemetry.total_wall_seconds
        : 0.0;
    const double tokens_per_second = telemetry.total_wall_seconds > 0.0
        ? static_cast<double>(telemetry.language_tokens_processed) /
            telemetry.total_wall_seconds
        : 0.0;
    const double images_per_second = telemetry.total_wall_seconds > 0.0
        ? static_cast<double>(telemetry.images_processed) / telemetry.total_wall_seconds
        : 0.0;
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"rlf-train-data-telemetry-v1\",\n"
           << "  \"hardware_profile\": " << std::quoted(options.profile) << ",\n"
           << "  \"backend\": " << std::quoted(options.backend) << ",\n"
           << "  \"ledger_sha256\": " << std::quoted(ledger.sha256) << ",\n"
           << "  \"checkpoint_path\": "
           << std::quoted(options.checkpoint.string()) << ",\n"
           << "  \"training_performed\": "
           << (telemetry.trained_shards > 0U ? "true" : "false") << ",\n"
           << "  \"separate_vision_analysis\": "
           << (telemetry.separate_vision_analysis ? "true" : "false") << ",\n"
           << "  \"tsv_field_policy\": "
           << std::quoted(telemetry.tsv_field_policy) << ",\n"
           << "  \"total_wall_seconds\": " << telemetry.total_wall_seconds << ",\n"
           << "  \"cpu_seconds\": " << telemetry.total_cpu_seconds << ",\n"
           << "  \"ledger_parse_seconds\": " << telemetry.ledger_parse_seconds << ",\n"
           << "  \"audit_and_dedup_seconds\": " << telemetry.audit_seconds << ",\n"
           << "  \"checkpoint_load_seconds\": " << telemetry.checkpoint_load_seconds << ",\n"
           << "  \"preprocessing_seconds\": "
           << telemetry.training.preprocessing_seconds << ",\n"
           << "  \"image_decode_seconds\": "
           << telemetry.training.image_decode_seconds << ",\n"
           << "  \"learning_update_seconds\": "
           << telemetry.training.update_seconds << ",\n"
           << "  \"checkpoint_write_seconds\": "
           << telemetry.checkpoint_write_seconds << ",\n"
           << "  \"evaluation_seconds\": null,\n"
           << "  \"retrieval_seconds\": null,\n"
           << "  \"consolidation_seconds\": null,\n"
           << "  \"gpu_active_seconds\": null,\n"
           << "  \"energy_joules\": null,\n"
           << "  \"peak_vram_bytes\": null,\n"
           << "  \"peak_ram_bytes\": ";
    if (telemetry.peak_ram_bytes.has_value()) {
        output << *telemetry.peak_ram_bytes;
    } else {
        output << "null";
    }
    output << ",\n"
           << "  \"backend_operations\": {\n"
           << "    \"batch_cosine_calls\": "
           << telemetry.backend_operations.batch_cosine_calls << ",\n"
           << "    \"indexed_batch_cosine_calls\": "
           << telemetry.backend_operations.indexed_batch_cosine_calls << ",\n"
           << "    \"indexed_cosine_pairs\": "
           << telemetry.backend_operations.indexed_cosine_pairs << ",\n"
           << "    \"cached_batch_cosine_calls\": "
           << telemetry.backend_operations.cached_batch_cosine_calls << ",\n"
           << "    \"candidate_cache_uploads\": "
           << telemetry.backend_operations.candidate_cache_uploads << ",\n"
           << "    \"candidate_cache_hits\": "
           << telemetry.backend_operations.candidate_cache_hits << ",\n"
           << "    \"inline_norm_cosine_calls\": "
           << telemetry.backend_operations.inline_norm_cosine_calls << ",\n"
           << "    \"precomputed_norm_cosine_calls\": "
           << telemetry.backend_operations.precomputed_norm_cosine_calls << ",\n"
           << "    \"host_batch_cosine_calls\": "
           << telemetry.backend_operations.host_batch_cosine_calls << ",\n"
           << "    \"device_batch_cosine_calls\": "
           << telemetry.backend_operations.device_batch_cosine_calls << ",\n"
           << "    \"host_cosine_fma_operations\": "
           << telemetry.backend_operations.host_cosine_fma_operations << ",\n"
           << "    \"avoided_device_cosine_fma_operations\": "
           << telemetry.backend_operations.avoided_device_cosine_fma_operations << ",\n"
           << "    \"candidate_norm_cache_uploads\": "
           << telemetry.backend_operations.candidate_norm_cache_uploads << ",\n"
           << "    \"avoided_pairwise_norm_fma_operations\": "
           << telemetry.backend_operations.avoided_pairwise_norm_fma_operations << ",\n"
           << "    \"host_precomputed_norm_fma_operations\": "
           << telemetry.backend_operations.host_precomputed_norm_fma_operations << ",\n"
           << "    \"local_update_calls\": "
           << telemetry.backend_operations.local_update_calls << ",\n"
           << "    \"host_local_update_calls\": "
           << telemetry.backend_operations.host_local_update_calls << ",\n"
           << "    \"device_local_update_calls\": "
           << telemetry.backend_operations.device_local_update_calls << ",\n"
           << "    \"host_to_device_bytes\": "
           << telemetry.backend_operations.host_to_device_bytes << ",\n"
           << "    \"device_to_host_bytes\": "
           << telemetry.backend_operations.device_to_host_bytes << ",\n"
           << "    \"kernel_launches\": "
           << telemetry.backend_operations.kernel_launches << ",\n"
           << "    \"stream_synchronizations\": "
           << telemetry.backend_operations.stream_synchronizations << ",\n"
           << "    \"avoided_host_to_device_bytes\": "
           << telemetry.backend_operations.avoided_host_to_device_bytes << ",\n"
           << "    \"avoided_device_to_host_bytes\": "
           << telemetry.backend_operations.avoided_device_to_host_bytes << ",\n"
           << "    \"avoided_kernel_launches\": "
           << telemetry.backend_operations.avoided_kernel_launches << ",\n"
           << "    \"avoided_stream_synchronizations\": "
           << telemetry.backend_operations.avoided_stream_synchronizations << ",\n"
           << "    \"host_local_update_maximum_dimensions\": "
           << telemetry.backend_operations.host_local_update_maximum_dimensions << ",\n"
           << "    \"host_batch_cosine_maximum_fma_operations\": "
           << telemetry.backend_operations.host_batch_cosine_maximum_fma_operations << ",\n"
           << "    \"hybrid_local_updates\": "
           << (telemetry.backend_operations.hybrid_local_updates ? "true" : "false")
           << ",\n    \"hybrid_small_batch_cosine\": "
           << (telemetry.backend_operations.hybrid_small_batch_cosine ? "true" : "false")
           << ",\n    \"precomputed_cached_cosine_norms\": "
           << (telemetry.backend_operations.precomputed_cached_cosine_norms ? "true" : "false")
           << "\n  },\n"
           << "  \"sparse_router_operations\": {\n"
           << "    \"full_rebuilds\": "
           << telemetry.sparse_router_operations.full_rebuilds << ",\n"
           << "    \"vectors_rebuilt\": "
           << telemetry.sparse_router_operations.vectors_rebuilt << ",\n"
           << "    \"incremental_updates\": "
           << telemetry.sparse_router_operations.incremental_updates << ",\n"
           << "    \"vectors_incrementally_updated\": "
           << telemetry.sparse_router_operations.vectors_incrementally_updated << ",\n"
           << "    \"vectors_appended\": "
           << telemetry.sparse_router_operations.vectors_appended
           << "\n  },\n"
           << "  \"visual_training_operations\": {\n"
           << "    \"concept_update_lookups\": "
           << telemetry.visual_training_operations.concept_update_lookups << ",\n"
           << "    \"linear_concept_comparisons\": "
           << telemetry.visual_training_operations.linear_concept_comparisons << ",\n"
           << "    \"indexed_concept_lookups\": "
           << telemetry.visual_training_operations.indexed_concept_lookups << ",\n"
           << "    \"concept_index_rebuilds\": "
           << telemetry.visual_training_operations.concept_index_rebuilds << ",\n"
           << "    \"concept_index_entries_built\": "
           << telemetry.visual_training_operations.concept_index_entries_built << ",\n"
           << "    \"example_duplicate_lookups\": "
           << telemetry.visual_training_operations.example_duplicate_lookups << ",\n"
           << "    \"linear_example_comparisons\": "
           << telemetry.visual_training_operations.linear_example_comparisons << ",\n"
           << "    \"indexed_example_candidates\": "
           << telemetry.visual_training_operations.indexed_example_candidates << ",\n"
           << "    \"example_index_rebuilds\": "
           << telemetry.visual_training_operations.example_index_rebuilds << ",\n"
           << "    \"example_index_entries_built\": "
           << telemetry.visual_training_operations.example_index_entries_built << ",\n"
           << "    \"mode_id_lookups\": "
           << telemetry.visual_training_operations.mode_id_lookups << ",\n"
           << "    \"mode_id_index_full_rebuilds\": "
           << telemetry.visual_training_operations.mode_id_index_full_rebuilds << ",\n"
           << "    \"mode_id_index_entries_rebuilt\": "
           << telemetry.visual_training_operations.mode_id_index_entries_rebuilt << ",\n"
           << "    \"mode_id_index_incremental_inserts\": "
           << telemetry.visual_training_operations.mode_id_index_incremental_inserts << ",\n"
           << "    \"region_mode_id_lookups\": "
           << telemetry.visual_training_operations.region_mode_id_lookups << ",\n"
           << "    \"linear_region_mode_comparisons\": "
           << telemetry.visual_training_operations.linear_region_mode_comparisons << ",\n"
           << "    \"indexed_region_mode_lookups\": "
           << telemetry.visual_training_operations.indexed_region_mode_lookups
           << "\n  },\n"
           << "  \"grounding_operations\": {\n"
           << "    \"link_lookups\": "
           << telemetry.grounding_operations.link_lookups << ",\n"
           << "    \"full_lookup_entries_rebuilt\": "
           << telemetry.grounding_operations.full_lookup_entries_rebuilt << ",\n"
           << "    \"indexed_link_candidates_examined\": "
           << telemetry.grounding_operations.indexed_link_candidates_examined << ",\n"
           << "    \"incremental_posting_inserts\": "
           << telemetry.grounding_operations.incremental_posting_inserts << ",\n"
           << "    \"confidence_recomputations\": "
           << telemetry.grounding_operations.confidence_recomputations << ",\n"
           << "    \"full_confidence_sweep_entries\": "
           << telemetry.grounding_operations.full_confidence_sweep_entries << ",\n"
           << "    \"derived_sort_entries\": "
           << telemetry.grounding_operations.derived_sort_entries << ",\n"
           << "    \"mode_query_full_scan_entries\": "
           << telemetry.grounding_operations.mode_query_full_scan_entries << ",\n"
           << "    \"mode_query_indexed_candidates\": "
           << telemetry.grounding_operations.mode_query_indexed_candidates << ",\n"
           << "    \"concept_query_full_scan_entries\": "
           << telemetry.grounding_operations.concept_query_full_scan_entries << ",\n"
           << "    \"concept_query_indexed_candidates\": "
           << telemetry.grounding_operations.concept_query_indexed_candidates
           << "\n  },\n"
           << "  \"language_training_operations\": {\n"
           << "    \"dialogue_training_calls\": "
           << telemetry.language_training_operations.dialogue_training_calls << ",\n"
           << "    \"dialogue_tokenizer_encode_calls\": "
           << telemetry.language_training_operations.dialogue_tokenizer_encode_calls
           << ",\n"
           << "    \"redundant_dialogue_encode_calls_avoided\": "
           << telemetry.language_training_operations
                  .redundant_dialogue_encode_calls_avoided
           << ",\n"
           << "    \"episode_duplicate_updates\": "
           << telemetry.language_training_operations.episode_duplicate_updates << ",\n"
           << "    \"episode_insert_attempts\": "
           << telemetry.language_training_operations.episode_insert_attempts << ",\n"
           << "    \"episode_inserts\": "
           << telemetry.language_training_operations.episode_inserts << ",\n"
           << "    \"episode_replacements\": "
           << telemetry.language_training_operations.episode_replacements << ",\n"
           << "    \"episode_capacity_skips\": "
           << telemetry.language_training_operations.episode_capacity_skips << ",\n"
           << "    \"context_insert_attempts\": "
           << telemetry.language_training_operations.context_insert_attempts << ",\n"
           << "    \"context_inserts\": "
           << telemetry.language_training_operations.context_inserts << ",\n"
           << "    \"context_replacements\": "
           << telemetry.language_training_operations.context_replacements << ",\n"
           << "    \"context_capacity_skips\": "
           << telemetry.language_training_operations.context_capacity_skips << ",\n"
           << "    \"outcome_update_lookups\": "
           << telemetry.language_training_operations.outcome_update_lookups << ",\n"
           << "    \"linear_outcome_comparisons\": "
           << telemetry.language_training_operations.linear_outcome_comparisons << ",\n"
           << "    \"indexed_outcome_lookups\": "
           << telemetry.language_training_operations.indexed_outcome_lookups << ",\n"
           << "    \"outcome_index_builds\": "
           << telemetry.language_training_operations.outcome_index_builds << ",\n"
           << "    \"outcome_index_entries_built\": "
           << telemetry.language_training_operations.outcome_index_entries_built << ",\n"
           << "    \"outcome_index_incremental_inserts\": "
           << telemetry.language_training_operations.outcome_index_incremental_inserts
           << "\n  },\n"
           << "  \"tool_router_training_operations\": {\n"
           << "    \"training_rows\": "
           << telemetry.tool_router_training_operations.training_rows << ",\n"
           << "    \"route_insert_attempts\": "
           << telemetry.tool_router_training_operations.route_insert_attempts << ",\n"
           << "    \"route_inserts\": "
           << telemetry.tool_router_training_operations.route_inserts << ",\n"
           << "    \"route_capacity_failures\": "
           << telemetry.tool_router_training_operations.route_capacity_failures << ",\n"
           << "    \"keyword_update_lookups\": "
           << telemetry.tool_router_training_operations.keyword_update_lookups << ",\n"
           << "    \"linear_keyword_comparisons\": "
           << telemetry.tool_router_training_operations.linear_keyword_comparisons << ",\n"
           << "    \"indexed_keyword_lookups\": "
           << telemetry.tool_router_training_operations.indexed_keyword_lookups << ",\n"
           << "    \"keyword_insert_attempts\": "
           << telemetry.tool_router_training_operations.keyword_insert_attempts << ",\n"
           << "    \"keyword_inserts\": "
           << telemetry.tool_router_training_operations.keyword_inserts << ",\n"
           << "    \"keyword_capacity_skips\": "
           << telemetry.tool_router_training_operations.keyword_capacity_skips << ",\n"
           << "    \"keyword_index_rebuilds\": "
           << telemetry.tool_router_training_operations.keyword_index_rebuilds << ",\n"
           << "    \"keyword_index_entries_built\": "
           << telemetry.tool_router_training_operations.keyword_index_entries_built << ",\n"
           << "    \"keyword_index_incremental_inserts\": "
           << telemetry.tool_router_training_operations.keyword_index_incremental_inserts
           << "\n  },\n"
           << "  \"general_training_operations\": {\n"
           << "    \"instruction_duplicate_prefilter_lookups\": "
           << telemetry.general_training_operations.instruction_duplicate_prefilter_lookups
           << ",\n"
           << "    \"instruction_duplicate_retrievals\": "
           << telemetry.general_training_operations.instruction_duplicate_retrievals
           << ",\n"
           << "    \"instruction_duplicate_retrievals_avoided\": "
           << telemetry.general_training_operations.instruction_duplicate_retrievals_avoided
           << ",\n"
           << "    \"instruction_index_rebuilds\": "
           << telemetry.general_training_operations.instruction_index_rebuilds << ",\n"
           << "    \"instruction_index_entries_built\": "
           << telemetry.general_training_operations.instruction_index_entries_built << ",\n"
           << "    \"instruction_index_incremental_inserts\": "
           << telemetry.general_training_operations.instruction_index_incremental_inserts
           << ",\n"
           << "    \"preference_duplicate_lookups\": "
           << telemetry.general_training_operations.preference_duplicate_lookups << ",\n"
           << "    \"linear_preference_comparisons\": "
           << telemetry.general_training_operations.linear_preference_comparisons << ",\n"
           << "    \"indexed_preference_candidates\": "
           << telemetry.general_training_operations.indexed_preference_candidates << ",\n"
           << "    \"preference_index_rebuilds\": "
           << telemetry.general_training_operations.preference_index_rebuilds << ",\n"
           << "    \"preference_index_entries_built\": "
           << telemetry.general_training_operations.preference_index_entries_built << ",\n"
           << "    \"preference_index_incremental_inserts\": "
           << telemetry.general_training_operations.preference_index_incremental_inserts
           << ",\n"
           << "    \"active_learning_duplicate_lookups\": "
           << telemetry.general_training_operations.active_learning_duplicate_lookups << ",\n"
           << "    \"linear_active_learning_comparisons\": "
           << telemetry.general_training_operations.linear_active_learning_comparisons << ",\n"
           << "    \"indexed_active_learning_candidates\": "
           << telemetry.general_training_operations.indexed_active_learning_candidates << ",\n"
           << "    \"active_learning_index_rebuilds\": "
           << telemetry.general_training_operations.active_learning_index_rebuilds << ",\n"
           << "    \"active_learning_index_entries_built\": "
           << telemetry.general_training_operations.active_learning_index_entries_built << ",\n"
           << "    \"active_learning_index_incremental_inserts\": "
           << telemetry.general_training_operations.active_learning_index_incremental_inserts
           << "\n  },\n"
           << "  \"bytes_read\": " << telemetry.bytes_read << ",\n"
           << "  \"bytes_written\": " << telemetry.bytes_written << ",\n"
           << "  \"checkpoint_bytes\": " << telemetry.bytes_written << ",\n"
           << "  \"audited_records\": " << telemetry.audited_records << ",\n"
           << "  \"trained_records\": " << telemetry.trained_records << ",\n"
           << "  \"target_training_records\": ";
    if (telemetry.target_training_records > 0U) {
        output << telemetry.target_training_records;
    } else {
        output << "null";
    }
    output << ",\n"
           << "  \"cumulative_training_records_before\": "
           << telemetry.cumulative_records_before << ",\n"
           << "  \"cumulative_training_records_after\": "
           << telemetry.cumulative_records_after << ",\n"
           << "  \"training_target_reached\": "
           << (telemetry.training_target_reached ? "true" : "false") << ",\n"
           << "  \"intentional_shard_stop\": "
           << (telemetry.intentional_shard_stop ? "true" : "false") << ",\n"
           << "  \"trained_shards\": " << telemetry.trained_shards << ",\n"
           << "  \"resumed_shards\": " << telemetry.resumed_shards << ",\n"
           << "  \"target_training_tokens\": "
           << telemetry.target_training_tokens << ",\n"
           << "  \"cumulative_tokens_before\": "
           << telemetry.cumulative_tokens_before << ",\n"
           << "  \"cumulative_tokens_after\": "
           << telemetry.cumulative_tokens_after << ",\n"
           << "  \"token_target_reached\": "
           << (telemetry.token_target_reached ? "true" : "false") << ",\n"
           << "  \"language_tokens_processed\": "
           << telemetry.language_tokens_processed << ",\n"
           << "  \"images_processed\": " << telemetry.images_processed << ",\n"
           << "  \"video_frames_processed\": "
           << telemetry.video_frames_processed << ",\n"
           << "  \"records_per_second\": " << records_per_second << ",\n"
           << "  \"tokens_per_second\": " << tokens_per_second << ",\n"
           << "  \"images_per_second\": " << images_per_second << ",\n"
           << "  \"claim_boundary\": \"GPU, energy, peak VRAM, evaluation, retrieval, and consolidation require separately bound instrumentation; Linux peak RAM is process high-water RSS, and null is not zero.\"\n"
           << "}\n";
    output.flush();
    if (!output) throw std::runtime_error("failed while writing train-data telemetry");
}

int run_train_data(const Options& options) {
    const auto total_start = SteadyClock::now();
    const std::clock_t cpu_start = std::clock();
    TrainDataTelemetry telemetry;
    telemetry.separate_vision_analysis = options.separate_vision_analysis;
    if (options.ledger.empty()) {
        throw std::invalid_argument("train-data requires --ledger");
    }
    const auto ledger_start = SteadyClock::now();
    const rlf::solstice::DataLedger ledger =
        rlf::solstice::load_data_ledger(options.ledger);
    telemetry.ledger_parse_seconds = elapsed_seconds(ledger_start);
    const auto audit_start = SteadyClock::now();
    const rlf::solstice::DataAuditReport audit =
        rlf::solstice::audit_data_ledger(ledger, audit_options(options));
    telemetry.audit_seconds = elapsed_seconds(audit_start);
    telemetry.audited_records = audit.records;
    telemetry.bytes_read = audit.shard_bytes + audit.referenced_media_bytes;
    emit_audit_report(options, audit);
    if (!audit.valid) {
        throw std::runtime_error("data ledger audit failed; training was not started");
    }
    if (options.target_training_records == 0U &&
        options.profile == "general-v100-32g-500m") {
        throw std::invalid_argument(
            "general-v100-32g-500m train-data requires --target-training-records"
        );
    }
    if (options.target_training_tokens == 0U &&
        options.profile == "general-h200-141g-30t") {
        throw std::invalid_argument(
            "general-h200-141g-30t train-data requires --target-training-tokens"
        );
    }
    std::uint64_t ledger_training_records = 0U;
    for (std::size_t index = 0U; index < ledger.shards.size(); ++index) {
        if (ledger.shards[index].split != rlf::solstice::DataShardSplit::train) {
            continue;
        }
        const std::uint64_t records = audit.audited_shards[index].records;
        if (records > std::numeric_limits<std::uint64_t>::max() -
                ledger_training_records) {
            throw std::runtime_error("training ledger record count overflow");
        }
        ledger_training_records += records;
    }
    if (options.target_training_records > 0U &&
        ledger_training_records != options.target_training_records) {
        throw std::runtime_error(
            "training ledger contains " + std::to_string(ledger_training_records) +
            " train records; exact target is " +
            std::to_string(options.target_training_records)
        );
    }
    const rlf::solstice::SolsticeProfile profile =
        rlf::solstice::parse_profile(options.profile);
    const bool fifty_million_rtx_profile =
        profile == rlf::solstice::SolsticeProfile::rtx_pro_6000_96g ||
        profile == rlf::solstice::SolsticeProfile::general_rtx_pro_6000_96g;
    const bool view_tsv_fields =
        fifty_million_rtx_profile && !options.copy_tsv_fields;
    telemetry.tsv_field_policy = view_tsv_fields ? "views" : "copied";
    if (options.enforce_profile) {
        for (const rlf::solstice::DataShard& shard : ledger.shards) {
            if (shard.split != rlf::solstice::DataShardSplit::train) continue;
            if (shard.kind == rlf::solstice::DataShardKind::vision &&
                !rlf::solstice::profile_allows_vision(profile)) {
                throw std::runtime_error(
                    "vision training shard is forbidden by enforced text-only profile"
                );
            }
            if (shard.kind == rlf::solstice::DataShardKind::video &&
                !rlf::solstice::profile_allows_video(profile)) {
                throw std::runtime_error(
                    "video training shard requires video-rtx-pro-6000-96g"
                );
            }
        }
    }
    const auto checkpoint_load_start = SteadyClock::now();
    rlf::solstice::SolsticeModel model = load_or_create(options);
    telemetry.checkpoint_load_seconds = elapsed_seconds(checkpoint_load_start);
    const rlf::solstice::SolsticeStats initial_stats = model.stats();
    telemetry.target_training_records = options.target_training_records;
    telemetry.cumulative_records_before = initial_stats.audited_training_records;
    telemetry.target_training_tokens = options.target_training_tokens;
    telemetry.cumulative_tokens_before = initial_stats.language_tokens_seen;
    if (options.target_training_records > 0U &&
        initial_stats.audited_training_records > options.target_training_records) {
        throw std::runtime_error("checkpoint already exceeds the exact training target");
    }
    if (options.target_training_tokens > 0U &&
        initial_stats.language_tokens_seen > options.target_training_tokens) {
        throw std::runtime_error("checkpoint already exceeds the exact token target");
    }
    if (options.target_training_records > 0U) {
        std::uint64_t ledger_completed_records = 0U;
        for (const rlf::solstice::TrainingShardRecord& completed :
             model.completed_training_shards()) {
            bool found = false;
            for (std::size_t index = 0U; index < ledger.shards.size(); ++index) {
                if (ledger.shards[index].split !=
                    rlf::solstice::DataShardSplit::train) {
                    continue;
                }
                const rlf::solstice::AuditedShard& audited =
                    audit.audited_shards[index];
                if (completed.shard_id == ledger.shards[index].shard_id &&
                    completed.shard_sha256 == audited.actual_sha256) {
                    found = true;
                    if (completed.records >
                        std::numeric_limits<std::uint64_t>::max() -
                            ledger_completed_records) {
                        throw std::runtime_error(
                            "completed training record count overflow"
                        );
                    }
                    ledger_completed_records += completed.records;
                    break;
                }
            }
            if (!found) {
                throw std::runtime_error(
                    "exact-target ledger omits or changes completed shard: " +
                    completed.shard_id
                );
            }
        }
        if (ledger_completed_records != initial_stats.audited_training_records) {
            throw std::runtime_error(
                "checkpoint completed-shard accounting is inconsistent"
            );
        }
    }
    const std::filesystem::path base = ledger.source_path.parent_path();
    std::size_t trained_shards = 0U;
    std::size_t skipped_shards = 0U;
    bool intentional_shard_stop = false;
    for (std::size_t index = 0U; index < ledger.shards.size(); ++index) {
        const rlf::solstice::DataShard& shard = ledger.shards[index];
        if (shard.split != rlf::solstice::DataShardSplit::train) continue;
        const rlf::solstice::AuditedShard& audited = audit.audited_shards[index];
        if (model.has_completed_training_shard(audited.actual_sha256)) {
            ++skipped_shards;
            continue;
        }
        for (const rlf::solstice::TrainingShardRecord& completed :
             model.completed_training_shards()) {
            if (completed.shard_id == shard.shard_id &&
                completed.shard_sha256 != audited.actual_sha256) {
                throw std::runtime_error(
                    "training shard ID was reused with changed content: " + shard.shard_id
                );
            }
        }
        const std::filesystem::path path = shard.path.is_absolute()
            ? shard.path : base / shard.path;
        if (options.target_training_records > 0U) {
            const std::uint64_t current_records = model.stats().audited_training_records;
            if (audited.records > options.target_training_records - current_records) {
                throw std::runtime_error(
                    "next immutable shard would exceed the exact training target"
                );
            }
        }
        const std::size_t trained_records = train_audited_shard(
            model, shard, path, audited.records, telemetry.training,
            options.separate_vision_analysis,
            view_tsv_fields
        );
        telemetry.trained_records += trained_records;
        const rlf::solstice::SolsticeStats shard_stats = model.stats();
        if (options.target_training_tokens > 0U &&
            shard_stats.language_tokens_seen > options.target_training_tokens) {
            throw std::runtime_error(
                "next immutable shard would exceed the exact token target"
            );
        }
        if (profile == rlf::solstice::SolsticeProfile::general_h200_141g_30t) {
            const auto language_operations =
                model.language_training_operation_stats();
            const auto tool_operations =
                model.tool_router().training_operation_stats();
            if (language_operations.episode_capacity_skips != 0U ||
                language_operations.context_capacity_skips != 0U ||
                tool_operations.keyword_capacity_skips != 0U) {
                throw std::runtime_error(
                    "H200 frontier training reached a state capacity ceiling"
                );
            }
        }
        if (audited.file_bytes > std::numeric_limits<std::uint64_t>::max() -
            audited.referenced_media_bytes) {
            throw std::runtime_error("training shard byte count overflow");
        }
        model.record_completed_training_shard({
            shard.shard_id,
            std::string(rlf::solstice::to_string(shard.kind)),
            audited.actual_sha256,
            ledger.sha256,
            shard.source_uri,
            shard.license,
            static_cast<std::uint64_t>(trained_records),
            audited.file_bytes + audited.referenced_media_bytes,
        });
        const auto checkpoint_write_start = SteadyClock::now();
        rlf::solstice::save_solstice_checkpoint(options.checkpoint, model);
        telemetry.checkpoint_write_seconds += elapsed_seconds(checkpoint_write_start);
        ++trained_shards;
        if (options.maximum_new_shards > 0U &&
            trained_shards >= options.maximum_new_shards) {
            intentional_shard_stop = true;
            break;
        }
    }
    std::cout << "trained_shards=" << trained_shards << '\n'
              << "resumed_shards=" << skipped_shards << '\n'
              << "checkpoint=" << options.checkpoint.string() << '\n';
    print_stats(model.stats());
    telemetry.trained_shards = trained_shards;
    telemetry.resumed_shards = skipped_shards;
    std::error_code size_error;
    telemetry.bytes_written = std::filesystem::file_size(options.checkpoint, size_error);
    if (size_error) telemetry.bytes_written = 0U;
    const rlf::solstice::SolsticeStats final_stats = model.stats();
    telemetry.cumulative_records_after = final_stats.audited_training_records;
    telemetry.cumulative_tokens_after = final_stats.language_tokens_seen;
    telemetry.record_target_reached = options.target_training_records == 0U ||
        final_stats.audited_training_records == options.target_training_records;
    telemetry.token_target_reached = options.target_training_tokens == 0U ||
        final_stats.language_tokens_seen == options.target_training_tokens;
    telemetry.training_target_reached =
        telemetry.record_target_reached && telemetry.token_target_reached;
    telemetry.intentional_shard_stop = intentional_shard_stop;
    if (!telemetry.training_target_reached && !intentional_shard_stop) {
        throw std::runtime_error(
            "checkpoint did not reach the exact cumulative training target"
        );
    }
    if (final_stats.language_tokens_seen < initial_stats.language_tokens_seen ||
        final_stats.images_seen < initial_stats.images_seen ||
        final_stats.video_frames_seen < initial_stats.video_frames_seen) {
        throw std::runtime_error("training statistics decreased during train-data");
    }
    telemetry.language_tokens_processed =
        final_stats.language_tokens_seen - initial_stats.language_tokens_seen;
    telemetry.images_processed = final_stats.images_seen - initial_stats.images_seen;
    telemetry.video_frames_processed =
        final_stats.video_frames_seen - initial_stats.video_frames_seen;
    telemetry.backend_operations = model.backend_operation_stats();
    telemetry.sparse_router_operations = model.sparse_router_operation_stats();
    telemetry.visual_training_operations = model.visual_training_operation_stats();
    telemetry.grounding_operations = model.grounding_operation_stats();
    telemetry.language_training_operations =
        model.language_training_operation_stats();
    telemetry.tool_router_training_operations =
        model.tool_router().training_operation_stats();
    telemetry.general_training_operations =
        model.general_training_operation_stats();
    telemetry.total_wall_seconds = elapsed_seconds(total_start);
    const std::clock_t cpu_end = std::clock();
    if (cpu_start != static_cast<std::clock_t>(-1) &&
        cpu_end != static_cast<std::clock_t>(-1)) {
        telemetry.total_cpu_seconds = static_cast<double>(cpu_end - cpu_start) /
            static_cast<double>(CLOCKS_PER_SEC);
    }
    telemetry.peak_ram_bytes = peak_resident_bytes();
    write_train_data_telemetry(options, ledger, telemetry);
    if (!telemetry.training_target_reached) return 4;
    return 0;
}

[[nodiscard]] std::string json_escape(const std::string_view value) {
    std::ostringstream output;
    output << '"';
    for (const char character : value) {
        const unsigned char byte = static_cast<unsigned char>(character);
        switch (character) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (byte < 0x20U) {
                output << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                       << static_cast<unsigned int>(byte) << std::dec;
            } else {
                output << character;
            }
        }
    }
    output << '"';
    return output.str();
}

int run_generate_video(const Options& options) {
    if (options.prompt.empty() || options.output.empty()) {
        throw std::invalid_argument(
            "generate-video requires --prompt and a new --output directory"
        );
    }
    const auto profile = rlf::solstice::parse_profile(options.profile);
    if (!rlf::solstice::profile_allows_video(profile)) {
        throw std::invalid_argument(
            "generate-video requires --profile video-rtx-pro-6000-96g"
        );
    }
    if (!std::filesystem::is_regular_file(options.checkpoint)) {
        throw std::invalid_argument("generate-video requires an existing checkpoint");
    }
    if (std::filesystem::exists(options.output)) {
        throw std::invalid_argument("generate-video output path already exists");
    }
    rlf::solstice::SolsticeModel model =
        rlf::solstice::load_solstice_checkpoint(
            options.checkpoint,
            checkpoint_limits(options)
        );
    if (!rlf::solstice::profile_config_matches(profile, model.config())) {
        throw std::runtime_error("checkpoint is not the enforced RTX video profile");
    }
    const rlf::solstice::VideoGeneration generation = model.generate_video(
        options.prompt, options.video_frames
    );
    const std::string checkpoint_sha = rlf::core::sha256_hex(
        rlf::core::sha256_file(options.checkpoint)
    );
    std::filesystem::path temporary = options.output;
    temporary += ".tmp." + checkpoint_sha.substr(0U, 12U);
    if (std::filesystem::exists(temporary)) {
        throw std::runtime_error("stale video generation temporary directory exists");
    }
    try {
        std::filesystem::create_directories(temporary / "frames");
        std::ofstream frame_manifest(
            temporary / "frame_manifest.tsv", std::ios::binary | std::ios::trunc
        );
        if (!frame_manifest) throw std::runtime_error("unable to create frame manifest");
        frame_manifest << "frame_index\trelative_path\tsha256\n";
        for (std::size_t index = 0U; index < generation.frames.size(); ++index) {
            std::ostringstream name;
            name << "frame_" << std::setw(6) << std::setfill('0') << index << ".ppm";
            const std::filesystem::path relative = std::filesystem::path("frames") / name.str();
            const std::filesystem::path absolute = temporary / relative;
            rlf::solstice::VideoPrototypeFabric::save_ppm(
                absolute, generation.frames[index]
            );
            frame_manifest << index << '\t' << relative.generic_string() << '\t'
                << rlf::core::sha256_hex(rlf::core::sha256_file(absolute)) << '\n';
        }
        frame_manifest.flush();
        if (!frame_manifest) throw std::runtime_error("failed while writing frame manifest");

        std::ofstream manifest(
            temporary / "generation_manifest.json", std::ios::binary | std::ios::trunc
        );
        if (!manifest) throw std::runtime_error("unable to create generation manifest");
        manifest << std::setprecision(17)
            << "{\n"
            << "  \"schema\": \"rlf-solstice-video-prototype-generation-v1\",\n"
            << "  \"claim_boundary\": \"deterministic learned motion-prototype rendering; not photorealistic video synthesis\",\n"
            << "  \"checkpoint_sha256\": \"" << checkpoint_sha << "\",\n"
            << "  \"profile\": " << json_escape(std::string(
                rlf::solstice::to_string(
                    rlf::solstice::parse_profile(options.profile)
                )
              )) << ",\n"
            << "  \"prompt\": " << json_escape(options.prompt) << ",\n"
            << "  \"prototype_id\": " << generation.prototype_id << ",\n"
            << "  \"source_sequence_id\": "
            << json_escape(generation.source_sequence_id) << ",\n"
            << "  \"prompt_similarity\": " << generation.prompt_similarity << ",\n"
            << "  \"confidence\": " << generation.confidence << ",\n"
            << "  \"frames_per_second\": " << generation.frames_per_second << ",\n"
            << "  \"frame_count\": " << generation.frames.size() << ",\n"
            << "  \"width\": " << model.video().config().output_width << ",\n"
            << "  \"height\": " << model.video().config().output_height << "\n"
            << "}\n";
        manifest.flush();
        if (!manifest) throw std::runtime_error("failed while writing generation manifest");
        std::filesystem::rename(temporary, options.output);
    } catch (...) {
        std::error_code cleanup_error;
        std::filesystem::remove_all(temporary, cleanup_error);
        throw;
    }
    std::cout << "prototype_video_output=" << options.output.string() << '\n'
              << "prototype_id=" << generation.prototype_id << '\n'
              << "frame_count=" << generation.frames.size() << '\n'
              << "prompt_similarity=" << generation.prompt_similarity << '\n'
              << "claim_boundary=prototype rendering, not photorealistic synthesis\n";
    return 0;
}

[[nodiscard]] double video_motion_error(
    const rlf::solstice::VideoMotionDescriptor& predicted,
    const rlf::solstice::VideoMotionDescriptor& observed,
    const std::size_t frame_count
) noexcept {
    const double transitions = static_cast<double>(frame_count > 0U ? frame_count - 1U : 0U);
    const double predicted_end_x = predicted.start_x + predicted.velocity_x * transitions;
    const double predicted_end_y = predicted.start_y + predicted.velocity_y * transitions;
    const double observed_end_x = observed.start_x + observed.velocity_x * transitions;
    const double observed_end_y = observed.start_y + observed.velocity_y * transitions;
    const double endpoint = std::hypot(
        predicted_end_x - observed_end_x, predicted_end_y - observed_end_y
    );
    const double velocity = std::hypot(
        predicted.velocity_x - observed.velocity_x,
        predicted.velocity_y - observed.velocity_y
    );
    const double extent = 0.5 * (
        std::abs(predicted.object_width - observed.object_width) +
        std::abs(predicted.object_height - observed.object_height)
    );
    return (endpoint + velocity + extent) / 3.0;
}

[[nodiscard]] double video_pixel_mae(
    const std::span<const rlf::solstice::ImageData> predicted,
    const std::span<const rlf::solstice::ImageData> observed
) {
    if (predicted.size() != observed.size() || predicted.empty()) {
        throw std::invalid_argument("video pixel metric frame counts do not match");
    }
    double absolute_error = 0.0;
    std::uint64_t samples = 0U;
    for (std::size_t frame_index = 0U; frame_index < predicted.size(); ++frame_index) {
        const auto& prediction = predicted[frame_index];
        const auto& truth = observed[frame_index];
        if (prediction.width == 0U || prediction.height == 0U || truth.width == 0U ||
            truth.height == 0U || prediction.rgb.size() != prediction.width * prediction.height * 3U ||
            truth.rgb.size() != truth.width * truth.height * 3U) {
            throw std::invalid_argument("invalid frame in video pixel metric");
        }
        for (std::size_t y = 0U; y < prediction.height; ++y) {
            const std::size_t truth_y = std::min(
                truth.height - 1U, y * truth.height / prediction.height
            );
            for (std::size_t x = 0U; x < prediction.width; ++x) {
                const std::size_t truth_x = std::min(
                    truth.width - 1U, x * truth.width / prediction.width
                );
                const std::size_t prediction_offset = (y * prediction.width + x) * 3U;
                const std::size_t truth_offset = (truth_y * truth.width + truth_x) * 3U;
                for (std::size_t channel = 0U; channel < 3U; ++channel) {
                    absolute_error += std::abs(
                        static_cast<double>(prediction.rgb[prediction_offset + channel]) -
                        static_cast<double>(truth.rgb[truth_offset + channel])
                    ) / 255.0;
                    ++samples;
                }
            }
        }
    }
    return absolute_error / static_cast<double>(samples);
}

int run_evaluate_video(const Options& options) {
    if (options.ledger.empty() || options.output.empty()) {
        throw std::invalid_argument(
            "evaluate-video requires --ledger and a new --output directory"
        );
    }
    const auto profile = rlf::solstice::parse_profile(options.profile);
    if (!rlf::solstice::profile_allows_video(profile)) {
        throw std::invalid_argument(
            "evaluate-video requires --profile video-rtx-pro-6000-96g"
        );
    }
    if (!std::filesystem::is_regular_file(options.checkpoint) ||
        std::filesystem::exists(options.output)) {
        throw std::invalid_argument(
            "evaluate-video needs an existing checkpoint and a new output path"
        );
    }
    const rlf::solstice::DataLedger ledger =
        rlf::solstice::load_data_ledger(options.ledger);
    rlf::solstice::DataAuditOptions audit_config = audit_options(options);
    audit_config.require_media_sha256 = true;
    const rlf::solstice::DataAuditReport audit =
        rlf::solstice::audit_data_ledger(ledger, audit_config);
    if (!audit.valid) {
        throw std::runtime_error("video evaluation ledger failed contamination audit");
    }
    const rlf::solstice::SolsticeModel model =
        rlf::solstice::load_solstice_checkpoint(
            options.checkpoint,
            checkpoint_limits(options)
        );
    if (!rlf::solstice::profile_config_matches(profile, model.config())) {
        throw std::runtime_error("checkpoint is not the enforced RTX video profile");
    }
    const std::string checkpoint_sha = rlf::core::sha256_hex(
        rlf::core::sha256_file(options.checkpoint)
    );
    std::filesystem::path temporary = options.output;
    temporary += ".tmp." + checkpoint_sha.substr(0U, 12U);
    if (std::filesystem::exists(temporary)) {
        throw std::runtime_error("stale video evaluation temporary directory exists");
    }
    std::size_t evaluated = 0U;
    std::size_t retrieval_failures = 0U;
    double motion_error_sum = 0.0;
    double pixel_mae_sum = 0.0;
    double held_frame_motion_error_sum = 0.0;
    double held_frame_pixel_mae_sum = 0.0;
    try {
        std::filesystem::create_directories(temporary / "sequences");
        {
            std::ofstream audit_output(
                temporary / "data_audit.json", std::ios::binary | std::ios::trunc
            );
            if (!audit_output) throw std::runtime_error("unable to create video audit artifact");
            rlf::solstice::write_data_audit_json(audit_output, audit);
        }
        std::ofstream predictions(
            temporary / "raw_predictions.tsv", std::ios::binary | std::ios::trunc
        );
        if (!predictions) throw std::runtime_error("unable to create video prediction artifact");
        predictions << "shard_id\tsequence_id\tprompt_sha256\tprototype_id\tprompt_similarity"
                       "\tmotion_error\tpixel_mae\theld_frame_motion_error"
                       "\theld_frame_pixel_mae\tstatus\n";
        const std::filesystem::path ledger_base = ledger.source_path.parent_path();
        for (const rlf::solstice::DataShard& shard : ledger.shards) {
            if (shard.kind != rlf::solstice::DataShardKind::video ||
                shard.split == rlf::solstice::DataShardSplit::train) {
                continue;
            }
            const std::filesystem::path shard_path = shard.path.is_absolute()
                ? shard.path : ledger_base / shard.path;
            auto [sequences, rows] = load_video_sequences(shard_path);
            static_cast<void>(rows);
            for (const VideoSequenceRows& sequence : sequences) {
                if (model.video().contains_source_sequence(sequence.id)) {
                    throw std::runtime_error(
                        "held-out video sequence ID occurs in the learned checkpoint: " + sequence.id
                    );
                }
                std::vector<rlf::solstice::ImageData> observed_frames;
                observed_frames.reserve(sequence.frames.size());
                for (const VideoFrameRow& frame : sequence.frames) {
                    observed_frames.push_back(rlf::solstice::load_image(frame.path));
                }
                const std::string prompt_sha = rlf::core::sha256_hex(
                    rlf::core::sha256(sequence.prompt)
                );
                const rlf::solstice::VideoMotionDescriptor observed =
                    rlf::solstice::VideoPrototypeFabric::describe(observed_frames);
                const std::vector<rlf::solstice::ImageData> held_frames(
                    observed_frames.size(), observed_frames.front()
                );
                const rlf::solstice::VideoMotionDescriptor held_motion =
                    rlf::solstice::VideoPrototypeFabric::describe(held_frames);
                const double held_motion_error = video_motion_error(
                    held_motion, observed, observed_frames.size()
                );
                const double held_pixel_mae = video_pixel_mae(
                    held_frames, observed_frames
                );
                held_frame_motion_error_sum += held_motion_error;
                held_frame_pixel_mae_sum += held_pixel_mae;
                std::optional<rlf::solstice::VideoGeneration> generation;
                try {
                    generation = model.generate_video(sequence.prompt, observed_frames.size());
                } catch (const std::runtime_error&) {
                    predictions << shard.shard_id << '\t' << sequence.id << '\t' << prompt_sha
                        << "\t0\t0\t1\t1\t" << std::setprecision(17)
                        << held_motion_error << '\t' << held_pixel_mae
                        << "\tretrieval_failed\n";
                    ++retrieval_failures;
                    motion_error_sum += 1.0;
                    pixel_mae_sum += 1.0;
                    ++evaluated;
                    continue;
                }
                const double motion_error = video_motion_error(
                    generation->motion, observed, observed_frames.size()
                );
                const double pixel_mae = video_pixel_mae(
                    generation->frames, observed_frames
                );
                const std::filesystem::path sequence_directory =
                    temporary / "sequences" / std::to_string(evaluated);
                for (std::size_t index = 0U; index < generation->frames.size(); ++index) {
                    rlf::solstice::VideoPrototypeFabric::save_ppm(
                        sequence_directory / ("frame_" + std::to_string(index) + ".ppm"),
                        generation->frames[index]
                    );
                }
                predictions << shard.shard_id << '\t' << sequence.id << '\t' << prompt_sha
                    << '\t' << generation->prototype_id << '\t'
                    << std::setprecision(17) << generation->prompt_similarity << '\t'
                    << motion_error << '\t' << pixel_mae << '\t'
                    << held_motion_error << '\t' << held_pixel_mae << "\tok\n";
                motion_error_sum += motion_error;
                pixel_mae_sum += pixel_mae;
                ++evaluated;
            }
        }
        if (evaluated == 0U) {
            throw std::runtime_error("video evaluation ledger contains no held-out video sequences");
        }
        predictions.flush();
        if (!predictions) throw std::runtime_error("failed while writing video predictions");
        std::ofstream summary(
            temporary / "summary.json", std::ios::binary | std::ios::trunc
        );
        if (!summary) throw std::runtime_error("unable to create video evaluation summary");
        summary << std::setprecision(17)
            << "{\n"
            << "  \"schema\": \"rlf-solstice-video-prototype-evaluation-v1\",\n"
            << "  \"claim_boundary\": \"prototype motion rendering; no photorealistic or frontier claim\",\n"
            << "  \"checkpoint_sha256\": \"" << checkpoint_sha << "\",\n"
            << "  \"ledger_sha256\": \"" << ledger.sha256 << "\",\n"
            << "  \"evaluated_sequences\": " << evaluated << ",\n"
            << "  \"retrieval_failures\": " << retrieval_failures << ",\n"
            << "  \"mean_motion_error\": "
            << motion_error_sum / static_cast<double>(evaluated) << ",\n"
            << "  \"mean_pixel_mae\": "
            << pixel_mae_sum / static_cast<double>(evaluated) << ",\n"
            << "  \"mean_held_frame_motion_error\": "
            << held_frame_motion_error_sum / static_cast<double>(evaluated) << ",\n"
            << "  \"mean_held_frame_pixel_mae\": "
            << held_frame_pixel_mae_sum / static_cast<double>(evaluated) << "\n"
            << "}\n";
        summary.flush();
        if (!summary) throw std::runtime_error("failed while writing video evaluation summary");
        std::filesystem::rename(temporary, options.output);
    } catch (...) {
        std::error_code cleanup_error;
        std::filesystem::remove_all(temporary, cleanup_error);
        throw;
    }
    std::cout << "video_evaluation_output=" << options.output.string() << '\n'
              << "evaluated_sequences=" << evaluated << '\n'
              << "retrieval_failures=" << retrieval_failures << '\n'
              << "mean_motion_error=" << motion_error_sum / static_cast<double>(evaluated) << '\n'
              << "mean_pixel_mae=" << pixel_mae_sum / static_cast<double>(evaluated) << '\n';
    std::cout << "mean_held_frame_motion_error="
              << held_frame_motion_error_sum / static_cast<double>(evaluated) << '\n'
              << "mean_held_frame_pixel_mae="
              << held_frame_pixel_mae_sum / static_cast<double>(evaluated) << '\n';
    return 0;
}

int run_evaluate_batch(const Options& options) {
    if (options.manifest.empty() || options.output.empty()) {
        throw std::invalid_argument(
            "evaluate-batch requires --manifest and --output"
        );
    }
    if (!std::filesystem::is_regular_file(options.checkpoint)) {
        throw std::invalid_argument(
            "evaluate-batch requires an existing checkpoint"
        );
    }
    if (options.enable_evaluation_tools && options.disable_tools) {
        throw std::invalid_argument(
            "--enable-tools and --no-tools cannot be combined"
        );
    }
    rlf::solstice::SolsticeModel model =
        rlf::solstice::load_solstice_checkpoint(
            options.checkpoint,
            checkpoint_limits(options)
        );
    enforce_loaded_profile(options, model);
    model.set_backend(parse_backend(options.backend));
    rlf::solstice::ToolRuntime tools = make_tools(options);
    rlf::solstice::ToolRuntime* const tool_pointer =
        options.enable_evaluation_tools ? &tools : nullptr;
    rlf::solstice::EvaluationBatchOptions batch_options;
    batch_options.manifest_path = options.manifest;
    batch_options.output_directory = options.output;
    batch_options.checkpoint_sha256 = rlf::core::sha256_hex(
        rlf::core::sha256_file(options.checkpoint)
    );
    batch_options.backend_name = options.backend;
    batch_options.generation = generation_settings(options);
    batch_options.maximum_prompt_bytes = options.maximum_prompt_bytes;
    batch_options.allow_images = rlf::solstice::profile_allows_vision(
        rlf::solstice::parse_profile(options.profile)
    );
    const rlf::solstice::EvaluationBatchReport report =
        rlf::solstice::run_evaluation_batch(
            model, batch_options, tool_pointer
        );
    std::cout << rlf::solstice::evaluation_batch_report_json(report);
    return 0;
}


int run_induce_rule(const Options& options) {
    if (options.subject.empty() || options.relation.empty() ||
        options.object.empty() || options.object.front() == '?') {
        throw std::invalid_argument(
            "induce-rule requires --subject, --relation, and a concrete --object"
        );
    }
    rlf::solstice::SolsticeModel model = load_or_create(options);
    const std::string name = options.prompt.empty()
        ? "induced_" + options.relation
        : options.prompt;
    const auto result = model.induce_chain_rule(
        name,
        options.subject,
        options.relation,
        options.object,
        options.maximum_hops,
        1.0
    );
    rlf::solstice::save_solstice_checkpoint(options.checkpoint, model);
    std::cout << "induced_rule_id=" << result.rule_id
              << " path_hops=" << result.path_hops
              << " edges_examined=" << result.edges_examined << '\n';
    std::cout << "relation_path=";
    for (std::size_t index = 0U; index < result.relation_path.size(); ++index) {
        if (index != 0U) {
            std::cout << ',';
        }
        std::cout << result.relation_path[index];
    }
    std::cout << '\n';
    return 0;
}

int run_reason(const Options& options) {
    if (options.subject.empty() || options.relation.empty()) {
        throw std::invalid_argument(
            "reason requires --subject and --relation"
        );
    }
    const rlf::solstice::SolsticeModel model =
        rlf::solstice::load_solstice_checkpoint(
            options.checkpoint,
            checkpoint_limits(options)
        );
    const auto answers = model.reason({
        options.subject, options.relation, options.object
    });
    if (answers.empty()) {
        std::cout << "no_answer\n";
        return 1;
    }
    for (const auto& answer : answers) {
        std::cout << "answer=" << answer.value
                  << " confidence=" << std::fixed << std::setprecision(6)
                  << answer.confidence << '\n';
        for (const auto& step : answer.proof) {
            std::cout << "  proof fact=" << step.fact_id
                      << " rule=" << step.rule_id
                      << " " << step.statement << '\n';
        }
    }
    return 0;
}

[[nodiscard]] rlf::solstice::ImageGenerationArchitecture
parse_image_generation_architecture(const std::string_view value) {
    if (value == "resonant-fabric" || value == "resonant") {
        return rlf::solstice::ImageGenerationArchitecture::resonant_fabric;
    }
    if (value == "patch-quilt-baseline" || value == "patch-quilt") {
        return rlf::solstice::ImageGenerationArchitecture::patch_quilt_baseline;
    }
    throw std::invalid_argument("unknown image-generation architecture");
}

void enforce_image_checkpoint_profile(
    const Options& options,
    const rlf::solstice::ImageGenerationCheckpointState& state
) {
    if (!options.profile_set) {
        return;
    }
    const auto requested =
        rlf::solstice::parse_image_generation_profile(options.profile);
    if (requested != state.profile) {
        throw std::runtime_error(
            "image-generation checkpoint does not match --profile"
        );
    }
}

int run_imagegen_profile_info(const Options& options) {
    if (!options.profile_set) {
        throw std::invalid_argument(
            "imagegen-profile-info requires an explicit --profile"
        );
    }
    const auto profile =
        rlf::solstice::parse_image_generation_profile(options.profile);
    const auto capacity =
        rlf::solstice::estimate_image_generation_capacity(profile);
    const auto resonant =
        rlf::solstice::make_resonant_image_profile_config(profile);
    constexpr std::uint64_t gib = 1024ULL * 1024ULL * 1024ULL;
    std::cout
        << "profile=" << rlf::solstice::to_string(profile) << '\n'
        << "architecture=resonant-fabric\n"
        << "gpu_working_set_gib=" << capacity.gpu_working_set_bytes / gib << '\n'
        << "peak_vram_limit_gib=" << capacity.peak_vram_limit_bytes / gib << '\n'
        << "host_ram_recommended_gib="
        << capacity.host_ram_recommended_bytes / gib << '\n'
        << "checkpoint_ceiling_gib="
        << capacity.checkpoint_ceiling_bytes / gib << '\n'
        << "maximum_modes=" << resonant.maximum_modes << '\n'
        << "patch_size=" << resonant.patch_size << '\n'
        << "candidate_count=" << resonant.candidate_count << '\n'
        << "active_count=" << resonant.active_count << '\n'
        << "maximum_prompt_concepts=" << resonant.maximum_prompt_concepts << '\n'
        << "maximum_semantic_candidates="
        << resonant.maximum_semantic_candidates << '\n'
        << "minimum_semantic_similarity="
        << resonant.minimum_semantic_similarity << '\n'
        << "claim_boundary=controlled non-neural image transformation learning; "
           "not unrestricted text-to-image or frontier evidence\n";
    return 0;
}

int run_imagegen_bootstrap(const Options& options) {
    if (!options.profile_set) {
        throw std::invalid_argument(
            "imagegen-bootstrap requires an explicit --profile"
        );
    }
    if (std::filesystem::exists(options.checkpoint)) {
        throw std::invalid_argument(
            "imagegen-bootstrap refuses to overwrite an existing checkpoint"
        );
    }
    rlf::solstice::ImageGenerationCheckpointState state;
    state.profile =
        rlf::solstice::parse_image_generation_profile(options.profile);
    state.architecture =
        parse_image_generation_architecture(options.architecture);
    state.master_seed = options.seed;
    if (state.architecture ==
        rlf::solstice::ImageGenerationArchitecture::resonant_fabric) {
        auto config =
            rlf::solstice::make_resonant_image_profile_config(state.profile);
        config.seed = options.seed;
        state.resonant_fabric =
            rlf::solstice::ResonantImageFabric(std::move(config));
    } else {
        state.fabric = rlf::solstice::PatchQuiltBaseline(
            rlf::solstice::make_image_generation_profile_config(state.profile)
        );
    }
    rlf::solstice::save_image_generation_checkpoint(options.checkpoint, state);
    std::cout << "image_generation_checkpoint="
              << options.checkpoint.string() << '\n'
              << "profile=" << rlf::solstice::to_string(state.profile) << '\n'
              << "architecture=" << rlf::solstice::to_string(state.architecture)
              << '\n';
    return 0;
}

constexpr std::string_view neutral_image_source_marker =
    "@neutral-gray128-target-size-v1";

[[nodiscard]] const std::string& neutral_image_source_hash() {
    static const std::string hash = rlf::core::sha256_hex(
        rlf::core::sha256(neutral_image_source_marker)
    );
    return hash;
}

[[nodiscard]] rlf::solstice::ImageData neutral_image_like(
    const rlf::solstice::ImageData& target
) {
    return {
        .width = target.width,
        .height = target.height,
        .rgb = std::vector<std::uint8_t>(target.rgb.size(), 128U),
    };
}

[[nodiscard]] rlf::solstice::ImageData make_neutral_image(
    const std::size_t width,
    const std::size_t height,
    const rlf::solstice::ResonantImageConfig& config
) {
    if (width == 0U || height == 0U ||
        width > config.maximum_image_side ||
        height > config.maximum_image_side ||
        width > config.maximum_image_pixels / height ||
        width % config.patch_size != 0U || height % config.patch_size != 0U) {
        throw std::invalid_argument(
            "prompt-only image dimensions exceed the profile or patch alignment"
        );
    }
    const std::size_t pixels = width * height;
    if (pixels > std::numeric_limits<std::size_t>::max() / 3U) {
        throw std::invalid_argument("prompt-only image byte count overflow");
    }
    return {
        .width = width,
        .height = height,
        .rgb = std::vector<std::uint8_t>(pixels * 3U, 128U),
    };
}

constexpr std::string_view image_prompt_language_shard_prefix =
    "prompt-language:";

[[nodiscard]] bool is_prompt_language_shard(
    const rlf::solstice::ImageGenerationShardRecord& shard
) noexcept {
    return std::string_view(shard.shard_id).starts_with(
        image_prompt_language_shard_prefix
    );
}

[[nodiscard]] std::size_t train_prompt_language_shard(
    rlf::solstice::ResonantImageFabric& fabric,
    const std::filesystem::path& path,
    const std::size_t maximum_record_bytes
) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "unable to open prompt-language shard: " + path.string()
        );
    }
    std::string line;
    std::size_t line_number = 0U;
    std::size_t records = 0U;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.size() > maximum_record_bytes) {
            throw std::runtime_error(
                "prompt-language record exceeds --max-prompt-bytes at " +
                path.string() + ":" + std::to_string(line_number)
            );
        }
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line.front() == '#') {
            continue;
        }
        fabric.train_prompt_language_record(line);
        if (records == std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error("prompt-language record count exhausted");
        }
        ++records;
    }
    if (!input.eof()) {
        throw std::runtime_error(
            "failed while reading prompt-language shard: " + path.string()
        );
    }
    return records;
}

int run_imagegen_train_language_ledger(const Options& options) {
    using rlf::solstice::DataRecordFormat;
    using rlf::solstice::DataShardKind;
    using rlf::solstice::DataShardSplit;
    if (options.ledger.empty()) {
        throw std::invalid_argument(
            "imagegen-train-language-ledger requires --ledger"
        );
    }
    if (options.target_training_records == 0U) {
        throw std::invalid_argument(
            "imagegen-train-language-ledger requires an exact "
            "--target-training-records"
        );
    }
    const auto ledger = rlf::solstice::load_data_ledger(options.ledger);
    const auto audit = rlf::solstice::audit_prompt_language_ledger_scalable(
        ledger, audit_options(options)
    );
    emit_audit_report(options, audit);
    if (!audit.valid) {
        throw std::runtime_error(
            "prompt-language ledger audit failed; training was not started"
        );
    }
    if (audit.audited_shards.size() != ledger.shards.size()) {
        throw std::runtime_error(
            "prompt-language audit did not account for every ledger shard"
        );
    }

    std::uint64_t ledger_training_records = 0U;
    for (std::size_t index = 0U; index < ledger.shards.size(); ++index) {
        const auto& shard = ledger.shards[index];
        if (shard.split != DataShardSplit::train) {
            continue;
        }
        if (shard.kind != DataShardKind::text ||
            shard.format != DataRecordFormat::text_lines) {
            throw std::runtime_error(
                "image prompt-language training accepts only train/text/"
                "text_lines shards: " + shard.shard_id
            );
        }
        const std::uint64_t records = static_cast<std::uint64_t>(
            audit.audited_shards[index].records
        );
        if (records > std::numeric_limits<std::uint64_t>::max() -
                ledger_training_records) {
            throw std::overflow_error(
                "prompt-language ledger record count overflow"
            );
        }
        ledger_training_records += records;
    }
    if (ledger_training_records != options.target_training_records) {
        throw std::runtime_error(
            "prompt-language ledger contains " +
            std::to_string(ledger_training_records) +
            " train records; exact target is " +
            std::to_string(options.target_training_records)
        );
    }

    auto state = rlf::solstice::load_image_generation_checkpoint(
        options.checkpoint
    );
    enforce_image_checkpoint_profile(options, state);
    if (state.architecture !=
        rlf::solstice::ImageGenerationArchitecture::resonant_fabric) {
        throw std::runtime_error(
            "imagegen-train-language-ledger requires resonant-fabric"
        );
    }
    const std::uint64_t learned_before =
        state.resonant_fabric.prompt_semantics().stats().records_seen;
    std::uint64_t completed_records = 0U;
    for (const auto& completed : state.completed_shards) {
        if (!is_prompt_language_shard(completed)) {
            continue;
        }
        const std::string_view original_id(completed.shard_id);
        const std::string ledger_id(original_id.substr(
            image_prompt_language_shard_prefix.size()
        ));
        bool matched = false;
        for (std::size_t index = 0U; index < ledger.shards.size(); ++index) {
            const auto& shard = ledger.shards[index];
            if (shard.split != DataShardSplit::train ||
                shard.shard_id != ledger_id) {
                continue;
            }
            const auto& audited = audit.audited_shards[index];
            if (completed.shard_sha256 != audited.actual_sha256 ||
                completed.ledger_sha256 != ledger.sha256 ||
                completed.records != audited.records) {
                throw std::runtime_error(
                    "completed prompt-language shard changed: " + ledger_id
                );
            }
            matched = true;
            break;
        }
        if (!matched) {
            throw std::runtime_error(
                "exact-target ledger omits completed prompt-language shard: " +
                ledger_id
            );
        }
        if (completed.records >
            std::numeric_limits<std::uint64_t>::max() - completed_records) {
            throw std::overflow_error(
                "completed prompt-language record count overflow"
            );
        }
        completed_records += completed.records;
    }
    if (completed_records != learned_before) {
        throw std::runtime_error(
            "prompt-language checkpoint shard accounting is inconsistent"
        );
    }

    const auto base = ledger.source_path.parent_path();
    std::size_t trained_shards = 0U;
    std::size_t resumed_shards = 0U;
    bool intentional_stop = false;
    for (std::size_t index = 0U; index < ledger.shards.size(); ++index) {
        const auto& shard = ledger.shards[index];
        if (shard.split != DataShardSplit::train) {
            continue;
        }
        const auto& audited = audit.audited_shards[index];
        const std::string checkpoint_shard_id =
            std::string(image_prompt_language_shard_prefix) + shard.shard_id;
        bool already_completed = false;
        for (const auto& completed : state.completed_shards) {
            if (completed.shard_id == checkpoint_shard_id) {
                already_completed = true;
                break;
            }
        }
        if (already_completed) {
            ++resumed_shards;
            continue;
        }
        const auto path = shard.path.is_absolute()
            ? shard.path
            : base / shard.path;
        const std::size_t records = train_prompt_language_shard(
            state.resonant_fabric,
            path,
            options.maximum_prompt_bytes
        );
        if (records != audited.records) {
            throw std::runtime_error(
                "prompt-language training/audit record mismatch for " +
                shard.shard_id
            );
        }
        state.resonant_fabric.rebuild_semantic_index();
        state.completed_shards.push_back({
            .shard_id = checkpoint_shard_id,
            .shard_sha256 = audited.actual_sha256,
            .ledger_sha256 = ledger.sha256,
            .source_uri = shard.source_uri,
            .license = shard.license,
            .records = static_cast<std::uint64_t>(records),
            .bytes = audited.file_bytes,
        });
        rlf::solstice::save_image_generation_checkpoint(
            options.checkpoint, state
        );
        ++trained_shards;
        if (options.maximum_new_shards > 0U &&
            trained_shards >= options.maximum_new_shards) {
            intentional_stop = true;
            break;
        }
    }
    const auto semantic_stats =
        state.resonant_fabric.prompt_semantics().stats();
    const bool target_reached =
        semantic_stats.records_seen == options.target_training_records;
    if (!target_reached && !intentional_stop) {
        throw std::runtime_error(
            "prompt-language checkpoint did not reach the exact target"
        );
    }
    std::cout << "trained_prompt_language_shards=" << trained_shards << '\n'
              << "resumed_prompt_language_shards=" << resumed_shards << '\n'
              << "prompt_language_records=" << semantic_stats.records_seen
              << '\n'
              << "prompt_language_words=" << semantic_stats.words_seen << '\n'
              << "prompt_semantic_modes="
              << state.resonant_fabric.prompt_semantics().modes().size() << '\n'
              << "target_training_records="
              << options.target_training_records << '\n'
              << "training_target_reached="
              << (target_reached ? "true" : "false") << '\n'
              << "checkpoint=" << options.checkpoint.string() << '\n';
    return target_reached ? 0 : 4;
}

int run_imagegen_train_pair(const Options& options) {
    if (!options.target_image.has_value() || options.prompt.empty()) {
        throw std::invalid_argument(
            "imagegen-train-pair requires --target-image and --prompt; --image is optional"
        );
    }
    auto state = rlf::solstice::load_image_generation_checkpoint(
        options.checkpoint
    );
    enforce_image_checkpoint_profile(options, state);
    if (state.architecture !=
        rlf::solstice::ImageGenerationArchitecture::resonant_fabric) {
        throw std::runtime_error(
            "imagegen-train-pair requires the resonant-fabric architecture"
        );
    }
    state.resonant_fabric.set_backend(parse_backend(options.backend));
    auto target = rlf::solstice::load_image(*options.target_image);
    auto source = options.image.has_value()
        ? rlf::solstice::load_image(*options.image)
        : neutral_image_like(target);
    const auto result = state.resonant_fabric.train({
        .source = std::move(source),
        .target = std::move(target),
        .semantic_label = options.prompt,
    });
    if (state.training_step == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("image-generation training step exhausted");
    }
    ++state.training_step;
    rlf::solstice::save_image_generation_checkpoint(options.checkpoint, state);
    std::cout << "training_step=" << state.training_step << '\n'
              << "patches=" << result.patches << '\n'
              << "modes_created=" << result.modes_created << '\n'
              << "modes_updated=" << result.modes_updated << '\n'
              << "source_kind="
              << (options.image.has_value() ? "provided-image" : "neutral-prompt")
              << '\n'
              << "backend=" << rlf::frontier::to_string(
                    state.resonant_fabric.backend_kind()
                 ) << '\n'
              << "backend_local_update_calls="
              << state.resonant_fabric.backend_operation_stats().local_update_calls
              << '\n'
              << "backend_device_local_update_calls="
              << state.resonant_fabric.backend_operation_stats()
                    .device_local_update_calls << '\n'
              << "deterministic_model_hash=0x" << std::hex
              << state.resonant_fabric.deterministic_hash() << std::dec << '\n';
    return 0;
}

[[nodiscard]] std::filesystem::path resolve_imagegen_manifest_path(
    const std::filesystem::path& root,
    const std::string_view relative_value
) {
    const std::filesystem::path relative(relative_value);
    if (relative.empty() || relative.is_absolute()) {
        throw std::invalid_argument(
            "image-generation manifest paths must be non-empty and relative"
        );
    }
    for (const auto& component : relative) {
        if (component == "..") {
            throw std::invalid_argument(
                "image-generation manifest path traversal is forbidden"
            );
        }
    }
    const std::filesystem::path resolved =
        std::filesystem::weakly_canonical(root / relative);
    const std::filesystem::path relative_to_root = resolved.lexically_relative(root);
    if (relative_to_root.empty() || *relative_to_root.begin() == "..") {
        throw std::invalid_argument(
            "image-generation manifest path escapes its shard directory"
        );
    }
    std::error_code status_error;
    const auto status = std::filesystem::symlink_status(resolved, status_error);
    if (status_error || std::filesystem::is_symlink(status) ||
        !std::filesystem::is_regular_file(status)) {
        throw std::invalid_argument(
            "image-generation source must be a regular non-symlink file"
        );
    }
    return resolved;
}

int run_imagegen_train_manifest(const Options& options) {
    if (options.manifest.empty()) {
        throw std::invalid_argument(
            "imagegen-train-manifest requires --manifest"
        );
    }
    std::error_code manifest_status_error;
    const auto manifest_status = std::filesystem::symlink_status(
        options.manifest,
        manifest_status_error
    );
    if (manifest_status_error || std::filesystem::is_symlink(manifest_status) ||
        !std::filesystem::is_regular_file(manifest_status)) {
        throw std::invalid_argument(
            "image-generation manifest must be a regular non-symlink file"
        );
    }
    if (std::filesystem::file_size(options.manifest) >
        options.maximum_train_shard_bytes) {
        throw std::runtime_error(
            "image-generation manifest exceeds --max-train-shard-bytes"
        );
    }
    auto state = rlf::solstice::load_image_generation_checkpoint(
        options.checkpoint
    );
    enforce_image_checkpoint_profile(options, state);
    if (state.architecture !=
        rlf::solstice::ImageGenerationArchitecture::resonant_fabric) {
        throw std::runtime_error(
            "imagegen-train-manifest requires resonant-fabric"
        );
    }
    state.resonant_fabric.set_backend(parse_backend(options.backend));

    const std::filesystem::path manifest =
        std::filesystem::weakly_canonical(options.manifest);
    const std::filesystem::path root = manifest.parent_path();
    const std::string manifest_sha = rlf::core::sha256_hex(
        rlf::core::sha256_file(manifest)
    );
    const std::string shard_id = manifest.filename().string();
    for (const auto& completed : state.completed_shards) {
        if (completed.shard_id == shard_id) {
            if (completed.shard_sha256 != manifest_sha) {
                throw std::runtime_error(
                    "completed image-generation shard ID was reused with new bytes"
                );
            }
            std::cout << "shard_already_completed=true\n"
                      << "shard_id=" << shard_id << '\n'
                      << "shard_sha256=" << manifest_sha << '\n';
            return 0;
        }
    }

    std::unordered_set<std::string> record_ids;
    std::unordered_set<std::string> exact_pairs;
    std::uint64_t source_bytes = 0ULL;
    const std::size_t records = for_each_tsv(
        manifest,
        8U,
        8U,
        [&](const std::vector<std::string>& fields) {
            const std::string& record_id = fields[0U];
            const std::string& source_hash = fields[2U];
            const std::string& target_hash = fields[4U];
            const std::string& semantic_label = fields[5U];
            const std::string& source_uri = fields[6U];
            const std::string& license = fields[7U];
            const bool neutral_source =
                fields[1U] == neutral_image_source_marker;
            if (record_id.empty() || !record_ids.insert(record_id).second ||
                !rlf::core::is_sha256_hex(source_hash) ||
                !rlf::core::is_sha256_hex(target_hash) ||
                semantic_label.empty() || source_uri.empty() || license.empty()) {
                throw std::runtime_error(
                    "invalid image-generation manifest identity/provenance row"
                );
            }
            const std::string pair_key = source_hash + ':' + target_hash + ':' +
                semantic_label;
            if (!exact_pairs.insert(pair_key).second) {
                throw std::runtime_error(
                    "duplicate image-generation training pair"
                );
            }
            const auto target_path = resolve_imagegen_manifest_path(
                root,
                fields[3U]
            );
            if ((neutral_source && source_hash != neutral_image_source_hash()) ||
                (!neutral_source &&
                 rlf::core::sha256_hex(rlf::core::sha256_file(
                     resolve_imagegen_manifest_path(root, fields[1U])
                 )) != source_hash) ||
                rlf::core::sha256_hex(rlf::core::sha256_file(target_path)) !=
                    target_hash) {
                throw std::runtime_error(
                    "image-generation manifest media SHA-256 mismatch"
                );
            }
            auto target = rlf::solstice::load_image(target_path);
            std::uint64_t pair_bytes = static_cast<std::uint64_t>(
                    std::filesystem::file_size(target_path)
                );
            rlf::solstice::ImageData source;
            if (neutral_source) {
                source = neutral_image_like(target);
            } else {
                const auto source_path = resolve_imagegen_manifest_path(
                    root, fields[1U]
                );
                pair_bytes += static_cast<std::uint64_t>(
                    std::filesystem::file_size(source_path)
                );
                source = rlf::solstice::load_image(source_path);
            }
            if (pair_bytes > options.maximum_train_shard_bytes - source_bytes) {
                throw std::runtime_error(
                    "image-generation shard media exceeds --max-train-shard-bytes"
                );
            }
            source_bytes += pair_bytes;
            static_cast<void>(state.resonant_fabric.train({
                .source = std::move(source),
                .target = std::move(target),
                .semantic_label = semantic_label,
            }));
        }
    );
    if (records == 0U) {
        throw std::runtime_error("image-generation manifest has no records");
    }
    if (static_cast<std::uint64_t>(records) >
        std::numeric_limits<std::uint64_t>::max() - state.training_step) {
        throw std::overflow_error("image-generation training step exhausted");
    }
    state.training_step += static_cast<std::uint64_t>(records);
    state.completed_shards.push_back({
        .shard_id = shard_id,
        .shard_sha256 = manifest_sha,
        .ledger_sha256 = manifest_sha,
        .source_uri = "manifest:" + manifest.generic_string(),
        .license = "per-row-audited",
        .records = static_cast<std::uint64_t>(records),
        .bytes = source_bytes,
    });
    // One manifest is the atomic recovery unit: any validation/training error
    // occurs before this transactional checkpoint install.
    rlf::solstice::save_image_generation_checkpoint(options.checkpoint, state);
    std::cout << "shard_already_completed=false\n"
              << "shard_id=" << shard_id << '\n'
              << "shard_sha256=" << manifest_sha << '\n'
              << "records=" << records << '\n'
              << "media_bytes=" << source_bytes << '\n'
              << "training_step=" << state.training_step << '\n'
              << "completed_shards=" << state.completed_shards.size() << '\n'
              << "backend=" << rlf::frontier::to_string(
                    state.resonant_fabric.backend_kind()
                 ) << '\n'
              << "backend_local_update_calls="
              << state.resonant_fabric.backend_operation_stats().local_update_calls
              << '\n'
              << "backend_device_local_update_calls="
              << state.resonant_fabric.backend_operation_stats()
                    .device_local_update_calls << '\n';
    return 0;
}

int run_imagegen_audit_pairs(const Options& options) {
    if (options.manifest.empty() || options.evaluation_manifest.empty() ||
        options.license_policy.empty() || options.output.empty()) {
        throw std::invalid_argument(
            "imagegen-audit-pairs requires --manifest, --evaluation-manifest, "
            "--license-policy, and --output"
        );
    }
    const auto report = rlf::solstice::audit_image_generation_data(
        options.manifest,
        options.evaluation_manifest,
        options.license_policy,
        options.output,
        {
            .maximum_records = options.maximum_audit_records,
            .near_duplicate_hamming_distance =
                options.near_duplicate_hamming_distance,
        }
    );
    std::cout << "audit_passed=" << (report.passed() ? "true" : "false")
              << '\n'
              << "records_audited=" << report.records_audited << '\n'
              << "evaluation_records_audited="
              << report.evaluation_records_audited << '\n'
              << "exact_duplicates=" << report.exact_duplicates << '\n'
              << "near_duplicates=" << report.near_duplicates << '\n'
              << "perceptual_duplicates=" << report.perceptual_duplicates << '\n'
              << "overlap_records=" << report.overlap_records << '\n'
              << "pair_manifest_sha256=" << report.pair_manifest_sha256 << '\n'
              << "evaluation_manifest_sha256="
              << report.evaluation_manifest_sha256 << '\n'
              << "frontier_claim_authorized=false\n";
    return report.passed() ? 0 : 3;
}

[[nodiscard]] double global_image_ssim(
    const rlf::solstice::ImageData& candidate,
    const rlf::solstice::ImageData& reference
) {
    if (candidate.width != reference.width ||
        candidate.height != reference.height ||
        candidate.rgb.size() != reference.rgb.size() || candidate.rgb.empty()) {
        throw std::invalid_argument("SSIM images must have matching non-empty shapes");
    }
    const double count = static_cast<double>(candidate.rgb.size());
    double candidate_mean = 0.0;
    double reference_mean = 0.0;
    for (std::size_t index = 0U; index < candidate.rgb.size(); ++index) {
        candidate_mean += static_cast<double>(candidate.rgb[index]);
        reference_mean += static_cast<double>(reference.rgb[index]);
    }
    candidate_mean /= count;
    reference_mean /= count;
    double candidate_variance = 0.0;
    double reference_variance = 0.0;
    double covariance = 0.0;
    for (std::size_t index = 0U; index < candidate.rgb.size(); ++index) {
        const double left = static_cast<double>(candidate.rgb[index]) - candidate_mean;
        const double right = static_cast<double>(reference.rgb[index]) - reference_mean;
        candidate_variance += left * left;
        reference_variance += right * right;
        covariance += left * right;
    }
    candidate_variance /= count;
    reference_variance /= count;
    covariance /= count;
    constexpr double c1 = 6.5025;
    constexpr double c2 = 58.5225;
    return ((2.0 * candidate_mean * reference_mean + c1) *
            (2.0 * covariance + c2)) /
        ((candidate_mean * candidate_mean + reference_mean * reference_mean + c1) *
         (candidate_variance + reference_variance + c2));
}

class OutputDirectoryTransaction final {
public:
    explicit OutputDirectoryTransaction(std::filesystem::path target)
        : target_(std::move(target)),
          temporary_(target_.string() + ".partial") {
        if (std::filesystem::exists(target_) ||
            std::filesystem::exists(temporary_)) {
            throw std::invalid_argument(
                "image evaluation target or partial output already exists"
            );
        }
        std::filesystem::create_directories(temporary_);
    }

    OutputDirectoryTransaction(const OutputDirectoryTransaction&) = delete;
    OutputDirectoryTransaction& operator=(const OutputDirectoryTransaction&) = delete;

    ~OutputDirectoryTransaction() {
        if (!committed_) {
            std::error_code cleanup_error;
            std::filesystem::remove_all(temporary_, cleanup_error);
        }
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return temporary_;
    }

    void commit() {
        std::filesystem::rename(temporary_, target_);
        committed_ = true;
    }

private:
    std::filesystem::path target_;
    std::filesystem::path temporary_;
    bool committed_{};
};

[[nodiscard]] std::vector<std::string> image_prompt_tokens(
    const std::string_view prompt
) {
    std::vector<std::string> tokens;
    std::string token;
    for (const char character : prompt) {
        const auto byte = static_cast<unsigned char>(character);
        if (std::isalnum(byte) != 0 || byte >= 0x80U) {
            token.push_back(static_cast<char>(
                byte < 0x80U ? std::tolower(byte) : byte
            ));
        } else if (!token.empty()) {
            tokens.push_back(std::move(token));
            token.clear();
        }
    }
    if (!token.empty()) {
        tokens.push_back(std::move(token));
    }
    std::sort(tokens.begin(), tokens.end());
    tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
    return tokens;
}

[[nodiscard]] double image_prompt_jaccard(
    const std::vector<std::string>& left,
    const std::vector<std::string>& right
) {
    std::size_t left_index = 0U;
    std::size_t right_index = 0U;
    std::size_t intersection = 0U;
    while (left_index < left.size() && right_index < right.size()) {
        if (left[left_index] == right[right_index]) {
            ++intersection;
            ++left_index;
            ++right_index;
        } else if (left[left_index] < right[right_index]) {
            ++left_index;
        } else {
            ++right_index;
        }
    }
    const std::size_t union_size = left.size() + right.size() - intersection;
    return union_size == 0U ? 0.0 :
        static_cast<double>(intersection) / static_cast<double>(union_size);
}

[[nodiscard]] std::unordered_set<std::string> image_evaluation_tags(
    const std::vector<std::string>& fields
) {
    if (fields.size() == 8U) {
        return {};
    }
    static const std::unordered_set<std::string> allowed{
        "unseen_prompt", "paraphrase", "composition", "natural_image",
        "multilingual", "spatial", "attribute_binding",
    };
    std::unordered_set<std::string> tags;
    std::size_t begin = 0U;
    const std::string_view value(fields.at(8U));
    while (begin <= value.size()) {
        const auto end = value.find(',', begin);
        const std::string token(value.substr(
            begin,
            end == std::string_view::npos ? value.size() - begin : end - begin
        ));
        if (token.empty() || !allowed.contains(token) ||
            !tags.insert(token).second) {
            throw std::runtime_error(
                "invalid or duplicate image evaluation tag"
            );
        }
        if (end == std::string_view::npos) {
            break;
        }
        begin = end + 1U;
    }
    return tags;
}

int run_imagegen_evaluate_manifest(const Options& options) {
    if (options.manifest.empty() || options.evaluation_manifest.empty() ||
        options.output.empty()) {
        throw std::invalid_argument(
            "imagegen-evaluate-manifest requires training --manifest, "
            "--evaluation-manifest, and --output"
        );
    }
    if (std::filesystem::exists(options.output)) {
        throw std::invalid_argument("image evaluation output already exists");
    }
    auto state = rlf::solstice::load_image_generation_checkpoint(options.checkpoint);
    enforce_image_checkpoint_profile(options, state);
    if (state.architecture !=
        rlf::solstice::ImageGenerationArchitecture::resonant_fabric) {
        throw std::runtime_error("image evaluation requires resonant-fabric");
    }
    state.resonant_fabric.set_backend(parse_backend(options.backend));
    const auto training_path = std::filesystem::weakly_canonical(options.manifest);
    const auto evaluation_path =
        std::filesystem::weakly_canonical(options.evaluation_manifest);
    struct TrainingReference final {
        std::filesystem::path target_path;
        std::vector<std::string> prompt_tokens;
        std::uint64_t difference_hash{};
        std::uint64_t average_hash{};
        std::array<std::uint8_t, 3U> mean_rgb{};
    };
    std::unordered_set<std::string> training_pixel_hashes;
    std::unordered_set<std::string> training_prompts;
    std::vector<TrainingReference> training_references;
    rlf::solstice::PatchQuiltBaseline patch_quilt(
        rlf::solstice::make_image_generation_profile_config(state.profile)
    );
    static_cast<void>(for_each_tsv(
        training_path, 8U, 8U,
        [&](const std::vector<std::string>& fields) {
            const auto target_path = resolve_imagegen_manifest_path(
                training_path.parent_path(), fields[3U]
            );
            if (rlf::core::sha256_hex(rlf::core::sha256_file(target_path)) !=
                fields[4U]) {
                throw std::runtime_error("training target changed before evaluation");
            }
            const auto image = rlf::solstice::load_image(target_path);
            training_pixel_hashes.insert(rlf::core::sha256_hex(
                rlf::core::sha256(std::span<const std::uint8_t>(image.rgb))
            ));
            training_references.push_back({
                .target_path = target_path,
                .prompt_tokens = image_prompt_tokens(fields[5U]),
                .difference_hash = rlf::solstice::image_difference_hash(image),
                .average_hash = rlf::solstice::image_average_hash(image),
                .mean_rgb = rlf::solstice::image_mean_rgb(image),
            });
            training_prompts.insert(fields[5U]);
            patch_quilt.train(image, fields[5U]);
        }
    ));
    OutputDirectoryTransaction transaction(options.output);
    const auto& evaluation_output = transaction.path();
    std::filesystem::create_directories(evaluation_output / "generated");
    std::filesystem::create_directories(evaluation_output / "nearest-example");
    std::filesystem::create_directories(evaluation_output / "patch-quilt");
    std::ofstream raw(evaluation_output / "predictions.tsv");
    if (!raw) {
        throw std::runtime_error("unable to create image evaluation predictions");
    }
    raw << "record_id\tprompt\tevaluation_tags\tprompt_seen_exactly_in_training\tgenerated_file\tgenerated_pixel_sha256\tmae\tmse\tpsnr_db\tssim\texact_pixel_fraction\tselected_modes\tunresolved_patches\texact_training_copy\tperceptual_training_copy\tcomposed_operations\tnearest_file\tnearest_prompt_jaccard\tnearest_ssim\tpatch_quilt_file\tpatch_quilt_ssim\n";
    std::size_t records = 0U;
    std::size_t unresolved = 0U;
    std::size_t exact_training_copies = 0U;
    std::size_t perceptual_training_copies = 0U;
    std::size_t composition_records = 0U;
    std::size_t unseen_prompt_records = 0U;
    std::size_t paraphrase_records = 0U;
    std::size_t natural_image_records = 0U;
    std::size_t multilingual_records = 0U;
    std::unordered_set<std::string> unique_outputs;
    struct PerceptualFingerprint final {
        std::uint64_t difference_hash{};
        std::uint64_t average_hash{};
        std::array<std::uint8_t, 3U> mean_rgb{};
    };
    std::vector<PerceptualFingerprint> perceptually_unique_outputs;
    double mae = 0.0;
    double mse = 0.0;
    double exact_pixels = 0.0;
    double ssim = 0.0;
    double nearest_ssim = 0.0;
    double patch_quilt_ssim = 0.0;
    double nearest_mae = 0.0;
    double patch_quilt_mae = 0.0;
    double composition_ssim = 0.0;
    double unseen_prompt_ssim = 0.0;
    double paraphrase_ssim = 0.0;
    double natural_image_ssim = 0.0;
    double multilingual_ssim = 0.0;
    static_cast<void>(for_each_tsv(
        evaluation_path, 8U, 9U,
        [&](const std::vector<std::string>& fields) {
            if (records >= options.maximum_audit_records) {
                throw std::runtime_error("image evaluation record limit exceeded");
            }
            const auto target_path = resolve_imagegen_manifest_path(
                evaluation_path.parent_path(), fields[3U]
            );
            if (rlf::core::sha256_hex(rlf::core::sha256_file(target_path)) !=
                fields[4U]) {
                throw std::runtime_error("evaluation target SHA-256 mismatch");
            }
            const auto target = rlf::solstice::load_image(target_path);
            rlf::solstice::ImageData base;
            if (fields[1U] == neutral_image_source_marker) {
                if (fields[2U] != neutral_image_source_hash()) {
                    throw std::runtime_error("evaluation neutral-source hash mismatch");
                }
                base = neutral_image_like(target);
            } else {
                const auto source_path = resolve_imagegen_manifest_path(
                    evaluation_path.parent_path(), fields[1U]
                );
                if (rlf::core::sha256_hex(rlf::core::sha256_file(source_path)) !=
                    fields[2U]) {
                    throw std::runtime_error("evaluation source SHA-256 mismatch");
                }
                base = rlf::solstice::load_image(source_path);
            }
            const auto operations = rlf::solstice::parse_resonant_image_prompt(fields[5U]);
            const auto tags = image_evaluation_tags(fields);
            const bool inferred_composition = operations.size() > 1U;
            if (!tags.empty() &&
                tags.contains("composition") != inferred_composition) {
                throw std::runtime_error(
                    "composition evaluation tag disagrees with parsed prompt"
                );
            }
            const bool is_composition = tags.empty()
                ? inferred_composition
                : tags.contains("composition");
            const bool prompt_seen_exactly = training_prompts.contains(fields[5U]);
            if (tags.contains("unseen_prompt") && prompt_seen_exactly) {
                throw std::runtime_error(
                    "unseen_prompt tag is contradicted by training manifest"
                );
            }
            const bool is_paraphrase = tags.contains("paraphrase");
            const bool is_natural_image = tags.contains("natural_image");
            const bool is_multilingual = tags.contains("multilingual");
            composition_records += is_composition ? 1U : 0U;
            unseen_prompt_records += prompt_seen_exactly ? 0U : 1U;
            paraphrase_records += is_paraphrase ? 1U : 0U;
            natural_image_records += is_natural_image ? 1U : 0U;
            multilingual_records += is_multilingual ? 1U : 0U;
            const auto generated = state.resonant_fabric.generate({
                .base_image = std::move(base),
                .transformations = operations,
                .capture_trace = false,
            });
            const auto quality = rlf::solstice::evaluate_resonant_image_quality(
                generated.image, target
            );
            const auto pixel_hash = rlf::core::sha256_hex(rlf::core::sha256(
                std::span<const std::uint8_t>(generated.image.rgb)
            ));
            const bool copied = training_pixel_hashes.contains(pixel_hash);
            const PerceptualFingerprint fingerprint{
                .difference_hash =
                    rlf::solstice::image_difference_hash(generated.image),
                .average_hash = rlf::solstice::image_average_hash(generated.image),
                .mean_rgb = rlf::solstice::image_mean_rgb(generated.image),
            };
            const bool perceptual_copy = std::any_of(
                training_references.begin(), training_references.end(),
                [&](const TrainingReference& trained) {
                    return (rlf::solstice::image_hash_hamming(
                                fingerprint.difference_hash,
                                trained.difference_hash
                            ) <= 3U ||
                            rlf::solstice::image_hash_hamming(
                                fingerprint.average_hash,
                                trained.average_hash
                            ) <= 3U) &&
                        rlf::solstice::image_mean_rgb_distance(
                            fingerprint.mean_rgb, trained.mean_rgb
                        ) <= 12U;
                }
            );
            const bool perceptually_new = std::none_of(
                perceptually_unique_outputs.begin(),
                perceptually_unique_outputs.end(),
                [&](const PerceptualFingerprint& previous) {
                    return (rlf::solstice::image_hash_hamming(
                                fingerprint.difference_hash,
                                previous.difference_hash
                            ) <= 3U ||
                            rlf::solstice::image_hash_hamming(
                                fingerprint.average_hash,
                                previous.average_hash
                            ) <= 3U) &&
                        rlf::solstice::image_mean_rgb_distance(
                            fingerprint.mean_rgb, previous.mean_rgb
                        ) <= 12U;
                }
            );
            if (perceptually_new) {
                perceptually_unique_outputs.push_back(fingerprint);
            }
            const bool unresolved_record =
                generated.operation_delta.unresolved_patch_transformations > 0U;
            exact_training_copies += copied ? 1U : 0U;
            perceptual_training_copies += perceptual_copy ? 1U : 0U;
            unresolved += unresolved_record ? 1U : 0U;
            unique_outputs.insert(pixel_hash);
            mae += quality.mean_absolute_error;
            mse += quality.mean_squared_error;
            exact_pixels += quality.exact_pixel_fraction;
            const double record_ssim = global_image_ssim(generated.image, target);
            ssim += record_ssim;
            composition_ssim += is_composition ? record_ssim : 0.0;
            unseen_prompt_ssim += prompt_seen_exactly ? 0.0 : record_ssim;
            paraphrase_ssim += is_paraphrase ? record_ssim : 0.0;
            natural_image_ssim += is_natural_image ? record_ssim : 0.0;
            multilingual_ssim += is_multilingual ? record_ssim : 0.0;
            const auto query_tokens = image_prompt_tokens(fields[5U]);
            std::size_t nearest_index = 0U;
            double nearest_score = -1.0;
            for (std::size_t index = 0U; index < training_references.size(); ++index) {
                const double score = image_prompt_jaccard(
                    query_tokens, training_references[index].prompt_tokens
                );
                if (score > nearest_score) {
                    nearest_score = score;
                    nearest_index = index;
                }
            }
            const auto nearest_image = rlf::solstice::load_image(
                training_references.at(nearest_index).target_path
            );
            if (nearest_image.width != target.width ||
                nearest_image.height != target.height) {
                throw std::runtime_error(
                    "nearest-example baseline requires matched target dimensions"
                );
            }
            const auto nearest_quality =
                rlf::solstice::evaluate_resonant_image_quality(nearest_image, target);
            const double nearest_record_ssim = global_image_ssim(nearest_image, target);
            nearest_ssim += nearest_record_ssim;
            nearest_mae += nearest_quality.mean_absolute_error;
            const auto quilt = patch_quilt.generate({
                .prompt = fields[5U],
                .width = target.width,
                .height = target.height,
                .seed = options.seed,
            });
            const auto quilt_quality =
                rlf::solstice::evaluate_resonant_image_quality(quilt.image, target);
            const double quilt_record_ssim = global_image_ssim(quilt.image, target);
            patch_quilt_ssim += quilt_record_ssim;
            patch_quilt_mae += quilt_quality.mean_absolute_error;
            const auto relative = std::filesystem::path("generated") /
                ("sample-" + std::to_string(records) + ".ppm");
            const auto nearest_relative = std::filesystem::path("nearest-example") /
                ("sample-" + std::to_string(records) + ".ppm");
            const auto quilt_relative = std::filesystem::path("patch-quilt") /
                ("sample-" + std::to_string(records) + ".ppm");
            rlf::solstice::VideoPrototypeFabric::save_ppm(
                evaluation_output / relative, generated.image
            );
            rlf::solstice::VideoPrototypeFabric::save_ppm(
                evaluation_output / nearest_relative, nearest_image
            );
            rlf::solstice::VideoPrototypeFabric::save_ppm(
                evaluation_output / quilt_relative, quilt.image
            );
            raw << fields[0U] << '\t' << fields[5U] << '\t'
                << (fields.size() == 9U ? fields[8U] : "") << '\t'
                << (prompt_seen_exactly ? 1 : 0) << '\t'
                << relative.generic_string() << '\t' << pixel_hash << '\t'
                << quality.mean_absolute_error << '\t'
                << quality.mean_squared_error << '\t'
                << quality.peak_signal_to_noise_db << '\t' << record_ssim << '\t'
                << quality.exact_pixel_fraction << '\t'
                << generated.selected_mode_ids.size() << '\t'
                << generated.operation_delta.unresolved_patch_transformations << '\t'
                << (copied ? 1 : 0) << '\t' << (perceptual_copy ? 1 : 0)
                << '\t' << operations.size() << '\t'
                << nearest_relative.generic_string() << '\t' << nearest_score
                << '\t' << nearest_record_ssim << '\t'
                << quilt_relative.generic_string() << '\t'
                << quilt_record_ssim << '\n';
            ++records;
        }
    ));
    if (records == 0U || !raw) {
        throw std::runtime_error("empty or unwritable image evaluation");
    }
    raw.close();
    if (!raw) {
        throw std::runtime_error("unable to finalize image evaluation predictions");
    }
    std::vector<std::filesystem::path> raw_paths;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(
             evaluation_output)) {
        std::error_code status_error;
        const auto status = entry.symlink_status(status_error);
        if (status_error || std::filesystem::is_symlink(status)) {
            throw std::runtime_error("image evaluation produced an unsafe artifact");
        }
        if (std::filesystem::is_regular_file(status)) {
            raw_paths.push_back(entry.path().lexically_relative(evaluation_output));
        }
    }
    std::sort(raw_paths.begin(), raw_paths.end());
    std::ofstream raw_manifest(evaluation_output / "raw_artifacts.tsv");
    if (!raw_manifest) {
        throw std::runtime_error("unable to create image evaluation artifact manifest");
    }
    raw_manifest << "path\tsha256\tbytes\n";
    for (const auto& relative : raw_paths) {
        const auto path = evaluation_output / relative;
        const auto bytes = std::filesystem::file_size(path);
        if (bytes == 0U) {
            throw std::runtime_error("image evaluation produced an empty artifact");
        }
        raw_manifest << relative.generic_string() << '\t'
                     << rlf::core::sha256_hex(rlf::core::sha256_file(path))
                     << '\t' << bytes << '\n';
    }
    raw_manifest.close();
    if (!raw_manifest) {
        throw std::runtime_error("unable to finalize image artifact manifest");
    }
    const std::string raw_manifest_sha = rlf::core::sha256_hex(
        rlf::core::sha256_file(evaluation_output / "raw_artifacts.tsv")
    );
    const double denominator = static_cast<double>(records);
    std::ofstream summary(evaluation_output / "summary.json");
    summary << "{\n"
            << "  \"schema\": \"rlf-imagegen-frozen-evaluation-v1\",\n"
            << "  \"checkpoint_sha256\": \""
            << rlf::core::sha256_hex(rlf::core::sha256_file(options.checkpoint))
            << "\",\n  \"training_manifest_sha256\": \""
            << rlf::core::sha256_hex(rlf::core::sha256_file(training_path))
            << "\",\n  \"evaluation_manifest_sha256\": \""
            << rlf::core::sha256_hex(rlf::core::sha256_file(evaluation_path))
            << "\",\n  \"raw_artifact_manifest_sha256\": \""
            << raw_manifest_sha
            << "\",\n  \"records\": " << records
            << ",\n  \"mean_absolute_error\": " << mae / denominator
            << ",\n  \"mean_squared_error\": " << mse / denominator
            << ",\n  \"mean_ssim\": " << ssim / denominator
            << ",\n  \"mean_exact_pixel_fraction\": " << exact_pixels / denominator
            << ",\n  \"nearest_example_mean_absolute_error\": "
            << nearest_mae / denominator
            << ",\n  \"nearest_example_mean_ssim\": "
            << nearest_ssim / denominator
            << ",\n  \"patch_quilt_mean_absolute_error\": "
            << patch_quilt_mae / denominator
            << ",\n  \"patch_quilt_mean_ssim\": "
            << patch_quilt_ssim / denominator
            << ",\n  \"unresolved_record_rate\": "
            << static_cast<double>(unresolved) / denominator
            << ",\n  \"exact_training_copy_rate\": "
            << static_cast<double>(exact_training_copies) / denominator
            << ",\n  \"perceptual_training_copy_rate\": "
            << static_cast<double>(perceptual_training_copies) / denominator
            << ",\n  \"unique_output_fraction\": "
            << static_cast<double>(unique_outputs.size()) / denominator
            << ",\n  \"perceptually_unique_output_fraction\": "
            << static_cast<double>(perceptually_unique_outputs.size()) /
                denominator
            << ",\n  \"composition_records\": " << composition_records
            << ",\n  \"composition_mean_ssim\": "
            << (composition_records == 0U ? 0.0 :
                composition_ssim / static_cast<double>(composition_records))
            << ",\n  \"unseen_prompt_records\": " << unseen_prompt_records
            << ",\n  \"unseen_prompt_mean_ssim\": "
            << (unseen_prompt_records == 0U ? 0.0 :
                unseen_prompt_ssim / static_cast<double>(unseen_prompt_records))
            << ",\n  \"paraphrase_records\": " << paraphrase_records
            << ",\n  \"paraphrase_mean_ssim\": "
            << (paraphrase_records == 0U ? 0.0 :
                paraphrase_ssim / static_cast<double>(paraphrase_records))
            << ",\n  \"natural_image_records\": " << natural_image_records
            << ",\n  \"natural_image_mean_ssim\": "
            << (natural_image_records == 0U ? 0.0 :
                natural_image_ssim / static_cast<double>(natural_image_records))
            << ",\n  \"multilingual_records\": " << multilingual_records
            << ",\n  \"multilingual_mean_ssim\": "
            << (multilingual_records == 0U ? 0.0 :
                multilingual_ssim / static_cast<double>(multilingual_records))
            << ",\n  \"internal_baselines_present\": true,\n"
            << "  \"external_diffusion_baseline_present\": false,\n"
            << "  \"frontier_claim_authorized\": false\n}\n";
    if (!summary) {
        throw std::runtime_error("unable to write image evaluation summary");
    }
    summary.close();
    if (!summary) {
        throw std::runtime_error("unable to finalize image evaluation summary");
    }
    transaction.commit();
    std::cout << "evaluation_records=" << records << '\n'
              << "mean_ssim=" << ssim / denominator << '\n'
              << "exact_training_copy_rate="
              << static_cast<double>(exact_training_copies) / denominator << '\n'
              << "frontier_claim_authorized=false\n";
    return 0;
}

int run_imagegen_generate(const Options& options) {
    if (options.output.empty() ||
        (options.transformations.empty() && options.prompt.empty())) {
        throw std::invalid_argument(
            "imagegen-generate requires --output and --prompt/--transform"
        );
    }
    if (std::filesystem::exists(options.output)) {
        throw std::invalid_argument("imagegen output already exists");
    }
    auto state = rlf::solstice::load_image_generation_checkpoint(
        options.checkpoint
    );
    enforce_image_checkpoint_profile(options, state);
    if (state.architecture !=
        rlf::solstice::ImageGenerationArchitecture::resonant_fabric) {
        throw std::runtime_error(
            "imagegen-generate requires the resonant-fabric architecture"
        );
    }
    state.resonant_fabric.set_backend(parse_backend(options.backend));
    const auto base = options.image.has_value()
        ? rlf::solstice::load_image(*options.image)
        : make_neutral_image(
            options.image_width,
            options.image_height,
            state.resonant_fabric.config()
        );
    std::vector<std::string> transformations = options.transformations;
    if (transformations.empty()) {
        transformations = rlf::solstice::parse_resonant_image_prompt(
            options.prompt
        );
    }
    const auto result = state.resonant_fabric.generate({
        .base_image = base,
        .transformations = transformations,
        .capture_trace = false,
    });
    rlf::solstice::VideoPrototypeFabric::save_ppm(options.output, result.image);
    const std::filesystem::path metadata = options.output.string() + ".json";
    std::ofstream report(metadata, std::ios::binary | std::ios::trunc);
    if (!report) {
        std::error_code cleanup_error;
        std::filesystem::remove(options.output, cleanup_error);
        throw std::runtime_error("unable to create image-generation metadata");
    }
    report << "{\n"
           << "  \"schema\": \"rlf-resonant-image-generation-v1\",\n"
           << "  \"generation_mode\": \""
           << (options.image.has_value() ? "source-transformation" : "prompt-only-neutral-seed")
           << "\",\n"
           << "  \"claim_boundary\": \"non-neural sparse open-vocabulary "
              "prompt retrieval with controlled learned image transformations; "
              "not diffusion parity or frontier evidence\",\n"
           << "  \"checkpoint_sha256\": \""
           << rlf::core::sha256_hex(
                rlf::core::sha256_file(options.checkpoint)
              ) << "\",\n"
           << "  \"deterministic_hash\": " << result.deterministic_hash << ",\n"
           << "  \"selected_modes\": " << result.selected_mode_ids.size()
           << "\n}\n";
    report.flush();
    if (!report) {
        throw std::runtime_error("failed while writing image-generation metadata");
    }
    std::cout << "generated_image=" << options.output.string() << '\n'
              << "metadata=" << metadata.string() << '\n'
              << "selected_modes=" << result.selected_mode_ids.size() << '\n'
              << "generation_mode="
              << (options.image.has_value() ? "source-transformation" : "prompt-only-neutral-seed")
              << '\n'
              << "claim_boundary=controlled sparse open-vocabulary transformation "
                 "retrieval, not diffusion parity or frontier image generation\n";
    return 0;
}

int run_imagegen_inspect(const Options& options) {
    const auto summary = rlf::solstice::inspect_image_generation_checkpoint(
        options.checkpoint
    );
    std::cout << "format_version=" << summary.format_version << '\n'
              << "profile=" << rlf::solstice::to_string(summary.profile) << '\n'
              << "architecture="
              << rlf::solstice::to_string(summary.architecture) << '\n'
              << "training_step=" << summary.training_step << '\n'
              << "images_seen=" << summary.images_seen << '\n'
              << "learned_modes=" << summary.learned_modes << '\n'
              << "prompt_language_records="
              << summary.prompt_language_records << '\n'
              << "prompt_language_words=" << summary.prompt_language_words
              << '\n'
              << "prompt_semantic_modes=" << summary.prompt_semantic_modes
              << '\n'
              << "completed_shards=" << summary.completed_shards << '\n'
              << "file_sha256=" << summary.file_sha256 << '\n'
              << "payload_sha256=" << summary.payload_sha256 << '\n'
              << "deterministic_model_hash=0x" << std::hex
              << summary.deterministic_model_hash << std::dec << '\n';
    return 0;
}

int run_demo(const Options& options) {
    rlf::solstice::SolsticeModel model({}, options.seed);
    model.set_backend(parse_backend(options.backend));
    model.bootstrap();
    rlf::solstice::ToolRuntime tools = make_tools(options);
    std::cout << "--- dialogue ---\n";
    print_response(model.respond(
        "What can you do?", nullptr, &tools, generation_settings(options)
    ));
    std::cout << "--- tool call ---\n";
    print_response(model.respond(
        "Calculate (12 + 8) * 3", nullptr, &tools, generation_settings(options)
    ));
    return 0;
}

int run(const int argument_count, char** argument_values) {
    const Options options = parse_options(argument_count, argument_values);
    if (options.help || options.command.empty() || options.command == "help") {
        print_help(std::cout);
        return 0;
    }
    if (options.command == "bootstrap") {
        return run_bootstrap(options);
    }
    if (options.command == "profile-info") {
        return run_profile_info(options);
    }
    if (options.command == "device-info") {
        return run_device_info(options);
    }
    if (options.command == "train-text") {
        return run_train_text(options);
    }
    if (options.command == "train-dialogue") {
        return run_train_dialogue(options);
    }
    if (options.command == "train-instructions") {
        return run_train_instructions(options);
    }
    if (options.command == "train-preferences") {
        return run_train_preferences(options);
    }
    if (options.command == "train-vision") {
        return run_train_vision(options);
    }
    if (options.command == "train-tools") {
        return run_train_tools(options);
    }
    if (options.command == "train-facts") {
        return run_train_facts(options);
    }
    if (options.command == "train-rules") {
        return run_train_rules(options);
    }
    if (options.command == "audit-data") {
        return run_audit_data(options);
    }
    if (options.command == "train-data") {
        return run_train_data(options);
    }
    if (options.command == "generate-video") {
        return run_generate_video(options);
    }
    if (options.command == "evaluate-video") {
        return run_evaluate_video(options);
    }
    if (options.command == "evaluate-batch") {
        return run_evaluate_batch(options);
    }
    if (options.command == "imagegen-profile-info") {
        return run_imagegen_profile_info(options);
    }
    if (options.command == "imagegen-bootstrap") {
        return run_imagegen_bootstrap(options);
    }
    if (options.command == "imagegen-train-language-ledger") {
        return run_imagegen_train_language_ledger(options);
    }
    if (options.command == "imagegen-train-pair") {
        return run_imagegen_train_pair(options);
    }
    if (options.command == "imagegen-train-manifest") {
        return run_imagegen_train_manifest(options);
    }
    if (options.command == "imagegen-audit-pairs") {
        return run_imagegen_audit_pairs(options);
    }
    if (options.command == "imagegen-evaluate-manifest") {
        return run_imagegen_evaluate_manifest(options);
    }
    if (options.command == "imagegen-generate") {
        return run_imagegen_generate(options);
    }
    if (options.command == "imagegen-inspect") {
        return run_imagegen_inspect(options);
    }
    if (options.command == "imagegen-verify") {
        const auto state = rlf::solstice::load_image_generation_checkpoint(
            options.checkpoint
        );
        enforce_image_checkpoint_profile(options, state);
        std::cout << "Image-generation checkpoint is valid: "
                  << options.checkpoint.string() << '\n';
        return 0;
    }
    if (options.command == "imagegen-verify-artifacts") {
        if (options.manifest.empty()) {
            throw std::invalid_argument(
                "imagegen-verify-artifacts requires --manifest"
            );
        }
        const auto report =
            rlf::solstice::verify_image_generation_artifact_manifest(
                options.manifest
            );
        std::cout << "bundle_integrity_verified="
                  << (report.bundle_integrity_verified ? "true" : "false")
                  << '\n'
                  << "origin_authenticated="
                  << (report.origin_authenticated ? "true" : "false") << '\n'
                  << "state_of_art_claim_proven="
                  << (report.state_of_art_claim_proven ? "true" : "false")
                  << '\n'
                  << "manifest_sha256=" << report.manifest_sha256 << '\n';
        for (const auto& failure : report.failures) {
            std::cout << "failure=" << failure << '\n';
        }
        return report.bundle_integrity_verified ? 0 : 1;
    }
    if (options.command == "induce-rule") {
        return run_induce_rule(options);
    }
    if (options.command == "reason") {
        return run_reason(options);
    }
    if (options.command == "ask") {
        return run_ask(options);
    }
    if (options.command == "chat") {
        return run_chat(options);
    }
    if (options.command == "inspect-image") {
        return run_inspect_image(options);
    }
    if (options.command == "inspect-checkpoint") {
        return run_inspect_checkpoint(options);
    }
    if (options.command == "verify-checkpoint") {
        if (options.enforce_profile) {
            static_cast<void>(
                rlf::solstice::inspect_solstice_checkpoint_for_profile(
                    options.checkpoint,
                    rlf::solstice::parse_profile(options.profile)
                )
            );
        } else {
            static_cast<void>(rlf::solstice::load_solstice_checkpoint(
                options.checkpoint,
                checkpoint_limits(options)
            ));
        }
        std::cout << "Checkpoint is valid: " << options.checkpoint.string() << '\n';
        return 0;
    }
    if (options.command == "demo") {
        return run_demo(options);
    }
    throw std::invalid_argument("unknown Solstice command: " + options.command);
}

}  // namespace

int main(const int argument_count, char** argument_values) {
    try {
        return run(argument_count, argument_values);
    } catch (const std::exception& error) {
        std::cerr << "solstice: " << error.what() << '\n';
        return 2;
    }
}
