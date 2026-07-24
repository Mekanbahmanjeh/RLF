#include "rlf/solstice/evaluation_runner.hpp"

#include "rlf/core/sha256.hpp"
#include "rlf/solstice/solstice_model.hpp"
#include "rlf/solstice/tool_protocol.hpp"

#include <atomic>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rlf::solstice {
namespace {

struct EvaluationRequest final {
    std::string id;
    std::filesystem::path prompt_path;
    std::string prompt_sha256;
    std::optional<std::filesystem::path> image_path;
    std::string image_sha256;
};

[[nodiscard]] std::string escape_json(const std::string_view value) {
    std::ostringstream output;
    for (const char raw : value) {
        const auto character = static_cast<unsigned char>(raw);
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << static_cast<unsigned int>(character)
                           << std::dec << std::setfill(' ');
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
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

[[nodiscard]] std::string lowercase_ascii(std::string value) {
    std::ranges::transform(value, value.begin(), [](const unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

[[nodiscard]] bool valid_id(const std::string_view value) noexcept {
    if (value.empty() || value.size() > 160U) {
        return false;
    }
    for (const char raw : value) {
        const auto character = static_cast<unsigned char>(raw);
        const bool accepted =
            (character >= static_cast<unsigned char>('a') &&
             character <= static_cast<unsigned char>('z')) ||
            (character >= static_cast<unsigned char>('A') &&
             character <= static_cast<unsigned char>('Z')) ||
            (character >= static_cast<unsigned char>('0') &&
             character <= static_cast<unsigned char>('9')) ||
            character == static_cast<unsigned char>('-') ||
            character == static_cast<unsigned char>('_') ||
            character == static_cast<unsigned char>('.');
        if (!accepted) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] std::filesystem::path resolve_below(
    const std::filesystem::path& root,
    const std::string_view relative_text
) {
    const std::filesystem::path relative(relative_text);
    if (relative.empty() || relative.is_absolute()) {
        throw std::runtime_error("evaluation paths must be nonempty and relative");
    }
    const std::filesystem::path candidate = std::filesystem::weakly_canonical(
        root / relative
    );
    const std::filesystem::path relation = candidate.lexically_relative(root);
    if (relation.empty() || relation.is_absolute()) {
        throw std::runtime_error("evaluation path escapes the manifest directory");
    }
    for (const std::filesystem::path& component : relation) {
        if (component == "..") {
            throw std::runtime_error("evaluation path escapes the manifest directory");
        }
    }
    std::error_code error;
    if (!std::filesystem::is_regular_file(candidate, error) || error) {
        throw std::runtime_error("evaluation input is not a regular file: " + candidate.string());
    }
    return candidate;
}

[[nodiscard]] std::vector<EvaluationRequest> load_requests(
    const std::filesystem::path& manifest_path
) {
    std::error_code error;
    const std::filesystem::path canonical_manifest =
        std::filesystem::weakly_canonical(manifest_path, error);
    if (error || !std::filesystem::is_regular_file(canonical_manifest, error) || error) {
        throw std::runtime_error("evaluation manifest is not a regular file");
    }
    const std::filesystem::path root = canonical_manifest.parent_path();
    std::ifstream input(canonical_manifest);
    if (!input) {
        throw std::runtime_error("unable to open evaluation manifest");
    }
    std::vector<EvaluationRequest> requests;
    std::unordered_set<std::string> ids;
    std::string line;
    std::size_t line_number = 0U;
    while (std::getline(input, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line.front() == '#') {
            continue;
        }
        const std::vector<std::string> fields = split_tabs(line);
        if (fields.size() != 5U) {
            throw std::runtime_error(
                "expected 5 evaluation-manifest fields at line " +
                std::to_string(line_number)
            );
        }
        if (!valid_id(fields[0U]) || !ids.emplace(fields[0U]).second) {
            throw std::runtime_error("invalid or duplicate evaluation ID at line " +
                                     std::to_string(line_number));
        }
        if (!core::is_sha256_hex(fields[2U])) {
            throw std::runtime_error("invalid prompt SHA-256 at line " +
                                     std::to_string(line_number));
        }
        EvaluationRequest request;
        request.id = fields[0U];
        request.prompt_path = resolve_below(root, fields[1U]);
        request.prompt_sha256 = core::sha256_hex(core::sha256_file(request.prompt_path));
        if (request.prompt_sha256 != lowercase_ascii(fields[2U])) {
            throw std::runtime_error("prompt SHA-256 mismatch for " + request.id);
        }
        if (fields[3U] == "-" && fields[4U] == "-") {
            request.image_sha256 = "-";
        } else {
            if (fields[3U] == "-" || !core::is_sha256_hex(fields[4U])) {
                throw std::runtime_error("invalid image fields for " + request.id);
            }
            request.image_path = resolve_below(root, fields[3U]);
            request.image_sha256 = core::sha256_hex(
                core::sha256_file(*request.image_path)
            );
            if (request.image_sha256 != lowercase_ascii(fields[4U])) {
                throw std::runtime_error("image SHA-256 mismatch for " + request.id);
            }
        }
        requests.push_back(std::move(request));
    }
    if (!input.eof()) {
        throw std::runtime_error("failed while reading evaluation manifest");
    }
    if (requests.empty()) {
        throw std::runtime_error("evaluation manifest contains no examples");
    }
    return requests;
}

[[nodiscard]] std::string read_bounded(
    const std::filesystem::path& path,
    const std::size_t maximum_bytes
) {
    const std::uintmax_t size = std::filesystem::file_size(path);
    if (size > maximum_bytes ||
        size > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("evaluation prompt exceeds configured byte limit");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("unable to read evaluation prompt");
    }
    std::string content(static_cast<std::size_t>(size), '\0');
    if (!content.empty()) {
        input.read(content.data(), static_cast<std::streamsize>(content.size()));
    }
    if (!input) {
        throw std::runtime_error("failed while reading evaluation prompt");
    }
    return content;
}

[[nodiscard]] std::string hash_text(const std::uint64_t value) {
    std::ostringstream output;
    output << std::hex << std::setw(16) << std::setfill('0') << value;
    return output.str();
}

[[nodiscard]] std::string response_json(
    const EvaluationRequest& request,
    const SolsticeResponse& response,
    const EvaluationBatchOptions& options,
    const std::uint64_t model_hash,
    const std::uint64_t elapsed_microseconds,
    const bool tools_enabled,
    const std::string_view manifest_sha256
) {
    std::ostringstream output;
    output << std::boolalpha << std::fixed << std::setprecision(9);
    output << "{\n  \"schema\": \"rlf-evaluation-result-v1\",\n"
           << "  \"id\": \"" << escape_json(request.id) << "\",\n"
           << "  \"manifest_sha256\": \"" << manifest_sha256 << "\",\n"
           << "  \"prompt_sha256\": \"" << request.prompt_sha256 << "\",\n"
           << "  \"image_sha256\": ";
    if (request.image_path.has_value()) {
        output << '"' << request.image_sha256 << '"';
    } else {
        output << "null";
    }
    output << ",\n  \"checkpoint_sha256\": \""
           << options.checkpoint_sha256 << "\",\n"
           << "  \"model_hash\": \"" << hash_text(model_hash) << "\",\n"
           << "  \"backend\": \"" << escape_json(options.backend_name) << "\",\n"
           << "  \"tools_enabled\": " << tools_enabled << ",\n"
           << "  \"generation\": {\"maximum_tokens\": "
           << options.generation.maximum_tokens << ", \"top_k\": "
           << options.generation.top_k << ", \"temperature\": "
           << options.generation.temperature << ", \"deterministic\": "
           << options.generation.deterministic << ", \"seed\": "
           << options.generation.seed << "},\n"
           << "  \"elapsed_microseconds\": " << elapsed_microseconds << ",\n"
           << "  \"response\": {\"text\": \"" << escape_json(response.text)
           << "\", \"uncertainty\": " << response.uncertainty;
    if (response.vision.has_value()) {
        output << ", \"vision\": {\"description\": \""
               << escape_json(response.vision->description)
               << "\", \"confidence\": " << response.vision->confidence
               << ", \"width\": " << response.vision->width
               << ", \"height\": " << response.vision->height
               << ", \"region_count\": " << response.vision->regions.size()
               << "}";
    }
    if (response.tool_proposal.has_value()) {
        output << ", \"tool_proposal\": "
               << ToolRuntime::serialize_call(response.tool_proposal->call)
               << ", \"tool_confidence\": "
               << response.tool_proposal->confidence;
    }
    if (response.tool_result.has_value()) {
        output << ", \"tool_result\": "
               << ToolRuntime::serialize_result(*response.tool_result);
    }
    output << "}\n}\n";
    return output.str();
}

[[nodiscard]] std::filesystem::path unique_sibling(
    const std::filesystem::path& target,
    const std::string_view marker
) {
    static std::atomic<std::uint64_t> counter{};
    for (std::size_t attempt = 0U; attempt < 1'000U; ++attempt) {
        const auto value = counter.fetch_add(1U, std::memory_order_relaxed);
        const std::filesystem::path candidate =
            target.string() + "." + std::string(marker) + "." + std::to_string(value);
        if (!std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
    throw std::runtime_error("unable to allocate evaluation temporary path");
}

void write_new_atomic(
    const std::filesystem::path& target,
    const std::string_view content
) {
    if (std::filesystem::exists(target)) {
        throw std::runtime_error("refusing to overwrite evaluation artifact: " +
                                 target.string());
    }
    const std::filesystem::path temporary = unique_sibling(target, "partial");
    try {
        {
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            if (!output) {
                throw std::runtime_error("unable to create evaluation artifact");
            }
            output.write(content.data(), static_cast<std::streamsize>(content.size()));
            output.flush();
            if (!output) {
                throw std::runtime_error("failed while writing evaluation artifact");
            }
        }
        std::filesystem::rename(temporary, target);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

void replace_preserving_prior(
    const std::filesystem::path& target,
    const std::string_view content
) {
    if (!std::filesystem::exists(target)) {
        write_new_atomic(target, content);
        return;
    }
    if (!std::filesystem::is_regular_file(target)) {
        throw std::runtime_error("evaluation summary target is not a regular file");
    }
    const std::filesystem::path backup = unique_sibling(target, "previous");
    std::filesystem::rename(target, backup);
    try {
        write_new_atomic(target, content);
        std::filesystem::remove(backup);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(target, ignored);
        std::filesystem::rename(backup, target, ignored);
        throw;
    }
}

[[nodiscard]] std::string run_identity_text(
    const EvaluationBatchOptions& options,
    const std::string_view manifest_sha256,
    const std::uint64_t model_hash,
    const bool tools_enabled
) {
    std::ostringstream output;
    output << std::boolalpha << std::setprecision(17)
           << "rlf-evaluation-identity-v1\t" << manifest_sha256 << '\t'
           << options.checkpoint_sha256 << '\t' << hash_text(model_hash) << '\t'
           << options.backend_name << '\t' << options.generation.maximum_tokens << '\t'
           << options.generation.top_k << '\t' << options.generation.temperature << '\t'
           << options.generation.deterministic << '\t' << options.generation.seed << '\t'
           << tools_enabled << '\t' << options.allow_images << '\n';
    return output.str();
}

void establish_run_identity(
    const std::filesystem::path& path,
    const std::string_view expected
) {
    if (!std::filesystem::exists(path)) {
        write_new_atomic(path, expected);
        return;
    }
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("evaluation run identity is not a regular file");
    }
    const std::uintmax_t size = std::filesystem::file_size(path);
    if (size > 16U * 1024U) {
        throw std::runtime_error("evaluation run identity is unexpectedly large");
    }
    const std::string actual = read_bounded(path, 16U * 1024U);
    if (actual != expected) {
        throw std::runtime_error(
            "evaluation output directory belongs to a different manifest, "
            "checkpoint, model, backend, generation setting, or tool policy"
        );
    }
}

[[nodiscard]] std::optional<std::uint64_t> completed_result_elapsed(
    const std::filesystem::path& result_path,
    const std::filesystem::path& metadata_path,
    const EvaluationRequest& request,
    const std::string_view checkpoint_sha256,
    const std::uint64_t model_hash
) {
    const bool result_exists = std::filesystem::exists(result_path);
    const bool metadata_exists = std::filesystem::exists(metadata_path);
    if (!result_exists && !metadata_exists) {
        return std::nullopt;
    }
    if (result_exists && !metadata_exists) {
        const std::filesystem::path orphan = unique_sibling(result_path, "orphan");
        std::filesystem::rename(result_path, orphan);
        return std::nullopt;
    }
    if (!result_exists || !std::filesystem::is_regular_file(result_path) ||
        !std::filesystem::is_regular_file(metadata_path)) {
        throw std::runtime_error("incomplete evaluation resume state for " + request.id);
    }
    std::ifstream input(metadata_path);
    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("empty evaluation metadata for " + request.id);
    }
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    const std::vector<std::string> fields = split_tabs(line);
    if ((fields.size() != 6U && fields.size() != 7U) ||
        fields[0U] != request.id ||
        fields[1U] != request.prompt_sha256 || fields[2U] != request.image_sha256 ||
        fields[3U] != checkpoint_sha256 || fields[4U] != hash_text(model_hash) ||
        !core::is_sha256_hex(fields[5U]) ||
        core::sha256_hex(core::sha256_file(result_path)) != fields[5U]) {
        throw std::runtime_error("evaluation resume metadata mismatch for " + request.id);
    }
    std::string extra;
    if (std::getline(input, extra)) {
        throw std::runtime_error("unexpected extra evaluation metadata for " + request.id);
    }
    if (fields.size() == 6U) {
        return 0U;
    }
    std::uint64_t elapsed{};
    const auto [end, error] = std::from_chars(
        fields[6U].data(), fields[6U].data() + fields[6U].size(), elapsed
    );
    if (error != std::errc{} || end != fields[6U].data() + fields[6U].size()) {
        throw std::runtime_error("invalid evaluation timing metadata for " + request.id);
    }
    return elapsed;
}

}  // namespace

EvaluationBatchReport run_evaluation_batch(
    const SolsticeModel& model,
    const EvaluationBatchOptions& options,
    ToolRuntime* tools
) {
    if (options.manifest_path.empty() || options.output_directory.empty() ||
        !core::is_sha256_hex(options.checkpoint_sha256) ||
        options.backend_name.empty() ||
        options.maximum_prompt_bytes == 0U ||
        options.generation.maximum_tokens == 0U || options.generation.top_k == 0U ||
        !std::isfinite(options.generation.temperature) ||
        options.generation.temperature <= 0.0) {
        throw std::invalid_argument("invalid evaluation batch options");
    }
    const std::vector<EvaluationRequest> requests = load_requests(options.manifest_path);
    for (const EvaluationRequest& request : requests) {
        if (!options.allow_images && request.image_path.has_value()) {
            throw std::runtime_error(
                "evaluation image input is disabled by the selected profile"
            );
        }
        const std::uintmax_t prompt_size = std::filesystem::file_size(request.prompt_path);
        if (prompt_size > options.maximum_prompt_bytes) {
            throw std::runtime_error("evaluation prompt exceeds configured byte limit");
        }
    }
    std::filesystem::create_directories(options.output_directory);
    if (!std::filesystem::is_directory(options.output_directory)) {
        throw std::runtime_error("evaluation output is not a directory");
    }

    EvaluationBatchReport report;
    report.manifest_sha256 = core::sha256_hex(core::sha256_file(options.manifest_path));
    report.checkpoint_sha256 = options.checkpoint_sha256;
    report.model_hash = model.deterministic_hash();
    report.total_examples = requests.size();
    report.output_directory = options.output_directory;
    establish_run_identity(
        options.output_directory / "run_identity.tsv",
        run_identity_text(
            options, report.manifest_sha256, report.model_hash, tools != nullptr
        )
    );

    for (const EvaluationRequest& request : requests) {
        const std::filesystem::path result_path =
            options.output_directory / (request.id + ".json");
        const std::filesystem::path metadata_path =
            options.output_directory / (request.id + ".meta.tsv");
        const std::optional<std::uint64_t> completed_elapsed =
            completed_result_elapsed(
                result_path, metadata_path, request, options.checkpoint_sha256,
                report.model_hash
            );
        if (completed_elapsed.has_value()) {
            ++report.resumed_examples;
            if (*completed_elapsed >
                std::numeric_limits<std::uint64_t>::max() -
                    report.inference_microseconds) {
                throw std::overflow_error("evaluation timing overflow");
            }
            report.inference_microseconds += *completed_elapsed;
            continue;
        }
        const std::string prompt = read_bounded(
            request.prompt_path, options.maximum_prompt_bytes
        );
        const auto begin = std::chrono::steady_clock::now();
        const SolsticeResponse response = model.respond_file(
            prompt, request.image_path, tools, options.generation
        );
        const auto end = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            end - begin
        ).count();
        const std::uint64_t elapsed_microseconds = elapsed < 0
            ? 0U
            : static_cast<std::uint64_t>(elapsed);
        const std::string result = response_json(
            request, response, options, report.model_hash, elapsed_microseconds,
            tools != nullptr, report.manifest_sha256
        );
        write_new_atomic(result_path, result);
        const std::string result_sha256 = core::sha256_hex(
            core::sha256_file(result_path)
        );
        const std::string metadata = request.id + '\t' + request.prompt_sha256 +
            '\t' + request.image_sha256 + '\t' + options.checkpoint_sha256 +
            '\t' + hash_text(report.model_hash) + '\t' + result_sha256 +
            '\t' + std::to_string(elapsed_microseconds) + '\n';
        write_new_atomic(metadata_path, metadata);
        ++report.produced_examples;
        if (elapsed_microseconds >
            std::numeric_limits<std::uint64_t>::max() - report.inference_microseconds) {
            throw std::overflow_error("evaluation timing overflow");
        }
        report.inference_microseconds += elapsed_microseconds;
        report.new_inference_microseconds += elapsed_microseconds;
    }
    replace_preserving_prior(
        options.output_directory / "run_summary.json",
        evaluation_batch_report_json(report)
    );
    return report;
}

std::string evaluation_batch_report_json(const EvaluationBatchReport& report) {
    std::ostringstream output;
    output << "{\n"
           << "  \"schema\": \"rlf-evaluation-batch-v1\",\n"
           << "  \"manifest_sha256\": \"" << report.manifest_sha256 << "\",\n"
           << "  \"checkpoint_sha256\": \"" << report.checkpoint_sha256 << "\",\n"
           << "  \"model_hash\": \"" << hash_text(report.model_hash) << "\",\n"
           << "  \"total_examples\": " << report.total_examples << ",\n"
           << "  \"produced_examples\": " << report.produced_examples << ",\n"
           << "  \"resumed_examples\": " << report.resumed_examples << ",\n"
           << "  \"artifact_inference_microseconds\": "
           << report.inference_microseconds << ",\n"
           << "  \"new_inference_microseconds\": "
           << report.new_inference_microseconds << ",\n"
           << "  \"output_directory\": \""
           << escape_json(report.output_directory.generic_string()) << "\"\n"
           << "}\n";
    return output.str();
}

}  // namespace rlf::solstice
