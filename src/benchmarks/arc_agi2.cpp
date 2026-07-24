#include "rlf/benchmarks/arc_agi2.hpp"

#include "rlf/core/sha256.hpp"

#include <algorithm>
#include <charconv>
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
#include <utility>
#include <vector>

namespace rlf::benchmarks {
namespace {

constexpr std::uintmax_t maximum_arc_file_bytes = 16U * 1024U * 1024U;

[[nodiscard]] std::string read_file(const std::filesystem::path& path) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error) {
        throw std::runtime_error("ARC input is not a regular file: " + path.string());
    }
    const std::uintmax_t size = std::filesystem::file_size(path, error);
    if (error || size > maximum_arc_file_bytes ||
        size > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("ARC input exceeds the file-size limit");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("unable to open ARC input: " + path.string());
    }
    std::string result(static_cast<std::size_t>(size), '\0');
    if (!result.empty()) {
        input.read(result.data(), static_cast<std::streamsize>(result.size()));
    }
    if (!input) {
        throw std::runtime_error("failed while reading ARC input: " + path.string());
    }
    return result;
}

[[nodiscard]] std::string escape_json(const std::string_view value) {
    std::ostringstream output;
    for (const char raw : value) {
        const auto character = static_cast<unsigned char>(raw);
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
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

class Cursor final {
public:
    explicit Cursor(const std::string_view text) : text_(text) {}

    void whitespace() noexcept {
        while (position_ < text_.size()) {
            const char character = text_[position_];
            if (character != ' ' && character != '\t' &&
                character != '\r' && character != '\n') {
                break;
            }
            ++position_;
        }
    }

    void expect(const char expected) {
        whitespace();
        if (position_ >= text_.size() || text_[position_] != expected) {
            throw std::invalid_argument("invalid ARC JSON structure");
        }
        ++position_;
    }

    [[nodiscard]] bool consume(const char expected) noexcept {
        whitespace();
        if (position_ < text_.size() && text_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    [[nodiscard]] bool finished() noexcept {
        whitespace();
        return position_ == text_.size();
    }

    [[nodiscard]] std::string string() {
        expect('"');
        std::string output;
        while (position_ < text_.size()) {
            const char character = text_[position_++];
            if (character == '"') {
                return output;
            }
            if (character != '\\') {
                output.push_back(character);
                continue;
            }
            if (position_ >= text_.size()) {
                throw std::invalid_argument("unterminated ARC JSON escape");
            }
            const char escaped = text_[position_++];
            switch (escaped) {
                case '"': output.push_back('"'); break;
                case '\\': output.push_back('\\'); break;
                case '/': output.push_back('/'); break;
                case 'b': output.push_back('\b'); break;
                case 'f': output.push_back('\f'); break;
                case 'n': output.push_back('\n'); break;
                case 'r': output.push_back('\r'); break;
                case 't': output.push_back('\t'); break;
                default: throw std::invalid_argument("unsupported ARC JSON escape");
            }
        }
        throw std::invalid_argument("unterminated ARC JSON string");
    }

    [[nodiscard]] std::uint8_t color() {
        whitespace();
        const char* const begin = text_.data() + position_;
        const char* const end = text_.data() + text_.size();
        unsigned int value{};
        const auto [parsed_end, error] = std::from_chars(begin, end, value);
        if (error != std::errc{} || parsed_end == begin || value > 9U) {
            throw std::invalid_argument("invalid ARC grid color");
        }
        position_ += static_cast<std::size_t>(parsed_end - begin);
        return static_cast<std::uint8_t>(value);
    }

    [[nodiscard]] ArcGrid grid() {
        expect('[');
        ArcGrid result;
        if (consume(']')) {
            throw std::invalid_argument("ARC grid must not be empty");
        }
        std::size_t width = 0U;
        while (true) {
            expect('[');
            std::vector<std::uint8_t> row;
            if (consume(']')) {
                throw std::invalid_argument("ARC grid row must not be empty");
            }
            while (true) {
                row.push_back(color());
                if (row.size() > 30U) {
                    throw std::invalid_argument("ARC grid exceeds 30 columns");
                }
                if (consume(']')) {
                    break;
                }
                expect(',');
            }
            if (width == 0U) {
                width = row.size();
            } else if (row.size() != width) {
                throw std::invalid_argument("ARC grid is not rectangular");
            }
            result.push_back(std::move(row));
            if (result.size() > 30U) {
                throw std::invalid_argument("ARC grid exceeds 30 rows");
            }
            if (consume(']')) {
                break;
            }
            expect(',');
        }
        return result;
    }

private:
    std::string_view text_;
    std::size_t position_{};
};

[[nodiscard]] ArcPair parse_pair(Cursor& cursor) {
    cursor.expect('{');
    ArcPair pair;
    bool input_seen = false;
    bool output_seen = false;
    while (true) {
        const std::string key = cursor.string();
        cursor.expect(':');
        if (key == "input" && !input_seen) {
            pair.input = cursor.grid();
            input_seen = true;
        } else if (key == "output" && !output_seen) {
            pair.output = cursor.grid();
            output_seen = true;
        } else {
            throw std::invalid_argument("invalid or duplicate ARC pair field");
        }
        if (cursor.consume('}')) {
            break;
        }
        cursor.expect(',');
    }
    if (!input_seen || !output_seen) {
        throw std::invalid_argument("ARC pair is missing input or output");
    }
    return pair;
}

[[nodiscard]] std::vector<ArcPair> parse_pairs(Cursor& cursor) {
    cursor.expect('[');
    std::vector<ArcPair> pairs;
    if (cursor.consume(']')) {
        throw std::invalid_argument("ARC pair list must not be empty");
    }
    while (true) {
        pairs.push_back(parse_pair(cursor));
        if (pairs.size() > 10U) {
            throw std::invalid_argument("ARC pair list exceeds supported bound");
        }
        if (cursor.consume(']')) {
            break;
        }
        cursor.expect(',');
    }
    return pairs;
}

[[nodiscard]] std::string grid_text(const ArcGrid& grid) {
    std::ostringstream output;
    output << '[';
    for (std::size_t row = 0U; row < grid.size(); ++row) {
        if (row != 0U) output << ',';
        output << '[';
        for (std::size_t column = 0U; column < grid[row].size(); ++column) {
            if (column != 0U) output << ',';
            output << static_cast<unsigned int>(grid[row][column]);
        }
        output << ']';
    }
    output << ']';
    return output.str();
}

[[nodiscard]] std::vector<std::string> split_tabs(const std::string_view line) {
    std::vector<std::string> fields;
    std::size_t start = 0U;
    while (start <= line.size()) {
        const std::size_t tab = line.find('\t', start);
        const std::size_t end = tab == std::string_view::npos ? line.size() : tab;
        fields.emplace_back(line.substr(start, end - start));
        if (tab == std::string_view::npos) break;
        start = tab + 1U;
    }
    return fields;
}

[[nodiscard]] std::string dataset_aggregate(const std::vector<ArcTask>& tasks) {
    std::string canonical;
    for (const ArcTask& task : tasks) {
        canonical += task.id;
        canonical.push_back('\t');
        canonical += task.source_sha256;
        canonical.push_back('\n');
    }
    return core::sha256_hex(core::sha256(canonical));
}

[[nodiscard]] std::string json_string_field(
    const std::string_view json,
    const std::string_view key,
    const std::size_t start_position = 0U
) {
    const std::string token = "\"" + std::string(key) + "\"";
    const std::size_t key_position = json.find(token, start_position);
    if (key_position == std::string_view::npos) {
        throw std::runtime_error("missing field in raw evaluation result: " + std::string(key));
    }
    const std::size_t colon = json.find(':', key_position + token.size());
    if (colon == std::string_view::npos) {
        throw std::runtime_error("invalid raw evaluation result field");
    }
    Cursor cursor(json.substr(colon + 1U));
    return cursor.string();
}

struct RawPrediction final {
    std::string manifest_sha256;
    std::string checkpoint_sha256;
    std::string model_hash;
    std::string text;
};

[[nodiscard]] RawPrediction load_raw_prediction(
    const std::filesystem::path& path
) {
    const std::string json = read_file(path);
    const std::filesystem::path metadata_path =
        path.parent_path() / (path.stem().string() + ".meta.tsv");
    std::string metadata = read_file(metadata_path);
    if (!metadata.empty() && metadata.back() == '\n') metadata.pop_back();
    if (!metadata.empty() && metadata.back() == '\r') metadata.pop_back();
    const std::vector<std::string> metadata_fields = split_tabs(metadata);
    if ((metadata_fields.size() != 6U && metadata_fields.size() != 7U) ||
        metadata_fields[0U] != path.stem().string() ||
        !core::is_sha256_hex(metadata_fields[1U]) ||
        (metadata_fields[2U] != "-" &&
         !core::is_sha256_hex(metadata_fields[2U])) ||
        !core::is_sha256_hex(metadata_fields[3U]) ||
        !core::is_sha256_hex(metadata_fields[5U]) ||
        core::sha256_hex(core::sha256(json)) != metadata_fields[5U]) {
        throw std::runtime_error("raw ARC prediction metadata mismatch: " + path.string());
    }
    const std::size_t response_position = json.find("\"response\"");
    if (response_position == std::string::npos) {
        throw std::runtime_error("raw prediction is missing response object");
    }
    RawPrediction prediction{
        json_string_field(json, "manifest_sha256"),
        json_string_field(json, "checkpoint_sha256"),
        json_string_field(json, "model_hash"),
        json_string_field(json, "text", response_position),
    };
    if (metadata_fields[3U] != prediction.checkpoint_sha256 ||
        metadata_fields[4U] != prediction.model_hash) {
        throw std::runtime_error("raw ARC prediction identity metadata mismatch");
    }
    return prediction;
}

void require_same_identity(
    const RawPrediction& prediction,
    const std::string_view manifest_sha256,
    std::string& checkpoint_sha256,
    std::string& model_hash
) {
    if (prediction.manifest_sha256 != manifest_sha256) {
        throw std::runtime_error("raw ARC prediction manifest hash mismatch");
    }
    if (checkpoint_sha256.empty()) {
        checkpoint_sha256 = prediction.checkpoint_sha256;
        model_hash = prediction.model_hash;
    } else if (prediction.checkpoint_sha256 != checkpoint_sha256 ||
               prediction.model_hash != model_hash) {
        throw std::runtime_error("raw ARC predictions mix checkpoints or model states");
    }
}

}  // namespace

ArcTask load_arc_task(const std::filesystem::path& path) {
    const std::string json = read_file(path);
    Cursor cursor(json);
    cursor.expect('{');
    ArcTask task;
    task.id = path.stem().string();
    task.source_sha256 = core::sha256_hex(core::sha256_file(path));
    bool train_seen = false;
    bool test_seen = false;
    while (true) {
        const std::string key = cursor.string();
        cursor.expect(':');
        if (key == "train" && !train_seen) {
            task.training = parse_pairs(cursor);
            train_seen = true;
        } else if (key == "test" && !test_seen) {
            task.test = parse_pairs(cursor);
            test_seen = true;
        } else {
            throw std::invalid_argument("invalid or duplicate ARC task field");
        }
        if (cursor.consume('}')) {
            break;
        }
        cursor.expect(',');
    }
    if (!cursor.finished() || !train_seen || !test_seen ||
        task.training.size() < 2U || task.test.size() > 3U) {
        throw std::invalid_argument("invalid ARC task bounds or trailing data");
    }
    return task;
}

std::vector<ArcTask> load_arc_dataset(
    const std::filesystem::path& evaluation_directory,
    const std::size_t expected_tasks
) {
    if (expected_tasks == 0U || !std::filesystem::is_directory(evaluation_directory)) {
        throw std::invalid_argument("invalid ARC evaluation directory or expected count");
    }
    std::vector<std::filesystem::path> paths;
    for (const std::filesystem::directory_entry& entry :
         std::filesystem::directory_iterator(evaluation_directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".json") {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    if (paths.size() != expected_tasks) {
        throw std::runtime_error(
            "ARC evaluation task count mismatch: expected " +
            std::to_string(expected_tasks) + ", found " + std::to_string(paths.size())
        );
    }
    std::vector<ArcTask> tasks;
    tasks.reserve(paths.size());
    for (const std::filesystem::path& path : paths) {
        tasks.push_back(load_arc_task(path));
    }
    return tasks;
}

std::string arc_task_prompt(const ArcTask& task) {
    std::ostringstream output;
    output << "Solve this ARC-AGI-2 task by inferring one transformation from the "
              "training pairs and applying it to every test input. Grid cells are "
              "integers 0 through 9.\n";
    for (std::size_t index = 0U; index < task.training.size(); ++index) {
        output << "TRAIN " << (index + 1U) << " INPUT\n"
               << grid_text(task.training[index].input) << "\nTRAIN "
               << (index + 1U) << " OUTPUT\n"
               << grid_text(task.training[index].output) << '\n';
    }
    for (std::size_t index = 0U; index < task.test.size(); ++index) {
        output << "TEST " << (index + 1U) << " INPUT\n"
               << grid_text(task.test[index].input) << '\n';
    }
    output << "Return only a JSON array containing one output grid per TEST input "
              "in order. For one 2x2 output, the required outer shape is "
              "[[[0,1],[1,0]]]. Do not include prose, labels, or code fences.\n";
    return output.str();
}

std::optional<std::vector<ArcGrid>> parse_arc_prediction(
    const std::string_view response,
    const std::size_t expected_grids
) {
    if (expected_grids == 0U) {
        return std::nullopt;
    }
    try {
        Cursor cursor(response);
        cursor.expect('[');
        std::vector<ArcGrid> grids;
        if (cursor.consume(']')) {
            return std::nullopt;
        }
        while (true) {
            grids.push_back(cursor.grid());
            if (grids.size() > expected_grids) {
                return std::nullopt;
            }
            if (cursor.consume(']')) {
                break;
            }
            cursor.expect(',');
        }
        if (!cursor.finished() || grids.size() != expected_grids) {
            return std::nullopt;
        }
        return grids;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

void write_verified_artifact(
    const std::filesystem::path& path,
    const std::string_view content
) {
    if (std::filesystem::exists(path)) {
        if (!std::filesystem::is_regular_file(path) || read_file(path) != content) {
            throw std::runtime_error("existing ARC artifact differs: " + path.string());
        }
        return;
    }
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    const std::filesystem::path temporary = path.string() + ".partial";
    if (std::filesystem::exists(temporary)) {
        throw std::runtime_error("stale ARC partial artifact: " + temporary.string());
    }
    try {
        {
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            if (!output) throw std::runtime_error("unable to create ARC artifact");
            output.write(content.data(), static_cast<std::streamsize>(content.size()));
            output.flush();
            if (!output) throw std::runtime_error("failed while writing ARC artifact");
        }
        std::filesystem::rename(temporary, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

void write_arc_artifact_manifest(
    const std::filesystem::path& root,
    const std::filesystem::path& output_path
) {
    std::error_code error;
    const std::filesystem::path canonical_root =
        std::filesystem::weakly_canonical(root, error);
    if (error || !std::filesystem::is_directory(canonical_root, error) || error) {
        throw std::runtime_error("invalid ARC artifact root");
    }
    const std::filesystem::path canonical_output = output_path.is_absolute()
        ? output_path.lexically_normal()
        : std::filesystem::absolute(output_path).lexically_normal();
    std::vector<std::filesystem::path> files;
    for (const std::filesystem::directory_entry& entry :
         std::filesystem::recursive_directory_iterator(canonical_root)) {
        if (!entry.is_regular_file()) continue;
        const std::filesystem::path absolute =
            std::filesystem::absolute(entry.path()).lexically_normal();
        if (absolute == canonical_output || entry.path().extension() == ".partial") {
            continue;
        }
        files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end(), [&canonical_root](
        const std::filesystem::path& left,
        const std::filesystem::path& right
    ) {
        return left.lexically_relative(canonical_root).generic_string() <
            right.lexically_relative(canonical_root).generic_string();
    });
    std::string manifest = "# relative_path\tbytes\tsha256\n";
    for (const std::filesystem::path& file : files) {
        manifest += file.lexically_relative(canonical_root).generic_string();
        manifest.push_back('\t');
        manifest += std::to_string(std::filesystem::file_size(file));
        manifest.push_back('\t');
        manifest += core::sha256_hex(core::sha256_file(file));
        manifest.push_back('\n');
    }
    write_verified_artifact(output_path, manifest);
}

ArcPreparationReport prepare_arc_evaluation(
    const std::filesystem::path& evaluation_directory,
    const std::filesystem::path& output_directory,
    const std::size_t expected_tasks
) {
    const std::vector<ArcTask> tasks = load_arc_dataset(
        evaluation_directory, expected_tasks
    );
    std::string request_manifest =
        "# id\tprompt_path\tprompt_sha256\timage_path\timage_sha256\n";
    std::string dataset_manifest =
        "# task_id\tsource_file\tsource_sha256\ttraining_pairs\ttest_inputs\tprompt_sha256\n";
    std::size_t test_inputs = 0U;
    for (const ArcTask& task : tasks) {
        const std::string prompt = arc_task_prompt(task);
        const std::string prompt_sha256 = core::sha256_hex(core::sha256(prompt));
        write_verified_artifact(
            output_directory / "prompts" / (task.id + ".txt"), prompt
        );
        request_manifest += task.id + "\tprompts/" + task.id + ".txt\t" +
            prompt_sha256 + "\t-\t-\n";
        dataset_manifest += task.id + '\t' + task.id + ".json\t" +
            task.source_sha256 + '\t' + std::to_string(task.training.size()) +
            '\t' + std::to_string(task.test.size()) + '\t' + prompt_sha256 + '\n';
        test_inputs += task.test.size();
    }
    const std::filesystem::path request_path = output_directory / "requests.tsv";
    write_verified_artifact(request_path, request_manifest);
    write_verified_artifact(output_directory / "dataset_manifest.tsv", dataset_manifest);
    ArcPreparationReport report;
    report.tasks = tasks.size();
    report.test_inputs = test_inputs;
    report.dataset_aggregate_sha256 = dataset_aggregate(tasks);
    report.request_manifest_sha256 = core::sha256_hex(core::sha256(request_manifest));
    report.request_manifest_path = request_path;
    write_verified_artifact(
        output_directory / "preparation.json", arc_preparation_report_json(report)
    );
    return report;
}

ArcScoreReport score_arc_evaluation(
    const std::filesystem::path& evaluation_directory,
    const std::filesystem::path& request_manifest,
    const std::filesystem::path& trial_one_directory,
    const std::filesystem::path& trial_two_directory,
    const std::size_t expected_tasks
) {
    std::error_code path_error;
    const std::filesystem::path canonical_one =
        std::filesystem::weakly_canonical(trial_one_directory, path_error);
    if (path_error) throw std::runtime_error("invalid first ARC trial directory");
    const std::filesystem::path canonical_two =
        std::filesystem::weakly_canonical(trial_two_directory, path_error);
    if (path_error || canonical_one == canonical_two) {
        throw std::runtime_error("ARC trials must use distinct directories");
    }
    const std::string identity_one = read_file(canonical_one / "run_identity.tsv");
    const std::string identity_two = read_file(canonical_two / "run_identity.tsv");
    if (identity_one == identity_two) {
        throw std::runtime_error("ARC trials must use distinct generation identities");
    }
    const std::vector<ArcTask> tasks = load_arc_dataset(
        evaluation_directory, expected_tasks
    );
    ArcScoreReport report;
    report.tasks = tasks.size();
    report.dataset_aggregate_sha256 = dataset_aggregate(tasks);
    report.request_manifest_sha256 = core::sha256_hex(core::sha256_file(request_manifest));
    report.task_scores.reserve(tasks.size());
    for (const ArcTask& task : tasks) {
        const RawPrediction first = load_raw_prediction(
            trial_one_directory / (task.id + ".json")
        );
        const RawPrediction second = load_raw_prediction(
            trial_two_directory / (task.id + ".json")
        );
        require_same_identity(
            first, report.request_manifest_sha256,
            report.checkpoint_sha256, report.model_hash
        );
        require_same_identity(
            second, report.request_manifest_sha256,
            report.checkpoint_sha256, report.model_hash
        );
        const auto first_grids = parse_arc_prediction(first.text, task.test.size());
        const auto second_grids = parse_arc_prediction(second.text, task.test.size());
        ArcTaskScore score;
        score.id = task.id;
        score.trial_one_valid = first_grids.has_value();
        score.trial_two_valid = second_grids.has_value();
        score.test_inputs = task.test.size();
        report.valid_trial_one_tasks += score.trial_one_valid ? 1U : 0U;
        report.valid_trial_two_tasks += score.trial_two_valid ? 1U : 0U;
        for (std::size_t index = 0U; index < task.test.size(); ++index) {
            const bool first_correct = first_grids.has_value() &&
                (*first_grids)[index] == task.test[index].output;
            const bool second_correct = second_grids.has_value() &&
                (*second_grids)[index] == task.test[index].output;
            score.correct_test_inputs += first_correct || second_correct ? 1U : 0U;
        }
        score.solved = score.correct_test_inputs == score.test_inputs;
        report.solved_tasks += score.solved ? 1U : 0U;
        report.test_inputs += score.test_inputs;
        report.correct_test_inputs += score.correct_test_inputs;
        report.task_scores.push_back(std::move(score));
    }
    report.task_accuracy = report.tasks == 0U ? 0.0 :
        static_cast<double>(report.solved_tasks) / static_cast<double>(report.tasks);
    report.test_input_accuracy = report.test_inputs == 0U ? 0.0 :
        static_cast<double>(report.correct_test_inputs) /
        static_cast<double>(report.test_inputs);
    report.target_passed = report.task_accuracy >= report.frontier_target;
    return report;
}

std::string arc_preparation_report_json(const ArcPreparationReport& report) {
    std::ostringstream output;
    output << "{\n"
           << "  \"schema\": \"rlf-arc-agi2-preparation-v1\",\n"
           << "  \"source\": \"https://github.com/arcprize/ARC-AGI-2\",\n"
           << "  \"split\": \"public_evaluation\",\n"
           << "  \"license\": \"Apache-2.0\",\n"
           << "  \"tasks\": " << report.tasks << ",\n"
           << "  \"test_inputs\": " << report.test_inputs << ",\n"
           << "  \"dataset_aggregate_sha256\": \""
           << report.dataset_aggregate_sha256 << "\",\n"
           << "  \"request_manifest_sha256\": \""
           << report.request_manifest_sha256 << "\",\n"
           << "  \"request_manifest_path\": \""
           << escape_json(report.request_manifest_path.generic_string()) << "\"\n"
           << "}\n";
    return output.str();
}

std::string arc_score_report_json(const ArcScoreReport& report) {
    std::ostringstream output;
    output << std::boolalpha << std::fixed << std::setprecision(9);
    output << "{\n"
           << "  \"schema\": \"rlf-arc-agi2-score-v1\",\n"
           << "  \"source\": \"https://github.com/arcprize/ARC-AGI-2\",\n"
           << "  \"split\": \"public_evaluation\",\n"
           << "  \"official_private_verified\": false,\n"
           << "  \"independent_evaluation\": false,\n"
           << "  \"contamination_audited\": false,\n"
           << "  \"contamination_note\": \"The frozen bootstrap checkpoint predates "
              "the campaign provenance ledger; no compatible broad-corpus audit exists.\",\n"
           << "  \"dataset_aggregate_sha256\": \""
           << report.dataset_aggregate_sha256 << "\",\n"
           << "  \"request_manifest_sha256\": \""
           << report.request_manifest_sha256 << "\",\n"
           << "  \"checkpoint_sha256\": \"" << report.checkpoint_sha256 << "\",\n"
           << "  \"model_hash\": \"" << report.model_hash << "\",\n"
           << "  \"tasks\": " << report.tasks << ",\n"
           << "  \"solved_tasks\": " << report.solved_tasks << ",\n"
           << "  \"test_inputs\": " << report.test_inputs << ",\n"
           << "  \"correct_test_inputs\": " << report.correct_test_inputs << ",\n"
           << "  \"valid_trial_one_tasks\": " << report.valid_trial_one_tasks << ",\n"
           << "  \"valid_trial_two_tasks\": " << report.valid_trial_two_tasks << ",\n"
           << "  \"task_accuracy\": " << report.task_accuracy << ",\n"
           << "  \"test_input_accuracy\": " << report.test_input_accuracy << ",\n"
           << "  \"frontier_target\": " << report.frontier_target << ",\n"
           << "  \"target_passed\": " << report.target_passed << ",\n"
           << "  \"tasks_detail\": [\n";
    for (std::size_t index = 0U; index < report.task_scores.size(); ++index) {
        const ArcTaskScore& score = report.task_scores[index];
        output << "    {\"id\": \"" << escape_json(score.id)
               << "\", \"trial_one_valid\": " << score.trial_one_valid
               << ", \"trial_two_valid\": " << score.trial_two_valid
               << ", \"test_inputs\": " << score.test_inputs
               << ", \"correct_test_inputs\": " << score.correct_test_inputs
               << ", \"solved\": " << score.solved << '}';
        if (index + 1U != report.task_scores.size()) output << ',';
        output << '\n';
    }
    output << "  ]\n}\n";
    return output.str();
}

}  // namespace rlf::benchmarks
