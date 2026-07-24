#include "rlf/solstice/efficiency_corpus.hpp"

#include "rlf/core/sha256.hpp"
#include "rlf/solstice/data_pipeline.hpp"

#include <array>
#include <charconv>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rlf::solstice {
namespace {

struct LedgerRow final {
    std::string id;
    std::string kind;
    std::string split;
    std::string modality;
    std::string domain;
    std::string format;
    std::filesystem::path path;
    std::string evaluation_family;
};

[[nodiscard]] std::string file_sha(const std::filesystem::path& path) {
    return core::sha256_hex(core::sha256_file(path));
}

void require_stream(const std::ofstream& stream, const char* description) {
    if (!stream) throw std::runtime_error(std::string("failed writing ") + description);
}

void write_ppm(const std::filesystem::path& path, const std::uint64_t index) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("unable to create generated image");
    output << "P6\n8 8\n255\n";
    for (std::uint64_t pixel = 0U; pixel < 64U; ++pixel) {
        std::uint64_t mixed = (index + 1U) * 0x9E37'79B9'7F4A'7C15ULL ^
            (pixel + 1U) * 0xD6E8'FEB8'6659'FD93ULL;
        mixed = (mixed ^ (mixed >> 30U)) * 0xBF58'476D'1CE4'E5B9ULL;
        mixed = (mixed ^ (mixed >> 27U)) * 0x94D0'49BB'1331'11EBULL;
        mixed ^= mixed >> 31U;
        output.put(static_cast<char>((mixed >> 8U) & 0xFFU));
        output.put(static_cast<char>((mixed >> 24U) & 0xFFU));
        output.put(static_cast<char>((mixed >> 40U) & 0xFFU));
    }
    require_stream(output, "generated image");
}

void write_ledger(
    const std::filesystem::path& root,
    const std::vector<LedgerRow>& rows
) {
    std::ofstream output(root / "ledger.tsv", std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("unable to create generated ledger");
    output << "# shard_id\tkind\tsplit\tmodality\tlanguage\tdomain\tformat\tpath\t"
              "source_uri\tlicense\tcreated_utc\tsha256\tpreprocessing_version\t"
              "teacher\tevaluation_family\tapproved\n";
    for (const LedgerRow& row : rows) {
        output << row.id << '\t' << row.kind << '\t' << row.split << '\t'
               << row.modality << "\ten\t" << row.domain << '\t' << row.format
               << '\t' << row.path.generic_string()
               << "\tgenerated:rlf-efficiency-fixture-v1\tCC0-1.0\t2026-07-21\t"
               << file_sha(root / row.path)
               << "\tefficiency-fixture-v1\tnone\t" << row.evaluation_family
               << "\ttrue\n";
    }
    require_stream(output, "generated ledger");
}

[[nodiscard]] std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> fields;
    std::size_t start = 0U;
    while (start <= line.size()) {
        const std::size_t tab = line.find('\t', start);
        fields.emplace_back(line.substr(
            start, tab == std::string::npos ? line.size() - start : tab - start
        ));
        if (tab == std::string::npos) break;
        start = tab + 1U;
    }
    return fields;
}

void write_prompt(
    const std::filesystem::path& path,
    const std::string& content
) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("unable to create evaluation prompt");
    output << content << '\n';
    require_stream(output, "evaluation prompt");
}

[[nodiscard]] std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("unable to read evaluation result");
    std::string content(
        static_cast<std::size_t>(std::filesystem::file_size(path)), '\0'
    );
    if (!content.empty()) {
        input.read(content.data(), static_cast<std::streamsize>(content.size()));
    }
    if (!input) throw std::runtime_error("failed reading evaluation result");
    return content;
}

[[nodiscard]] std::string parse_json_string(
    const std::string_view input,
    std::size_t position
) {
    if (position >= input.size() || input[position] != '"') {
        throw std::runtime_error("expected JSON string");
    }
    ++position;
    std::string output;
    while (position < input.size()) {
        const char character = input[position++];
        if (character == '"') return output;
        if (character != '\\') {
            output.push_back(character);
            continue;
        }
        if (position >= input.size()) throw std::runtime_error("truncated JSON escape");
        const char escaped = input[position++];
        switch (escaped) {
            case '"': output.push_back('"'); break;
            case '\\': output.push_back('\\'); break;
            case '/': output.push_back('/'); break;
            case 'b': output.push_back('\b'); break;
            case 'f': output.push_back('\f'); break;
            case 'n': output.push_back('\n'); break;
            case 'r': output.push_back('\r'); break;
            case 't': output.push_back('\t'); break;
            case 'u': {
                if (position + 4U > input.size()) {
                    throw std::runtime_error("truncated JSON unicode escape");
                }
                unsigned int value = 0U;
                const auto result = std::from_chars(
                    input.data() + position, input.data() + position + 4U,
                    value, 16
                );
                if (result.ec != std::errc{} || result.ptr != input.data() + position + 4U ||
                    value > 0x7FU) {
                    throw std::runtime_error("unsupported JSON unicode escape in result");
                }
                output.push_back(static_cast<char>(value));
                position += 4U;
                break;
            }
            default: throw std::runtime_error("unsupported JSON escape in result");
        }
    }
    throw std::runtime_error("unterminated JSON string");
}

[[nodiscard]] std::string extract_result_text(const std::string& json) {
    const std::size_t response = json.find("\"response\": {");
    if (response == std::string::npos) throw std::runtime_error("result has no response object");
    const std::size_t key = json.find("\"text\": ", response);
    if (key == std::string::npos) throw std::runtime_error("result has no response text");
    return parse_json_string(json, key + std::string_view("\"text\": ").size());
}

[[nodiscard]] std::string normalize_answer(const std::string_view input) {
    std::string output;
    bool pending_space = false;
    for (const char raw : input) {
        const unsigned char character = static_cast<unsigned char>(raw);
        if (std::isspace(character) != 0) {
            pending_space = !output.empty();
            continue;
        }
        if (pending_space) {
            output.push_back(' ');
            pending_space = false;
        }
        output.push_back(character < 128U
            ? static_cast<char>(std::tolower(character)) : raw);
    }
    return output;
}

[[nodiscard]] std::string escape_tsv(const std::string_view input) {
    std::string output;
    for (const char character : input) {
        if (character == '\t') output += "\\t";
        else if (character == '\n') output += "\\n";
        else if (character == '\r') output += "\\r";
        else if (character == '\\') output += "\\\\";
        else output.push_back(character);
    }
    return output;
}

}  // namespace

EfficiencyCorpusReport generate_efficiency_corpus(
    const EfficiencyCorpusConfig& config
) {
    if (config.output_directory.empty()) {
        throw std::invalid_argument("efficiency corpus output directory is required");
    }
    if (config.training_records < 1'000U) {
        throw std::invalid_argument("efficiency corpus requires at least 1,000 training records");
    }
    if (config.evaluation_records_per_category == 0U) {
        throw std::invalid_argument("evaluation records per category must be positive");
    }
    if (std::filesystem::exists(config.output_directory)) {
        throw std::invalid_argument("efficiency corpus output already exists");
    }
    std::filesystem::path temporary = config.output_directory;
    temporary += ".tmp";
    if (std::filesystem::exists(temporary)) {
        throw std::invalid_argument("efficiency corpus temporary output already exists");
    }
    std::filesystem::create_directories(temporary / "shards");
    std::filesystem::create_directories(temporary / "media");

    EfficiencyCorpusReport report;
    report.output_directory = config.output_directory;
    report.training_records = config.training_records;
    const std::size_t caption_records = config.training_records / 100U;
    const std::size_t vqa_records = config.training_records / 100U;
    const std::size_t nonvisual = config.training_records - caption_records - vqa_records;
    const std::array<std::string, 6U> text_categories{
        "text_instruction", "mathematics", "coding", "tool_trajectory",
        "preference_correction", "continual_update",
    };
    std::array<std::size_t, 6U> text_counts{};
    for (std::size_t index = 0U; index < text_counts.size(); ++index) {
        text_counts[index] = nonvisual / text_counts.size() +
            (index < nonvisual % text_counts.size() ? 1U : 0U);
        report.category_counts[text_categories[index]] = text_counts[index];
    }
    report.category_counts["image_caption"] = caption_records;
    report.category_counts["visual_question_answering"] = vqa_records;

    std::vector<LedgerRow> rows;
    const auto instruction_shard = [&](const std::string& category,
                                       const std::size_t count,
                                       const std::string& domain) {
        const std::filesystem::path relative = "shards/train_" + category + ".tsv";
        std::ofstream output(temporary / relative, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("unable to create instruction shard");
        for (std::size_t index = 0U; index < count; ++index) {
            output << category << '_' << index << '\t' << domain << '\t'
                   << "training instance " << index << " seed " << config.seed
                   << " requires a reusable " << category << " transformation\t"
                   << "derive variables for instance " << index << " then verify the result\t"
                   << "verified " << category << " result "
                   << (index * 17U + static_cast<std::size_t>(config.seed % 997U))
                   << "\t1\n";
        }
        require_stream(output, "instruction shard");
        rows.push_back({"train_" + category, "instruction", "train", "text",
                        domain, "tsv", relative, "none"});
    };
    instruction_shard("text_instruction", text_counts[0U], "instruction");
    instruction_shard("mathematics", text_counts[1U], "mathematics");
    instruction_shard("coding", text_counts[2U], "software");

    {
        const std::filesystem::path relative = "shards/train_tools.tsv";
        std::ofstream output(temporary / relative, std::ios::binary | std::ios::trunc);
        for (std::size_t index = 0U; index < text_counts[3U]; ++index) {
            output << "tool trajectory request " << index << " argument value "
                   << (index * 31U + 7U) << '\t'
                   << (index % 3U == 0U ? "calculator" :
                       (index % 3U == 1U ? "read_text_file" : "list_directory")) << '\n';
        }
        require_stream(output, "tool shard");
        rows.push_back({"train_tool_trajectory", "tools", "train", "text",
                        "tools", "tsv", relative, "none"});
    }
    {
        const std::filesystem::path relative = "shards/train_preferences.tsv";
        std::ofstream output(temporary / relative, std::ios::binary | std::ios::trunc);
        for (std::size_t index = 0U; index < text_counts[4U]; ++index) {
            output << "preference request " << index << " with constraint " << (index % 19U)
                   << "\tgrounded concise correction " << index
                   << "\tunsupported verbose response " << index
                   << "\tprefer verified evidence\t1\n";
        }
        require_stream(output, "preference shard");
        rows.push_back({"train_preference_correction", "preference", "train", "text",
                        "preference", "tsv", relative, "none"});
    }
    {
        const std::size_t count = text_counts[5U];
        const std::array<std::string, 4U> waves{"a", "b", "c", "d"};
        std::size_t emitted = 0U;
        for (std::size_t wave = 0U; wave < waves.size(); ++wave) {
            const std::size_t wave_count = count / waves.size() +
                (wave < count % waves.size() ? 1U : 0U);
            const std::filesystem::path relative =
                "shards/train_continual_" + waves[wave] + ".tsv";
            std::ofstream output(temporary / relative, std::ios::binary | std::ios::trunc);
            for (std::size_t index = 0U; index < wave_count; ++index) {
                output << "continual_entity_" << wave << '_' << index
                       << "\thas_revision\trevision_" << wave << '_' << (index * 13U + 5U)
                       << "\t1\n";
            }
            require_stream(output, "continual shard");
            rows.push_back({"train_continual_" + waves[wave], "facts", "train", "text",
                            "continual", "tsv", relative, "none"});
            emitted += wave_count;
        }
        if (emitted != count) throw std::logic_error("continual count mismatch");
    }

    std::uint64_t image_index = config.seed;
    const auto vision_shard = [&](const std::string& category,
                                  const std::size_t count,
                                  const bool question_answer) {
        const std::filesystem::path relative = "shards/train_" + category + ".tsv";
        std::ofstream output(temporary / relative, std::ios::binary | std::ios::trunc);
        for (std::size_t index = 0U; index < count; ++index, ++image_index) {
            const std::string filename = category + "_" + std::to_string(index) + ".ppm";
            const std::filesystem::path image_relative = std::filesystem::path("media") / filename;
            write_ppm(temporary / image_relative, image_index);
            output << "../" << image_relative.generic_string() << '\t'
                   << file_sha(temporary / image_relative) << '\t';
            if (question_answer) {
                output << "Question: identify visual code " << index
                       << ". Answer: visual-code-" << (image_index % 100'003U) << '\n';
            } else {
                output << "synthetic color-grid caption " << index
                       << " signature " << (image_index % 100'003U) << '\n';
            }
            ++report.image_files;
        }
        require_stream(output, "vision shard");
        rows.push_back({"train_" + category, "vision", "train", "image-text",
                        category, "vision_tsv", relative, "none"});
    };
    vision_shard("image_caption", caption_records, false);
    vision_shard("visual_question_answering", vqa_records, true);

    const std::array<std::string, 14U> evaluation_categories{
        "language_instruction", "factual_grounding", "mathematics",
        "compositional_reasoning", "coding", "image_captioning",
        "visual_question_answering", "ocr_document", "tool_selection",
        "tool_argument_correctness", "unnecessary_tool_avoidance",
        "continual_retention", "unseen_template_transfer", "distribution_shift",
    };
    const std::filesystem::path eval_text_relative = "shards/evaluation_text.tsv";
    const std::filesystem::path eval_vision_relative = "shards/evaluation_vision.tsv";
    std::ofstream eval_text(temporary / eval_text_relative, std::ios::binary | std::ios::trunc);
    std::ofstream eval_vision(temporary / eval_vision_relative, std::ios::binary | std::ios::trunc);
    for (const std::string& category : evaluation_categories) {
        const bool visual = category == "image_captioning" ||
            category == "visual_question_answering" || category == "ocr_document";
        for (std::size_t index = 0U; index < config.evaluation_records_per_category;
             ++index) {
            if (visual) {
                const std::string filename = "eval_" + category + "_" +
                    std::to_string(index) + ".ppm";
                const std::filesystem::path image_relative =
                    std::filesystem::path("media") / filename;
                write_ppm(temporary / image_relative, image_index++);
                eval_vision << "../" << image_relative.generic_string() << '\t'
                            << file_sha(temporary / image_relative) << '\t'
                            << "sealed heldout " << category << " probe " << index
                            << " answer eval-" << category << '-' << (index * 23U + 11U) << '\n';
                ++report.image_files;
            } else {
                eval_text << "heldout_" << category << '_' << index << '\t' << category << '\t'
                          << "sealed evaluation scenario " << category << " nonce "
                          << (index * 29U + 101U) << "\t"
                          << "apply unseen constraints and verify independently\t"
                          << "evaluation answer " << category << ' ' << (index * 23U + 11U)
                          << "\t1\n";
            }
            ++report.evaluation_records;
        }
    }
    require_stream(eval_text, "evaluation text shard");
    require_stream(eval_vision, "evaluation vision shard");
    eval_text.close();
    eval_vision.close();
    require_stream(eval_text, "evaluation text shard");
    require_stream(eval_vision, "evaluation vision shard");
    rows.push_back({"evaluation_text", "instruction", "evaluation", "text",
                    "mixed", "tsv", eval_text_relative, "efficiency_fixed_suite_v1"});
    rows.push_back({"evaluation_vision", "vision", "evaluation", "image-text",
                    "mixed", "vision_tsv", eval_vision_relative,
                    "efficiency_fixed_suite_v1"});

    write_ledger(temporary, rows);
    DataAuditOptions audit_options;
    audit_options.maximum_records = config.training_records + report.evaluation_records + 1U;
    audit_options.require_media_sha256 = true;
    const DataLedger ledger = load_data_ledger(temporary / "ledger.tsv");
    const DataAuditReport audit = audit_data_ledger(ledger, audit_options);
    if (!audit.valid) {
        throw std::runtime_error("generated efficiency corpus failed its contamination audit");
    }
    report.ledger_sha256 = ledger.sha256;
    report.shard_bytes = audit.shard_bytes;
    report.media_bytes = audit.referenced_media_bytes;
    report.audit_passed = true;
    std::filesystem::rename(temporary, config.output_directory);
    report.ledger_path = config.output_directory / "ledger.tsv";
    return report;
}

void write_efficiency_corpus_report_json(
    std::ostream& output,
    const EfficiencyCorpusReport& report
) {
    output << "{\n"
           << "  \"schema\": \"rlf-efficiency-corpus-v1\",\n"
           << "  \"output_directory\": " << std::quoted(report.output_directory.string()) << ",\n"
           << "  \"ledger_path\": " << std::quoted(report.ledger_path.string()) << ",\n"
           << "  \"ledger_sha256\": \"" << report.ledger_sha256 << "\",\n"
           << "  \"training_records\": " << report.training_records << ",\n"
           << "  \"evaluation_records\": " << report.evaluation_records << ",\n"
           << "  \"image_files\": " << report.image_files << ",\n"
           << "  \"shard_bytes\": " << report.shard_bytes << ",\n"
           << "  \"media_bytes\": " << report.media_bytes << ",\n"
           << "  \"audit_passed\": " << (report.audit_passed ? "true" : "false") << ",\n"
           << "  \"internal_performance_fixture\": true,\n"
           << "  \"external_quality_evidence_eligible\": false,\n"
           << "  \"category_counts\": {\n";
    std::size_t emitted = 0U;
    for (const auto& [name, count] : report.category_counts) {
        output << "    \"" << name << "\": " << count
               << (++emitted == report.category_counts.size() ? "\n" : ",\n");
    }
    output << "  },\n"
           << "  \"claim_boundary\": \"This deterministic generated corpus validates pipeline mechanics and scaling only; it is not external general-quality evidence.\"\n"
           << "}\n";
}

EfficiencyEvaluationReport materialize_efficiency_evaluation(
    const std::filesystem::path& ledger_path,
    const std::filesystem::path& output_directory
) {
    if (output_directory.empty() || std::filesystem::exists(output_directory)) {
        throw std::invalid_argument("evaluation output must be a new directory");
    }
    const DataLedger ledger = load_data_ledger(ledger_path);
    DataAuditOptions audit_options;
    audit_options.require_media_sha256 = true;
    const DataAuditReport audit = audit_data_ledger(ledger, audit_options);
    if (!audit.valid) throw std::runtime_error("source ledger failed evaluation audit");
    std::filesystem::path temporary = output_directory;
    temporary += ".tmp";
    if (std::filesystem::exists(temporary)) {
        throw std::invalid_argument("evaluation temporary output already exists");
    }
    std::filesystem::create_directories(temporary / "prompts");
    std::filesystem::create_directories(temporary / "media");
    std::ofstream requests(temporary / "requests.tsv", std::ios::binary | std::ios::trunc);
    std::ofstream expected(temporary / "expected.tsv", std::ios::binary | std::ios::trunc);
    if (!requests || !expected) throw std::runtime_error("unable to create evaluation manifests");
    requests << "# id\tprompt_path\tprompt_sha256\timage_path\timage_sha256\n";
    expected << "# id\tcategory\tscoring\texpected\n";
    EfficiencyEvaluationReport report;
    report.output_directory = output_directory;
    report.source_ledger_sha256 = ledger.sha256;
    const std::filesystem::path ledger_root = ledger.source_path.parent_path();
    for (const DataShard& shard : ledger.shards) {
        if (shard.split != DataShardSplit::evaluation) continue;
        const std::filesystem::path shard_path = shard.path.is_absolute()
            ? shard.path : ledger_root / shard.path;
        std::ifstream input(shard_path);
        if (!input) throw std::runtime_error("unable to read evaluation shard");
        std::string line;
        while (std::getline(input, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || line.front() == '#') continue;
            const std::vector<std::string> fields = split_tabs(line);
            std::string id;
            std::string category;
            std::string prompt;
            std::string answer;
            std::filesystem::path copied_image;
            std::string copied_image_sha{"-"};
            if (shard.kind == DataShardKind::instruction) {
                if (fields.size() < 5U) throw std::runtime_error("invalid evaluation instruction row");
                id = fields[0U];
                category = fields[1U];
                prompt = fields[2U];
                answer = fields[4U];
            } else if (shard.kind == DataShardKind::vision) {
                if (fields.size() != 3U) throw std::runtime_error("invalid evaluation vision row");
                constexpr std::string_view prefix = "sealed heldout ";
                const std::size_t probe = fields[2U].find(" probe ");
                const std::size_t answer_marker = fields[2U].find(" answer ");
                if (!fields[2U].starts_with(prefix) || probe == std::string::npos ||
                    answer_marker == std::string::npos || answer_marker <= probe + 7U) {
                    throw std::runtime_error("invalid generated visual evaluation caption");
                }
                category = fields[2U].substr(prefix.size(), probe - prefix.size());
                const std::string probe_id = fields[2U].substr(
                    probe + 7U, answer_marker - (probe + 7U)
                );
                id = "heldout_" + category + '_' + probe_id;
                answer = fields[2U].substr(answer_marker + 8U);
                prompt = "Inspect the image for sealed " + category + " probe " +
                    probe_id + ". Return only the answer token.";
                std::filesystem::path source_image(fields[0U]);
                if (source_image.is_relative()) source_image = shard_path.parent_path() / source_image;
                copied_image = std::filesystem::path("media") / (id + ".ppm");
                std::filesystem::copy_file(source_image, temporary / copied_image);
                copied_image_sha = file_sha(temporary / copied_image);
                if (copied_image_sha != fields[1U]) {
                    throw std::runtime_error("evaluation image hash changed while materializing");
                }
                ++report.image_examples;
            } else {
                throw std::runtime_error("unsupported evaluation shard kind");
            }
            const std::filesystem::path prompt_relative =
                std::filesystem::path("prompts") / (id + ".txt");
            write_prompt(temporary / prompt_relative, prompt);
            requests << id << '\t' << prompt_relative.generic_string() << '\t'
                     << file_sha(temporary / prompt_relative) << '\t';
            if (copied_image.empty()) requests << "-\t-\n";
            else requests << copied_image.generic_string() << '\t' << copied_image_sha << '\n';
            expected << id << '\t' << category << "\texact_normalized\t" << answer << '\n';
            ++report.examples;
        }
        if (!input.eof()) throw std::runtime_error("failed reading evaluation shard");
    }
    require_stream(requests, "evaluation request manifest");
    require_stream(expected, "evaluation expected answers");
    requests.close();
    expected.close();
    require_stream(requests, "evaluation request manifest");
    require_stream(expected, "evaluation expected answers");
    if (report.examples == 0U) throw std::runtime_error("ledger has no evaluation records");
    report.request_manifest_sha256 = file_sha(temporary / "requests.tsv");
    report.expected_answers_sha256 = file_sha(temporary / "expected.tsv");
    std::filesystem::rename(temporary, output_directory);
    return report;
}

void write_efficiency_evaluation_report_json(
    std::ostream& output,
    const EfficiencyEvaluationReport& report
) {
    output << "{\n"
           << "  \"schema\": \"rlf-efficiency-evaluation-suite-v1\",\n"
           << "  \"output_directory\": " << std::quoted(report.output_directory.string()) << ",\n"
           << "  \"source_ledger_sha256\": \"" << report.source_ledger_sha256 << "\",\n"
           << "  \"request_manifest_sha256\": \"" << report.request_manifest_sha256 << "\",\n"
           << "  \"expected_answers_sha256\": \"" << report.expected_answers_sha256 << "\",\n"
           << "  \"examples\": " << report.examples << ",\n"
           << "  \"image_examples\": " << report.image_examples << ",\n"
           << "  \"fixed_categories\": 14,\n"
           << "  \"external_quality_evidence_eligible\": false,\n"
           << "  \"claim_boundary\": \"This is a sealed controlled fixture for matched-pipeline comparisons, not an external frontier benchmark.\"\n"
           << "}\n";
}

EfficiencyEvaluationScore score_efficiency_evaluation(
    const std::filesystem::path& expected_path,
    const std::filesystem::path& results_directory,
    const std::filesystem::path& output_directory
) {
    if (!std::filesystem::is_regular_file(expected_path) ||
        !std::filesystem::is_directory(results_directory) ||
        output_directory.empty() || std::filesystem::exists(output_directory)) {
        throw std::invalid_argument("invalid or existing evaluation scoring path");
    }
    std::filesystem::path temporary = output_directory;
    temporary += ".tmp";
    if (std::filesystem::exists(temporary)) {
        throw std::invalid_argument("evaluation scoring temporary output exists");
    }
    std::filesystem::create_directories(temporary);
    std::ofstream raw(temporary / "raw_scores.tsv", std::ios::binary | std::ios::trunc);
    if (!raw) throw std::runtime_error("unable to create raw evaluation scores");
    raw << "id\tcategory\tcorrect\texpected\tactual\tresult_sha256\n";
    EfficiencyEvaluationScore score;
    score.expected_sha256 = file_sha(expected_path);
    std::ifstream expected_input(expected_path);
    std::string line;
    while (std::getline(expected_input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.front() == '#') continue;
        const std::vector<std::string> fields = split_tabs(line);
        if (fields.size() != 4U || fields[2U] != "exact_normalized") {
            throw std::runtime_error("invalid expected-answer row");
        }
        const std::filesystem::path result_path = results_directory / (fields[0U] + ".json");
        const std::filesystem::path metadata_path = results_directory / (fields[0U] + ".meta.tsv");
        if (!std::filesystem::is_regular_file(result_path) ||
            !std::filesystem::is_regular_file(metadata_path)) {
            throw std::runtime_error("missing evaluation result for " + fields[0U]);
        }
        std::ifstream metadata(metadata_path);
        std::string metadata_line;
        if (!std::getline(metadata, metadata_line)) {
            throw std::runtime_error("empty evaluation metadata");
        }
        const std::vector<std::string> metadata_fields = split_tabs(metadata_line);
        const std::string result_sha = file_sha(result_path);
        if (metadata_fields.size() != 7U || metadata_fields[0U] != fields[0U] ||
            metadata_fields[5U] != result_sha) {
            throw std::runtime_error("evaluation result metadata identity mismatch");
        }
        const std::string actual = extract_result_text(read_file(result_path));
        const bool correct = normalize_answer(actual) == normalize_answer(fields[3U]);
        ++score.examples;
        if (correct) ++score.correct;
        EfficiencyCategoryScore& category = score.categories[fields[1U]];
        ++category.examples;
        if (correct) ++category.correct;
        raw << fields[0U] << '\t' << fields[1U] << '\t'
            << (correct ? "true" : "false") << '\t' << fields[3U] << '\t'
            << escape_tsv(actual) << '\t' << result_sha << '\n';
    }
    if (!expected_input.eof() || score.examples == 0U) {
        throw std::runtime_error("failed or empty expected-answer manifest");
    }
    require_stream(raw, "raw evaluation scores");
    raw.close();
    score.accuracy = static_cast<double>(score.correct) /
        static_cast<double>(score.examples);
    {
        std::ofstream summary(temporary / "summary.json", std::ios::binary | std::ios::trunc);
        if (!summary) throw std::runtime_error("unable to create evaluation score summary");
        write_efficiency_evaluation_score_json(summary, score);
        require_stream(summary, "evaluation score summary");
    }
    std::filesystem::rename(temporary, output_directory);
    return score;
}

void write_efficiency_evaluation_score_json(
    std::ostream& output,
    const EfficiencyEvaluationScore& score
) {
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"rlf-efficiency-evaluation-score-v1\",\n"
           << "  \"expected_sha256\": \"" << score.expected_sha256 << "\",\n"
           << "  \"examples\": " << score.examples << ",\n"
           << "  \"correct\": " << score.correct << ",\n"
           << "  \"accuracy\": " << score.accuracy << ",\n"
           << "  \"categories\": {\n";
    std::size_t emitted = 0U;
    for (const auto& [name, category] : score.categories) {
        const double accuracy = category.examples > 0U
            ? static_cast<double>(category.correct) / static_cast<double>(category.examples)
            : 0.0;
        output << "    \"" << name << "\": {\"examples\": " << category.examples
               << ", \"correct\": " << category.correct << ", \"accuracy\": "
               << accuracy << "}" << (++emitted == score.categories.size() ? "\n" : ",\n");
    }
    output << "  },\n"
           << "  \"external_quality_evidence_eligible\": false,\n"
           << "  \"claim_boundary\": \"Exact-normalized controlled-fixture scores support matched internal comparisons only.\"\n"
           << "}\n";
}

}  // namespace rlf::solstice
