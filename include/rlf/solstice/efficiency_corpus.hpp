#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <map>
#include <string>

namespace rlf::solstice {

struct EfficiencyCorpusConfig final {
    std::filesystem::path output_directory;
    std::size_t training_records{};
    std::uint64_t seed{0x524C'4645'4646'3031ULL};
    std::size_t evaluation_records_per_category{16U};
};

struct EfficiencyCorpusReport final {
    std::filesystem::path output_directory;
    std::filesystem::path ledger_path;
    std::string ledger_sha256;
    std::size_t training_records{};
    std::size_t evaluation_records{};
    std::size_t image_files{};
    std::uint64_t shard_bytes{};
    std::uint64_t media_bytes{};
    std::map<std::string, std::size_t, std::less<>> category_counts;
    bool audit_passed{};
};

[[nodiscard]] EfficiencyCorpusReport generate_efficiency_corpus(
    const EfficiencyCorpusConfig& config
);

void write_efficiency_corpus_report_json(
    std::ostream& output,
    const EfficiencyCorpusReport& report
);

struct EfficiencyEvaluationReport final {
    std::filesystem::path output_directory;
    std::string source_ledger_sha256;
    std::string request_manifest_sha256;
    std::string expected_answers_sha256;
    std::size_t examples{};
    std::size_t image_examples{};
};

[[nodiscard]] EfficiencyEvaluationReport materialize_efficiency_evaluation(
    const std::filesystem::path& ledger_path,
    const std::filesystem::path& output_directory
);

void write_efficiency_evaluation_report_json(
    std::ostream& output,
    const EfficiencyEvaluationReport& report
);

struct EfficiencyCategoryScore final {
    std::size_t examples{};
    std::size_t correct{};
};

struct EfficiencyEvaluationScore final {
    std::string expected_sha256;
    std::size_t examples{};
    std::size_t correct{};
    double accuracy{};
    std::map<std::string, EfficiencyCategoryScore, std::less<>> categories;
};

[[nodiscard]] EfficiencyEvaluationScore score_efficiency_evaluation(
    const std::filesystem::path& expected_path,
    const std::filesystem::path& results_directory,
    const std::filesystem::path& output_directory
);

void write_efficiency_evaluation_score_json(
    std::ostream& output,
    const EfficiencyEvaluationScore& score
);

}  // namespace rlf::solstice
