#include "rlf/solstice/frontier_gate.hpp"
#include "rlf/solstice/checkpoint.hpp"
#include "rlf/solstice/profile.hpp"

#include "rlf/core/sha256.hpp"

#include <charconv>
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

struct ModelIdentity final {
    std::string checkpoint_sha256;
    std::string training_manifest_sha256;
};

[[nodiscard]] rlf::solstice::SolsticeProfile read_readiness_profile(
    const std::filesystem::path& path
) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("unable to read manifest readiness report");
    }
    std::optional<std::string> selected;
    std::string line;
    while (std::getline(input, line)) {
        const std::size_t key = line.find("\"profile\"");
        if (key == std::string::npos) continue;
        if (selected.has_value()) {
            throw std::runtime_error("readiness report contains duplicate profile fields");
        }
        const std::size_t colon = line.find(':', key + 9U);
        const std::size_t begin = colon == std::string::npos
            ? std::string::npos : line.find('"', colon + 1U);
        const std::size_t end = begin == std::string::npos
            ? std::string::npos : line.find('"', begin + 1U);
        if (end == std::string::npos || end == begin + 1U) {
            throw std::runtime_error("readiness report profile is not a JSON string");
        }
        const std::string value = line.substr(begin + 1U, end - begin - 1U);
        if (value.find('\\') != std::string::npos) {
            throw std::runtime_error("readiness report profile may not contain escapes");
        }
        selected = value;
    }
    if (!input.eof() || !selected.has_value()) {
        throw std::runtime_error("readiness report does not bind a training profile");
    }
    const rlf::solstice::SolsticeProfile profile =
        rlf::solstice::parse_profile(*selected);
    if (rlf::solstice::to_string(profile) != *selected) {
        throw std::runtime_error("readiness report profile is not canonical");
    }
    return profile;
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

[[nodiscard]] double parse_double(const std::string_view value) {
    double result{};
    const auto [end, error] = std::from_chars(
        value.data(), value.data() + value.size(), result
    );
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::invalid_argument("invalid score: " + std::string(value));
    }
    return result;
}

[[nodiscard]] std::size_t parse_size(const std::string_view value) {
    std::size_t result{};
    const auto [end, error] = std::from_chars(
        value.data(), value.data() + value.size(), result
    );
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::invalid_argument("invalid example count: " + std::string(value));
    }
    return result;
}

[[nodiscard]] bool parse_bool(const std::string_view value) {
    if (value == "true" || value == "1" || value == "yes") {
        return true;
    }
    if (value == "false" || value == "0" || value == "no") {
        return false;
    }
    throw std::invalid_argument("invalid boolean: " + std::string(value));
}

[[nodiscard]] std::string lowercase_ascii(std::string value) {
    std::ranges::transform(value, value.begin(), [](const unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

[[nodiscard]] ModelIdentity verify_training_manifest(
    const std::filesystem::path& path
) {
    const std::filesystem::path sidecar = path.string() + ".sha256";
    if (!std::filesystem::is_regular_file(path) ||
        !std::filesystem::is_regular_file(sidecar)) {
        throw std::runtime_error("training manifest or self-hash sidecar is missing");
    }
    const std::string manifest_sha256 =
        rlf::core::sha256_hex(rlf::core::sha256_file(path));
    std::ifstream hash_input(sidecar);
    std::string declared_manifest_sha256;
    hash_input >> declared_manifest_sha256;
    if (!hash_input || !rlf::core::is_sha256_hex(declared_manifest_sha256) ||
        lowercase_ascii(declared_manifest_sha256) != manifest_sha256) {
        throw std::runtime_error("training manifest self-hash is missing or mismatched");
    }

    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("unable to read training manifest");
    }
    std::set<std::string> kinds;
    std::string checkpoint_sha256;
    std::filesystem::path checkpoint_path;
    std::filesystem::path readiness_path;
    std::string line;
    std::size_t records = 0U;
    std::size_t expected_records = 0U;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (line.front() == '#') {
            std::size_t declared_records = 0U;
            if (line == "# rlf-training-artifact-manifest-v1") {
                declared_records = 9U;
            } else if (line == "# rlf-training-artifact-manifest-v2") {
                declared_records = 10U;
            }
            if (declared_records > 0U) {
                if (expected_records > 0U) {
                    throw std::runtime_error("duplicate training manifest schema");
                }
                expected_records = declared_records;
            }
            continue;
        }
        const std::vector<std::string> fields = split_tabs(line);
        if (fields.size() != 4U || !kinds.insert(fields[0U]).second ||
            !rlf::core::is_sha256_hex(fields[1U])) {
            throw std::runtime_error("invalid or duplicate training manifest record");
        }
        const std::filesystem::path artifact = std::filesystem::path(fields[3U]).is_absolute()
            ? std::filesystem::path(fields[3U])
            : path.parent_path() / fields[3U];
        std::error_code error;
        const bool regular = std::filesystem::is_regular_file(artifact, error);
        if (error || !regular) {
            throw std::runtime_error("training manifest artifact is missing or mismatched");
        }
        const std::uintmax_t bytes = std::filesystem::file_size(artifact, error);
        if (error || bytes != parse_size(fields[2U]) ||
            rlf::core::sha256_hex(rlf::core::sha256_file(artifact)) !=
                lowercase_ascii(fields[1U])) {
            throw std::runtime_error("training manifest artifact is missing or mismatched");
        }
        if (fields[0U] == "checkpoint") {
            checkpoint_sha256 = lowercase_ascii(fields[1U]);
            checkpoint_path = artifact;
        } else if (fields[0U] == "readiness_report") {
            readiness_path = artifact;
        }
        ++records;
    }
    if (!input.eof() || expected_records == 0U || records != expected_records ||
        checkpoint_sha256.empty() || readiness_path.empty()) {
        throw std::runtime_error(
            "training manifest schema and complete artifact count do not match"
        );
    }
    std::set<std::string> expected_kinds{
        "checkpoint", "ledger", "source_manifest", "data_audit",
        "resource_summary", "vram_trace", "environment",
        "checkpoint_inspection", "readiness_report"
    };
    if (expected_records == 10U) expected_kinds.insert("telemetry");
    if (kinds != expected_kinds) {
        throw std::runtime_error(
            "training manifest does not contain the exact schema artifact kinds"
        );
    }
    const rlf::solstice::SolsticeProfile profile =
        read_readiness_profile(readiness_path);
    const rlf::solstice::SolsticeCheckpointSummary checkpoint =
        rlf::solstice::inspect_solstice_checkpoint_for_profile(
            checkpoint_path, profile
        );
    if (checkpoint.format_version != 6U) {
        throw std::runtime_error("training manifest checkpoint is not format 6");
    }
    return {checkpoint_sha256, manifest_sha256};
}

[[nodiscard]] std::vector<rlf::solstice::FrontierBenchmarkEvidence> read_evidence(
    const std::filesystem::path& path,
    const std::optional<ModelIdentity>& model_identity
) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("unable to open benchmark evidence: " + path.string());
    }
    std::vector<rlf::solstice::FrontierBenchmarkEvidence> evidence;
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
        if (fields.size() != 17U) {
            throw std::runtime_error(
                "expected 17 TSV fields at " + path.string() + ":" +
                std::to_string(line_number)
            );
        }
        const auto resolve = [&path](const std::string& value) {
            std::filesystem::path resolved(value);
            return resolved.is_absolute() ? resolved : path.parent_path() / resolved;
        };
        const auto verify_artifact = [](const std::filesystem::path& artifact) {
            std::error_code error;
            const bool present = std::filesystem::is_regular_file(artifact, error) &&
                !error && std::filesystem::file_size(artifact, error) > 0U && !error;
            return present;
        };
        const std::filesystem::path artifact_path = resolve(fields[11U]);
        const std::filesystem::path contamination_path = resolve(fields[13U]);
        const bool artifact_present = !fields[11U].empty() &&
            verify_artifact(artifact_path);
        const bool contamination_present = !fields[13U].empty() &&
            verify_artifact(contamination_path);
        if (!rlf::core::is_sha256_hex(fields[12U]) ||
            !rlf::core::is_sha256_hex(fields[14U]) ||
            !rlf::core::is_sha256_hex(fields[15U]) ||
            !rlf::core::is_sha256_hex(fields[16U])) {
            throw std::runtime_error(
                "invalid artifact SHA-256 at " + path.string() + ":" +
                std::to_string(line_number)
            );
        }
        const bool artifact_verified = artifact_present &&
            rlf::core::sha256_hex(rlf::core::sha256_file(artifact_path)) ==
                lowercase_ascii(fields[12U]);
        const bool contamination_verified = contamination_present &&
            rlf::core::sha256_hex(rlf::core::sha256_file(contamination_path)) ==
                lowercase_ascii(fields[14U]);
        const bool model_identity_verified = model_identity.has_value() &&
            lowercase_ascii(fields[15U]) == model_identity->checkpoint_sha256 &&
            lowercase_ascii(fields[16U]) ==
                model_identity->training_manifest_sha256;
        evidence.push_back(rlf::solstice::FrontierBenchmarkEvidence{
            fields[0U],
            fields[1U],
            parse_double(fields[2U]),
            parse_size(fields[3U]),
            parse_bool(fields[4U]),
            parse_bool(fields[5U]),
            parse_bool(fields[6U]),
            artifact_present,
            fields[11U],
            fields[7U],
            fields[8U],
            fields[9U],
            fields[10U],
            fields[12U],
            artifact_verified,
            fields[13U],
            fields[14U],
            contamination_present,
            contamination_verified,
            fields[15U],
            fields[16U],
            model_identity_verified,
        });
    }
    if (!input.eof()) {
        throw std::runtime_error("failed while reading benchmark evidence");
    }
    return evidence;
}

void write_text(const std::filesystem::path& path, const std::string_view text) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("unable to write report: " + path.string());
    }
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!output) {
        throw std::runtime_error("failed while writing report: " + path.string());
    }
}

int run(const int argc, char** argv) {
    std::filesystem::path evidence_path;
    std::filesystem::path output_path;
    std::filesystem::path training_manifest_path;
    bool print_targets = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--evidence") {
            if (index + 1 >= argc) {
                throw std::invalid_argument("--evidence requires a path");
            }
            evidence_path = argv[++index];
        } else if (argument == "--output") {
            if (index + 1 >= argc) {
                throw std::invalid_argument("--output requires a path");
            }
            output_path = argv[++index];
        } else if (argument == "--print-targets") {
            print_targets = true;
        } else if (argument == "--training-manifest") {
            if (index + 1 >= argc) {
                throw std::invalid_argument("--training-manifest requires a path");
            }
            training_manifest_path = argv[++index];
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: rlf_general_benchmark --evidence scores.tsv "
                   "[--training-manifest FILE] [--output report.json]\n"
                << "       rlf_general_benchmark --print-targets\n\n"
                << "Evidence TSV fields:\n"
                << "benchmark capability score examples external_dataset "
                   "contamination_audited independent_harness dataset_version "
                   "harness harness_version evaluator evidence_path evidence_sha256 "
                   "contamination_report_path contamination_report_sha256 "
                   "model_checkpoint_sha256 training_manifest_sha256\n";
            return 0;
        } else {
            throw std::invalid_argument("unknown option: " + std::string(argument));
        }
    }

    const std::vector<rlf::solstice::FrontierBenchmarkTarget> targets =
        rlf::solstice::leading_system_targets_2026();
    if (print_targets) {
        for (const auto& target : targets) {
            std::cout << target.benchmark << '\t' << target.capability << '\t'
                      << target.minimum_score << '\t' << target.minimum_examples
                      << '\t' << target.reference_date << '\t'
                      << target.reference_source << '\n';
        }
        return 0;
    }
    if (evidence_path.empty()) {
        throw std::invalid_argument("--evidence is required");
    }
    std::optional<ModelIdentity> model_identity;
    if (!training_manifest_path.empty()) {
        model_identity = verify_training_manifest(training_manifest_path);
    }
    const auto evidence = read_evidence(evidence_path, model_identity);
    const auto report = rlf::solstice::evaluate_frontier_gate(evidence, targets);
    const std::string json = rlf::solstice::frontier_gate_json(report);
    if (output_path.empty()) {
        std::cout << json;
    } else {
        write_text(output_path, json);
        std::cout << "Wrote " << output_path.string() << '\n';
    }
    return report.broad_frontier_parity_proven ? 0 : 3;
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "rlf_general_benchmark: " << error.what() << '\n';
        return 2;
    }
}
