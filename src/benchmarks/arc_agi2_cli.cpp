#include "rlf/benchmarks/arc_agi2.hpp"

#include <charconv>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

[[nodiscard]] std::size_t parse_size(const std::string_view text) {
    std::size_t value{};
    const auto [end, error] = std::from_chars(
        text.data(), text.data() + text.size(), value
    );
    if (error != std::errc{} || end != text.data() + text.size() || value == 0U) {
        throw std::invalid_argument("invalid positive task count");
    }
    return value;
}

int run(const int argc, char** argv) {
    if (argc < 2) {
        throw std::invalid_argument("prepare or score command is required");
    }
    const std::string command = argv[1];
    std::filesystem::path dataset;
    std::filesystem::path output;
    std::filesystem::path requests;
    std::filesystem::path trial_one;
    std::filesystem::path trial_two;
    std::size_t expected_tasks = 120U;
    for (int index = 2; index < argc; ++index) {
        const std::string_view argument = argv[index];
        const auto value = [&](const std::string_view option) -> std::string_view {
            if (index + 1 >= argc) {
                throw std::invalid_argument(std::string(option) + " requires a value");
            }
            return argv[++index];
        };
        if (argument == "--dataset") dataset = value(argument);
        else if (argument == "--output") output = value(argument);
        else if (argument == "--requests") requests = value(argument);
        else if (argument == "--trial-one") trial_one = value(argument);
        else if (argument == "--trial-two") trial_two = value(argument);
        else if (argument == "--expected-tasks") expected_tasks = parse_size(value(argument));
        else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage:\n"
                << "  rlf_arc_agi2 prepare --dataset DIR --output DIR [--expected-tasks 120]\n"
                << "  rlf_arc_agi2 score --dataset DIR --requests FILE --trial-one DIR "
                   "--trial-two DIR --output FILE [--expected-tasks 120]\n";
            return 0;
        } else {
            throw std::invalid_argument("unknown option: " + std::string(argument));
        }
    }
    if (dataset.empty() || output.empty()) {
        throw std::invalid_argument("--dataset and --output are required");
    }
    if (command == "prepare") {
        const auto report = rlf::benchmarks::prepare_arc_evaluation(
            dataset, output, expected_tasks
        );
        std::cout << rlf::benchmarks::arc_preparation_report_json(report);
        return 0;
    }
    if (command == "score") {
        if (requests.empty() || trial_one.empty() || trial_two.empty()) {
            throw std::invalid_argument(
                "score requires --requests, --trial-one, and --trial-two"
            );
        }
        const auto report = rlf::benchmarks::score_arc_evaluation(
            dataset, requests, trial_one, trial_two, expected_tasks
        );
        const std::string json = rlf::benchmarks::arc_score_report_json(report);
        rlf::benchmarks::write_verified_artifact(output, json);
        rlf::benchmarks::write_arc_artifact_manifest(
            output.parent_path(), output.parent_path() / "artifact_manifest.tsv"
        );
        std::cout << json;
        return report.target_passed ? 0 : 3;
    }
    throw std::invalid_argument("unknown ARC command: " + command);
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "rlf_arc_agi2: " << error.what() << '\n';
        return 2;
    }
}
