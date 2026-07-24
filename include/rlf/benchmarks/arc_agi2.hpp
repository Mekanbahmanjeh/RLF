#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rlf::benchmarks {

using ArcGrid = std::vector<std::vector<std::uint8_t>>;

struct ArcPair final {
    ArcGrid input;
    ArcGrid output;
};

struct ArcTask final {
    std::string id;
    std::string source_sha256;
    std::vector<ArcPair> training;
    std::vector<ArcPair> test;
};

struct ArcPreparationReport final {
    std::size_t tasks{};
    std::size_t test_inputs{};
    std::string dataset_aggregate_sha256;
    std::string request_manifest_sha256;
    std::filesystem::path request_manifest_path;
};

struct ArcTaskScore final {
    std::string id;
    bool trial_one_valid{};
    bool trial_two_valid{};
    std::size_t test_inputs{};
    std::size_t correct_test_inputs{};
    bool solved{};
};

struct ArcScoreReport final {
    std::size_t tasks{};
    std::size_t solved_tasks{};
    std::size_t test_inputs{};
    std::size_t correct_test_inputs{};
    std::size_t valid_trial_one_tasks{};
    std::size_t valid_trial_two_tasks{};
    double task_accuracy{};
    double test_input_accuracy{};
    double frontier_target{0.925};
    bool target_passed{};
    std::string dataset_aggregate_sha256;
    std::string request_manifest_sha256;
    std::string checkpoint_sha256;
    std::string model_hash;
    std::vector<ArcTaskScore> task_scores;
};

[[nodiscard]] ArcTask load_arc_task(const std::filesystem::path& path);
[[nodiscard]] std::vector<ArcTask> load_arc_dataset(
    const std::filesystem::path& evaluation_directory,
    std::size_t expected_tasks
);
[[nodiscard]] std::string arc_task_prompt(const ArcTask& task);
[[nodiscard]] std::optional<std::vector<ArcGrid>> parse_arc_prediction(
    std::string_view response,
    std::size_t expected_grids
);
[[nodiscard]] ArcPreparationReport prepare_arc_evaluation(
    const std::filesystem::path& evaluation_directory,
    const std::filesystem::path& output_directory,
    std::size_t expected_tasks
);
[[nodiscard]] ArcScoreReport score_arc_evaluation(
    const std::filesystem::path& evaluation_directory,
    const std::filesystem::path& request_manifest,
    const std::filesystem::path& trial_one_directory,
    const std::filesystem::path& trial_two_directory,
    std::size_t expected_tasks
);
[[nodiscard]] std::string arc_preparation_report_json(
    const ArcPreparationReport& report
);
[[nodiscard]] std::string arc_score_report_json(const ArcScoreReport& report);
void write_verified_artifact(
    const std::filesystem::path& path,
    std::string_view content
);
void write_arc_artifact_manifest(
    const std::filesystem::path& root,
    const std::filesystem::path& output_path
);

}  // namespace rlf::benchmarks
