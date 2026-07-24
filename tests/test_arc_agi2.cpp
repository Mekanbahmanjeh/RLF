#include "test_framework.hpp"

#include "rlf/benchmarks/arc_agi2.hpp"
#include "rlf/core/sha256.hpp"

#include <filesystem>
#include <fstream>
#include <string>

namespace {

void write_text(const std::filesystem::path& path, const std::string& content) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << content;
}

void write_raw_result(
    const std::filesystem::path& directory,
    const std::string& id,
    const std::string& manifest_sha256,
    const std::string& response,
    const std::string& identity
) {
    std::filesystem::create_directories(directory);
    write_text(directory / "run_identity.tsv", identity);
    const std::string result =
        "{\"manifest_sha256\":\"" + manifest_sha256 +
        "\",\"checkpoint_sha256\":\"" + std::string(64U, 'a') +
        "\",\"model_hash\":\"0123456789abcdef\",\"response\":{\"text\":\"" +
        response + "\"}}\n";
    const std::filesystem::path result_path = directory / (id + ".json");
    write_text(result_path, result);
    const std::string result_hash = rlf::core::sha256_hex(rlf::core::sha256(result));
    write_text(
        directory / (id + ".meta.tsv"),
        id + '\t' + std::string(64U, 'b') + "\t-\t" + std::string(64U, 'a') +
        "\t0123456789abcdef\t" + result_hash + "\t1\n"
    );
}

}  // namespace

RLF_TEST_CASE("ARC adapter excludes test answers and applies exact two-trial scoring") {
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "rlf_arc_agi2_adapter_test";
    std::filesystem::remove_all(root);
    const std::filesystem::path dataset = root / "dataset";
    std::filesystem::create_directories(dataset);
    write_text(
        dataset / "task1.json",
        "{\"train\":[{\"input\":[[0]],\"output\":[[1]]},"
        "{\"input\":[[3]],\"output\":[[4]]}],"
        "\"test\":[{\"input\":[[8]],\"output\":[[9]]}]}\n"
    );
    const auto task = rlf::benchmarks::load_arc_task(dataset / "task1.json");
    const std::string prompt = rlf::benchmarks::arc_task_prompt(task);
    RLF_CHECK(prompt.find("TEST 1 OUTPUT") == std::string::npos);
    RLF_CHECK(prompt.find("[[9]]") == std::string::npos);
    RLF_CHECK(!rlf::benchmarks::parse_arc_prediction("prose [[[9]]]", 1U).has_value());
    RLF_CHECK(rlf::benchmarks::parse_arc_prediction("[[[9]]]", 1U).has_value());

    const std::filesystem::path prepared = root / "prepared";
    const auto preparation = rlf::benchmarks::prepare_arc_evaluation(
        dataset, prepared, 1U
    );
    write_raw_result(
        root / "trial1", "task1", preparation.request_manifest_sha256,
        "not a grid", "trial-one\n"
    );
    write_raw_result(
        root / "trial2", "task1", preparation.request_manifest_sha256,
        "[[[9]]]", "trial-two\n"
    );
    const auto score = rlf::benchmarks::score_arc_evaluation(
        dataset, preparation.request_manifest_path,
        root / "trial1", root / "trial2", 1U
    );
    RLF_CHECK(score.tasks == 1U);
    RLF_CHECK(score.solved_tasks == 1U);
    RLF_CHECK_NEAR(score.task_accuracy, 1.0, 1.0e-12);
    RLF_CHECK(score.target_passed);
    std::filesystem::remove_all(root);
}

RLF_TEST_CASE("ARC adapter rejects non-rectangular and out-of-range grids") {
    RLF_CHECK(!rlf::benchmarks::parse_arc_prediction("[[[1],[1,2]]]", 1U).has_value());
    RLF_CHECK(!rlf::benchmarks::parse_arc_prediction("[[[10]]]", 1U).has_value());
    RLF_CHECK(!rlf::benchmarks::parse_arc_prediction("[]", 1U).has_value());
}
