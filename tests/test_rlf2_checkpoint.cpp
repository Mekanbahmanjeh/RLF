#include "test_framework.hpp"

#include "rlf/experiments/rlf2_predictive_reasoning.hpp"
#include "rlf/storage/rlf2_checkpoint.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

RLF_TEST_CASE("RLF-2 checkpoint round trips and rejects corruption") {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "rlf2_test_checkpoint.rlf";
    const std::filesystem::path corrupted =
        std::filesystem::temp_directory_path() / "rlf2_test_checkpoint_bad.rlf";
    const rlf::experiments::Rlf2Config config{
        .seed = 0x524C4632434B5054ULL,
        .dimension = 8U,
        .training_episodes = 12U,
        .development_episodes = 4U,
        .evaluation_episodes = 4U,
        .training_min_route_length = 1U,
        .training_max_route_length = 2U,
        .evaluation_min_route_length = 3U,
        .evaluation_max_route_length = 3U,
        .maximum_cycles = 8U,
        .operator_count = 6U,
        .state_noise_radians = 0.01,
        .goal_similarity_threshold = 0.9995,
    };
    const auto training =
        rlf::experiments::train_rlf2_checkpoint(config, path);
    const auto summary = rlf::storage::inspect_rlf2_checkpoint(path);
    RLF_CHECK(summary.format_version == 4U);
    RLF_CHECK(summary.dimension == config.dimension);
    RLF_CHECK(summary.operator_count == config.operator_count);
    RLF_CHECK(summary.skill_count >= summary.operator_count);
    RLF_CHECK(training.prototypes > 0U);

    auto fabric = rlf::storage::load_rlf2_checkpoint(path);
    RLF_CHECK(fabric.prototypes().size() == summary.prototype_count);
    RLF_CHECK(fabric.skills().size() == summary.skill_count);

    std::ifstream input(path, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>()
    );
    input.close();
    RLF_CHECK(bytes.size() > 40U);
    bytes.back() = static_cast<char>(bytes.back() ^ 0x1);
    std::ofstream output(corrupted, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    output.close();
    RLF_CHECK_THROWS_AS(rlf::storage::load_rlf2_checkpoint(corrupted), std::runtime_error);
    std::filesystem::remove(path);
    std::filesystem::remove(corrupted);
}
