#include "test_framework.hpp"

#include "rlf/core/deterministic_rng.hpp"
#include "rlf/core/sparse_world_model.hpp"
#include "rlf/storage/rlf3_checkpoint.hpp"

#include <filesystem>
#include <fstream>
#include <vector>

RLF_TEST_CASE("RLF-3 checkpoint round trip and corruption rejection") {
    constexpr std::size_t dimension = 8U;
    rlf::core::SparseWorldModelConfig config;
    config.dimension = dimension;
    config.hash_dimensions = 3U;
    config.minimum_transition_support = 1U;
    rlf::core::SparseWorldModel model(config, 0x33445566ULL);
    const auto action = model.register_action("advance", 1.0);
    rlf::core::DeterministicRng rng(0x99887766ULL);
    const auto first = rlf::core::PhaseVector::random(dimension, rng);
    const auto second = rlf::core::PhaseVector::random(dimension, rng);
    const auto memory = rlf::core::PhaseVector::random(dimension, rng);
    model.observe_transition({
        {first, memory}, action, {second, memory}, 1.0, true
    });
    model.observe_successful_route({
        {{first, memory}, {second, memory}}, {action}, true
    });

    const auto directory = std::filesystem::temp_directory_path();
    const auto path = directory / "rlf3_checkpoint_test.rlf";
    const auto corrupt = directory / "rlf3_checkpoint_corrupt.rlf";
    rlf::storage::save_rlf3_checkpoint(path, model);
    const auto summary = rlf::storage::inspect_rlf3_checkpoint(path);
    RLF_CHECK(summary.format_version == 5U);
    RLF_CHECK(summary.transition_count == 1U);
    RLF_CHECK(summary.subgoal_count == 1U);
    const auto restored = rlf::storage::load_rlf3_checkpoint(path);
    RLF_CHECK(restored.predict({first, memory}, action).has_value());
    RLF_CHECK(restored.subgoals().size() == model.subgoals().size());

    std::ifstream input(path, std::ios::binary);
    std::vector<char> bytes{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()
    };
    input.close();
    RLF_CHECK(!bytes.empty());
    bytes.back() = static_cast<char>(bytes.back() ^ 0x01);
    std::ofstream output(corrupt, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    output.close();
    RLF_CHECK_THROWS_AS(
        rlf::storage::load_rlf3_checkpoint(corrupt),
        std::runtime_error
    );
    std::filesystem::remove(path);
    std::filesystem::remove(corrupt);
}
