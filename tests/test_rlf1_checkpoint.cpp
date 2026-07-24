#include "test_framework.hpp"

#include "rlf/core/latent_routing.hpp"
#include "rlf/storage/rlf1_checkpoint.hpp"

#include <filesystem>
#include <fstream>
#include <utility>
#include <vector>

namespace {

[[nodiscard]] rlf::core::TransformationOperator make_shift(
    const std::size_t dimension
) {
    std::vector<float> shift(dimension, 0.0F);
    for (std::size_t index = 0U; index < dimension; ++index) {
        shift[index] = 0.05F * static_cast<float>(index + 1U);
    }
    return rlf::core::TransformationOperator(
        dimension,
        {rlf::core::OperatorPrimitive::shift(
            rlf::core::PhaseVector(std::move(shift))
        )}
    );
}

}  // namespace

RLF_TEST_CASE("RLF-1 checkpoint round trips latent router state") {
    constexpr std::size_t dimension = 10U;
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "rlf1_checkpoint_test.rlf";
    rlf::core::LatentRouter router(
        {
            .dimension = dimension,
            .maximum_cycles = 6U,
            .successor_familiarity_weight = 0.42,
            .route_repetition_penalty = 0.77,
            .abstention_resonance_threshold = 0.12,
            .search_beam_width = 3U,
            .search_lookahead_depth = 2U,
        },
        42ULL
    );
    const std::uint64_t action = router.register_operator(
        "shift",
        make_shift(dimension)
    );
    const rlf::core::PhaseVector start =
        rlf::core::PhaseVector::zeros(dimension);
    const rlf::core::PhaseVector goal =
        router.operator_by_id(action).transformation.apply(start);
    router.reinforce_route(
        start,
        goal,
        std::vector<std::uint64_t>{action},
        1.0
    );
    rlf::storage::save_rlf1_checkpoint(path, router);
    const rlf::storage::Rlf1CheckpointSummary summary =
        rlf::storage::inspect_rlf1_checkpoint(path);
    RLF_CHECK(summary.format_version == 3U);
    RLF_CHECK(summary.dimension == dimension);
    RLF_CHECK(summary.operator_count == 1U);
    RLF_CHECK(summary.routing_mode_count == 1U);
    RLF_CHECK(summary.route_memory_count == 1U);

    rlf::core::LatentRouter restored =
        rlf::storage::load_rlf1_checkpoint(path);
    RLF_CHECK_NEAR(restored.config().successor_familiarity_weight, 0.42, 1.0e-12);
    RLF_CHECK_NEAR(restored.config().route_repetition_penalty, 0.77, 1.0e-12);
    RLF_CHECK_NEAR(restored.config().abstention_resonance_threshold, 0.12, 1.0e-12);
    RLF_CHECK(restored.config().search_beam_width == 3U);
    RLF_CHECK(restored.config().search_lookahead_depth == 2U);
    const auto result = restored.execute(start, goal);
    RLF_CHECK(result.success);
    RLF_CHECK(result.route.size() == 1U);
    std::filesystem::remove(path);
}

RLF_TEST_CASE("RLF-1 checkpoint rejects corruption") {
    constexpr std::size_t dimension = 6U;
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "rlf1_checkpoint_corrupt.rlf";
    rlf::core::LatentRouter router({.dimension = dimension}, 91ULL);
    static_cast<void>(router.register_operator("shift", make_shift(dimension)));
    rlf::storage::save_rlf1_checkpoint(path, router);

    std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
    RLF_CHECK(static_cast<bool>(file));
    file.seekg(-1, std::ios::end);
    char byte{};
    file.read(&byte, 1);
    byte = static_cast<char>(byte ^ 0x5A);
    file.seekp(-1, std::ios::end);
    file.write(&byte, 1);
    file.close();

    RLF_CHECK_THROWS_AS(
        rlf::storage::load_rlf1_checkpoint(path),
        std::runtime_error
    );
    std::filesystem::remove(path);
}
