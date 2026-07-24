#include "test_framework.hpp"

#include "rlf/agent/agent_fabric.hpp"
#include "rlf/storage/rlf6_checkpoint.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

RLF_TEST_CASE("RLF-6 checkpoint version 8 round trip inspect corruption and truncation") {
    rlf::agent::AgentFabric fabric({}, 0x524C4636434B50ULL);
    rlf::agent::Goal goal;
    goal.specification = "complete";
    goal.completion_conditions = {{"goal", "done", false}};
    goal.provenance = "test";
    static_cast<void>(fabric.add_goal(goal));
    rlf::agent::EvidenceRecord evidence;
    evidence.fact = {"state", "ready", false};
    evidence.kind = rlf::agent::EvidenceKind::verified_fact;
    evidence.confidence = 1.0;
    evidence.source_reliability = 1.0;
    evidence.verified = true;
    evidence.provenance = "test";
    fabric.ingest_evidence(evidence);

    const auto path = std::filesystem::temp_directory_path() / "rlf6_checkpoint_test.rlf";
    const auto corrupt = std::filesystem::temp_directory_path() / "rlf6_checkpoint_corrupt.rlf";
    const auto truncated = std::filesystem::temp_directory_path() / "rlf6_checkpoint_truncated.rlf";
    rlf::storage::save_rlf6_checkpoint(path, fabric);
    const auto restored = rlf::storage::load_rlf6_checkpoint(path);
    RLF_CHECK(restored.deterministic_hash() == fabric.deterministic_hash());
    const auto summary = rlf::storage::inspect_rlf6_checkpoint(path);
    RLF_CHECK(summary.format_version == 8U);
    RLF_CHECK(summary.goal_count == 1U);

    std::ifstream input(path, std::ios::binary);
    std::vector<char> bytes{
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()
    };
    input.close();
    RLF_CHECK(bytes.size() > 32U);
    auto corrupt_bytes = bytes;
    corrupt_bytes.back() = static_cast<char>(corrupt_bytes.back() ^ 0x01);
    std::ofstream corrupt_output(corrupt, std::ios::binary | std::ios::trunc);
    corrupt_output.write(corrupt_bytes.data(), static_cast<std::streamsize>(corrupt_bytes.size()));
    corrupt_output.close();
    RLF_CHECK_THROWS_AS(rlf::storage::load_rlf6_checkpoint(corrupt), std::runtime_error);

    bytes.resize(bytes.size() - 7U);
    std::ofstream truncated_output(truncated, std::ios::binary | std::ios::trunc);
    truncated_output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    truncated_output.close();
    RLF_CHECK_THROWS_AS(rlf::storage::load_rlf6_checkpoint(truncated), std::runtime_error);

    rlf::storage::Rlf6CheckpointLoadOptions tight;
    tight.maximum_records = 0U;
    RLF_CHECK_THROWS_AS(rlf::storage::load_rlf6_checkpoint(path, tight), std::runtime_error);
    std::filesystem::remove(path);
    std::filesystem::remove(corrupt);
    std::filesystem::remove(truncated);
}

RLF_TEST_CASE("RLF-6 rejects duplicate IDs and cyclic goal graphs") {
    rlf::agent::AgentFabric fabric({}, 17U);
    rlf::agent::Goal first;
    first.stable_id = 10U;
    first.specification = "first";
    first.completion_conditions = {{"a", "yes", false}};
    first.provenance = "test";
    static_cast<void>(fabric.add_goal(first));
    rlf::agent::Goal duplicate = first;
    duplicate.specification = "duplicate";
    RLF_CHECK_THROWS_AS(fabric.add_goal(duplicate), std::invalid_argument);

    rlf::agent::Goal second;
    second.stable_id = 11U;
    second.specification = "second";
    second.completion_conditions = {{"b", "yes", false}};
    second.provenance = "test";
    static_cast<void>(fabric.add_goal(second));
    auto snapshot = fabric.snapshot();
    snapshot.state.goal_stack[0].dependencies = {11U};
    snapshot.state.goal_stack[1].dependencies = {10U};
    RLF_CHECK_THROWS_AS(
        rlf::agent::AgentFabric::from_snapshot(snapshot), std::runtime_error
    );
}
