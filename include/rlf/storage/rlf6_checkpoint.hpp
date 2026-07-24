#pragma once

#include "rlf/agent/agent_fabric.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace rlf::storage {

constexpr std::uint32_t rlf6_checkpoint_format_version = 8U;

struct Rlf6CheckpointLoadOptions final {
    std::size_t maximum_file_bytes{1U << 30U};
    std::size_t maximum_records{10'000'000U};
    std::size_t maximum_nested_records{100'000'000U};
    std::size_t maximum_string_bytes{1U << 20U};
    std::size_t maximum_route_length{1'000'000U};
};

struct Rlf6CheckpointSummary final {
    std::uint32_t format_version{};
    std::uint64_t seed{};
    std::uint64_t episode_id{};
    std::uint64_t step_index{};
    std::size_t observation_count{};
    std::size_t belief_count{};
    std::size_t goal_count{};
    std::size_t tool_count{};
    std::size_t transition_count{};
    std::size_t memory_count{};
    std::size_t skill_count{};
    std::size_t error_count{};
    std::uint64_t deterministic_hash{};
    std::uint64_t payload_checksum{};
    std::size_t file_bytes{};
};

void save_rlf6_checkpoint(
    const std::filesystem::path& path,
    const agent::AgentFabric& fabric
);
[[nodiscard]] agent::AgentFabric load_rlf6_checkpoint(
    const std::filesystem::path& path,
    const Rlf6CheckpointLoadOptions& options = {}
);
[[nodiscard]] Rlf6CheckpointSummary inspect_rlf6_checkpoint(
    const std::filesystem::path& path,
    const Rlf6CheckpointLoadOptions& options = {}
);

}  // namespace rlf::storage
