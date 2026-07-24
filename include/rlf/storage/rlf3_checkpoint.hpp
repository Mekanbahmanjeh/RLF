#pragma once

#include "rlf/core/sparse_world_model.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace rlf::storage {

constexpr std::uint32_t rlf3_checkpoint_format_version = 5U;

struct Rlf3CheckpointLoadOptions final {
    std::size_t maximum_file_bytes{1U << 30U};
    std::size_t maximum_dimension{1'048'576U};
    std::size_t maximum_actions{1'000'000U};
    std::size_t maximum_states{10'000'000U};
    std::size_t maximum_contexts{1'000'000U};
    std::size_t maximum_transitions{20'000'000U};
    std::size_t maximum_outcomes{100'000'000U};
    std::size_t maximum_subgoals{20'000'000U};
    std::size_t maximum_string_bytes{1U << 20U};
};

struct Rlf3CheckpointSummary final {
    std::uint32_t format_version{};
    std::uint64_t seed{};
    std::uint64_t training_step{};
    std::size_t dimension{};
    std::size_t action_count{};
    std::size_t state_count{};
    std::size_t context_count{};
    std::size_t transition_count{};
    std::size_t outcome_count{};
    std::size_t subgoal_count{};
    std::uint64_t payload_checksum{};
    std::size_t file_bytes{};
};

void save_rlf3_checkpoint(
    const std::filesystem::path& path,
    const core::SparseWorldModel& model
);
[[nodiscard]] core::SparseWorldModel load_rlf3_checkpoint(
    const std::filesystem::path& path,
    const Rlf3CheckpointLoadOptions& options = {}
);
[[nodiscard]] Rlf3CheckpointSummary inspect_rlf3_checkpoint(
    const std::filesystem::path& path,
    const Rlf3CheckpointLoadOptions& options = {}
);

}  // namespace rlf::storage
