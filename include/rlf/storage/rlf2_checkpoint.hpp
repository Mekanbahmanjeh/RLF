#pragma once

#include "rlf/core/predictive_skill_fabric.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace rlf::storage {

constexpr std::uint32_t rlf2_checkpoint_format_version = 4U;

struct Rlf2CheckpointLoadOptions final {
    std::size_t maximum_file_bytes{1U << 30U};
    std::size_t maximum_dimension{1'048'576U};
    std::size_t maximum_operators{1'000'000U};
    std::size_t maximum_skills{10'000'000U};
    std::size_t maximum_prototypes{10'000'000U};
    std::size_t maximum_route_length{4'096U};
    std::size_t maximum_profile_values{16'777'216U};
    std::size_t maximum_string_bytes{1U << 20U};
};

struct Rlf2CheckpointSummary final {
    std::uint32_t format_version{};
    std::uint64_t seed{};
    std::uint64_t training_step{};
    std::size_t dimension{};
    std::size_t operator_count{};
    std::size_t skill_count{};
    std::size_t compound_skill_count{};
    std::size_t prototype_count{};
    std::uint64_t payload_checksum{};
    std::size_t file_bytes{};
};

void save_rlf2_checkpoint(
    const std::filesystem::path& path,
    const core::PredictiveSkillFabric& fabric
);
[[nodiscard]] core::PredictiveSkillFabric load_rlf2_checkpoint(
    const std::filesystem::path& path,
    const Rlf2CheckpointLoadOptions& options = {}
);
[[nodiscard]] Rlf2CheckpointSummary inspect_rlf2_checkpoint(
    const std::filesystem::path& path,
    const Rlf2CheckpointLoadOptions& options = {}
);

}  // namespace rlf::storage
