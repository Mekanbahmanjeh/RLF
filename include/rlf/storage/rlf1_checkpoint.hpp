#pragma once

#include "rlf/core/latent_routing.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace rlf::storage {

constexpr std::uint32_t rlf1_checkpoint_format_version = 3U;

struct Rlf1CheckpointLoadOptions final {
    std::size_t maximum_file_bytes{1U << 30U};
    std::size_t maximum_dimension{1'048'576U};
    std::size_t maximum_operators{1'000'000U};
    std::size_t maximum_modes{10'000'000U};
    std::size_t maximum_halt_modes{1'000'000U};
    std::size_t maximum_route_records{1'000'000U};
    std::size_t maximum_route_length{4'096U};
    std::size_t maximum_string_bytes{1U << 20U};
};

struct Rlf1CheckpointSummary final {
    std::uint32_t format_version{};
    std::uint64_t seed{};
    std::uint64_t training_step{};
    std::size_t dimension{};
    std::size_t operator_count{};
    std::size_t macro_operator_count{};
    std::size_t routing_mode_count{};
    std::size_t halt_mode_count{};
    std::size_t route_memory_count{};
    std::uint64_t payload_checksum{};
    std::size_t file_bytes{};
};

void save_rlf1_checkpoint(
    const std::filesystem::path& path,
    const core::LatentRouter& router
);
[[nodiscard]] core::LatentRouter load_rlf1_checkpoint(
    const std::filesystem::path& path,
    const Rlf1CheckpointLoadOptions& options = {}
);
[[nodiscard]] Rlf1CheckpointSummary inspect_rlf1_checkpoint(
    const std::filesystem::path& path,
    const Rlf1CheckpointLoadOptions& options = {}
);

}  // namespace rlf::storage
