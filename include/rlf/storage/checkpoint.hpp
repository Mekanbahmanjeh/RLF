#pragma once

#include "rlf/core/resonant_fabric.hpp"
#include "rlf/memory/associative_memory.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace rlf::storage {

constexpr std::uint32_t checkpoint_format_version = 2U;

struct CheckpointData final {
    core::FabricConfig config;
    std::uint64_t master_seed;
    std::uint64_t training_step;
    std::vector<core::ResonantMode> modes;
    memory::AssociativeMemory associative_memory;
    learning::StructuralStatistics structural_statistics;
    std::string update_strategy;
    std::map<std::string, std::string> experiment_metadata;
};

struct CheckpointLoadOptions final {
    std::size_t maximum_file_bytes{1U << 30U};
    std::size_t maximum_dimension{
        core::PhaseVector::default_max_serialized_dimension
    };
    std::size_t maximum_modes{1'000'000U};
    std::size_t maximum_corrections_per_mode{4'096U};
    std::size_t maximum_memory_capacity{1'000'000U};
    std::size_t maximum_memory_records{1'000'000U};
    std::size_t maximum_payload_bytes{1U << 28U};
    std::size_t maximum_metadata_entries{4'096U};
    std::size_t maximum_string_bytes{1U << 20U};
    std::optional<std::size_t> expected_dimension{};
};

struct CheckpointSummary final {
    std::uint32_t format_version;
    std::uint64_t master_seed;
    std::uint64_t training_step;
    std::size_t dimension;
    std::size_t mode_count;
    std::size_t enabled_mode_count;
    std::size_t associative_record_count;
    std::size_t associative_capacity;
    learning::StructuralStatistics structural_statistics;
    std::string update_strategy;
    std::map<std::string, std::string> experiment_metadata;
    std::uint64_t payload_checksum;
    std::size_t file_bytes;
};

void save_checkpoint(
    const std::filesystem::path& path,
    const CheckpointData& checkpoint
);
[[nodiscard]] CheckpointData load_checkpoint(
    const std::filesystem::path& path,
    const CheckpointLoadOptions& options = {}
);
[[nodiscard]] CheckpointSummary inspect_checkpoint(
    const std::filesystem::path& path,
    const CheckpointLoadOptions& options = {}
);

}  // namespace rlf::storage
