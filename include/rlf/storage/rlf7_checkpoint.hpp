#pragma once

#include "rlf/frontier/frontier_trainer.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace rlf::storage {

struct FrontierCheckpointLoadOptions final {
    std::size_t maximum_file_bytes{4ULL << 30U};
    std::size_t maximum_records{100'000'000U};
    std::size_t maximum_modes{10'000'000U};
    std::size_t maximum_string_bytes{4U << 20U};
    std::size_t maximum_prototype_values{1U << 20U};
};

struct FrontierCheckpointSummary final {
    std::string architecture;
    std::uint32_t format_version{};
    std::uint64_t seed{};
    std::uint64_t training_step{};
    std::uint64_t training_examples{};
    std::uint64_t evaluation_examples{};
    std::size_t knowledge_records{};
    std::size_t modes{};
    std::uint64_t deterministic_hash{};
    std::uint64_t payload_checksum{};
    std::size_t file_bytes{};
    std::string backend;
};

void save_rlf7_checkpoint(
    const std::filesystem::path& path,
    const frontier::FrontierModel& model
);
[[nodiscard]] frontier::FrontierModel load_rlf7_checkpoint(
    const std::filesystem::path& path,
    const FrontierCheckpointLoadOptions& options = {}
);

void save_frontier_checkpoint(
    const std::filesystem::path& path,
    const frontier::FrontierModel& model,
    std::string backend
);
[[nodiscard]] frontier::FrontierModel load_frontier_checkpoint(
    const std::filesystem::path& path,
    const FrontierCheckpointLoadOptions& options = {}
);

[[nodiscard]] FrontierCheckpointSummary inspect_frontier_checkpoint(
    const std::filesystem::path& path,
    const FrontierCheckpointLoadOptions& options = {}
);

}  // namespace rlf::storage
