#pragma once

#include "rlf/core/temporal_predictive_fabric.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace rlf::storage {

constexpr std::uint32_t rlf4_checkpoint_format_version = 6U;

struct Rlf4CheckpointLoadOptions final {
    std::size_t maximum_file_bytes{1U << 30U};
    std::size_t maximum_dimension{1'048'576U};
    std::size_t maximum_prototypes{10'000'000U};
    std::size_t maximum_contexts{20'000'000U};
    std::size_t maximum_outcomes{100'000'000U};
    std::size_t maximum_options{10'000'000U};
    std::size_t maximum_sequence_length{1'024U};
};

struct Rlf4CheckpointSummary final {
    std::uint32_t format_version{};
    std::uint64_t seed{};
    std::uint64_t training_step{};
    std::size_t dimension{};
    std::size_t prototype_count{};
    std::size_t context_count{};
    std::size_t outcome_count{};
    std::size_t option_count{};
    std::uint64_t payload_checksum{};
    std::size_t file_bytes{};
};

void save_rlf4_checkpoint(
    const std::filesystem::path& path,
    const core::TemporalPredictiveFabric& fabric
);
[[nodiscard]] core::TemporalPredictiveFabric load_rlf4_checkpoint(
    const std::filesystem::path& path,
    const Rlf4CheckpointLoadOptions& options = {}
);
[[nodiscard]] Rlf4CheckpointSummary inspect_rlf4_checkpoint(
    const std::filesystem::path& path,
    const Rlf4CheckpointLoadOptions& options = {}
);

}  // namespace rlf::storage
