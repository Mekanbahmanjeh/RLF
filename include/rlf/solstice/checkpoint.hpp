#pragma once

#include "rlf/solstice/solstice_model.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace rlf::solstice {

enum class SolsticeProfile : std::uint8_t;

struct SolsticeCheckpointLimits final {
    std::uint64_t maximum_file_bytes{1'025ULL * 1024ULL * 1024ULL * 1024ULL};
    std::size_t maximum_string_bytes{16U * 1024U * 1024U};
    std::size_t maximum_collection_entries{250'000'000U};
};

struct SolsticeCheckpointSummary final {
    std::uint32_t format_version{};
    std::uint64_t file_bytes{};
    std::uint64_t payload_checksum{};
    std::uint64_t seed{};
    SolsticeStats stats;
};

// Returns defensive deserialization limits for an explicitly selected
// profile. The unusually large 500M campaign limits are opt-in: ordinary
// checkpoints retain the established 1,025 GiB / 250M-entry ceilings.
[[nodiscard]] SolsticeCheckpointLimits checkpoint_limits_for_profile(
    SolsticeProfile profile
) noexcept;

void save_solstice_checkpoint(
    const std::filesystem::path& path,
    const SolsticeModel& model
);

[[nodiscard]] SolsticeModel load_solstice_checkpoint(
    const std::filesystem::path& path,
    SolsticeCheckpointLimits limits = {}
);

[[nodiscard]] SolsticeCheckpointSummary inspect_solstice_checkpoint(
    const std::filesystem::path& path,
    SolsticeCheckpointLimits limits = {}
);

// Loads the checkpoint once using the selected profile's defensive limits and
// rejects it unless its serialized configuration exactly matches that profile.
// This is the required inspection path for serving and external evaluation.
[[nodiscard]] SolsticeCheckpointSummary inspect_solstice_checkpoint_for_profile(
    const std::filesystem::path& path,
    SolsticeProfile profile
);

}  // namespace rlf::solstice
