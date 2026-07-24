#pragma once

#include "rlf/solstice/image_generation_fabric.hpp"
#include "rlf/solstice/resonant_image_fabric.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace rlf::solstice {

struct ImageGenerationShardRecord final {
    std::string shard_id;
    std::string shard_sha256;
    std::string ledger_sha256;
    std::string source_uri;
    std::string license;
    std::uint64_t records{};
    std::uint64_t bytes{};
};

struct ImageGenerationCheckpointState final {
    ImageGenerationProfile profile{ImageGenerationProfile::reference};
    ImageGenerationArchitecture architecture{
        ImageGenerationArchitecture::patch_quilt_baseline
    };
    std::uint64_t master_seed{};
    std::uint64_t training_step{};
    PatchQuiltBaseline fabric;
    ResonantImageFabric resonant_fabric;
    ImageGenerationOperationStats cumulative_operations;
    std::vector<ImageGenerationShardRecord> completed_shards;
};

struct ImageGenerationCheckpointLimits final {
    std::uint64_t maximum_file_bytes{513ULL * 1024ULL * 1024ULL * 1024ULL};
    std::size_t maximum_source_images{4'000'000U};
    std::size_t maximum_tile_prototypes{48'000'000U};
    std::size_t maximum_string_bytes{16'384U};
    std::size_t maximum_total_concepts{192'000'000U};
    std::size_t maximum_completed_shards{1'000'000U};
    std::uint64_t maximum_rgb_bytes{48ULL * 1024ULL * 1024ULL * 1024ULL};
    std::size_t maximum_resonant_modes{48'000'000U};
    std::uint64_t maximum_phase_values{96ULL * 1024ULL * 1024ULL * 1024ULL};
    std::size_t maximum_corrections_per_mode{1'024U};
    std::size_t maximum_prompt_semantic_modes{2'000'000U};
};

struct ImageGenerationCheckpointSummary final {
    std::uint32_t format_version{};
    std::uint64_t file_bytes{};
    std::string file_sha256;
    std::string payload_sha256;
    ImageGenerationProfile profile{ImageGenerationProfile::reference};
    ImageGenerationArchitecture architecture{
        ImageGenerationArchitecture::patch_quilt_baseline
    };
    std::uint64_t master_seed{};
    std::uint64_t training_step{};
    std::uint64_t images_seen{};
    std::size_t source_images{};
    std::size_t tile_prototypes{};
    std::size_t learned_modes{};
    std::uint64_t prompt_language_records{};
    std::uint64_t prompt_language_words{};
    std::size_t prompt_semantic_modes{};
    std::size_t completed_shards{};
    std::uint64_t deterministic_model_hash{};
};

void save_image_generation_checkpoint(
    const std::filesystem::path& path,
    const ImageGenerationCheckpointState& state
);

[[nodiscard]] ImageGenerationCheckpointState load_image_generation_checkpoint(
    const std::filesystem::path& path,
    ImageGenerationCheckpointLimits limits = {}
);

[[nodiscard]] ImageGenerationCheckpointSummary inspect_image_generation_checkpoint(
    const std::filesystem::path& path,
    ImageGenerationCheckpointLimits limits = {}
);

}  // namespace rlf::solstice
