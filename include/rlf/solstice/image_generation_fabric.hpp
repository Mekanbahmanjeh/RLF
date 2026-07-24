#pragma once

#include "rlf/solstice/vision_fabric.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rlf::solstice {

struct ImageGenerationConfig final {
    std::size_t tile_size{16U};
    std::size_t coordinate_bins{16U};
    std::size_t maximum_source_images{100'000U};
    std::size_t maximum_tile_prototypes{1'000'000U};
    std::size_t maximum_source_side{8'192U};
    std::size_t maximum_source_pixels{32U * 1024U * 1024U};
    std::size_t maximum_caption_bytes{4'096U};
    std::size_t maximum_caption_concepts{32U};
    std::uint64_t maximum_total_caption_bytes{256ULL * 1024ULL * 1024ULL};
    std::uint64_t maximum_total_concept_bytes{256ULL * 1024ULL * 1024ULL};
    std::uint64_t maximum_posting_entries{16'000'000ULL};
    std::size_t maximum_candidates_per_cell{4'096U};
    std::size_t default_output_width{256U};
    std::size_t default_output_height{256U};
    std::size_t maximum_output_side{2'048U};
    std::size_t maximum_output_pixels{4U * 1024U * 1024U};
    double semantic_weight{2.0};
    double spatial_weight{0.35};
    double seam_weight{0.85};
    double support_weight{0.05};
};

struct ImageGenerationSource final {
    std::uint64_t id{};
    std::string caption;
    std::vector<std::string> concepts;
    std::size_t width{};
    std::size_t height{};
    std::size_t first_tile{};
    std::size_t tile_count{};
};

struct ImageTilePrototype final {
    std::uint64_t id{};
    std::uint64_t source_id{};
    std::uint16_t x_bin{};
    std::uint16_t y_bin{};
    std::array<float, 12U> descriptor{};
    std::vector<std::uint8_t> rgb;
    std::uint64_t support{1U};
};

struct ImageGenerationRequest final {
    std::string prompt;
    std::size_t width{};
    std::size_t height{};
    std::uint64_t seed{0x494D41474547454EULL};
};

struct GeneratedImage final {
    ImageData image;
    std::vector<std::uint64_t> selected_prototype_ids;
    double mean_semantic_score{};
    double mean_seam_score{};
    std::size_t fallback_cells{};
    std::uint64_t deterministic_hash{};
};

struct ImageGenerationOperationStats final {
    std::uint64_t training_calls{};
    std::uint64_t source_images_inserted{};
    std::uint64_t tile_prototypes_inserted{};
    std::uint64_t source_capacity_rejections{};
    std::uint64_t tile_capacity_rejections{};
    std::uint64_t string_budget_rejections{};
    std::uint64_t posting_budget_rejections{};
    std::uint64_t generation_calls{};
    std::uint64_t candidate_bucket_lookups{};
    std::uint64_t candidates_scored{};
    std::uint64_t fallback_cells{};
};

struct PatchQuiltSnapshot final {
    ImageGenerationConfig config;
    std::uint64_t next_source_id{1U};
    std::uint64_t next_tile_id{1U};
    std::uint64_t images_seen{};
    std::vector<ImageGenerationSource> sources;
    std::vector<ImageTilePrototype> tiles;
};

enum class ImageGenerationProfile : std::uint8_t {
    reference,
    a100_80g,
    v100_32g,
};

enum class ImageGenerationArchitecture : std::uint8_t {
    patch_quilt_baseline,
    resonant_fabric,
};

[[nodiscard]] std::string_view to_string(
    ImageGenerationArchitecture architecture
) noexcept;

struct ImageGenerationCapacity final {
    std::uint64_t gpu_working_set_bytes{};
    std::uint64_t peak_vram_limit_bytes{};
    std::uint64_t host_ram_recommended_bytes{};
    std::uint64_t checkpoint_ceiling_bytes{};
};

struct PatchQuiltScaleProjection final {
    std::size_t width{};
    std::size_t height{};
    std::uint64_t tiles_per_image{};
    std::uint64_t maximum_images_by_tile_capacity{};
    std::uint64_t maximum_images_by_source_capacity{};
    std::uint64_t maximum_simultaneous_images{};
    std::uint64_t raw_rgb_bytes_at_tile_capacity{};
};

[[nodiscard]] ImageGenerationConfig make_image_generation_profile_config(
    ImageGenerationProfile profile
);
[[nodiscard]] ImageGenerationCapacity estimate_image_generation_capacity(
    ImageGenerationProfile profile
) noexcept;
[[nodiscard]] PatchQuiltScaleProjection project_patch_quilt_scale(
    const ImageGenerationConfig& config,
    std::size_t width,
    std::size_t height
);
[[nodiscard]] bool image_generation_profile_config_matches(
    ImageGenerationProfile profile,
    const ImageGenerationConfig& config
) noexcept;
[[nodiscard]] std::string_view to_string(ImageGenerationProfile profile) noexcept;
[[nodiscard]] ImageGenerationProfile parse_image_generation_profile(
    std::string_view value
);

class PatchQuiltBaseline final {
public:
    explicit PatchQuiltBaseline(ImageGenerationConfig config = {});

    void train(const ImageData& image, std::string_view caption);
    [[nodiscard]] GeneratedImage generate(const ImageGenerationRequest& request) const;

    [[nodiscard]] const ImageGenerationConfig& config() const noexcept;
    [[nodiscard]] std::span<const ImageGenerationSource> sources() const noexcept;
    [[nodiscard]] std::span<const ImageTilePrototype> tiles() const noexcept;
    [[nodiscard]] std::uint64_t images_seen() const noexcept;
    [[nodiscard]] ImageGenerationOperationStats operation_stats() const noexcept;
    [[nodiscard]] std::uint64_t deterministic_hash() const noexcept;

    [[nodiscard]] PatchQuiltSnapshot snapshot() const;
    [[nodiscard]] static PatchQuiltBaseline from_snapshot(
        PatchQuiltSnapshot snapshot
    );

private:
    [[nodiscard]] static std::vector<std::string> caption_concepts(
        std::string_view caption,
        std::size_t maximum_concepts
    );
    [[nodiscard]] std::uint64_t bucket_key(
        std::uint64_t concept_hash,
        std::uint16_t x_bin,
        std::uint16_t y_bin
    ) const noexcept;
    void rebuild_indices();
    void index_tile(std::size_t tile_index);

    ImageGenerationConfig config_;
    std::uint64_t next_source_id_{1U};
    std::uint64_t next_tile_id_{1U};
    std::uint64_t images_seen_{};
    std::uint64_t stored_caption_bytes_{};
    std::uint64_t stored_concept_bytes_{};
    std::uint64_t posting_entries_{};
    std::vector<ImageGenerationSource> sources_;
    std::vector<ImageTilePrototype> tiles_;
    std::unordered_map<std::uint64_t, std::size_t> source_id_index_;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> tile_buckets_;
    mutable ImageGenerationOperationStats operation_stats_;
};

}  // namespace rlf::solstice
