#pragma once

#include "rlf/frontier/frontier_backend.hpp"
#include "rlf/solstice/sparse_router.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace rlf::solstice {

struct ImageLimits final {
    std::size_t maximum_file_bytes{64U * 1024U * 1024U};
    std::size_t maximum_width{8'192U};
    std::size_t maximum_height{8'192U};
    std::size_t maximum_pixels{32U * 1024U * 1024U};
};

struct ImageData final {
    std::size_t width{};
    std::size_t height{};
    std::vector<std::uint8_t> rgb;
};

[[nodiscard]] ImageData load_image(
    const std::filesystem::path& path,
    ImageLimits limits = {}
);

[[nodiscard]] ImageData decode_image(
    std::span<const std::uint8_t> encoded,
    std::string_view extension,
    ImageLimits limits = {}
);

struct VisionConfig final {
    std::size_t patch_size{16U};
    std::vector<std::size_t> patch_sizes;
    std::size_t descriptor_dimensions{32U};
    std::size_t maximum_input_side{512U};
    std::size_t maximum_patches{16'384U};
    std::size_t retrieval_query_batch{256U};
    std::size_t retrieval_candidate_batch{4'096U};
    std::size_t training_patch_batch{256U};
    std::size_t sparse_routing_minimum_modes{8'192U};
    SparseRouterConfig sparse_router{18U, 2'048U, 2U, 0x564953494F4E524FULL};
    std::size_t maximum_modes{4'096U};
    std::size_t maximum_examples{100'000U};
    std::size_t maximum_regions{32U};
    std::size_t maximum_concepts_per_mode{24U};
    double mode_creation_similarity{0.86};
    double example_match_similarity{0.72};
    double local_learning_rate{0.12};
};

struct VisualConceptCount final {
    std::string concept_name;
    std::uint64_t count{};
};

struct VisualMode final {
    std::uint64_t id{};
    std::vector<float> prototype;
    std::uint64_t support{};
    std::vector<VisualConceptCount> concepts;
};

struct VisualExample final {
    std::uint64_t id{};
    std::vector<float> global_descriptor;
    std::string caption;
    std::vector<std::string> concepts;
    std::uint64_t support{1U};
};

struct VisualRegion final {
    std::uint64_t mode_id{};
    std::size_t x{};
    std::size_t y{};
    std::size_t width{};
    std::size_t height{};
    std::size_t patch_count{};
    std::string concept_name;
    double confidence{};
};

struct VisionAnalysis final {
    std::size_t width{};
    std::size_t height{};
    std::string description;
    std::vector<std::string> concepts;
    std::vector<VisualRegion> regions;
    double confidence{};
    std::uint64_t nearest_example_id{};
};

struct VisionSnapshot final {
    VisionConfig config;
    std::uint64_t next_mode_id{1U};
    std::uint64_t next_example_id{1U};
    std::uint64_t images_seen{};
    std::vector<VisualMode> modes;
    std::vector<VisualExample> examples;
};

struct VisualTrainingOperationStats final {
    std::uint64_t concept_update_lookups{};
    std::uint64_t linear_concept_comparisons{};
    std::uint64_t indexed_concept_lookups{};
    std::uint64_t concept_index_rebuilds{};
    std::uint64_t concept_index_entries_built{};
    std::uint64_t example_duplicate_lookups{};
    std::uint64_t linear_example_comparisons{};
    std::uint64_t indexed_example_candidates{};
    std::uint64_t example_index_rebuilds{};
    std::uint64_t example_index_entries_built{};
    std::uint64_t mode_id_lookups{};
    std::uint64_t mode_id_index_full_rebuilds{};
    std::uint64_t mode_id_index_entries_rebuilt{};
    std::uint64_t mode_id_index_incremental_inserts{};
    std::uint64_t region_mode_id_lookups{};
    std::uint64_t linear_region_mode_comparisons{};
    std::uint64_t indexed_region_mode_lookups{};
};

class VisualPatchFabric final {
public:
    explicit VisualPatchFabric(VisionConfig config = {});

    void train(const ImageData& image, std::string_view caption);
    [[nodiscard]] VisionAnalysis train_and_analyze(
        const ImageData& image,
        std::string_view caption
    );
    void train_file(
        const std::filesystem::path& path,
        std::string_view caption,
        ImageLimits limits = {}
    );

    [[nodiscard]] VisionAnalysis analyze(const ImageData& image) const;
    [[nodiscard]] VisionAnalysis analyze_file(
        const std::filesystem::path& path,
        ImageLimits limits = {}
    ) const;
    [[nodiscard]] std::string grounding_text(const VisionAnalysis& analysis) const;

    void set_backend(rlf::frontier::FrontierBackendKind kind);
    [[nodiscard]] rlf::frontier::FrontierBackendKind backend_kind() const noexcept;
    [[nodiscard]] rlf::frontier::BackendCapabilities backend_capabilities() const noexcept;
    [[nodiscard]] rlf::frontier::BackendOperationStats backend_operation_stats() const noexcept;
    [[nodiscard]] SparseRouterOperationStats sparse_router_operation_stats() const noexcept;
    [[nodiscard]] VisualTrainingOperationStats training_operation_stats() const noexcept;
    void set_incremental_sparse_router_updates(bool enabled) noexcept;
    void set_batched_sparse_reranking(bool enabled) noexcept;
    void set_indexed_concept_updates(bool enabled);
    void set_indexed_example_duplicate_lookup(bool enabled);
    void set_persistent_mode_id_index(bool enabled);

    [[nodiscard]] const VisionConfig& config() const noexcept;
    [[nodiscard]] std::span<const VisualMode> modes() const noexcept;
    [[nodiscard]] std::span<const VisualExample> examples() const noexcept;
    [[nodiscard]] std::uint64_t images_seen() const noexcept;
    [[nodiscard]] std::uint64_t deterministic_hash() const noexcept;

    [[nodiscard]] VisionSnapshot snapshot() const;
    [[nodiscard]] static VisualPatchFabric from_snapshot(VisionSnapshot snapshot);

private:
    struct PreparedImage final {
        ImageData image;
        double scale_x{1.0};
        double scale_y{1.0};
    };

    struct PatchRecord final {
        std::size_t scale_index{};
        std::size_t patch_size{};
        std::size_t grid_x{};
        std::size_t grid_y{};
        std::size_t pixel_x{};
        std::size_t pixel_y{};
        std::size_t pixel_width{};
        std::size_t pixel_height{};
        std::vector<float> descriptor;
        std::uint64_t mode_id{};
        double mode_similarity{};
    };

    [[nodiscard]] PreparedImage prepare_image(const ImageData& image) const;
    [[nodiscard]] std::vector<PatchRecord> extract_patches(
        const ImageData& image
    ) const;
    [[nodiscard]] std::vector<float> global_descriptor(
        const ImageData& image,
        std::span<const PatchRecord> patches
    ) const;
    void train_prepared(
        const PreparedImage& prepared,
        std::vector<PatchRecord>& patches,
        std::string_view caption
    );
    [[nodiscard]] VisionAnalysis analyze_prepared(
        const ImageData& original,
        const PreparedImage& prepared,
        std::vector<PatchRecord> patches
    ) const;
    [[nodiscard]] static double descriptor_similarity(
        std::span<const float> left,
        std::span<const float> right
    ) noexcept;
    [[nodiscard]] std::pair<std::size_t, double> nearest_mode(
        std::span<const float> descriptor
    ) const;
    void update_mode_router() const;
    void assign_existing_modes(std::span<PatchRecord> patches) const;
    [[nodiscard]] static std::vector<std::string> caption_concepts(
        std::string_view caption
    );
    void rebuild_concept_indices();
    void rebuild_example_caption_index();
    void rebuild_mode_id_index();
    [[nodiscard]] std::vector<VisualRegion> build_regions(
        const ImageData& image,
        std::vector<PatchRecord> patches
    ) const;
    [[nodiscard]] std::string fallback_description(
        std::span<const float> descriptor,
        std::size_t region_count
    ) const;

    VisionConfig config_;
    std::uint64_t next_mode_id_{1U};
    std::uint64_t next_example_id_{1U};
    std::uint64_t images_seen_{};
    std::uint64_t mode_revision_{1U};
    std::vector<VisualMode> modes_;
    std::vector<VisualExample> examples_;
    std::unique_ptr<rlf::frontier::FrontierComputeBackend> backend_;
    mutable SparseRoutingIndex mode_router_;
    mutable std::uint64_t mode_router_revision_{};
    std::vector<std::vector<std::size_t>> mode_concept_indices_;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>>
        example_caption_index_;
    std::unordered_map<std::uint64_t, std::size_t> mode_id_index_;
    std::size_t mode_id_index_mode_count_{};
    mutable VisualTrainingOperationStats training_operation_stats_;
    bool incremental_sparse_router_updates_{true};
    bool batched_sparse_reranking_{true};
    bool sparse_rerank_policy_explicit_{};
    bool indexed_concept_updates_{true};
    bool indexed_example_duplicate_lookup_{true};
    bool persistent_mode_id_index_{true};
};

}  // namespace rlf::solstice
