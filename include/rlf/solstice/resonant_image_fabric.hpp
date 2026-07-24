#pragma once

#include "rlf/core/phase_vector.hpp"
#include "rlf/core/resonant_mode.hpp"
#include "rlf/frontier/frontier_backend.hpp"
#include "rlf/solstice/image_generation_fabric.hpp"
#include "rlf/solstice/prompt_semantic_fabric.hpp"
#include "rlf/solstice/vision_fabric.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rlf::solstice {

// CPU reference configuration for a controlled, non-neural image-
// transformation learner. This is a research component, not a claim of
// unrestricted text-to-image generation or state-of-the-art quality.
struct ResonantImageConfig final {
    std::size_t patch_size{1U};
    std::size_t phase_redundancy{4U};
    std::size_t coordinate_bins{16U};
    std::size_t maximum_modes{65'536U};
    std::size_t maximum_concept_bytes{256U};
    std::size_t maximum_image_side{1'024U};
    std::size_t maximum_image_pixels{1U * 1'024U * 1'024U};
    std::size_t candidate_count{16U};
    std::size_t active_count{1U};
    std::size_t maximum_settling_cycles{12U};
    std::size_t maximum_trace_entries{1'000'000U};
    std::size_t maximum_prompt_concepts{64U};
    std::size_t maximum_semantic_candidates{64U};
    double minimum_resonance{0.50};
    double minimum_semantic_similarity{0.12};
    double semantic_resonance_weight{0.65};
    double convergence_tolerance_radians{1.0e-4};
    double settling_relaxation{0.75};
    double transformation_learning_rate{0.25};
    double context_learning_rate{0.05};
    double confidence_learning_rate{0.05};
    std::uint64_t seed{0x524C46494D475245ULL};
    PromptSemanticConfig prompt_semantics{};
};

struct ResonantImageTrainingPair final {
    ImageData source;
    ImageData target;
    std::string semantic_label;
};

struct ResonantImageMode final {
    std::string semantic_label;
    std::uint16_t x_bin{};
    std::uint16_t y_bin{};
    core::ResonantMode resonant_mode;
    std::uint64_t example_count{1ULL};
};

struct ResonantImageOperationStats final {
    std::uint64_t training_examples{};
    std::uint64_t training_patches{};
    std::uint64_t modes_created{};
    std::uint64_t local_mode_updates{};
    std::uint64_t generation_calls{};
    std::uint64_t generated_patches{};
    std::uint64_t sparse_bucket_lookups{};
    std::uint64_t resonance_evaluations{};
    std::uint64_t active_mode_applications{};
    std::uint64_t settling_cycles{};
    std::uint64_t unresolved_patch_transformations{};
    std::uint64_t decoded_channels{};
    std::uint64_t semantic_bucket_lookups{};
    std::uint64_t semantic_candidates_scored{};
    std::uint64_t semantic_matches{};
};

// Complete persistent state. Derived carrier tables and the sparse cell index
// are rebuilt deterministically from this snapshot when loading.
struct ResonantImageSnapshot final {
    ResonantImageConfig config;
    std::uint64_t next_mode_id{1ULL};
    std::vector<ResonantImageMode> modes;
    ResonantImageOperationStats operation_stats;
    PromptSemanticSnapshot prompt_semantics;
};

struct ResonantImageTrainingResult final {
    std::size_t patches{};
    std::size_t modes_created{};
    std::size_t modes_updated{};
    ResonantImageOperationStats operation_delta;
};

struct ResonantImageTraceEntry final {
    std::string semantic_label;
    std::size_t patch_x{};
    std::size_t patch_y{};
    std::size_t cycle{};
    std::vector<std::uint64_t> active_mode_ids;
    double strongest_resonance{};
    double state_change_radians{};
};

struct ResonantImageGenerateRequest final {
    ImageData base_image;
    std::vector<std::string> transformations;
    bool capture_trace{false};
};

struct ResonantGeneratedImage final {
    ImageData image;
    std::vector<std::uint64_t> selected_mode_ids;
    std::vector<ResonantImageTraceEntry> trace;
    ResonantImageOperationStats operation_delta;
    std::uint64_t deterministic_hash{};
};

struct ResonantImageQuality final {
    double mean_absolute_error{};
    double mean_squared_error{};
    double peak_signal_to_noise_db{
        std::numeric_limits<double>::infinity()
    };
    double exact_channel_fraction{1.0};
    double exact_pixel_fraction{1.0};
};

// Keeps quality and work accounting side by side so a faster but lower-quality
// method cannot look better merely by omitting its accuracy loss.
struct ResonantImageComparison final {
    ResonantImageQuality quality;
    ResonantImageOperationStats generation_operations;
    std::size_t learned_modes{};
    std::size_t estimated_model_bytes{};
};

[[nodiscard]] ResonantImageConfig make_resonant_image_profile_config(
    ImageGenerationProfile profile
);
[[nodiscard]] bool resonant_image_profile_config_matches(
    ImageGenerationProfile profile,
    const ResonantImageConfig& config
) noexcept;

[[nodiscard]] ResonantImageQuality evaluate_resonant_image_quality(
    const ImageData& candidate,
    const ImageData& reference
);
[[nodiscard]] std::vector<std::string> parse_resonant_image_prompt(
    std::string_view prompt
);

class ResonantImageFabric final {
public:
    explicit ResonantImageFabric(ResonantImageConfig config = {});

    [[nodiscard]] const ResonantImageConfig& config() const noexcept;
    [[nodiscard]] std::span<const ResonantImageMode> modes() const noexcept;
    [[nodiscard]] ResonantImageOperationStats operation_stats() const noexcept;
    void set_backend(frontier::FrontierBackendKind kind);
    [[nodiscard]] frontier::FrontierBackendKind backend_kind() const noexcept;
    [[nodiscard]] frontier::BackendOperationStats backend_operation_stats()
        const noexcept;
    void reset_operation_stats() noexcept;
    void train_prompt_language_record(std::string_view text);
    [[nodiscard]] const PromptSemanticFabric& prompt_semantics() const noexcept;
    void rebuild_semantic_index();

    [[nodiscard]] core::PhaseVector encode_patch(
        const ImageData& image,
        std::size_t patch_x,
        std::size_t patch_y
    ) const;
    [[nodiscard]] ImageData decode_patch(
        const core::PhaseVector& state
    ) const;

    [[nodiscard]] ResonantImageTrainingResult train(
        const ResonantImageTrainingPair& example
    );
    [[nodiscard]] ResonantGeneratedImage generate(
        const ResonantImageGenerateRequest& request
    );

    [[nodiscard]] ResonantImageComparison compare(
        const ResonantGeneratedImage& generated,
        const ImageData& reference
    ) const;
    [[nodiscard]] std::size_t estimated_model_bytes() const noexcept;
    [[nodiscard]] std::uint64_t deterministic_hash() const noexcept;
    [[nodiscard]] ResonantImageSnapshot snapshot() const;
    [[nodiscard]] static ResonantImageFabric from_snapshot(
        ResonantImageSnapshot snapshot
    );

private:
    struct CellKey final {
        std::string semantic_label;
        std::uint16_t x_bin{};
        std::uint16_t y_bin{};

        [[nodiscard]] bool operator<(const CellKey& other) const noexcept;
    };

    struct RetrievedMode final {
        std::size_t mode_index{};
        double resonance{};
        double semantic_similarity{1.0};
    };

    ResonantImageConfig config_;
    std::size_t phase_dimension_{};
    std::vector<float> carriers_;
    std::vector<std::int8_t> carrier_signs_;
    std::vector<ResonantImageMode> modes_;
    std::map<CellKey, std::size_t> cell_index_;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>>
        semantic_cell_index_;
    std::uint64_t next_mode_id_{1ULL};
    ResonantImageOperationStats operation_stats_;
    PromptSemanticFabric prompt_semantics_;
    bool semantic_index_dirty_{false};
    std::unique_ptr<frontier::FrontierComputeBackend> backend_;

    void validate_image(const ImageData& image) const;
    void validate_concept(std::string_view semantic_label) const;
    [[nodiscard]] std::pair<std::uint16_t, std::uint16_t> coordinate_bin(
        std::size_t patch_x,
        std::size_t patch_y,
        std::size_t patch_columns,
        std::size_t patch_rows
    ) const;
    [[nodiscard]] core::PhaseVector position_context(
        std::uint16_t x_bin,
        std::uint16_t y_bin
    ) const;
    [[nodiscard]] std::vector<RetrievedMode> retrieve(
        std::string_view semantic_label,
        std::uint16_t x_bin,
        std::uint16_t y_bin,
        const core::PhaseVector& context
    );
    void index_semantic_mode(std::size_t mode_index);
    void ensure_semantic_index();
    [[nodiscard]] std::vector<std::uint64_t> prompt_concepts(
        std::string_view prompt
    ) const;
    [[nodiscard]] core::PhaseVector settle_patch(
        const core::PhaseVector& input,
        std::string_view semantic_label,
        std::size_t patch_x,
        std::size_t patch_y,
        std::size_t patch_columns,
        std::size_t patch_rows,
        bool capture_trace,
        std::vector<ResonantImageTraceEntry>& trace,
        std::vector<std::uint64_t>& selected_mode_ids
    );
};

}  // namespace rlf::solstice
