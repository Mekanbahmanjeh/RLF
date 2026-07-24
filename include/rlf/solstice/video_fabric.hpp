#pragma once

#include "rlf/solstice/general_fabric.hpp"
#include "rlf/solstice/vision_fabric.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rlf::solstice {

// This component learns and renders compact motion prototypes.  It is not a
// photorealistic pixel generator and must not be reported as one.
struct VideoFabricConfig final {
    std::size_t maximum_sequences{1U};
    std::size_t maximum_frames_per_sequence{64U};
    std::size_t maximum_generation_frames{256U};
    std::size_t maximum_prompt_bytes{64U * 1024U};
    std::size_t output_width{96U};
    std::size_t output_height{96U};
    double minimum_prompt_similarity{0.08};
    double local_learning_rate{0.20};
};

struct VideoMotionDescriptor final {
    double start_x{0.5};
    double start_y{0.5};
    double velocity_x{};
    double velocity_y{};
    double object_width{0.25};
    double object_height{0.25};
    std::array<double, 3U> foreground_rgb{0.75, 0.35, 0.20};
    std::array<double, 3U> background_rgb{};
};

struct VideoPrototype final {
    std::uint64_t id{};
    std::string source_sequence_id;
    std::string prompt;
    SemanticSignature prompt_signature;
    VideoMotionDescriptor motion;
    double frames_per_second{24.0};
    std::size_t observed_frames{};
    std::uint64_t support{1U};
};

struct VideoGeneration final {
    std::uint64_t prototype_id{};
    std::string source_sequence_id;
    double prompt_similarity{};
    double confidence{};
    double frames_per_second{};
    VideoMotionDescriptor motion;
    std::vector<ImageData> frames;
};

struct VideoFabricSnapshot final {
    VideoFabricConfig config;
    std::uint64_t next_prototype_id{1U};
    std::uint64_t sequences_seen{};
    std::uint64_t frames_seen{};
    std::vector<VideoPrototype> prototypes;
};

class VideoPrototypeFabric final {
public:
    explicit VideoPrototypeFabric(VideoFabricConfig config = {});

    std::uint64_t train(
        std::string_view sequence_id,
        std::string_view prompt,
        double frames_per_second,
        std::span<const ImageData> frames
    );

    [[nodiscard]] VideoGeneration generate(
        std::string_view prompt,
        std::size_t frame_count
    ) const;

    [[nodiscard]] static VideoMotionDescriptor describe(
        std::span<const ImageData> frames
    );
    static void save_ppm(const std::filesystem::path& path, const ImageData& image);

    [[nodiscard]] const VideoFabricConfig& config() const noexcept;
    [[nodiscard]] std::span<const VideoPrototype> prototypes() const noexcept;
    [[nodiscard]] std::uint64_t sequences_seen() const noexcept;
    [[nodiscard]] std::uint64_t frames_seen() const noexcept;
    [[nodiscard]] bool contains_source_sequence(std::string_view sequence_id) const noexcept;
    [[nodiscard]] std::uint64_t deterministic_hash() const noexcept;

    [[nodiscard]] VideoFabricSnapshot snapshot() const;
    [[nodiscard]] static VideoPrototypeFabric from_snapshot(
        VideoFabricSnapshot snapshot
    );

private:
    [[nodiscard]] static ImageData render(
        const VideoMotionDescriptor& motion,
        std::size_t width,
        std::size_t height,
        std::size_t frame_index
    );

    VideoFabricConfig config_;
    std::uint64_t next_prototype_id_{1U};
    std::uint64_t sequences_seen_{};
    std::uint64_t frames_seen_{};
    std::vector<VideoPrototype> prototypes_;
};

}  // namespace rlf::solstice
