#pragma once

#include "rlf/frontier/knowledge_fabric.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace rlf::frontier {

struct Image final {
    std::size_t width{};
    std::size_t height{};
    std::size_t channels{};
    std::vector<std::uint8_t> pixels;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint8_t at(
        std::size_t x,
        std::size_t y,
        std::size_t channel
    ) const;
};

struct BoundingBox final {
    std::size_t x{};
    std::size_t y{};
    std::size_t width{};
    std::size_t height{};
    double confidence{};
};

struct VisualObservation final {
    std::vector<float> descriptor;
    std::vector<BoundingBox> regions;
    double foreground_fraction{};
    std::uint64_t content_hash{};
};

struct VideoFrameObservation final {
    std::size_t frame_index{};
    VisualObservation visual;
};

struct ObjectTrack final {
    std::uint64_t stable_id{};
    std::vector<std::size_t> frame_indices;
    std::vector<BoundingBox> boxes;
    double mean_dx{};
    double mean_dy{};
    double continuity{};
};

struct VideoObservation final {
    std::vector<VideoFrameObservation> frames;
    std::vector<ObjectTrack> tracks;
    std::vector<float> descriptor;
    std::uint64_t content_hash{};
};

struct AudioObservation final {
    std::uint32_t sample_rate{};
    std::size_t channels{};
    std::size_t frame_count{};
    std::vector<float> descriptor;
    double duration_seconds{};
    double rms{};
    double dominant_frequency_hz{};
    std::uint64_t content_hash{};
};

class ImageUnderstanding final {
public:
    [[nodiscard]] static Image load_pnm(const std::filesystem::path& path);
    static void save_ppm(const std::filesystem::path& path, const Image& image);
    [[nodiscard]] static VisualObservation analyze(const Image& image);
};

class VideoUnderstanding final {
public:
    [[nodiscard]] static VideoObservation analyze(
        std::span<const Image> frames
    );
    [[nodiscard]] static std::vector<Image> predict_next_frames(
        const VideoObservation& observation,
        std::size_t count
    );
};

class AudioUnderstanding final {
public:
    [[nodiscard]] static AudioObservation analyze_wav(
        const std::filesystem::path& path
    );
    static void synthesize_prototype_wav(
        const std::filesystem::path& path,
        const AudioObservation& observation,
        double duration_seconds = 1.0
    );
};

class PrototypeGenerator final {
public:
    static void generate_visual_mode(
        const std::filesystem::path& path,
        const ModeRecord& mode,
        std::size_t width = 96U,
        std::size_t height = 96U
    );
    static void generate_video_mode(
        const std::filesystem::path& directory,
        const ModeRecord& mode,
        std::size_t frame_count = 8U
    );
    static void generate_audio_mode(
        const std::filesystem::path& path,
        const ModeRecord& mode,
        std::uint32_t sample_rate = 16'000U
    );
};

[[nodiscard]] std::uint64_t hash_bytes(std::span<const std::uint8_t> bytes) noexcept;

}  // namespace rlf::frontier
