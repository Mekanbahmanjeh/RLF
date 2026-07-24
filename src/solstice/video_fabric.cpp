#include "rlf/solstice/video_fabric.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace rlf::solstice {
namespace {

constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;

void hash_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (unsigned int index = 0U; index < 8U; ++index) {
        hash ^= (value >> (index * 8U)) & 0xFFULL;
        hash *= fnv_prime;
    }
}

void hash_string(std::uint64_t& hash, const std::string_view value) noexcept {
    hash_u64(hash, value.size());
    for (const char character : value) {
        hash ^= static_cast<unsigned char>(character);
        hash *= fnv_prime;
    }
}

void hash_double(std::uint64_t& hash, const double value) noexcept {
    hash_u64(hash, std::bit_cast<std::uint64_t>(value));
}

[[nodiscard]] bool valid_image(const ImageData& image) noexcept {
    if (image.width == 0U || image.height == 0U ||
        image.width > std::numeric_limits<std::size_t>::max() / image.height) {
        return false;
    }
    const std::size_t pixels = image.width * image.height;
    return pixels <= std::numeric_limits<std::size_t>::max() / 3U &&
        image.rgb.size() == pixels * 3U;
}

[[nodiscard]] bool valid_config(const VideoFabricConfig& config) noexcept {
    return config.maximum_sequences > 0U &&
        config.maximum_frames_per_sequence >= 2U &&
        config.maximum_generation_frames > 0U &&
        config.maximum_prompt_bytes > 0U &&
        config.output_width > 0U && config.output_height > 0U &&
        config.output_width <= 4'096U && config.output_height <= 4'096U &&
        std::isfinite(config.minimum_prompt_similarity) &&
        config.minimum_prompt_similarity >= 0.0 &&
        config.minimum_prompt_similarity <= 1.0 &&
        std::isfinite(config.local_learning_rate) &&
        config.local_learning_rate > 0.0 && config.local_learning_rate <= 1.0;
}

[[nodiscard]] bool valid_motion(const VideoMotionDescriptor& motion) noexcept {
    const auto unit = [](const double value) {
        return std::isfinite(value) && value >= 0.0 && value <= 1.0;
    };
    return unit(motion.start_x) && unit(motion.start_y) &&
        std::isfinite(motion.velocity_x) && std::abs(motion.velocity_x) <= 1.0 &&
        std::isfinite(motion.velocity_y) && std::abs(motion.velocity_y) <= 1.0 &&
        unit(motion.object_width) && motion.object_width > 0.0 &&
        unit(motion.object_height) && motion.object_height > 0.0 &&
        std::all_of(motion.foreground_rgb.begin(), motion.foreground_rgb.end(), unit) &&
        std::all_of(motion.background_rgb.begin(), motion.background_rgb.end(), unit);
}

[[nodiscard]] std::uint8_t byte_color(const double value) noexcept {
    return static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0, 1.0) * 255.0));
}

struct FrameFeature final {
    double x{0.5};
    double y{0.5};
    double width{0.25};
    double height{0.25};
    std::array<double, 3U> foreground{};
    std::array<double, 3U> background{};
};

[[nodiscard]] FrameFeature describe_frame(const ImageData& image) {
    if (!valid_image(image)) {
        throw std::invalid_argument("video contains an invalid RGB frame");
    }
    std::array<double, 3U> border{};
    std::size_t border_count = 0U;
    for (std::size_t y = 0U; y < image.height; ++y) {
        for (std::size_t x = 0U; x < image.width; ++x) {
            if (x != 0U && y != 0U && x + 1U != image.width && y + 1U != image.height) {
                continue;
            }
            const std::size_t offset = (y * image.width + x) * 3U;
            for (std::size_t channel = 0U; channel < 3U; ++channel) {
                border[channel] += static_cast<double>(image.rgb[offset + channel]) / 255.0;
            }
            ++border_count;
        }
    }
    for (double& value : border) value /= static_cast<double>(border_count);

    double mean_distance = 0.0;
    for (std::size_t pixel = 0U; pixel < image.width * image.height; ++pixel) {
        double squared = 0.0;
        for (std::size_t channel = 0U; channel < 3U; ++channel) {
            const double value = static_cast<double>(image.rgb[pixel * 3U + channel]) / 255.0;
            const double difference = value - border[channel];
            squared += difference * difference;
        }
        mean_distance += std::sqrt(squared / 3.0);
    }
    mean_distance /= static_cast<double>(image.width * image.height);
    const double threshold = std::clamp(mean_distance * 1.35 + 0.035, 0.055, 0.45);

    std::size_t minimum_x = image.width;
    std::size_t minimum_y = image.height;
    std::size_t maximum_x = 0U;
    std::size_t maximum_y = 0U;
    std::size_t count = 0U;
    double sum_x = 0.0;
    double sum_y = 0.0;
    std::array<double, 3U> foreground{};
    for (std::size_t y = 0U; y < image.height; ++y) {
        for (std::size_t x = 0U; x < image.width; ++x) {
            const std::size_t offset = (y * image.width + x) * 3U;
            double squared = 0.0;
            for (std::size_t channel = 0U; channel < 3U; ++channel) {
                const double value = static_cast<double>(image.rgb[offset + channel]) / 255.0;
                const double difference = value - border[channel];
                squared += difference * difference;
            }
            if (std::sqrt(squared / 3.0) <= threshold) continue;
            minimum_x = std::min(minimum_x, x);
            minimum_y = std::min(minimum_y, y);
            maximum_x = std::max(maximum_x, x);
            maximum_y = std::max(maximum_y, y);
            sum_x += static_cast<double>(x) + 0.5;
            sum_y += static_cast<double>(y) + 0.5;
            for (std::size_t channel = 0U; channel < 3U; ++channel) {
                foreground[channel] += static_cast<double>(image.rgb[offset + channel]) / 255.0;
            }
            ++count;
        }
    }
    FrameFeature result;
    result.background = border;
    if (count == 0U) {
        result.foreground = border;
        result.width = 1.0;
        result.height = 1.0;
        return result;
    }
    result.x = sum_x / static_cast<double>(count * image.width);
    result.y = sum_y / static_cast<double>(count * image.height);
    result.width = static_cast<double>(maximum_x - minimum_x + 1U) /
        static_cast<double>(image.width);
    result.height = static_cast<double>(maximum_y - minimum_y + 1U) /
        static_cast<double>(image.height);
    for (double& value : foreground) value /= static_cast<double>(count);
    result.foreground = foreground;
    return result;
}

void blend(double& target, const double value, const double rate) noexcept {
    target += rate * (value - target);
}

}  // namespace

VideoPrototypeFabric::VideoPrototypeFabric(VideoFabricConfig config)
    : config_(std::move(config)) {
    if (!valid_config(config_)) {
        throw std::invalid_argument("invalid Solstice video prototype configuration");
    }
    prototypes_.reserve(std::min<std::size_t>(config_.maximum_sequences, 65'536U));
}

VideoMotionDescriptor VideoPrototypeFabric::describe(
    const std::span<const ImageData> frames
) {
    if (frames.size() < 2U) {
        throw std::invalid_argument("video prototype learning requires at least two frames");
    }
    std::vector<FrameFeature> features;
    features.reserve(frames.size());
    for (const ImageData& frame : frames) features.push_back(describe_frame(frame));
    VideoMotionDescriptor result;
    result.start_x = features.front().x;
    result.start_y = features.front().y;
    const double transitions = static_cast<double>(features.size() - 1U);
    result.velocity_x = (features.back().x - features.front().x) / transitions;
    result.velocity_y = (features.back().y - features.front().y) / transitions;
    result.object_width = 0.0;
    result.object_height = 0.0;
    result.foreground_rgb.fill(0.0);
    result.background_rgb.fill(0.0);
    for (const FrameFeature& feature : features) {
        result.object_width += feature.width;
        result.object_height += feature.height;
        for (std::size_t channel = 0U; channel < 3U; ++channel) {
            result.foreground_rgb[channel] += feature.foreground[channel];
            result.background_rgb[channel] += feature.background[channel];
        }
    }
    const double count = static_cast<double>(features.size());
    result.object_width = std::clamp(result.object_width / count, 1.0 / 4'096.0, 1.0);
    result.object_height = std::clamp(result.object_height / count, 1.0 / 4'096.0, 1.0);
    for (std::size_t channel = 0U; channel < 3U; ++channel) {
        result.foreground_rgb[channel] /= count;
        result.background_rgb[channel] /= count;
    }
    return result;
}

std::uint64_t VideoPrototypeFabric::train(
    const std::string_view sequence_id,
    const std::string_view prompt,
    const double frames_per_second,
    const std::span<const ImageData> frames
) {
    if (sequence_id.empty() || prompt.empty() || prompt.size() > config_.maximum_prompt_bytes ||
        !std::isfinite(frames_per_second) || frames_per_second <= 0.0 ||
        frames.size() > config_.maximum_frames_per_sequence) {
        throw std::invalid_argument("invalid video prototype training example");
    }
    const VideoMotionDescriptor motion = describe(frames);
    if (sequences_seen_ == std::numeric_limits<std::uint64_t>::max() ||
        frames.size() > std::numeric_limits<std::uint64_t>::max() - frames_seen_) {
        throw std::runtime_error("video training statistics overflow");
    }
    for (VideoPrototype& prototype : prototypes_) {
        if (prototype.source_sequence_id != sequence_id) continue;
        if (prototype.prompt != prompt) {
            throw std::invalid_argument("video sequence ID reused with a different prompt");
        }
        const double rate = config_.local_learning_rate /
            std::sqrt(static_cast<double>(prototype.support));
        if (prototype.support == std::numeric_limits<std::uint64_t>::max() ||
            frames.size() > std::numeric_limits<std::size_t>::max() -
                prototype.observed_frames) {
            throw std::runtime_error("video prototype support overflow");
        }
        blend(prototype.motion.start_x, motion.start_x, rate);
        blend(prototype.motion.start_y, motion.start_y, rate);
        blend(prototype.motion.velocity_x, motion.velocity_x, rate);
        blend(prototype.motion.velocity_y, motion.velocity_y, rate);
        blend(prototype.motion.object_width, motion.object_width, rate);
        blend(prototype.motion.object_height, motion.object_height, rate);
        for (std::size_t channel = 0U; channel < 3U; ++channel) {
            blend(prototype.motion.foreground_rgb[channel], motion.foreground_rgb[channel], rate);
            blend(prototype.motion.background_rgb[channel], motion.background_rgb[channel], rate);
        }
        blend(prototype.frames_per_second, frames_per_second, rate);
        prototype.observed_frames += frames.size();
        ++prototype.support;
        ++sequences_seen_;
        frames_seen_ += frames.size();
        return prototype.id;
    }
    if (prototypes_.size() >= config_.maximum_sequences) {
        throw std::runtime_error("Solstice video prototype capacity exceeded");
    }
    if (next_prototype_id_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::runtime_error("video prototype ID space exhausted");
    }
    const std::uint64_t id = next_prototype_id_++;
    prototypes_.push_back(VideoPrototype{
        id, std::string(sequence_id), std::string(prompt),
        GeneralInstructionFabric::make_signature(prompt), motion,
        frames_per_second, frames.size(), 1U,
    });
    ++sequences_seen_;
    frames_seen_ += frames.size();
    return id;
}

ImageData VideoPrototypeFabric::render(
    const VideoMotionDescriptor& motion,
    const std::size_t width,
    const std::size_t height,
    const std::size_t frame_index
) {
    ImageData image;
    image.width = width;
    image.height = height;
    if (width == 0U || height == 0U ||
        width > std::numeric_limits<std::size_t>::max() / height ||
        width * height > std::numeric_limits<std::size_t>::max() / 3U) {
        throw std::invalid_argument("video prototype render size overflows");
    }
    const std::size_t pixel_count = width * height;
    image.rgb.assign(pixel_count * 3U, std::uint8_t{0U});
    const std::array<std::uint8_t, 3U> background{
        byte_color(motion.background_rgb[0U]), byte_color(motion.background_rgb[1U]),
        byte_color(motion.background_rgb[2U]),
    };
    const std::array<std::uint8_t, 3U> foreground{
        byte_color(motion.foreground_rgb[0U]), byte_color(motion.foreground_rgb[1U]),
        byte_color(motion.foreground_rgb[2U]),
    };
    for (std::size_t pixel = 0U; pixel < pixel_count; ++pixel) {
        for (std::size_t channel = 0U; channel < 3U; ++channel) {
            image.rgb[pixel * 3U + channel] = background[channel];
        }
    }
    const double center_x = std::clamp(
        motion.start_x + motion.velocity_x * static_cast<double>(frame_index), 0.0, 1.0
    );
    const double center_y = std::clamp(
        motion.start_y + motion.velocity_y * static_cast<double>(frame_index), 0.0, 1.0
    );
    const double half_width = motion.object_width * 0.5;
    const double half_height = motion.object_height * 0.5;
    const std::size_t x0 = static_cast<std::size_t>(std::floor(
        std::clamp(center_x - half_width, 0.0, 1.0) * static_cast<double>(width)
    ));
    const std::size_t y0 = static_cast<std::size_t>(std::floor(
        std::clamp(center_y - half_height, 0.0, 1.0) * static_cast<double>(height)
    ));
    const std::size_t x1 = std::min(width, static_cast<std::size_t>(std::ceil(
        std::clamp(center_x + half_width, 0.0, 1.0) * static_cast<double>(width)
    )));
    const std::size_t y1 = std::min(height, static_cast<std::size_t>(std::ceil(
        std::clamp(center_y + half_height, 0.0, 1.0) * static_cast<double>(height)
    )));
    for (std::size_t y = y0; y < y1; ++y) {
        for (std::size_t x = x0; x < x1; ++x) {
            const std::size_t offset = (y * width + x) * 3U;
            for (std::size_t channel = 0U; channel < 3U; ++channel) {
                image.rgb[offset + channel] = foreground[channel];
            }
        }
    }
    return image;
}

VideoGeneration VideoPrototypeFabric::generate(
    const std::string_view prompt,
    const std::size_t frame_count
) const {
    if (prompt.empty() || prompt.size() > config_.maximum_prompt_bytes ||
        frame_count == 0U || frame_count > config_.maximum_generation_frames) {
        throw std::invalid_argument("invalid video prototype generation request");
    }
    if (prototypes_.empty()) {
        throw std::runtime_error("no learned video prototypes are available");
    }
    const SemanticSignature signature = GeneralInstructionFabric::make_signature(prompt);
    const VideoPrototype* best = nullptr;
    double best_similarity = -1.0;
    for (const VideoPrototype& prototype : prototypes_) {
        const double similarity = GeneralInstructionFabric::signature_similarity(
            signature, prototype.prompt_signature
        );
        if (similarity > best_similarity ||
            (similarity == best_similarity && best != nullptr && prototype.id < best->id)) {
            best = &prototype;
            best_similarity = similarity;
        }
    }
    if (best == nullptr || best_similarity < config_.minimum_prompt_similarity) {
        throw std::runtime_error("no video prototype meets the prompt-similarity threshold");
    }
    VideoGeneration result;
    result.prototype_id = best->id;
    result.source_sequence_id = best->source_sequence_id;
    result.prompt_similarity = best_similarity;
    result.confidence = std::clamp(
        best_similarity * (0.75 + 0.25 * std::min(1.0, std::log2(
            static_cast<double>(best->support) + 1.0
        ) / 4.0)), 0.0, 1.0
    );
    result.frames_per_second = best->frames_per_second;
    result.motion = best->motion;
    result.frames.reserve(frame_count);
    for (std::size_t index = 0U; index < frame_count; ++index) {
        result.frames.push_back(render(
            best->motion, config_.output_width, config_.output_height, index
        ));
    }
    return result;
}

void VideoPrototypeFabric::save_ppm(
    const std::filesystem::path& path,
    const ImageData& image
) {
    if (!valid_image(image)) throw std::invalid_argument("cannot save invalid RGB frame");
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("unable to create prototype frame: " + path.string());
    output << "P6\n" << image.width << ' ' << image.height << "\n255\n";
    output.write(
        reinterpret_cast<const char*>(image.rgb.data()),
        static_cast<std::streamsize>(image.rgb.size())
    );
    if (!output) throw std::runtime_error("failed while writing prototype frame");
}

const VideoFabricConfig& VideoPrototypeFabric::config() const noexcept { return config_; }
std::span<const VideoPrototype> VideoPrototypeFabric::prototypes() const noexcept {
    return prototypes_;
}
std::uint64_t VideoPrototypeFabric::sequences_seen() const noexcept { return sequences_seen_; }
std::uint64_t VideoPrototypeFabric::frames_seen() const noexcept { return frames_seen_; }
bool VideoPrototypeFabric::contains_source_sequence(
    const std::string_view sequence_id
) const noexcept {
    return std::any_of(prototypes_.begin(), prototypes_.end(), [sequence_id](const auto& item) {
        return item.source_sequence_id == sequence_id;
    });
}

std::uint64_t VideoPrototypeFabric::deterministic_hash() const noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, next_prototype_id_);
    hash_u64(hash, sequences_seen_);
    hash_u64(hash, frames_seen_);
    for (const VideoPrototype& prototype : prototypes_) {
        hash_u64(hash, prototype.id);
        hash_string(hash, prototype.source_sequence_id);
        hash_string(hash, prototype.prompt);
        for (const std::uint64_t word : prototype.prompt_signature.words) hash_u64(hash, word);
        hash_double(hash, prototype.motion.start_x);
        hash_double(hash, prototype.motion.start_y);
        hash_double(hash, prototype.motion.velocity_x);
        hash_double(hash, prototype.motion.velocity_y);
        hash_double(hash, prototype.motion.object_width);
        hash_double(hash, prototype.motion.object_height);
        for (const double value : prototype.motion.foreground_rgb) hash_double(hash, value);
        for (const double value : prototype.motion.background_rgb) hash_double(hash, value);
        hash_double(hash, prototype.frames_per_second);
        hash_u64(hash, prototype.observed_frames);
        hash_u64(hash, prototype.support);
    }
    return hash;
}

VideoFabricSnapshot VideoPrototypeFabric::snapshot() const {
    return {config_, next_prototype_id_, sequences_seen_, frames_seen_, prototypes_};
}

VideoPrototypeFabric VideoPrototypeFabric::from_snapshot(VideoFabricSnapshot snapshot) {
    VideoPrototypeFabric fabric(snapshot.config);
    if (snapshot.next_prototype_id == 0U || snapshot.prototypes.size() > snapshot.config.maximum_sequences) {
        throw std::invalid_argument("invalid Solstice video prototype snapshot");
    }
    std::uint64_t maximum_id = 0U;
    for (const VideoPrototype& prototype : snapshot.prototypes) {
        if (prototype.id == 0U || prototype.source_sequence_id.empty() || prototype.prompt.empty() ||
            prototype.prompt.size() > snapshot.config.maximum_prompt_bytes ||
            prototype.observed_frames < 2U || prototype.support == 0U ||
            !std::isfinite(prototype.frames_per_second) || prototype.frames_per_second <= 0.0 ||
            !valid_motion(prototype.motion) ||
            prototype.prompt_signature != GeneralInstructionFabric::make_signature(prototype.prompt)) {
            throw std::invalid_argument("invalid Solstice video prototype record");
        }
        maximum_id = std::max(maximum_id, prototype.id);
    }
    if (snapshot.next_prototype_id <= maximum_id) {
        throw std::invalid_argument("invalid Solstice video prototype next ID");
    }
    fabric.next_prototype_id_ = snapshot.next_prototype_id;
    fabric.sequences_seen_ = snapshot.sequences_seen;
    fabric.frames_seen_ = snapshot.frames_seen;
    fabric.prototypes_ = std::move(snapshot.prototypes);
    return fabric;
}

}  // namespace rlf::solstice
