#include "rlf/frontier/multimodal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numbers>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>

namespace rlf::frontier {
namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

[[nodiscard]] std::string read_pnm_token(std::istream& input) {
    std::string token;
    while (true) {
        int next = input.peek();
        if (next == std::char_traits<char>::eof()) break;
        if (std::isspace(static_cast<unsigned char>(next)) != 0) {
            static_cast<void>(input.get());
            continue;
        }
        if (next == '#') {
            std::string ignored;
            std::getline(input, ignored);
            continue;
        }
        break;
    }
    input >> token;
    return token;
}

[[nodiscard]] std::size_t parse_size_token(
    const std::string& token,
    const char* description
) {
    try {
        std::size_t consumed = 0U;
        const unsigned long long value = std::stoull(token, &consumed, 10);
        if (consumed != token.size() || value == 0ULL ||
            value > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error("invalid value");
        }
        return static_cast<std::size_t>(value);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid PNM ") + description);
    }
}

[[nodiscard]] std::uint16_t read_u16(const std::uint8_t* bytes) noexcept {
    return static_cast<std::uint16_t>(bytes[0]) |
        static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[1]) << 8U);
}

[[nodiscard]] std::uint32_t read_u32(const std::uint8_t* bytes) noexcept {
    return static_cast<std::uint32_t>(bytes[0]) |
        (static_cast<std::uint32_t>(bytes[1]) << 8U) |
        (static_cast<std::uint32_t>(bytes[2]) << 16U) |
        (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

void write_u16(std::ostream& output, const std::uint16_t value) {
    const std::array<char, 2U> bytes{
        static_cast<char>(value & 0xFFU),
        static_cast<char>((value >> 8U) & 0xFFU),
    };
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void write_u32(std::ostream& output, const std::uint32_t value) {
    const std::array<char, 4U> bytes{
        static_cast<char>(value & 0xFFU),
        static_cast<char>((value >> 8U) & 0xFFU),
        static_cast<char>((value >> 16U) & 0xFFU),
        static_cast<char>((value >> 24U) & 0xFFU),
    };
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

[[nodiscard]] std::array<double, 3U> pixel_rgb(
    const Image& image,
    const std::size_t x,
    const std::size_t y
) {
    if (image.channels == 1U) {
        const double value = static_cast<double>(image.at(x, y, 0U)) / 255.0;
        return {value, value, value};
    }
    return {
        static_cast<double>(image.at(x, y, 0U)) / 255.0,
        static_cast<double>(image.at(x, y, 1U)) / 255.0,
        static_cast<double>(image.at(x, y, 2U)) / 255.0,
    };
}

[[nodiscard]] double color_distance(
    const std::array<double, 3U>& left,
    const std::array<double, 3U>& right
) noexcept {
    const double red = left[0] - right[0];
    const double green = left[1] - right[1];
    const double blue = left[2] - right[2];
    return std::sqrt(red * red + green * green + blue * blue);
}

[[nodiscard]] std::vector<BoundingBox> connected_regions(
    const Image& image,
    const std::vector<bool>& foreground
) {
    const std::size_t pixel_count = image.width * image.height;
    std::vector<bool> visited(pixel_count, false);
    std::vector<BoundingBox> regions;
    constexpr std::array<int, 4U> dx{-1, 1, 0, 0};
    constexpr std::array<int, 4U> dy{0, 0, -1, 1};
    const std::size_t minimum_area = std::max<std::size_t>(4U, pixel_count / 500U);
    for (std::size_t start = 0U; start < pixel_count; ++start) {
        if (!foreground[start] || visited[start]) continue;
        std::queue<std::size_t> pending;
        pending.push(start);
        visited[start] = true;
        std::size_t min_x = start % image.width;
        std::size_t max_x = min_x;
        std::size_t min_y = start / image.width;
        std::size_t max_y = min_y;
        std::size_t area = 0U;
        while (!pending.empty()) {
            const std::size_t current = pending.front();
            pending.pop();
            ++area;
            const std::size_t x = current % image.width;
            const std::size_t y = current / image.width;
            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
            min_y = std::min(min_y, y);
            max_y = std::max(max_y, y);
            for (std::size_t direction = 0U; direction < dx.size(); ++direction) {
                const int nx = static_cast<int>(x) + dx[direction];
                const int ny = static_cast<int>(y) + dy[direction];
                if (nx < 0 || ny < 0 ||
                    nx >= static_cast<int>(image.width) ||
                    ny >= static_cast<int>(image.height)) continue;
                const std::size_t neighbor =
                    static_cast<std::size_t>(ny) * image.width +
                    static_cast<std::size_t>(nx);
                if (foreground[neighbor] && !visited[neighbor]) {
                    visited[neighbor] = true;
                    pending.push(neighbor);
                }
            }
        }
        if (area >= minimum_area) {
            const double fraction = static_cast<double>(area) /
                static_cast<double>(pixel_count);
            regions.push_back({
                .x = min_x,
                .y = min_y,
                .width = max_x - min_x + 1U,
                .height = max_y - min_y + 1U,
                .confidence = std::clamp(0.5 + fraction * 4.0, 0.0, 1.0),
            });
        }
    }
    std::sort(regions.begin(), regions.end(), [](const BoundingBox& left, const BoundingBox& right) {
        return left.width * left.height > right.width * right.height;
    });
    if (regions.size() > 32U) regions.resize(32U);
    return regions;
}

[[nodiscard]] std::array<double, 3U> border_mean(const Image& image) {
    std::array<double, 3U> sum{};
    std::size_t count = 0U;
    for (std::size_t y = 0U; y < image.height; ++y) {
        for (std::size_t x = 0U; x < image.width; ++x) {
            if (x != 0U && y != 0U && x + 1U != image.width && y + 1U != image.height) continue;
            const auto color = pixel_rgb(image, x, y);
            for (std::size_t channel = 0U; channel < 3U; ++channel) sum[channel] += color[channel];
            ++count;
        }
    }
    if (count == 0U) return {};
    for (double& value : sum) value /= static_cast<double>(count);
    return sum;
}

[[nodiscard]] std::vector<float> visual_descriptor(
    const Image& image,
    const std::vector<BoundingBox>& regions,
    const double foreground_fraction
) {
    std::vector<float> descriptor;
    descriptor.reserve(64U);
    std::array<double, 3U> mean{};
    std::array<double, 3U> square{};
    for (std::size_t y = 0U; y < image.height; ++y) {
        for (std::size_t x = 0U; x < image.width; ++x) {
            const auto color = pixel_rgb(image, x, y);
            for (std::size_t channel = 0U; channel < 3U; ++channel) {
                mean[channel] += color[channel];
                square[channel] += color[channel] * color[channel];
            }
        }
    }
    const double count = static_cast<double>(image.width * image.height);
    for (std::size_t channel = 0U; channel < 3U; ++channel) {
        mean[channel] /= count;
        square[channel] = std::sqrt(std::max(0.0, square[channel] / count - mean[channel] * mean[channel]));
        descriptor.push_back(static_cast<float>(mean[channel]));
    }
    for (const double value : square) descriptor.push_back(static_cast<float>(value));

    for (std::size_t grid_y = 0U; grid_y < 4U; ++grid_y) {
        for (std::size_t grid_x = 0U; grid_x < 4U; ++grid_x) {
            const std::size_t start_x = grid_x * image.width / 4U;
            const std::size_t end_x = (grid_x + 1U) * image.width / 4U;
            const std::size_t start_y = grid_y * image.height / 4U;
            const std::size_t end_y = (grid_y + 1U) * image.height / 4U;
            std::array<double, 3U> cell{};
            std::size_t cell_count = 0U;
            for (std::size_t y = start_y; y < end_y; ++y) {
                for (std::size_t x = start_x; x < end_x; ++x) {
                    const auto color = pixel_rgb(image, x, y);
                    for (std::size_t channel = 0U; channel < 3U; ++channel) cell[channel] += color[channel];
                    ++cell_count;
                }
            }
            for (double& value : cell) value /= static_cast<double>(std::max<std::size_t>(1U, cell_count));
            for (const double value : cell) descriptor.push_back(static_cast<float>(value));
        }
    }

    double horizontal_gradient = 0.0;
    double vertical_gradient = 0.0;
    for (std::size_t y = 0U; y + 1U < image.height; ++y) {
        for (std::size_t x = 0U; x + 1U < image.width; ++x) {
            const auto current = pixel_rgb(image, x, y);
            horizontal_gradient += color_distance(current, pixel_rgb(image, x + 1U, y));
            vertical_gradient += color_distance(current, pixel_rgb(image, x, y + 1U));
        }
    }
    const double gradient_count = static_cast<double>(
        std::max<std::size_t>(1U, (image.width - 1U) * (image.height - 1U))
    );
    descriptor.push_back(static_cast<float>(horizontal_gradient / gradient_count));
    descriptor.push_back(static_cast<float>(vertical_gradient / gradient_count));
    descriptor.push_back(static_cast<float>(foreground_fraction));
    descriptor.push_back(static_cast<float>(std::min<std::size_t>(regions.size(), 16U)) / 16.0F);
    if (regions.empty()) {
        descriptor.insert(descriptor.end(), 4U, 0.0F);
    } else {
        const BoundingBox& box = regions.front();
        descriptor.push_back(static_cast<float>(box.x) / static_cast<float>(image.width));
        descriptor.push_back(static_cast<float>(box.y) / static_cast<float>(image.height));
        descriptor.push_back(static_cast<float>(box.width) / static_cast<float>(image.width));
        descriptor.push_back(static_cast<float>(box.height) / static_cast<float>(image.height));
    }
    while (descriptor.size() < 64U) descriptor.push_back(0.0F);
    if (descriptor.size() > 64U) descriptor.resize(64U);
    return descriptor;
}

[[nodiscard]] double box_center_x(const BoundingBox& box) noexcept {
    return static_cast<double>(box.x) + static_cast<double>(box.width) * 0.5;
}

[[nodiscard]] double box_center_y(const BoundingBox& box) noexcept {
    return static_cast<double>(box.y) + static_cast<double>(box.height) * 0.5;
}

[[nodiscard]] Image draw_motion_frame(
    const std::size_t width,
    const std::size_t height,
    const double normalized_x,
    const double normalized_y,
    const std::array<std::uint8_t, 3U>& color
) {
    Image image{
        .width = width,
        .height = height,
        .channels = 3U,
        .pixels = std::vector<std::uint8_t>(width * height * 3U, 18U),
    };
    const std::size_t box_width = std::max<std::size_t>(4U, width / 5U);
    const std::size_t box_height = std::max<std::size_t>(4U, height / 5U);
    const std::size_t max_x = width > box_width ? width - box_width : 0U;
    const std::size_t max_y = height > box_height ? height - box_height : 0U;
    const std::size_t start_x = static_cast<std::size_t>(
        std::clamp(normalized_x, 0.0, 1.0) * static_cast<double>(max_x)
    );
    const std::size_t start_y = static_cast<std::size_t>(
        std::clamp(normalized_y, 0.0, 1.0) * static_cast<double>(max_y)
    );
    for (std::size_t y = start_y; y < std::min(height, start_y + box_height); ++y) {
        for (std::size_t x = start_x; x < std::min(width, start_x + box_width); ++x) {
            const std::size_t index = (y * width + x) * 3U;
            image.pixels[index] = color[0];
            image.pixels[index + 1U] = color[1];
            image.pixels[index + 2U] = color[2];
        }
    }
    return image;
}

}  // namespace

std::uint64_t hash_bytes(const std::span<const std::uint8_t> bytes) noexcept {
    std::uint64_t hash = fnv_offset;
    for (const std::uint8_t byte : bytes) {
        hash ^= byte;
        hash *= fnv_prime;
    }
    return hash;
}

bool Image::valid() const noexcept {
    if (width == 0U || height == 0U || (channels != 1U && channels != 3U)) return false;
    if (width > std::numeric_limits<std::size_t>::max() / height) return false;
    const std::size_t pixels_count = width * height;
    if (pixels_count > std::numeric_limits<std::size_t>::max() / channels) return false;
    return pixels.size() == pixels_count * channels;
}

std::uint8_t Image::at(
    const std::size_t x,
    const std::size_t y,
    const std::size_t channel
) const {
    if (!valid() || x >= width || y >= height || channel >= channels) {
        throw std::out_of_range("image pixel coordinate is out of range");
    }
    return pixels[(y * width + x) * channels + channel];
}

Image ImageUnderstanding::load_pnm(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("unable to open PNM image: " + path.string());
    const std::string magic = read_pnm_token(input);
    if (magic != "P5" && magic != "P6") {
        throw std::runtime_error("only binary P5/P6 PNM images are supported");
    }
    const std::size_t width = parse_size_token(read_pnm_token(input), "width");
    const std::size_t height = parse_size_token(read_pnm_token(input), "height");
    const std::size_t maximum = parse_size_token(read_pnm_token(input), "maximum value");
    if (maximum != 255U) throw std::runtime_error("PNM maximum value must be 255");
    static_cast<void>(input.get());
    const std::size_t channels = magic == "P6" ? 3U : 1U;
    if (width > 100'000U || height > 100'000U || width > std::numeric_limits<std::size_t>::max() / height) {
        throw std::runtime_error("PNM dimensions are too large");
    }
    const std::size_t pixel_count = width * height;
    if (pixel_count > std::numeric_limits<std::size_t>::max() / channels) {
        throw std::runtime_error("PNM byte count overflows");
    }
    Image image{.width = width, .height = height, .channels = channels, .pixels = {}};
    image.pixels.resize(pixel_count * channels);
    input.read(
        reinterpret_cast<char*>(image.pixels.data()),
        static_cast<std::streamsize>(image.pixels.size())
    );
    if (input.gcount() != static_cast<std::streamsize>(image.pixels.size())) {
        throw std::runtime_error("PNM image is truncated");
    }
    return image;
}

void ImageUnderstanding::save_ppm(
    const std::filesystem::path& path,
    const Image& image
) {
    if (!image.valid()) throw std::invalid_argument("cannot save invalid image");
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("unable to create PPM image: " + path.string());
    output << "P6\n" << image.width << ' ' << image.height << "\n255\n";
    if (image.channels == 3U) {
        output.write(
            reinterpret_cast<const char*>(image.pixels.data()),
            static_cast<std::streamsize>(image.pixels.size())
        );
    } else {
        for (const std::uint8_t value : image.pixels) {
            const std::array<char, 3U> rgb{
                static_cast<char>(value),
                static_cast<char>(value),
                static_cast<char>(value),
            };
            output.write(rgb.data(), static_cast<std::streamsize>(rgb.size()));
        }
    }
    if (!output) throw std::runtime_error("failed while writing PPM image");
}

VisualObservation ImageUnderstanding::analyze(const Image& image) {
    if (!image.valid()) throw std::invalid_argument("cannot analyze invalid image");
    const auto background = border_mean(image);
    std::vector<bool> foreground(image.width * image.height, false);
    std::size_t foreground_count = 0U;
    double mean_distance = 0.0;
    for (std::size_t y = 0U; y < image.height; ++y) {
        for (std::size_t x = 0U; x < image.width; ++x) {
            mean_distance += color_distance(pixel_rgb(image, x, y), background);
        }
    }
    mean_distance /= static_cast<double>(image.width * image.height);
    const double threshold = std::clamp(mean_distance * 1.35 + 0.05, 0.08, 0.45);
    for (std::size_t y = 0U; y < image.height; ++y) {
        for (std::size_t x = 0U; x < image.width; ++x) {
            const std::size_t index = y * image.width + x;
            foreground[index] = color_distance(pixel_rgb(image, x, y), background) > threshold;
            if (foreground[index]) ++foreground_count;
        }
    }
    std::vector<BoundingBox> regions = connected_regions(image, foreground);
    const double fraction = static_cast<double>(foreground_count) /
        static_cast<double>(image.width * image.height);
    return {
        .descriptor = visual_descriptor(image, regions, fraction),
        .regions = std::move(regions),
        .foreground_fraction = fraction,
        .content_hash = hash_bytes(image.pixels),
    };
}

VideoObservation VideoUnderstanding::analyze(const std::span<const Image> frames) {
    if (frames.empty()) throw std::invalid_argument("video requires at least one frame");
    VideoObservation result;
    result.frames.reserve(frames.size());
    std::uint64_t combined_hash = fnv_offset;
    for (std::size_t index = 0U; index < frames.size(); ++index) {
        VisualObservation visual = ImageUnderstanding::analyze(frames[index]);
        combined_hash ^= visual.content_hash;
        combined_hash *= fnv_prime;
        result.frames.push_back({.frame_index = index, .visual = std::move(visual)});
    }

    std::uint64_t next_track_id = 1U;
    std::vector<bool> used;
    for (std::size_t frame_index = 0U; frame_index < result.frames.size(); ++frame_index) {
        const auto& regions = result.frames[frame_index].visual.regions;
        used.assign(regions.size(), false);
        if (frame_index == 0U) {
            for (std::size_t region_index = 0U; region_index < regions.size(); ++region_index) {
                result.tracks.push_back({
                    .stable_id = next_track_id++,
                    .frame_indices = {frame_index},
                    .boxes = {regions[region_index]},
                });
            }
            continue;
        }
        for (auto& track : result.tracks) {
            if (track.frame_indices.empty() || track.frame_indices.back() + 1U != frame_index) continue;
            const BoundingBox& previous = track.boxes.back();
            double best_distance = std::numeric_limits<double>::infinity();
            std::size_t best_index = regions.size();
            for (std::size_t region_index = 0U; region_index < regions.size(); ++region_index) {
                if (used[region_index]) continue;
                const double x_distance = box_center_x(regions[region_index]) - box_center_x(previous);
                const double y_distance = box_center_y(regions[region_index]) - box_center_y(previous);
                const double distance = std::sqrt(x_distance * x_distance + y_distance * y_distance);
                if (distance < best_distance) {
                    best_distance = distance;
                    best_index = region_index;
                }
            }
            const double maximum_distance = 0.35 * static_cast<double>(
                std::max(frames[frame_index].width, frames[frame_index].height)
            );
            if (best_index < regions.size() && best_distance <= maximum_distance) {
                used[best_index] = true;
                track.frame_indices.push_back(frame_index);
                track.boxes.push_back(regions[best_index]);
            }
        }
        for (std::size_t region_index = 0U; region_index < regions.size(); ++region_index) {
            if (!used[region_index]) {
                result.tracks.push_back({
                    .stable_id = next_track_id++,
                    .frame_indices = {frame_index},
                    .boxes = {regions[region_index]},
                });
            }
        }
    }

    double aggregate_dx = 0.0;
    double aggregate_dy = 0.0;
    double aggregate_continuity = 0.0;
    for (auto& track : result.tracks) {
        if (track.boxes.size() > 1U) {
            double dx = 0.0;
            double dy = 0.0;
            for (std::size_t index = 1U; index < track.boxes.size(); ++index) {
                dx += box_center_x(track.boxes[index]) - box_center_x(track.boxes[index - 1U]);
                dy += box_center_y(track.boxes[index]) - box_center_y(track.boxes[index - 1U]);
            }
            const double transitions = static_cast<double>(track.boxes.size() - 1U);
            track.mean_dx = dx / transitions;
            track.mean_dy = dy / transitions;
        }
        track.continuity = static_cast<double>(track.frame_indices.size()) /
            static_cast<double>(frames.size());
        aggregate_dx += track.mean_dx * track.continuity;
        aggregate_dy += track.mean_dy * track.continuity;
        aggregate_continuity += track.continuity;
    }
    const double track_count = static_cast<double>(std::max<std::size_t>(1U, result.tracks.size()));
    result.descriptor = {
        static_cast<float>(aggregate_dx / track_count),
        static_cast<float>(aggregate_dy / track_count),
        static_cast<float>(aggregate_continuity / track_count),
        static_cast<float>(result.tracks.size()),
        static_cast<float>(frames.size()),
    };
    if (!result.frames.empty()) {
        const auto& first = result.frames.front().visual.descriptor;
        const auto& last = result.frames.back().visual.descriptor;
        const std::size_t compare = std::min<std::size_t>(16U, std::min(first.size(), last.size()));
        for (std::size_t index = 0U; index < compare; ++index) {
            result.descriptor.push_back(last[index] - first[index]);
        }
    }
    while (result.descriptor.size() < 32U) result.descriptor.push_back(0.0F);
    result.descriptor.resize(32U);
    result.content_hash = combined_hash;
    return result;
}

std::vector<Image> VideoUnderstanding::predict_next_frames(
    const VideoObservation& observation,
    const std::size_t count
) {
    if (observation.frames.empty() || count == 0U) return {};
    const Image& last_source = observation.frames.back().visual.regions.empty()
        ? Image{}
        : Image{};
    static_cast<void>(last_source);
    double x = 0.4;
    double y = 0.4;
    double dx = observation.descriptor.empty() ? 0.0 : static_cast<double>(observation.descriptor[0]);
    double dy = observation.descriptor.size() < 2U ? 0.0 : static_cast<double>(observation.descriptor[1]);
    if (!observation.tracks.empty() && !observation.tracks.front().boxes.empty()) {
        const BoundingBox& box = observation.tracks.front().boxes.back();
        x = box_center_x(box) / 96.0;
        y = box_center_y(box) / 96.0;
        dx /= 96.0;
        dy /= 96.0;
    }
    std::vector<Image> frames;
    frames.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        x += dx;
        y += dy;
        frames.push_back(draw_motion_frame(96U, 96U, x, y, {64U, 210U, 150U}));
    }
    return frames;
}

AudioObservation AudioUnderstanding::analyze_wav(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("unable to open WAV file: " + path.string());
    std::vector<std::uint8_t> bytes(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>()
    );
    if (bytes.size() < 44U || std::memcmp(bytes.data(), "RIFF", 4U) != 0 ||
        std::memcmp(bytes.data() + 8U, "WAVE", 4U) != 0) {
        throw std::runtime_error("invalid WAV header");
    }
    std::size_t offset = 12U;
    std::uint16_t format = 0U;
    std::uint16_t channels = 0U;
    std::uint32_t sample_rate = 0U;
    std::uint16_t bits_per_sample = 0U;
    std::span<const std::uint8_t> data;
    while (offset + 8U <= bytes.size()) {
        const char* chunk_id = reinterpret_cast<const char*>(bytes.data() + offset);
        const std::uint32_t chunk_size = read_u32(bytes.data() + offset + 4U);
        offset += 8U;
        if (offset + chunk_size > bytes.size()) throw std::runtime_error("WAV chunk is truncated");
        if (std::memcmp(chunk_id, "fmt ", 4U) == 0) {
            if (chunk_size < 16U) throw std::runtime_error("WAV format chunk is too small");
            format = read_u16(bytes.data() + offset);
            channels = read_u16(bytes.data() + offset + 2U);
            sample_rate = read_u32(bytes.data() + offset + 4U);
            bits_per_sample = read_u16(bytes.data() + offset + 14U);
        } else if (std::memcmp(chunk_id, "data", 4U) == 0) {
            data = std::span<const std::uint8_t>(bytes.data() + offset, chunk_size);
        }
        offset += chunk_size + (chunk_size & 1U);
    }
    if (format != 1U || channels == 0U || sample_rate == 0U || bits_per_sample != 16U || data.empty()) {
        throw std::runtime_error("only PCM16 WAV audio is supported");
    }
    const std::size_t sample_count = data.size() / 2U;
    if (sample_count % channels != 0U) throw std::runtime_error("WAV sample data is misaligned");
    std::vector<double> mono(sample_count / channels, 0.0);
    for (std::size_t frame = 0U; frame < mono.size(); ++frame) {
        double value = 0.0;
        for (std::size_t channel = 0U; channel < channels; ++channel) {
            const std::size_t byte_index = (frame * channels + channel) * 2U;
            const std::uint16_t raw = read_u16(data.data() + byte_index);
            const std::int16_t signed_value = static_cast<std::int16_t>(raw);
            value += static_cast<double>(signed_value) / 32768.0;
        }
        mono[frame] = value / static_cast<double>(channels);
    }
    double square_sum = 0.0;
    std::size_t zero_crossings = 0U;
    double peak = 0.0;
    for (std::size_t index = 0U; index < mono.size(); ++index) {
        square_sum += mono[index] * mono[index];
        peak = std::max(peak, std::abs(mono[index]));
        if (index != 0U && (mono[index] >= 0.0) != (mono[index - 1U] >= 0.0)) ++zero_crossings;
    }
    const double rms = std::sqrt(square_sum / static_cast<double>(mono.size()));
    const std::size_t analysis_count = std::min<std::size_t>(mono.size(), 4'096U);
    constexpr std::size_t band_count = 24U;
    std::array<double, band_count> bands{};
    std::size_t dominant_band = 0U;
    for (std::size_t band = 0U; band < band_count; ++band) {
        const double frequency = 50.0 + static_cast<double>(band) *
            (static_cast<double>(sample_rate) * 0.45 - 50.0) /
            static_cast<double>(band_count - 1U);
        double real = 0.0;
        double imaginary = 0.0;
        for (std::size_t index = 0U; index < analysis_count; ++index) {
            const double phase = 2.0 * std::numbers::pi_v<double> * frequency *
                static_cast<double>(index) / static_cast<double>(sample_rate);
            real += mono[index] * std::cos(phase);
            imaginary -= mono[index] * std::sin(phase);
        }
        bands[band] = std::sqrt(real * real + imaginary * imaginary) /
            static_cast<double>(std::max<std::size_t>(1U, analysis_count));
        if (bands[band] > bands[dominant_band]) dominant_band = band;
    }
    const double dominant_frequency = 50.0 + static_cast<double>(dominant_band) *
        (static_cast<double>(sample_rate) * 0.45 - 50.0) /
        static_cast<double>(band_count - 1U);
    std::vector<float> descriptor;
    descriptor.reserve(32U);
    descriptor.push_back(static_cast<float>(rms));
    descriptor.push_back(static_cast<float>(peak));
    descriptor.push_back(static_cast<float>(zero_crossings) /
        static_cast<float>(std::max<std::size_t>(1U, mono.size())));
    descriptor.push_back(static_cast<float>(dominant_frequency / static_cast<double>(sample_rate)));
    for (const double value : bands) descriptor.push_back(static_cast<float>(value));
    descriptor.push_back(static_cast<float>(mono.size()) / static_cast<float>(sample_rate));
    while (descriptor.size() < 32U) descriptor.push_back(0.0F);
    descriptor.resize(32U);
    return {
        .sample_rate = sample_rate,
        .channels = channels,
        .frame_count = mono.size(),
        .descriptor = std::move(descriptor),
        .duration_seconds = static_cast<double>(mono.size()) / static_cast<double>(sample_rate),
        .rms = rms,
        .dominant_frequency_hz = dominant_frequency,
        .content_hash = hash_bytes(bytes),
    };
}

void AudioUnderstanding::synthesize_prototype_wav(
    const std::filesystem::path& path,
    const AudioObservation& observation,
    const double duration_seconds
) {
    if (observation.sample_rate == 0U || !std::isfinite(duration_seconds) || duration_seconds <= 0.0) {
        throw std::invalid_argument("invalid audio synthesis configuration");
    }
    const std::size_t sample_count = static_cast<std::size_t>(
        std::round(duration_seconds * static_cast<double>(observation.sample_rate))
    );
    if (sample_count > std::numeric_limits<std::uint32_t>::max() / 2U) {
        throw std::invalid_argument("audio synthesis request is too large");
    }
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("unable to create WAV file: " + path.string());
    const std::uint32_t data_bytes = static_cast<std::uint32_t>(sample_count * 2U);
    output.write("RIFF", 4);
    write_u32(output, 36U + data_bytes);
    output.write("WAVEfmt ", 8);
    write_u32(output, 16U);
    write_u16(output, 1U);
    write_u16(output, 1U);
    write_u32(output, observation.sample_rate);
    write_u32(output, observation.sample_rate * 2U);
    write_u16(output, 2U);
    write_u16(output, 16U);
    output.write("data", 4);
    write_u32(output, data_bytes);
    const double frequency = std::clamp(
        observation.dominant_frequency_hz,
        20.0,
        static_cast<double>(observation.sample_rate) * 0.45
    );
    const double amplitude = std::clamp(observation.rms * 1.414, 0.05, 0.8);
    for (std::size_t index = 0U; index < sample_count; ++index) {
        const double envelope = std::min(1.0, static_cast<double>(index) / 256.0) *
            std::min(1.0, static_cast<double>(sample_count - index) / 256.0);
        const double sample = amplitude * envelope * std::sin(
            2.0 * std::numbers::pi_v<double> * frequency *
            static_cast<double>(index) / static_cast<double>(observation.sample_rate)
        );
        const auto integer = static_cast<std::int16_t>(
            std::clamp(sample, -1.0, 1.0) * 32767.0
        );
        write_u16(output, static_cast<std::uint16_t>(integer));
    }
    if (!output) throw std::runtime_error("failed while writing WAV prototype");
}

void PrototypeGenerator::generate_visual_mode(
    const std::filesystem::path& path,
    const ModeRecord& mode,
    const std::size_t width,
    const std::size_t height
) {
    if (mode.modality != Modality::image || mode.prototype.empty()) {
        throw std::invalid_argument("visual generation requires an image mode");
    }
    const auto component = [&mode](const std::size_t index, const float fallback) {
        return index < mode.prototype.size() ? std::clamp(mode.prototype[index], 0.0F, 1.0F) : fallback;
    };
    Image image{
        .width = width,
        .height = height,
        .channels = 3U,
        .pixels = std::vector<std::uint8_t>(width * height * 3U),
    };
    for (std::size_t y = 0U; y < height; ++y) {
        for (std::size_t x = 0U; x < width; ++x) {
            const std::size_t grid_x = std::min<std::size_t>(3U, x * 4U / width);
            const std::size_t grid_y = std::min<std::size_t>(3U, y * 4U / height);
            const std::size_t base = 6U + (grid_y * 4U + grid_x) * 3U;
            const std::array<float, 3U> values{
                component(base, component(0U, 0.3F)),
                component(base + 1U, component(1U, 0.3F)),
                component(base + 2U, component(2U, 0.3F)),
            };
            const std::size_t pixel = (y * width + x) * 3U;
            for (std::size_t channel = 0U; channel < 3U; ++channel) {
                image.pixels[pixel + channel] = static_cast<std::uint8_t>(
                    std::round(values[channel] * 255.0F)
                );
            }
        }
    }
    ImageUnderstanding::save_ppm(path, image);
}

void PrototypeGenerator::generate_video_mode(
    const std::filesystem::path& directory,
    const ModeRecord& mode,
    const std::size_t frame_count
) {
    if (mode.modality != Modality::video || mode.prototype.size() < 2U) {
        throw std::invalid_argument("video generation requires a video mode");
    }
    std::filesystem::create_directories(directory);
    double x = 0.35;
    double y = 0.35;
    const double dx = std::clamp(static_cast<double>(mode.prototype[0]) / 96.0, -0.15, 0.15);
    const double dy = std::clamp(static_cast<double>(mode.prototype[1]) / 96.0, -0.15, 0.15);
    for (std::size_t index = 0U; index < frame_count; ++index) {
        Image image = draw_motion_frame(96U, 96U, x, y, {230U, 110U, 70U});
        ImageUnderstanding::save_ppm(
            directory / ("frame_" + std::to_string(index) + ".ppm"),
            image
        );
        x += dx;
        y += dy;
    }
}

void PrototypeGenerator::generate_audio_mode(
    const std::filesystem::path& path,
    const ModeRecord& mode,
    const std::uint32_t sample_rate
) {
    if (mode.modality != Modality::audio || mode.prototype.empty()) {
        throw std::invalid_argument("audio generation requires an audio mode");
    }
    AudioObservation observation;
    observation.sample_rate = sample_rate;
    observation.rms = mode.prototype.empty() ? 0.2 :
        std::clamp(static_cast<double>(mode.prototype[0]), 0.05, 0.8);
    observation.dominant_frequency_hz = mode.prototype.size() > 3U
        ? std::clamp(static_cast<double>(mode.prototype[3]) * static_cast<double>(sample_rate), 40.0, 4'000.0)
        : 440.0;
    AudioUnderstanding::synthesize_prototype_wav(path, observation, 1.0);
}

}  // namespace rlf::frontier
