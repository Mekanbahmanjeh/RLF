#include "rlf/solstice/image_generation_fabric.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
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

[[nodiscard]] std::uint64_t stable_string_hash(
    const std::string_view value
) noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_string(hash, value);
    return hash;
}

[[nodiscard]] std::uint64_t mix64(std::uint64_t value) noexcept {
    value ^= value >> 30U;
    value *= 0xBF58476D1CE4E5B9ULL;
    value ^= value >> 27U;
    value *= 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

void atomic_saturating_add(
    std::uint64_t& counter,
    const std::uint64_t increment = 1U
) noexcept {
    std::atomic_ref<std::uint64_t> atomic_counter(counter);
    std::uint64_t current = atomic_counter.load(std::memory_order_relaxed);
    while (current != std::numeric_limits<std::uint64_t>::max()) {
        const std::uint64_t desired = increment >
                std::numeric_limits<std::uint64_t>::max() - current
            ? std::numeric_limits<std::uint64_t>::max()
            : current + increment;
        if (atomic_counter.compare_exchange_weak(
                current,
                desired,
                std::memory_order_relaxed,
                std::memory_order_relaxed
            )) {
            return;
        }
    }
}

[[nodiscard]] std::uint64_t atomic_load(std::uint64_t& counter) noexcept {
    return std::atomic_ref<std::uint64_t>(counter).load(std::memory_order_relaxed);
}

void validate_config(const ImageGenerationConfig& config) {
    if (config.tile_size == 0U || config.coordinate_bins == 0U ||
        config.coordinate_bins > std::numeric_limits<std::uint16_t>::max() ||
        config.maximum_source_images == 0U ||
        config.maximum_tile_prototypes == 0U ||
        config.maximum_source_side == 0U || config.maximum_source_pixels == 0U ||
        config.maximum_caption_bytes == 0U ||
        config.maximum_caption_concepts == 0U ||
        config.maximum_total_caption_bytes == 0U ||
        config.maximum_total_concept_bytes == 0U ||
        config.maximum_posting_entries == 0U ||
        config.maximum_caption_bytes > config.maximum_total_caption_bytes ||
        config.maximum_candidates_per_cell == 0U ||
        config.default_output_width == 0U || config.default_output_height == 0U ||
        config.maximum_output_side == 0U || config.maximum_output_pixels == 0U ||
        config.default_output_width > config.maximum_output_side ||
        config.default_output_height > config.maximum_output_side ||
        config.default_output_width > config.maximum_output_pixels /
            config.default_output_height ||
        !std::isfinite(config.semantic_weight) || config.semantic_weight < 0.0 ||
        !std::isfinite(config.spatial_weight) || config.spatial_weight < 0.0 ||
        !std::isfinite(config.seam_weight) || config.seam_weight < 0.0 ||
        !std::isfinite(config.support_weight) || config.support_weight < 0.0) {
        throw std::invalid_argument("invalid image-generation configuration");
    }
    if (config.tile_size > std::numeric_limits<std::size_t>::max() /
            config.tile_size ||
        config.tile_size * config.tile_size >
            std::numeric_limits<std::size_t>::max() / 3U ||
        config.maximum_source_side >
            std::numeric_limits<std::size_t>::max() / config.coordinate_bins ||
        config.maximum_output_side >
            std::numeric_limits<std::size_t>::max() / config.coordinate_bins) {
        throw std::invalid_argument("image-generation tile size is too large");
    }
}

void validate_image(const ImageData& image) {
    if (image.width == 0U || image.height == 0U ||
        image.width > std::numeric_limits<std::size_t>::max() / image.height ||
        image.width * image.height > std::numeric_limits<std::size_t>::max() / 3U ||
        image.rgb.size() != image.width * image.height * 3U) {
        throw std::invalid_argument("invalid image-generation training image");
    }
}

[[nodiscard]] std::array<float, 12U> tile_descriptor(
    const std::span<const std::uint8_t> rgb,
    const std::size_t side
) {
    std::array<double, 3U> sums{};
    std::array<double, 3U> squares{};
    std::array<double, 3U> horizontal{};
    std::array<double, 3U> vertical{};
    for (std::size_t y = 0U; y < side; ++y) {
        for (std::size_t x = 0U; x < side; ++x) {
            const std::size_t offset = (y * side + x) * 3U;
            for (std::size_t channel = 0U; channel < 3U; ++channel) {
                const double value = static_cast<double>(rgb[offset + channel]) / 255.0;
                sums[channel] += value;
                squares[channel] += value * value;
                if (x != 0U) {
                    horizontal[channel] += std::abs(
                        static_cast<double>(rgb[offset + channel]) -
                        static_cast<double>(rgb[offset + channel - 3U])
                    ) / 255.0;
                }
                if (y != 0U) {
                    vertical[channel] += std::abs(
                        static_cast<double>(rgb[offset + channel]) -
                        static_cast<double>(rgb[offset + channel - side * 3U])
                    ) / 255.0;
                }
            }
        }
    }
    const double pixels = static_cast<double>(side * side);
    const double horizontal_edges = static_cast<double>(side * (side - 1U));
    const double vertical_edges = static_cast<double>((side - 1U) * side);
    std::array<float, 12U> descriptor{};
    for (std::size_t channel = 0U; channel < 3U; ++channel) {
        const double mean = sums[channel] / pixels;
        descriptor[channel] = static_cast<float>(mean);
        descriptor[3U + channel] = static_cast<float>(std::sqrt(std::max(
            0.0,
            squares[channel] / pixels - mean * mean
        )));
        descriptor[6U + channel] = static_cast<float>(
            horizontal_edges == 0.0 ? 0.0 : horizontal[channel] / horizontal_edges
        );
        descriptor[9U + channel] = static_cast<float>(
            vertical_edges == 0.0 ? 0.0 : vertical[channel] / vertical_edges
        );
    }
    return descriptor;
}

[[nodiscard]] bool same_config(
    const ImageGenerationConfig& left,
    const ImageGenerationConfig& right
) noexcept {
    return left.tile_size == right.tile_size &&
        left.coordinate_bins == right.coordinate_bins &&
        left.maximum_source_images == right.maximum_source_images &&
        left.maximum_tile_prototypes == right.maximum_tile_prototypes &&
        left.maximum_source_side == right.maximum_source_side &&
        left.maximum_source_pixels == right.maximum_source_pixels &&
        left.maximum_caption_bytes == right.maximum_caption_bytes &&
        left.maximum_caption_concepts == right.maximum_caption_concepts &&
        left.maximum_total_caption_bytes == right.maximum_total_caption_bytes &&
        left.maximum_total_concept_bytes == right.maximum_total_concept_bytes &&
        left.maximum_posting_entries == right.maximum_posting_entries &&
        left.maximum_candidates_per_cell == right.maximum_candidates_per_cell &&
        left.default_output_width == right.default_output_width &&
        left.default_output_height == right.default_output_height &&
        left.maximum_output_side == right.maximum_output_side &&
        left.maximum_output_pixels == right.maximum_output_pixels &&
        left.semantic_weight == right.semantic_weight &&
        left.spatial_weight == right.spatial_weight &&
        left.seam_weight == right.seam_weight &&
        left.support_weight == right.support_weight;
}

template <typename Value>
void reserve_geometrically(
    std::vector<Value>& values,
    const std::size_t required,
    const std::size_t maximum
) {
    if (required <= values.capacity()) {
        return;
    }
    const std::size_t capacity = values.capacity();
    const std::size_t increment = std::max(std::size_t{1U}, capacity / 2U);
    const std::size_t grown = capacity > maximum - std::min(increment, maximum)
        ? maximum
        : capacity + increment;
    values.reserve(std::min(maximum, std::max(required, grown)));
}

}  // namespace

ImageGenerationConfig make_image_generation_profile_config(
    const ImageGenerationProfile profile
) {
    ImageGenerationConfig config;
    if (profile == ImageGenerationProfile::a100_80g ||
        profile == ImageGenerationProfile::v100_32g) {
        config.tile_size = 16U;
        config.coordinate_bins = 32U;
        config.maximum_source_images = 4'000'000U;
        config.maximum_tile_prototypes = 48'000'000U;
        config.maximum_source_side = 16'384U;
        config.maximum_source_pixels = 128U * 1024U * 1024U;
        config.maximum_caption_bytes = 16'384U;
        config.maximum_caption_concepts = 48U;
        config.maximum_total_caption_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
        config.maximum_total_concept_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
        config.maximum_posting_entries = 512'000'000ULL;
        config.maximum_candidates_per_cell = 8'192U;
        config.default_output_width = 1'024U;
        config.default_output_height = 1'024U;
        config.maximum_output_side = 4'096U;
        config.maximum_output_pixels = 16U * 1024U * 1024U;
    }
    return config;
}

ImageGenerationCapacity estimate_image_generation_capacity(
    const ImageGenerationProfile profile
) noexcept {
    constexpr std::uint64_t gib = 1024ULL * 1024ULL * 1024ULL;
    if (profile == ImageGenerationProfile::a100_80g) {
        return {72ULL * gib, 76ULL * gib, 512ULL * gib, 512ULL * gib};
    }
    if (profile == ImageGenerationProfile::v100_32g) {
        // Logical model and quality settings are identical to the A100
        // profile. Only the bounded device-resident working set changes;
        // host-backed prototype storage is not represented as VRAM capacity.
        return {30ULL * gib, 30ULL * gib, 512ULL * gib, 512ULL * gib};
    }
    return {4ULL * gib, 6ULL * gib, 16ULL * gib, 32ULL * gib};
}

PatchQuiltScaleProjection project_patch_quilt_scale(
    const ImageGenerationConfig& config,
    const std::size_t width,
    const std::size_t height
) {
    validate_config(config);
    if (width == 0U || height == 0U || width > config.maximum_source_side ||
        height > config.maximum_source_side ||
        width > config.maximum_source_pixels / height ||
        width * height > config.maximum_source_pixels) {
        throw std::invalid_argument("patch-quilt projection dimensions exceed profile");
    }
    const std::uint64_t columns = 1U + (width - 1U) / config.tile_size;
    const std::uint64_t rows = 1U + (height - 1U) / config.tile_size;
    if (columns > std::numeric_limits<std::uint64_t>::max() / rows) {
        throw std::overflow_error("patch-quilt projection tile count overflow");
    }
    const std::uint64_t tiles_per_image = columns * rows;
    const std::uint64_t by_tiles = config.maximum_tile_prototypes / tiles_per_image;
    const std::uint64_t by_sources = config.maximum_source_images;
    const std::uint64_t tile_bytes = static_cast<std::uint64_t>(config.tile_size) *
        static_cast<std::uint64_t>(config.tile_size) * 3ULL;
    if (static_cast<std::uint64_t>(config.maximum_tile_prototypes) >
        std::numeric_limits<std::uint64_t>::max() / tile_bytes) {
        throw std::overflow_error("patch-quilt projection byte count overflow");
    }
    return {
        width,
        height,
        tiles_per_image,
        by_tiles,
        by_sources,
        std::min(by_tiles, by_sources),
        static_cast<std::uint64_t>(config.maximum_tile_prototypes) * tile_bytes,
    };
}

bool image_generation_profile_config_matches(
    const ImageGenerationProfile profile,
    const ImageGenerationConfig& config
) noexcept {
    try {
        return same_config(config, make_image_generation_profile_config(profile));
    } catch (...) {
        return false;
    }
}

std::string_view to_string(const ImageGenerationProfile profile) noexcept {
    switch (profile) {
        case ImageGenerationProfile::reference:
            return "imagegen-reference";
        case ImageGenerationProfile::a100_80g:
            return "imagegen-a100-80g";
        case ImageGenerationProfile::v100_32g:
            return "imagegen-v100-32g";
    }
    return "unknown";
}

std::string_view to_string(
    const ImageGenerationArchitecture architecture
) noexcept {
    switch (architecture) {
        case ImageGenerationArchitecture::patch_quilt_baseline:
            return "patch-quilt-baseline";
        case ImageGenerationArchitecture::resonant_fabric:
            return "resonant-fabric";
    }
    return "unknown";
}

ImageGenerationProfile parse_image_generation_profile(const std::string_view value) {
    if (value == "imagegen-reference" || value == "reference") {
        return ImageGenerationProfile::reference;
    }
    if (value == "imagegen-a100-80g" || value == "a100-80g") {
        return ImageGenerationProfile::a100_80g;
    }
    if (value == "imagegen-v100-32g" || value == "v100-32g") {
        return ImageGenerationProfile::v100_32g;
    }
    throw std::invalid_argument("unknown image-generation profile");
}

PatchQuiltBaseline::PatchQuiltBaseline(ImageGenerationConfig config)
    : config_(std::move(config)) {
    validate_config(config_);
}

std::vector<std::string> PatchQuiltBaseline::caption_concepts(
    const std::string_view caption,
    const std::size_t maximum_concepts
) {
    std::vector<std::string> concepts;
    std::string current;
    const auto flush = [&]() {
        if (current.empty() || concepts.size() >= maximum_concepts) {
            current.clear();
            return;
        }
        if (std::find(concepts.begin(), concepts.end(), current) == concepts.end()) {
            concepts.push_back(current);
        }
        current.clear();
    };
    for (const char raw_byte : caption) {
        const auto byte = static_cast<unsigned char>(raw_byte);
        const bool ascii_alphanumeric =
            (byte >= static_cast<unsigned char>('a') &&
             byte <= static_cast<unsigned char>('z')) ||
            (byte >= static_cast<unsigned char>('A') &&
             byte <= static_cast<unsigned char>('Z')) ||
            (byte >= static_cast<unsigned char>('0') &&
             byte <= static_cast<unsigned char>('9')) || byte >= 0x80U;
        if (ascii_alphanumeric) {
            current.push_back(static_cast<char>(
                byte >= static_cast<unsigned char>('A') &&
                        byte <= static_cast<unsigned char>('Z')
                    ? byte + static_cast<unsigned char>('a' - 'A')
                    : byte
            ));
        } else {
            flush();
        }
    }
    flush();
    if (concepts.empty()) {
        concepts.emplace_back("<unlabeled>");
    }
    return concepts;
}

std::uint64_t PatchQuiltBaseline::bucket_key(
    const std::uint64_t concept_hash,
    const std::uint16_t x_bin,
    const std::uint16_t y_bin
) const noexcept {
    std::uint64_t key = mix64(concept_hash ^ 0x494D475F4255434BULL);
    key ^= static_cast<std::uint64_t>(x_bin) << 16U;
    key ^= static_cast<std::uint64_t>(y_bin);
    return mix64(key);
}

void PatchQuiltBaseline::index_tile(const std::size_t tile_index) {
    const ImageTilePrototype& tile = tiles_[tile_index];
    const auto source_iterator = source_id_index_.find(tile.source_id);
    if (source_iterator == source_id_index_.end()) {
        throw std::logic_error("image-generation tile references missing source");
    }
    tile_buckets_[bucket_key(0U, tile.x_bin, tile.y_bin)].push_back(tile_index);
    for (const std::string& concept_name : sources_[source_iterator->second].concepts) {
        tile_buckets_[bucket_key(stable_string_hash(concept_name), tile.x_bin, tile.y_bin)]
            .push_back(tile_index);
    }
}

void PatchQuiltBaseline::rebuild_indices() {
    source_id_index_.clear();
    tile_buckets_.clear();
    stored_caption_bytes_ = 0U;
    stored_concept_bytes_ = 0U;
    posting_entries_ = 0U;
    source_id_index_.reserve(sources_.size());
    for (std::size_t index = 0U; index < sources_.size(); ++index) {
        const ImageGenerationSource& source = sources_[index];
        if (source.caption.size() > config_.maximum_total_caption_bytes -
                stored_caption_bytes_) {
            throw std::invalid_argument("image-generation caption budget exceeded");
        }
        stored_caption_bytes_ += source.caption.size();
        for (const std::string& concept_name : source.concepts) {
            if (concept_name.size() > config_.maximum_total_concept_bytes -
                    stored_concept_bytes_) {
                throw std::invalid_argument("image-generation concept budget exceeded");
            }
            stored_concept_bytes_ += concept_name.size();
        }
        const std::uint64_t entries_per_tile =
            static_cast<std::uint64_t>(source.concepts.size()) + 1U;
        if (source.tile_count >
            (config_.maximum_posting_entries - posting_entries_) /
                entries_per_tile) {
            throw std::invalid_argument("image-generation posting budget exceeded");
        }
        posting_entries_ += source.tile_count * entries_per_tile;
        if (!source_id_index_.emplace(sources_[index].id, index).second) {
            throw std::invalid_argument("duplicate image-generation source ID");
        }
    }
    for (std::size_t index = 0U; index < tiles_.size(); ++index) {
        index_tile(index);
    }
}

void PatchQuiltBaseline::train(
    const ImageData& image,
    const std::string_view caption
) {
    atomic_saturating_add(operation_stats_.training_calls);
    validate_image(image);
    if (image.width > config_.maximum_source_side ||
        image.height > config_.maximum_source_side ||
        image.width * image.height > config_.maximum_source_pixels) {
        throw std::length_error("image-generation source dimensions exceed profile");
    }
    if (caption.size() > config_.maximum_caption_bytes) {
        throw std::length_error("image-generation caption exceeds profile");
    }
    if (sources_.size() >= config_.maximum_source_images) {
        atomic_saturating_add(operation_stats_.source_capacity_rejections);
        throw std::length_error("image-generation source capacity exhausted");
    }
    const std::size_t columns = 1U + (image.width - 1U) / config_.tile_size;
    const std::size_t rows = 1U + (image.height - 1U) / config_.tile_size;
    if (columns > std::numeric_limits<std::size_t>::max() / rows) {
        atomic_saturating_add(operation_stats_.tile_capacity_rejections);
        throw std::length_error("image-generation tile count overflow");
    }
    const std::size_t tile_count = columns * rows;
    if (tile_count > config_.maximum_tile_prototypes - tiles_.size()) {
        atomic_saturating_add(operation_stats_.tile_capacity_rejections);
        throw std::length_error("image-generation tile capacity exhausted");
    }
    if (next_source_id_ == std::numeric_limits<std::uint64_t>::max() ||
        static_cast<std::uint64_t>(tile_count) >
            std::numeric_limits<std::uint64_t>::max() - next_tile_id_) {
        throw std::length_error("image-generation stable ID space exhausted");
    }

    std::vector<std::string> concepts = caption_concepts(
        caption,
        config_.maximum_caption_concepts
    );
    std::uint64_t concept_bytes = 0U;
    for (const std::string& concept_name : concepts) {
        concept_bytes += concept_name.size();
    }
    if (caption.size() > config_.maximum_total_caption_bytes -
            stored_caption_bytes_ ||
        concept_bytes > config_.maximum_total_concept_bytes -
            stored_concept_bytes_) {
        atomic_saturating_add(operation_stats_.string_budget_rejections);
        throw std::length_error("image-generation string storage budget exhausted");
    }
    const std::uint64_t entries_per_tile =
        static_cast<std::uint64_t>(concepts.size()) + 1U;
    if (static_cast<std::uint64_t>(tile_count) >
        (config_.maximum_posting_entries - posting_entries_) /
            entries_per_tile) {
        atomic_saturating_add(operation_stats_.posting_budget_rejections);
        throw std::length_error("image-generation posting budget exhausted");
    }
    const std::uint64_t new_posting_entries =
        static_cast<std::uint64_t>(tile_count) * entries_per_tile;

    const std::uint64_t source_id = next_source_id_;
    std::vector<ImageTilePrototype> pending;
    pending.reserve(tile_count);
    const std::size_t tile_bytes = config_.tile_size * config_.tile_size * 3U;
    for (std::size_t grid_y = 0U; grid_y < rows; ++grid_y) {
        for (std::size_t grid_x = 0U; grid_x < columns; ++grid_x) {
            ImageTilePrototype tile;
            tile.id = next_tile_id_ + pending.size();
            tile.source_id = source_id;
            tile.x_bin = static_cast<std::uint16_t>(std::min(
                config_.coordinate_bins - 1U,
                (grid_x * config_.tile_size * config_.coordinate_bins) /
                    image.width
            ));
            tile.y_bin = static_cast<std::uint16_t>(std::min(
                config_.coordinate_bins - 1U,
                (grid_y * config_.tile_size * config_.coordinate_bins) /
                    image.height
            ));
            std::vector<std::uint8_t> tile_rgb(tile_bytes, std::uint8_t{0U});
            for (std::size_t y = 0U; y < config_.tile_size; ++y) {
                const std::size_t source_y = std::min(
                    image.height - 1U,
                    grid_y * config_.tile_size + y
                );
                for (std::size_t x = 0U; x < config_.tile_size; ++x) {
                    const std::size_t source_x = std::min(
                        image.width - 1U,
                        grid_x * config_.tile_size + x
                    );
                    const std::size_t source_offset =
                        (source_y * image.width + source_x) * 3U;
                    const std::size_t tile_offset =
                        (y * config_.tile_size + x) * 3U;
                    std::copy_n(
                        image.rgb.begin() + static_cast<std::ptrdiff_t>(source_offset),
                        3U,
                        tile_rgb.begin() + static_cast<std::ptrdiff_t>(tile_offset)
                    );
                }
            }
            tile.descriptor = tile_descriptor(tile_rgb, config_.tile_size);
            tile.rgb = std::move(tile_rgb);
            pending.push_back(std::move(tile));
        }
    }

    ImageGenerationSource source;
    source.id = source_id;
    source.caption.assign(caption);
    source.concepts = std::move(concepts);
    source.width = image.width;
    source.height = image.height;
    source.first_tile = tiles_.size();
    source.tile_count = pending.size();

    std::unordered_map<std::uint64_t, std::size_t> bucket_additions;
    for (const ImageTilePrototype& tile : pending) {
        ++bucket_additions[bucket_key(0U, tile.x_bin, tile.y_bin)];
        for (const std::string& concept_name : source.concepts) {
            ++bucket_additions[bucket_key(
                stable_string_hash(concept_name),
                tile.x_bin,
                tile.y_bin
            )];
        }
    }

    reserve_geometrically(
        sources_,
        sources_.size() + 1U,
        config_.maximum_source_images
    );
    reserve_geometrically(
        tiles_,
        tiles_.size() + pending.size(),
        config_.maximum_tile_prototypes
    );

    if (source_id_index_.size() == source_id_index_.max_size() ||
        bucket_additions.size() > tile_buckets_.max_size() - tile_buckets_.size()) {
        throw std::length_error("image-generation derived index capacity exhausted");
    }
    source_id_index_.reserve(source_id_index_.size() + 1U);
    tile_buckets_.reserve(tile_buckets_.size() + bucket_additions.size());
    std::vector<std::uint64_t> inserted_bucket_keys;
    inserted_bucket_keys.reserve(bucket_additions.size());
    bool source_index_inserted = false;
    try {
        source_index_inserted = source_id_index_.emplace(
            source_id,
            sources_.size()
        ).second;
        if (!source_index_inserted) {
            throw std::logic_error("duplicate image-generation source ID");
        }
        for (const auto& [key, additions] : bucket_additions) {
            auto [iterator, inserted] = tile_buckets_.try_emplace(key);
            if (inserted) {
                inserted_bucket_keys.push_back(key);
            }
            if (additions > iterator->second.max_size() - iterator->second.size()) {
                throw std::length_error(
                    "image-generation bucket index capacity exhausted"
                );
            }
            iterator->second.reserve(iterator->second.size() + additions);
        }
    } catch (...) {
        for (const std::uint64_t key : inserted_bucket_keys) {
            tile_buckets_.erase(key);
        }
        if (source_index_inserted) {
            source_id_index_.erase(source_id);
        }
        throw;
    }

    const std::size_t previous_source_count = sources_.size();
    const std::size_t previous_tile_count = tiles_.size();
    try {
        sources_.push_back(std::move(source));
        for (ImageTilePrototype& tile : pending) {
            tiles_.push_back(std::move(tile));
        }
    } catch (...) {
        sources_.resize(previous_source_count);
        tiles_.resize(previous_tile_count);
        source_id_index_.erase(source_id);
        for (const std::uint64_t key : inserted_bucket_keys) {
            tile_buckets_.erase(key);
        }
        throw;
    }
    for (std::size_t tile_index = previous_tile_count;
         tile_index < tiles_.size();
         ++tile_index) {
        index_tile(tile_index);
    }
    ++next_source_id_;
    next_tile_id_ += tile_count;
    ++images_seen_;
    stored_caption_bytes_ += caption.size();
    stored_concept_bytes_ += concept_bytes;
    posting_entries_ += new_posting_entries;
    atomic_saturating_add(operation_stats_.source_images_inserted);
    atomic_saturating_add(
        operation_stats_.tile_prototypes_inserted,
        static_cast<std::uint64_t>(tile_count)
    );
}

GeneratedImage PatchQuiltBaseline::generate(
    const ImageGenerationRequest& request
) const {
    if (tiles_.empty()) {
        throw std::runtime_error("cannot generate without image tile prototypes");
    }
    const std::size_t width = request.width == 0U
        ? config_.default_output_width
        : request.width;
    const std::size_t height = request.height == 0U
        ? config_.default_output_height
        : request.height;
    if (width == 0U || height == 0U || width > config_.maximum_output_side ||
        height > config_.maximum_output_side ||
        width > config_.maximum_output_pixels / height ||
        width * height > config_.maximum_output_pixels ||
        width * height > std::numeric_limits<std::size_t>::max() / 3U) {
        throw std::invalid_argument("image-generation output dimensions exceed profile");
    }
    atomic_saturating_add(operation_stats_.generation_calls);
    const std::vector<std::string> prompt_concepts = caption_concepts(
        request.prompt,
        config_.maximum_caption_concepts
    );
    const std::size_t columns = 1U + (width - 1U) / config_.tile_size;
    const std::size_t rows = 1U + (height - 1U) / config_.tile_size;
    GeneratedImage result;
    result.image.width = width;
    result.image.height = height;
    result.image.rgb.assign(width * height * 3U, 0U);
    result.selected_prototype_ids.reserve(columns * rows);
    double semantic_sum = 0.0;
    double seam_sum = 0.0;

    for (std::size_t grid_y = 0U; grid_y < rows; ++grid_y) {
        for (std::size_t grid_x = 0U; grid_x < columns; ++grid_x) {
            const std::size_t cell_x = grid_x * config_.tile_size;
            const std::size_t cell_y = grid_y * config_.tile_size;
            const std::uint16_t x_bin = static_cast<std::uint16_t>(std::min(
                config_.coordinate_bins - 1U,
                (cell_x * config_.coordinate_bins) / width
            ));
            const std::uint16_t y_bin = static_cast<std::uint16_t>(std::min(
                config_.coordinate_bins - 1U,
                (cell_y * config_.coordinate_bins) / height
            ));
            std::vector<std::size_t> candidates;
            candidates.reserve(config_.maximum_candidates_per_cell);
            std::unordered_set<std::size_t> seen;
            seen.reserve(config_.maximum_candidates_per_cell);
            const std::size_t per_concept_budget = std::max(
                std::size_t{1U},
                config_.maximum_candidates_per_cell / prompt_concepts.size()
            );
            const auto append_bucket = [&](
                const std::uint64_t key,
                const std::size_t bucket_budget
            ) {
                atomic_saturating_add(operation_stats_.candidate_bucket_lookups);
                const auto iterator = tile_buckets_.find(key);
                if (iterator == tile_buckets_.end()) {
                    return;
                }
                const std::vector<std::size_t>& bucket = iterator->second;
                const std::size_t remaining =
                    config_.maximum_candidates_per_cell - candidates.size();
                const std::size_t take = std::min({remaining, bucket_budget, bucket.size()});
                if (take == 0U) {
                    return;
                }
                const std::uint64_t cell =
                    static_cast<std::uint64_t>(grid_y * columns + grid_x);
                const std::size_t offset = static_cast<std::size_t>(
                    mix64(request.seed ^ key ^ cell) % bucket.size()
                );
                for (std::size_t sample = 0U; sample < take; ++sample) {
                    const std::size_t position =
                        (offset + (sample * bucket.size()) / take) % bucket.size();
                    const std::size_t candidate = bucket[position];
                    if (seen.insert(candidate).second) {
                        candidates.push_back(candidate);
                    }
                }
            };
            for (std::size_t radius = 0U;
                 radius <= 2U && candidates.size() < config_.maximum_candidates_per_cell;
                 ++radius) {
                const std::size_t minimum_x = x_bin > radius ? x_bin - radius : 0U;
                const std::size_t minimum_y = y_bin > radius ? y_bin - radius : 0U;
                const std::size_t maximum_x = std::min(
                    config_.coordinate_bins - 1U,
                    static_cast<std::size_t>(x_bin) + radius
                );
                const std::size_t maximum_y = std::min(
                    config_.coordinate_bins - 1U,
                    static_cast<std::size_t>(y_bin) + radius
                );
                for (const std::string& concept_name : prompt_concepts) {
                    for (std::size_t bucket_y = minimum_y; bucket_y <= maximum_y; ++bucket_y) {
                        for (std::size_t bucket_x = minimum_x; bucket_x <= maximum_x; ++bucket_x) {
                            if (radius != 0U && bucket_x > minimum_x && bucket_x < maximum_x &&
                                bucket_y > minimum_y && bucket_y < maximum_y) {
                                continue;
                            }
                            append_bucket(
                                bucket_key(
                                    stable_string_hash(concept_name),
                                    static_cast<std::uint16_t>(bucket_x),
                                    static_cast<std::uint16_t>(bucket_y)
                                ),
                                per_concept_budget
                            );
                        }
                    }
                }
                if (!candidates.empty()) {
                    break;
                }
            }
            if (candidates.empty()) {
                ++result.fallback_cells;
                atomic_saturating_add(operation_stats_.fallback_cells);
                append_bucket(
                    bucket_key(0U, x_bin, y_bin),
                    config_.maximum_candidates_per_cell
                );
            }
            if (candidates.empty()) {
                candidates.push_back(
                    (grid_y * columns + grid_x) % tiles_.size()
                );
            }

            std::size_t best_index = candidates.front();
            double best_score = -std::numeric_limits<double>::infinity();
            double best_semantic = 0.0;
            double best_seam = 0.0;
            for (const std::size_t candidate_index : candidates) {
                const ImageTilePrototype& tile = tiles_[candidate_index];
                const auto source_iterator = source_id_index_.find(tile.source_id);
                if (source_iterator == source_id_index_.end()) {
                    throw std::logic_error("image-generation index references missing source");
                }
                const ImageGenerationSource& source = sources_[source_iterator->second];
                std::size_t shared = 0U;
                for (const std::string& concept_name : prompt_concepts) {
                    if (std::find(source.concepts.begin(), source.concepts.end(), concept_name) !=
                        source.concepts.end()) {
                        ++shared;
                    }
                }
                const double semantic = static_cast<double>(shared) /
                    static_cast<double>(prompt_concepts.size());
                const double delta_x = std::abs(
                    static_cast<double>(tile.x_bin) - static_cast<double>(x_bin)
                ) / static_cast<double>(config_.coordinate_bins);
                const double delta_y = std::abs(
                    static_cast<double>(tile.y_bin) - static_cast<double>(y_bin)
                ) / static_cast<double>(config_.coordinate_bins);
                const double spatial = std::max(0.0, 1.0 - (delta_x + delta_y) * 0.5);
                double seam_total = 0.0;
                std::size_t seam_samples = 0U;
                if (cell_x != 0U) {
                    const std::size_t copy_height = std::min(config_.tile_size, height - cell_y);
                    for (std::size_t y = 0U; y < copy_height; ++y) {
                        const std::size_t output_offset =
                            ((cell_y + y) * width + cell_x - 1U) * 3U;
                        const std::size_t tile_offset = y * config_.tile_size * 3U;
                        for (std::size_t channel = 0U; channel < 3U; ++channel) {
                            seam_total += 1.0 - std::abs(
                                static_cast<double>(result.image.rgb[output_offset + channel]) -
                                static_cast<double>(tile.rgb[tile_offset + channel])
                            ) / 255.0;
                            ++seam_samples;
                        }
                    }
                }
                if (cell_y != 0U) {
                    const std::size_t copy_width = std::min(config_.tile_size, width - cell_x);
                    for (std::size_t x = 0U; x < copy_width; ++x) {
                        const std::size_t output_offset =
                            ((cell_y - 1U) * width + cell_x + x) * 3U;
                        const std::size_t tile_offset = x * 3U;
                        for (std::size_t channel = 0U; channel < 3U; ++channel) {
                            seam_total += 1.0 - std::abs(
                                static_cast<double>(result.image.rgb[output_offset + channel]) -
                                static_cast<double>(tile.rgb[tile_offset + channel])
                            ) / 255.0;
                            ++seam_samples;
                        }
                    }
                }
                const double seam = seam_samples == 0U
                    ? 0.5
                    : seam_total / static_cast<double>(seam_samples);
                const double support = std::min(
                    1.0,
                    std::log1p(static_cast<double>(tile.support)) / std::log(1024.0)
                );
                const std::uint64_t jitter_hash = mix64(
                    request.seed ^ stable_string_hash(request.prompt) ^ tile.id ^
                    (static_cast<std::uint64_t>(grid_y * columns + grid_x) << 32U)
                );
                const double jitter = static_cast<double>(jitter_hash & 0xFFFFULL) /
                    65'535.0 * 1.0e-9;
                const double score = config_.semantic_weight * semantic +
                    config_.spatial_weight * spatial + config_.seam_weight * seam +
                    config_.support_weight * support + jitter;
                atomic_saturating_add(operation_stats_.candidates_scored);
                if (score > best_score ||
                    (score == best_score && tile.id < tiles_[best_index].id)) {
                    best_score = score;
                    best_index = candidate_index;
                    best_semantic = semantic;
                    best_seam = seam;
                }
            }

            const ImageTilePrototype& selected = tiles_[best_index];
            const std::size_t copy_width = std::min(config_.tile_size, width - cell_x);
            const std::size_t copy_height = std::min(config_.tile_size, height - cell_y);
            for (std::size_t y = 0U; y < copy_height; ++y) {
                for (std::size_t x = 0U; x < copy_width; ++x) {
                    const std::size_t output_offset =
                        ((cell_y + y) * width + cell_x + x) * 3U;
                    const std::size_t tile_offset =
                        (y * config_.tile_size + x) * 3U;
                    std::copy_n(
                        selected.rgb.begin() + static_cast<std::ptrdiff_t>(tile_offset),
                        3U,
                        result.image.rgb.begin() + static_cast<std::ptrdiff_t>(output_offset)
                    );
                }
            }
            result.selected_prototype_ids.push_back(selected.id);
            semantic_sum += best_semantic;
            seam_sum += best_seam;
        }
    }

    const double cells = static_cast<double>(result.selected_prototype_ids.size());
    result.mean_semantic_score = semantic_sum / cells;
    result.mean_seam_score = seam_sum / cells;
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, width);
    hash_u64(hash, height);
    hash_string(hash, request.prompt);
    hash_u64(hash, request.seed);
    for (const std::uint64_t id : result.selected_prototype_ids) {
        hash_u64(hash, id);
    }
    for (const std::uint8_t value : result.image.rgb) {
        hash ^= value;
        hash *= fnv_prime;
    }
    result.deterministic_hash = hash;
    return result;
}

const ImageGenerationConfig& PatchQuiltBaseline::config() const noexcept {
    return config_;
}

std::span<const ImageGenerationSource> PatchQuiltBaseline::sources() const noexcept {
    return sources_;
}

std::span<const ImageTilePrototype> PatchQuiltBaseline::tiles() const noexcept {
    return tiles_;
}

std::uint64_t PatchQuiltBaseline::images_seen() const noexcept {
    return images_seen_;
}

ImageGenerationOperationStats PatchQuiltBaseline::operation_stats() const noexcept {
    ImageGenerationOperationStats result;
    result.training_calls = atomic_load(operation_stats_.training_calls);
    result.source_images_inserted = atomic_load(
        operation_stats_.source_images_inserted
    );
    result.tile_prototypes_inserted = atomic_load(
        operation_stats_.tile_prototypes_inserted
    );
    result.source_capacity_rejections = atomic_load(
        operation_stats_.source_capacity_rejections
    );
    result.tile_capacity_rejections = atomic_load(
        operation_stats_.tile_capacity_rejections
    );
    result.string_budget_rejections = atomic_load(
        operation_stats_.string_budget_rejections
    );
    result.posting_budget_rejections = atomic_load(
        operation_stats_.posting_budget_rejections
    );
    result.generation_calls = atomic_load(operation_stats_.generation_calls);
    result.candidate_bucket_lookups = atomic_load(
        operation_stats_.candidate_bucket_lookups
    );
    result.candidates_scored = atomic_load(operation_stats_.candidates_scored);
    result.fallback_cells = atomic_load(operation_stats_.fallback_cells);
    return result;
}

std::uint64_t PatchQuiltBaseline::deterministic_hash() const noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, config_.tile_size);
    hash_u64(hash, config_.coordinate_bins);
    hash_u64(hash, config_.maximum_source_images);
    hash_u64(hash, config_.maximum_tile_prototypes);
    hash_u64(hash, config_.maximum_source_side);
    hash_u64(hash, config_.maximum_source_pixels);
    hash_u64(hash, config_.maximum_caption_bytes);
    hash_u64(hash, config_.maximum_caption_concepts);
    hash_u64(hash, config_.maximum_total_caption_bytes);
    hash_u64(hash, config_.maximum_total_concept_bytes);
    hash_u64(hash, config_.maximum_posting_entries);
    hash_u64(hash, config_.maximum_candidates_per_cell);
    hash_u64(hash, config_.default_output_width);
    hash_u64(hash, config_.default_output_height);
    hash_u64(hash, config_.maximum_output_side);
    hash_u64(hash, config_.maximum_output_pixels);
    hash_u64(hash, std::bit_cast<std::uint64_t>(config_.semantic_weight));
    hash_u64(hash, std::bit_cast<std::uint64_t>(config_.spatial_weight));
    hash_u64(hash, std::bit_cast<std::uint64_t>(config_.seam_weight));
    hash_u64(hash, std::bit_cast<std::uint64_t>(config_.support_weight));
    hash_u64(hash, next_source_id_);
    hash_u64(hash, next_tile_id_);
    hash_u64(hash, images_seen_);
    for (const ImageGenerationSource& source : sources_) {
        hash_u64(hash, source.id);
        hash_string(hash, source.caption);
        hash_u64(hash, source.width);
        hash_u64(hash, source.height);
        hash_u64(hash, source.first_tile);
        hash_u64(hash, source.tile_count);
        hash_u64(hash, source.concepts.size());
        for (const std::string& concept_name : source.concepts) {
            hash_string(hash, concept_name);
        }
    }
    for (const ImageTilePrototype& tile : tiles_) {
        hash_u64(hash, tile.id);
        hash_u64(hash, tile.source_id);
        hash_u64(hash, tile.x_bin);
        hash_u64(hash, tile.y_bin);
        hash_u64(hash, tile.support);
        for (const float value : tile.descriptor) {
            hash_u64(hash, std::bit_cast<std::uint32_t>(value));
        }
        for (const std::uint8_t value : tile.rgb) {
            hash ^= value;
            hash *= fnv_prime;
        }
    }
    return hash;
}

PatchQuiltSnapshot PatchQuiltBaseline::snapshot() const {
    return {
        config_,
        next_source_id_,
        next_tile_id_,
        images_seen_,
        sources_,
        tiles_,
    };
}

PatchQuiltBaseline PatchQuiltBaseline::from_snapshot(
    PatchQuiltSnapshot snapshot
) {
    validate_config(snapshot.config);
    if (snapshot.sources.size() > snapshot.config.maximum_source_images ||
        snapshot.tiles.size() > snapshot.config.maximum_tile_prototypes ||
        snapshot.images_seen != snapshot.sources.size() ||
        snapshot.next_source_id != snapshot.sources.size() + 1U ||
        snapshot.next_tile_id != snapshot.tiles.size() + 1U) {
        throw std::invalid_argument("invalid image-generation snapshot limits");
    }
    const std::size_t expected_tile_bytes =
        snapshot.config.tile_size * snapshot.config.tile_size * 3U;
    std::unordered_set<std::uint64_t> source_ids;
    std::unordered_set<std::uint64_t> tile_ids;
    source_ids.reserve(snapshot.sources.size());
    tile_ids.reserve(snapshot.tiles.size());
    std::size_t covered_tiles = 0U;
    for (std::size_t source_index = 0U;
         source_index < snapshot.sources.size();
         ++source_index) {
        const ImageGenerationSource& source = snapshot.sources[source_index];
        if (source.id != source_index + 1U ||
            !source_ids.insert(source.id).second || source.width == 0U ||
            source.height == 0U || source.width > snapshot.config.maximum_source_side ||
            source.height > snapshot.config.maximum_source_side ||
            source.width > snapshot.config.maximum_source_pixels / source.height ||
            source.width * source.height > snapshot.config.maximum_source_pixels ||
            source.caption.size() > snapshot.config.maximum_caption_bytes ||
            source.concepts.empty() || source.tile_count == 0U ||
            source.concepts.size() > snapshot.config.maximum_caption_concepts ||
            source.first_tile != covered_tiles ||
            source.tile_count > snapshot.tiles.size() - source.first_tile) {
            throw std::invalid_argument("invalid image-generation source snapshot");
        }
        for (const std::string& concept_name : source.concepts) {
            if (concept_name.empty() ||
                concept_name.size() > snapshot.config.maximum_caption_bytes) {
                throw std::invalid_argument("invalid image-generation source concept");
            }
        }
        for (std::size_t tile_offset = 0U;
             tile_offset < source.tile_count;
             ++tile_offset) {
            if (snapshot.tiles[source.first_tile + tile_offset].source_id != source.id) {
                throw std::invalid_argument("image-generation tile ownership mismatch");
            }
        }
        covered_tiles += source.tile_count;
    }
    if (covered_tiles != snapshot.tiles.size()) {
        throw std::invalid_argument("image-generation source ranges do not cover tiles");
    }
    for (std::size_t tile_index = 0U; tile_index < snapshot.tiles.size(); ++tile_index) {
        const ImageTilePrototype& tile = snapshot.tiles[tile_index];
        if (tile.id != tile_index + 1U ||
            !tile_ids.insert(tile.id).second || source_ids.count(tile.source_id) == 0U ||
            tile.x_bin >= snapshot.config.coordinate_bins ||
            tile.y_bin >= snapshot.config.coordinate_bins || tile.support == 0U ||
            tile.rgb.size() != expected_tile_bytes) {
            throw std::invalid_argument("invalid image-generation tile snapshot");
        }
        for (const float value : tile.descriptor) {
            if (!std::isfinite(value) || value < 0.0F || value > 1.0F) {
                throw std::invalid_argument("non-finite image-generation descriptor");
            }
        }
    }
    PatchQuiltBaseline fabric(std::move(snapshot.config));
    fabric.next_source_id_ = snapshot.next_source_id;
    fabric.next_tile_id_ = snapshot.next_tile_id;
    fabric.images_seen_ = snapshot.images_seen;
    fabric.sources_ = std::move(snapshot.sources);
    fabric.tiles_ = std::move(snapshot.tiles);
    fabric.rebuild_indices();
    return fabric;
}

}  // namespace rlf::solstice
