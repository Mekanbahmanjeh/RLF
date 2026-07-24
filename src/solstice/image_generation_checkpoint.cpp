#include "rlf/solstice/image_generation_checkpoint.hpp"

#include "rlf/core/sha256.hpp"

#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace rlf::solstice {
namespace {

constexpr std::array<char, 8U> checkpoint_magic{
    'R', 'L', 'F', 'I', 'M', 'G', '0', '1'
};
constexpr std::uint32_t checkpoint_version = 4U;
constexpr std::uint32_t minimum_checkpoint_version = 1U;
constexpr std::uint32_t header_bytes = 64U;

class PayloadWriter final {
public:
    explicit PayloadWriter(std::ostream& output) : output_(output) {}

    void u8(const std::uint8_t value) {
        output_.put(static_cast<char>(value));
        if (!output_) {
            throw std::runtime_error("failed while streaming image-generation checkpoint");
        }
        if (bytes_written_ == std::numeric_limits<std::uint64_t>::max()) {
            throw std::runtime_error("image-generation checkpoint size overflow");
        }
        ++bytes_written_;
    }

    void u16(const std::uint16_t value) {
        for (unsigned int index = 0U; index < 2U; ++index) {
            u8(static_cast<std::uint8_t>((value >> (index * 8U)) & 0xFFU));
        }
    }

    void u32(const std::uint32_t value) { u64_low(value); }

    void u64(const std::uint64_t value) {
        for (unsigned int index = 0U; index < 8U; ++index) {
            u8(static_cast<std::uint8_t>((value >> (index * 8U)) & 0xFFULL));
        }
    }

    void size(const std::size_t value) { u64(static_cast<std::uint64_t>(value)); }
    void float32(const float value) { u64_low(std::bit_cast<std::uint32_t>(value)); }
    void float64(const double value) { u64(std::bit_cast<std::uint64_t>(value)); }

    void string(const std::string_view value) {
        size(value.size());
        for (const char character : value) {
            u8(static_cast<std::uint8_t>(static_cast<unsigned char>(character)));
        }
    }

    void bytes(const std::span<const std::uint8_t> values) {
        size(values.size());
        for (const std::uint8_t value : values) {
            u8(value);
        }
    }

    [[nodiscard]] std::uint64_t bytes_written() const noexcept {
        return bytes_written_;
    }

private:
    void u64_low(const std::uint32_t value) {
        for (unsigned int index = 0U; index < 4U; ++index) {
            u8(static_cast<std::uint8_t>((value >> (index * 8U)) & 0xFFU));
        }
    }

    std::ostream& output_;
    std::uint64_t bytes_written_{};
};

class PayloadReader final {
public:
    PayloadReader(
        std::istream& input,
        const std::uint64_t payload_bytes_value,
        ImageGenerationCheckpointLimits limits
    ) : input_(input), remaining_(payload_bytes_value), limits_(limits) {}

    [[nodiscard]] std::uint8_t u8() {
        require(1U);
        const int value = input_.get();
        if (value == std::char_traits<char>::eof()) {
            throw std::runtime_error("truncated image-generation checkpoint payload");
        }
        --remaining_;
        return static_cast<std::uint8_t>(static_cast<unsigned char>(value));
    }

    [[nodiscard]] std::uint16_t u16() {
        std::uint16_t value = 0U;
        for (unsigned int index = 0U; index < 2U; ++index) {
            value |= static_cast<std::uint16_t>(
                static_cast<std::uint16_t>(u8()) << (index * 8U)
            );
        }
        return value;
    }

    [[nodiscard]] std::uint32_t u32() {
        std::uint32_t value = 0U;
        for (unsigned int index = 0U; index < 4U; ++index) {
            value |= static_cast<std::uint32_t>(u8()) << (index * 8U);
        }
        return value;
    }

    [[nodiscard]] std::uint64_t u64() {
        std::uint64_t value = 0U;
        for (unsigned int index = 0U; index < 8U; ++index) {
            value |= static_cast<std::uint64_t>(u8()) << (index * 8U);
        }
        return value;
    }

    [[nodiscard]] std::size_t size() {
        const std::uint64_t value = u64();
        if (value > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max()
            )) {
            throw std::runtime_error("checkpoint size exceeds platform range");
        }
        return static_cast<std::size_t>(value);
    }

    [[nodiscard]] float float32() { return std::bit_cast<float>(u32()); }
    [[nodiscard]] double float64() { return std::bit_cast<double>(u64()); }

    [[nodiscard]] std::string string() {
        const std::size_t length = size();
        if (length > limits_.maximum_string_bytes) {
            throw std::runtime_error("checkpoint string exceeds configured limit");
        }
        require(length);
        std::string value(length, '\0');
        for (char& character : value) {
            character = static_cast<char>(u8());
        }
        return value;
    }

    [[nodiscard]] std::vector<std::uint8_t> bytes(
        const std::size_t expected_length,
        std::uint64_t& cumulative_bytes
    ) {
        const std::size_t length = size();
        if (length != expected_length ||
            static_cast<std::uint64_t>(length) >
                limits_.maximum_rgb_bytes - cumulative_bytes) {
            throw std::runtime_error("invalid image-generation tile byte count");
        }
        require(length);
        std::vector<std::uint8_t> value(length, std::uint8_t{0U});
        for (std::uint8_t& byte : value) {
            byte = u8();
        }
        cumulative_bytes += length;
        return value;
    }

    void require_minimum_records(
        const std::size_t count,
        const std::uint64_t minimum_record_bytes
    ) const {
        if (minimum_record_bytes != 0U &&
            static_cast<std::uint64_t>(count) > remaining_ / minimum_record_bytes) {
            throw std::runtime_error("checkpoint count exceeds remaining payload");
        }
    }

    [[nodiscard]] bool finished() const noexcept { return remaining_ == 0U; }
    [[nodiscard]] const ImageGenerationCheckpointLimits& limits() const noexcept {
        return limits_;
    }

private:
    void require(const std::size_t count) const {
        if (static_cast<std::uint64_t>(count) > remaining_) {
            throw std::runtime_error("truncated image-generation checkpoint payload");
        }
    }

    std::istream& input_;
    std::uint64_t remaining_{};
    ImageGenerationCheckpointLimits limits_;
};

void write_u32_stream(std::ostream& output, const std::uint32_t value) {
    for (unsigned int index = 0U; index < 4U; ++index) {
        output.put(static_cast<char>((value >> (index * 8U)) & 0xFFU));
    }
    if (!output) throw std::runtime_error("failed while writing checkpoint header");
}

void write_u64_stream(std::ostream& output, const std::uint64_t value) {
    for (unsigned int index = 0U; index < 8U; ++index) {
        output.put(static_cast<char>((value >> (index * 8U)) & 0xFFULL));
    }
    if (!output) throw std::runtime_error("failed while writing checkpoint header");
}

[[nodiscard]] std::uint32_t read_u32_stream(std::istream& input) {
    std::uint32_t value = 0U;
    for (unsigned int index = 0U; index < 4U; ++index) {
        const int byte = input.get();
        if (byte == std::char_traits<char>::eof()) {
            throw std::runtime_error("truncated image-generation checkpoint header");
        }
        value |= static_cast<std::uint32_t>(
            static_cast<unsigned char>(byte)
        ) << (index * 8U);
    }
    return value;
}

[[nodiscard]] std::uint64_t read_u64_stream(std::istream& input) {
    std::uint64_t value = 0U;
    for (unsigned int index = 0U; index < 8U; ++index) {
        const int byte = input.get();
        if (byte == std::char_traits<char>::eof()) {
            throw std::runtime_error("truncated image-generation checkpoint header");
        }
        value |= static_cast<std::uint64_t>(
            static_cast<unsigned char>(byte)
        ) << (index * 8U);
    }
    return value;
}

void write_config(PayloadWriter& writer, const ImageGenerationConfig& config) {
    writer.size(config.tile_size);
    writer.size(config.coordinate_bins);
    writer.size(config.maximum_source_images);
    writer.size(config.maximum_tile_prototypes);
    writer.size(config.maximum_source_side);
    writer.size(config.maximum_source_pixels);
    writer.size(config.maximum_caption_bytes);
    writer.size(config.maximum_caption_concepts);
    writer.u64(config.maximum_total_caption_bytes);
    writer.u64(config.maximum_total_concept_bytes);
    writer.u64(config.maximum_posting_entries);
    writer.size(config.maximum_candidates_per_cell);
    writer.size(config.default_output_width);
    writer.size(config.default_output_height);
    writer.size(config.maximum_output_side);
    writer.size(config.maximum_output_pixels);
    writer.float64(config.semantic_weight);
    writer.float64(config.spatial_weight);
    writer.float64(config.seam_weight);
    writer.float64(config.support_weight);
}

[[nodiscard]] ImageGenerationConfig read_config(PayloadReader& reader) {
    ImageGenerationConfig config;
    config.tile_size = reader.size();
    config.coordinate_bins = reader.size();
    config.maximum_source_images = reader.size();
    config.maximum_tile_prototypes = reader.size();
    config.maximum_source_side = reader.size();
    config.maximum_source_pixels = reader.size();
    config.maximum_caption_bytes = reader.size();
    config.maximum_caption_concepts = reader.size();
    config.maximum_total_caption_bytes = reader.u64();
    config.maximum_total_concept_bytes = reader.u64();
    config.maximum_posting_entries = reader.u64();
    config.maximum_candidates_per_cell = reader.size();
    config.default_output_width = reader.size();
    config.default_output_height = reader.size();
    config.maximum_output_side = reader.size();
    config.maximum_output_pixels = reader.size();
    config.semantic_weight = reader.float64();
    config.spatial_weight = reader.float64();
    config.seam_weight = reader.float64();
    config.support_weight = reader.float64();
    return config;
}

void write_operations(
    PayloadWriter& writer,
    const ImageGenerationOperationStats& stats
) {
    writer.u64(stats.training_calls);
    writer.u64(stats.source_images_inserted);
    writer.u64(stats.tile_prototypes_inserted);
    writer.u64(stats.source_capacity_rejections);
    writer.u64(stats.tile_capacity_rejections);
    writer.u64(stats.string_budget_rejections);
    writer.u64(stats.posting_budget_rejections);
    writer.u64(stats.generation_calls);
    writer.u64(stats.candidate_bucket_lookups);
    writer.u64(stats.candidates_scored);
    writer.u64(stats.fallback_cells);
}

[[nodiscard]] ImageGenerationOperationStats read_operations(PayloadReader& reader) {
    ImageGenerationOperationStats stats;
    stats.training_calls = reader.u64();
    stats.source_images_inserted = reader.u64();
    stats.tile_prototypes_inserted = reader.u64();
    stats.source_capacity_rejections = reader.u64();
    stats.tile_capacity_rejections = reader.u64();
    stats.string_budget_rejections = reader.u64();
    stats.posting_budget_rejections = reader.u64();
    stats.generation_calls = reader.u64();
    stats.candidate_bucket_lookups = reader.u64();
    stats.candidates_scored = reader.u64();
    stats.fallback_cells = reader.u64();
    return stats;
}

void write_resonant_config(
    PayloadWriter& writer,
    const ResonantImageConfig& config
) {
    writer.size(config.patch_size);
    writer.size(config.phase_redundancy);
    writer.size(config.coordinate_bins);
    writer.size(config.maximum_modes);
    writer.size(config.maximum_concept_bytes);
    writer.size(config.maximum_image_side);
    writer.size(config.maximum_image_pixels);
    writer.size(config.candidate_count);
    writer.size(config.active_count);
    writer.size(config.maximum_settling_cycles);
    writer.size(config.maximum_trace_entries);
    writer.size(config.maximum_prompt_concepts);
    writer.size(config.maximum_semantic_candidates);
    writer.float64(config.minimum_resonance);
    writer.float64(config.minimum_semantic_similarity);
    writer.float64(config.semantic_resonance_weight);
    writer.float64(config.convergence_tolerance_radians);
    writer.float64(config.settling_relaxation);
    writer.float64(config.transformation_learning_rate);
    writer.float64(config.context_learning_rate);
    writer.float64(config.confidence_learning_rate);
    writer.u64(config.seed);
    writer.size(config.prompt_semantics.phase_dimension);
    writer.size(config.prompt_semantics.maximum_words);
    writer.size(config.prompt_semantics.maximum_word_bytes);
    writer.size(config.prompt_semantics.maximum_words_per_record);
    writer.size(config.prompt_semantics.context_window);
    writer.size(config.prompt_semantics.bucket_bits);
    writer.size(config.prompt_semantics.maximum_expansions_per_word);
    writer.u64(config.prompt_semantics.minimum_support);
    writer.float64(config.prompt_semantics.learning_rate);
    writer.u64(config.prompt_semantics.seed);
}

[[nodiscard]] ResonantImageConfig read_resonant_config(
    PayloadReader& reader,
    const std::uint32_t format_version
) {
    ResonantImageConfig config;
    config.patch_size = reader.size();
    config.phase_redundancy = reader.size();
    config.coordinate_bins = reader.size();
    config.maximum_modes = reader.size();
    config.maximum_concept_bytes = reader.size();
    config.maximum_image_side = reader.size();
    config.maximum_image_pixels = reader.size();
    config.candidate_count = reader.size();
    config.active_count = reader.size();
    config.maximum_settling_cycles = reader.size();
    config.maximum_trace_entries = reader.size();
    if (format_version >= 3U) {
        config.maximum_prompt_concepts = reader.size();
        config.maximum_semantic_candidates = reader.size();
    }
    config.minimum_resonance = reader.float64();
    if (format_version >= 3U) {
        config.minimum_semantic_similarity = reader.float64();
        config.semantic_resonance_weight = reader.float64();
    }
    config.convergence_tolerance_radians = reader.float64();
    config.settling_relaxation = reader.float64();
    config.transformation_learning_rate = reader.float64();
    config.context_learning_rate = reader.float64();
    config.confidence_learning_rate = reader.float64();
    config.seed = reader.u64();
    if (format_version >= 4U) {
        config.prompt_semantics.phase_dimension = reader.size();
        config.prompt_semantics.maximum_words = reader.size();
        config.prompt_semantics.maximum_word_bytes = reader.size();
        config.prompt_semantics.maximum_words_per_record = reader.size();
        config.prompt_semantics.context_window = reader.size();
        config.prompt_semantics.bucket_bits = reader.size();
        config.prompt_semantics.maximum_expansions_per_word = reader.size();
        config.prompt_semantics.minimum_support = reader.u64();
        config.prompt_semantics.learning_rate = reader.float64();
        config.prompt_semantics.seed = reader.u64();
    }
    return config;
}

void write_resonant_operations(
    PayloadWriter& writer,
    const ResonantImageOperationStats& stats
) {
    writer.u64(stats.training_examples);
    writer.u64(stats.training_patches);
    writer.u64(stats.modes_created);
    writer.u64(stats.local_mode_updates);
    writer.u64(stats.generation_calls);
    writer.u64(stats.generated_patches);
    writer.u64(stats.sparse_bucket_lookups);
    writer.u64(stats.resonance_evaluations);
    writer.u64(stats.active_mode_applications);
    writer.u64(stats.settling_cycles);
    writer.u64(stats.unresolved_patch_transformations);
    writer.u64(stats.decoded_channels);
    writer.u64(stats.semantic_bucket_lookups);
    writer.u64(stats.semantic_candidates_scored);
    writer.u64(stats.semantic_matches);
}

[[nodiscard]] ResonantImageOperationStats read_resonant_operations(
    PayloadReader& reader,
    const std::uint32_t format_version
) {
    ResonantImageOperationStats stats{
        .training_examples = reader.u64(),
        .training_patches = reader.u64(),
        .modes_created = reader.u64(),
        .local_mode_updates = reader.u64(),
        .generation_calls = reader.u64(),
        .generated_patches = reader.u64(),
        .sparse_bucket_lookups = reader.u64(),
        .resonance_evaluations = reader.u64(),
        .active_mode_applications = reader.u64(),
        .settling_cycles = reader.u64(),
        .unresolved_patch_transformations = reader.u64(),
        .decoded_channels = reader.u64(),
    };
    if (format_version >= 3U) {
        stats.semantic_bucket_lookups = reader.u64();
        stats.semantic_candidates_scored = reader.u64();
        stats.semantic_matches = reader.u64();
    }
    return stats;
}

void write_phase_vector(
    PayloadWriter& writer,
    const core::PhaseVector& phase
) {
    writer.size(phase.size());
    for (const float angle : phase.angles()) {
        writer.float32(angle);
    }
}

[[nodiscard]] core::PhaseVector read_phase_vector(
    PayloadReader& reader,
    std::uint64_t& cumulative_phase_values
) {
    const std::size_t count = reader.size();
    if (static_cast<std::uint64_t>(count) >
        reader.limits().maximum_phase_values - cumulative_phase_values) {
        throw std::runtime_error(
            "resonant-image checkpoint phase-value limit exceeded"
        );
    }
    reader.require_minimum_records(count, 4U);
    std::vector<float> angles;
    angles.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        angles.push_back(reader.float32());
    }
    cumulative_phase_values += static_cast<std::uint64_t>(count);
    return core::PhaseVector(std::move(angles));
}

void write_prompt_semantics(
    PayloadWriter& writer,
    const PromptSemanticSnapshot& snapshot
) {
    writer.u64(snapshot.next_mode_id);
    writer.u64(snapshot.stats.records_seen);
    writer.u64(snapshot.stats.words_seen);
    writer.u64(snapshot.stats.modes_created);
    writer.u64(snapshot.stats.modes_updated);
    writer.u64(snapshot.stats.capacity_skips);
    writer.u64(snapshot.stats.semantic_queries);
    writer.u64(snapshot.stats.bucket_candidates_scored);
    writer.size(snapshot.modes.size());
    for (const PromptSemanticMode& mode : snapshot.modes) {
        writer.u64(mode.id);
        writer.string(mode.word);
        write_phase_vector(writer, mode.context_prototype);
        writer.u64(mode.support);
    }
}

[[nodiscard]] PromptSemanticSnapshot read_prompt_semantics(
    PayloadReader& reader,
    const PromptSemanticConfig& config,
    std::uint64_t& cumulative_phase_values
) {
    PromptSemanticSnapshot snapshot;
    snapshot.config = config;
    snapshot.next_mode_id = reader.u64();
    snapshot.stats.records_seen = reader.u64();
    snapshot.stats.words_seen = reader.u64();
    snapshot.stats.modes_created = reader.u64();
    snapshot.stats.modes_updated = reader.u64();
    snapshot.stats.capacity_skips = reader.u64();
    snapshot.stats.semantic_queries = reader.u64();
    snapshot.stats.bucket_candidates_scored = reader.u64();
    const std::size_t count = reader.size();
    if (count > config.maximum_words ||
        count > reader.limits().maximum_prompt_semantic_modes) {
        throw std::runtime_error(
            "prompt-semantic mode count exceeds load limit"
        );
    }
    reader.require_minimum_records(count, 32U);
    snapshot.modes.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        snapshot.modes.push_back({
            .id = reader.u64(),
            .word = reader.string(),
            .context_prototype = read_phase_vector(
                reader, cumulative_phase_values
            ),
            .support = reader.u64(),
        });
    }
    return snapshot;
}

void validate_shards(const std::span<const ImageGenerationShardRecord> shards) {
    std::unordered_set<std::string> identifiers;
    identifiers.reserve(shards.size());
    for (const ImageGenerationShardRecord& shard : shards) {
        if (shard.shard_id.empty() || !identifiers.insert(shard.shard_id).second ||
            !rlf::core::is_sha256_hex(shard.shard_sha256) ||
            !rlf::core::is_sha256_hex(shard.ledger_sha256) ||
            shard.source_uri.empty() || shard.license.empty()) {
            throw std::invalid_argument("invalid image-generation completed shard");
        }
    }
}

void write_shards(
    PayloadWriter& writer,
    const std::span<const ImageGenerationShardRecord> shards
) {
    writer.size(shards.size());
    for (const ImageGenerationShardRecord& shard : shards) {
        writer.string(shard.shard_id);
        writer.string(shard.shard_sha256);
        writer.string(shard.ledger_sha256);
        writer.string(shard.source_uri);
        writer.string(shard.license);
        writer.u64(shard.records);
        writer.u64(shard.bytes);
    }
}

[[nodiscard]] std::vector<ImageGenerationShardRecord> read_shards(
    PayloadReader& reader
) {
    const std::size_t shard_count = reader.size();
    if (shard_count > reader.limits().maximum_completed_shards) {
        throw std::runtime_error("image-generation shard count exceeds load limit");
    }
    reader.require_minimum_records(shard_count, 56U);
    std::vector<ImageGenerationShardRecord> shards;
    shards.reserve(shard_count);
    for (std::size_t index = 0U; index < shard_count; ++index) {
        ImageGenerationShardRecord shard;
        shard.shard_id = reader.string();
        shard.shard_sha256 = reader.string();
        shard.ledger_sha256 = reader.string();
        shard.source_uri = reader.string();
        shard.license = reader.string();
        shard.records = reader.u64();
        shard.bytes = reader.u64();
        shards.push_back(std::move(shard));
    }
    validate_shards(shards);
    return shards;
}

void serialize(
    PayloadWriter& writer,
    const ImageGenerationCheckpointState& state
) {
    const bool patch_quilt =
        state.architecture == ImageGenerationArchitecture::patch_quilt_baseline;
    const bool resonant =
        state.architecture == ImageGenerationArchitecture::resonant_fabric;
    if ((!patch_quilt && !resonant) ||
        (patch_quilt && !image_generation_profile_config_matches(
            state.profile,
            state.fabric.config()
        )) ||
        (resonant && !resonant_image_profile_config_matches(
            state.profile,
            state.resonant_fabric.config()
        ))) {
        throw std::invalid_argument("image-generation checkpoint profile mismatch");
    }
    validate_shards(state.completed_shards);
    writer.u8(static_cast<std::uint8_t>(state.profile));
    writer.u8(static_cast<std::uint8_t>(state.architecture));
    writer.u64(state.master_seed);
    writer.u64(state.training_step);
    if (patch_quilt) {
        write_config(writer, state.fabric.config());
        writer.u64(state.fabric.sources().size() + 1U);
        writer.u64(state.fabric.tiles().size() + 1U);
        writer.u64(state.fabric.images_seen());
        write_operations(writer, state.cumulative_operations);
        writer.size(state.fabric.sources().size());
        for (const ImageGenerationSource& source : state.fabric.sources()) {
            writer.u64(source.id);
            writer.string(source.caption);
            writer.size(source.concepts.size());
            for (const std::string& concept_name : source.concepts) {
                writer.string(concept_name);
            }
            writer.size(source.width);
            writer.size(source.height);
            writer.size(source.first_tile);
            writer.size(source.tile_count);
        }
        writer.size(state.fabric.tiles().size());
        for (const ImageTilePrototype& tile : state.fabric.tiles()) {
            writer.u64(tile.id);
            writer.u64(tile.source_id);
            writer.u16(tile.x_bin);
            writer.u16(tile.y_bin);
            for (const float value : tile.descriptor) {
                writer.float32(value);
            }
            writer.bytes(tile.rgb);
            writer.u64(tile.support);
        }
    } else {
        const ResonantImageSnapshot snapshot = state.resonant_fabric.snapshot();
        write_resonant_config(writer, snapshot.config);
        writer.u64(snapshot.next_mode_id);
        write_resonant_operations(writer, snapshot.operation_stats);
        writer.size(snapshot.modes.size());
        for (const ResonantImageMode& image_mode : snapshot.modes) {
            writer.string(image_mode.semantic_label);
            writer.u16(image_mode.x_bin);
            writer.u16(image_mode.y_bin);
            const core::ResonantMode& mode = image_mode.resonant_mode;
            writer.u64(mode.id);
            write_phase_vector(writer, mode.context_key);
            write_phase_vector(writer, mode.transformation);
            writer.float32(mode.selectivity);
            writer.float32(mode.confidence);
            writer.float32(mode.utility);
            writer.u64(mode.activation_count);
            writer.u64(mode.successful_update_count);
            writer.u64(mode.unsuccessful_update_count);
            writer.u64(mode.creation_step);
            writer.u64(mode.last_used_step);
            writer.u8(mode.enabled ? 1U : 0U);
            writer.u64(image_mode.example_count);
            writer.size(mode.recent_corrections.size());
            for (const core::CorrectionSummary& correction :
                 mode.recent_corrections) {
                write_phase_vector(writer, correction.context);
                write_phase_vector(writer, correction.desired_transformation);
                writer.float64(correction.proposal_quality);
                writer.u8(correction.improved_prediction ? 1U : 0U);
                writer.u64(correction.step);
            }
        }
        write_prompt_semantics(writer, snapshot.prompt_semantics);
    }
    write_shards(writer, state.completed_shards);
}

[[nodiscard]] ImageGenerationCheckpointState deserialize(
    PayloadReader& reader,
    const std::uint32_t format_version
) {
    const std::uint8_t profile_value = reader.u8();
    const std::uint8_t maximum_profile = format_version == 1U
        ? static_cast<std::uint8_t>(ImageGenerationProfile::a100_80g)
        : static_cast<std::uint8_t>(ImageGenerationProfile::v100_32g);
    if (profile_value > maximum_profile) {
        throw std::runtime_error("unknown image-generation checkpoint profile");
    }
    const auto profile = static_cast<ImageGenerationProfile>(profile_value);
    const std::uint8_t architecture_value = reader.u8();
    if (architecture_value > static_cast<std::uint8_t>(
            ImageGenerationArchitecture::resonant_fabric
        ) ||
        (format_version == 1U && architecture_value != static_cast<std::uint8_t>(
            ImageGenerationArchitecture::patch_quilt_baseline
        ))) {
        throw std::runtime_error("unsupported image-generation checkpoint architecture");
    }
    const auto architecture = static_cast<ImageGenerationArchitecture>(
        architecture_value
    );
    const std::uint64_t master_seed = reader.u64();
    const std::uint64_t training_step = reader.u64();
    ImageGenerationCheckpointState state;
    state.profile = profile;
    state.architecture = architecture;
    state.master_seed = master_seed;
    state.training_step = training_step;
    if (architecture == ImageGenerationArchitecture::patch_quilt_baseline) {
        PatchQuiltSnapshot snapshot;
        snapshot.config = read_config(reader);
        if (!image_generation_profile_config_matches(profile, snapshot.config)) {
            throw std::runtime_error("image-generation checkpoint config is not profile-bound");
        }
        snapshot.next_source_id = reader.u64();
        snapshot.next_tile_id = reader.u64();
        snapshot.images_seen = reader.u64();
        const ImageGenerationOperationStats operations = read_operations(reader);

        const std::size_t source_count = reader.size();
        if (source_count > reader.limits().maximum_source_images ||
            source_count > snapshot.config.maximum_source_images) {
            throw std::runtime_error("image-generation source count exceeds load limit");
        }
        reader.require_minimum_records(source_count, 56U);
        snapshot.sources.reserve(source_count);
        std::size_t total_concepts = 0U;
        for (std::size_t index = 0U; index < source_count; ++index) {
            ImageGenerationSource source;
            source.id = reader.u64();
            source.caption = reader.string();
            const std::size_t concept_count = reader.size();
            if (concept_count > snapshot.config.maximum_caption_concepts ||
                concept_count > reader.limits().maximum_total_concepts - total_concepts) {
                throw std::runtime_error("image-generation concept count exceeds load limit");
            }
            reader.require_minimum_records(concept_count, 8U);
            source.concepts.reserve(concept_count);
            for (std::size_t concept_index = 0U;
                 concept_index < concept_count;
                 ++concept_index) {
                source.concepts.push_back(reader.string());
            }
            total_concepts += concept_count;
            source.width = reader.size();
            source.height = reader.size();
            source.first_tile = reader.size();
            source.tile_count = reader.size();
            snapshot.sources.push_back(std::move(source));
        }

        const std::size_t tile_count = reader.size();
        if (tile_count > reader.limits().maximum_tile_prototypes ||
            tile_count > snapshot.config.maximum_tile_prototypes) {
            throw std::runtime_error("image-generation tile count exceeds load limit");
        }
        const std::size_t tile_rgb_bytes =
            snapshot.config.tile_size * snapshot.config.tile_size * 3U;
        reader.require_minimum_records(
            tile_count,
            84U + static_cast<std::uint64_t>(tile_rgb_bytes)
        );
        snapshot.tiles.reserve(tile_count);
        std::uint64_t cumulative_rgb_bytes = 0U;
        for (std::size_t index = 0U; index < tile_count; ++index) {
            ImageTilePrototype tile;
            tile.id = reader.u64();
            tile.source_id = reader.u64();
            tile.x_bin = reader.u16();
            tile.y_bin = reader.u16();
            for (float& value : tile.descriptor) {
                value = reader.float32();
            }
            tile.rgb = reader.bytes(tile_rgb_bytes, cumulative_rgb_bytes);
            tile.support = reader.u64();
            snapshot.tiles.push_back(std::move(tile));
        }
        state.fabric = PatchQuiltBaseline::from_snapshot(std::move(snapshot));
        state.cumulative_operations = operations;
    } else {
        ResonantImageSnapshot snapshot;
        snapshot.config = read_resonant_config(reader, format_version);
        if (!resonant_image_profile_config_matches(profile, snapshot.config)) {
            throw std::runtime_error(
                "resonant-image checkpoint config is not profile-bound"
            );
        }
        snapshot.next_mode_id = reader.u64();
        snapshot.operation_stats = read_resonant_operations(
            reader,
            format_version
        );
        const std::size_t mode_count = reader.size();
        if (mode_count > reader.limits().maximum_resonant_modes ||
            mode_count > snapshot.config.maximum_modes) {
            throw std::runtime_error("resonant-image mode count exceeds load limit");
        }
        reader.require_minimum_records(mode_count, 121U);
        snapshot.modes.reserve(mode_count);
        std::uint64_t phase_values = 0ULL;
        for (std::size_t index = 0U; index < mode_count; ++index) {
            const std::string label = reader.string();
            const std::uint16_t x_bin = reader.u16();
            const std::uint16_t y_bin = reader.u16();
            const std::uint64_t id = reader.u64();
            core::PhaseVector context = read_phase_vector(reader, phase_values);
            core::PhaseVector transformation = read_phase_vector(
                reader,
                phase_values
            );
            const float selectivity = reader.float32();
            const float confidence = reader.float32();
            const float utility = reader.float32();
            const std::uint64_t activation_count = reader.u64();
            const std::uint64_t successful_update_count = reader.u64();
            const std::uint64_t unsuccessful_update_count = reader.u64();
            const std::uint64_t creation_step = reader.u64();
            const std::uint64_t last_used_step = reader.u64();
            const std::uint8_t enabled = reader.u8();
            const std::uint64_t example_count = reader.u64();
            core::ResonantMode mode(
                id,
                std::move(context),
                std::move(transformation),
                selectivity,
                confidence,
                utility,
                creation_step
            );
            mode.activation_count = activation_count;
            mode.successful_update_count = successful_update_count;
            mode.unsuccessful_update_count = unsuccessful_update_count;
            mode.last_used_step = last_used_step;
            if (enabled > 1U) {
                throw std::runtime_error(
                    "invalid resonant-image enabled checkpoint flag"
                );
            }
            mode.enabled = enabled == 1U;
            const std::size_t correction_count = reader.size();
            if (correction_count >
                reader.limits().maximum_corrections_per_mode) {
                throw std::runtime_error(
                    "resonant-image correction count exceeds load limit"
                );
            }
            mode.recent_corrections.reserve(correction_count);
            for (std::size_t correction_index = 0U;
                 correction_index < correction_count;
                 ++correction_index) {
                core::PhaseVector correction_context = read_phase_vector(
                    reader,
                    phase_values
                );
                core::PhaseVector desired_transformation = read_phase_vector(
                    reader,
                    phase_values
                );
                const double proposal_quality = reader.float64();
                const std::uint8_t improved = reader.u8();
                if (improved > 1U) {
                    throw std::runtime_error(
                        "invalid resonant-image correction checkpoint flag"
                    );
                }
                mode.recent_corrections.push_back({
                    .context = std::move(correction_context),
                    .desired_transformation =
                        std::move(desired_transformation),
                    .proposal_quality = proposal_quality,
                    .improved_prediction = improved == 1U,
                    .step = reader.u64(),
                });
            }
            snapshot.modes.push_back({
                .semantic_label = label,
                .x_bin = x_bin,
                .y_bin = y_bin,
                .resonant_mode = std::move(mode),
                .example_count = example_count,
            });
        }
        if (format_version >= 4U) {
            snapshot.prompt_semantics = read_prompt_semantics(
                reader,
                snapshot.config.prompt_semantics,
                phase_values
            );
        } else {
            snapshot.prompt_semantics.config =
                snapshot.config.prompt_semantics;
        }
        state.resonant_fabric = ResonantImageFabric::from_snapshot(
            std::move(snapshot)
        );
    }
    std::vector<ImageGenerationShardRecord> shards = read_shards(reader);
    if (!reader.finished()) {
        throw std::runtime_error("unexpected trailing checkpoint payload data");
    }
    state.completed_shards = std::move(shards);
    return state;
}

struct LoadedCheckpoint final {
    ImageGenerationCheckpointState state;
    std::uint32_t format_version{};
    std::uint64_t file_bytes{};
    rlf::core::Sha256Digest payload_digest{};
};

[[nodiscard]] LoadedCheckpoint read_checkpoint(
    const std::filesystem::path& path,
    const ImageGenerationCheckpointLimits limits
) {
    std::error_code status_error;
    const std::filesystem::file_status status =
        std::filesystem::symlink_status(path, status_error);
    if (status_error || !std::filesystem::is_regular_file(status) ||
        std::filesystem::is_symlink(status)) {
        throw std::runtime_error("image-generation checkpoint must be a regular non-symlink file");
    }
    const std::uintmax_t size = std::filesystem::file_size(path);
    if (size < header_bytes || size > limits.maximum_file_bytes ||
        size > std::numeric_limits<std::uint64_t>::max()) {
        throw std::runtime_error("image-generation checkpoint file size is invalid");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("unable to open image-generation checkpoint");
    std::array<char, 8U> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (input.gcount() != static_cast<std::streamsize>(magic.size()) ||
        magic != checkpoint_magic) {
        throw std::runtime_error("invalid image-generation checkpoint magic");
    }
    const std::uint32_t version = read_u32_stream(input);
    const std::uint32_t stored_header_bytes = read_u32_stream(input);
    const std::uint64_t payload_bytes_value = read_u64_stream(input);
    rlf::core::Sha256Digest stored_digest{};
    input.read(
        reinterpret_cast<char*>(stored_digest.data()),
        static_cast<std::streamsize>(stored_digest.size())
    );
    if (input.gcount() != static_cast<std::streamsize>(stored_digest.size())) {
        throw std::runtime_error("truncated image-generation checkpoint digest");
    }
    const std::uint64_t reserved = read_u64_stream(input);
    if (version < minimum_checkpoint_version || version > checkpoint_version ||
        stored_header_bytes != header_bytes ||
        reserved != 0U || payload_bytes_value != size - header_bytes) {
        throw std::runtime_error("unsupported image-generation checkpoint header");
    }
    const rlf::core::Sha256Digest actual_digest = rlf::core::sha256_file_range(
        path,
        header_bytes,
        payload_bytes_value
    );
    if (actual_digest != stored_digest) {
        throw std::runtime_error("image-generation checkpoint payload SHA-256 mismatch");
    }
    input.seekg(header_bytes, std::ios::beg);
    if (!input) throw std::runtime_error("unable to seek to checkpoint payload");
    PayloadReader reader(input, payload_bytes_value, limits);
    LoadedCheckpoint loaded;
    loaded.state = deserialize(reader, version);
    loaded.format_version = version;
    loaded.file_bytes = static_cast<std::uint64_t>(size);
    loaded.payload_digest = actual_digest;
    return loaded;
}

#if defined(__unix__) || defined(__APPLE__)
void sync_file(const std::filesystem::path& path) {
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0) {
        throw std::runtime_error("unable to open checkpoint for fsync");
    }
    const int result = ::fsync(descriptor);
    const int saved_error = errno;
    static_cast<void>(::close(descriptor));
    if (result != 0) {
        throw std::runtime_error(
            "unable to fsync checkpoint: " + std::to_string(saved_error)
        );
    }
}

void sync_directory(const std::filesystem::path& directory) {
    const int descriptor = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY);
    if (descriptor < 0) {
        throw std::runtime_error("unable to open checkpoint directory for fsync");
    }
    const int result = ::fsync(descriptor);
    const int saved_error = errno;
    static_cast<void>(::close(descriptor));
    if (result != 0) {
        throw std::runtime_error(
            "unable to fsync checkpoint directory: " + std::to_string(saved_error)
        );
    }
}
#else
void sync_file(const std::filesystem::path&) {}
void sync_directory(const std::filesystem::path&) {}
#endif

}  // namespace

void save_image_generation_checkpoint(
    const std::filesystem::path& path,
    const ImageGenerationCheckpointState& state
) {
    if (path.empty()) {
        throw std::invalid_argument("image-generation checkpoint path must not be empty");
    }
    std::error_code status_error;
    const std::filesystem::file_status status =
        std::filesystem::symlink_status(path, status_error);
    if (status_error && status_error != std::errc::no_such_file_or_directory) {
        throw std::runtime_error("unable to inspect image-generation checkpoint target");
    }
    if (!status_error && status.type() != std::filesystem::file_type::not_found &&
        (std::filesystem::is_symlink(status) ||
         !std::filesystem::is_regular_file(status))) {
        throw std::invalid_argument(
            "image-generation checkpoint target must be a regular non-symlink file"
        );
    }
    const std::filesystem::path parent = path.parent_path().empty()
        ? std::filesystem::current_path()
        : path.parent_path();
    std::filesystem::create_directories(parent);
    const auto stamp = std::chrono::high_resolution_clock::now()
        .time_since_epoch().count();
    const std::filesystem::path temporary =
        path.string() + ".tmp." + std::to_string(stamp);
    try {
        std::uint64_t payload_size = 0U;
        {
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            if (!output) {
                throw std::runtime_error("unable to create image-generation checkpoint temporary");
            }
            output.write(
                checkpoint_magic.data(),
                static_cast<std::streamsize>(checkpoint_magic.size())
            );
            write_u32_stream(output, checkpoint_version);
            write_u32_stream(output, header_bytes);
            write_u64_stream(output, 0U);
            const rlf::core::Sha256Digest placeholder{};
            output.write(
                reinterpret_cast<const char*>(placeholder.data()),
                static_cast<std::streamsize>(placeholder.size())
            );
            write_u64_stream(output, 0U);
            PayloadWriter writer(output);
            serialize(writer, state);
            payload_size = writer.bytes_written();
            output.flush();
            if (!output) {
                throw std::runtime_error("failed while writing image-generation checkpoint");
            }
        }
        const rlf::core::Sha256Digest payload_digest = rlf::core::sha256_file_range(
            temporary,
            header_bytes,
            payload_size
        );
        {
            std::fstream output(
                temporary,
                std::ios::binary | std::ios::in | std::ios::out
            );
            if (!output) {
                throw std::runtime_error("unable to reopen image-generation checkpoint header");
            }
            output.seekp(16, std::ios::beg);
            write_u64_stream(output, payload_size);
            output.write(
                reinterpret_cast<const char*>(payload_digest.data()),
                static_cast<std::streamsize>(payload_digest.size())
            );
            output.flush();
            if (!output) {
                throw std::runtime_error("failed while finalizing checkpoint header");
            }
        }
        sync_file(temporary);
        const LoadedCheckpoint verified = read_checkpoint(temporary, {});
        const std::uint64_t verified_model_hash =
            state.architecture == ImageGenerationArchitecture::resonant_fabric
            ? verified.state.resonant_fabric.deterministic_hash()
            : verified.state.fabric.deterministic_hash();
        const std::uint64_t expected_model_hash =
            state.architecture == ImageGenerationArchitecture::resonant_fabric
            ? state.resonant_fabric.deterministic_hash()
            : state.fabric.deterministic_hash();
        if (verified.state.profile != state.profile ||
            verified.state.architecture != state.architecture ||
            verified.state.master_seed != state.master_seed ||
            verified.state.training_step != state.training_step ||
            verified_model_hash != expected_model_hash) {
            throw std::runtime_error("temporary image-generation checkpoint verification failed");
        }
        std::error_code rename_error;
        std::filesystem::rename(temporary, path, rename_error);
        if (rename_error) {
            const std::filesystem::path backup =
                path.string() + ".backup." + std::to_string(stamp);
            std::error_code backup_error;
            std::filesystem::rename(path, backup, backup_error);
            if (backup_error) {
                throw std::runtime_error("unable to preserve previous image-generation checkpoint");
            }
            rename_error.clear();
            std::filesystem::rename(temporary, path, rename_error);
            if (rename_error) {
                std::error_code restore_error;
                std::filesystem::rename(backup, path, restore_error);
                if (restore_error) {
                    throw std::runtime_error(
                        "checkpoint install and recovery failed; backup retained at " +
                        backup.string()
                    );
                }
                throw std::runtime_error("unable to install image-generation checkpoint");
            }
            std::filesystem::remove(backup, backup_error);
        }
        sync_directory(parent);
    } catch (...) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary, cleanup_error);
        throw;
    }
}

ImageGenerationCheckpointState load_image_generation_checkpoint(
    const std::filesystem::path& path,
    const ImageGenerationCheckpointLimits limits
) {
    return read_checkpoint(path, limits).state;
}

ImageGenerationCheckpointSummary inspect_image_generation_checkpoint(
    const std::filesystem::path& path,
    const ImageGenerationCheckpointLimits limits
) {
    LoadedCheckpoint loaded = read_checkpoint(path, limits);
    ImageGenerationCheckpointSummary summary;
    summary.format_version = loaded.format_version;
    summary.file_bytes = loaded.file_bytes;
    summary.file_sha256 = rlf::core::sha256_hex(rlf::core::sha256_file(path));
    summary.payload_sha256 = rlf::core::sha256_hex(loaded.payload_digest);
    summary.profile = loaded.state.profile;
    summary.architecture = loaded.state.architecture;
    summary.master_seed = loaded.state.master_seed;
    summary.training_step = loaded.state.training_step;
    if (loaded.state.architecture ==
        ImageGenerationArchitecture::resonant_fabric) {
        summary.images_seen =
            loaded.state.resonant_fabric.operation_stats().training_examples;
        summary.learned_modes = loaded.state.resonant_fabric.modes().size();
        const PromptSemanticStats prompt_stats =
            loaded.state.resonant_fabric.prompt_semantics().stats();
        summary.prompt_language_records = prompt_stats.records_seen;
        summary.prompt_language_words = prompt_stats.words_seen;
        summary.prompt_semantic_modes =
            loaded.state.resonant_fabric.prompt_semantics().modes().size();
        summary.deterministic_model_hash =
            loaded.state.resonant_fabric.deterministic_hash();
    } else {
        summary.images_seen = loaded.state.fabric.images_seen();
        summary.source_images = loaded.state.fabric.sources().size();
        summary.tile_prototypes = loaded.state.fabric.tiles().size();
        summary.deterministic_model_hash =
            loaded.state.fabric.deterministic_hash();
    }
    summary.completed_shards = loaded.state.completed_shards.size();
    return summary;
}

}  // namespace rlf::solstice
