#include "rlf/storage/rlf5_checkpoint.hpp"

#include "storage/binary_codec.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rlf::storage {
namespace {

constexpr std::array<std::uint8_t, 8U> magic{
    'R', 'L', 'F', '5', 'C', 'K', 'P', '7'
};
constexpr std::size_t header_size = 8U + 4U + 8U + 8U;

void write_phase_vector(
    detail::BufferWriter& writer,
    const core::PhaseVector& value
) {
    writer.write_u64(value.size());
    for (const float angle : value.angles()) {
        writer.write_float(angle);
    }
}

[[nodiscard]] core::PhaseVector read_phase_vector(
    detail::BufferReader& reader,
    const std::size_t expected_dimension,
    const Rlf5CheckpointLoadOptions& options
) {
    const std::size_t dimension = detail::checked_size(
        reader.read_u64(), options.maximum_dimension,
        "RLF-5 phase-vector dimension"
    );
    if (dimension != expected_dimension) {
        throw std::runtime_error("RLF-5 checkpoint dimension mismatch");
    }
    constexpr float tau = 2.0F * std::numbers::pi_v<float>;
    std::vector<float> angles;
    angles.reserve(dimension);
    for (std::size_t index = 0U; index < dimension; ++index) {
        const float angle = reader.read_float();
        if (!std::isfinite(angle) || angle < 0.0F || angle >= tau) {
            throw std::runtime_error("invalid RLF-5 phase angle");
        }
        angles.push_back(angle);
    }
    return core::PhaseVector(std::move(angles));
}

void write_config(
    detail::BufferWriter& writer,
    const core::LanguageFabricConfig& config
) {
    writer.write_u64(config.phase_dimension);
    writer.write_u64(config.maximum_lexemes);
    writer.write_u64(config.maximum_merges);
    writer.write_u64(config.minimum_pair_support);
    writer.write_u64(config.maximum_contexts);
    writer.write_u64(config.maximum_context_order);
    writer.write_u64(config.minimum_context_support);
    writer.write_u64(config.maximum_constructions);
    writer.write_u64(config.minimum_construction_support);
    writer.write_u64(config.maximum_generation_tokens);
    writer.write_u64(config.maximum_semantic_values);
    writer.write_u64(config.maximum_surfaces_per_concept);
    writer.write_double(config.smoothing);
    writer.write_double(config.minimum_lexical_score);
    writer.write_double(config.construction_support_weight);
    writer.write_double(config.literal_match_weight);
    writer.write_double(config.slot_match_weight);
}

[[nodiscard]] core::LanguageFabricConfig read_config(
    detail::BufferReader& reader,
    const Rlf5CheckpointLoadOptions& options
) {
    core::LanguageFabricConfig config;
    config.phase_dimension = detail::checked_size(
        reader.read_u64(), options.maximum_dimension, "RLF-5 phase dimension"
    );
    config.maximum_lexemes = detail::checked_size(
        reader.read_u64(), options.maximum_lexemes, "RLF-5 maximum lexemes"
    );
    config.maximum_merges = detail::checked_size(
        reader.read_u64(), options.maximum_merges, "RLF-5 maximum merges"
    );
    config.minimum_pair_support = detail::checked_size(
        reader.read_u64(), std::numeric_limits<std::size_t>::max(),
        "RLF-5 minimum pair support"
    );
    config.maximum_contexts = detail::checked_size(
        reader.read_u64(), options.maximum_contexts, "RLF-5 maximum contexts"
    );
    config.maximum_context_order = detail::checked_size(
        reader.read_u64(), options.maximum_pattern_items,
        "RLF-5 maximum context order"
    );
    config.minimum_context_support = detail::checked_size(
        reader.read_u64(), std::numeric_limits<std::size_t>::max(),
        "RLF-5 minimum context support"
    );
    config.maximum_constructions = detail::checked_size(
        reader.read_u64(), options.maximum_constructions,
        "RLF-5 maximum constructions"
    );
    config.minimum_construction_support = detail::checked_size(
        reader.read_u64(), std::numeric_limits<std::size_t>::max(),
        "RLF-5 minimum construction support"
    );
    config.maximum_generation_tokens = detail::checked_size(
        reader.read_u64(), options.maximum_pattern_items,
        "RLF-5 maximum generation tokens"
    );
    config.maximum_semantic_values = detail::checked_size(
        reader.read_u64(), options.maximum_concepts,
        "RLF-5 maximum semantic values"
    );
    config.maximum_surfaces_per_concept = detail::checked_size(
        reader.read_u64(), options.maximum_surfaces,
        "RLF-5 maximum surfaces per concept"
    );
    config.smoothing = reader.read_double();
    config.minimum_lexical_score = reader.read_double();
    config.construction_support_weight = reader.read_double();
    config.literal_match_weight = reader.read_double();
    config.slot_match_weight = reader.read_double();
    return config;
}

void write_stats(
    detail::BufferWriter& writer,
    const core::LanguageFabricStats& stats
) {
    writer.write_u64(stats.raw_bytes_seen);
    writer.write_u64(stats.lexicon_merges);
    writer.write_u64(stats.language_tokens_seen);
    writer.write_u64(stats.contexts_created);
    writer.write_u64(stats.contexts_updated);
    writer.write_u64(stats.supervised_examples_seen);
    writer.write_u64(stats.concepts_created);
    writer.write_u64(stats.constructions_created);
    writer.write_u64(stats.parse_queries);
    writer.write_u64(stats.generation_queries);
    writer.write_u64(stats.answer_queries);
}

[[nodiscard]] core::LanguageFabricStats read_stats(
    detail::BufferReader& reader
) {
    return {
        reader.read_u64(), reader.read_u64(), reader.read_u64(),
        reader.read_u64(), reader.read_u64(), reader.read_u64(),
        reader.read_u64(), reader.read_u64(), reader.read_u64(),
        reader.read_u64(), reader.read_u64(),
    };
}

[[nodiscard]] std::vector<std::uint8_t> encode_payload(
    const core::LanguageFabric& fabric
) {
    const auto snapshot = fabric.snapshot();
    detail::BufferWriter writer;
    write_config(writer, snapshot.config);
    writer.write_u64(snapshot.seed);
    writer.write_u64(snapshot.next_lexeme_id);
    writer.write_u64(snapshot.next_context_id);
    writer.write_u64(snapshot.next_concept_id);
    writer.write_u64(snapshot.next_construction_id);
    write_stats(writer, snapshot.stats);

    writer.write_u64(snapshot.lexemes.size());
    for (const auto& lexeme : snapshot.lexemes) {
        writer.write_u64(lexeme.id);
        writer.write_string(lexeme.bytes);
        writer.write_u64(lexeme.support);
        write_phase_vector(writer, lexeme.key);
    }

    writer.write_u64(snapshot.merges.size());
    for (const auto& merge : snapshot.merges) {
        writer.write_u64(merge.left_id);
        writer.write_u64(merge.right_id);
        writer.write_u64(merge.result_id);
        writer.write_u64(merge.support);
        writer.write_u64(static_cast<std::uint64_t>(merge.description_gain));
    }

    writer.write_u64(snapshot.contexts.size());
    for (const auto& context : snapshot.contexts) {
        writer.write_u64(context.id);
        writer.write_u64(context.history.size());
        for (const auto token : context.history) {
            writer.write_u64(token);
        }
        writer.write_u64(context.support);
        writer.write_u64(context.outcomes.size());
        for (const auto& outcome : context.outcomes) {
            writer.write_u64(outcome.token_id);
            writer.write_u64(outcome.count);
        }
    }

    writer.write_u64(snapshot.concepts.size());
    for (const auto& semantic_concept : snapshot.concepts) {
        writer.write_u64(semantic_concept.id);
        writer.write_u8(static_cast<std::uint8_t>(semantic_concept.role));
        writer.write_string(semantic_concept.value);
        write_phase_vector(writer, semantic_concept.key);
        writer.write_u64(semantic_concept.support);
        writer.write_u64(semantic_concept.surfaces.size());
        for (const auto& surface : semantic_concept.surfaces) {
            writer.write_u64(surface.token_id);
            writer.write_u64(surface.count);
            writer.write_double(surface.association);
        }
    }

    writer.write_u64(snapshot.constructions.size());
    for (const auto& construction : snapshot.constructions) {
        writer.write_u64(construction.id);
        writer.write_u8(static_cast<std::uint8_t>(construction.act));
        writer.write_u64(construction.support);
        writer.write_double(construction.confidence);
        writer.write_u64(construction.pattern.size());
        for (const auto& item : construction.pattern) {
            writer.write_u8(static_cast<std::uint8_t>(item.kind));
            writer.write_u64(item.token_id);
            writer.write_u8(static_cast<std::uint8_t>(item.role));
            writer.write_u8(item.surface_form);
        }
    }
    return writer.take();
}

[[nodiscard]] core::LanguageFabricSnapshot decode_payload(
    const std::span<const std::uint8_t> payload,
    const Rlf5CheckpointLoadOptions& options
) {
    detail::BufferReader reader(payload);
    core::LanguageFabricSnapshot snapshot;
    snapshot.config = read_config(reader, options);
    snapshot.seed = reader.read_u64();
    snapshot.next_lexeme_id = reader.read_u64();
    snapshot.next_context_id = reader.read_u64();
    snapshot.next_concept_id = reader.read_u64();
    snapshot.next_construction_id = reader.read_u64();
    snapshot.stats = read_stats(reader);

    const std::size_t lexeme_count = detail::checked_size(
        reader.read_u64(), options.maximum_lexemes, "RLF-5 lexeme count"
    );
    snapshot.lexemes.reserve(lexeme_count);
    for (std::size_t index = 0U; index < lexeme_count; ++index) {
        core::LanguageLexeme lexeme;
        lexeme.id = reader.read_u64();
        lexeme.bytes = reader.read_string(options.maximum_lexeme_bytes);
        lexeme.support = reader.read_u64();
        lexeme.key = read_phase_vector(
            reader, snapshot.config.phase_dimension, options
        );
        snapshot.lexemes.push_back(std::move(lexeme));
    }

    const std::size_t merge_count = detail::checked_size(
        reader.read_u64(), options.maximum_merges, "RLF-5 merge count"
    );
    snapshot.merges.reserve(merge_count);
    for (std::size_t index = 0U; index < merge_count; ++index) {
        snapshot.merges.push_back({
            reader.read_u64(), reader.read_u64(), reader.read_u64(),
            reader.read_u64(), static_cast<std::int64_t>(reader.read_u64()),
        });
    }

    const std::size_t context_count = detail::checked_size(
        reader.read_u64(), options.maximum_contexts, "RLF-5 context count"
    );
    std::size_t total_outcomes = 0U;
    snapshot.contexts.reserve(context_count);
    for (std::size_t index = 0U; index < context_count; ++index) {
        core::LanguageContext context;
        context.id = reader.read_u64();
        const std::size_t history_size = detail::checked_size(
            reader.read_u64(), snapshot.config.maximum_context_order,
            "RLF-5 context history"
        );
        context.history.reserve(history_size);
        for (std::size_t item = 0U; item < history_size; ++item) {
            context.history.push_back(reader.read_u64());
        }
        context.support = reader.read_u64();
        const std::size_t outcome_count = detail::checked_size(
            reader.read_u64(), options.maximum_outcomes,
            "RLF-5 context outcomes"
        );
        if (total_outcomes > options.maximum_outcomes - outcome_count) {
            throw std::runtime_error("RLF-5 outcomes exceed configured limit");
        }
        total_outcomes += outcome_count;
        context.outcomes.reserve(outcome_count);
        for (std::size_t item = 0U; item < outcome_count; ++item) {
            context.outcomes.push_back({reader.read_u64(), reader.read_u64()});
        }
        snapshot.contexts.push_back(std::move(context));
    }

    const std::size_t concept_count = detail::checked_size(
        reader.read_u64(), options.maximum_concepts, "RLF-5 concept count"
    );
    std::size_t total_surfaces = 0U;
    snapshot.concepts.reserve(concept_count);
    for (std::size_t index = 0U; index < concept_count; ++index) {
        core::LanguageConcept semantic_concept;
        semantic_concept.id = reader.read_u64();
        const auto role = reader.read_u8();
        if (role > static_cast<std::uint8_t>(core::LanguageRole::location)) {
            throw std::runtime_error("invalid RLF-5 language role");
        }
        semantic_concept.role = static_cast<core::LanguageRole>(role);
        semantic_concept.value = reader.read_string(options.maximum_string_bytes);
        semantic_concept.key = read_phase_vector(
            reader, snapshot.config.phase_dimension, options
        );
        semantic_concept.support = reader.read_u64();
        const std::size_t surface_count = detail::checked_size(
            reader.read_u64(), options.maximum_surfaces,
            "RLF-5 concept surfaces"
        );
        if (total_surfaces > options.maximum_surfaces - surface_count) {
            throw std::runtime_error("RLF-5 surfaces exceed configured limit");
        }
        total_surfaces += surface_count;
        semantic_concept.surfaces.reserve(surface_count);
        for (std::size_t item = 0U; item < surface_count; ++item) {
            semantic_concept.surfaces.push_back({
                reader.read_u64(), reader.read_u64(), reader.read_double(),
            });
        }
        snapshot.concepts.push_back(std::move(semantic_concept));
    }

    const std::size_t construction_count = detail::checked_size(
        reader.read_u64(), options.maximum_constructions,
        "RLF-5 construction count"
    );
    snapshot.constructions.reserve(construction_count);
    for (std::size_t index = 0U; index < construction_count; ++index) {
        core::LanguageConstruction construction;
        construction.id = reader.read_u64();
        const auto act = reader.read_u8();
        if (act > static_cast<std::uint8_t>(core::LanguageAct::answer_location)) {
            throw std::runtime_error("invalid RLF-5 language act");
        }
        construction.act = static_cast<core::LanguageAct>(act);
        construction.support = reader.read_u64();
        construction.confidence = reader.read_double();
        const std::size_t item_count = detail::checked_size(
            reader.read_u64(), options.maximum_pattern_items,
            "RLF-5 construction pattern"
        );
        construction.pattern.reserve(item_count);
        for (std::size_t item_index = 0U; item_index < item_count; ++item_index) {
            const auto kind = reader.read_u8();
            const auto token_id = reader.read_u64();
            const auto role = reader.read_u8();
            const auto form = reader.read_u8();
            if (kind > static_cast<std::uint8_t>(core::LanguagePatternKind::slot) ||
                role > static_cast<std::uint8_t>(core::LanguageRole::location)) {
                throw std::runtime_error("invalid RLF-5 construction item");
            }
            construction.pattern.push_back({
                static_cast<core::LanguagePatternKind>(kind), token_id,
                static_cast<core::LanguageRole>(role), form,
            });
        }
        snapshot.constructions.push_back(std::move(construction));
    }
    if (!reader.empty()) {
        throw std::runtime_error("RLF-5 checkpoint contains trailing payload");
    }
    return snapshot;
}

struct ParsedCheckpoint final {
    std::vector<std::uint8_t> payload;
    std::uint64_t checksum{};
    std::size_t file_bytes{};
};

[[nodiscard]] ParsedCheckpoint parse_file(
    const std::filesystem::path& path,
    const Rlf5CheckpointLoadOptions& options
) {
    const auto bytes = detail::read_file(path, options.maximum_file_bytes);
    if (bytes.size() < header_size) {
        throw std::runtime_error("truncated RLF-5 checkpoint header");
    }
    detail::BufferReader reader(bytes);
    for (const std::uint8_t expected : magic) {
        if (reader.read_u8() != expected) {
            throw std::runtime_error("invalid RLF-5 checkpoint magic");
        }
    }
    const auto version = reader.read_u32();
    if (version != rlf5_checkpoint_format_version) {
        throw std::runtime_error("unsupported RLF-5 checkpoint version");
    }
    const std::size_t payload_size = detail::checked_size(
        reader.read_u64(), options.maximum_file_bytes,
        "RLF-5 checkpoint payload"
    );
    const std::uint64_t expected_checksum = reader.read_u64();
    if (payload_size != reader.remaining()) {
        throw std::runtime_error("RLF-5 checkpoint payload size mismatch");
    }
    auto payload = reader.read_bytes(payload_size);
    const std::uint64_t actual_checksum = detail::checksum(payload);
    if (actual_checksum != expected_checksum) {
        throw std::runtime_error("RLF-5 checkpoint checksum mismatch");
    }
    return {std::move(payload), actual_checksum, bytes.size()};
}

}  // namespace

void save_rlf5_checkpoint(
    const std::filesystem::path& path,
    const core::LanguageFabric& fabric
) {
    const auto payload = encode_payload(fabric);
    detail::BufferWriter writer;
    writer.write_bytes(magic);
    writer.write_u32(rlf5_checkpoint_format_version);
    writer.write_u64(payload.size());
    writer.write_u64(detail::checksum(payload));
    writer.write_bytes(payload);
    detail::write_file_transactionally(path, writer.bytes());
}

core::LanguageFabric load_rlf5_checkpoint(
    const std::filesystem::path& path,
    const Rlf5CheckpointLoadOptions& options
) {
    const auto parsed = parse_file(path, options);
    return core::LanguageFabric::from_snapshot(
        decode_payload(parsed.payload, options)
    );
}

Rlf5CheckpointSummary inspect_rlf5_checkpoint(
    const std::filesystem::path& path,
    const Rlf5CheckpointLoadOptions& options
) {
    const auto parsed = parse_file(path, options);
    const auto snapshot = decode_payload(parsed.payload, options);
    std::size_t outcomes = 0U;
    for (const auto& context : snapshot.contexts) {
        outcomes += context.outcomes.size();
    }
    return {
        rlf5_checkpoint_format_version,
        snapshot.seed,
        snapshot.config.phase_dimension,
        snapshot.lexemes.size(),
        snapshot.merges.size(),
        snapshot.contexts.size(),
        outcomes,
        snapshot.concepts.size(),
        snapshot.constructions.size(),
        parsed.checksum,
        parsed.file_bytes,
    };
}

}  // namespace rlf::storage
