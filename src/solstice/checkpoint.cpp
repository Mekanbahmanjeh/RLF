#include "rlf/solstice/checkpoint.hpp"

#include "rlf/solstice/profile.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rlf::solstice {
namespace {

constexpr std::array<char, 8U> checkpoint_magic{
    'R', 'L', 'F', 'S', 'L', 'S', 'T', '1'
};
constexpr std::uint32_t checkpoint_version = 6U;
constexpr std::uint32_t minimum_checkpoint_version = 1U;
constexpr std::uint64_t video_extension_magic = 0x524C465649443631ULL; // RLFVID61
constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;

class BinaryWriter final {
public:
    explicit BinaryWriter(std::ostream& output) : output_(output) {}

    void u8(const std::uint8_t value) {
        output_.put(static_cast<char>(value));
        if (!output_) {
            throw std::runtime_error("failed while streaming Solstice checkpoint");
        }
        checksum_ ^= value;
        checksum_ *= fnv_prime;
        if (bytes_written_ == std::numeric_limits<std::uint64_t>::max()) {
            throw std::runtime_error("Solstice checkpoint payload size overflow");
        }
        ++bytes_written_;
    }

    void u32(const std::uint32_t value) {
        for (unsigned int index = 0U; index < 4U; ++index) {
            u8(static_cast<std::uint8_t>((value >> (index * 8U)) & 0xFFU));
        }
    }

    void u64(const std::uint64_t value) {
        for (unsigned int index = 0U; index < 8U; ++index) {
            u8(static_cast<std::uint8_t>((value >> (index * 8U)) & 0xFFULL));
        }
    }

    void size(const std::size_t value) { u64(static_cast<std::uint64_t>(value)); }
    void boolean(const bool value) { u8(value ? 1U : 0U); }
    void floating(const float value) { u32(std::bit_cast<std::uint32_t>(value)); }
    void floating(const double value) { u64(std::bit_cast<std::uint64_t>(value)); }

    void string(const std::string_view value) {
        size(value.size());
        for (const char character : value) {
            u8(static_cast<std::uint8_t>(static_cast<unsigned char>(character)));
        }
    }

    [[nodiscard]] std::uint64_t bytes_written() const noexcept { return bytes_written_; }
    [[nodiscard]] std::uint64_t checksum() const noexcept { return checksum_; }

private:
    std::ostream& output_;
    std::uint64_t bytes_written_{};
    std::uint64_t checksum_{fnv_offset_basis};
};

class BinaryReader final {
public:
    BinaryReader(
        std::istream& input,
        const std::uint64_t payload_bytes,
        SolsticeCheckpointLimits limits
    ) : input_(input), remaining_(payload_bytes), limits_(limits) {}

    [[nodiscard]] std::uint8_t u8() {
        require(1U);
        const int character = input_.get();
        if (character == std::char_traits<char>::eof()) {
            throw std::runtime_error("truncated Solstice checkpoint payload");
        }
        const auto value = static_cast<std::uint8_t>(
            static_cast<unsigned char>(character)
        );
        --remaining_;
        checksum_ ^= value;
        checksum_ *= fnv_prime;
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
        if (value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error("Solstice checkpoint size value exceeds platform range");
        }
        return static_cast<std::size_t>(value);
    }

    [[nodiscard]] std::size_t count() {
        const std::size_t value = size();
        if (value > limits_.maximum_collection_entries) {
            throw std::runtime_error("Solstice checkpoint collection exceeds configured limit");
        }
        return value;
    }

    [[nodiscard]] bool boolean() {
        const std::uint8_t value = u8();
        if (value > 1U) {
            throw std::runtime_error("invalid boolean in Solstice checkpoint");
        }
        return value != 0U;
    }

    [[nodiscard]] float float32() { return std::bit_cast<float>(u32()); }
    [[nodiscard]] double float64() { return std::bit_cast<double>(u64()); }

    [[nodiscard]] std::string string() {
        const std::size_t length = size();
        if (length > limits_.maximum_string_bytes) {
            throw std::runtime_error("Solstice checkpoint string exceeds configured limit");
        }
        require(length);
        std::string value(length, '\0');
        for (char& character : value) {
            character = static_cast<char>(u8());
        }
        return value;
    }

    [[nodiscard]] bool finished() const noexcept {
        return remaining_ == 0U;
    }

    [[nodiscard]] std::uint64_t checksum() const noexcept { return checksum_; }

private:
    void require(const std::size_t count_value) const {
        if (static_cast<std::uint64_t>(count_value) > remaining_) {
            throw std::runtime_error("truncated Solstice checkpoint payload");
        }
    }

    std::istream& input_;
    std::uint64_t remaining_{};
    SolsticeCheckpointLimits limits_;
    std::uint64_t checksum_{fnv_offset_basis};
};

void write_tokenizer_config(BinaryWriter& writer, const TokenizerConfig& config) {
    writer.size(config.maximum_vocabulary);
    writer.size(config.maximum_merges);
    writer.size(config.minimum_pair_support);
    writer.size(config.maximum_piece_bytes);
    writer.size(config.maximum_training_bytes);
}

[[nodiscard]] TokenizerConfig read_tokenizer_config(BinaryReader& reader) {
    TokenizerConfig config;
    config.maximum_vocabulary = reader.size();
    config.maximum_merges = reader.size();
    config.minimum_pair_support = reader.size();
    config.maximum_piece_bytes = reader.size();
    config.maximum_training_bytes = reader.size();
    return config;
}

void write_language_config(
    BinaryWriter& writer,
    const HierarchicalLanguageConfig& config
) {
    writer.size(config.context_orders.size());
    for (const std::size_t order : config.context_orders) {
        writer.size(order);
    }
    writer.size(config.maximum_contexts);
    writer.size(config.maximum_episodes);
    writer.size(config.maximum_episode_cue_tokens);
    writer.size(config.maximum_episode_response_tokens);
    writer.size(config.maximum_generation_tokens);
    writer.size(config.prediction_candidate_limit);
    writer.floating(config.smoothing);
    writer.floating(config.long_context_weight);
    writer.floating(config.episode_conditioning_weight);
    writer.floating(config.repetition_penalty);
}

[[nodiscard]] HierarchicalLanguageConfig read_language_config(BinaryReader& reader) {
    HierarchicalLanguageConfig config;
    config.context_orders.resize(reader.count());
    for (std::size_t& order : config.context_orders) {
        order = reader.size();
    }
    config.maximum_contexts = reader.size();
    config.maximum_episodes = reader.size();
    config.maximum_episode_cue_tokens = reader.size();
    config.maximum_episode_response_tokens = reader.size();
    config.maximum_generation_tokens = reader.size();
    config.prediction_candidate_limit = reader.size();
    config.smoothing = reader.float64();
    config.long_context_weight = reader.float64();
    config.episode_conditioning_weight = reader.float64();
    config.repetition_penalty = reader.float64();
    return config;
}

void write_sparse_router_config(BinaryWriter& writer, const SparseRouterConfig& config);
[[nodiscard]] SparseRouterConfig read_sparse_router_config(BinaryReader& reader);

void write_vision_config(BinaryWriter& writer, const VisionConfig& config) {
    writer.size(config.patch_size);
    writer.size(config.patch_sizes.size());
    for (const std::size_t patch_size : config.patch_sizes) {
        writer.size(patch_size);
    }
    writer.size(config.maximum_modes);
    writer.size(config.maximum_examples);
    writer.size(config.maximum_regions);
    writer.size(config.maximum_concepts_per_mode);
    writer.floating(config.mode_creation_similarity);
    writer.floating(config.example_match_similarity);
    writer.floating(config.local_learning_rate);
    writer.size(config.descriptor_dimensions);
    writer.size(config.maximum_input_side);
    writer.size(config.maximum_patches);
    writer.size(config.retrieval_query_batch);
    writer.size(config.retrieval_candidate_batch);
    writer.size(config.training_patch_batch);
    writer.size(config.sparse_routing_minimum_modes);
    write_sparse_router_config(writer, config.sparse_router);
}

[[nodiscard]] VisionConfig read_vision_config(
    BinaryReader& reader,
    const std::uint32_t version
) {
    VisionConfig config;
    config.patch_size = reader.size();
    if (version >= 2U) {
        config.patch_sizes.resize(reader.count());
        for (std::size_t& patch_size : config.patch_sizes) {
            patch_size = reader.size();
        }
    }
    config.maximum_modes = reader.size();
    config.maximum_examples = reader.size();
    config.maximum_regions = reader.size();
    config.maximum_concepts_per_mode = reader.size();
    config.mode_creation_similarity = reader.float64();
    config.example_match_similarity = reader.float64();
    config.local_learning_rate = reader.float64();
    if (version >= 3U) {
        config.descriptor_dimensions = reader.size();
        config.maximum_input_side = reader.size();
        config.maximum_patches = reader.size();
        config.retrieval_query_batch = reader.size();
        config.retrieval_candidate_batch = reader.size();
        config.training_patch_batch = reader.size();
        if (version >= 4U) {
            config.sparse_routing_minimum_modes = reader.size();
            config.sparse_router = read_sparse_router_config(reader);
        }
    } else {
        config.descriptor_dimensions = 16U;
    }
    return config;
}

void write_tool_config(BinaryWriter& writer, const ToolRouterConfig& config) {
    writer.size(config.maximum_tools);
    writer.size(config.maximum_keywords_per_tool);
    writer.floating(config.minimum_confidence);
}

[[nodiscard]] ToolRouterConfig read_tool_config(BinaryReader& reader) {
    ToolRouterConfig config;
    config.maximum_tools = reader.size();
    config.maximum_keywords_per_tool = reader.size();
    config.minimum_confidence = reader.float64();
    return config;
}

void write_tokenizer(BinaryWriter& writer, const TokenizerSnapshot& snapshot) {
    write_tokenizer_config(writer, snapshot.config);
    writer.size(snapshot.pieces.size());
    for (const TokenPiece& piece : snapshot.pieces) {
        writer.u32(piece.id);
        writer.string(piece.bytes);
        writer.boolean(piece.special);
    }
    writer.size(snapshot.merges.size());
    for (const TokenMerge& merge : snapshot.merges) {
        writer.u32(merge.left);
        writer.u32(merge.right);
        writer.u32(merge.result);
        writer.u64(merge.support);
    }
}

[[nodiscard]] TokenizerSnapshot read_tokenizer(BinaryReader& reader) {
    TokenizerSnapshot snapshot;
    snapshot.config = read_tokenizer_config(reader);
    snapshot.pieces.resize(reader.count());
    for (TokenPiece& piece : snapshot.pieces) {
        piece.id = reader.u32();
        piece.bytes = reader.string();
        piece.special = reader.boolean();
    }
    snapshot.merges.resize(reader.count());
    for (TokenMerge& merge : snapshot.merges) {
        merge.left = reader.u32();
        merge.right = reader.u32();
        merge.result = reader.u32();
        merge.support = reader.u64();
    }
    return snapshot;
}

void write_language(BinaryWriter& writer, const HierarchicalLanguageSnapshot& snapshot) {
    write_language_config(writer, snapshot.config);
    writer.u64(snapshot.next_context_id);
    writer.u64(snapshot.next_episode_id);
    writer.u64(snapshot.tokens_seen);
    writer.size(snapshot.contexts.size());
    for (const PredictiveContext& context : snapshot.contexts) {
        writer.u64(context.id);
        writer.size(context.history.size());
        for (const TokenId token : context.history) {
            writer.u32(token);
        }
        writer.u64(context.support);
        writer.size(context.outcomes.size());
        for (const TokenOutcome& outcome : context.outcomes) {
            writer.u32(outcome.token);
            writer.u64(outcome.count);
        }
    }
    writer.size(snapshot.episodes.size());
    for (const LanguageEpisode& episode : snapshot.episodes) {
        writer.u64(episode.id);
        writer.size(episode.cue.size());
        for (const TokenId token : episode.cue) {
            writer.u32(token);
        }
        writer.size(episode.response.size());
        for (const TokenId token : episode.response) {
            writer.u32(token);
        }
        writer.u64(episode.support);
    }
}

[[nodiscard]] HierarchicalLanguageSnapshot read_language(BinaryReader& reader) {
    HierarchicalLanguageSnapshot snapshot;
    snapshot.config = read_language_config(reader);
    snapshot.next_context_id = reader.u64();
    snapshot.next_episode_id = reader.u64();
    snapshot.tokens_seen = reader.u64();
    snapshot.contexts.resize(reader.count());
    for (PredictiveContext& context : snapshot.contexts) {
        context.id = reader.u64();
        context.history.resize(reader.count());
        for (TokenId& token : context.history) {
            token = reader.u32();
        }
        context.support = reader.u64();
        context.outcomes.resize(reader.count());
        for (TokenOutcome& outcome : context.outcomes) {
            outcome.token = reader.u32();
            outcome.count = reader.u64();
        }
    }
    snapshot.episodes.resize(reader.count());
    for (LanguageEpisode& episode : snapshot.episodes) {
        episode.id = reader.u64();
        episode.cue.resize(reader.count());
        for (TokenId& token : episode.cue) {
            token = reader.u32();
        }
        episode.response.resize(reader.count());
        for (TokenId& token : episode.response) {
            token = reader.u32();
        }
        episode.support = reader.u64();
    }
    return snapshot;
}

void write_vision(BinaryWriter& writer, const VisionSnapshot& snapshot) {
    write_vision_config(writer, snapshot.config);
    writer.u64(snapshot.next_mode_id);
    writer.u64(snapshot.next_example_id);
    writer.u64(snapshot.images_seen);
    writer.size(snapshot.modes.size());
    for (const VisualMode& mode : snapshot.modes) {
        writer.u64(mode.id);
        writer.size(mode.prototype.size());
        for (const float value : mode.prototype) {
            writer.floating(value);
        }
        writer.u64(mode.support);
        writer.size(mode.concepts.size());
        for (const VisualConceptCount& concept_name : mode.concepts) {
            writer.string(concept_name.concept_name);
            writer.u64(concept_name.count);
        }
    }
    writer.size(snapshot.examples.size());
    for (const VisualExample& example : snapshot.examples) {
        writer.u64(example.id);
        writer.size(example.global_descriptor.size());
        for (const float value : example.global_descriptor) {
            writer.floating(value);
        }
        writer.string(example.caption);
        writer.size(example.concepts.size());
        for (const std::string& concept_name : example.concepts) {
            writer.string(concept_name);
        }
        writer.u64(example.support);
    }
}

[[nodiscard]] VisionSnapshot read_vision(
    BinaryReader& reader,
    const std::uint32_t version
) {
    VisionSnapshot snapshot;
    snapshot.config = read_vision_config(reader, version);
    snapshot.next_mode_id = reader.u64();
    snapshot.next_example_id = reader.u64();
    snapshot.images_seen = reader.u64();
    snapshot.modes.resize(reader.count());
    for (VisualMode& mode : snapshot.modes) {
        mode.id = reader.u64();
        mode.prototype.resize(reader.count());
        for (float& value : mode.prototype) {
            value = reader.float32();
        }
        mode.support = reader.u64();
        mode.concepts.resize(reader.count());
        for (VisualConceptCount& concept_name : mode.concepts) {
            concept_name.concept_name = reader.string();
            concept_name.count = reader.u64();
        }
    }
    snapshot.examples.resize(reader.count());
    for (VisualExample& example : snapshot.examples) {
        example.id = reader.u64();
        example.global_descriptor.resize(reader.count());
        for (float& value : example.global_descriptor) {
            value = reader.float32();
        }
        example.caption = reader.string();
        example.concepts.resize(reader.count());
        for (std::string& concept_name : example.concepts) {
            concept_name = reader.string();
        }
        example.support = reader.u64();
    }
    return snapshot;
}

void write_tool_router(BinaryWriter& writer, const ToolRouterSnapshot& snapshot) {
    write_tool_config(writer, snapshot.config);
    writer.size(snapshot.routes.size());
    for (const ToolRoute& route : snapshot.routes) {
        writer.string(route.tool_name);
        writer.u64(route.examples);
        writer.size(route.keywords.size());
        for (const ToolKeywordCount& keyword : route.keywords) {
            writer.string(keyword.keyword);
            writer.u64(keyword.count);
        }
    }
}

[[nodiscard]] ToolRouterSnapshot read_tool_router(BinaryReader& reader) {
    ToolRouterSnapshot snapshot;
    snapshot.config = read_tool_config(reader);
    snapshot.routes.resize(reader.count());
    for (ToolRoute& route : snapshot.routes) {
        route.tool_name = reader.string();
        route.examples = reader.u64();
        route.keywords.resize(reader.count());
        for (ToolKeywordCount& keyword : route.keywords) {
            keyword.keyword = reader.string();
            keyword.count = reader.u64();
        }
    }
    return snapshot;
}


void write_abstraction(BinaryWriter& writer, const AbstractionSnapshot& snapshot) {
    writer.size(snapshot.config.maximum_facts);
    writer.size(snapshot.config.maximum_rules);
    writer.size(snapshot.config.maximum_inference_depth);
    writer.size(snapshot.config.maximum_derivations_per_query);
    writer.floating(snapshot.config.minimum_confidence);
    writer.floating(snapshot.config.inferred_confidence_decay);
    writer.u64(snapshot.next_fact_id);
    writer.u64(snapshot.next_rule_id);
    writer.size(snapshot.facts.size());
    for (const ReasoningFact& fact : snapshot.facts) {
        writer.u64(fact.id);
        writer.string(fact.subject);
        writer.string(fact.relation);
        writer.string(fact.object);
        writer.floating(fact.confidence);
        writer.u64(fact.support);
        writer.boolean(fact.inferred);
        writer.string(fact.provenance);
    }
    writer.size(snapshot.rules.size());
    for (const ReasoningRule& rule : snapshot.rules) {
        writer.u64(rule.id);
        writer.string(rule.name);
        writer.size(rule.premises.size());
        for (const RelationalPattern& premise : rule.premises) {
            writer.string(premise.subject);
            writer.string(premise.relation);
            writer.string(premise.object);
        }
        writer.string(rule.conclusion.subject);
        writer.string(rule.conclusion.relation);
        writer.string(rule.conclusion.object);
        writer.floating(rule.confidence);
        writer.u64(rule.support);
    }
    writer.size(snapshot.relation_aliases.size());
    for (const auto& [left, right] : snapshot.relation_aliases) {
        writer.string(left);
        writer.string(right);
    }
}

[[nodiscard]] AbstractionSnapshot read_abstraction(BinaryReader& reader) {
    AbstractionSnapshot snapshot;
    snapshot.config.maximum_facts = reader.size();
    snapshot.config.maximum_rules = reader.size();
    snapshot.config.maximum_inference_depth = reader.size();
    snapshot.config.maximum_derivations_per_query = reader.size();
    snapshot.config.minimum_confidence = reader.float64();
    snapshot.config.inferred_confidence_decay = reader.float64();
    snapshot.next_fact_id = reader.u64();
    snapshot.next_rule_id = reader.u64();
    snapshot.facts.resize(reader.count());
    for (ReasoningFact& fact : snapshot.facts) {
        fact.id = reader.u64();
        fact.subject = reader.string();
        fact.relation = reader.string();
        fact.object = reader.string();
        fact.confidence = reader.float64();
        fact.support = reader.u64();
        fact.inferred = reader.boolean();
        fact.provenance = reader.string();
    }
    snapshot.rules.resize(reader.count());
    for (ReasoningRule& rule : snapshot.rules) {
        rule.id = reader.u64();
        rule.name = reader.string();
        rule.premises.resize(reader.count());
        for (RelationalPattern& premise : rule.premises) {
            premise.subject = reader.string();
            premise.relation = reader.string();
            premise.object = reader.string();
        }
        rule.conclusion.subject = reader.string();
        rule.conclusion.relation = reader.string();
        rule.conclusion.object = reader.string();
        rule.confidence = reader.float64();
        rule.support = reader.u64();
    }
    snapshot.relation_aliases.resize(reader.count());
    for (auto& alias : snapshot.relation_aliases) {
        alias.first = reader.string();
        alias.second = reader.string();
    }
    return snapshot;
}

void write_sparse_router_config(BinaryWriter& writer, const SparseRouterConfig& config) {
    writer.size(config.signature_bits);
    writer.size(config.maximum_candidates);
    writer.size(config.probe_radius);
    writer.u64(config.seed);
}

[[nodiscard]] SparseRouterConfig read_sparse_router_config(BinaryReader& reader) {
    SparseRouterConfig config;
    config.signature_bits = reader.size();
    config.maximum_candidates = reader.size();
    config.probe_radius = reader.size();
    config.seed = reader.u64();
    return config;
}

void write_continual(BinaryWriter& writer, const ContinualLearningSnapshot& snapshot) {
    const ContinualLearningConfig& config = snapshot.config;
    writer.size(config.feature_dimensions);
    writer.size(config.maximum_prototypes);
    writer.size(config.replay_capacity);
    writer.size(config.consolidation_interval);
    writer.size(config.replay_batch_size);
    writer.floating(config.base_learning_rate);
    writer.floating(config.stability_strength);
    writer.floating(config.novelty_threshold);
    writer.floating(config.contrastive_margin);
    write_sparse_router_config(writer, config.router);
    writer.u64(snapshot.next_prototype_id);
    writer.u64(snapshot.next_experience_id);
    writer.u64(snapshot.step);
    writer.u64(snapshot.consolidations);
    writer.size(snapshot.prototypes.size());
    for (const ContinualPrototype& prototype : snapshot.prototypes) {
        writer.u64(prototype.id);
        writer.string(prototype.task);
        writer.string(prototype.label);
        writer.size(prototype.centroid.size());
        for (const float value : prototype.centroid) writer.floating(value);
        writer.u64(prototype.support);
        writer.floating(prototype.importance);
        writer.floating(prototype.plasticity);
        writer.u64(prototype.last_update_step);
    }
    writer.size(snapshot.replay.size());
    for (const ReplayExperience& experience : snapshot.replay) {
        writer.u64(experience.id);
        writer.string(experience.task);
        writer.string(experience.label);
        writer.size(experience.features.size());
        for (const float value : experience.features) writer.floating(value);
        writer.floating(experience.priority);
        writer.u64(experience.seen_step);
    }
}

[[nodiscard]] ContinualLearningSnapshot read_continual(BinaryReader& reader) {
    ContinualLearningSnapshot snapshot;
    ContinualLearningConfig& config = snapshot.config;
    config.feature_dimensions = reader.size();
    config.maximum_prototypes = reader.size();
    config.replay_capacity = reader.size();
    config.consolidation_interval = reader.size();
    config.replay_batch_size = reader.size();
    config.base_learning_rate = reader.float64();
    config.stability_strength = reader.float64();
    config.novelty_threshold = reader.float64();
    config.contrastive_margin = reader.float64();
    config.router = read_sparse_router_config(reader);
    snapshot.next_prototype_id = reader.u64();
    snapshot.next_experience_id = reader.u64();
    snapshot.step = reader.u64();
    snapshot.consolidations = reader.u64();
    snapshot.prototypes.resize(reader.count());
    for (ContinualPrototype& prototype : snapshot.prototypes) {
        prototype.id = reader.u64();
        prototype.task = reader.string();
        prototype.label = reader.string();
        prototype.centroid.resize(reader.count());
        for (float& value : prototype.centroid) value = reader.float32();
        prototype.support = reader.u64();
        prototype.importance = reader.float64();
        prototype.plasticity = reader.float64();
        prototype.last_update_step = reader.u64();
    }
    snapshot.replay.resize(reader.count());
    for (ReplayExperience& experience : snapshot.replay) {
        experience.id = reader.u64();
        experience.task = reader.string();
        experience.label = reader.string();
        experience.features.resize(reader.count());
        for (float& value : experience.features) value = reader.float32();
        experience.priority = reader.float64();
        experience.seen_step = reader.u64();
    }
    return snapshot;
}

void write_grounding(BinaryWriter& writer, const GroundingSnapshot& snapshot) {
    writer.size(snapshot.config.maximum_links);
    writer.size(snapshot.config.maximum_concepts);
    writer.size(snapshot.config.maximum_results);
    writer.floating(snapshot.config.smoothing);
    writer.floating(snapshot.config.minimum_score);
    writer.floating(snapshot.config.negative_weight);
    writer.u64(snapshot.observations);
    writer.size(snapshot.links.size());
    for (const GroundingLink& link : snapshot.links) {
        writer.u64(link.visual_mode_id);
        writer.string(link.concept_name);
        writer.u64(link.positive_count);
        writer.u64(link.negative_count);
        writer.floating(link.confidence);
    }
}

[[nodiscard]] GroundingSnapshot read_grounding(BinaryReader& reader) {
    GroundingSnapshot snapshot;
    snapshot.config.maximum_links = reader.size();
    snapshot.config.maximum_concepts = reader.size();
    snapshot.config.maximum_results = reader.size();
    snapshot.config.smoothing = reader.float64();
    snapshot.config.minimum_score = reader.float64();
    snapshot.config.negative_weight = reader.float64();
    snapshot.observations = reader.u64();
    snapshot.links.resize(reader.count());
    for (GroundingLink& link : snapshot.links) {
        link.visual_mode_id = reader.u64();
        link.concept_name = reader.string();
        link.positive_count = reader.u64();
        link.negative_count = reader.u64();
        link.confidence = reader.float64();
    }
    return snapshot;
}

void write_general(BinaryWriter& writer, const GeneralFabricSnapshot& snapshot) {
    const GeneralFabricConfig& config = snapshot.config;
    writer.size(config.maximum_demonstrations);
    writer.size(config.maximum_preferences);
    writer.size(config.maximum_active_learning_items);
    writer.size(config.maximum_concepts_per_item);
    writer.size(config.maximum_retrieval_candidates);
    writer.size(config.maximum_retrieved_demonstrations);
    writer.size(config.maximum_context_characters);
    writer.size(config.deliberation_candidates);
    writer.floating(config.minimum_retrieval_similarity);
    writer.floating(config.exact_task_bonus);
    writer.floating(config.exact_domain_bonus);
    writer.floating(config.preference_weight);
    writer.floating(config.direct_recall_threshold);
    writer.floating(config.active_learning_uncertainty);
    writer.u64(snapshot.next_demonstration_id);
    writer.u64(snapshot.next_preference_id);
    writer.u64(snapshot.next_active_learning_id);
    writer.size(snapshot.demonstrations.size());
    for (const InstructionDemonstration& demonstration : snapshot.demonstrations) {
        writer.u64(demonstration.id);
        writer.string(demonstration.task);
        writer.string(demonstration.domain);
        writer.string(demonstration.prompt);
        writer.string(demonstration.rationale);
        writer.string(demonstration.response);
        for (const std::uint64_t word : demonstration.signature.words) {
            writer.u64(word);
        }
        writer.u64(demonstration.support);
        writer.floating(demonstration.quality);
    }
    writer.size(snapshot.preferences.size());
    for (const PreferenceExample& preference : snapshot.preferences) {
        writer.u64(preference.id);
        writer.string(preference.prompt);
        writer.string(preference.chosen);
        writer.string(preference.rejected);
        writer.string(preference.feedback);
        for (const std::uint64_t word : preference.prompt_signature.words) {
            writer.u64(word);
        }
        for (const std::uint64_t word : preference.chosen_signature.words) {
            writer.u64(word);
        }
        for (const std::uint64_t word : preference.rejected_signature.words) {
            writer.u64(word);
        }
        writer.floating(preference.weight);
    }
    writer.size(snapshot.active_learning_items.size());
    for (const ActiveLearningItem& item : snapshot.active_learning_items) {
        writer.u64(item.id);
        writer.string(item.prompt);
        writer.string(item.grounding);
        writer.floating(item.uncertainty);
        writer.u64(item.observations);
    }
}

[[nodiscard]] GeneralFabricSnapshot read_general(BinaryReader& reader) {
    GeneralFabricSnapshot snapshot;
    GeneralFabricConfig& config = snapshot.config;
    config.maximum_demonstrations = reader.size();
    config.maximum_preferences = reader.size();
    config.maximum_active_learning_items = reader.size();
    config.maximum_concepts_per_item = reader.size();
    config.maximum_retrieval_candidates = reader.size();
    config.maximum_retrieved_demonstrations = reader.size();
    config.maximum_context_characters = reader.size();
    config.deliberation_candidates = reader.size();
    config.minimum_retrieval_similarity = reader.float64();
    config.exact_task_bonus = reader.float64();
    config.exact_domain_bonus = reader.float64();
    config.preference_weight = reader.float64();
    config.direct_recall_threshold = reader.float64();
    config.active_learning_uncertainty = reader.float64();
    snapshot.next_demonstration_id = reader.u64();
    snapshot.next_preference_id = reader.u64();
    snapshot.next_active_learning_id = reader.u64();
    snapshot.demonstrations.resize(reader.count());
    for (InstructionDemonstration& demonstration : snapshot.demonstrations) {
        demonstration.id = reader.u64();
        demonstration.task = reader.string();
        demonstration.domain = reader.string();
        demonstration.prompt = reader.string();
        demonstration.rationale = reader.string();
        demonstration.response = reader.string();
        for (std::uint64_t& word : demonstration.signature.words) {
            word = reader.u64();
        }
        demonstration.support = reader.u64();
        demonstration.quality = reader.float64();
    }
    snapshot.preferences.resize(reader.count());
    for (PreferenceExample& preference : snapshot.preferences) {
        preference.id = reader.u64();
        preference.prompt = reader.string();
        preference.chosen = reader.string();
        preference.rejected = reader.string();
        preference.feedback = reader.string();
        for (std::uint64_t& word : preference.prompt_signature.words) {
            word = reader.u64();
        }
        for (std::uint64_t& word : preference.chosen_signature.words) {
            word = reader.u64();
        }
        for (std::uint64_t& word : preference.rejected_signature.words) {
            word = reader.u64();
        }
        preference.weight = reader.float64();
    }
    snapshot.active_learning_items.resize(reader.count());
    for (ActiveLearningItem& item : snapshot.active_learning_items) {
        item.id = reader.u64();
        item.prompt = reader.string();
        item.grounding = reader.string();
        item.uncertainty = reader.float64();
        item.observations = reader.u64();
    }
    return snapshot;
}

void write_training_shards(
    BinaryWriter& writer,
    const std::vector<TrainingShardRecord>& records
) {
    writer.size(records.size());
    for (const TrainingShardRecord& record : records) {
        writer.string(record.shard_id);
        writer.string(record.kind);
        writer.string(record.shard_sha256);
        writer.string(record.ledger_sha256);
        writer.string(record.source_uri);
        writer.string(record.license);
        writer.u64(record.records);
        writer.u64(record.bytes);
    }
}

[[nodiscard]] std::vector<TrainingShardRecord> read_training_shards(
    BinaryReader& reader
) {
    std::vector<TrainingShardRecord> records(reader.count());
    for (TrainingShardRecord& record : records) {
        record.shard_id = reader.string();
        record.kind = reader.string();
        record.shard_sha256 = reader.string();
        record.ledger_sha256 = reader.string();
        record.source_uri = reader.string();
        record.license = reader.string();
        record.records = reader.u64();
        record.bytes = reader.u64();
    }
    return records;
}

void write_video(BinaryWriter& writer, const VideoFabricSnapshot& snapshot) {
    writer.u64(video_extension_magic);
    writer.size(snapshot.config.maximum_sequences);
    writer.size(snapshot.config.maximum_frames_per_sequence);
    writer.size(snapshot.config.maximum_generation_frames);
    writer.size(snapshot.config.maximum_prompt_bytes);
    writer.size(snapshot.config.output_width);
    writer.size(snapshot.config.output_height);
    writer.floating(snapshot.config.minimum_prompt_similarity);
    writer.floating(snapshot.config.local_learning_rate);
    writer.u64(snapshot.next_prototype_id);
    writer.u64(snapshot.sequences_seen);
    writer.u64(snapshot.frames_seen);
    writer.size(snapshot.prototypes.size());
    for (const VideoPrototype& prototype : snapshot.prototypes) {
        writer.u64(prototype.id);
        writer.string(prototype.source_sequence_id);
        writer.string(prototype.prompt);
        for (const std::uint64_t word : prototype.prompt_signature.words) writer.u64(word);
        writer.floating(prototype.motion.start_x);
        writer.floating(prototype.motion.start_y);
        writer.floating(prototype.motion.velocity_x);
        writer.floating(prototype.motion.velocity_y);
        writer.floating(prototype.motion.object_width);
        writer.floating(prototype.motion.object_height);
        for (const double value : prototype.motion.foreground_rgb) writer.floating(value);
        for (const double value : prototype.motion.background_rgb) writer.floating(value);
        writer.floating(prototype.frames_per_second);
        writer.size(prototype.observed_frames);
        writer.u64(prototype.support);
    }
}

[[nodiscard]] VideoFabricSnapshot read_video(BinaryReader& reader) {
    if (reader.u64() != video_extension_magic) {
        throw std::runtime_error("unknown Solstice format-6 checkpoint extension");
    }
    VideoFabricSnapshot snapshot;
    snapshot.config.maximum_sequences = reader.size();
    snapshot.config.maximum_frames_per_sequence = reader.size();
    snapshot.config.maximum_generation_frames = reader.size();
    snapshot.config.maximum_prompt_bytes = reader.size();
    snapshot.config.output_width = reader.size();
    snapshot.config.output_height = reader.size();
    snapshot.config.minimum_prompt_similarity = reader.float64();
    snapshot.config.local_learning_rate = reader.float64();
    snapshot.next_prototype_id = reader.u64();
    snapshot.sequences_seen = reader.u64();
    snapshot.frames_seen = reader.u64();
    snapshot.prototypes.resize(reader.count());
    for (VideoPrototype& prototype : snapshot.prototypes) {
        prototype.id = reader.u64();
        prototype.source_sequence_id = reader.string();
        prototype.prompt = reader.string();
        for (std::uint64_t& word : prototype.prompt_signature.words) word = reader.u64();
        prototype.motion.start_x = reader.float64();
        prototype.motion.start_y = reader.float64();
        prototype.motion.velocity_x = reader.float64();
        prototype.motion.velocity_y = reader.float64();
        prototype.motion.object_width = reader.float64();
        prototype.motion.object_height = reader.float64();
        for (double& value : prototype.motion.foreground_rgb) value = reader.float64();
        for (double& value : prototype.motion.background_rgb) value = reader.float64();
        prototype.frames_per_second = reader.float64();
        prototype.observed_frames = reader.size();
        prototype.support = reader.u64();
    }
    return snapshot;
}

void serialize(BinaryWriter& writer, const SolsticeModel& model) {
    writer.u64(model.seed());
    writer.size(model.config().maximum_tool_result_characters);
    {
        const TokenizerSnapshot snapshot = model.tokenizer().snapshot();
        write_tokenizer(writer, snapshot);
    }
    {
        const HierarchicalLanguageSnapshot snapshot = model.language().snapshot();
        write_language(writer, snapshot);
    }
    {
        const VisionSnapshot snapshot = model.vision().snapshot();
        write_vision(writer, snapshot);
    }
    {
        const ToolRouterSnapshot snapshot = model.tool_router().snapshot();
        write_tool_router(writer, snapshot);
    }
    {
        const AbstractionSnapshot snapshot = model.abstraction().snapshot();
        write_abstraction(writer, snapshot);
    }
    {
        const ContinualLearningSnapshot snapshot = model.continual().snapshot();
        write_continual(writer, snapshot);
    }
    {
        const GroundingSnapshot snapshot = model.grounding().snapshot();
        write_grounding(writer, snapshot);
    }
    {
        const GeneralFabricSnapshot snapshot = model.general().snapshot();
        write_general(writer, snapshot);
    }
    write_training_shards(writer, model.completed_training_shards());
    const VideoFabricSnapshot video = model.video().snapshot();
    if (video.config.maximum_sequences > 1U || !video.prototypes.empty()) {
        write_video(writer, video);
    }
}

[[nodiscard]] SolsticeModel deserialize(
    BinaryReader& reader,
    const std::uint32_t version
) {
    SolsticeSnapshot snapshot;
    snapshot.seed = reader.u64();
    snapshot.config.maximum_tool_result_characters = reader.size();
    snapshot.tokenizer = read_tokenizer(reader);
    snapshot.language = read_language(reader);
    snapshot.vision = read_vision(reader, version);
    snapshot.tool_router = read_tool_router(reader);
    if (version >= 4U) {
        snapshot.abstraction = read_abstraction(reader);
        snapshot.continual = read_continual(reader);
        snapshot.grounding = read_grounding(reader);
    }
    if (version >= 5U) {
        snapshot.general = read_general(reader);
    }
    if (version >= 6U) {
        snapshot.completed_training_shards = read_training_shards(reader);
        if (!reader.finished()) {
            snapshot.video = read_video(reader);
        }
    }
    if (!reader.finished()) {
        throw std::runtime_error("unexpected trailing data in Solstice checkpoint");
    }
    snapshot.config.tokenizer = snapshot.tokenizer.config;
    snapshot.config.language = snapshot.language.config;
    snapshot.config.vision = snapshot.vision.config;
    snapshot.config.video = snapshot.video.config;
    snapshot.config.tool_router = snapshot.tool_router.config;
    snapshot.config.abstraction = snapshot.abstraction.config;
    snapshot.config.continual = snapshot.continual.config;
    snapshot.config.grounding = snapshot.grounding.config;
    snapshot.config.general = snapshot.general.config;
    return SolsticeModel::from_snapshot(std::move(snapshot));
}

void write_u32_stream(std::ostream& output, const std::uint32_t value) {
    for (unsigned int index = 0U; index < 4U; ++index) {
        output.put(static_cast<char>((value >> (index * 8U)) & 0xFFU));
    }
}

void write_u64_stream(std::ostream& output, const std::uint64_t value) {
    for (unsigned int index = 0U; index < 8U; ++index) {
        output.put(static_cast<char>((value >> (index * 8U)) & 0xFFULL));
    }
}

[[nodiscard]] std::uint32_t read_u32_stream(std::istream& input) {
    std::uint32_t value = 0U;
    for (unsigned int index = 0U; index < 4U; ++index) {
        const int character = input.get();
        if (character == std::char_traits<char>::eof()) {
            throw std::runtime_error("truncated Solstice checkpoint header");
        }
        value |= static_cast<std::uint32_t>(static_cast<unsigned char>(character)) <<
            (index * 8U);
    }
    return value;
}

[[nodiscard]] std::uint64_t read_u64_stream(std::istream& input) {
    std::uint64_t value = 0U;
    for (unsigned int index = 0U; index < 8U; ++index) {
        const int character = input.get();
        if (character == std::char_traits<char>::eof()) {
            throw std::runtime_error("truncated Solstice checkpoint header");
        }
        value |= static_cast<std::uint64_t>(static_cast<unsigned char>(character)) <<
            (index * 8U);
    }
    return value;
}

struct CheckpointHeader final {
    std::uint32_t version{};
    std::uint64_t file_bytes{};
    std::uint64_t payload_bytes{};
    std::uint64_t stored_checksum{};
};

[[nodiscard]] CheckpointHeader read_checkpoint_header(
    const std::filesystem::path& path,
    std::istream& input,
    const SolsticeCheckpointLimits limits
) {
    const std::uintmax_t file_size = std::filesystem::file_size(path);
    if (file_size > limits.maximum_file_bytes || file_size < 28U) {
        throw std::runtime_error("Solstice checkpoint file size is invalid or exceeds limit");
    }
    std::array<char, checkpoint_magic.size()> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!input || magic != checkpoint_magic) {
        throw std::runtime_error("invalid Solstice checkpoint magic");
    }
    CheckpointHeader loaded;
    loaded.version = read_u32_stream(input);
    if (loaded.version < minimum_checkpoint_version ||
        loaded.version > checkpoint_version) {
        throw std::runtime_error("unsupported Solstice checkpoint version");
    }
    loaded.payload_bytes = read_u64_stream(input);
    loaded.stored_checksum = read_u64_stream(input);
    if (loaded.payload_bytes != static_cast<std::uint64_t>(file_size) - 28ULL) {
        throw std::runtime_error("invalid Solstice checkpoint payload size");
    }
    loaded.file_bytes = static_cast<std::uint64_t>(file_size);
    return loaded;
}

[[nodiscard]] SolsticeModel read_checkpoint_model(
    const std::filesystem::path& path,
    const SolsticeCheckpointLimits limits,
    CheckpointHeader& header
) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("unable to open Solstice checkpoint: " + path.string());
    }
    header = read_checkpoint_header(path, input, limits);
    BinaryReader reader(input, header.payload_bytes, limits);
    SolsticeModel model = deserialize(reader, header.version);
    if (!reader.finished()) {
        throw std::runtime_error("unexpected trailing data in Solstice checkpoint");
    }
    if (reader.checksum() != header.stored_checksum) {
        throw std::runtime_error("Solstice checkpoint checksum mismatch");
    }
    return model;
}

}  // namespace

SolsticeCheckpointLimits checkpoint_limits_for_profile(
    const SolsticeProfile profile
) noexcept {
    if (profile == SolsticeProfile::general_v100_32g_500m) {
        constexpr std::uint64_t gib = 1024ULL * 1024ULL * 1024ULL;
        return {
            10'241ULL * gib,
            16U * 1024U * 1024U,
            1'200'000'000U,
        };
    }
    if (profile == SolsticeProfile::general_h200_141g_30t) {
        constexpr std::uint64_t gib = 1024ULL * 1024ULL * 1024ULL;
        return {
            16'385ULL * gib,
            16U * 1024U * 1024U,
            4'000'000'000ULL,
        };
    }
    return {};
}

void save_solstice_checkpoint(
    const std::filesystem::path& path,
    const SolsticeModel& model
) {
    if (path.empty()) {
        throw std::invalid_argument("Solstice checkpoint path must not be empty");
    }
    std::error_code target_error;
    if (std::filesystem::exists(path, target_error) &&
        !std::filesystem::is_regular_file(path, target_error)) {
        throw std::invalid_argument(
            "Solstice checkpoint target exists and is not a regular file"
        );
    }
    if (target_error) {
        throw std::runtime_error(
            "unable to inspect Solstice checkpoint target: " + target_error.message()
        );
    }
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    const auto stamp = std::chrono::high_resolution_clock::now()
        .time_since_epoch().count();
    const std::filesystem::path temporary =
        path.string() + ".tmp." + std::to_string(stamp);
    try {
        {
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            if (!output) {
                throw std::runtime_error("unable to create Solstice checkpoint temporary file");
            }
            output.write(checkpoint_magic.data(),
                         static_cast<std::streamsize>(checkpoint_magic.size()));
            write_u32_stream(output, checkpoint_version);
            write_u64_stream(output, 0U);
            write_u64_stream(output, 0U);
            BinaryWriter writer(output);
            serialize(writer, model);
            const std::uint64_t payload_size = writer.bytes_written();
            const std::uint64_t payload_checksum = writer.checksum();
            output.seekp(12, std::ios::beg);
            if (!output) {
                throw std::runtime_error(
                    "unable to seek in Solstice checkpoint temporary file"
                );
            }
            write_u64_stream(output, payload_size);
            write_u64_stream(output, payload_checksum);
            output.flush();
            if (!output) {
                throw std::runtime_error("failed while writing Solstice checkpoint");
            }
        }
    } catch (...) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary, cleanup_error);
        throw;
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (!error) return;

    std::error_code exists_error;
    const bool target_exists = std::filesystem::exists(path, exists_error);
    if (exists_error || !target_exists) {
        std::filesystem::remove(temporary);
        throw std::runtime_error("unable to install Solstice checkpoint: " + error.message());
    }
    const std::filesystem::path backup =
        path.string() + ".backup." + std::to_string(stamp);
    error.clear();
    std::filesystem::rename(path, backup, error);
    if (error) {
        std::filesystem::remove(temporary);
        throw std::runtime_error(
            "unable to preserve prior Solstice checkpoint: " + error.message()
        );
    }
    error.clear();
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::error_code restore_error;
        std::filesystem::rename(backup, path, restore_error);
        std::filesystem::remove(temporary);
        if (restore_error) {
            throw std::runtime_error(
                "unable to install checkpoint or restore prior file; recovery copy: " +
                backup.string()
            );
        }
        throw std::runtime_error("unable to install Solstice checkpoint: " + error.message());
    }
    std::filesystem::remove(backup, error);
}

SolsticeModel load_solstice_checkpoint(
    const std::filesystem::path& path,
    const SolsticeCheckpointLimits limits
) {
    CheckpointHeader header;
    return read_checkpoint_model(path, limits, header);
}

SolsticeCheckpointSummary inspect_solstice_checkpoint(
    const std::filesystem::path& path,
    const SolsticeCheckpointLimits limits
) {
    CheckpointHeader loaded;
    const SolsticeModel model = read_checkpoint_model(path, limits, loaded);
    return SolsticeCheckpointSummary{
        loaded.version,
        loaded.file_bytes,
        loaded.stored_checksum,
        model.seed(),
        model.stats(),
    };
}

SolsticeCheckpointSummary inspect_solstice_checkpoint_for_profile(
    const std::filesystem::path& path,
    const SolsticeProfile profile
) {
    CheckpointHeader loaded;
    const SolsticeModel model = read_checkpoint_model(
        path, checkpoint_limits_for_profile(profile), loaded
    );
    if (!profile_config_matches(profile, model.config())) {
        throw std::runtime_error(
            "Solstice checkpoint configuration does not match selected profile"
        );
    }
    return SolsticeCheckpointSummary{
        loaded.version,
        loaded.file_bytes,
        loaded.stored_checksum,
        model.seed(),
        model.stats(),
    };
}

}  // namespace rlf::solstice
