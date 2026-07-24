#pragma once

#include "rlf/core/phase_vector.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rlf::core {

struct LanguageFabricConfig final {
    std::size_t phase_dimension{64U};
    std::size_t maximum_lexemes{2'048U};
    std::size_t maximum_merges{512U};
    std::size_t minimum_pair_support{4U};
    std::size_t maximum_contexts{1'000'000U};
    std::size_t maximum_context_order{8U};
    std::size_t minimum_context_support{2U};
    std::size_t maximum_constructions{8'192U};
    std::size_t minimum_construction_support{2U};
    std::size_t maximum_generation_tokens{256U};
    std::size_t maximum_semantic_values{16'384U};
    std::size_t maximum_surfaces_per_concept{8U};
    double smoothing{0.10};
    double minimum_lexical_score{0.08};
    double construction_support_weight{0.20};
    double literal_match_weight{0.35};
    double slot_match_weight{1.0};
};

enum class LanguageAct : std::uint8_t {
    statement = 0U,
    query_agent = 1U,
    query_patient = 2U,
    query_location = 3U,
    answer_agent = 4U,
    answer_patient = 5U,
    answer_location = 6U,
};

enum class LanguageRole : std::uint8_t {
    predicate = 0U,
    agent = 1U,
    patient = 2U,
    agent_attribute = 3U,
    patient_attribute = 4U,
    location = 5U,
};

struct LanguageFrame final {
    LanguageAct act{LanguageAct::statement};
    std::string predicate;
    std::string agent;
    std::string patient;
    std::string agent_attribute;
    std::string patient_attribute;
    std::string location;

    [[nodiscard]] bool operator==(const LanguageFrame&) const noexcept = default;
};

struct LanguageSupervisedExample final {
    std::string text;
    LanguageFrame frame;
};

struct LanguageLexeme final {
    std::uint64_t id{};
    std::string bytes;
    std::uint64_t support{};
    PhaseVector key{std::vector<float>{0.0F}};
};

struct LanguageMerge final {
    std::uint64_t left_id{};
    std::uint64_t right_id{};
    std::uint64_t result_id{};
    std::uint64_t support{};
    std::int64_t description_gain{};
};

struct LanguageOutcomeCount final {
    std::uint64_t token_id{};
    std::uint64_t count{};
};

struct LanguageContext final {
    std::uint64_t id{};
    std::vector<std::uint64_t> history;
    std::uint64_t support{};
    std::vector<LanguageOutcomeCount> outcomes;
};

struct LanguagePredictionOutcome final {
    std::uint64_t token_id{};
    double probability{};
};

struct LanguagePrediction final {
    std::vector<LanguagePredictionOutcome> outcomes;
    std::size_t context_order{};
    std::uint64_t context_id{};
    double uncertainty{1.0};
};

struct LanguageSurfaceCount final {
    std::uint64_t token_id{};
    std::uint64_t count{};
    double association{};
};

struct LanguageConcept final {
    std::uint64_t id{};
    LanguageRole role{LanguageRole::predicate};
    std::string value;
    PhaseVector key{std::vector<float>{0.0F}};
    std::uint64_t support{};
    std::vector<LanguageSurfaceCount> surfaces;
};

enum class LanguagePatternKind : std::uint8_t {
    literal = 0U,
    slot = 1U,
};

struct LanguagePatternItem final {
    LanguagePatternKind kind{LanguagePatternKind::literal};
    std::uint64_t token_id{};
    LanguageRole role{LanguageRole::predicate};
    std::uint8_t surface_form{};

    [[nodiscard]] bool operator==(const LanguagePatternItem&) const noexcept = default;
};

struct LanguageConstruction final {
    std::uint64_t id{};
    LanguageAct act{LanguageAct::statement};
    std::vector<LanguagePatternItem> pattern;
    std::uint64_t support{};
    double confidence{};
};

struct LanguageParse final {
    bool success{false};
    LanguageFrame frame;
    std::uint64_t construction_id{};
    double score{};
    double uncertainty{1.0};
    std::vector<std::uint64_t> tokens;
};

struct LanguageGeneration final {
    bool success{false};
    std::string text;
    std::uint64_t construction_id{};
    std::vector<std::uint64_t> tokens;
};

struct LanguageAnswer final {
    bool success{false};
    LanguageFrame query;
    LanguageFrame matched_fact;
    LanguageFrame answer_frame;
    std::string text;
};

struct LanguageFabricStats final {
    std::uint64_t raw_bytes_seen{};
    std::uint64_t lexicon_merges{};
    std::uint64_t language_tokens_seen{};
    std::uint64_t contexts_created{};
    std::uint64_t contexts_updated{};
    std::uint64_t supervised_examples_seen{};
    std::uint64_t concepts_created{};
    std::uint64_t constructions_created{};
    std::uint64_t parse_queries{};
    std::uint64_t generation_queries{};
    std::uint64_t answer_queries{};
};

struct LanguageFabricSnapshot final {
    LanguageFabricConfig config;
    std::uint64_t seed{};
    std::uint64_t next_lexeme_id{};
    std::uint64_t next_context_id{};
    std::uint64_t next_concept_id{};
    std::uint64_t next_construction_id{};
    std::vector<LanguageLexeme> lexemes;
    std::vector<LanguageMerge> merges;
    std::vector<LanguageContext> contexts;
    std::vector<LanguageConcept> concepts;
    std::vector<LanguageConstruction> constructions;
    LanguageFabricStats stats;
};

class LanguageFabric final {
public:
    explicit LanguageFabric(LanguageFabricConfig config, std::uint64_t seed);

    [[nodiscard]] const LanguageFabricConfig& config() const noexcept;
    [[nodiscard]] std::uint64_t seed() const noexcept;
    [[nodiscard]] std::span<const LanguageLexeme> lexemes() const noexcept;
    [[nodiscard]] std::span<const LanguageMerge> merges() const noexcept;
    [[nodiscard]] std::span<const LanguageContext> contexts() const noexcept;
    [[nodiscard]] std::span<const LanguageConcept> concepts() const noexcept;
    [[nodiscard]] std::span<const LanguageConstruction> constructions() const noexcept;
    [[nodiscard]] const LanguageFabricStats& stats() const noexcept;

    void learn_lexicon(std::string_view raw_corpus);
    [[nodiscard]] std::vector<std::uint64_t> encode(std::string_view text) const;
    [[nodiscard]] std::string decode(std::span<const std::uint64_t> tokens) const;

    void train_language_model(std::string_view raw_corpus);
    [[nodiscard]] LanguagePrediction predict_next(
        std::span<const std::uint64_t> history
    ) const;
    [[nodiscard]] std::string generate(
        std::string_view prompt,
        std::size_t maximum_tokens,
        bool stop_at_newline = true
    ) const;
    [[nodiscard]] double sequence_nll(std::string_view text) const;

    void train_semantics(std::span<const LanguageSupervisedExample> examples);
    [[nodiscard]] LanguageParse parse(std::string_view text) const;
    [[nodiscard]] LanguageGeneration generate_frame(const LanguageFrame& frame) const;
    [[nodiscard]] LanguageAnswer answer(
        std::span<const std::string> context_sentences,
        std::string_view question
    ) const;

    [[nodiscard]] std::optional<std::uint64_t> lexeme_id(std::string_view bytes) const;
    [[nodiscard]] const LanguageLexeme& lexeme_by_id(std::uint64_t id) const;
    [[nodiscard]] std::size_t estimated_storage_bytes() const noexcept;
    [[nodiscard]] std::uint64_t deterministic_hash() const noexcept;

    [[nodiscard]] LanguageFabricSnapshot snapshot() const;
    [[nodiscard]] static LanguageFabric from_snapshot(LanguageFabricSnapshot snapshot);

private:
    struct VectorHash final {
        [[nodiscard]] std::size_t operator()(
            const std::vector<std::uint64_t>& value
        ) const noexcept;
    };

    struct ConceptKey final {
        LanguageRole role{LanguageRole::predicate};
        std::string value;
        [[nodiscard]] bool operator==(const ConceptKey&) const noexcept = default;
    };

    struct ConceptKeyHash final {
        [[nodiscard]] std::size_t operator()(const ConceptKey& value) const noexcept;
    };

    [[nodiscard]] static std::vector<std::pair<LanguageRole, std::string_view>>
    frame_values(const LanguageFrame& frame);
    [[nodiscard]] static std::string_view frame_value(
        const LanguageFrame& frame,
        LanguageRole role
    );
    static void set_frame_value(
        LanguageFrame& frame,
        LanguageRole role,
        std::string value
    );

    [[nodiscard]] PhaseVector lexeme_phase(std::string_view bytes) const;
    [[nodiscard]] bool merge_allowed(
        const LanguageLexeme& left,
        const LanguageLexeme& right
    ) const noexcept;
    [[nodiscard]] std::vector<std::uint64_t> apply_merges(
        std::span<const std::uint64_t> base_tokens
    ) const;
    [[nodiscard]] std::vector<std::uint64_t> base_encode(
        std::string_view text
    ) const;
    void rebuild_indices();
    void validate_snapshot() const;

    [[nodiscard]] std::optional<std::size_t> context_index(
        std::span<const std::uint64_t> history
    ) const;
    [[nodiscard]] std::size_t get_or_create_context(
        std::span<const std::uint64_t> history
    );
    void update_context(
        std::span<const std::uint64_t> history,
        std::uint64_t next_id
    );

    [[nodiscard]] const LanguageConcept* find_concept(
        LanguageRole role,
        std::string_view value
    ) const;
    [[nodiscard]] double lexical_score(
        LanguageRole role,
        std::uint64_t token_id,
        std::string_view value
    ) const;
    [[nodiscard]] std::optional<std::pair<std::string, double>> best_value(
        LanguageRole role,
        std::uint64_t token_id
    ) const;
    [[nodiscard]] std::uint64_t best_surface(
        LanguageRole role,
        std::string_view value,
        std::uint8_t surface_form
    ) const;
    [[nodiscard]] LanguageFrame answer_frame_for(
        const LanguageFrame& fact,
        const LanguageFrame& query
    ) const;

    LanguageFabricConfig config_;
    std::uint64_t seed_{};
    std::uint64_t next_lexeme_id_{1ULL};
    std::uint64_t next_context_id_{1ULL};
    std::uint64_t next_concept_id_{1ULL};
    std::uint64_t next_construction_id_{1ULL};
    std::vector<LanguageLexeme> lexemes_;
    std::vector<LanguageMerge> merges_;
    std::vector<LanguageContext> contexts_;
    std::vector<LanguageConcept> concepts_;
    std::vector<LanguageConstruction> constructions_;
    std::unordered_map<std::uint64_t, std::size_t> lexeme_index_by_id_;
    std::unordered_map<std::string, std::uint64_t> lexeme_id_by_bytes_;
    std::unordered_map<std::vector<std::uint64_t>, std::size_t, VectorHash>
        context_index_by_history_;
    std::unordered_map<ConceptKey, std::size_t, ConceptKeyHash>
        concept_index_;
    LanguageFabricStats stats_;
};

[[nodiscard]] std::string_view rlf5_architecture_name() noexcept;
[[nodiscard]] std::string_view language_act_name(LanguageAct act) noexcept;
[[nodiscard]] std::string_view language_role_name(LanguageRole role) noexcept;

}  // namespace rlf::core
