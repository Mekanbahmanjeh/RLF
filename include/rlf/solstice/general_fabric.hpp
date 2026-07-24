#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rlf::solstice {

struct GeneralFabricConfig final {
    std::size_t maximum_demonstrations{500'000U};
    std::size_t maximum_preferences{250'000U};
    std::size_t maximum_active_learning_items{100'000U};
    std::size_t maximum_concepts_per_item{256U};
    std::size_t maximum_retrieval_candidates{8'192U};
    std::size_t maximum_retrieved_demonstrations{16U};
    std::size_t maximum_context_characters{32'768U};
    std::size_t deliberation_candidates{4U};
    double minimum_retrieval_similarity{0.04};
    double exact_task_bonus{0.20};
    double exact_domain_bonus{0.12};
    double preference_weight{1.50};
    double direct_recall_threshold{0.93};
    double active_learning_uncertainty{0.58};
};

struct SemanticSignature final {
    std::array<std::uint64_t, 4U> words{};

    [[nodiscard]] bool operator==(const SemanticSignature&) const noexcept = default;
};

struct InstructionDemonstration final {
    std::uint64_t id{};
    std::string task;
    std::string domain;
    std::string prompt;
    std::string rationale;
    std::string response;
    SemanticSignature signature;
    std::uint64_t support{1U};
    double quality{1.0};
};

struct PreferenceExample final {
    std::uint64_t id{};
    std::string prompt;
    std::string chosen;
    std::string rejected;
    std::string feedback;
    SemanticSignature prompt_signature;
    SemanticSignature chosen_signature;
    SemanticSignature rejected_signature;
    double weight{1.0};
};

struct ActiveLearningItem final {
    std::uint64_t id{};
    std::string prompt;
    std::string grounding;
    double uncertainty{1.0};
    std::uint64_t observations{1U};
};

struct GeneralRetrievalMatch final {
    std::uint64_t demonstration_id{};
    std::size_t demonstration_index{};
    double score{};
};

struct DeliberationContext final {
    std::string task;
    std::string domain;
    std::string context;
    std::string direct_response;
    std::string plan_hint;
    std::vector<std::uint64_t> demonstration_ids;
    double confidence{};
};

struct GeneralFabricSnapshot final {
    GeneralFabricConfig config;
    std::uint64_t next_demonstration_id{1U};
    std::uint64_t next_preference_id{1U};
    std::uint64_t next_active_learning_id{1U};
    std::vector<InstructionDemonstration> demonstrations;
    std::vector<PreferenceExample> preferences;
    std::vector<ActiveLearningItem> active_learning_items;
};

struct GeneralTrainingOperationStats final {
    std::uint64_t instruction_duplicate_prefilter_lookups{};
    std::uint64_t instruction_duplicate_retrievals{};
    std::uint64_t instruction_duplicate_retrievals_avoided{};
    std::uint64_t instruction_index_rebuilds{};
    std::uint64_t instruction_index_entries_built{};
    std::uint64_t instruction_index_incremental_inserts{};
    std::uint64_t preference_duplicate_lookups{};
    std::uint64_t linear_preference_comparisons{};
    std::uint64_t indexed_preference_candidates{};
    std::uint64_t preference_index_rebuilds{};
    std::uint64_t preference_index_entries_built{};
    std::uint64_t preference_index_incremental_inserts{};
    std::uint64_t active_learning_duplicate_lookups{};
    std::uint64_t linear_active_learning_comparisons{};
    std::uint64_t indexed_active_learning_candidates{};
    std::uint64_t active_learning_index_rebuilds{};
    std::uint64_t active_learning_index_entries_built{};
    std::uint64_t active_learning_index_incremental_inserts{};
};

class GeneralInstructionFabric final {
public:
    explicit GeneralInstructionFabric(GeneralFabricConfig config = {});

    std::uint64_t train_instruction(
        std::string_view task,
        std::string_view domain,
        std::string_view prompt,
        std::string_view rationale,
        std::string_view response,
        double quality = 1.0
    );
    std::uint64_t train_preference(
        std::string_view prompt,
        std::string_view chosen,
        std::string_view rejected,
        std::string_view feedback = {},
        double weight = 1.0
    );
    std::uint64_t observe_uncertain(
        std::string_view prompt,
        std::string_view grounding,
        double uncertainty
    );

    [[nodiscard]] std::vector<GeneralRetrievalMatch> retrieve(
        std::string_view task,
        std::string_view domain,
        std::string_view prompt,
        std::string_view grounding = {},
        std::size_t maximum_results = 0U
    ) const;
    [[nodiscard]] DeliberationContext build_context(
        std::string_view task,
        std::string_view domain,
        std::string_view prompt,
        std::string_view grounding,
        std::string_view knowledge
    ) const;
    [[nodiscard]] double score_response(
        std::string_view prompt,
        std::string_view response,
        std::span<const GeneralRetrievalMatch> matches
    ) const;

    [[nodiscard]] std::span<const InstructionDemonstration> demonstrations() const noexcept;
    [[nodiscard]] std::span<const PreferenceExample> preferences() const noexcept;
    [[nodiscard]] std::span<const ActiveLearningItem> active_learning_items() const noexcept;
    [[nodiscard]] const GeneralFabricConfig& config() const noexcept;
    [[nodiscard]] std::uint64_t deterministic_hash() const noexcept;
    void set_indexed_preference_duplicates(bool enabled);
    void set_indexed_active_learning_duplicates(bool enabled);
    void set_indexed_instruction_duplicates(bool enabled);
    [[nodiscard]] GeneralTrainingOperationStats training_operation_stats() const noexcept;

    [[nodiscard]] GeneralFabricSnapshot snapshot() const;
    [[nodiscard]] static GeneralInstructionFabric from_snapshot(
        GeneralFabricSnapshot snapshot
    );

    [[nodiscard]] static SemanticSignature make_signature(std::string_view text);
    [[nodiscard]] static double signature_similarity(
        const SemanticSignature& left,
        const SemanticSignature& right
    ) noexcept;

private:
    [[nodiscard]] static std::string normalize_label(std::string_view value);
    [[nodiscard]] static std::vector<std::uint64_t> concept_hashes(
        std::string_view text,
        std::size_t maximum_concepts
    );
    [[nodiscard]] static std::array<std::uint64_t, 8U> band_keys(
        const SemanticSignature& signature
    ) noexcept;
    void index_demonstration(std::size_t index);
    void rebuild_index();
    void rebuild_preference_index();
    void rebuild_active_learning_index();
    void rebuild_instruction_index();

    GeneralFabricConfig config_;
    std::uint64_t next_demonstration_id_{1U};
    std::uint64_t next_preference_id_{1U};
    std::uint64_t next_active_learning_id_{1U};
    std::vector<InstructionDemonstration> demonstrations_;
    std::vector<PreferenceExample> preferences_;
    std::vector<ActiveLearningItem> active_learning_items_;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> band_postings_;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>>
        preference_postings_;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>>
        active_learning_postings_;
    std::unordered_set<std::uint64_t> instruction_keys_;
    bool indexed_preference_duplicates_{true};
    bool indexed_active_learning_duplicates_{true};
    bool indexed_instruction_duplicates_{true};
    GeneralTrainingOperationStats training_operation_stats_;
};

}  // namespace rlf::solstice
