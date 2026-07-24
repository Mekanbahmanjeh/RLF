#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rlf::frontier {

enum class KnowledgeKind : std::uint8_t {
    claim,
    observed_fact,
    verified_fact,
    inferred_belief,
    prediction,
    procedure,
    concept_record,
};

enum class MemoryTier : std::uint8_t {
    active,
    hot,
    warm,
    cold,
};

enum class Modality : std::uint8_t {
    text,
    structured,
    image,
    video,
    audio,
    sensor,
    action,
};

struct KnowledgeRecord final {
    std::uint64_t stable_id{};
    KnowledgeKind kind{KnowledgeKind::claim};
    std::string subject;
    std::string predicate;
    std::string object;
    std::string source;
    double confidence{0.5};
    double utility{};
    std::uint64_t creation_step{};
    std::uint64_t last_used_step{};
    std::uint64_t valid_from{};
    std::uint64_t valid_until{};
    std::uint64_t version{1U};
    std::size_t use_count{};
    std::size_t contradiction_count{};
    bool verified{};
    bool stale{};
    bool invalidated{};
    MemoryTier tier{MemoryTier::warm};
};

struct ModeRecord final {
    std::uint64_t stable_id{};
    Modality modality{Modality::structured};
    std::string label;
    std::vector<float> prototype;
    std::optional<std::uint64_t> parent_id;
    std::set<std::uint64_t> child_ids;
    std::set<std::uint64_t> linked_modes;
    double confidence{0.5};
    double utility{};
    std::size_t support{};
    std::uint64_t creation_step{};
    std::uint64_t last_used_step{};
    bool enabled{true};
};

struct KnowledgeQuery final {
    std::string subject;
    std::string predicate;
    std::vector<std::string> terms;
    std::size_t maximum_results{8U};
    bool include_stale{};
    bool include_invalidated{};
};

struct KnowledgeHit final {
    std::uint64_t stable_id{};
    double score{};
    bool exact{};
};

struct ModeHit final {
    std::uint64_t stable_id{};
    double score{};
};

struct KnowledgeStatistics final {
    std::size_t records{};
    std::size_t active_records{};
    std::size_t hot_records{};
    std::size_t warm_records{};
    std::size_t cold_records{};
    std::size_t modes{};
    std::size_t contradictions{};
    std::size_t stale_records{};
    std::size_t invalidated_records{};
    std::size_t index_keys{};
    std::size_t last_candidates_examined{};
    std::size_t total_candidates_examined{};
    std::uint64_t approximate_persistent_bytes{};
};

class KnowledgeFabric final {
public:
    explicit KnowledgeFabric(std::uint64_t seed = 0x524C4637ULL);

    [[nodiscard]] std::uint64_t seed() const noexcept;
    [[nodiscard]] std::uint64_t step() const noexcept;
    void set_step(std::uint64_t step) noexcept;
    void reserve_records(std::size_t expected_records, std::size_t expected_terms_per_record = 3U);

    std::uint64_t insert(KnowledgeRecord record);
    bool update(std::uint64_t stable_id, const KnowledgeRecord& replacement);
    bool invalidate(std::uint64_t stable_id);
    bool mark_stale(std::uint64_t stable_id, bool stale = true);
    bool verify(std::uint64_t stable_id, double confidence = 1.0);

    [[nodiscard]] std::vector<KnowledgeHit> query(
        const KnowledgeQuery& query
    );
    [[nodiscard]] const KnowledgeRecord* find(std::uint64_t stable_id) const;
    [[nodiscard]] KnowledgeRecord* find(std::uint64_t stable_id);

    std::uint64_t learn_mode(
        Modality modality,
        std::string label,
        std::span<const float> prototype,
        std::optional<std::uint64_t> parent_id = std::nullopt,
        double confidence = 0.5
    );
    bool link_modes(std::uint64_t left, std::uint64_t right);
    [[nodiscard]] std::vector<ModeHit> retrieve_modes(
        Modality modality,
        std::span<const float> query,
        std::size_t maximum_results
    ) const;
    [[nodiscard]] const ModeRecord* find_mode(std::uint64_t stable_id) const;
    [[nodiscard]] ModeRecord* find_mode(std::uint64_t stable_id);

    void consolidate(std::size_t hot_limit, std::size_t active_limit);
    void prune(std::size_t maximum_records, double minimum_utility);

    [[nodiscard]] const std::map<std::uint64_t, KnowledgeRecord>& records() const noexcept;
    [[nodiscard]] const std::map<std::uint64_t, ModeRecord>& modes() const noexcept;
    void import_record(KnowledgeRecord record);
    void import_mode(ModeRecord mode);

    [[nodiscard]] KnowledgeStatistics statistics() const;
    [[nodiscard]] std::uint64_t deterministic_hash() const noexcept;

private:
    std::uint64_t seed_{};
    std::uint64_t step_{};
    std::uint64_t next_record_id_{1U};
    std::uint64_t next_mode_id_{1U};
    std::map<std::uint64_t, KnowledgeRecord> records_;
    std::map<std::uint64_t, ModeRecord> modes_;
    std::unordered_multimap<std::uint64_t, std::uint64_t> exact_index_;
    std::unordered_multimap<std::uint64_t, std::uint64_t> term_index_;
    mutable std::size_t last_candidates_examined_{};
    mutable std::size_t total_candidates_examined_{};

    static void validate_record(const KnowledgeRecord& record);
    static void validate_mode(const ModeRecord& mode);
    static std::uint64_t exact_key(std::string_view subject, std::string_view predicate) noexcept;
    static std::vector<std::string> tokenize(std::string_view value);
    static std::uint64_t token_key(std::string_view token) noexcept;
    static double cosine_similarity(std::span<const float> left, std::span<const float> right);
    void index_record(const KnowledgeRecord& record);
    void unindex_record(const KnowledgeRecord& record);
    void detect_contradictions(KnowledgeRecord& record);
    [[nodiscard]] bool would_create_cycle(
        std::uint64_t child_id,
        std::uint64_t parent_id
    ) const;
};

[[nodiscard]] std::string_view to_string(KnowledgeKind kind) noexcept;
[[nodiscard]] std::string_view to_string(MemoryTier tier) noexcept;
[[nodiscard]] std::string_view to_string(Modality modality) noexcept;

}  // namespace rlf::frontier
