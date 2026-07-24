#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rlf::solstice {

struct GroundingConfig final {
    std::size_t maximum_links{20'000'000U};
    std::size_t maximum_concepts{2'000'000U};
    std::size_t maximum_results{64U};
    double smoothing{0.5};
    double minimum_score{0.05};
    double negative_weight{1.0};
};

struct GroundingLink final {
    std::uint64_t visual_mode_id{};
    std::string concept_name;
    std::uint64_t positive_count{};
    std::uint64_t negative_count{};
    double confidence{};
};

struct GroundingHit final {
    std::uint64_t visual_mode_id{};
    std::string concept_name;
    double score{};
    std::uint64_t support{};
};

struct GroundingSnapshot final {
    GroundingConfig config;
    std::uint64_t observations{};
    std::vector<GroundingLink> links;
};

struct GroundingOperationStats final {
    std::uint64_t link_lookups{};
    std::uint64_t full_lookup_entries_rebuilt{};
    std::uint64_t indexed_link_candidates_examined{};
    std::uint64_t incremental_posting_inserts{};
    std::uint64_t confidence_recomputations{};
    std::uint64_t full_confidence_sweep_entries{};
    std::uint64_t derived_sort_entries{};
    std::uint64_t mode_query_full_scan_entries{};
    std::uint64_t mode_query_indexed_candidates{};
    std::uint64_t concept_query_full_scan_entries{};
    std::uint64_t concept_query_indexed_candidates{};
};

class CrossModalGroundingFabric final {
public:
    explicit CrossModalGroundingFabric(GroundingConfig config = {});

    void observe(
        std::span<const std::uint64_t> visual_mode_ids,
        std::span<const std::string> positive_concepts,
        std::span<const std::string> negative_concepts = {}
    );
    [[nodiscard]] std::vector<GroundingHit> concepts_for_mode(
        std::uint64_t visual_mode_id,
        std::size_t maximum_results = 0U
    ) const;
    [[nodiscard]] std::vector<GroundingHit> modes_for_concept(
        std::string_view concept_name,
        std::size_t maximum_results = 0U
    ) const;
    [[nodiscard]] std::vector<GroundingHit> compose_concepts(
        std::span<const std::string> concepts,
        std::size_t maximum_results = 0U
    ) const;

    [[nodiscard]] std::span<const GroundingLink> links() const noexcept;
    [[nodiscard]] const GroundingConfig& config() const noexcept;
    [[nodiscard]] std::uint64_t observations() const noexcept;
    [[nodiscard]] std::uint64_t deterministic_hash() const noexcept;
    void set_persistent_link_index(bool enabled);
    [[nodiscard]] GroundingOperationStats operation_stats() const noexcept;

    [[nodiscard]] GroundingSnapshot snapshot() const;
    [[nodiscard]] static CrossModalGroundingFabric from_snapshot(
        GroundingSnapshot snapshot
    );

private:
    [[nodiscard]] static std::string normalize_concept(
        std::string_view concept_name
    );
    [[nodiscard]] double link_score(const GroundingLink& link) const noexcept;
    void rebuild_index();

    GroundingConfig config_;
    std::uint64_t observations_{};
    std::vector<GroundingLink> links_;
    std::vector<std::size_t> order_by_mode_;
    std::vector<std::size_t> order_by_concept_;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>>
        link_indices_by_mode_;
    std::unordered_map<std::string, std::vector<std::size_t>>
        link_indices_by_concept_;
    std::size_t indexed_link_count_{};
    bool persistent_link_index_{true};
    bool confidence_cache_valid_{true};
    mutable GroundingOperationStats operation_stats_;
};

}  // namespace rlf::solstice
