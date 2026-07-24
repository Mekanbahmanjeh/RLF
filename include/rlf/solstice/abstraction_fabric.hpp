#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rlf::solstice {

struct AbstractionConfig final {
    std::size_t maximum_facts{5'000'000U};
    std::size_t maximum_rules{250'000U};
    std::size_t maximum_inference_depth{8U};
    std::size_t maximum_derivations_per_query{100'000U};
    double minimum_confidence{0.05};
    double inferred_confidence_decay{0.97};
};

struct RelationalPattern final {
    std::string subject;
    std::string relation;
    std::string object;
};

struct ReasoningFact final {
    std::uint64_t id{};
    std::string subject;
    std::string relation;
    std::string object;
    double confidence{1.0};
    std::uint64_t support{1U};
    bool inferred{};
    std::string provenance;
};

struct ReasoningRule final {
    std::uint64_t id{};
    std::string name;
    std::vector<RelationalPattern> premises;
    RelationalPattern conclusion;
    double confidence{1.0};
    std::uint64_t support{1U};
};

struct ReasoningProofStep final {
    std::uint64_t fact_id{};
    std::uint64_t rule_id{};
    std::string statement;
};

struct ReasoningAnswer final {
    std::string value;
    double confidence{};
    std::vector<ReasoningProofStep> proof;
    std::string matched_subject;
    std::string matched_relation;
    std::string matched_object;
};

struct ReasoningInferenceStats final {
    std::uint64_t candidate_facts_examined{};
    std::uint64_t unification_attempts{};
    std::uint64_t derivations{};
    std::uint64_t index_lookups{};
    std::uint64_t naive_candidate_upper_bound{};
};

struct ReasoningQueryResult final {
    std::vector<ReasoningAnswer> answers;
    ReasoningInferenceStats stats;
};

struct SchemaInductionResult final {
    std::uint64_t rule_id{};
    std::size_t path_hops{};
    std::uint64_t edges_examined{};
    std::vector<std::string> relation_path;
};

struct AbstractionSnapshot final {
    AbstractionConfig config;
    std::uint64_t next_fact_id{1U};
    std::uint64_t next_rule_id{1U};
    std::vector<ReasoningFact> facts;
    std::vector<ReasoningRule> rules;
    std::vector<std::pair<std::string, std::string>> relation_aliases;
};

class AbstractionFabric final {
public:
    explicit AbstractionFabric(AbstractionConfig config = {});

    std::uint64_t learn_fact(
        std::string_view subject,
        std::string_view relation,
        std::string_view object,
        double confidence = 1.0,
        std::string_view provenance = {}
    );
    std::uint64_t learn_rule(
        std::string_view name,
        std::span<const RelationalPattern> premises,
        const RelationalPattern& conclusion,
        double confidence = 1.0
    );
    void learn_relation_equivalence(
        std::string_view left,
        std::string_view right
    );
    std::uint64_t transfer_rule(
        std::uint64_t source_rule_id,
        std::string_view new_name,
        const std::map<std::string, std::string>& relation_mapping
    );
    [[nodiscard]] SchemaInductionResult induce_chain_rule(
        std::string_view name,
        std::string_view demonstration_subject,
        std::string_view conclusion_relation,
        std::string_view demonstration_object,
        std::size_t maximum_hops = 4U,
        double confidence = 1.0
    );

    [[nodiscard]] ReasoningQueryResult infer_with_stats(
        const RelationalPattern& query,
        std::size_t maximum_answers = 16U
    ) const;
    [[nodiscard]] std::vector<ReasoningAnswer> infer(
        const RelationalPattern& query,
        std::size_t maximum_answers = 16U
    ) const;
    [[nodiscard]] std::vector<ReasoningAnswer> answer(
        std::string_view subject,
        std::string_view relation,
        std::size_t maximum_answers = 16U
    ) const;

    [[nodiscard]] std::span<const ReasoningFact> facts() const noexcept;
    [[nodiscard]] std::span<const ReasoningRule> rules() const noexcept;
    [[nodiscard]] const AbstractionConfig& config() const noexcept;
    [[nodiscard]] std::uint64_t deterministic_hash() const noexcept;

    [[nodiscard]] AbstractionSnapshot snapshot() const;
    [[nodiscard]] static AbstractionFabric from_snapshot(
        AbstractionSnapshot snapshot
    );

private:
    [[nodiscard]] std::string canonical_relation(std::string_view relation) const;
    void rebuild_indices();

    AbstractionConfig config_;
    std::uint64_t next_fact_id_{1U};
    std::uint64_t next_rule_id_{1U};
    std::vector<ReasoningFact> facts_;
    std::vector<ReasoningRule> rules_;
    std::map<std::string, std::string> relation_aliases_;
    std::map<std::string, std::size_t> fact_index_;
};

}  // namespace rlf::solstice
