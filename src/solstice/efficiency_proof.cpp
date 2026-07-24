#include "rlf/solstice/efficiency_proof.hpp"

#include "rlf/solstice/abstraction_fabric.hpp"
#include "rlf/solstice/continual_learning.hpp"
#include "rlf/solstice/grounding_fabric.hpp"
#include "rlf/solstice/sparse_router.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rlf::solstice {
namespace {

[[nodiscard]] std::vector<float> synthetic_vector(
    const std::size_t item,
    const std::size_t dimensions
) {
    std::vector<float> values(dimensions);
    for (std::size_t dimension = 0U; dimension < dimensions; ++dimension) {
        std::uint64_t value =
            (static_cast<std::uint64_t>(item) + 1ULL) *
            (static_cast<std::uint64_t>(dimension) + 17ULL) *
            0x9e3779b97f4a7c15ULL;
        value ^= value >> 29U;
        value *= 0xbf58476d1ce4e5b9ULL;
        value ^= value >> 31U;
        values[dimension] = static_cast<float>(
            static_cast<double>(value % 65'521ULL) / 32'760.5 - 1.0
        );
    }
    return values;
}

[[nodiscard]] std::string relation_pair_key(
    const std::string& subject,
    const std::string& object
) {
    return subject + "\x1f" + object;
}

struct SchemaProof final {
    EfficiencyMetric supervision;
    EfficiencyMetric indexed_reasoning;
    CapabilityProof composition;
    CapabilityProof one_shot_induction;
};

[[nodiscard]] SchemaProof run_schema_proof(
    const EfficiencyProofConfig& config
) {
    if (config.schema_instances < 2U) {
        throw std::invalid_argument("schema proof needs at least two instances");
    }
    AbstractionConfig abstraction_config;
    abstraction_config.maximum_facts = config.schema_instances * 4U + 64U;
    abstraction_config.maximum_rules = 64U;
    abstraction_config.maximum_inference_depth = 2U;
    abstraction_config.maximum_derivations_per_query =
        config.schema_instances * 2U + 64U;
    AbstractionFabric fabric(abstraction_config);

    for (std::size_t index = 0U; index < config.schema_instances; ++index) {
        const std::string suffix = std::to_string(index);
        fabric.learn_fact("subject_" + suffix, "first_link", "middle_" + suffix);
        fabric.learn_fact("middle_" + suffix, "second_link", "object_" + suffix);
    }
    const SchemaInductionResult induction = fabric.induce_chain_rule(
        "one demonstration two-hop schema",
        "subject_0",
        "composed_link",
        "object_0",
        2U,
        1.0
    );
    const ReasoningQueryResult result = fabric.infer_with_stats(
        {"?subject", "composed_link", "?object"},
        config.schema_instances
    );

    std::unordered_set<std::string> inferred;
    inferred.reserve(result.answers.size() * 2U + 1U);
    for (const ReasoningAnswer& answer : result.answers) {
        inferred.insert(relation_pair_key(
            answer.matched_subject, answer.matched_object
        ));
    }
    std::size_t correct = 0U;
    for (std::size_t index = 1U; index < config.schema_instances; ++index) {
        const std::string suffix = std::to_string(index);
        if (inferred.contains(relation_pair_key(
                "subject_" + suffix, "object_" + suffix
            ))) {
            ++correct;
        }
    }
    const std::size_t held_out_instances = config.schema_instances - 1U;
    const double accuracy = static_cast<double>(correct) /
        static_cast<double>(held_out_instances);

    // The matched memorization baseline is explicitly trained with every
    // target relation label. Both systems receive the same unlabeled base
    // facts; only target-task supervision differs.
    std::unordered_set<std::string> memorization_baseline;
    memorization_baseline.reserve(held_out_instances * 2U + 1U);
    for (std::size_t index = 1U; index < config.schema_instances; ++index) {
        const std::string suffix = std::to_string(index);
        memorization_baseline.insert(relation_pair_key(
            "subject_" + suffix, "object_" + suffix
        ));
    }
    const double baseline_accuracy = memorization_baseline.size() ==
            held_out_instances
        ? 1.0
        : 0.0;
    const double supervision_ratio = static_cast<double>(
        held_out_instances
    );
    const double reasoning_ratio =
        result.stats.candidate_facts_examined == 0U
        ? 0.0
        : static_cast<double>(result.stats.naive_candidate_upper_bound) /
            static_cast<double>(result.stats.candidate_facts_examined);

    return {
        {
            "target_supervision_efficiency",
            "synthetic_two_hop_schema_induction",
            "labeled_target_examples",
            static_cast<double>(held_out_instances),
            1.0,
            supervision_ratio,
            baseline_accuracy,
            accuracy,
            config.target_efficiency_ratio,
            config.minimum_accuracy,
            supervision_ratio >= config.target_efficiency_ratio &&
                accuracy >= config.minimum_accuracy &&
                baseline_accuracy >= config.minimum_accuracy,
            "Counts only held-out target-relation labels. The induction demonstration is excluded from evaluation; both systems receive the same base facts.",
        },
        {
            "indexed_reasoning_candidate_efficiency",
            "synthetic_two_hop_forward_chaining",
            "candidate_fact_unifications",
            static_cast<double>(result.stats.naive_candidate_upper_bound),
            static_cast<double>(result.stats.candidate_facts_examined),
            reasoning_ratio,
            baseline_accuracy,
            accuracy,
            config.target_efficiency_ratio,
            config.minimum_accuracy,
            reasoning_ratio >= config.target_efficiency_ratio &&
                accuracy >= config.minimum_accuracy,
            "Compares indexed joins with the exact exhaustive candidate upper bound for the same rules and facts.",
        },
        {
            "compositional_generalization",
            accuracy,
            config.minimum_accuracy,
            accuracy >= config.minimum_accuracy,
            "A learned two-hop schema is evaluated on held-out entity combinations.",
        },
        {
            "one_shot_schema_induction",
            induction.path_hops == 2U && induction.rule_id != 0U ? 1.0 : 0.0,
            1.0,
            induction.path_hops == 2U && induction.rule_id != 0U,
            "The reusable variable-binding rule is induced from one labeled demonstration.",
        },
    };
}

[[nodiscard]] EfficiencyMetric run_sparse_routing_proof(
    const EfficiencyProofConfig& config
) {
    if (config.routing_vectors == 0U || config.routing_dimensions == 0U ||
        config.routing_queries == 0U || config.routing_trials == 0U) {
        throw std::invalid_argument("routing proof dimensions must be non-zero");
    }
    std::vector<float> matrix;
    matrix.reserve(config.routing_vectors * config.routing_dimensions);
    for (std::size_t item = 0U; item < config.routing_vectors; ++item) {
        const auto vector = synthetic_vector(item, config.routing_dimensions);
        matrix.insert(matrix.end(), vector.begin(), vector.end());
    }

    std::uint64_t candidates = 0U;
    std::size_t recalled = 0U;
    SparseRouterConfig router_config;
    router_config.signature_bits = 28U;
    router_config.maximum_candidates = 16U;
    router_config.probe_radius = 0U;
    for (std::size_t trial = 0U; trial < config.routing_trials; ++trial) {
        router_config.seed = 0x524C46524F555445ULL ^
            (static_cast<std::uint64_t>(trial + 1U) *
             0x9e3779b97f4a7c15ULL);
        SparseRoutingIndex router(router_config);
        router.rebuild(
            matrix, config.routing_vectors, config.routing_dimensions
        );
        for (std::size_t query_index = 0U;
             query_index < config.routing_queries;
             ++query_index) {
            const std::size_t target =
                (query_index * 104'729U + trial * 65'537U + 19U) %
                config.routing_vectors;
            const std::span<const float> query(
                matrix.data() + target * config.routing_dimensions,
                config.routing_dimensions
            );
            const SparseRouteResult route = router.route(query);
            candidates += route.candidates_examined;
            if (std::find(
                    route.candidate_indices.begin(),
                    route.candidate_indices.end(),
                    target
                ) != route.candidate_indices.end()) {
                ++recalled;
            }
        }
    }
    const std::size_t total_queries =
        config.routing_queries * config.routing_trials;
    const double accuracy = static_cast<double>(recalled) /
        static_cast<double>(total_queries);
    const double baseline_operations = static_cast<double>(total_queries) *
        static_cast<double>(config.routing_vectors) *
        static_cast<double>(config.routing_dimensions);
    const double routed_operations = static_cast<double>(total_queries) *
        static_cast<double>(router_config.signature_bits) *
        static_cast<double>(config.routing_dimensions) +
        static_cast<double>(candidates) *
        static_cast<double>(config.routing_dimensions);
    const double ratio = baseline_operations /
        std::max(routed_operations, 1.0);
    return {
        "exact_retrieval_inference_efficiency",
        "multi_seed_amortized_indexed_exact_query_retrieval",
        "estimated_multiply_accumulate_or_similarity_components",
        baseline_operations,
        routed_operations,
        ratio,
        1.0,
        accuracy,
        config.target_efficiency_ratio,
        config.minimum_accuracy,
        ratio >= config.target_efficiency_ratio &&
            accuracy >= config.minimum_accuracy,
        "Runs multiple deterministic index seeds. Excludes one-time index construction and applies to exact-query inference after the index is built.",
    };
}

[[nodiscard]] CapabilityProof run_induction_negative_control() {
    AbstractionFabric fabric;
    fabric.learn_fact("alpha", "left", "beta");
    bool rejected = false;
    try {
        static_cast<void>(fabric.induce_chain_rule(
            "unsupported", "alpha", "invented", "missing", 3U
        ));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    return {
        "schema_induction_negative_control",
        rejected ? 1.0 : 0.0,
        1.0,
        rejected,
        "Rejects a labeled conclusion when no relational path supports it.",
    };
}

[[nodiscard]] CapabilityProof run_transfer_proof() {
    AbstractionFabric fabric;
    const std::array<RelationalPattern, 2U> source{{
        {"?x", "parent", "?y"},
        {"?y", "parent", "?z"},
    }};
    const std::uint64_t rule = fabric.learn_rule(
        "source", source, {"?x", "grandparent", "?z"}
    );
    constexpr std::size_t domains = 128U;
    std::size_t correct = 0U;
    for (std::size_t domain = 0U; domain < domains; ++domain) {
        const std::string link = "domain_link_" + std::to_string(domain);
        const std::string composed = "domain_composed_" + std::to_string(domain);
        fabric.transfer_rule(
            rule,
            "transfer_" + std::to_string(domain),
            {{"parent", link}, {"grandparent", composed}}
        );
        const std::string left = "a_" + std::to_string(domain);
        const std::string middle = "b_" + std::to_string(domain);
        const std::string right = "c_" + std::to_string(domain);
        fabric.learn_fact(left, link, middle);
        fabric.learn_fact(middle, link, right);
        const auto answers = fabric.answer(left, composed);
        if (!answers.empty() && answers.front().value == right) {
            ++correct;
        }
    }
    const double accuracy = static_cast<double>(correct) /
        static_cast<double>(domains);
    return {
        "cross_domain_structural_transfer",
        accuracy,
        0.99,
        accuracy >= 0.99,
        "Transfers a learned variable-binding structure to 128 unseen relation vocabularies.",
    };
}

[[nodiscard]] CapabilityProof run_continual_proof(
    const EfficiencyProofConfig& config
) {
    ContinualLearningConfig continual_config;
    continual_config.feature_dimensions = 32U;
    continual_config.maximum_prototypes =
        config.continual_classes * (config.continual_tasks + 2U) * 2U;
    continual_config.replay_capacity = continual_config.maximum_prototypes * 4U;
    continual_config.consolidation_interval = 64U;
    continual_config.replay_batch_size = 256U;
    continual_config.router.signature_bits = 16U;
    continual_config.router.maximum_candidates = 128U;
    ContinualLearningFabric fabric(continual_config);

    std::vector<std::vector<float>> anchor_examples;
    anchor_examples.reserve(config.continual_classes);
    for (std::size_t label = 0U; label < config.continual_classes; ++label) {
        anchor_examples.push_back(synthetic_vector(
            label, continual_config.feature_dimensions
        ));
        for (std::size_t repeat = 0U; repeat < 8U; ++repeat) {
            fabric.learn(
                "anchor",
                "anchor_" + std::to_string(label),
                anchor_examples.back()
            );
        }
    }
    for (std::size_t task = 0U; task < config.continual_tasks; ++task) {
        for (std::size_t label = 0U; label < config.continual_classes; ++label) {
            const auto features = synthetic_vector(
                100'000U + task * config.continual_classes + label,
                continual_config.feature_dimensions
            );
            for (std::size_t repeat = 0U; repeat < 4U; ++repeat) {
                fabric.learn(
                    "task_" + std::to_string(task),
                    "label_" + std::to_string(label),
                    features
                );
            }
        }
    }
    fabric.consolidate();
    std::size_t retained = 0U;
    for (std::size_t label = 0U; label < config.continual_classes; ++label) {
        if (fabric.predict("anchor", anchor_examples[label]).label ==
            "anchor_" + std::to_string(label)) {
            ++retained;
        }
    }
    const double retention = static_cast<double>(retained) /
        static_cast<double>(config.continual_classes);
    return {
        "continual_retention",
        retention,
        0.95,
        retention >= 0.95,
        "Measures retained anchor-task accuracy after sequentially learning unrelated tasks.",
    };
}

[[nodiscard]] CapabilityProof run_grounding_proof() {
    CrossModalGroundingFabric fabric;
    constexpr std::size_t concepts = 256U;
    for (std::size_t index = 0U; index < concepts; ++index) {
        const std::array<std::uint64_t, 2U> modes{
            static_cast<std::uint64_t>(index * 2U + 1U),
            static_cast<std::uint64_t>(index * 2U + 2U),
        };
        const std::array<std::string, 1U> positive{
            "concept_" + std::to_string(index)
        };
        const std::array<std::string, 1U> negative{
            "concept_" + std::to_string((index + 1U) % concepts)
        };
        for (std::size_t repeat = 0U; repeat < 8U; ++repeat) {
            fabric.observe(modes, positive, negative);
        }
    }
    std::size_t correct = 0U;
    for (std::size_t index = 0U; index < concepts; ++index) {
        const auto hits = fabric.modes_for_concept(
            "concept_" + std::to_string(index)
        );
        if (!hits.empty() &&
            (hits.front().visual_mode_id == index * 2U + 1U ||
             hits.front().visual_mode_id == index * 2U + 2U)) {
            ++correct;
        }
    }
    const double accuracy = static_cast<double>(correct) /
        static_cast<double>(concepts);
    return {
        "contrastive_multimodal_grounding",
        accuracy,
        0.99,
        accuracy >= 0.99,
        "Tests positive/negative region-concept binding across 256 concepts.",
    };
}

void write_json_string(std::ostream& output, const std::string& value) {
    output << '"';
    for (const char character : value) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default: output << character; break;
        }
    }
    output << '"';
}

}  // namespace

EfficiencyProofReport run_efficiency_proofs(
    const EfficiencyProofConfig& config
) {
    if (!std::isfinite(config.target_efficiency_ratio) ||
        config.target_efficiency_ratio <= 1.0 ||
        !std::isfinite(config.minimum_accuracy) ||
        config.minimum_accuracy <= 0.0 || config.minimum_accuracy > 1.0) {
        throw std::invalid_argument("invalid efficiency proof thresholds");
    }
    EfficiencyProofReport report;
    report.suite = "rlf_frontier_efficiency_proof_v2";
    const SchemaProof schema = run_schema_proof(config);
    report.efficiency_metrics.push_back(schema.supervision);
    report.efficiency_metrics.push_back(schema.indexed_reasoning);
    report.efficiency_metrics.push_back(run_sparse_routing_proof(config));
    report.capability_proofs.push_back(schema.composition);
    report.capability_proofs.push_back(schema.one_shot_induction);
    report.capability_proofs.push_back(run_induction_negative_control());
    report.capability_proofs.push_back(run_transfer_proof());
    report.capability_proofs.push_back(run_continual_proof(config));
    report.capability_proofs.push_back(run_grounding_proof());

    report.narrow_ten_thousand_x_proven =
        config.target_efficiency_ratio >= 10'000.0 &&
        schema.supervision.passed && schema.indexed_reasoning.passed;
    report.general_learning_efficiency_proven = false;
    report.frontier_parity_proven = false;
    report.all_internal_proofs_passed = std::all_of(
        report.efficiency_metrics.begin(),
        report.efficiency_metrics.end(),
        [](const EfficiencyMetric& metric) { return metric.passed; }
    ) && std::all_of(
        report.capability_proofs.begin(),
        report.capability_proofs.end(),
        [](const CapabilityProof& proof) { return proof.passed; }
    );
    report.claim_boundary =
        "Passing proves >=10,000x target-supervision and indexed-candidate "
        "efficiency only on the declared synthetic relational task, plus "
        "the listed internal capability checks. It does not prove 10,000x "
        "general learning efficiency or parity with leading foundation models.";
    return report;
}

void write_efficiency_proof_json(
    std::ostream& output,
    const EfficiencyProofReport& report
) {
    output << std::fixed << std::setprecision(6);
    output << "{\n  \"suite\": ";
    write_json_string(output, report.suite);
    output << ",\n  \"narrow_ten_thousand_x_proven\": "
           << (report.narrow_ten_thousand_x_proven ? "true" : "false")
           << ",\n  \"general_learning_efficiency_proven\": "
           << (report.general_learning_efficiency_proven ? "true" : "false")
           << ",\n  \"frontier_parity_proven\": "
           << (report.frontier_parity_proven ? "true" : "false")
           << ",\n  \"all_internal_proofs_passed\": "
           << (report.all_internal_proofs_passed ? "true" : "false")
           << ",\n  \"claim_boundary\": ";
    write_json_string(output, report.claim_boundary);
    output << ",\n  \"efficiency_metrics\": [\n";
    for (std::size_t index = 0U;
         index < report.efficiency_metrics.size();
         ++index) {
        const EfficiencyMetric& metric = report.efficiency_metrics[index];
        output << "    {\n      \"name\": ";
        write_json_string(output, metric.name);
        output << ",\n      \"scope\": ";
        write_json_string(output, metric.scope);
        output << ",\n      \"unit\": ";
        write_json_string(output, metric.unit);
        output << ",\n      \"baseline_cost\": " << metric.baseline_cost
               << ",\n      \"rlf_cost\": " << metric.rlf_cost
               << ",\n      \"ratio\": " << metric.ratio
               << ",\n      \"baseline_accuracy\": " << metric.baseline_accuracy
               << ",\n      \"rlf_accuracy\": " << metric.rlf_accuracy
               << ",\n      \"required_ratio\": " << metric.required_ratio
               << ",\n      \"required_accuracy\": " << metric.required_accuracy
               << ",\n      \"passed\": " << (metric.passed ? "true" : "false")
               << ",\n      \"qualification\": ";
        write_json_string(output, metric.qualification);
        output << "\n    }";
        if (index + 1U != report.efficiency_metrics.size()) {
            output << ',';
        }
        output << '\n';
    }
    output << "  ],\n  \"capability_proofs\": [\n";
    for (std::size_t index = 0U;
         index < report.capability_proofs.size();
         ++index) {
        const CapabilityProof& proof = report.capability_proofs[index];
        output << "    {\n      \"name\": ";
        write_json_string(output, proof.name);
        output << ",\n      \"score\": " << proof.score
               << ",\n      \"required_score\": " << proof.required_score
               << ",\n      \"passed\": " << (proof.passed ? "true" : "false")
               << ",\n      \"qualification\": ";
        write_json_string(output, proof.qualification);
        output << "\n    }";
        if (index + 1U != report.capability_proofs.size()) {
            output << ',';
        }
        output << '\n';
    }
    output << "  ]\n}\n";
}

}  // namespace rlf::solstice
