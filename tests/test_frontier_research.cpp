#include "test_framework.hpp"

#include "rlf/solstice/abstraction_fabric.hpp"
#include "rlf/solstice/continual_learning.hpp"
#include "rlf/solstice/grounding_fabric.hpp"
#include "rlf/solstice/sparse_router.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {

[[nodiscard]] std::vector<float> basis(
    const std::size_t dimensions,
    const std::size_t active,
    const float noise = 0.0F
) {
    std::vector<float> values(dimensions, noise);
    values.at(active) = 1.0F;
    return values;
}

}  // namespace

RLF_TEST_CASE("Frontier abstraction fabric performs multi-hop variable binding") {
    rlf::solstice::AbstractionFabric fabric;
    fabric.learn_fact("ada", "parent", "bea");
    fabric.learn_fact("bea", "parent", "cy");
    const std::array<rlf::solstice::RelationalPattern, 2U> premises{{
        {"?x", "parent", "?y"},
        {"?y", "parent", "?z"},
    }};
    fabric.learn_rule(
        "grandparent composition",
        premises,
        {"?x", "grandparent", "?z"},
        0.98
    );
    const auto answers = fabric.answer("ada", "grandparent");
    RLF_CHECK(!answers.empty());
    RLF_CHECK(answers.front().value == "cy");
    RLF_CHECK(answers.front().confidence > 0.8);
    RLF_CHECK(answers.front().proof.size() >= 3U);
}

RLF_TEST_CASE("Frontier abstraction transfers relational structure to a new domain") {
    rlf::solstice::AbstractionFabric fabric;
    const std::array<rlf::solstice::RelationalPattern, 2U> source{{
        {"?x", "parent", "?y"},
        {"?y", "parent", "?z"},
    }};
    const std::uint64_t source_rule = fabric.learn_rule(
        "two-hop family", source, {"?x", "grandparent", "?z"}
    );
    fabric.transfer_rule(
        source_rule,
        "two-hop organization",
        {{"parent", "manages"}, {"grandparent", "indirectly_manages"}}
    );
    fabric.learn_fact("alice", "manages", "bob");
    fabric.learn_fact("bob", "manages", "carol");
    const auto answers = fabric.answer("alice", "indirectly_manages");
    RLF_CHECK(!answers.empty());
    RLF_CHECK(answers.front().value == "carol");
}

RLF_TEST_CASE("Frontier sparse router finds exact items with bounded candidates") {
    constexpr std::size_t dimensions = 32U;
    constexpr std::size_t count = 4'096U;
    std::vector<float> matrix;
    matrix.reserve(dimensions * count);
    for (std::size_t item = 0U; item < count; ++item) {
        for (std::size_t dimension = 0U; dimension < dimensions; ++dimension) {
            const std::uint64_t value =
                (item + 1U) * (dimension + 17U) * 0x9e3779b9ULL;
            matrix.push_back(static_cast<float>(
                static_cast<double>(value % 10'007U) / 5'003.5 - 1.0
            ));
        }
    }
    rlf::solstice::SparseRouterConfig config;
    config.signature_bits = 18U;
    config.maximum_candidates = 128U;
    config.probe_radius = 2U;
    rlf::solstice::SparseRoutingIndex router(config);
    router.rebuild(matrix, count, dimensions);
    const std::size_t target = 1'337U;
    const auto result = router.route(std::span<const float>(
        matrix.data() + target * dimensions, dimensions
    ));
    RLF_CHECK(std::find(
        result.candidate_indices.begin(),
        result.candidate_indices.end(),
        target
    ) != result.candidate_indices.end());
    RLF_CHECK(result.candidates_examined <= config.maximum_candidates);
    RLF_CHECK(result.candidates_examined * 20U < result.exhaustive_candidates);
}

RLF_TEST_CASE("Frontier sparse router incremental updates exactly match rebuild") {
    constexpr std::size_t dimensions = 16U;
    constexpr std::size_t initial_count = 512U;
    std::vector<float> matrix(initial_count * dimensions, 0.0F);
    for (std::size_t item = 0U; item < initial_count; ++item) {
        for (std::size_t dimension = 0U; dimension < dimensions; ++dimension) {
            matrix[item * dimensions + dimension] = static_cast<float>(
                static_cast<double>(
                    ((item + 11U) * (dimension + 23U) * 2'654'435'761ULL) %
                    65'521ULL
                ) / 32'760.5 - 1.0
            );
        }
    }
    rlf::solstice::SparseRouterConfig config;
    config.signature_bits = 14U;
    config.maximum_candidates = 96U;
    config.probe_radius = 2U;
    rlf::solstice::SparseRoutingIndex incremental(config);
    rlf::solstice::SparseRoutingIndex reference(config);
    incremental.rebuild(matrix, initial_count, dimensions);
    reference.rebuild(matrix, initial_count, dimensions);

    std::size_t updated = 0U;
    for (std::size_t index = 3U; index < initial_count; index += 13U) {
        for (std::size_t dimension = 0U; dimension < dimensions; ++dimension) {
            matrix[index * dimensions + dimension] +=
                (dimension % 2U == 0U ? 0.35F : -0.27F);
        }
        incremental.update(
            index,
            std::span<const float>(
                matrix.data() + index * dimensions, dimensions
            )
        );
        ++updated;
    }
    std::vector<float> appended(dimensions, 0.0F);
    for (std::size_t dimension = 0U; dimension < dimensions; ++dimension) {
        appended[dimension] = dimension % 3U == 0U ? 0.75F : -0.125F;
    }
    incremental.append(appended);
    matrix.insert(matrix.end(), appended.begin(), appended.end());
    reference.rebuild(matrix, initial_count + 1U, dimensions);

    for (std::size_t query = 0U; query < initial_count + 1U; query += 7U) {
        const auto vector = std::span<const float>(
            matrix.data() + query * dimensions, dimensions
        );
        const auto actual = incremental.route(vector);
        const auto expected = reference.route(vector);
        RLF_CHECK(actual.candidate_indices == expected.candidate_indices);
        RLF_CHECK(actual.signatures_probed == expected.signatures_probed);
        RLF_CHECK(actual.candidates_examined == expected.candidates_examined);
        RLF_CHECK(actual.exhaustive_candidates == expected.exhaustive_candidates);
    }
    const auto stats = incremental.operation_stats();
    RLF_CHECK(stats.full_rebuilds == 1U);
    RLF_CHECK(stats.vectors_rebuilt == initial_count);
    RLF_CHECK(stats.incremental_updates == updated + 1U);
    RLF_CHECK(stats.vectors_incrementally_updated == updated);
    RLF_CHECK(stats.vectors_appended == 1U);
}

RLF_TEST_CASE("Frontier continual fabric retains old skills after new-task learning") {
    rlf::solstice::ContinualLearningConfig config;
    config.feature_dimensions = 8U;
    config.maximum_prototypes = 128U;
    config.replay_capacity = 128U;
    config.consolidation_interval = 8U;
    config.replay_batch_size = 32U;
    config.router.signature_bits = 8U;
    config.router.maximum_candidates = 32U;
    rlf::solstice::ContinualLearningFabric fabric(config);

    const std::vector<float> old_a = basis(8U, 0U, 0.01F);
    const std::vector<float> old_b = basis(8U, 1U, 0.01F);
    for (std::size_t repeat = 0U; repeat < 24U; ++repeat) {
        fabric.learn("old", "alpha", old_a);
        fabric.learn("old", "beta", old_b);
    }
    RLF_CHECK(fabric.predict("old", old_a).label == "alpha");
    RLF_CHECK(fabric.predict("old", old_b).label == "beta");

    for (std::size_t label = 0U; label < 5U; ++label) {
        const std::vector<float> example = basis(8U, label + 2U, 0.02F);
        for (std::size_t repeat = 0U; repeat < 24U; ++repeat) {
            fabric.learn(
                "new",
                "class_" + std::to_string(label),
                example
            );
        }
    }
    fabric.consolidate();
    RLF_CHECK(fabric.predict("old", old_a).label == "alpha");
    RLF_CHECK(fabric.predict("old", old_b).label == "beta");
    RLF_CHECK(fabric.stats().consolidations > 0U);
}

RLF_TEST_CASE("Frontier grounding learns contrastive region-word bindings") {
    rlf::solstice::CrossModalGroundingFabric grounding;
    const std::array<std::uint64_t, 2U> red_modes{10U, 11U};
    const std::array<std::uint64_t, 2U> blue_modes{20U, 21U};
    const std::array<std::string, 2U> red_concepts{"red", "warning"};
    const std::array<std::string, 2U> blue_concepts{"blue", "information"};
    for (std::size_t repeat = 0U; repeat < 12U; ++repeat) {
        grounding.observe(red_modes, red_concepts, blue_concepts);
        grounding.observe(blue_modes, blue_concepts, red_concepts);
    }
    const auto red_hits = grounding.modes_for_concept("red");
    RLF_CHECK(!red_hits.empty());
    RLF_CHECK(red_hits.front().visual_mode_id == 10U ||
              red_hits.front().visual_mode_id == 11U);
    const auto composed = grounding.compose_concepts(red_concepts);
    RLF_CHECK(!composed.empty());
    RLF_CHECK(composed.front().score > 0.5);
}

RLF_TEST_CASE("Persistent grounding indexes preserve exact learning and recall") {
    rlf::solstice::CrossModalGroundingFabric reference;
    rlf::solstice::CrossModalGroundingFabric indexed;
    reference.set_persistent_link_index(false);
    indexed.set_persistent_link_index(true);
    const std::array<std::uint64_t, 3U> modes{7U, 11U, 19U};
    const std::array<std::string, 3U> positives{"Red alert", "round", "metal"};
    const std::array<std::string, 2U> negatives{"blue", "soft"};
    for (std::size_t repeat = 0U; repeat < 5U; ++repeat) {
        reference.observe(modes, positives, negatives);
        indexed.observe(modes, positives, negatives);
        RLF_CHECK(reference.deterministic_hash() == indexed.deterministic_hash());
    }

    const auto check_hits = [](const auto& left, const auto& right) {
        RLF_CHECK(left.size() == right.size());
        for (std::size_t index = 0U; index < left.size(); ++index) {
            RLF_CHECK(left[index].visual_mode_id == right[index].visual_mode_id);
            RLF_CHECK(left[index].concept_name == right[index].concept_name);
            RLF_CHECK(left[index].score == right[index].score);
            RLF_CHECK(left[index].support == right[index].support);
        }
    };
    check_hits(reference.concepts_for_mode(11U), indexed.concepts_for_mode(11U));
    check_hits(reference.modes_for_concept("RED ALERT"),
               indexed.modes_for_concept("RED ALERT"));
    check_hits(reference.compose_concepts(positives),
               indexed.compose_concepts(positives));

    auto resumed = rlf::solstice::CrossModalGroundingFabric::from_snapshot(
        indexed.snapshot()
    );
    resumed.set_persistent_link_index(true);
    indexed.observe(modes, positives, negatives);
    resumed.observe(modes, positives, negatives);
    RLF_CHECK(indexed.deterministic_hash() == resumed.deterministic_hash());

    auto adversarial_snapshot = indexed.snapshot();
    adversarial_snapshot.links.front().confidence = 0.0;
    adversarial_snapshot.links.push_back(adversarial_snapshot.links.front());
    auto adversarial_reference =
        rlf::solstice::CrossModalGroundingFabric::from_snapshot(
            adversarial_snapshot
        );
    auto adversarial_indexed =
        rlf::solstice::CrossModalGroundingFabric::from_snapshot(
            std::move(adversarial_snapshot)
        );
    adversarial_reference.set_persistent_link_index(false);
    adversarial_indexed.set_persistent_link_index(true);
    adversarial_reference.observe(modes, positives, negatives);
    adversarial_indexed.observe(modes, positives, negatives);
    RLF_CHECK(adversarial_reference.deterministic_hash() ==
              adversarial_indexed.deterministic_hash());

    const auto reference_stats = reference.operation_stats();
    const auto indexed_stats = indexed.operation_stats();
    RLF_CHECK(reference_stats.link_lookups > 0U);
    RLF_CHECK(reference_stats.full_lookup_entries_rebuilt > 0U);
    RLF_CHECK(reference_stats.full_confidence_sweep_entries > 0U);
    RLF_CHECK(reference_stats.derived_sort_entries > 0U);
    RLF_CHECK(indexed_stats.link_lookups > 0U);
    RLF_CHECK(indexed_stats.full_lookup_entries_rebuilt == 0U);
    RLF_CHECK(indexed_stats.full_confidence_sweep_entries == 0U);
    RLF_CHECK(indexed_stats.incremental_posting_inserts > 0U);
    RLF_CHECK(indexed_stats.mode_query_indexed_candidates > 0U);
    RLF_CHECK(indexed_stats.concept_query_indexed_candidates > 0U);
}
