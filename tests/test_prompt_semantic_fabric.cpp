#include "test_framework.hpp"

#include "rlf/solstice/prompt_semantic_fabric.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace {

using rlf::solstice::PromptSemanticConfig;
using rlf::solstice::PromptSemanticFabric;

[[nodiscard]] PromptSemanticFabric trained_semantics() {
    PromptSemanticConfig config;
    config.phase_dimension = 64U;
    config.bucket_bits = 6U;
    config.minimum_support = 2U;
    config.maximum_expansions_per_word = 8U;
    PromptSemanticFabric fabric(config);
    for (std::size_t repeat = 0U; repeat < 8U; ++repeat) {
        fabric.train_record("bright crimson pigment on canvas");
        fabric.train_record("bright scarlet pigment on canvas");
        fabric.train_record("small crimson object beside square");
        fabric.train_record("small scarlet object beside square");
    }
    return fabric;
}

}  // namespace

RLF_TEST_CASE("prompt semantic fabric learns distributional prompt aliases") {
    const auto fabric = trained_semantics();
    const auto aliases = fabric.similar_words("scarlet", 8U);
    RLF_CHECK(std::find(aliases.begin(), aliases.end(), "crimson") !=
              aliases.end());
    const auto concepts = fabric.semantic_concept_hashes(
        "a scarlet square", 64U
    );
    RLF_CHECK(!concepts.empty());
    RLF_CHECK(fabric.stats().records_seen == 32U);
    RLF_CHECK(fabric.stats().modes_created > 0U);
}

RLF_TEST_CASE("prompt semantic snapshot restores deterministically") {
    const auto original = trained_semantics();
    const auto restored = PromptSemanticFabric::from_snapshot(
        original.snapshot()
    );
    RLF_CHECK(restored.deterministic_hash() == original.deterministic_hash());
    RLF_CHECK(restored.similar_words("crimson", 8U) ==
              original.similar_words("crimson", 8U));
}

RLF_TEST_CASE("prompt semantic snapshot rejects duplicate words") {
    auto snapshot = trained_semantics().snapshot();
    snapshot.modes.push_back(snapshot.modes.front());
    snapshot.modes.back().id = snapshot.next_mode_id++;
    bool rejected = false;
    try {
        static_cast<void>(PromptSemanticFabric::from_snapshot(
            std::move(snapshot)
        ));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    RLF_CHECK(rejected);
}

RLF_TEST_CASE("prompt semantic records never truncate silently") {
    PromptSemanticConfig config;
    config.maximum_words_per_record = 2U;
    PromptSemanticFabric fabric(config);
    fabric.train_record("alpha beta");
    RLF_CHECK_THROWS_AS(
        fabric.train_record("alpha beta gamma"),
        std::length_error
    );
}
