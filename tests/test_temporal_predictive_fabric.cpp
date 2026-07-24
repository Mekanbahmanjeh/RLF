#include "test_framework.hpp"

#include "rlf/core/deterministic_rng.hpp"
#include "rlf/core/temporal_predictive_fabric.hpp"

#include <array>
#include <cstddef>
#include <vector>

RLF_TEST_CASE("temporal fabric learns noisy prototypes and variable contexts") {
    rlf::core::TemporalFabricConfig config;
    config.dimension = 12U;
    config.maximum_context_order = 6U;
    config.minimum_context_support = 1U;
    config.prototype_merge_distance = 0.10;
    config.minimum_option_support = 3U;
    config.maximum_options = 64U;
    rlf::core::TemporalPredictiveFabric fabric(config, 0x524C4634ULL);
    rlf::core::DeterministicRng rng(0x1234ULL);
    const auto a = rlf::core::PhaseVector::random(config.dimension, rng);
    const auto b = rlf::core::PhaseVector::random(config.dimension, rng);
    const auto c = rlf::core::PhaseVector::random(config.dimension, rng);
    const std::array<rlf::core::PhaseVector, 3U> pattern{a, b, c};
    std::vector<rlf::core::PhaseVector> sequence;
    for (std::size_t repetition = 0U; repetition < 20U; ++repetition) {
        sequence.insert(sequence.end(), pattern.begin(), pattern.end());
    }
    fabric.observe_sequence(sequence);
    fabric.discover_options();
    RLF_CHECK(fabric.prototypes().size() == 3U);
    RLF_CHECK(!fabric.contexts().empty());
    RLF_CHECK(!fabric.options().empty());

    const auto a_id = fabric.match_prototype(a);
    const auto b_id = fabric.match_prototype(b);
    const auto c_id = fabric.match_prototype(c);
    RLF_CHECK(a_id.has_value());
    RLF_CHECK(b_id.has_value());
    RLF_CHECK(c_id.has_value());
    const std::array<std::uint64_t, 2U> history{*a_id, *b_id};
    const auto prediction = fabric.predict_next(history, true);
    RLF_CHECK(!prediction.outcomes.empty());
    RLF_CHECK(prediction.outcomes.front().prototype_id == *c_id);
    const auto forecast = fabric.forecast(history, 6U, true);
    RLF_CHECK(forecast.predicted_tokens == 6U);
    RLF_CHECK(forecast.decision_operations <= 6U);
}

RLF_TEST_CASE("temporal fabric snapshot preserves predictions") {
    rlf::core::TemporalFabricConfig config;
    config.dimension = 8U;
    config.maximum_context_order = 4U;
    config.minimum_context_support = 1U;
    config.minimum_option_support = 2U;
    rlf::core::TemporalPredictiveFabric fabric(config, 0xBEEFULL);
    rlf::core::DeterministicRng rng(0x9988ULL);
    const auto x = rlf::core::PhaseVector::random(config.dimension, rng);
    const auto y = rlf::core::PhaseVector::random(config.dimension, rng);
    std::vector<rlf::core::PhaseVector> sequence{x, y, x, y, x, y};
    fabric.observe_sequence(sequence);
    fabric.discover_options();
    const auto x_id = fabric.match_prototype(x);
    RLF_CHECK(x_id.has_value());
    const std::array<std::uint64_t, 1U> history{*x_id};
    const auto before = fabric.predict_next(history, true);
    auto restored = rlf::core::TemporalPredictiveFabric::from_snapshot(
        fabric.snapshot()
    );
    const auto after = restored.predict_next(history, true);
    RLF_CHECK(before.outcomes.size() == after.outcomes.size());
    RLF_CHECK(before.outcomes.front().prototype_id ==
              after.outcomes.front().prototype_id);
    RLF_CHECK(fabric.deterministic_hash() == restored.deterministic_hash());
}

RLF_TEST_CASE("temporal fabric detects a high-surprise regime change") {
    rlf::core::TemporalFabricConfig config;
    config.dimension = 8U;
    config.maximum_context_order = 3U;
    config.minimum_context_support = 1U;
    config.prototype_merge_distance = 0.08;
    config.surprise_slow_rate = 0.01;
    config.surprise_fast_rate = 0.90;
    config.change_threshold = 0.05;
    config.change_cooldown = 1U;
    rlf::core::TemporalPredictiveFabric fabric(config, 0x4348414E4745ULL);
    rlf::core::DeterministicRng rng(0x123456ULL);
    const auto a = rlf::core::PhaseVector::random(config.dimension, rng);
    const auto b = rlf::core::PhaseVector::random(config.dimension, rng);
    const auto c = rlf::core::PhaseVector::random(config.dimension, rng);
    for (std::size_t repetition = 0U; repetition < 24U; ++repetition) {
        static_cast<void>(fabric.observe(a));
        static_cast<void>(fabric.observe(b));
    }
    static_cast<void>(fabric.observe(a));
    const auto unexpected = fabric.observe(c);
    RLF_CHECK(unexpected.prediction_available);
    RLF_CHECK(unexpected.surprise > 1.0);
    RLF_CHECK(unexpected.change_detected);
}

RLF_TEST_CASE("temporal fabric rejects unknown phase observations without mutation") {
    rlf::core::TemporalFabricConfig config;
    config.dimension = 8U;
    config.prototype_merge_distance = 0.01;
    rlf::core::TemporalPredictiveFabric fabric(config, 0x554E4B4E4F574EULL);
    rlf::core::DeterministicRng rng(0x9876ULL);
    const auto known = rlf::core::PhaseVector::random(config.dimension, rng);
    const auto unknown = rlf::core::PhaseVector::random(config.dimension, rng);
    static_cast<void>(fabric.observe(known));
    const std::size_t before = fabric.prototypes().size();
    RLF_CHECK(!fabric.match_prototype(unknown).has_value());
    RLF_CHECK(fabric.prototypes().size() == before);
}

RLF_TEST_CASE("temporal fabric rejects duplicate snapshot structures") {
    rlf::core::TemporalFabricConfig config;
    config.dimension = 8U;
    config.maximum_context_order = 3U;
    config.minimum_context_support = 1U;
    rlf::core::TemporalPredictiveFabric fabric(config, 0x4455504C49434154ULL);
    rlf::core::DeterministicRng rng(0x1111ULL);
    const auto a = rlf::core::PhaseVector::random(config.dimension, rng);
    const auto b = rlf::core::PhaseVector::random(config.dimension, rng);
    for (std::size_t repetition = 0U; repetition < 4U; ++repetition) {
        static_cast<void>(fabric.observe(a));
        static_cast<void>(fabric.observe(b));
    }
    auto snapshot = fabric.snapshot();
    RLF_CHECK(!snapshot.contexts.empty());
    auto duplicate = snapshot.contexts.front();
    duplicate.id = snapshot.next_context_id++;
    snapshot.contexts.push_back(std::move(duplicate));
    bool rejected = false;
    try {
        static_cast<void>(
            rlf::core::TemporalPredictiveFabric::from_snapshot(std::move(snapshot))
        );
    } catch (...) {
        rejected = true;
    }
    RLF_CHECK(rejected);
}
