#include "test_framework.hpp"

#include "rlf/baselines/nearest_neighbor.hpp"
#include "rlf/baselines/prototype_classifier.hpp"
#include "rlf/baselines/transition_table.hpp"
#include "rlf/core/phase_vector.hpp"

RLF_TEST_CASE("nearest-neighbor baseline retrieves exact keys") {
    rlf::baselines::NearestNeighborMemory memory(4U);
    const rlf::core::PhaseVector key({0.1F, 0.2F, 0.3F, 0.4F});
    const rlf::core::PhaseVector value({1.1F, 1.2F, 1.3F, 1.4F});
    const std::uint64_t id = memory.insert(key, value);
    const auto matches = memory.retrieve(key, 1U);

    RLF_CHECK(matches.size() == 1U);
    RLF_CHECK(matches[0U].record_id == id);
    RLF_CHECK(matches[0U].similarity == 1.0);
}

RLF_TEST_CASE("transition-table baseline chooses majority then stable ID") {
    rlf::baselines::TransitionTablePredictor predictor;
    predictor.observe(1ULL, 3ULL);
    predictor.observe(1ULL, 2ULL);
    RLF_CHECK(predictor.predict(1ULL).value() == 2ULL);
    predictor.observe(1ULL, 3ULL);
    RLF_CHECK(predictor.predict(1ULL).value() == 3ULL);
}

RLF_TEST_CASE("prototype classifier separates phase clusters") {
    rlf::baselines::FixedPrototypeClassifier classifier(4U);
    classifier.observe(
        1ULL,
        rlf::core::PhaseVector({0.1F, 0.1F, 0.1F, 0.1F})
    );
    classifier.observe(
        1ULL,
        rlf::core::PhaseVector({0.2F, 0.2F, 0.2F, 0.2F})
    );
    classifier.observe(
        2ULL,
        rlf::core::PhaseVector({0.0F, 1.0F, 2.0F, 3.0F})
    );

    const auto prediction = classifier.predict(
        rlf::core::PhaseVector({0.15F, 0.15F, 0.15F, 0.15F})
    );
    RLF_CHECK(prediction.has_value());
    RLF_CHECK(prediction->label == 1ULL);
}
