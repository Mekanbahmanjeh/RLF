#include "test_framework.hpp"

#include "rlf/experiments/persistence_roundtrip.hpp"

#include <filesystem>

RLF_TEST_CASE("persistence experiment round trips and rejects mutations") {
    const std::filesystem::path checkpoint =
        std::filesystem::temp_directory_path() /
        "rlf-persistence-experiment.rlf";
    const rlf::experiments::PersistenceRoundtripResult result =
        rlf::experiments::run_persistence_roundtrip({
            .seed = 811ULL,
            .dimension = 32U,
            .mode_count = 3U,
            .memory_records = 4U,
            .checkpoint_path = checkpoint,
        });

    RLF_CHECK(result.mode_count == 3U);
    RLF_CHECK(result.memory_records == 4U);
    RLF_CHECK(result.minimum_mode_key_similarity == 1.0);
    RLF_CHECK(result.minimum_mode_transformation_similarity == 1.0);
    RLF_CHECK(result.minimum_memory_key_similarity == 1.0);
    RLF_CHECK(result.corruption_rejected);
    RLF_CHECK(result.truncation_rejected);
    std::filesystem::remove(checkpoint);
}
