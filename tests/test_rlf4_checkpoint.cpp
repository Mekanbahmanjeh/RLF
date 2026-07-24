#include "test_framework.hpp"

#include "rlf/core/deterministic_rng.hpp"
#include "rlf/core/temporal_predictive_fabric.hpp"
#include "rlf/storage/rlf4_checkpoint.hpp"

#include <filesystem>
#include <fstream>
#include <vector>

RLF_TEST_CASE("RLF-4 checkpoint round trip and corruption rejection") {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "rlf4_checkpoint_test.rlf";
    const std::filesystem::path corrupt =
        std::filesystem::temp_directory_path() / "rlf4_checkpoint_corrupt.rlf";
    rlf::core::TemporalFabricConfig config;
    config.dimension = 8U;
    config.maximum_context_order = 4U;
    config.minimum_context_support = 1U;
    config.minimum_option_support = 2U;
    rlf::core::TemporalPredictiveFabric fabric(config, 0x4444ULL);
    rlf::core::DeterministicRng rng(0x5555ULL);
    std::vector<rlf::core::PhaseVector> sequence;
    for (std::size_t index = 0U; index < 4U; ++index) {
        sequence.push_back(rlf::core::PhaseVector::random(config.dimension, rng));
    }
    for (std::size_t repetition = 0U; repetition < 8U; ++repetition) {
        fabric.observe_sequence(sequence);
    }
    fabric.discover_options();
    rlf::storage::save_rlf4_checkpoint(path, fabric);
    auto restored = rlf::storage::load_rlf4_checkpoint(path);
    RLF_CHECK(restored.deterministic_hash() == fabric.deterministic_hash());
    const auto summary = rlf::storage::inspect_rlf4_checkpoint(path);
    RLF_CHECK(summary.format_version == 6U);
    RLF_CHECK(summary.prototype_count == fabric.prototypes().size());

    std::ifstream input(path, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>()
    );
    input.close();
    RLF_CHECK(bytes.size() > 32U);
    bytes.back() = static_cast<char>(bytes.back() ^ 0x01);
    std::ofstream output(corrupt, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    output.close();
    bool rejected = false;
    try {
        static_cast<void>(rlf::storage::load_rlf4_checkpoint(corrupt));
    } catch (...) {
        rejected = true;
    }
    RLF_CHECK(rejected);
    std::filesystem::remove(path);
    std::filesystem::remove(corrupt);
}

RLF_TEST_CASE("RLF-4 checkpoint rejects truncation and configured size bounds") {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "rlf4_checkpoint_bounds.rlf";
    const std::filesystem::path truncated =
        std::filesystem::temp_directory_path() / "rlf4_checkpoint_truncated.rlf";
    rlf::core::TemporalFabricConfig config;
    config.dimension = 8U;
    config.maximum_context_order = 3U;
    config.minimum_context_support = 1U;
    rlf::core::TemporalPredictiveFabric fabric(config, 0xB0A4D5ULL);
    rlf::core::DeterministicRng rng(0xABCDULL);
    for (std::size_t index = 0U; index < 12U; ++index) {
        static_cast<void>(fabric.observe(
            rlf::core::PhaseVector::random(config.dimension, rng)
        ));
    }
    rlf::storage::save_rlf4_checkpoint(path, fabric);
    std::ifstream input(path, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>()
    );
    input.close();
    RLF_CHECK(bytes.size() > 32U);
    bytes.resize(bytes.size() - 11U);
    std::ofstream output(truncated, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    output.close();

    bool truncation_rejected = false;
    try {
        static_cast<void>(rlf::storage::load_rlf4_checkpoint(truncated));
    } catch (...) {
        truncation_rejected = true;
    }
    RLF_CHECK(truncation_rejected);

    rlf::storage::Rlf4CheckpointLoadOptions limits;
    limits.maximum_file_bytes = 16U;
    bool limit_rejected = false;
    try {
        static_cast<void>(rlf::storage::load_rlf4_checkpoint(path, limits));
    } catch (...) {
        limit_rejected = true;
    }
    RLF_CHECK(limit_rejected);
    std::filesystem::remove(path);
    std::filesystem::remove(truncated);
}
