#include "test_framework.hpp"

#include "rlf/core/phase_vector.hpp"
#include "rlf/core/resonant_mode.hpp"
#include "rlf/memory/associative_memory.hpp"
#include "rlf/storage/checkpoint.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

namespace {

[[nodiscard]] std::filesystem::path checkpoint_path(
    const std::string& name
) {
    return std::filesystem::temp_directory_path() /
        ("rlf-" + name + ".rlf");
}

[[nodiscard]] rlf::storage::CheckpointData make_checkpoint() {
    rlf::core::FabricConfig config;
    config.dimension = 4U;
    config.maximum_modes = 16U;
    config.settling.candidate_count = 4U;
    config.settling.active_count = 2U;
    config.settling.maximum_cycles = 3U;
    config.structural_learning.enabled = true;

    std::vector<rlf::core::ResonantMode> modes;
    modes.emplace_back(
        1ULL,
        rlf::core::PhaseVector({0.1F, 0.2F, 0.3F, 0.4F}),
        rlf::core::PhaseVector({1.1F, 1.2F, 1.3F, 1.4F}),
        0.7F,
        0.8F,
        0.2F,
        3ULL
    );
    modes[0U].activation_count = 9ULL;
    modes[0U].successful_update_count = 7ULL;
    modes[0U].unsuccessful_update_count = 2ULL;
    modes[0U].last_used_step = 12ULL;
    modes[0U].recent_corrections.push_back({
        .context =
            rlf::core::PhaseVector({0.2F, 0.3F, 0.4F, 0.5F}),
        .desired_transformation =
            rlf::core::PhaseVector({1.0F, 1.1F, 1.2F, 1.3F}),
        .proposal_quality = 0.9,
        .improved_prediction = true,
        .step = 11ULL,
    });

    rlf::memory::AssociativeMemory memory(4U, 8U);
    static_cast<void>(memory.insert(
        rlf::core::PhaseVector({2.1F, 2.2F, 2.3F, 2.4F}),
        rlf::memory::BytePayload{7U, 8U, 9U},
        0.75F,
        5ULL
    ));

    return {
        .config = config,
        .master_seed = 123456ULL,
        .training_step = 12ULL,
        .modes = std::move(modes),
        .associative_memory = std::move(memory),
        .structural_statistics = {
            .modes_created = 3ULL,
            .modes_split = 1ULL,
            .modes_merged = 0ULL,
            .modes_pruned = 1ULL,
        },
        .update_strategy = "winner_only",
        .experiment_metadata = {
            {"experiment", "checkpoint_test"},
            {"note", "roundtrip"},
        },
    };
}

[[nodiscard]] std::vector<char> read_bytes(
    const std::filesystem::path& path
) {
    std::ifstream input(path, std::ios::binary);
    return {
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()
    };
}

void write_bytes(
    const std::filesystem::path& path,
    const std::vector<char>& bytes
) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

}  // namespace

RLF_TEST_CASE("checkpoint round trip preserves fabric and memory") {
    const std::filesystem::path path =
        checkpoint_path("checkpoint-roundtrip");
    std::filesystem::remove(path);
    rlf::storage::save_checkpoint(path, make_checkpoint());

    const rlf::storage::CheckpointData loaded =
        rlf::storage::load_checkpoint(path);
    RLF_CHECK(loaded.config.dimension == 4U);
    RLF_CHECK(loaded.master_seed == 123456ULL);
    RLF_CHECK(loaded.training_step == 12ULL);
    RLF_CHECK(loaded.modes.size() == 1U);
    RLF_CHECK(loaded.modes[0U].activation_count == 9ULL);
    RLF_CHECK(loaded.modes[0U].recent_corrections.size() == 1U);
    RLF_CHECK(loaded.associative_memory.size() == 1U);
    RLF_CHECK(loaded.structural_statistics.modes_split == 1ULL);
    RLF_CHECK(loaded.update_strategy == "winner_only");
    RLF_CHECK(
        loaded.experiment_metadata.at("experiment") ==
        "checkpoint_test"
    );

    const rlf::storage::CheckpointSummary summary =
        rlf::storage::inspect_checkpoint(path);
    RLF_CHECK(summary.format_version == 2U);
    RLF_CHECK(summary.mode_count == 1U);
    RLF_CHECK(summary.enabled_mode_count == 1U);
    RLF_CHECK(summary.associative_record_count == 1U);
    RLF_CHECK(summary.file_bytes > 0U);
    RLF_CHECK(summary.payload_checksum != 0ULL);
    std::filesystem::remove(path);
}

RLF_TEST_CASE("checkpoint rejects incompatible expected dimensions") {
    const std::filesystem::path path =
        checkpoint_path("checkpoint-dimension");
    rlf::storage::save_checkpoint(path, make_checkpoint());

    rlf::storage::CheckpointLoadOptions options;
    options.expected_dimension = 8U;
    RLF_CHECK_THROWS_AS(
        rlf::storage::load_checkpoint(path, options),
        std::runtime_error
    );
    std::filesystem::remove(path);
}

RLF_TEST_CASE("checkpoint rejects corruption truncation and versions") {
    const std::filesystem::path path =
        checkpoint_path("checkpoint-source");
    const std::filesystem::path corrupt =
        checkpoint_path("checkpoint-corrupt");
    const std::filesystem::path truncated =
        checkpoint_path("checkpoint-truncated");
    const std::filesystem::path version =
        checkpoint_path("checkpoint-version");
    rlf::storage::save_checkpoint(path, make_checkpoint());

    std::vector<char> bytes = read_bytes(path);
    std::vector<char> corrupted = bytes;
    corrupted.back() = static_cast<char>(corrupted.back() ^ 0x01);
    write_bytes(corrupt, corrupted);

    std::vector<char> shortened = bytes;
    shortened.resize(shortened.size() - 1U);
    write_bytes(truncated, shortened);

    std::vector<char> unsupported = bytes;
    unsupported[8U] = 3;
    unsupported[9U] = 0;
    unsupported[10U] = 0;
    unsupported[11U] = 0;
    write_bytes(version, unsupported);

    RLF_CHECK_THROWS_AS(
        rlf::storage::load_checkpoint(corrupt),
        std::runtime_error
    );
    RLF_CHECK_THROWS_AS(
        rlf::storage::load_checkpoint(truncated),
        std::runtime_error
    );
    RLF_CHECK_THROWS_AS(
        rlf::storage::load_checkpoint(version),
        std::runtime_error
    );

    std::filesystem::remove(path);
    std::filesystem::remove(corrupt);
    std::filesystem::remove(truncated);
    std::filesystem::remove(version);
}

RLF_TEST_CASE("checkpoint enforces configured file and count limits") {
    const std::filesystem::path path =
        checkpoint_path("checkpoint-limits");
    rlf::storage::save_checkpoint(path, make_checkpoint());

    rlf::storage::CheckpointLoadOptions file_options;
    file_options.maximum_file_bytes = 16U;
    RLF_CHECK_THROWS_AS(
        rlf::storage::load_checkpoint(path, file_options),
        std::runtime_error
    );

    rlf::storage::CheckpointLoadOptions mode_options;
    mode_options.maximum_modes = 0U;
    RLF_CHECK_THROWS_AS(
        rlf::storage::load_checkpoint(path, mode_options),
        std::runtime_error
    );
    std::filesystem::remove(path);
}
