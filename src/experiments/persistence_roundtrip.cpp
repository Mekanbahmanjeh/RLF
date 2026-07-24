#include "rlf/experiments/persistence_roundtrip.hpp"

#include "rlf/core/deterministic_rng.hpp"
#include "rlf/core/phase_vector.hpp"
#include "rlf/core/resonant_mode.hpp"
#include "rlf/experiments/phase_vector_smoke.hpp"
#include "rlf/memory/associative_memory.hpp"
#include "rlf/storage/checkpoint.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <ostream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace rlf::experiments {
namespace {

constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;

void hash_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (unsigned int byte_index = 0U; byte_index < 8U; ++byte_index) {
        hash ^= (value >> (byte_index * 8U)) & 0xFFULL;
        hash *= fnv_prime;
    }
}

void hash_phase_vector(
    std::uint64_t& hash,
    const core::PhaseVector& value
) noexcept {
    for (const float angle : value.angles()) {
        hash_u64(
            hash,
            static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(angle))
        );
    }
}

[[nodiscard]] std::vector<char> read_file_bytes(
    const std::filesystem::path& path
) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "unable to read generated checkpoint"
        );
    }
    return {
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()
    };
}

void write_file_bytes(
    const std::filesystem::path& path,
    const std::vector<char>& bytes
) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "unable to write checkpoint mutation"
        );
    }
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!output) {
        throw std::runtime_error(
            "failed to write checkpoint mutation"
        );
    }
}

[[nodiscard]] bool load_is_rejected(
    const std::filesystem::path& path
) {
    try {
        static_cast<void>(storage::load_checkpoint(path));
        return false;
    } catch (const std::runtime_error&) {
        return true;
    }
}

}  // namespace

PersistenceRoundtripResult run_persistence_roundtrip(
    const PersistenceRoundtripConfig& config
) {
    if (config.dimension == 0U ||
        config.mode_count == 0U ||
        config.memory_records == 0U ||
        config.checkpoint_path.empty()) {
        throw std::invalid_argument(
            "invalid persistence-roundtrip configuration"
        );
    }

    core::DeterministicRng random_number_generator(config.seed);
    core::FabricConfig fabric_config;
    fabric_config.dimension = config.dimension;
    fabric_config.maximum_modes = std::max(
        config.mode_count,
        std::size_t{64U}
    );
    fabric_config.settling.candidate_count = std::min(
        config.mode_count,
        std::size_t{16U}
    );
    fabric_config.settling.active_count = std::min(
        fabric_config.settling.candidate_count,
        std::size_t{8U}
    );
    fabric_config.structural_learning.enabled = true;

    std::vector<core::ResonantMode> modes;
    modes.reserve(config.mode_count);
    for (std::size_t mode_index = 0U;
         mode_index < config.mode_count;
         ++mode_index) {
        core::ResonantMode mode(
            static_cast<std::uint64_t>(mode_index + 1U),
            core::PhaseVector::random(
                config.dimension,
                random_number_generator
            ),
            core::PhaseVector::random(
                config.dimension,
                random_number_generator
            ),
            0.5F + static_cast<float>(mode_index % 3U) * 0.25F,
            0.25F + static_cast<float>(mode_index % 4U) * 0.15F,
            static_cast<float>(
                static_cast<double>(mode_index % 5U) * 0.1 - 0.2
            ),
            static_cast<std::uint64_t>(mode_index)
        );
        mode.activation_count =
            static_cast<std::uint64_t>(mode_index * 3U);
        mode.successful_update_count =
            static_cast<std::uint64_t>(mode_index * 2U);
        mode.unsuccessful_update_count =
            static_cast<std::uint64_t>(mode_index);
        mode.last_used_step =
            static_cast<std::uint64_t>(mode_index + 10U);
        mode.recent_corrections.push_back({
            .context = mode.context_key,
            .desired_transformation = mode.transformation,
            .proposal_quality = 0.75,
            .improved_prediction = true,
            .step = static_cast<std::uint64_t>(mode_index + 1U),
        });
        modes.push_back(std::move(mode));
    }

    memory::AssociativeMemory associative_memory(
        config.dimension,
        config.memory_records + 8U
    );
    for (std::size_t record_index = 0U;
         record_index < config.memory_records;
         ++record_index) {
        core::PhaseVector key = core::PhaseVector::random(
            config.dimension,
            random_number_generator
        );
        memory::AssociativeValue value =
            (record_index % 2U) == 0U
            ? memory::AssociativeValue(
                  core::PhaseVector::random(
                      config.dimension,
                      random_number_generator
                  )
              )
            : memory::AssociativeValue(memory::BytePayload{
                  static_cast<std::uint8_t>(record_index & 0xFFU),
                  static_cast<std::uint8_t>(
                      (record_index * 3U) & 0xFFU
                  ),
                  static_cast<std::uint8_t>(
                      (record_index * 7U) & 0xFFU
                  ),
              });
        static_cast<void>(associative_memory.insert(
            std::move(key),
            std::move(value),
            0.5F + static_cast<float>(record_index % 5U) * 0.1F,
            static_cast<std::uint64_t>(record_index + 1U)
        ));
    }

    const storage::CheckpointData checkpoint{
        .config = fabric_config,
        .master_seed = config.seed,
        .training_step = 777ULL,
        .modes = modes,
        .associative_memory = std::move(associative_memory),
        .structural_statistics = {
            .modes_created =
                static_cast<std::uint64_t>(config.mode_count),
            .modes_split = 1ULL,
            .modes_merged = 2ULL,
            .modes_pruned = 3ULL,
        },
        .update_strategy = "normalized_responsibility",
        .experiment_metadata = {
            {"experiment", "persistence_roundtrip"},
            {"status", "observed"},
        },
    };
    storage::save_checkpoint(config.checkpoint_path, checkpoint);
    const storage::CheckpointData loaded =
        storage::load_checkpoint(config.checkpoint_path, {
            .maximum_file_bytes = 1U << 30U,
            .maximum_dimension =
                core::PhaseVector::default_max_serialized_dimension,
            .maximum_modes = 1'000'000U,
            .maximum_corrections_per_mode = 4'096U,
            .maximum_memory_capacity = 1'000'000U,
            .maximum_memory_records = 1'000'000U,
            .maximum_payload_bytes = 1U << 28U,
            .maximum_metadata_entries = 4'096U,
            .maximum_string_bytes = 1U << 20U,
            .expected_dimension = config.dimension,
        });
    const storage::CheckpointSummary summary =
        storage::inspect_checkpoint(config.checkpoint_path);

    double minimum_mode_key_similarity = 1.0;
    double minimum_mode_transformation_similarity = 1.0;
    for (std::size_t mode_index = 0U;
         mode_index < modes.size();
         ++mode_index) {
        minimum_mode_key_similarity = std::min(
            minimum_mode_key_similarity,
            modes[mode_index].context_key.similarity(
                loaded.modes[mode_index].context_key
            )
        );
        minimum_mode_transformation_similarity = std::min(
            minimum_mode_transformation_similarity,
            modes[mode_index].transformation.similarity(
                loaded.modes[mode_index].transformation
            )
        );
    }

    double minimum_memory_key_similarity = 1.0;
    for (std::size_t record_index = 0U;
         record_index < checkpoint.associative_memory.size();
         ++record_index) {
        minimum_memory_key_similarity = std::min(
            minimum_memory_key_similarity,
            checkpoint.associative_memory.records()[record_index]
                .key.similarity(
                    loaded.associative_memory.records()[record_index]
                        .key
                )
        );
    }

    const std::vector<char> original_bytes = read_file_bytes(
        config.checkpoint_path
    );
    std::filesystem::path corrupt_path = config.checkpoint_path;
    corrupt_path += ".corrupt";
    std::filesystem::path truncated_path = config.checkpoint_path;
    truncated_path += ".truncated";
    std::vector<char> corrupt_bytes = original_bytes;
    corrupt_bytes.back() = static_cast<char>(
        corrupt_bytes.back() ^ 0x01
    );
    write_file_bytes(corrupt_path, corrupt_bytes);
    std::vector<char> truncated_bytes = original_bytes;
    truncated_bytes.resize(truncated_bytes.size() - 1U);
    write_file_bytes(truncated_path, truncated_bytes);
    const bool corruption_rejected = load_is_rejected(corrupt_path);
    const bool truncation_rejected = load_is_rejected(truncated_path);
    std::filesystem::remove(corrupt_path);
    std::filesystem::remove(truncated_path);

    std::uint64_t run_hash = fnv_offset_basis;
    hash_u64(run_hash, config.seed);
    hash_u64(run_hash, summary.payload_checksum);
    hash_u64(
        run_hash,
        static_cast<std::uint64_t>(summary.file_bytes)
    );
    for (const core::ResonantMode& mode : loaded.modes) {
        hash_u64(run_hash, mode.id);
        hash_phase_vector(run_hash, mode.context_key);
        hash_phase_vector(run_hash, mode.transformation);
    }
    for (const memory::AssociativeRecord& record :
         loaded.associative_memory.records()) {
        hash_u64(run_hash, record.id);
        hash_phase_vector(run_hash, record.key);
    }
    hash_u64(run_hash, corruption_rejected ? 1ULL : 0ULL);
    hash_u64(run_hash, truncation_rejected ? 1ULL : 0ULL);

    return {
        .seed = config.seed,
        .dimension = config.dimension,
        .mode_count = loaded.modes.size(),
        .memory_records = loaded.associative_memory.size(),
        .checkpoint_bytes = summary.file_bytes,
        .minimum_mode_key_similarity =
            minimum_mode_key_similarity,
        .minimum_mode_transformation_similarity =
            minimum_mode_transformation_similarity,
        .minimum_memory_key_similarity =
            minimum_memory_key_similarity,
        .corruption_rejected = corruption_rejected,
        .truncation_rejected = truncation_rejected,
        .payload_checksum = summary.payload_checksum,
        .deterministic_run_hash = run_hash,
    };
}

void write_persistence_roundtrip_json(
    std::ostream& output,
    const PersistenceRoundtripResult& result
) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "{\n"
           << "  \"experiment\": \"persistence_roundtrip\",\n"
           << "  \"status\": \"observed\",\n"
           << "  \"seed\": " << result.seed << ",\n"
           << "  \"dimension\": " << result.dimension << ",\n"
           << "  \"mode_count\": " << result.mode_count << ",\n"
           << "  \"memory_records\": "
           << result.memory_records << ",\n"
           << "  \"checkpoint_bytes\": "
           << result.checkpoint_bytes << ",\n"
           << "  \"minimum_mode_key_similarity\": "
           << result.minimum_mode_key_similarity << ",\n"
           << "  \"minimum_mode_transformation_similarity\": "
           << result.minimum_mode_transformation_similarity << ",\n"
           << "  \"minimum_memory_key_similarity\": "
           << result.minimum_memory_key_similarity << ",\n"
           << "  \"corruption_rejected\": "
           << (result.corruption_rejected ? "true" : "false") << ",\n"
           << "  \"truncation_rejected\": "
           << (result.truncation_rejected ? "true" : "false") << ",\n"
           << "  \"payload_checksum\": \""
           << format_run_hash(result.payload_checksum) << "\",\n"
           << "  \"deterministic_run_hash\": \""
           << format_run_hash(result.deterministic_run_hash) << "\"\n"
           << "}\n";
    if (!output) {
        throw std::runtime_error(
            "failed to write persistence-roundtrip result"
        );
    }
}

}  // namespace rlf::experiments
