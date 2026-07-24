#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>

namespace rlf::experiments {

struct PersistenceRoundtripConfig final {
    std::uint64_t seed{0x524C4633ULL};
    std::size_t dimension{256U};
    std::size_t mode_count{8U};
    std::size_t memory_records{32U};
    std::filesystem::path checkpoint_path{
        "results/milestone4_checkpoint.rlf"
    };
};

struct PersistenceRoundtripResult final {
    std::uint64_t seed{};
    std::size_t dimension{};
    std::size_t mode_count{};
    std::size_t memory_records{};
    std::size_t checkpoint_bytes{};
    double minimum_mode_key_similarity{};
    double minimum_mode_transformation_similarity{};
    double minimum_memory_key_similarity{};
    bool corruption_rejected{};
    bool truncation_rejected{};
    std::uint64_t payload_checksum{};
    std::uint64_t deterministic_run_hash{};
};

[[nodiscard]] PersistenceRoundtripResult run_persistence_roundtrip(
    const PersistenceRoundtripConfig& config
);
void write_persistence_roundtrip_json(
    std::ostream& output,
    const PersistenceRoundtripResult& result
);

}  // namespace rlf::experiments
