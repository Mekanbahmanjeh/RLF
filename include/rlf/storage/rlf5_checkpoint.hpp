#pragma once

#include "rlf/core/language_fabric.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace rlf::storage {

constexpr std::uint32_t rlf5_checkpoint_format_version = 7U;

struct Rlf5CheckpointLoadOptions final {
    std::size_t maximum_file_bytes{1U << 30U};
    std::size_t maximum_dimension{1'048'576U};
    std::size_t maximum_lexemes{10'000'000U};
    std::size_t maximum_lexeme_bytes{1U << 20U};
    std::size_t maximum_merges{10'000'000U};
    std::size_t maximum_contexts{20'000'000U};
    std::size_t maximum_outcomes{100'000'000U};
    std::size_t maximum_concepts{10'000'000U};
    std::size_t maximum_surfaces{100'000'000U};
    std::size_t maximum_constructions{10'000'000U};
    std::size_t maximum_pattern_items{4'096U};
    std::size_t maximum_string_bytes{1U << 20U};
};

struct Rlf5CheckpointSummary final {
    std::uint32_t format_version{};
    std::uint64_t seed{};
    std::size_t phase_dimension{};
    std::size_t lexeme_count{};
    std::size_t merge_count{};
    std::size_t context_count{};
    std::size_t outcome_count{};
    std::size_t concept_count{};
    std::size_t construction_count{};
    std::uint64_t payload_checksum{};
    std::size_t file_bytes{};
};

void save_rlf5_checkpoint(
    const std::filesystem::path& path,
    const core::LanguageFabric& fabric
);
[[nodiscard]] core::LanguageFabric load_rlf5_checkpoint(
    const std::filesystem::path& path,
    const Rlf5CheckpointLoadOptions& options = {}
);
[[nodiscard]] Rlf5CheckpointSummary inspect_rlf5_checkpoint(
    const std::filesystem::path& path,
    const Rlf5CheckpointLoadOptions& options = {}
);

}  // namespace rlf::storage
