#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace rlf::experiments {

struct StructuralAdaptationConfig final {
    std::uint64_t seed{0x524C4632ULL};
    std::size_t dimension{256U};
    std::size_t training_examples{48U};
    std::size_t evaluation_examples{128U};
    double context_noise_radians{0.05};
};

struct StructuralEventResult final {
    std::string type;
    std::uint64_t step{};
    std::uint64_t primary_mode_id{};
    std::vector<std::uint64_t> related_mode_ids;
    std::string reason;
    double metric{};
};

struct StructuralAdaptationResult final {
    std::uint64_t seed{};
    std::size_t dimension{};
    std::size_t training_examples{};
    std::size_t evaluation_examples{};
    double initial_mean_similarity{};
    double final_mean_similarity{};
    double final_task_accuracy{};
    std::size_t total_modes{};
    std::size_t enabled_modes{};
    std::uint64_t modes_created{};
    std::uint64_t modes_split{};
    std::uint64_t modes_merged{};
    std::uint64_t modes_pruned{};
    std::vector<StructuralEventResult> events;
    std::uint64_t deterministic_run_hash{};
};

[[nodiscard]] StructuralAdaptationResult run_structural_adaptation(
    const StructuralAdaptationConfig& config
);
void write_structural_adaptation_json(
    std::ostream& output,
    const StructuralAdaptationResult& result
);

}  // namespace rlf::experiments
