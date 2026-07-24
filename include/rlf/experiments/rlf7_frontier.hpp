#pragma once

#include "rlf/frontier/frontier_backend.hpp"
#include "rlf/frontier/frontier_trainer.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace rlf::experiments {

struct Rlf7FrontierConfig final {
    std::uint64_t seed{0x524C4638ULL};
    std::size_t knowledge_records{10'000U};
    std::size_t knowledge_queries{1'000U};
    std::size_t media_training_per_class{6U};
    std::size_t media_evaluation_per_class{4U};
    std::size_t agent_evaluation_episodes{12U};
    std::size_t threads{1U};
    bool include_audio{true};
    bool run_agent_gate{false};
    bool frontier_mode{true};
    frontier::FrontierBackendKind backend{frontier::FrontierBackendKind::optimized_cpu};
};

struct FrontierProjection final {
    std::size_t gpu_memory_gb{};
    std::uint64_t projected_hot_modes{};
    std::uint64_t projected_hot_bytes{};
    std::uint64_t cpu_ram_bytes{};
    std::uint64_t nvme_bytes{};
};

struct Rlf7FrontierResult final {
    Rlf7FrontierConfig config;
    std::size_t knowledge_records{};
    double knowledge_retrieval_accuracy{};
    double candidates_per_query{};
    double image_accuracy{};
    double image_localization_iou{};
    double video_accuracy{};
    double video_track_continuity{};
    double audio_accuracy{};
    double cross_modal_binding_accuracy{};
    bool agent_gate_executed{};
    double agent_success_rate{};
    double agent_recovery_rate{};
    std::size_t accepted_skills{};
    bool scalar_optimized_agreement{};
    bool cuda_compiled{};
    bool cuda_available{};
    bool checkpoint_ready{true};
    bool prototype_generation_ready{true};
    std::uint64_t persistent_bytes{};
    std::uint64_t deterministic_hash{};
    std::vector<FrontierProjection> projections;
    std::string scientific_decision;
    bool frontier_claim_justified{};
    std::vector<std::string> limitations;
};

[[nodiscard]] Rlf7FrontierResult run_rlf7_frontier(
    const Rlf7FrontierConfig& config,
    frontier::FrontierModel* trained_model = nullptr
);
void write_rlf7_frontier_json(
    std::ostream& output,
    const Rlf7FrontierResult& result
);

}  // namespace rlf::experiments
