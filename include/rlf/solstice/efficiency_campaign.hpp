#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>

namespace rlf::solstice {

struct PhysicalRuntimeInputs final {
    std::uint64_t compressed_dataset_bytes{};
    std::uint64_t decompressed_dataset_bytes{};
    std::uint64_t pcie_transfer_bytes{};
    std::uint64_t image_count{};
    std::uint64_t record_count{};
    std::uint64_t checkpoint_bytes{};
    double nvme_bytes_per_second{};
    double decompression_bytes_per_second{};
    double pcie_bytes_per_second{};
    double image_decodes_per_second{};
    double preprocessing_records_per_second{};
    double checkpoint_write_bytes_per_second{};
    double evaluation_seconds{};
    double baseline_seconds{1'800'000.0};
    double target_speedup{100'000.0};
    std::string evidence_kind{"planning_assumption"};
    std::string hardware_profile{"rtx-pro-6000-96g"};
};

struct PhysicalRuntimeFloor final {
    PhysicalRuntimeInputs inputs;
    double minimum_read_seconds{};
    double minimum_decompression_seconds{};
    double minimum_pcie_seconds{};
    double minimum_decode_seconds{};
    double minimum_preprocessing_seconds{};
    double minimum_checkpoint_seconds{};
    double minimum_evaluation_seconds{};
    double serial_floor_seconds{};
    double theoretical_best_case_seconds{};
    double target_runtime_seconds{};
    double maximum_possible_end_to_end_speedup{};
    bool target_physically_possible_under_inputs{};
    bool measured_hardware_evidence{};
};

struct FiftyMillionTrainingEvidence final {
    std::string hardware_profile;
    std::string device_name;
    std::string device_uuid;
    std::string backend;
    std::uint64_t device_count{};
    bool mig_disabled{};
    std::uint64_t total_records{};
    std::uint64_t instruction_records{};
    std::uint64_t preference_records{};
    std::uint64_t tool_records{};
    std::uint64_t fact_records{};
    std::uint64_t vision_records{};
    std::uint64_t episode_capacity_skips{};
    std::uint64_t context_capacity_skips{};
    std::uint64_t tool_keyword_capacity_skips{};
    double wall_seconds{};
    double gpu_active_seconds{};
    std::uint64_t peak_vram_bytes{};
    bool checkpoint_verified{};
    bool raw_gpu_trace_bound{};
    bool raw_energy_trace_bound{};
    double baseline_gpu_hours{};
    double candidate_gpu_hours{};
    bool matched_quality{};
    bool external_quality_gate_passed{};
    bool independent_reproduction_complete{};
};

struct FiftyMillionTrainingContract final {
    FiftyMillionTrainingEvidence evidence;
    double stated_hour_target_seconds{100.8};
    double strict_eighteen_second_target_seconds{18.0};
    double stated_hour_required_records_per_second{};
    double strict_required_records_per_second{};
    double measured_records_per_second{};
    double measured_gpu_hour_speedup{};
    bool target_wording_is_time_consistent{};
    bool exact_record_mix{};
    bool exact_rtx_profile{};
    bool physical_device_identity_present{};
    bool single_gpu_without_mig{};
    bool cuda_backend{};
    bool no_silent_capacity_saturation{};
    bool vram_within_profile_limit{};
    bool physical_evidence_complete{};
    bool stated_hour_target_passed{};
    bool strict_eighteen_second_target_passed{};
    bool general_100000x_gpu_hour_efficiency_proven{};
};

struct VisionGroundingAblationInputs final {
    std::size_t images{64U};
    std::size_t width{64U};
    std::size_t height{64U};
    std::size_t repetitions{3U};
};

struct VisionGroundingAblation final {
    VisionGroundingAblationInputs inputs;
    double separate_train_analyze_seconds{};
    double fused_train_analyze_seconds{};
    double speedup{};
    bool analyses_identical{};
    bool model_states_identical{};
    std::uint64_t reference_model_hash{};
    std::uint64_t fused_model_hash{};
    std::uint64_t reference_local_updates{};
    std::uint64_t fused_local_updates{};
};

struct SparseRouterUpdateAblationInputs final {
    std::size_t vector_count{8'192U};
    std::size_t dimensions{32U};
    std::size_t updates_per_batch{128U};
    std::size_t batches{8U};
    std::size_t queries_per_batch{32U};
    std::size_t repetitions{3U};
};

struct SparseRouterUpdateAblation final {
    SparseRouterUpdateAblationInputs inputs;
    double full_rebuild_seconds{};
    double incremental_update_seconds{};
    double speedup{};
    bool all_routes_identical{};
    std::uint64_t route_hash{};
    std::uint64_t reference_vectors_rebuilt{};
    std::uint64_t incremental_vectors_rebuilt{};
    std::uint64_t vectors_incrementally_updated{};
};

struct SparseRerankBatchAblationInputs final {
    std::size_t vector_count{8'192U};
    std::size_t query_count{1'024U};
    std::size_t dimensions{32U};
    std::size_t maximum_candidates{256U};
    std::size_t repetitions{5U};
};

struct SparseRerankBatchAblation final {
    SparseRerankBatchAblationInputs inputs;
    double per_query_seconds{};
    double batched_seconds{};
    double speedup{};
    bool results_identical{};
    std::uint64_t result_hash{};
    std::uint64_t per_query_backend_calls{};
    std::uint64_t batched_backend_calls{};
    std::uint64_t cosine_pairs{};
};

struct ConceptUpdateAblationInputs final {
    std::size_t images{32U};
    std::size_t width{64U};
    std::size_t height{64U};
    std::size_t concepts{128U};
    std::size_t repetitions{3U};
};

struct ConceptUpdateAblation final {
    ConceptUpdateAblationInputs inputs;
    double linear_seconds{};
    double indexed_seconds{};
    double speedup{};
    bool model_states_identical{};
    std::uint64_t model_hash{};
    std::uint64_t concept_update_lookups{};
    std::uint64_t linear_concept_comparisons{};
    std::uint64_t indexed_concept_lookups{};
};

struct ExampleDuplicateAblationInputs final {
    std::size_t examples{8'192U};
    std::size_t repetitions{3U};
};

struct ExampleDuplicateAblation final {
    ExampleDuplicateAblationInputs inputs;
    double linear_seconds{};
    double indexed_seconds{};
    double speedup{};
    bool model_states_identical{};
    std::uint64_t model_hash{};
    std::uint64_t example_duplicate_lookups{};
    std::uint64_t linear_example_comparisons{};
    std::uint64_t indexed_example_candidates{};
    std::uint64_t learned_examples{};
};

struct ModeIdIndexAblationInputs final {
    std::size_t modes{65'536U};
    std::size_t images{256U};
    std::size_t repetitions{3U};
};

struct ModeIdIndexAblation final {
    ModeIdIndexAblationInputs inputs;
    double rebuild_seconds{};
    double persistent_seconds{};
    double speedup{};
    bool model_states_identical{};
    bool analyses_identical{};
    std::uint64_t model_hash{};
    std::uint64_t mode_id_lookups{};
    std::uint64_t rebuilt_entries{};
    std::uint64_t persistent_rebuilt_entries{};
    std::uint64_t incremental_inserts{};
    std::uint64_t region_mode_id_lookups{};
    std::uint64_t linear_region_mode_comparisons{};
    std::uint64_t indexed_region_mode_lookups{};
};

struct GroundingIndexAblationInputs final {
    std::size_t initial_links{65'536U};
    std::size_t concepts_per_mode{8U};
    std::size_t observations{128U};
    std::size_t queries{128U};
    std::size_t repetitions{3U};
};

struct GroundingIndexAblation final {
    GroundingIndexAblationInputs inputs;
    double rebuild_seconds{};
    double persistent_seconds{};
    double speedup{};
    bool model_states_identical{};
    bool query_results_identical{};
    std::uint64_t model_hash{};
    std::uint64_t link_lookups{};
    std::uint64_t rebuilt_lookup_entries{};
    std::uint64_t indexed_link_candidates_examined{};
    std::uint64_t full_confidence_sweep_entries{};
    std::uint64_t sparse_confidence_recomputations{};
    std::uint64_t derived_sort_entries{};
    std::uint64_t query_full_scan_entries{};
    std::uint64_t query_indexed_candidates{};
};

struct LanguageOutcomeIndexAblationInputs final {
    std::size_t outcomes{65'536U};
    std::size_t updates{8'192U};
    std::size_t repetitions{3U};
};

struct LanguageOutcomeIndexAblation final {
    LanguageOutcomeIndexAblationInputs inputs;
    double linear_seconds{};
    double indexed_seconds{};
    double speedup{};
    bool model_states_identical{};
    bool predictions_identical{};
    std::uint64_t model_hash{};
    std::uint64_t outcome_update_lookups{};
    std::uint64_t linear_outcome_comparisons{};
    std::uint64_t indexed_outcome_lookups{};
};

struct PreferenceDuplicateIndexAblationInputs final {
    std::size_t preferences{65'536U};
    std::size_t updates{8'192U};
    std::size_t repetitions{3U};
};

struct PreferenceDuplicateIndexAblation final {
    PreferenceDuplicateIndexAblationInputs inputs;
    double linear_seconds{};
    double indexed_seconds{};
    double speedup{};
    bool model_states_identical{};
    bool scores_identical{};
    std::uint64_t model_hash{};
    std::uint64_t preference_duplicate_lookups{};
    std::uint64_t linear_preference_comparisons{};
    std::uint64_t indexed_preference_candidates{};
};

struct ActiveLearningDuplicateIndexAblationInputs final {
    std::size_t items{65'536U};
    std::size_t updates{8'192U};
    std::size_t repetitions{3U};
};

struct ActiveLearningDuplicateIndexAblation final {
    ActiveLearningDuplicateIndexAblationInputs inputs;
    double linear_seconds{};
    double indexed_seconds{};
    double speedup{};
    bool model_states_identical{};
    std::uint64_t model_hash{};
    std::uint64_t active_learning_duplicate_lookups{};
    std::uint64_t linear_active_learning_comparisons{};
    std::uint64_t indexed_active_learning_candidates{};
};

struct InstructionDuplicatePrefilterAblationInputs final {
    std::size_t demonstrations{16'384U};
    std::size_t updates{1'024U};
    std::size_t repetitions{3U};
};

struct InstructionDuplicatePrefilterAblation final {
    InstructionDuplicatePrefilterAblationInputs inputs;
    double retrieval_seconds{};
    double indexed_seconds{};
    double speedup{};
    bool model_states_identical{};
    std::uint64_t model_hash{};
    std::uint64_t reference_retrievals{};
    std::uint64_t indexed_retrievals{};
    std::uint64_t indexed_retrievals_avoided{};
};

struct ToolKeywordIndexAblationInputs final {
    std::size_t keywords{65'536U};
    std::size_t updates{8'192U};
    std::size_t repetitions{3U};
};

struct ToolKeywordIndexAblation final {
    ToolKeywordIndexAblationInputs inputs;
    double linear_seconds{};
    double indexed_seconds{};
    double speedup{};
    bool model_states_identical{};
    std::uint64_t model_hash{};
    std::uint64_t keyword_update_lookups{};
    std::uint64_t linear_keyword_comparisons{};
    std::uint64_t indexed_keyword_lookups{};
    std::uint64_t keyword_capacity_skips{};
};

struct DialogueEncodingAblationInputs final {
    std::size_t dialogues{4'096U};
    std::size_t repetitions{3U};
};

struct DialogueEncodingAblation final {
    DialogueEncodingAblationInputs inputs;
    double redundant_seconds{};
    double fused_seconds{};
    double speedup{};
    bool model_states_identical{};
    std::uint64_t model_hash{};
    std::uint64_t redundant_encode_calls{};
    std::uint64_t fused_encode_calls{};
    std::uint64_t encode_calls_avoided{};
};

[[nodiscard]] PhysicalRuntimeFloor calculate_physical_runtime_floor(
    const PhysicalRuntimeInputs& inputs
);

void write_physical_runtime_floor_json(
    std::ostream& output,
    const PhysicalRuntimeFloor& report
);

[[nodiscard]] FiftyMillionTrainingContract evaluate_fifty_million_contract(
    const FiftyMillionTrainingEvidence& evidence
);

void write_fifty_million_contract_json(
    std::ostream& output,
    const FiftyMillionTrainingContract& report
);

[[nodiscard]] VisionGroundingAblation measure_vision_grounding_ablation(
    const VisionGroundingAblationInputs& inputs
);

void write_vision_grounding_ablation_json(
    std::ostream& output,
    const VisionGroundingAblation& report
);

[[nodiscard]] SparseRouterUpdateAblation measure_sparse_router_update_ablation(
    const SparseRouterUpdateAblationInputs& inputs
);

void write_sparse_router_update_ablation_json(
    std::ostream& output,
    const SparseRouterUpdateAblation& report
);

[[nodiscard]] SparseRerankBatchAblation measure_sparse_rerank_batch_ablation(
    const SparseRerankBatchAblationInputs& inputs
);

void write_sparse_rerank_batch_ablation_json(
    std::ostream& output,
    const SparseRerankBatchAblation& report
);

[[nodiscard]] ConceptUpdateAblation measure_concept_update_ablation(
    const ConceptUpdateAblationInputs& inputs
);

void write_concept_update_ablation_json(
    std::ostream& output,
    const ConceptUpdateAblation& report
);

[[nodiscard]] ExampleDuplicateAblation measure_example_duplicate_ablation(
    const ExampleDuplicateAblationInputs& inputs
);

void write_example_duplicate_ablation_json(
    std::ostream& output,
    const ExampleDuplicateAblation& report
);

[[nodiscard]] ModeIdIndexAblation measure_mode_id_index_ablation(
    const ModeIdIndexAblationInputs& inputs
);

void write_mode_id_index_ablation_json(
    std::ostream& output,
    const ModeIdIndexAblation& report
);

[[nodiscard]] GroundingIndexAblation measure_grounding_index_ablation(
    const GroundingIndexAblationInputs& inputs
);

void write_grounding_index_ablation_json(
    std::ostream& output,
    const GroundingIndexAblation& report
);

[[nodiscard]] LanguageOutcomeIndexAblation
measure_language_outcome_index_ablation(
    const LanguageOutcomeIndexAblationInputs& inputs
);

void write_language_outcome_index_ablation_json(
    std::ostream& output,
    const LanguageOutcomeIndexAblation& report
);

[[nodiscard]] PreferenceDuplicateIndexAblation
measure_preference_duplicate_index_ablation(
    const PreferenceDuplicateIndexAblationInputs& inputs
);

void write_preference_duplicate_index_ablation_json(
    std::ostream& output,
    const PreferenceDuplicateIndexAblation& report
);

[[nodiscard]] ActiveLearningDuplicateIndexAblation
measure_active_learning_duplicate_index_ablation(
    const ActiveLearningDuplicateIndexAblationInputs& inputs
);

void write_active_learning_duplicate_index_ablation_json(
    std::ostream& output,
    const ActiveLearningDuplicateIndexAblation& report
);

[[nodiscard]] InstructionDuplicatePrefilterAblation
measure_instruction_duplicate_prefilter_ablation(
    const InstructionDuplicatePrefilterAblationInputs& inputs
);

void write_instruction_duplicate_prefilter_ablation_json(
    std::ostream& output,
    const InstructionDuplicatePrefilterAblation& report
);

[[nodiscard]] ToolKeywordIndexAblation measure_tool_keyword_index_ablation(
    const ToolKeywordIndexAblationInputs& inputs
);

void write_tool_keyword_index_ablation_json(
    std::ostream& output,
    const ToolKeywordIndexAblation& report
);

[[nodiscard]] DialogueEncodingAblation measure_dialogue_encoding_ablation(
    const DialogueEncodingAblationInputs& inputs
);

void write_dialogue_encoding_ablation_json(
    std::ostream& output,
    const DialogueEncodingAblation& report
);

}  // namespace rlf::solstice
