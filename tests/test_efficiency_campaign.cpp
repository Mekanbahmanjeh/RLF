#include "rlf/solstice/efficiency_campaign.hpp"
#include "rlf/solstice/efficiency_corpus.hpp"
#include "rlf/core/sha256.hpp"
#include "test_framework.hpp"

#include <sstream>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <vector>

RLF_TEST_CASE("Physical runtime floor uses overlapped ingress and fail-closed evidence") {
    rlf::solstice::PhysicalRuntimeInputs inputs;
    inputs.compressed_dataset_bytes = 1'000U;
    inputs.decompressed_dataset_bytes = 2'000U;
    inputs.pcie_transfer_bytes = 500U;
    inputs.image_count = 100U;
    inputs.record_count = 500U;
    inputs.checkpoint_bytes = 400U;
    inputs.nvme_bytes_per_second = 100.0;
    inputs.decompression_bytes_per_second = 100.0;
    inputs.pcie_bytes_per_second = 100.0;
    inputs.image_decodes_per_second = 20.0;
    inputs.preprocessing_records_per_second = 100.0;
    inputs.checkpoint_write_bytes_per_second = 100.0;
    inputs.evaluation_seconds = 6.0;
    inputs.baseline_seconds = 1'000.0;
    inputs.target_speedup = 100.0;
    const auto report = rlf::solstice::calculate_physical_runtime_floor(inputs);
    RLF_CHECK(report.minimum_read_seconds == 10.0);
    RLF_CHECK(report.minimum_decompression_seconds == 20.0);
    RLF_CHECK(report.serial_floor_seconds == 55.0);
    RLF_CHECK(report.theoretical_best_case_seconds == 30.0);
    RLF_CHECK(report.maximum_possible_end_to_end_speedup < 34.0);
    RLF_CHECK(!report.target_physically_possible_under_inputs);
    RLF_CHECK(!report.measured_hardware_evidence);
    std::ostringstream json;
    rlf::solstice::write_physical_runtime_floor_json(json, report);
    RLF_CHECK(json.str().find("\"measured_hardware_evidence\": false") != std::string::npos);
}

RLF_TEST_CASE("Physical runtime floor rejects missing throughput") {
    rlf::solstice::PhysicalRuntimeInputs inputs;
    inputs.compressed_dataset_bytes = 1U;
    bool rejected = false;
    try {
        static_cast<void>(rlf::solstice::calculate_physical_runtime_floor(inputs));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    RLF_CHECK(rejected);
}

RLF_TEST_CASE("Physical runtime floor does not trust a measured label") {
    rlf::solstice::PhysicalRuntimeInputs inputs;
    inputs.evidence_kind = "measured";
    const auto report = rlf::solstice::calculate_physical_runtime_floor(inputs);
    RLF_CHECK(!report.measured_hardware_evidence);
}

RLF_TEST_CASE("Fifty million contract separates 0.028 hours from 18 seconds") {
    rlf::solstice::FiftyMillionTrainingEvidence evidence;
    evidence.hardware_profile = "general-rtx-pro-6000-96g";
    evidence.device_name = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition";
    evidence.device_uuid = "GPU-test-uuid";
    evidence.backend = "cuda";
    evidence.device_count = 1U;
    evidence.mig_disabled = true;
    evidence.total_records = 50'000'000U;
    evidence.instruction_records = 24'500'001U;
    evidence.preference_records = 8'166'666U;
    evidence.tool_records = 8'166'667U;
    evidence.fact_records = 8'166'666U;
    evidence.vision_records = 1'000'000U;
    evidence.wall_seconds = 100.0;
    evidence.gpu_active_seconds = 90.0;
    evidence.peak_vram_bytes = 89ULL * 1024ULL * 1024ULL * 1024ULL;
    evidence.checkpoint_verified = true;
    evidence.raw_gpu_trace_bound = true;
    evidence.raw_energy_trace_bound = true;

    const auto report =
        rlf::solstice::evaluate_fifty_million_contract(evidence);
    RLF_CHECK(!report.target_wording_is_time_consistent);
    RLF_CHECK(report.exact_record_mix);
    RLF_CHECK(report.physical_evidence_complete);
    RLF_CHECK(report.stated_hour_target_passed);
    RLF_CHECK(!report.strict_eighteen_second_target_passed);
    RLF_CHECK(report.stated_hour_required_records_per_second > 496'000.0);
    RLF_CHECK(report.strict_required_records_per_second > 2'777'000.0);
    RLF_CHECK(!report.general_100000x_gpu_hour_efficiency_proven);
}

RLF_TEST_CASE("Fifty million contract rejects silent saturation and gates 100000x") {
    rlf::solstice::FiftyMillionTrainingEvidence evidence;
    evidence.hardware_profile = "rtx-pro-6000-96g";
    evidence.device_name = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition";
    evidence.device_uuid = "GPU-test-uuid";
    evidence.backend = "cuda";
    evidence.device_count = 1U;
    evidence.mig_disabled = true;
    evidence.total_records = 50'000'000U;
    evidence.instruction_records = 24'500'001U;
    evidence.preference_records = 8'166'666U;
    evidence.tool_records = 8'166'667U;
    evidence.fact_records = 8'166'666U;
    evidence.vision_records = 1'000'000U;
    evidence.wall_seconds = 18.0;
    evidence.gpu_active_seconds = 18.0;
    evidence.peak_vram_bytes = 87ULL * 1024ULL * 1024ULL * 1024ULL;
    evidence.checkpoint_verified = true;
    evidence.raw_gpu_trace_bound = true;
    evidence.raw_energy_trace_bound = true;
    evidence.baseline_gpu_hours = 500.0;
    evidence.candidate_gpu_hours = 0.005;
    evidence.matched_quality = true;
    evidence.external_quality_gate_passed = true;
    evidence.independent_reproduction_complete = true;

    const auto complete =
        rlf::solstice::evaluate_fifty_million_contract(evidence);
    RLF_CHECK(complete.strict_eighteen_second_target_passed);
    RLF_CHECK(complete.measured_gpu_hour_speedup >= 100'000.0);
    RLF_CHECK(complete.general_100000x_gpu_hour_efficiency_proven);

    evidence.context_capacity_skips = 1U;
    const auto saturated =
        rlf::solstice::evaluate_fifty_million_contract(evidence);
    RLF_CHECK(!saturated.no_silent_capacity_saturation);
    RLF_CHECK(!saturated.physical_evidence_complete);
    RLF_CHECK(!saturated.strict_eighteen_second_target_passed);
    RLF_CHECK(!saturated.general_100000x_gpu_hour_efficiency_proven);
    std::ostringstream json;
    rlf::solstice::write_fifty_million_contract_json(json, saturated);
    RLF_CHECK(json.str().find(
        "\"general_100000x_gpu_hour_efficiency_proven\": false"
    ) != std::string::npos);
}

RLF_TEST_CASE("Fused vision grounding ablation preserves model and analysis") {
    rlf::solstice::VisionGroundingAblationInputs inputs;
    inputs.images = 3U;
    inputs.width = 32U;
    inputs.height = 32U;
    inputs.repetitions = 1U;
    const auto report = rlf::solstice::measure_vision_grounding_ablation(inputs);
    RLF_CHECK(report.analyses_identical);
    RLF_CHECK(report.model_states_identical);
    RLF_CHECK(report.reference_model_hash == report.fused_model_hash);
    RLF_CHECK(report.reference_local_updates == report.fused_local_updates);
    RLF_CHECK(report.separate_train_analyze_seconds > 0.0);
    RLF_CHECK(report.fused_train_analyze_seconds > 0.0);
    std::ostringstream json;
    rlf::solstice::write_vision_grounding_ablation_json(json, report);
    RLF_CHECK(json.str().find("\"gpu_efficiency_evidence_eligible\": false") !=
              std::string::npos);
}

RLF_TEST_CASE("Sparse router update ablation preserves every routed candidate") {
    rlf::solstice::SparseRouterUpdateAblationInputs inputs;
    inputs.vector_count = 512U;
    inputs.dimensions = 16U;
    inputs.updates_per_batch = 16U;
    inputs.batches = 3U;
    inputs.queries_per_batch = 8U;
    inputs.repetitions = 1U;
    const auto report =
        rlf::solstice::measure_sparse_router_update_ablation(inputs);
    RLF_CHECK(report.all_routes_identical);
    RLF_CHECK(report.full_rebuild_seconds > 0.0);
    RLF_CHECK(report.incremental_update_seconds > 0.0);
    RLF_CHECK(report.reference_vectors_rebuilt == 2'048U);
    RLF_CHECK(report.incremental_vectors_rebuilt == 512U);
    RLF_CHECK(report.vectors_incrementally_updated == 48U);
    std::ostringstream json;
    rlf::solstice::write_sparse_router_update_ablation_json(json, report);
    RLF_CHECK(json.str().find("\"all_routes_identical\": true") !=
              std::string::npos);
    RLF_CHECK(json.str().find("\"gpu_efficiency_evidence_eligible\": false") !=
              std::string::npos);
}

RLF_TEST_CASE("Sparse rerank batching preserves every selected mode and score") {
    rlf::solstice::SparseRerankBatchAblationInputs inputs;
    inputs.vector_count = 512U;
    inputs.query_count = 32U;
    inputs.dimensions = 16U;
    inputs.maximum_candidates = 64U;
    inputs.repetitions = 1U;
    const auto report =
        rlf::solstice::measure_sparse_rerank_batch_ablation(inputs);
    RLF_CHECK(report.results_identical);
    RLF_CHECK(report.per_query_seconds > 0.0);
    RLF_CHECK(report.batched_seconds > 0.0);
    RLF_CHECK(report.per_query_backend_calls == inputs.query_count);
    RLF_CHECK(report.batched_backend_calls == 1U);
    RLF_CHECK(report.cosine_pairs > 0U);
    std::ostringstream json;
    rlf::solstice::write_sparse_rerank_batch_ablation_json(json, report);
    RLF_CHECK(json.str().find("\"results_identical\": true") !=
              std::string::npos);
    RLF_CHECK(json.str().find("\"gpu_efficiency_evidence_eligible\": false") !=
              std::string::npos);
}

RLF_TEST_CASE("Indexed concept update ablation preserves learned visual state") {
    rlf::solstice::ConceptUpdateAblationInputs inputs;
    inputs.images = 2U;
    inputs.width = 32U;
    inputs.height = 32U;
    inputs.concepts = 16U;
    inputs.repetitions = 1U;
    const auto report = rlf::solstice::measure_concept_update_ablation(inputs);
    RLF_CHECK(report.model_states_identical);
    RLF_CHECK(report.linear_seconds > 0.0);
    RLF_CHECK(report.indexed_seconds > 0.0);
    RLF_CHECK(report.concept_update_lookups == report.indexed_concept_lookups);
    RLF_CHECK(report.linear_concept_comparisons > report.concept_update_lookups);
    std::ostringstream json;
    rlf::solstice::write_concept_update_ablation_json(json, report);
    RLF_CHECK(json.str().find("\"model_states_identical\": true") !=
              std::string::npos);
    RLF_CHECK(json.str().find("\"gpu_efficiency_evidence_eligible\": false") !=
              std::string::npos);
}

RLF_TEST_CASE("Indexed example duplicate ablation preserves learned visual state") {
    rlf::solstice::ExampleDuplicateAblationInputs inputs;
    inputs.examples = 32U;
    inputs.repetitions = 1U;
    const auto report =
        rlf::solstice::measure_example_duplicate_ablation(inputs);
    RLF_CHECK(report.model_states_identical);
    RLF_CHECK(report.linear_seconds > 0.0);
    RLF_CHECK(report.indexed_seconds > 0.0);
    RLF_CHECK(report.learned_examples == inputs.examples);
    RLF_CHECK(report.example_duplicate_lookups == inputs.examples + 1U);
    RLF_CHECK(report.indexed_example_candidates == 1U);
    RLF_CHECK(report.linear_example_comparisons >
              report.indexed_example_candidates);
    std::ostringstream json;
    rlf::solstice::write_example_duplicate_ablation_json(json, report);
    RLF_CHECK(json.str().find("\"model_states_identical\": true") !=
              std::string::npos);
    RLF_CHECK(json.str().find("\"gpu_efficiency_evidence_eligible\": false") !=
              std::string::npos);
}

RLF_TEST_CASE("Persistent mode ID index ablation preserves learned visual state") {
    rlf::solstice::ModeIdIndexAblationInputs inputs;
    inputs.modes = 128U;
    inputs.images = 4U;
    inputs.repetitions = 1U;
    const auto report = rlf::solstice::measure_mode_id_index_ablation(inputs);
    RLF_CHECK(report.model_states_identical);
    RLF_CHECK(report.analyses_identical);
    RLF_CHECK(report.rebuild_seconds > 0.0);
    RLF_CHECK(report.persistent_seconds > 0.0);
    RLF_CHECK(report.mode_id_lookups == inputs.images);
    RLF_CHECK(report.rebuilt_entries == inputs.modes * inputs.images);
    RLF_CHECK(report.persistent_rebuilt_entries == 0U);
    RLF_CHECK(report.incremental_inserts == 0U);
    RLF_CHECK(report.region_mode_id_lookups == inputs.images);
    RLF_CHECK(report.linear_region_mode_comparisons > 0U);
    RLF_CHECK(report.indexed_region_mode_lookups == inputs.images);
    std::ostringstream json;
    rlf::solstice::write_mode_id_index_ablation_json(json, report);
    RLF_CHECK(json.str().find("\"model_states_identical\": true") !=
              std::string::npos);
    RLF_CHECK(json.str().find("\"analyses_identical\": true") !=
              std::string::npos);
    RLF_CHECK(json.str().find("\"gpu_efficiency_evidence_eligible\": false") !=
              std::string::npos);
}

RLF_TEST_CASE("Persistent grounding index ablation preserves learning and recall") {
    rlf::solstice::GroundingIndexAblationInputs inputs;
    inputs.initial_links = 128U;
    inputs.concepts_per_mode = 8U;
    inputs.observations = 4U;
    inputs.queries = 4U;
    inputs.repetitions = 1U;
    const auto report = rlf::solstice::measure_grounding_index_ablation(inputs);
    RLF_CHECK(report.model_states_identical);
    RLF_CHECK(report.query_results_identical);
    RLF_CHECK(report.rebuild_seconds > 0.0);
    RLF_CHECK(report.persistent_seconds > 0.0);
    RLF_CHECK(report.link_lookups == inputs.observations * 3U);
    RLF_CHECK(report.rebuilt_lookup_entries ==
              inputs.initial_links * inputs.observations);
    RLF_CHECK(report.full_confidence_sweep_entries ==
              inputs.initial_links * inputs.observations);
    RLF_CHECK(report.sparse_confidence_recomputations ==
              inputs.observations * 3U);
    RLF_CHECK(report.derived_sort_entries ==
              inputs.initial_links * inputs.observations * 2U);
    RLF_CHECK(report.query_full_scan_entries >
              report.query_indexed_candidates);
    std::ostringstream json;
    rlf::solstice::write_grounding_index_ablation_json(json, report);
    RLF_CHECK(json.str().find("\"model_states_identical\": true") !=
              std::string::npos);
    RLF_CHECK(json.str().find("\"query_results_identical\": true") !=
              std::string::npos);
    RLF_CHECK(json.str().find("\"gpu_efficiency_evidence_eligible\": false") !=
              std::string::npos);
}

RLF_TEST_CASE("Indexed language outcome ablation preserves learning and prediction") {
    rlf::solstice::LanguageOutcomeIndexAblationInputs inputs;
    inputs.outcomes = 64U;
    inputs.updates = 32U;
    inputs.repetitions = 1U;
    const auto report =
        rlf::solstice::measure_language_outcome_index_ablation(inputs);
    RLF_CHECK(report.model_states_identical);
    RLF_CHECK(report.predictions_identical);
    RLF_CHECK(report.linear_seconds > 0.0);
    RLF_CHECK(report.indexed_seconds > 0.0);
    RLF_CHECK(report.outcome_update_lookups == inputs.updates);
    RLF_CHECK(report.indexed_outcome_lookups == inputs.updates);
    RLF_CHECK(report.linear_outcome_comparisons >
              report.indexed_outcome_lookups);
    std::ostringstream json;
    rlf::solstice::write_language_outcome_index_ablation_json(json, report);
    RLF_CHECK(json.str().find("\"model_states_identical\": true") !=
              std::string::npos);
    RLF_CHECK(json.str().find("\"predictions_identical\": true") !=
              std::string::npos);
    RLF_CHECK(json.str().find("\"gpu_efficiency_evidence_eligible\": false") !=
              std::string::npos);
}

RLF_TEST_CASE("Indexed preference duplicate ablation preserves general learning") {
    rlf::solstice::PreferenceDuplicateIndexAblationInputs inputs;
    inputs.preferences = 64U;
    inputs.updates = 32U;
    inputs.repetitions = 1U;
    const auto report =
        rlf::solstice::measure_preference_duplicate_index_ablation(inputs);
    RLF_CHECK(report.model_states_identical);
    RLF_CHECK(report.scores_identical);
    RLF_CHECK(report.linear_seconds > 0.0);
    RLF_CHECK(report.indexed_seconds > 0.0);
    RLF_CHECK(report.preference_duplicate_lookups == inputs.updates);
    RLF_CHECK(report.indexed_preference_candidates == inputs.updates);
    RLF_CHECK(report.linear_preference_comparisons >
              report.indexed_preference_candidates);
    std::ostringstream json;
    rlf::solstice::write_preference_duplicate_index_ablation_json(json, report);
    RLF_CHECK(json.str().find("\"model_states_identical\": true") !=
              std::string::npos);
    RLF_CHECK(json.str().find("\"scores_identical\": true") !=
              std::string::npos);
    RLF_CHECK(json.str().find("\"gpu_efficiency_evidence_eligible\": false") !=
              std::string::npos);
}

RLF_TEST_CASE("Indexed active learning duplicate ablation preserves admission") {
    rlf::solstice::ActiveLearningDuplicateIndexAblationInputs inputs;
    inputs.items = 64U;
    inputs.updates = 32U;
    inputs.repetitions = 1U;
    const auto report = rlf::solstice::
        measure_active_learning_duplicate_index_ablation(inputs);
    RLF_CHECK(report.model_states_identical);
    RLF_CHECK(report.linear_seconds > 0.0);
    RLF_CHECK(report.indexed_seconds > 0.0);
    RLF_CHECK(report.active_learning_duplicate_lookups == inputs.updates);
    RLF_CHECK(report.indexed_active_learning_candidates == inputs.updates);
    RLF_CHECK(report.linear_active_learning_comparisons >
              report.indexed_active_learning_candidates);
    std::ostringstream json;
    rlf::solstice::write_active_learning_duplicate_index_ablation_json(
        json, report
    );
    RLF_CHECK(json.str().find("\"model_states_identical\": true") !=
              std::string::npos);
    RLF_CHECK(json.str().find("\"gpu_efficiency_evidence_eligible\": false") !=
              std::string::npos);
}

RLF_TEST_CASE("Instruction duplicate prefilter ablation preserves exact learning") {
    rlf::solstice::InstructionDuplicatePrefilterAblationInputs inputs;
    inputs.demonstrations = 64U;
    inputs.updates = 8U;
    inputs.repetitions = 1U;
    const auto report = rlf::solstice::
        measure_instruction_duplicate_prefilter_ablation(inputs);
    RLF_CHECK(report.model_states_identical);
    RLF_CHECK(report.retrieval_seconds > 0.0);
    RLF_CHECK(report.indexed_seconds > 0.0);
    RLF_CHECK(report.reference_retrievals == inputs.updates);
    RLF_CHECK(report.indexed_retrievals == 0U);
    RLF_CHECK(report.indexed_retrievals_avoided == inputs.updates);
    std::ostringstream json;
    rlf::solstice::write_instruction_duplicate_prefilter_ablation_json(
        json, report
    );
    RLF_CHECK(json.str().find("\"model_states_identical\": true") !=
              std::string::npos);
    RLF_CHECK(json.str().find("\"gpu_efficiency_evidence_eligible\": false") !=
              std::string::npos);
}

RLF_TEST_CASE("Indexed tool keyword ablation preserves exact learned routes") {
    rlf::solstice::ToolKeywordIndexAblationInputs inputs;
    inputs.keywords = 128U;
    inputs.updates = 16U;
    inputs.repetitions = 1U;
    const auto report =
        rlf::solstice::measure_tool_keyword_index_ablation(inputs);
    RLF_CHECK(report.model_states_identical);
    RLF_CHECK(report.linear_seconds > 0.0);
    RLF_CHECK(report.indexed_seconds > 0.0);
    RLF_CHECK(report.keyword_update_lookups == inputs.updates);
    RLF_CHECK(report.indexed_keyword_lookups == inputs.updates);
    RLF_CHECK(report.linear_keyword_comparisons >
              report.indexed_keyword_lookups);
    RLF_CHECK(report.keyword_capacity_skips == 0U);
    std::ostringstream json;
    rlf::solstice::write_tool_keyword_index_ablation_json(json, report);
    RLF_CHECK(json.str().find("\"model_states_identical\": true") !=
              std::string::npos);
    RLF_CHECK(json.str().find("\"gpu_efficiency_evidence_eligible\": false") !=
              std::string::npos);
}

RLF_TEST_CASE("Fused dialogue encoding ablation preserves exact language state") {
    rlf::solstice::DialogueEncodingAblationInputs inputs;
    inputs.dialogues = 16U;
    inputs.repetitions = 1U;
    const auto report =
        rlf::solstice::measure_dialogue_encoding_ablation(inputs);
    RLF_CHECK(report.model_states_identical);
    RLF_CHECK(report.redundant_seconds > 0.0);
    RLF_CHECK(report.fused_seconds > 0.0);
    RLF_CHECK(report.redundant_encode_calls == inputs.dialogues * 5U);
    RLF_CHECK(report.fused_encode_calls == inputs.dialogues * 3U);
    RLF_CHECK(report.encode_calls_avoided == inputs.dialogues * 2U);
    std::ostringstream json;
    rlf::solstice::write_dialogue_encoding_ablation_json(json, report);
    RLF_CHECK(json.str().find("\"model_states_identical\": true") !=
              std::string::npos);
    RLF_CHECK(json.str().find("\"gpu_efficiency_evidence_eligible\": false") !=
              std::string::npos);
}

RLF_TEST_CASE("Efficiency corpus has exact counts and passes its ledger audit") {
    const std::filesystem::path output =
        std::filesystem::temp_directory_path() / "rlf_efficiency_corpus_test_v1";
    std::error_code error;
    std::filesystem::remove_all(output, error);
    std::filesystem::remove_all(output.string() + ".tmp", error);
    rlf::solstice::EfficiencyCorpusConfig config;
    config.output_directory = output;
    config.training_records = 1'000U;
    config.evaluation_records_per_category = 1U;
    const auto report = rlf::solstice::generate_efficiency_corpus(config);
    RLF_CHECK(report.audit_passed);
    RLF_CHECK(report.training_records == 1'000U);
    RLF_CHECK(report.evaluation_records == 14U);
    std::size_t categories = 0U;
    for (const auto& [name, count] : report.category_counts) {
        static_cast<void>(name);
        categories += count;
    }
    RLF_CHECK(categories == 1'000U);
    RLF_CHECK(std::filesystem::is_regular_file(report.ledger_path));
    const std::filesystem::path evaluation_output = output.string() + "_evaluation";
    std::filesystem::remove_all(evaluation_output, error);
    const auto evaluation = rlf::solstice::materialize_efficiency_evaluation(
        report.ledger_path, evaluation_output
    );
    RLF_CHECK(evaluation.examples == 14U);
    RLF_CHECK(evaluation.image_examples == 3U);
    RLF_CHECK(std::filesystem::is_regular_file(evaluation_output / "requests.tsv"));
    RLF_CHECK(std::filesystem::is_regular_file(evaluation_output / "expected.tsv"));
    const std::filesystem::path fake_results = output.string() + "_results";
    const std::filesystem::path score_output = output.string() + "_scores";
    std::filesystem::remove_all(fake_results, error);
    std::filesystem::remove_all(score_output, error);
    std::filesystem::create_directories(fake_results);
    std::ifstream answers(evaluation_output / "expected.tsv");
    std::string line;
    while (std::getline(answers, line)) {
        if (line.empty() || line.front() == '#') continue;
        std::vector<std::string> fields;
        std::size_t start = 0U;
        while (start <= line.size()) {
            const std::size_t tab = line.find('\t', start);
            fields.push_back(line.substr(
                start, tab == std::string::npos ? line.size() - start : tab - start
            ));
            if (tab == std::string::npos) break;
            start = tab + 1U;
        }
        RLF_CHECK(fields.size() == 4U);
        const std::filesystem::path result_path = fake_results / (fields[0U] + ".json");
        {
            std::ofstream result(result_path);
            result << "{\"response\": {\"text\": \"" << fields[3U] << "\"}}\n";
        }
        const std::string result_sha = rlf::core::sha256_hex(
            rlf::core::sha256_file(result_path)
        );
        std::ofstream metadata(fake_results / (fields[0U] + ".meta.tsv"));
        metadata << fields[0U] << "\tprompt\t-\tcheckpoint\tmodel\t"
                 << result_sha << "\t1\n";
    }
    const auto score = rlf::solstice::score_efficiency_evaluation(
        evaluation_output / "expected.tsv", fake_results, score_output
    );
    RLF_CHECK(score.examples == 14U);
    RLF_CHECK(score.correct == 14U);
    RLF_CHECK(score.accuracy == 1.0);
    std::filesystem::remove_all(output, error);
    std::filesystem::remove_all(evaluation_output, error);
    std::filesystem::remove_all(fake_results, error);
    std::filesystem::remove_all(score_output, error);
}
