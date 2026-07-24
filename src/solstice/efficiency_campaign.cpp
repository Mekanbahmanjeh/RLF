#include "rlf/solstice/efficiency_campaign.hpp"
#include "rlf/frontier/frontier_backend.hpp"
#include "rlf/solstice/grounding_fabric.hpp"
#include "rlf/solstice/language_fabric.hpp"
#include "rlf/solstice/general_fabric.hpp"
#include "rlf/solstice/sparse_router.hpp"
#include "rlf/solstice/tool_protocol.hpp"
#include "rlf/solstice/vision_fabric.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace rlf::solstice {
namespace {

double duration(const std::uint64_t amount, const double rate, const char* name) {
    if (amount == 0U) {
        return 0.0;
    }
    if (!std::isfinite(rate) || rate <= 0.0) {
        throw std::invalid_argument(std::string(name) + " must be positive when work is nonzero");
    }
    return static_cast<double>(amount) / rate;
}

void require_nonnegative_finite(const double value, const char* name) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(std::string(name) + " must be finite and nonnegative");
    }
}

[[nodiscard]] ImageData ablation_image(
    const std::size_t image_index,
    const std::size_t width,
    const std::size_t height
) {
    ImageData image;
    image.width = width;
    image.height = height;
    image.rgb.reserve(width * height * 3U);
    for (std::size_t y = 0U; y < height; ++y) {
        for (std::size_t x = 0U; x < width; ++x) {
            image.rgb.push_back(static_cast<std::uint8_t>(
                (x * 17U + y * 3U + image_index * 29U) & 0xFFU
            ));
            image.rgb.push_back(static_cast<std::uint8_t>(
                (x * 5U + y * 19U + image_index * 11U) & 0xFFU
            ));
            image.rgb.push_back(static_cast<std::uint8_t>(
                (x * 13U + y * 7U + image_index * 23U) & 0xFFU
            ));
        }
    }
    return image;
}

[[nodiscard]] bool same_analysis(
    const VisionAnalysis& left,
    const VisionAnalysis& right
) {
    if (left.width != right.width || left.height != right.height ||
        left.description != right.description || left.concepts != right.concepts ||
        left.confidence != right.confidence ||
        left.nearest_example_id != right.nearest_example_id ||
        left.regions.size() != right.regions.size()) {
        return false;
    }
    for (std::size_t index = 0U; index < left.regions.size(); ++index) {
        const VisualRegion& a = left.regions[index];
        const VisualRegion& b = right.regions[index];
        if (a.mode_id != b.mode_id || a.x != b.x || a.y != b.y ||
            a.width != b.width || a.height != b.height ||
            a.patch_count != b.patch_count ||
            a.concept_name != b.concept_name || a.confidence != b.confidence) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool same_grounding_hits(
    const std::vector<GroundingHit>& left,
    const std::vector<GroundingHit>& right
) {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t index = 0U; index < left.size(); ++index) {
        if (left[index].visual_mode_id != right[index].visual_mode_id ||
            left[index].concept_name != right[index].concept_name ||
            left[index].score != right[index].score ||
            left[index].support != right[index].support) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool same_prediction(
    const HierarchicalPrediction& left,
    const HierarchicalPrediction& right
) {
    if (left.deepest_context_order != right.deepest_context_order ||
        left.uncertainty != right.uncertainty ||
        left.candidates.size() != right.candidates.size()) {
        return false;
    }
    for (std::size_t index = 0U; index < left.candidates.size(); ++index) {
        if (left.candidates[index].token != right.candidates[index].token ||
            left.candidates[index].probability != right.candidates[index].probability ||
            left.candidates[index].score != right.candidates[index].score) {
            return false;
        }
    }
    return true;
}

void hash_route(std::uint64_t& hash, const SparseRouteResult& route) noexcept {
    constexpr std::uint64_t prime = 1'099'511'628'211ULL;
    const auto mix = [&](const std::uint64_t value) {
        hash ^= value;
        hash *= prime;
    };
    mix(route.signatures_probed);
    mix(route.candidates_examined);
    mix(route.exhaustive_candidates);
    for (const std::size_t index : route.candidate_indices) {
        mix(static_cast<std::uint64_t>(index));
    }
}

}  // namespace

PhysicalRuntimeFloor calculate_physical_runtime_floor(
    const PhysicalRuntimeInputs& inputs
) {
    if (!std::isfinite(inputs.baseline_seconds) || inputs.baseline_seconds <= 0.0) {
        throw std::invalid_argument("baseline_seconds must be positive");
    }
    if (!std::isfinite(inputs.target_speedup) || inputs.target_speedup <= 0.0) {
        throw std::invalid_argument("target_speedup must be positive");
    }
    require_nonnegative_finite(inputs.evaluation_seconds, "evaluation_seconds");
    PhysicalRuntimeFloor report;
    report.inputs = inputs;
    report.minimum_read_seconds = duration(
        inputs.compressed_dataset_bytes, inputs.nvme_bytes_per_second, "nvme_bytes_per_second"
    );
    report.minimum_decompression_seconds = duration(
        inputs.decompressed_dataset_bytes,
        inputs.decompression_bytes_per_second,
        "decompression_bytes_per_second"
    );
    report.minimum_pcie_seconds = duration(
        inputs.pcie_transfer_bytes, inputs.pcie_bytes_per_second, "pcie_bytes_per_second"
    );
    report.minimum_decode_seconds = duration(
        inputs.image_count, inputs.image_decodes_per_second, "image_decodes_per_second"
    );
    report.minimum_preprocessing_seconds = duration(
        inputs.record_count,
        inputs.preprocessing_records_per_second,
        "preprocessing_records_per_second"
    );
    report.minimum_checkpoint_seconds = duration(
        inputs.checkpoint_bytes,
        inputs.checkpoint_write_bytes_per_second,
        "checkpoint_write_bytes_per_second"
    );
    report.minimum_evaluation_seconds = inputs.evaluation_seconds;
    report.serial_floor_seconds = report.minimum_read_seconds +
        report.minimum_decompression_seconds + report.minimum_pcie_seconds +
        report.minimum_decode_seconds + report.minimum_preprocessing_seconds +
        report.minimum_checkpoint_seconds + report.minimum_evaluation_seconds;
    const double overlapped_ingress_floor = std::max({
        report.minimum_read_seconds,
        report.minimum_decompression_seconds,
        report.minimum_pcie_seconds,
        report.minimum_decode_seconds,
        report.minimum_preprocessing_seconds,
    });
    report.theoretical_best_case_seconds = overlapped_ingress_floor +
        report.minimum_checkpoint_seconds + report.minimum_evaluation_seconds;
    report.target_runtime_seconds = inputs.baseline_seconds / inputs.target_speedup;
    report.maximum_possible_end_to_end_speedup =
        report.theoretical_best_case_seconds > 0.0
        ? inputs.baseline_seconds / report.theoretical_best_case_seconds
        : 0.0;
    report.target_physically_possible_under_inputs =
        report.theoretical_best_case_seconds <= report.target_runtime_seconds;
    // A command-line label alone cannot authenticate target-host measurements.
    // A later evidence verifier must bind raw logs before this can become true.
    report.measured_hardware_evidence = false;
    return report;
}

void write_physical_runtime_floor_json(
    std::ostream& output,
    const PhysicalRuntimeFloor& report
) {
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"rlf-efficiency-physical-floor-v1\",\n"
           << "  \"hardware_profile\": \"" << report.inputs.hardware_profile << "\",\n"
           << "  \"evidence_kind\": \"" << report.inputs.evidence_kind << "\",\n"
           << "  \"measured_hardware_evidence\": "
           << (report.measured_hardware_evidence ? "true" : "false") << ",\n"
           << "  \"dataset_size_bytes\": " << report.inputs.compressed_dataset_bytes << ",\n"
           << "  \"decompressed_size_bytes\": " << report.inputs.decompressed_dataset_bytes << ",\n"
           << "  \"pcie_transfer_bytes\": " << report.inputs.pcie_transfer_bytes << ",\n"
           << "  \"image_count\": " << report.inputs.image_count << ",\n"
           << "  \"record_count\": " << report.inputs.record_count << ",\n"
           << "  \"checkpoint_size_bytes\": " << report.inputs.checkpoint_bytes << ",\n"
           << "  \"nvme_bytes_per_second\": " << report.inputs.nvme_bytes_per_second << ",\n"
           << "  \"decompression_bytes_per_second\": "
           << report.inputs.decompression_bytes_per_second << ",\n"
           << "  \"pcie_bytes_per_second\": " << report.inputs.pcie_bytes_per_second << ",\n"
           << "  \"image_decodes_per_second\": "
           << report.inputs.image_decodes_per_second << ",\n"
           << "  \"preprocessing_records_per_second\": "
           << report.inputs.preprocessing_records_per_second << ",\n"
           << "  \"checkpoint_write_bytes_per_second\": "
           << report.inputs.checkpoint_write_bytes_per_second << ",\n"
           << "  \"minimum_read_time_seconds\": " << report.minimum_read_seconds << ",\n"
           << "  \"minimum_decompression_time_seconds\": " << report.minimum_decompression_seconds << ",\n"
           << "  \"minimum_pcie_time_seconds\": " << report.minimum_pcie_seconds << ",\n"
           << "  \"minimum_decode_time_seconds\": " << report.minimum_decode_seconds << ",\n"
           << "  \"minimum_preprocessing_time_seconds\": " << report.minimum_preprocessing_seconds << ",\n"
           << "  \"minimum_checkpoint_time_seconds\": " << report.minimum_checkpoint_seconds << ",\n"
           << "  \"minimum_evaluation_time_seconds\": " << report.minimum_evaluation_seconds << ",\n"
           << "  \"serial_floor_seconds\": " << report.serial_floor_seconds << ",\n"
           << "  \"theoretical_best_case_runtime_seconds\": "
           << report.theoretical_best_case_seconds << ",\n"
           << "  \"baseline_runtime_seconds\": " << report.inputs.baseline_seconds << ",\n"
           << "  \"target_speedup\": " << report.inputs.target_speedup << ",\n"
           << "  \"target_runtime_seconds\": " << report.target_runtime_seconds << ",\n"
           << "  \"maximum_possible_end_to_end_speedup\": "
           << report.maximum_possible_end_to_end_speedup << ",\n"
           << "  \"target_physically_possible_under_inputs\": "
           << (report.target_physically_possible_under_inputs ? "true" : "false") << ",\n"
           << "  \"claim_boundary\": \"The evidence-kind field is self-declared and does not authenticate raw logs. The best-case floor assumes steady-state overlap and zero learning-update cost.\"\n"
           << "}\n";
}

FiftyMillionTrainingContract evaluate_fifty_million_contract(
    const FiftyMillionTrainingEvidence& evidence
) {
    constexpr std::uint64_t expected_total = 50'000'000U;
    constexpr std::uint64_t expected_instructions = 24'500'001U;
    constexpr std::uint64_t expected_preferences = 8'166'666U;
    constexpr std::uint64_t expected_tools = 8'166'667U;
    constexpr std::uint64_t expected_facts = 8'166'666U;
    constexpr std::uint64_t expected_vision = 1'000'000U;

    FiftyMillionTrainingContract report;
    report.evidence = evidence;
    report.stated_hour_required_records_per_second =
        static_cast<double>(expected_total) / report.stated_hour_target_seconds;
    report.strict_required_records_per_second =
        static_cast<double>(expected_total) /
        report.strict_eighteen_second_target_seconds;
    report.measured_records_per_second = evidence.wall_seconds > 0.0
        ? static_cast<double>(evidence.total_records) / evidence.wall_seconds
        : 0.0;
    report.measured_gpu_hour_speedup =
        std::isfinite(evidence.baseline_gpu_hours) &&
        std::isfinite(evidence.candidate_gpu_hours) &&
        evidence.baseline_gpu_hours > 0.0 && evidence.candidate_gpu_hours > 0.0
        ? evidence.baseline_gpu_hours / evidence.candidate_gpu_hours
        : 0.0;
    report.target_wording_is_time_consistent =
        report.stated_hour_target_seconds ==
        report.strict_eighteen_second_target_seconds;
    report.exact_record_mix =
        evidence.total_records == expected_total &&
        evidence.instruction_records == expected_instructions &&
        evidence.preference_records == expected_preferences &&
        evidence.tool_records == expected_tools &&
        evidence.fact_records == expected_facts &&
        evidence.vision_records == expected_vision;
    report.exact_rtx_profile =
        evidence.hardware_profile == "rtx-pro-6000-96g" ||
        evidence.hardware_profile == "general-rtx-pro-6000-96g";
    report.physical_device_identity_present =
        evidence.device_name.find("RTX PRO 6000") != std::string::npos &&
        !evidence.device_uuid.empty();
    report.single_gpu_without_mig =
        evidence.device_count == 1U && evidence.mig_disabled;
    report.cuda_backend = evidence.backend == "cuda";
    report.no_silent_capacity_saturation =
        evidence.episode_capacity_skips == 0U &&
        evidence.context_capacity_skips == 0U &&
        evidence.tool_keyword_capacity_skips == 0U;
    const std::uint64_t maximum_vram =
        evidence.hardware_profile == "rtx-pro-6000-96g"
        ? 88ULL * 1024ULL * 1024ULL * 1024ULL
        : 90ULL * 1024ULL * 1024ULL * 1024ULL;
    report.vram_within_profile_limit =
        evidence.peak_vram_bytes > 0U &&
        evidence.peak_vram_bytes <= maximum_vram;
    report.physical_evidence_complete =
        report.exact_record_mix && report.exact_rtx_profile &&
        report.physical_device_identity_present &&
        report.single_gpu_without_mig && report.cuda_backend &&
        report.no_silent_capacity_saturation &&
        report.vram_within_profile_limit &&
        evidence.wall_seconds > 0.0 && evidence.gpu_active_seconds > 0.0 &&
        evidence.checkpoint_verified && evidence.raw_gpu_trace_bound &&
        evidence.raw_energy_trace_bound;
    report.stated_hour_target_passed =
        report.physical_evidence_complete &&
        evidence.wall_seconds <= report.stated_hour_target_seconds;
    report.strict_eighteen_second_target_passed =
        report.physical_evidence_complete &&
        evidence.wall_seconds <= report.strict_eighteen_second_target_seconds;
    report.general_100000x_gpu_hour_efficiency_proven =
        report.physical_evidence_complete && evidence.matched_quality &&
        evidence.external_quality_gate_passed &&
        evidence.independent_reproduction_complete &&
        report.measured_gpu_hour_speedup >= 100'000.0;
    return report;
}

void write_fifty_million_contract_json(
    std::ostream& output,
    const FiftyMillionTrainingContract& report
) {
    const auto boolean = [](const bool value) { return value ? "true" : "false"; };
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"rlf-fifty-million-training-contract-v1\",\n"
           << "  \"hardware_profile\": "
           << std::quoted(report.evidence.hardware_profile) << ",\n"
           << "  \"device_name\": " << std::quoted(report.evidence.device_name)
           << ",\n"
           << "  \"device_uuid\": " << std::quoted(report.evidence.device_uuid)
           << ",\n"
           << "  \"backend\": " << std::quoted(report.evidence.backend) << ",\n"
           << "  \"device_count\": " << report.evidence.device_count << ",\n"
           << "  \"mig_disabled\": " << boolean(report.evidence.mig_disabled) << ",\n"
           << "  \"records\": {\n"
           << "    \"total\": " << report.evidence.total_records << ",\n"
           << "    \"instructions\": " << report.evidence.instruction_records << ",\n"
           << "    \"preferences\": " << report.evidence.preference_records << ",\n"
           << "    \"tools\": " << report.evidence.tool_records << ",\n"
           << "    \"facts\": " << report.evidence.fact_records << ",\n"
           << "    \"vision\": " << report.evidence.vision_records << "\n"
           << "  },\n"
           << "  \"capacity_skips\": {\n"
           << "    \"episodes\": " << report.evidence.episode_capacity_skips << ",\n"
           << "    \"contexts\": " << report.evidence.context_capacity_skips << ",\n"
           << "    \"tool_keywords\": "
           << report.evidence.tool_keyword_capacity_skips << "\n"
           << "  },\n"
           << "  \"wall_seconds\": " << report.evidence.wall_seconds << ",\n"
           << "  \"gpu_active_seconds\": " << report.evidence.gpu_active_seconds << ",\n"
           << "  \"peak_vram_bytes\": " << report.evidence.peak_vram_bytes << ",\n"
           << "  \"stated_0_028_hour_target_seconds\": "
           << report.stated_hour_target_seconds << ",\n"
           << "  \"strict_eighteen_second_target_seconds\": "
           << report.strict_eighteen_second_target_seconds << ",\n"
           << "  \"target_wording_is_time_consistent\": "
           << boolean(report.target_wording_is_time_consistent) << ",\n"
           << "  \"stated_hour_required_records_per_second\": "
           << report.stated_hour_required_records_per_second << ",\n"
           << "  \"strict_required_records_per_second\": "
           << report.strict_required_records_per_second << ",\n"
           << "  \"measured_records_per_second\": "
           << report.measured_records_per_second << ",\n"
           << "  \"exact_record_mix\": " << boolean(report.exact_record_mix) << ",\n"
           << "  \"exact_rtx_profile\": " << boolean(report.exact_rtx_profile) << ",\n"
           << "  \"physical_device_identity_present\": "
           << boolean(report.physical_device_identity_present) << ",\n"
           << "  \"single_gpu_without_mig\": "
           << boolean(report.single_gpu_without_mig) << ",\n"
           << "  \"cuda_backend\": " << boolean(report.cuda_backend) << ",\n"
           << "  \"no_silent_capacity_saturation\": "
           << boolean(report.no_silent_capacity_saturation) << ",\n"
           << "  \"vram_within_profile_limit\": "
           << boolean(report.vram_within_profile_limit) << ",\n"
           << "  \"checkpoint_verified\": "
           << boolean(report.evidence.checkpoint_verified) << ",\n"
           << "  \"raw_gpu_trace_bound\": "
           << boolean(report.evidence.raw_gpu_trace_bound) << ",\n"
           << "  \"raw_energy_trace_bound\": "
           << boolean(report.evidence.raw_energy_trace_bound) << ",\n"
           << "  \"physical_evidence_complete\": "
           << boolean(report.physical_evidence_complete) << ",\n"
           << "  \"stated_hour_target_passed\": "
           << boolean(report.stated_hour_target_passed) << ",\n"
           << "  \"strict_eighteen_second_target_passed\": "
           << boolean(report.strict_eighteen_second_target_passed) << ",\n"
           << "  \"baseline_gpu_hours\": "
           << report.evidence.baseline_gpu_hours << ",\n"
           << "  \"candidate_gpu_hours\": "
           << report.evidence.candidate_gpu_hours << ",\n"
           << "  \"measured_gpu_hour_speedup\": "
           << report.measured_gpu_hour_speedup << ",\n"
           << "  \"matched_quality\": "
           << boolean(report.evidence.matched_quality) << ",\n"
           << "  \"external_quality_gate_passed\": "
           << boolean(report.evidence.external_quality_gate_passed) << ",\n"
           << "  \"independent_reproduction_complete\": "
           << boolean(report.evidence.independent_reproduction_complete) << ",\n"
           << "  \"general_100000x_gpu_hour_efficiency_proven\": "
           << boolean(report.general_100000x_gpu_hour_efficiency_proven) << ",\n"
           << "  \"claim_boundary\": \"The 18-second ingest/training target and 0.028-hour target are separate because 0.028 hours equals 100.8 seconds. A throughput pass is not a general efficiency claim without matched external quality and independently reproduced GPU-hour evidence.\"\n"
           << "}\n";
}

VisionGroundingAblation measure_vision_grounding_ablation(
    const VisionGroundingAblationInputs& inputs
) {
    if (inputs.images == 0U || inputs.width == 0U || inputs.height == 0U ||
        inputs.repetitions == 0U || inputs.width > 4'096U ||
        inputs.height > 4'096U ||
        inputs.width > std::numeric_limits<std::size_t>::max() / inputs.height ||
        inputs.width * inputs.height >
            std::numeric_limits<std::size_t>::max() / 3U) {
        throw std::invalid_argument("invalid vision grounding ablation dimensions");
    }
    std::vector<ImageData> images;
    images.reserve(inputs.images);
    for (std::size_t index = 0U; index < inputs.images; ++index) {
        images.push_back(ablation_image(index, inputs.width, inputs.height));
    }

    VisionGroundingAblation report;
    report.inputs = inputs;
    report.analyses_identical = true;
    report.model_states_identical = true;
    using Clock = std::chrono::steady_clock;
    for (std::size_t repetition = 0U; repetition < inputs.repetitions; ++repetition) {
        VisualPatchFabric reference;
        VisualPatchFabric fused;
        std::vector<VisionAnalysis> expected;
        std::vector<VisionAnalysis> actual;
        expected.reserve(images.size());
        actual.reserve(images.size());
        const auto run_reference = [&]() {
            const auto start = Clock::now();
            for (std::size_t index = 0U; index < images.size(); ++index) {
                const std::string caption = "deterministic visual sample " +
                    std::to_string(index % 17U);
                reference.train(images[index], caption);
                expected.push_back(reference.analyze(images[index]));
            }
            report.separate_train_analyze_seconds +=
                std::chrono::duration<double>(Clock::now() - start).count();
        };
        const auto run_fused = [&]() {
            const auto start = Clock::now();
            for (std::size_t index = 0U; index < images.size(); ++index) {
                const std::string caption = "deterministic visual sample " +
                    std::to_string(index % 17U);
                actual.push_back(fused.train_and_analyze(images[index], caption));
            }
            report.fused_train_analyze_seconds +=
                std::chrono::duration<double>(Clock::now() - start).count();
        };
        if ((repetition & 1U) == 0U) {
            run_reference();
            run_fused();
        } else {
            run_fused();
            run_reference();
        }
        if (expected.size() != actual.size()) {
            report.analyses_identical = false;
        } else {
            for (std::size_t index = 0U; index < expected.size(); ++index) {
                report.analyses_identical = report.analyses_identical &&
                    same_analysis(expected[index], actual[index]);
            }
        }
        report.reference_model_hash = reference.deterministic_hash();
        report.fused_model_hash = fused.deterministic_hash();
        report.reference_local_updates +=
            reference.backend_operation_stats().local_update_calls;
        report.fused_local_updates +=
            fused.backend_operation_stats().local_update_calls;
        report.model_states_identical = report.model_states_identical &&
            report.reference_model_hash == report.fused_model_hash &&
            report.reference_local_updates == report.fused_local_updates;
    }
    report.speedup = report.fused_train_analyze_seconds > 0.0
        ? report.separate_train_analyze_seconds /
            report.fused_train_analyze_seconds
        : 0.0;
    return report;
}

void write_vision_grounding_ablation_json(
    std::ostream& output,
    const VisionGroundingAblation& report
) {
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"rlf-vision-grounding-ablation-v1\",\n"
           << "  \"scope\": \"CPU preparation-host visual train-plus-grounding path; not GPU-hour or general-efficiency evidence\",\n"
           << "  \"images_per_repetition\": " << report.inputs.images << ",\n"
           << "  \"width\": " << report.inputs.width << ",\n"
           << "  \"height\": " << report.inputs.height << ",\n"
           << "  \"repetitions\": " << report.inputs.repetitions << ",\n"
           << "  \"separate_train_analyze_seconds\": "
           << report.separate_train_analyze_seconds << ",\n"
           << "  \"fused_train_analyze_seconds\": "
           << report.fused_train_analyze_seconds << ",\n"
           << "  \"speedup\": " << report.speedup << ",\n"
           << "  \"analyses_identical\": "
           << (report.analyses_identical ? "true" : "false") << ",\n"
           << "  \"model_states_identical\": "
           << (report.model_states_identical ? "true" : "false") << ",\n"
           << "  \"reference_model_hash\": " << report.reference_model_hash << ",\n"
           << "  \"fused_model_hash\": " << report.fused_model_hash << ",\n"
           << "  \"reference_local_updates\": "
           << report.reference_local_updates << ",\n"
           << "  \"fused_local_updates\": "
           << report.fused_local_updates << ",\n"
           << "  \"external_quality_evidence_eligible\": false,\n"
           << "  \"gpu_efficiency_evidence_eligible\": false\n"
           << "}\n";
}

SparseRouterUpdateAblation measure_sparse_router_update_ablation(
    const SparseRouterUpdateAblationInputs& inputs
) {
    if (inputs.vector_count == 0U || inputs.dimensions == 0U ||
        inputs.updates_per_batch == 0U || inputs.batches == 0U ||
        inputs.queries_per_batch == 0U || inputs.repetitions == 0U ||
        inputs.vector_count > 1'000'000U || inputs.dimensions > 4'096U ||
        inputs.batches > 1'000'000U ||
        inputs.queries_per_batch > 1'000'000U ||
        inputs.repetitions > 1'000U ||
        inputs.updates_per_batch > inputs.vector_count ||
        inputs.vector_count >
            std::numeric_limits<std::size_t>::max() / inputs.dimensions) {
        throw std::invalid_argument("invalid sparse-router ablation inputs");
    }
    std::vector<float> initial(inputs.vector_count * inputs.dimensions, 0.0F);
    for (std::size_t vector = 0U; vector < inputs.vector_count; ++vector) {
        for (std::size_t dimension = 0U; dimension < inputs.dimensions; ++dimension) {
            const std::uint64_t mixed =
                (static_cast<std::uint64_t>(vector) + 17ULL) *
                (static_cast<std::uint64_t>(dimension) + 31ULL) *
                0x9e3779b97f4a7c15ULL;
            initial[vector * inputs.dimensions + dimension] =
                static_cast<float>(
                    static_cast<double>(mixed % 65'521ULL) / 32'760.5 - 1.0
                );
        }
    }
    SparseRouterConfig config;
    config.signature_bits = 18U;
    config.maximum_candidates = 256U;
    config.probe_radius = 2U;
    SparseRouterUpdateAblation report;
    report.inputs = inputs;
    report.all_routes_identical = true;
    report.route_hash = 14'695'981'039'346'656'037ULL;
    using Clock = std::chrono::steady_clock;
    for (std::size_t repetition = 0U; repetition < inputs.repetitions; ++repetition) {
        std::vector<float> matrix = initial;
        SparseRoutingIndex reference(config);
        SparseRoutingIndex incremental(config);
        reference.rebuild(matrix, inputs.vector_count, inputs.dimensions);
        incremental.rebuild(matrix, inputs.vector_count, inputs.dimensions);
        for (std::size_t batch = 0U; batch < inputs.batches; ++batch) {
            std::vector<std::size_t> updated;
            updated.reserve(inputs.updates_per_batch);
            for (std::size_t offset = 0U;
                 offset < inputs.updates_per_batch;
                 ++offset) {
                const std::size_t index =
                    (batch * 131U + offset * 257U) % inputs.vector_count;
                updated.push_back(index);
                for (std::size_t dimension = 0U;
                     dimension < inputs.dimensions;
                     ++dimension) {
                    matrix[index * inputs.dimensions + dimension] +=
                        ((dimension + offset + batch) & 1U) == 0U
                        ? 0.0005F : -0.0004F;
                }
            }
            const auto run_reference = [&]() {
                const auto start = Clock::now();
                reference.rebuild(
                    matrix, inputs.vector_count, inputs.dimensions
                );
                report.full_rebuild_seconds +=
                    std::chrono::duration<double>(Clock::now() - start).count();
            };
            const auto run_incremental = [&]() {
                const auto start = Clock::now();
                for (const std::size_t index : updated) {
                    incremental.update(
                        index,
                        std::span<const float>(
                            matrix.data() + index * inputs.dimensions,
                            inputs.dimensions
                        )
                    );
                }
                report.incremental_update_seconds +=
                    std::chrono::duration<double>(Clock::now() - start).count();
            };
            if (((batch + repetition) & 1U) == 0U) {
                run_reference();
                run_incremental();
            } else {
                run_incremental();
                run_reference();
            }
            for (std::size_t query = 0U;
                 query < inputs.queries_per_batch;
                 ++query) {
                const std::size_t index =
                    (query * 509U + batch * 37U) % inputs.vector_count;
                const auto vector = std::span<const float>(
                    matrix.data() + index * inputs.dimensions,
                    inputs.dimensions
                );
                const SparseRouteResult expected = reference.route(vector);
                const SparseRouteResult actual = incremental.route(vector);
                report.all_routes_identical = report.all_routes_identical &&
                    expected.candidate_indices == actual.candidate_indices &&
                    expected.signatures_probed == actual.signatures_probed &&
                    expected.candidates_examined == actual.candidates_examined &&
                    expected.exhaustive_candidates == actual.exhaustive_candidates;
                hash_route(report.route_hash, actual);
            }
        }
        const auto reference_stats = reference.operation_stats();
        const auto incremental_stats = incremental.operation_stats();
        report.reference_vectors_rebuilt += reference_stats.vectors_rebuilt;
        report.incremental_vectors_rebuilt += incremental_stats.vectors_rebuilt;
        report.vectors_incrementally_updated +=
            incremental_stats.vectors_incrementally_updated;
    }
    report.speedup = report.incremental_update_seconds > 0.0
        ? report.full_rebuild_seconds / report.incremental_update_seconds
        : 0.0;
    return report;
}

void write_sparse_router_update_ablation_json(
    std::ostream& output,
    const SparseRouterUpdateAblation& report
) {
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"rlf-sparse-router-update-ablation-v1\",\n"
           << "  \"scope\": \"CPU preparation-host sparse-index maintenance only; not full training, GPU-hour, or external-quality evidence\",\n"
           << "  \"vector_count\": " << report.inputs.vector_count << ",\n"
           << "  \"dimensions\": " << report.inputs.dimensions << ",\n"
           << "  \"updates_per_batch\": " << report.inputs.updates_per_batch << ",\n"
           << "  \"batches\": " << report.inputs.batches << ",\n"
           << "  \"queries_per_batch\": " << report.inputs.queries_per_batch << ",\n"
           << "  \"repetitions\": " << report.inputs.repetitions << ",\n"
           << "  \"full_rebuild_seconds\": " << report.full_rebuild_seconds << ",\n"
           << "  \"incremental_update_seconds\": " << report.incremental_update_seconds << ",\n"
           << "  \"speedup\": " << report.speedup << ",\n"
           << "  \"all_routes_identical\": "
           << (report.all_routes_identical ? "true" : "false") << ",\n"
           << "  \"route_hash\": " << report.route_hash << ",\n"
           << "  \"reference_vectors_rebuilt\": "
           << report.reference_vectors_rebuilt << ",\n"
           << "  \"incremental_vectors_rebuilt\": "
           << report.incremental_vectors_rebuilt << ",\n"
           << "  \"vectors_incrementally_updated\": "
           << report.vectors_incrementally_updated << ",\n"
           << "  \"capacity_changed\": false,\n"
           << "  \"candidate_limit_changed\": false,\n"
           << "  \"external_quality_evidence_eligible\": false,\n"
           << "  \"gpu_efficiency_evidence_eligible\": false\n"
           << "}\n";
}

SparseRerankBatchAblation measure_sparse_rerank_batch_ablation(
    const SparseRerankBatchAblationInputs& inputs
) {
    if (inputs.vector_count == 0U || inputs.query_count == 0U ||
        inputs.dimensions == 0U || inputs.maximum_candidates == 0U ||
        inputs.repetitions == 0U || inputs.vector_count > 1'000'000U ||
        inputs.query_count > 1'000'000U || inputs.dimensions > 4'096U ||
        inputs.maximum_candidates > inputs.vector_count ||
        inputs.repetitions > 1'000U ||
        inputs.vector_count >
            std::numeric_limits<std::size_t>::max() / inputs.dimensions ||
        inputs.query_count >
            std::numeric_limits<std::size_t>::max() / inputs.dimensions) {
        throw std::invalid_argument("invalid sparse-rerank ablation inputs");
    }
    std::vector<float> modes(inputs.vector_count * inputs.dimensions, 0.0F);
    for (std::size_t vector = 0U; vector < inputs.vector_count; ++vector) {
        for (std::size_t dimension = 0U; dimension < inputs.dimensions; ++dimension) {
            const std::uint64_t mixed =
                (static_cast<std::uint64_t>(vector) + 97ULL) *
                (static_cast<std::uint64_t>(dimension) + 53ULL) *
                0x9e3779b97f4a7c15ULL;
            modes[vector * inputs.dimensions + dimension] = static_cast<float>(
                static_cast<double>(mixed % 65'521ULL) / 32'760.5 - 1.0
            );
        }
    }
    std::vector<float> queries(inputs.query_count * inputs.dimensions, 0.0F);
    for (std::size_t query = 0U; query < inputs.query_count; ++query) {
        const std::size_t source = (query * 509U + 17U) % inputs.vector_count;
        for (std::size_t dimension = 0U; dimension < inputs.dimensions; ++dimension) {
            queries[query * inputs.dimensions + dimension] =
                modes[source * inputs.dimensions + dimension] +
                (((query + dimension) & 1U) == 0U ? 0.0003F : -0.0002F);
        }
    }
    SparseRouterConfig config;
    config.signature_bits = 18U;
    config.maximum_candidates = inputs.maximum_candidates;
    config.probe_radius = 2U;
    SparseRoutingIndex router(config);
    router.rebuild(modes, inputs.vector_count, inputs.dimensions);
    std::vector<SparseRouteResult> routes;
    routes.reserve(inputs.query_count);
    for (std::size_t query = 0U; query < inputs.query_count; ++query) {
        routes.push_back(router.route(std::span<const float>(
            queries.data() + query * inputs.dimensions, inputs.dimensions
        )));
    }

    struct Ranked final {
        std::size_t index{};
        float similarity{};
        [[nodiscard]] bool operator==(const Ranked&) const noexcept = default;
    };
    const auto reference_backend = frontier::make_frontier_backend(
        frontier::FrontierBackendKind::optimized_cpu
    );
    const auto batched_backend = frontier::make_frontier_backend(
        frontier::FrontierBackendKind::optimized_cpu
    );
    SparseRerankBatchAblation report;
    report.inputs = inputs;
    report.results_identical = true;
    report.result_hash = 14'695'981'039'346'656'037ULL;
    using Clock = std::chrono::steady_clock;
    for (std::size_t repetition = 0U; repetition < inputs.repetitions; ++repetition) {
        std::vector<Ranked> expected;
        std::vector<Ranked> actual;
        const auto run_reference = [&]() {
            const auto start = Clock::now();
            expected.clear();
            expected.reserve(inputs.query_count);
            for (std::size_t query = 0U; query < inputs.query_count; ++query) {
                const auto& indices = routes[query].candidate_indices;
                std::vector<float> candidates;
                candidates.reserve(indices.size() * inputs.dimensions);
                for (const std::size_t index : indices) {
                    candidates.insert(
                        candidates.end(),
                        modes.begin() + static_cast<std::ptrdiff_t>(index * inputs.dimensions),
                        modes.begin() + static_cast<std::ptrdiff_t>((index + 1U) * inputs.dimensions)
                    );
                }
                const std::vector<float> scores = reference_backend->batch_cosine(
                    std::span<const float>(
                        queries.data() + query * inputs.dimensions, inputs.dimensions
                    ),
                    1U, candidates, indices.size(), inputs.dimensions
                );
                std::size_t best = 0U;
                for (std::size_t candidate = 1U; candidate < scores.size(); ++candidate) {
                    if (scores[candidate] > scores[best] ||
                        (scores[candidate] == scores[best] &&
                         indices[candidate] < indices[best])) best = candidate;
                }
                expected.push_back(Ranked{indices[best], scores[best]});
            }
            report.per_query_seconds +=
                std::chrono::duration<double>(Clock::now() - start).count();
        };
        const auto run_batched = [&]() {
            const auto start = Clock::now();
            std::vector<float> candidates;
            std::vector<std::size_t> candidate_queries;
            std::vector<std::size_t> candidate_indices;
            std::vector<std::size_t> offsets;
            offsets.reserve(inputs.query_count + 1U);
            offsets.push_back(0U);
            for (std::size_t query = 0U; query < inputs.query_count; ++query) {
                for (const std::size_t index : routes[query].candidate_indices) {
                    candidates.insert(
                        candidates.end(),
                        modes.begin() + static_cast<std::ptrdiff_t>(index * inputs.dimensions),
                        modes.begin() + static_cast<std::ptrdiff_t>((index + 1U) * inputs.dimensions)
                    );
                    candidate_queries.push_back(query);
                    candidate_indices.push_back(index);
                }
                offsets.push_back(candidate_indices.size());
            }
            const std::vector<float> scores = batched_backend->batch_cosine_indexed(
                queries, inputs.query_count, candidates, candidate_queries,
                inputs.dimensions
            );
            actual.clear();
            actual.reserve(inputs.query_count);
            for (std::size_t query = 0U; query < inputs.query_count; ++query) {
                std::size_t best = offsets[query];
                for (std::size_t pair = best + 1U; pair < offsets[query + 1U]; ++pair) {
                    if (scores[pair] > scores[best] ||
                        (scores[pair] == scores[best] &&
                         candidate_indices[pair] < candidate_indices[best])) best = pair;
                }
                actual.push_back(Ranked{candidate_indices[best], scores[best]});
            }
            report.batched_seconds +=
                std::chrono::duration<double>(Clock::now() - start).count();
        };
        if ((repetition & 1U) == 0U) {
            run_reference();
            run_batched();
        } else {
            run_batched();
            run_reference();
        }
        report.results_identical = report.results_identical && expected == actual;
        for (const Ranked& ranked : actual) {
            report.result_hash ^= static_cast<std::uint64_t>(ranked.index);
            report.result_hash *= 1'099'511'628'211ULL;
            report.result_hash ^= std::bit_cast<std::uint32_t>(ranked.similarity);
            report.result_hash *= 1'099'511'628'211ULL;
        }
    }
    report.speedup = report.batched_seconds > 0.0
        ? report.per_query_seconds / report.batched_seconds : 0.0;
    report.per_query_backend_calls = reference_backend->operation_stats().batch_cosine_calls;
    report.batched_backend_calls = batched_backend->operation_stats().batch_cosine_calls;
    report.cosine_pairs = batched_backend->operation_stats().indexed_cosine_pairs;
    return report;
}

void write_sparse_rerank_batch_ablation_json(
    std::ostream& output,
    const SparseRerankBatchAblation& report
) {
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"rlf-sparse-rerank-batch-ablation-v1\",\n"
           << "  \"scope\": \"CPU preparation-host sparse reranking only; not full training, GPU-hour, or external-quality evidence\",\n"
           << "  \"vector_count\": " << report.inputs.vector_count << ",\n"
           << "  \"query_count\": " << report.inputs.query_count << ",\n"
           << "  \"dimensions\": " << report.inputs.dimensions << ",\n"
           << "  \"maximum_candidates\": " << report.inputs.maximum_candidates << ",\n"
           << "  \"repetitions\": " << report.inputs.repetitions << ",\n"
           << "  \"per_query_seconds\": " << report.per_query_seconds << ",\n"
           << "  \"batched_seconds\": " << report.batched_seconds << ",\n"
           << "  \"speedup\": " << report.speedup << ",\n"
           << "  \"results_identical\": "
           << (report.results_identical ? "true" : "false") << ",\n"
           << "  \"result_hash\": " << report.result_hash << ",\n"
           << "  \"per_query_backend_calls\": "
           << report.per_query_backend_calls << ",\n"
           << "  \"batched_backend_calls\": "
           << report.batched_backend_calls << ",\n"
           << "  \"cosine_pairs\": " << report.cosine_pairs << ",\n"
           << "  \"capacity_changed\": false,\n"
           << "  \"candidate_limit_changed\": false,\n"
           << "  \"external_quality_evidence_eligible\": false,\n"
           << "  \"gpu_efficiency_evidence_eligible\": false\n"
           << "}\n";
}

ConceptUpdateAblation measure_concept_update_ablation(
    const ConceptUpdateAblationInputs& inputs
) {
    if (inputs.images == 0U || inputs.width == 0U || inputs.height == 0U ||
        inputs.concepts == 0U || inputs.repetitions == 0U ||
        inputs.images > 10'000U || inputs.width > 1'024U ||
        inputs.height > 1'024U || inputs.concepts > 1'024U ||
        inputs.repetitions > 1'000U ||
        inputs.width > std::numeric_limits<std::size_t>::max() / inputs.height ||
        inputs.width * inputs.height >
            std::numeric_limits<std::size_t>::max() / 3U) {
        throw std::invalid_argument("invalid concept-update ablation inputs");
    }
    std::vector<ImageData> images;
    images.reserve(inputs.images);
    for (std::size_t image = 0U; image < inputs.images; ++image) {
        images.push_back(ablation_image(image, inputs.width, inputs.height));
    }
    std::string caption;
    for (std::size_t concept_index = 0U;
         concept_index < inputs.concepts;
         ++concept_index) {
        if (!caption.empty()) caption.push_back(' ');
        caption += "concept_" + std::to_string(concept_index);
    }
    VisionConfig config;
    config.patch_size = 8U;
    config.patch_sizes = {8U, 16U, 32U};
    config.descriptor_dimensions = 32U;
    config.maximum_input_side = std::max(inputs.width, inputs.height);
    config.maximum_patches = 4'096U;
    config.maximum_modes = inputs.images * 128U + 1U;
    config.maximum_examples = inputs.images + 1U;
    config.maximum_concepts_per_mode = inputs.concepts;
    config.training_patch_batch = 256U;
    config.retrieval_query_batch = 128U;
    config.retrieval_candidate_batch = 256U;
    config.sparse_routing_minimum_modes = config.maximum_modes;
    config.mode_creation_similarity = 0.99999;

    ConceptUpdateAblation report;
    report.inputs = inputs;
    report.model_states_identical = true;
    using Clock = std::chrono::steady_clock;
    for (std::size_t repetition = 0U; repetition < inputs.repetitions; ++repetition) {
        VisualPatchFabric linear(config);
        VisualPatchFabric indexed(config);
        linear.set_indexed_concept_updates(false);
        indexed.set_indexed_concept_updates(true);
        const auto run_linear = [&]() {
            const auto start = Clock::now();
            for (const ImageData& image : images) linear.train(image, caption);
            report.linear_seconds +=
                std::chrono::duration<double>(Clock::now() - start).count();
        };
        const auto run_indexed = [&]() {
            const auto start = Clock::now();
            for (const ImageData& image : images) indexed.train(image, caption);
            report.indexed_seconds +=
                std::chrono::duration<double>(Clock::now() - start).count();
        };
        if ((repetition & 1U) == 0U) {
            run_linear();
            run_indexed();
        } else {
            run_indexed();
            run_linear();
        }
        const auto linear_stats = linear.training_operation_stats();
        const auto indexed_stats = indexed.training_operation_stats();
        report.model_states_identical = report.model_states_identical &&
            linear.deterministic_hash() == indexed.deterministic_hash() &&
            linear_stats.concept_update_lookups ==
                indexed_stats.concept_update_lookups;
        report.model_hash = indexed.deterministic_hash();
        report.concept_update_lookups += indexed_stats.concept_update_lookups;
        report.linear_concept_comparisons +=
            linear_stats.linear_concept_comparisons;
        report.indexed_concept_lookups += indexed_stats.indexed_concept_lookups;
    }
    report.speedup = report.indexed_seconds > 0.0
        ? report.linear_seconds / report.indexed_seconds : 0.0;
    return report;
}

void write_concept_update_ablation_json(
    std::ostream& output,
    const ConceptUpdateAblation& report
) {
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"rlf-concept-update-ablation-v1\",\n"
           << "  \"scope\": \"CPU preparation-host visual training with unchanged concept capacity and learned state; not GPU-hour or external-quality evidence\",\n"
           << "  \"images\": " << report.inputs.images << ",\n"
           << "  \"width\": " << report.inputs.width << ",\n"
           << "  \"height\": " << report.inputs.height << ",\n"
           << "  \"concepts\": " << report.inputs.concepts << ",\n"
           << "  \"repetitions\": " << report.inputs.repetitions << ",\n"
           << "  \"linear_seconds\": " << report.linear_seconds << ",\n"
           << "  \"indexed_seconds\": " << report.indexed_seconds << ",\n"
           << "  \"speedup\": " << report.speedup << ",\n"
           << "  \"model_states_identical\": "
           << (report.model_states_identical ? "true" : "false") << ",\n"
           << "  \"model_hash\": " << report.model_hash << ",\n"
           << "  \"concept_update_lookups\": "
           << report.concept_update_lookups << ",\n"
           << "  \"linear_concept_comparisons\": "
           << report.linear_concept_comparisons << ",\n"
           << "  \"indexed_concept_lookups\": "
           << report.indexed_concept_lookups << ",\n"
           << "  \"capacity_changed\": false,\n"
           << "  \"training_examples_changed\": false,\n"
           << "  \"learning_updates_changed\": false,\n"
           << "  \"external_quality_evidence_eligible\": false,\n"
           << "  \"gpu_efficiency_evidence_eligible\": false\n"
           << "}\n";
}

ExampleDuplicateAblation measure_example_duplicate_ablation(
    const ExampleDuplicateAblationInputs& inputs
) {
    if (inputs.examples == 0U || inputs.repetitions == 0U ||
        inputs.examples > 100'000U || inputs.repetitions > 1'000U) {
        throw std::invalid_argument("invalid example-duplicate ablation inputs");
    }
    std::vector<ImageData> images;
    std::vector<std::string> captions;
    images.reserve(inputs.examples);
    captions.reserve(inputs.examples);
    for (std::size_t index = 0U; index < inputs.examples; ++index) {
        images.push_back(ablation_image(index, 8U, 8U));
        captions.push_back("unique visual example " + std::to_string(index));
    }

    VisionConfig config;
    config.patch_size = 8U;
    config.patch_sizes = {8U};
    config.descriptor_dimensions = 32U;
    config.maximum_input_side = 8U;
    config.maximum_patches = 1U;
    config.maximum_modes = 1U;
    config.maximum_examples = inputs.examples;
    config.maximum_concepts_per_mode = 8U;
    config.training_patch_batch = 1U;
    config.retrieval_query_batch = 1U;
    config.retrieval_candidate_batch = 1U;
    config.sparse_routing_minimum_modes = 2U;

    ExampleDuplicateAblation report;
    report.inputs = inputs;
    report.model_states_identical = true;
    using Clock = std::chrono::steady_clock;
    for (std::size_t repetition = 0U; repetition < inputs.repetitions; ++repetition) {
        VisualPatchFabric linear(config);
        VisualPatchFabric indexed(config);
        linear.set_indexed_example_duplicate_lookup(false);
        indexed.set_indexed_example_duplicate_lookup(true);
        const auto train_all = [&](VisualPatchFabric& fabric) {
            for (std::size_t index = 0U; index < inputs.examples; ++index) {
                fabric.train(images[index], captions[index]);
            }
            fabric.train(images.front(), captions.front());
        };
        const auto run_linear = [&]() {
            const auto start = Clock::now();
            train_all(linear);
            report.linear_seconds +=
                std::chrono::duration<double>(Clock::now() - start).count();
        };
        const auto run_indexed = [&]() {
            const auto start = Clock::now();
            train_all(indexed);
            report.indexed_seconds +=
                std::chrono::duration<double>(Clock::now() - start).count();
        };
        if ((repetition & 1U) == 0U) {
            run_linear();
            run_indexed();
        } else {
            run_indexed();
            run_linear();
        }
        const auto linear_stats = linear.training_operation_stats();
        const auto indexed_stats = indexed.training_operation_stats();
        report.model_states_identical = report.model_states_identical &&
            linear.deterministic_hash() == indexed.deterministic_hash() &&
            linear_stats.example_duplicate_lookups ==
                indexed_stats.example_duplicate_lookups &&
            linear.examples().size() == indexed.examples().size();
        report.model_hash = indexed.deterministic_hash();
        report.example_duplicate_lookups +=
            indexed_stats.example_duplicate_lookups;
        report.linear_example_comparisons +=
            linear_stats.linear_example_comparisons;
        report.indexed_example_candidates +=
            indexed_stats.indexed_example_candidates;
        report.learned_examples = indexed.examples().size();
    }
    report.speedup = report.indexed_seconds > 0.0
        ? report.linear_seconds / report.indexed_seconds : 0.0;
    return report;
}

void write_example_duplicate_ablation_json(
    std::ostream& output,
    const ExampleDuplicateAblation& report
) {
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"rlf-example-duplicate-ablation-v1\",\n"
           << "  \"scope\": \"CPU preparation-host complete visual training with unchanged example capacity and learned state; not GPU-hour or external-quality evidence\",\n"
           << "  \"examples\": " << report.inputs.examples << ",\n"
           << "  \"repetitions\": " << report.inputs.repetitions << ",\n"
           << "  \"linear_seconds\": " << report.linear_seconds << ",\n"
           << "  \"indexed_seconds\": " << report.indexed_seconds << ",\n"
           << "  \"speedup\": " << report.speedup << ",\n"
           << "  \"model_states_identical\": "
           << (report.model_states_identical ? "true" : "false") << ",\n"
           << "  \"model_hash\": " << report.model_hash << ",\n"
           << "  \"example_duplicate_lookups\": "
           << report.example_duplicate_lookups << ",\n"
           << "  \"linear_example_comparisons\": "
           << report.linear_example_comparisons << ",\n"
           << "  \"indexed_example_candidates\": "
           << report.indexed_example_candidates << ",\n"
           << "  \"learned_examples\": " << report.learned_examples << ",\n"
           << "  \"capacity_changed\": false,\n"
           << "  \"training_examples_changed\": false,\n"
           << "  \"learning_updates_changed\": false,\n"
           << "  \"external_quality_evidence_eligible\": false,\n"
           << "  \"gpu_efficiency_evidence_eligible\": false\n"
           << "}\n";
}

ModeIdIndexAblation measure_mode_id_index_ablation(
    const ModeIdIndexAblationInputs& inputs
) {
    if (inputs.modes == 0U || inputs.images == 0U ||
        inputs.repetitions == 0U || inputs.modes > 1'000'000U ||
        inputs.images > 100'000U || inputs.repetitions > 1'000U) {
        throw std::invalid_argument("invalid mode-ID-index ablation inputs");
    }

    VisionConfig config;
    config.patch_size = 8U;
    config.patch_sizes = {8U};
    config.descriptor_dimensions = 32U;
    config.maximum_input_side = 8U;
    config.maximum_patches = 1U;
    config.maximum_modes = inputs.modes;
    config.maximum_examples = inputs.images;
    config.maximum_concepts_per_mode = 8U;
    config.training_patch_batch = 1U;
    config.retrieval_query_batch = 1U;
    config.retrieval_candidate_batch = 256U;
    config.sparse_routing_minimum_modes = 1U;
    config.sparse_router.signature_bits = 18U;
    config.sparse_router.maximum_candidates = 256U;
    config.sparse_router.probe_radius = 2U;

    VisionSnapshot initial;
    initial.config = config;
    initial.next_mode_id = static_cast<std::uint64_t>(inputs.modes) + 1U;
    initial.modes.reserve(inputs.modes);
    for (std::size_t mode = 0U; mode < inputs.modes; ++mode) {
        std::vector<float> prototype(config.descriptor_dimensions, 0.0F);
        for (std::size_t dimension = 0U;
             dimension < prototype.size();
             ++dimension) {
            const std::uint64_t mixed =
                (static_cast<std::uint64_t>(mode) + 17ULL) *
                (static_cast<std::uint64_t>(dimension) + 31ULL) *
                0x9e3779b97f4a7c15ULL;
            prototype[dimension] = static_cast<float>(
                static_cast<double>(mixed % 65'521ULL) / 32'760.5 - 1.0
            );
        }
        initial.modes.push_back(VisualMode{
            static_cast<std::uint64_t>(mode) + 1U,
            std::move(prototype),
            1U,
            {},
        });
    }
    std::vector<ImageData> images;
    std::vector<std::string> captions;
    images.reserve(inputs.images);
    captions.reserve(inputs.images);
    for (std::size_t image = 0U; image < inputs.images; ++image) {
        images.push_back(ablation_image(image, 8U, 8U));
        captions.push_back("mode index image " + std::to_string(image));
    }

    ModeIdIndexAblation report;
    report.inputs = inputs;
    report.model_states_identical = true;
    report.analyses_identical = true;
    using Clock = std::chrono::steady_clock;
    for (std::size_t repetition = 0U; repetition < inputs.repetitions; ++repetition) {
        VisualPatchFabric rebuild = VisualPatchFabric::from_snapshot(initial);
        VisualPatchFabric persistent = VisualPatchFabric::from_snapshot(initial);
        rebuild.set_persistent_mode_id_index(false);
        persistent.set_persistent_mode_id_index(true);
        rebuild.set_incremental_sparse_router_updates(true);
        persistent.set_incremental_sparse_router_updates(true);
        static_cast<void>(rebuild.analyze(images.front()));
        static_cast<void>(persistent.analyze(images.front()));
        const auto rebuild_before = rebuild.training_operation_stats();
        const auto persistent_before = persistent.training_operation_stats();
        std::vector<VisionAnalysis> rebuild_analyses;
        std::vector<VisionAnalysis> persistent_analyses;
        rebuild_analyses.reserve(inputs.images);
        persistent_analyses.reserve(inputs.images);
        const auto train_all = [&](
            VisualPatchFabric& fabric,
            std::vector<VisionAnalysis>& analyses
        ) {
            for (std::size_t image = 0U; image < inputs.images; ++image) {
                analyses.push_back(fabric.train_and_analyze(
                    images[image], captions[image]
                ));
            }
        };
        const auto run_rebuild = [&]() {
            const auto start = Clock::now();
            train_all(rebuild, rebuild_analyses);
            report.rebuild_seconds +=
                std::chrono::duration<double>(Clock::now() - start).count();
        };
        const auto run_persistent = [&]() {
            const auto start = Clock::now();
            train_all(persistent, persistent_analyses);
            report.persistent_seconds +=
                std::chrono::duration<double>(Clock::now() - start).count();
        };
        if ((repetition & 1U) == 0U) {
            run_rebuild();
            run_persistent();
        } else {
            run_persistent();
            run_rebuild();
        }
        const auto rebuild_after = rebuild.training_operation_stats();
        const auto persistent_after = persistent.training_operation_stats();
        const std::uint64_t rebuild_lookups =
            rebuild_after.mode_id_lookups - rebuild_before.mode_id_lookups;
        const std::uint64_t persistent_lookups =
            persistent_after.mode_id_lookups - persistent_before.mode_id_lookups;
        report.model_states_identical = report.model_states_identical &&
            rebuild.deterministic_hash() == persistent.deterministic_hash() &&
            rebuild_lookups == persistent_lookups;
        if (rebuild_analyses.size() != persistent_analyses.size()) {
            report.analyses_identical = false;
        } else {
            for (std::size_t image = 0U;
                 image < rebuild_analyses.size();
                 ++image) {
                report.analyses_identical = report.analyses_identical &&
                    same_analysis(
                        rebuild_analyses[image], persistent_analyses[image]
                    );
            }
        }
        report.model_hash = persistent.deterministic_hash();
        report.mode_id_lookups += persistent_lookups;
        report.rebuilt_entries +=
            rebuild_after.mode_id_index_entries_rebuilt -
            rebuild_before.mode_id_index_entries_rebuilt;
        report.persistent_rebuilt_entries +=
            persistent_after.mode_id_index_entries_rebuilt -
            persistent_before.mode_id_index_entries_rebuilt;
        report.incremental_inserts +=
            persistent_after.mode_id_index_incremental_inserts -
            persistent_before.mode_id_index_incremental_inserts;
        const std::uint64_t rebuild_region_lookups =
            rebuild_after.region_mode_id_lookups -
            rebuild_before.region_mode_id_lookups;
        const std::uint64_t persistent_region_lookups =
            persistent_after.region_mode_id_lookups -
            persistent_before.region_mode_id_lookups;
        report.model_states_identical = report.model_states_identical &&
            rebuild_region_lookups == persistent_region_lookups;
        report.region_mode_id_lookups += persistent_region_lookups;
        report.linear_region_mode_comparisons +=
            rebuild_after.linear_region_mode_comparisons -
            rebuild_before.linear_region_mode_comparisons;
        report.indexed_region_mode_lookups +=
            persistent_after.indexed_region_mode_lookups -
            persistent_before.indexed_region_mode_lookups;
    }
    report.speedup = report.persistent_seconds > 0.0
        ? report.rebuild_seconds / report.persistent_seconds : 0.0;
    return report;
}

void write_mode_id_index_ablation_json(
    std::ostream& output,
    const ModeIdIndexAblation& report
) {
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"rlf-mode-id-index-ablation-v1\",\n"
           << "  \"scope\": \"CPU preparation-host complete fused visual training and analysis from an identical large-mode snapshot; not GPU-hour or external-quality evidence\",\n"
           << "  \"modes\": " << report.inputs.modes << ",\n"
           << "  \"images\": " << report.inputs.images << ",\n"
           << "  \"repetitions\": " << report.inputs.repetitions << ",\n"
           << "  \"rebuild_seconds\": " << report.rebuild_seconds << ",\n"
           << "  \"persistent_seconds\": " << report.persistent_seconds << ",\n"
           << "  \"speedup\": " << report.speedup << ",\n"
           << "  \"model_states_identical\": "
           << (report.model_states_identical ? "true" : "false") << ",\n"
           << "  \"analyses_identical\": "
           << (report.analyses_identical ? "true" : "false") << ",\n"
           << "  \"model_hash\": " << report.model_hash << ",\n"
           << "  \"mode_id_lookups\": " << report.mode_id_lookups << ",\n"
           << "  \"rebuilt_entries\": " << report.rebuilt_entries << ",\n"
           << "  \"persistent_rebuilt_entries\": "
           << report.persistent_rebuilt_entries << ",\n"
           << "  \"incremental_inserts\": " << report.incremental_inserts << ",\n"
           << "  \"region_mode_id_lookups\": "
           << report.region_mode_id_lookups << ",\n"
           << "  \"linear_region_mode_comparisons\": "
           << report.linear_region_mode_comparisons << ",\n"
           << "  \"indexed_region_mode_lookups\": "
           << report.indexed_region_mode_lookups << ",\n"
           << "  \"capacity_changed\": false,\n"
           << "  \"training_examples_changed\": false,\n"
           << "  \"learning_updates_changed\": false,\n"
           << "  \"checkpoint_format_changed\": false,\n"
           << "  \"external_quality_evidence_eligible\": false,\n"
           << "  \"gpu_efficiency_evidence_eligible\": false\n"
           << "}\n";
}

GroundingIndexAblation measure_grounding_index_ablation(
    const GroundingIndexAblationInputs& inputs
) {
    if (inputs.initial_links == 0U || inputs.concepts_per_mode < 3U ||
        inputs.observations == 0U || inputs.queries == 0U ||
        inputs.repetitions == 0U || inputs.initial_links > 2'000'000U ||
        inputs.concepts_per_mode > 1'024U || inputs.observations > 100'000U ||
        inputs.queries > 100'000U || inputs.repetitions > 1'000U) {
        throw std::invalid_argument("invalid grounding-index ablation inputs");
    }

    GroundingSnapshot initial;
    initial.config.maximum_links = inputs.initial_links +
        inputs.observations * inputs.concepts_per_mode;
    initial.config.maximum_concepts = inputs.initial_links;
    initial.links.reserve(inputs.initial_links);
    const std::size_t mode_count =
        (inputs.initial_links + inputs.concepts_per_mode - 1U) /
        inputs.concepts_per_mode;
    const double positive = 1.0 + initial.config.smoothing;
    const double negative = initial.config.smoothing;
    const double confidence = std::clamp(
        1.0 / (1.0 + std::exp(-(
            std::log(positive / negative) + 0.15 * std::log1p(1.0)
        ))),
        0.0,
        1.0
    );
    for (std::size_t link = 0U; link < inputs.initial_links; ++link) {
        initial.links.push_back(GroundingLink{
            static_cast<std::uint64_t>(link / inputs.concepts_per_mode) + 1U,
            "concept_" + std::to_string(link % inputs.concepts_per_mode),
            1U,
            0U,
            confidence,
        });
    }
    const std::vector<std::string> positives{"concept_0", "concept_1"};
    const std::vector<std::string> negatives{"concept_2"};
    const std::vector<std::string> composed{"concept_0", "concept_1"};

    GroundingIndexAblation report;
    report.inputs = inputs;
    report.model_states_identical = true;
    report.query_results_identical = true;
    using Clock = std::chrono::steady_clock;
    for (std::size_t repetition = 0U; repetition < inputs.repetitions; ++repetition) {
        CrossModalGroundingFabric rebuild =
            CrossModalGroundingFabric::from_snapshot(initial);
        CrossModalGroundingFabric persistent =
            CrossModalGroundingFabric::from_snapshot(initial);
        rebuild.set_persistent_link_index(false);
        persistent.set_persistent_link_index(true);
        const auto rebuild_before = rebuild.operation_stats();
        const auto persistent_before = persistent.operation_stats();
        std::vector<std::vector<GroundingHit>> rebuild_hits;
        std::vector<std::vector<GroundingHit>> persistent_hits;
        rebuild_hits.reserve(inputs.queries * 3U);
        persistent_hits.reserve(inputs.queries * 3U);
        const auto run_workload = [&](CrossModalGroundingFabric& fabric,
                                      auto& collected_hits) {
            for (std::size_t observation = 0U;
                 observation < inputs.observations;
                 ++observation) {
                const std::array<std::uint64_t, 1U> modes{
                    static_cast<std::uint64_t>(observation % mode_count) + 1U
                };
                fabric.observe(modes, positives, negatives);
            }
            for (std::size_t query = 0U; query < inputs.queries; ++query) {
                collected_hits.push_back(fabric.concepts_for_mode(
                    static_cast<std::uint64_t>(query % mode_count) + 1U
                ));
                collected_hits.push_back(fabric.modes_for_concept(
                    "concept_" + std::to_string(query % inputs.concepts_per_mode)
                ));
                collected_hits.push_back(fabric.compose_concepts(composed));
            }
        };
        const auto run_rebuild = [&]() {
            const auto start = Clock::now();
            run_workload(rebuild, rebuild_hits);
            report.rebuild_seconds +=
                std::chrono::duration<double>(Clock::now() - start).count();
        };
        const auto run_persistent = [&]() {
            const auto start = Clock::now();
            run_workload(persistent, persistent_hits);
            report.persistent_seconds +=
                std::chrono::duration<double>(Clock::now() - start).count();
        };
        if ((repetition & 1U) == 0U) {
            run_rebuild();
            run_persistent();
        } else {
            run_persistent();
            run_rebuild();
        }
        report.model_states_identical = report.model_states_identical &&
            rebuild.deterministic_hash() == persistent.deterministic_hash();
        if (rebuild_hits.size() != persistent_hits.size()) {
            report.query_results_identical = false;
        } else {
            for (std::size_t query = 0U; query < rebuild_hits.size(); ++query) {
                report.query_results_identical = report.query_results_identical &&
                    same_grounding_hits(rebuild_hits[query], persistent_hits[query]);
            }
        }
        report.model_hash = persistent.deterministic_hash();
        const auto rebuild_after = rebuild.operation_stats();
        const auto persistent_after = persistent.operation_stats();
        const std::uint64_t rebuild_lookups =
            rebuild_after.link_lookups - rebuild_before.link_lookups;
        const std::uint64_t persistent_lookups =
            persistent_after.link_lookups - persistent_before.link_lookups;
        report.model_states_identical = report.model_states_identical &&
            rebuild_lookups == persistent_lookups;
        report.link_lookups += persistent_lookups;
        report.rebuilt_lookup_entries +=
            rebuild_after.full_lookup_entries_rebuilt -
            rebuild_before.full_lookup_entries_rebuilt;
        report.indexed_link_candidates_examined +=
            persistent_after.indexed_link_candidates_examined -
            persistent_before.indexed_link_candidates_examined;
        report.full_confidence_sweep_entries +=
            rebuild_after.full_confidence_sweep_entries -
            rebuild_before.full_confidence_sweep_entries;
        report.sparse_confidence_recomputations +=
            persistent_after.confidence_recomputations -
            persistent_before.confidence_recomputations;
        report.derived_sort_entries +=
            rebuild_after.derived_sort_entries - rebuild_before.derived_sort_entries;
        report.query_full_scan_entries +=
            rebuild_after.mode_query_full_scan_entries -
                rebuild_before.mode_query_full_scan_entries +
            rebuild_after.concept_query_full_scan_entries -
                rebuild_before.concept_query_full_scan_entries;
        report.query_indexed_candidates +=
            persistent_after.mode_query_indexed_candidates -
                persistent_before.mode_query_indexed_candidates +
            persistent_after.concept_query_indexed_candidates -
                persistent_before.concept_query_indexed_candidates;
    }
    report.speedup = report.persistent_seconds > 0.0
        ? report.rebuild_seconds / report.persistent_seconds
        : 0.0;
    return report;
}

void write_grounding_index_ablation_json(
    std::ostream& output,
    const GroundingIndexAblation& report
) {
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"rlf-grounding-index-ablation-v1\",\n"
           << "  \"scope\": \"CPU preparation-host cross-modal grounding observation and exact recall from an identical learned-link snapshot; not GPU-hour or external-quality evidence\",\n"
           << "  \"initial_links\": " << report.inputs.initial_links << ",\n"
           << "  \"concepts_per_mode\": " << report.inputs.concepts_per_mode << ",\n"
           << "  \"observations\": " << report.inputs.observations << ",\n"
           << "  \"queries\": " << report.inputs.queries << ",\n"
           << "  \"repetitions\": " << report.inputs.repetitions << ",\n"
           << "  \"rebuild_seconds\": " << report.rebuild_seconds << ",\n"
           << "  \"persistent_seconds\": " << report.persistent_seconds << ",\n"
           << "  \"speedup\": " << report.speedup << ",\n"
           << "  \"model_states_identical\": "
           << (report.model_states_identical ? "true" : "false") << ",\n"
           << "  \"query_results_identical\": "
           << (report.query_results_identical ? "true" : "false") << ",\n"
           << "  \"model_hash\": " << report.model_hash << ",\n"
           << "  \"link_lookups\": " << report.link_lookups << ",\n"
           << "  \"rebuilt_lookup_entries\": "
           << report.rebuilt_lookup_entries << ",\n"
           << "  \"indexed_link_candidates_examined\": "
           << report.indexed_link_candidates_examined << ",\n"
           << "  \"full_confidence_sweep_entries\": "
           << report.full_confidence_sweep_entries << ",\n"
           << "  \"sparse_confidence_recomputations\": "
           << report.sparse_confidence_recomputations << ",\n"
           << "  \"derived_sort_entries\": "
           << report.derived_sort_entries << ",\n"
           << "  \"query_full_scan_entries\": "
           << report.query_full_scan_entries << ",\n"
           << "  \"query_indexed_candidates\": "
           << report.query_indexed_candidates << ",\n"
           << "  \"capacity_changed\": false,\n"
           << "  \"training_examples_changed\": false,\n"
           << "  \"learning_updates_changed\": false,\n"
           << "  \"checkpoint_format_changed\": false,\n"
           << "  \"external_quality_evidence_eligible\": false,\n"
           << "  \"gpu_efficiency_evidence_eligible\": false\n"
           << "}\n";
}

LanguageOutcomeIndexAblation measure_language_outcome_index_ablation(
    const LanguageOutcomeIndexAblationInputs& inputs
) {
    if (inputs.outcomes < 32U || inputs.updates == 0U ||
        inputs.repetitions == 0U || inputs.outcomes > 1'000'000U ||
        inputs.updates > 10'000'000U || inputs.repetitions > 1'000U ||
        inputs.outcomes >= std::numeric_limits<TokenId>::max()) {
        throw std::invalid_argument("invalid language-outcome-index inputs");
    }
    HierarchicalLanguageSnapshot initial;
    initial.config.context_orders = {0U};
    initial.config.maximum_contexts = 1U;
    initial.config.maximum_episodes = 1U;
    initial.config.prediction_candidate_limit = 128U;
    initial.next_context_id = 2U;
    initial.tokens_seen = inputs.outcomes;
    PredictiveContext context;
    context.id = 1U;
    context.support = inputs.outcomes;
    context.outcomes.reserve(inputs.outcomes);
    for (std::size_t outcome = 0U; outcome < inputs.outcomes; ++outcome) {
        context.outcomes.push_back(TokenOutcome{
            static_cast<TokenId>(outcome + 1U), 1U
        });
    }
    initial.contexts.push_back(std::move(context));

    LanguageOutcomeIndexAblation report;
    report.inputs = inputs;
    report.model_states_identical = true;
    report.predictions_identical = true;
    using Clock = std::chrono::steady_clock;
    const std::size_t late_window = std::min<std::size_t>(inputs.outcomes, 4'096U);
    for (std::size_t repetition = 0U; repetition < inputs.repetitions; ++repetition) {
        HierarchicalLanguageFabric linear =
            HierarchicalLanguageFabric::from_snapshot(initial);
        HierarchicalLanguageFabric indexed =
            HierarchicalLanguageFabric::from_snapshot(initial);
        linear.set_indexed_outcome_updates(false);
        indexed.set_indexed_outcome_updates(true);
        const auto linear_before = linear.training_operation_stats();
        const auto indexed_before = indexed.training_operation_stats();
        const auto train_all = [&](HierarchicalLanguageFabric& fabric) {
            for (std::size_t update = 0U; update < inputs.updates; ++update) {
                const TokenId target = static_cast<TokenId>(
                    inputs.outcomes - (update % late_window)
                );
                const std::array<TokenId, 2U> sequence{0U, target};
                fabric.train_token_sequence(sequence);
            }
        };
        const auto run_linear = [&]() {
            const auto start = Clock::now();
            train_all(linear);
            report.linear_seconds +=
                std::chrono::duration<double>(Clock::now() - start).count();
        };
        const auto run_indexed = [&]() {
            const auto start = Clock::now();
            train_all(indexed);
            report.indexed_seconds +=
                std::chrono::duration<double>(Clock::now() - start).count();
        };
        if ((repetition & 1U) == 0U) {
            run_linear();
            run_indexed();
        } else {
            run_indexed();
            run_linear();
        }
        report.model_states_identical = report.model_states_identical &&
            linear.deterministic_hash() == indexed.deterministic_hash();
        report.predictions_identical = report.predictions_identical &&
            same_prediction(linear.predict_next({}), indexed.predict_next({}));
        report.model_hash = indexed.deterministic_hash();
        const auto linear_after = linear.training_operation_stats();
        const auto indexed_after = indexed.training_operation_stats();
        const std::uint64_t linear_lookups =
            linear_after.outcome_update_lookups -
            linear_before.outcome_update_lookups;
        const std::uint64_t indexed_lookups =
            indexed_after.outcome_update_lookups -
            indexed_before.outcome_update_lookups;
        report.model_states_identical = report.model_states_identical &&
            linear_lookups == indexed_lookups;
        report.outcome_update_lookups += indexed_lookups;
        report.linear_outcome_comparisons +=
            linear_after.linear_outcome_comparisons -
            linear_before.linear_outcome_comparisons;
        report.indexed_outcome_lookups +=
            indexed_after.indexed_outcome_lookups -
            indexed_before.indexed_outcome_lookups;
    }
    report.speedup = report.indexed_seconds > 0.0
        ? report.linear_seconds / report.indexed_seconds
        : 0.0;
    return report;
}

void write_language_outcome_index_ablation_json(
    std::ostream& output,
    const LanguageOutcomeIndexAblation& report
) {
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"rlf-language-outcome-index-ablation-v1\",\n"
           << "  \"scope\": \"CPU preparation-host exact variable-order language outcome-count training from an identical high-cardinality context snapshot; not GPU-hour or external-quality evidence\",\n"
           << "  \"outcomes\": " << report.inputs.outcomes << ",\n"
           << "  \"updates\": " << report.inputs.updates << ",\n"
           << "  \"repetitions\": " << report.inputs.repetitions << ",\n"
           << "  \"linear_seconds\": " << report.linear_seconds << ",\n"
           << "  \"indexed_seconds\": " << report.indexed_seconds << ",\n"
           << "  \"speedup\": " << report.speedup << ",\n"
           << "  \"model_states_identical\": "
           << (report.model_states_identical ? "true" : "false") << ",\n"
           << "  \"predictions_identical\": "
           << (report.predictions_identical ? "true" : "false") << ",\n"
           << "  \"model_hash\": " << report.model_hash << ",\n"
           << "  \"outcome_update_lookups\": "
           << report.outcome_update_lookups << ",\n"
           << "  \"linear_outcome_comparisons\": "
           << report.linear_outcome_comparisons << ",\n"
           << "  \"indexed_outcome_lookups\": "
           << report.indexed_outcome_lookups << ",\n"
           << "  \"capacity_changed\": false,\n"
           << "  \"training_examples_changed\": false,\n"
           << "  \"learning_updates_changed\": false,\n"
           << "  \"checkpoint_format_changed\": false,\n"
           << "  \"external_quality_evidence_eligible\": false,\n"
           << "  \"gpu_efficiency_evidence_eligible\": false\n"
           << "}\n";
}

PreferenceDuplicateIndexAblation measure_preference_duplicate_index_ablation(
    const PreferenceDuplicateIndexAblationInputs& inputs
) {
    if (inputs.preferences == 0U || inputs.updates == 0U ||
        inputs.repetitions == 0U || inputs.preferences > 1'000'000U ||
        inputs.updates > 10'000'000U || inputs.repetitions > 1'000U) {
        throw std::invalid_argument("invalid preference-duplicate-index inputs");
    }
    GeneralFabricSnapshot initial;
    initial.config.maximum_preferences = inputs.preferences;
    initial.next_preference_id = static_cast<std::uint64_t>(inputs.preferences) + 1U;
    initial.preferences.reserve(inputs.preferences);
    for (std::size_t item = 0U; item < inputs.preferences; ++item) {
        const std::string suffix = std::to_string(item);
        initial.preferences.push_back(PreferenceExample{
            static_cast<std::uint64_t>(item) + 1U,
            "preference prompt " + suffix,
            "chosen response " + suffix,
            "rejected response " + suffix,
            {},
            {}, {}, {},
            1.0,
        });
    }
    const std::size_t late_window =
        std::min<std::size_t>(inputs.preferences, 4'096U);
    std::vector<std::string> prompts;
    std::vector<std::string> chosen;
    std::vector<std::string> rejected;
    prompts.reserve(late_window);
    chosen.reserve(late_window);
    rejected.reserve(late_window);
    for (std::size_t offset = 0U; offset < late_window; ++offset) {
        const std::string suffix = std::to_string(inputs.preferences - offset - 1U);
        prompts.push_back("preference prompt " + suffix);
        chosen.push_back("chosen response " + suffix);
        rejected.push_back("rejected response " + suffix);
    }

    PreferenceDuplicateIndexAblation report;
    report.inputs = inputs;
    report.model_states_identical = true;
    report.scores_identical = true;
    using Clock = std::chrono::steady_clock;
    for (std::size_t repetition = 0U; repetition < inputs.repetitions; ++repetition) {
        GeneralInstructionFabric linear =
            GeneralInstructionFabric::from_snapshot(initial);
        GeneralInstructionFabric indexed =
            GeneralInstructionFabric::from_snapshot(initial);
        linear.set_indexed_preference_duplicates(false);
        indexed.set_indexed_preference_duplicates(true);
        const auto linear_before = linear.training_operation_stats();
        const auto indexed_before = indexed.training_operation_stats();
        const auto train_all = [&](GeneralInstructionFabric& fabric) {
            for (std::size_t update = 0U; update < inputs.updates; ++update) {
                const std::size_t offset = update % late_window;
                static_cast<void>(fabric.train_preference(
                    prompts[offset], chosen[offset], rejected[offset], {}, 0.25
                ));
            }
        };
        const auto run_linear = [&]() {
            const auto start = Clock::now();
            train_all(linear);
            report.linear_seconds +=
                std::chrono::duration<double>(Clock::now() - start).count();
        };
        const auto run_indexed = [&]() {
            const auto start = Clock::now();
            train_all(indexed);
            report.indexed_seconds +=
                std::chrono::duration<double>(Clock::now() - start).count();
        };
        if ((repetition & 1U) == 0U) {
            run_linear();
            run_indexed();
        } else {
            run_indexed();
            run_linear();
        }
        report.model_states_identical = report.model_states_identical &&
            linear.deterministic_hash() == indexed.deterministic_hash();
        const std::vector<GeneralRetrievalMatch> no_matches;
        report.scores_identical = report.scores_identical &&
            linear.score_response(prompts.front(), chosen.front(), no_matches) ==
            indexed.score_response(prompts.front(), chosen.front(), no_matches);
        report.model_hash = indexed.deterministic_hash();
        const auto linear_after = linear.training_operation_stats();
        const auto indexed_after = indexed.training_operation_stats();
        const std::uint64_t linear_lookups =
            linear_after.preference_duplicate_lookups -
            linear_before.preference_duplicate_lookups;
        const std::uint64_t indexed_lookups =
            indexed_after.preference_duplicate_lookups -
            indexed_before.preference_duplicate_lookups;
        report.model_states_identical = report.model_states_identical &&
            linear_lookups == indexed_lookups;
        report.preference_duplicate_lookups += indexed_lookups;
        report.linear_preference_comparisons +=
            linear_after.linear_preference_comparisons -
            linear_before.linear_preference_comparisons;
        report.indexed_preference_candidates +=
            indexed_after.indexed_preference_candidates -
            indexed_before.indexed_preference_candidates;
    }
    report.speedup = report.indexed_seconds > 0.0
        ? report.linear_seconds / report.indexed_seconds
        : 0.0;
    return report;
}

void write_preference_duplicate_index_ablation_json(
    std::ostream& output,
    const PreferenceDuplicateIndexAblation& report
) {
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"rlf-preference-duplicate-index-ablation-v1\",\n"
           << "  \"scope\": \"CPU preparation-host exact preference duplicate-update training from an identical retained-preference snapshot; not GPU-hour or external-quality evidence\",\n"
           << "  \"preferences\": " << report.inputs.preferences << ",\n"
           << "  \"updates\": " << report.inputs.updates << ",\n"
           << "  \"repetitions\": " << report.inputs.repetitions << ",\n"
           << "  \"linear_seconds\": " << report.linear_seconds << ",\n"
           << "  \"indexed_seconds\": " << report.indexed_seconds << ",\n"
           << "  \"speedup\": " << report.speedup << ",\n"
           << "  \"model_states_identical\": "
           << (report.model_states_identical ? "true" : "false") << ",\n"
           << "  \"scores_identical\": "
           << (report.scores_identical ? "true" : "false") << ",\n"
           << "  \"model_hash\": " << report.model_hash << ",\n"
           << "  \"preference_duplicate_lookups\": "
           << report.preference_duplicate_lookups << ",\n"
           << "  \"linear_preference_comparisons\": "
           << report.linear_preference_comparisons << ",\n"
           << "  \"indexed_preference_candidates\": "
           << report.indexed_preference_candidates << ",\n"
           << "  \"capacity_changed\": false,\n"
           << "  \"training_examples_changed\": false,\n"
           << "  \"learning_updates_changed\": false,\n"
           << "  \"checkpoint_format_changed\": false,\n"
           << "  \"external_quality_evidence_eligible\": false,\n"
           << "  \"gpu_efficiency_evidence_eligible\": false\n"
           << "}\n";
}

ActiveLearningDuplicateIndexAblation
measure_active_learning_duplicate_index_ablation(
    const ActiveLearningDuplicateIndexAblationInputs& inputs
) {
    if (inputs.items == 0U || inputs.updates == 0U ||
        inputs.repetitions == 0U || inputs.items > 1'000'000U ||
        inputs.updates > 10'000'000U || inputs.repetitions > 1'000U) {
        throw std::invalid_argument(
            "invalid active-learning-duplicate-index inputs"
        );
    }
    GeneralFabricSnapshot initial;
    initial.config.maximum_active_learning_items = inputs.items;
    initial.config.active_learning_uncertainty = 0.0;
    initial.next_active_learning_id =
        static_cast<std::uint64_t>(inputs.items) + 1U;
    initial.active_learning_items.reserve(inputs.items);
    for (std::size_t item = 0U; item < inputs.items; ++item) {
        const std::string suffix = std::to_string(item);
        initial.active_learning_items.push_back(ActiveLearningItem{
            static_cast<std::uint64_t>(item) + 1U,
            "active prompt " + suffix,
            "active grounding " + suffix,
            0.6,
            1U,
        });
    }
    const std::size_t late_window =
        std::min<std::size_t>(inputs.items, 4'096U);
    std::vector<std::string> prompts;
    std::vector<std::string> grounding;
    prompts.reserve(late_window);
    grounding.reserve(late_window);
    for (std::size_t offset = 0U; offset < late_window; ++offset) {
        const std::string suffix = std::to_string(inputs.items - offset - 1U);
        prompts.push_back("active prompt " + suffix);
        grounding.push_back("active grounding " + suffix);
    }

    ActiveLearningDuplicateIndexAblation report;
    report.inputs = inputs;
    report.model_states_identical = true;
    using Clock = std::chrono::steady_clock;
    for (std::size_t repetition = 0U; repetition < inputs.repetitions; ++repetition) {
        GeneralInstructionFabric linear =
            GeneralInstructionFabric::from_snapshot(initial);
        GeneralInstructionFabric indexed =
            GeneralInstructionFabric::from_snapshot(initial);
        linear.set_indexed_active_learning_duplicates(false);
        indexed.set_indexed_active_learning_duplicates(true);
        const auto linear_before = linear.training_operation_stats();
        const auto indexed_before = indexed.training_operation_stats();
        const auto update_all = [&](GeneralInstructionFabric& fabric) {
            for (std::size_t update = 0U; update < inputs.updates; ++update) {
                const std::size_t offset = update % late_window;
                static_cast<void>(fabric.observe_uncertain(
                    prompts[offset], grounding[offset], 0.9
                ));
            }
        };
        const auto run_linear = [&]() {
            const auto start = Clock::now();
            update_all(linear);
            report.linear_seconds +=
                std::chrono::duration<double>(Clock::now() - start).count();
        };
        const auto run_indexed = [&]() {
            const auto start = Clock::now();
            update_all(indexed);
            report.indexed_seconds +=
                std::chrono::duration<double>(Clock::now() - start).count();
        };
        if ((repetition & 1U) == 0U) {
            run_linear();
            run_indexed();
        } else {
            run_indexed();
            run_linear();
        }
        report.model_states_identical = report.model_states_identical &&
            linear.deterministic_hash() == indexed.deterministic_hash();
        report.model_hash = indexed.deterministic_hash();
        const auto linear_after = linear.training_operation_stats();
        const auto indexed_after = indexed.training_operation_stats();
        const std::uint64_t linear_lookups =
            linear_after.active_learning_duplicate_lookups -
            linear_before.active_learning_duplicate_lookups;
        const std::uint64_t indexed_lookups =
            indexed_after.active_learning_duplicate_lookups -
            indexed_before.active_learning_duplicate_lookups;
        report.model_states_identical = report.model_states_identical &&
            linear_lookups == indexed_lookups;
        report.active_learning_duplicate_lookups += indexed_lookups;
        report.linear_active_learning_comparisons +=
            linear_after.linear_active_learning_comparisons -
            linear_before.linear_active_learning_comparisons;
        report.indexed_active_learning_candidates +=
            indexed_after.indexed_active_learning_candidates -
            indexed_before.indexed_active_learning_candidates;
    }
    report.speedup = report.indexed_seconds > 0.0
        ? report.linear_seconds / report.indexed_seconds
        : 0.0;
    return report;
}

void write_active_learning_duplicate_index_ablation_json(
    std::ostream& output,
    const ActiveLearningDuplicateIndexAblation& report
) {
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"rlf-active-learning-duplicate-index-ablation-v1\",\n"
           << "  \"scope\": \"CPU preparation-host exact active-learning duplicate updates from an identical retained-item snapshot; not GPU-hour or external-quality evidence\",\n"
           << "  \"items\": " << report.inputs.items << ",\n"
           << "  \"updates\": " << report.inputs.updates << ",\n"
           << "  \"repetitions\": " << report.inputs.repetitions << ",\n"
           << "  \"linear_seconds\": " << report.linear_seconds << ",\n"
           << "  \"indexed_seconds\": " << report.indexed_seconds << ",\n"
           << "  \"speedup\": " << report.speedup << ",\n"
           << "  \"model_states_identical\": "
           << (report.model_states_identical ? "true" : "false") << ",\n"
           << "  \"model_hash\": " << report.model_hash << ",\n"
           << "  \"active_learning_duplicate_lookups\": "
           << report.active_learning_duplicate_lookups << ",\n"
           << "  \"linear_active_learning_comparisons\": "
           << report.linear_active_learning_comparisons << ",\n"
           << "  \"indexed_active_learning_candidates\": "
           << report.indexed_active_learning_candidates << ",\n"
           << "  \"capacity_changed\": false,\n"
           << "  \"training_examples_changed\": false,\n"
           << "  \"learning_updates_changed\": false,\n"
           << "  \"checkpoint_format_changed\": false,\n"
           << "  \"external_quality_evidence_eligible\": false,\n"
           << "  \"gpu_efficiency_evidence_eligible\": false\n"
           << "}\n";
}

InstructionDuplicatePrefilterAblation
measure_instruction_duplicate_prefilter_ablation(
    const InstructionDuplicatePrefilterAblationInputs& inputs
) {
    if (inputs.demonstrations == 0U || inputs.updates == 0U ||
        inputs.repetitions == 0U || inputs.demonstrations > 1'000'000U ||
        inputs.updates > 1'000'000U || inputs.repetitions > 1'000U) {
        throw std::invalid_argument(
            "invalid instruction-duplicate-prefilter inputs"
        );
    }
    GeneralFabricSnapshot initial;
    initial.config.maximum_demonstrations =
        inputs.demonstrations + inputs.updates;
    initial.config.maximum_retrieval_candidates =
        inputs.demonstrations + inputs.updates;
    initial.next_demonstration_id =
        static_cast<std::uint64_t>(inputs.demonstrations) + 1U;
    initial.demonstrations.reserve(inputs.demonstrations);
    for (std::size_t item = 0U; item < inputs.demonstrations; ++item) {
        const std::string suffix = std::to_string(item);
        const std::string prompt = "instruction prompt " + suffix;
        initial.demonstrations.push_back(InstructionDemonstration{
            static_cast<std::uint64_t>(item) + 1U,
            "task",
            "domain",
            prompt,
            "instruction rationale " + suffix,
            "instruction response " + suffix,
            GeneralInstructionFabric::make_signature(
                "task domain " + prompt
            ),
            1U,
            1.0,
        });
    }

    InstructionDuplicatePrefilterAblation report;
    report.inputs = inputs;
    report.model_states_identical = true;
    using Clock = std::chrono::steady_clock;
    for (std::size_t repetition = 0U; repetition < inputs.repetitions; ++repetition) {
        GeneralInstructionFabric reference =
            GeneralInstructionFabric::from_snapshot(initial);
        GeneralInstructionFabric indexed =
            GeneralInstructionFabric::from_snapshot(initial);
        reference.set_indexed_instruction_duplicates(false);
        indexed.set_indexed_instruction_duplicates(true);
        const auto reference_before = reference.training_operation_stats();
        const auto indexed_before = indexed.training_operation_stats();
        const auto train_all = [&](GeneralInstructionFabric& fabric) {
            for (std::size_t update = 0U; update < inputs.updates; ++update) {
                const std::string suffix = std::to_string(update);
                static_cast<void>(fabric.train_instruction(
                    "task", "domain", "new instruction prompt " + suffix,
                    "new instruction rationale " + suffix,
                    "new instruction response " + suffix, 1.0
                ));
            }
        };
        const auto run_reference = [&]() {
            const auto start = Clock::now();
            train_all(reference);
            report.retrieval_seconds +=
                std::chrono::duration<double>(Clock::now() - start).count();
        };
        const auto run_indexed = [&]() {
            const auto start = Clock::now();
            train_all(indexed);
            report.indexed_seconds +=
                std::chrono::duration<double>(Clock::now() - start).count();
        };
        if ((repetition & 1U) == 0U) {
            run_reference();
            run_indexed();
        } else {
            run_indexed();
            run_reference();
        }
        report.model_states_identical = report.model_states_identical &&
            reference.deterministic_hash() == indexed.deterministic_hash();
        report.model_hash = indexed.deterministic_hash();
        const auto reference_after = reference.training_operation_stats();
        const auto indexed_after = indexed.training_operation_stats();
        report.reference_retrievals +=
            reference_after.instruction_duplicate_retrievals -
            reference_before.instruction_duplicate_retrievals;
        report.indexed_retrievals +=
            indexed_after.instruction_duplicate_retrievals -
            indexed_before.instruction_duplicate_retrievals;
        report.indexed_retrievals_avoided +=
            indexed_after.instruction_duplicate_retrievals_avoided -
            indexed_before.instruction_duplicate_retrievals_avoided;
    }
    report.speedup = report.indexed_seconds > 0.0
        ? report.retrieval_seconds / report.indexed_seconds
        : 0.0;
    return report;
}

void write_instruction_duplicate_prefilter_ablation_json(
    std::ostream& output,
    const InstructionDuplicatePrefilterAblation& report
) {
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"rlf-instruction-duplicate-prefilter-ablation-v1\",\n"
           << "  \"scope\": \"CPU preparation-host exact unique-instruction training from an identical retained-demonstration snapshot; not GPU-hour or external-quality evidence\",\n"
           << "  \"demonstrations\": " << report.inputs.demonstrations << ",\n"
           << "  \"updates\": " << report.inputs.updates << ",\n"
           << "  \"repetitions\": " << report.inputs.repetitions << ",\n"
           << "  \"retrieval_seconds\": " << report.retrieval_seconds << ",\n"
           << "  \"indexed_seconds\": " << report.indexed_seconds << ",\n"
           << "  \"speedup\": " << report.speedup << ",\n"
           << "  \"model_states_identical\": "
           << (report.model_states_identical ? "true" : "false") << ",\n"
           << "  \"model_hash\": " << report.model_hash << ",\n"
           << "  \"reference_retrievals\": "
           << report.reference_retrievals << ",\n"
           << "  \"indexed_retrievals\": " << report.indexed_retrievals << ",\n"
           << "  \"indexed_retrievals_avoided\": "
           << report.indexed_retrievals_avoided << ",\n"
           << "  \"capacity_changed\": false,\n"
           << "  \"training_examples_changed\": false,\n"
           << "  \"learning_updates_changed\": false,\n"
           << "  \"checkpoint_format_changed\": false,\n"
           << "  \"external_quality_evidence_eligible\": false,\n"
           << "  \"gpu_efficiency_evidence_eligible\": false\n"
           << "}\n";
}

ToolKeywordIndexAblation measure_tool_keyword_index_ablation(
    const ToolKeywordIndexAblationInputs& inputs
) {
    if (inputs.keywords == 0U || inputs.updates == 0U ||
        inputs.repetitions == 0U || inputs.keywords > 1'000'000U ||
        inputs.updates > 10'000'000U || inputs.repetitions > 1'000U) {
        throw std::invalid_argument("invalid tool-keyword-index inputs");
    }
    ToolRouterSnapshot initial;
    initial.config.maximum_tools = 1U;
    initial.config.maximum_keywords_per_tool = inputs.keywords;
    ToolRoute route;
    route.tool_name = "example_tool";
    route.examples = inputs.keywords;
    route.keywords.reserve(inputs.keywords);
    for (std::size_t item = 0U; item < inputs.keywords; ++item) {
        route.keywords.push_back(ToolKeywordCount{
            "keyword_" + std::to_string(item), 1U
        });
    }
    initial.routes.push_back(std::move(route));
    const std::size_t late_window =
        std::min<std::size_t>(inputs.keywords, 4'096U);

    ToolKeywordIndexAblation report;
    report.inputs = inputs;
    report.model_states_identical = true;
    using Clock = std::chrono::steady_clock;
    for (std::size_t repetition = 0U; repetition < inputs.repetitions; ++repetition) {
        ToolRouter linear = ToolRouter::from_snapshot(initial);
        ToolRouter indexed = ToolRouter::from_snapshot(initial);
        linear.set_indexed_keyword_updates(false);
        indexed.set_indexed_keyword_updates(true);
        const auto linear_before = linear.training_operation_stats();
        const auto indexed_before = indexed.training_operation_stats();
        const auto train_all = [&](ToolRouter& router) {
            for (std::size_t update = 0U; update < inputs.updates; ++update) {
                const std::size_t item =
                    inputs.keywords - (update % late_window) - 1U;
                router.train(
                    "keyword_" + std::to_string(item), "example_tool"
                );
            }
        };
        const auto run_linear = [&]() {
            const auto start = Clock::now();
            train_all(linear);
            report.linear_seconds +=
                std::chrono::duration<double>(Clock::now() - start).count();
        };
        const auto run_indexed = [&]() {
            const auto start = Clock::now();
            train_all(indexed);
            report.indexed_seconds +=
                std::chrono::duration<double>(Clock::now() - start).count();
        };
        if ((repetition & 1U) == 0U) {
            run_linear();
            run_indexed();
        } else {
            run_indexed();
            run_linear();
        }
        report.model_states_identical = report.model_states_identical &&
            linear.deterministic_hash() == indexed.deterministic_hash();
        report.model_hash = indexed.deterministic_hash();
        const auto linear_after = linear.training_operation_stats();
        const auto indexed_after = indexed.training_operation_stats();
        const std::uint64_t linear_lookups =
            linear_after.keyword_update_lookups -
            linear_before.keyword_update_lookups;
        const std::uint64_t indexed_lookups =
            indexed_after.keyword_update_lookups -
            indexed_before.keyword_update_lookups;
        report.model_states_identical = report.model_states_identical &&
            linear_lookups == indexed_lookups;
        report.keyword_update_lookups += indexed_lookups;
        report.linear_keyword_comparisons +=
            linear_after.linear_keyword_comparisons -
            linear_before.linear_keyword_comparisons;
        report.indexed_keyword_lookups +=
            indexed_after.indexed_keyword_lookups -
            indexed_before.indexed_keyword_lookups;
        report.keyword_capacity_skips +=
            indexed_after.keyword_capacity_skips -
            indexed_before.keyword_capacity_skips;
    }
    report.speedup = report.indexed_seconds > 0.0
        ? report.linear_seconds / report.indexed_seconds
        : 0.0;
    return report;
}

void write_tool_keyword_index_ablation_json(
    std::ostream& output,
    const ToolKeywordIndexAblation& report
) {
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"rlf-tool-keyword-index-ablation-v1\",\n"
           << "  \"scope\": \"CPU preparation-host exact tool-keyword count updates from an identical retained-route snapshot; not GPU-hour or external-quality evidence\",\n"
           << "  \"keywords\": " << report.inputs.keywords << ",\n"
           << "  \"updates\": " << report.inputs.updates << ",\n"
           << "  \"repetitions\": " << report.inputs.repetitions << ",\n"
           << "  \"linear_seconds\": " << report.linear_seconds << ",\n"
           << "  \"indexed_seconds\": " << report.indexed_seconds << ",\n"
           << "  \"speedup\": " << report.speedup << ",\n"
           << "  \"model_states_identical\": "
           << (report.model_states_identical ? "true" : "false") << ",\n"
           << "  \"model_hash\": " << report.model_hash << ",\n"
           << "  \"keyword_update_lookups\": "
           << report.keyword_update_lookups << ",\n"
           << "  \"linear_keyword_comparisons\": "
           << report.linear_keyword_comparisons << ",\n"
           << "  \"indexed_keyword_lookups\": "
           << report.indexed_keyword_lookups << ",\n"
           << "  \"keyword_capacity_skips\": "
           << report.keyword_capacity_skips << ",\n"
           << "  \"capacity_changed\": false,\n"
           << "  \"training_examples_changed\": false,\n"
           << "  \"learning_updates_changed\": false,\n"
           << "  \"checkpoint_format_changed\": false,\n"
           << "  \"external_quality_evidence_eligible\": false,\n"
           << "  \"gpu_efficiency_evidence_eligible\": false\n"
           << "}\n";
}

DialogueEncodingAblation measure_dialogue_encoding_ablation(
    const DialogueEncodingAblationInputs& inputs
) {
    if (inputs.dialogues == 0U || inputs.repetitions == 0U ||
        inputs.dialogues > 1'000'000U || inputs.repetitions > 1'000U) {
        throw std::invalid_argument("invalid dialogue-encoding inputs");
    }
    SolsticeTokenizer tokenizer;
    tokenizer.train(
        "dialogue prompt response grounded context reusable evidence value"
    );
    HierarchicalLanguageConfig config;
    config.context_orders = {0U};
    config.maximum_contexts = 1U;
    config.maximum_episodes = inputs.dialogues;

    DialogueEncodingAblation report;
    report.inputs = inputs;
    report.model_states_identical = true;
    using Clock = std::chrono::steady_clock;
    for (std::size_t repetition = 0U; repetition < inputs.repetitions; ++repetition) {
        HierarchicalLanguageFabric redundant(config);
        HierarchicalLanguageFabric fused(config);
        redundant.set_fused_dialogue_encoding(false);
        fused.set_fused_dialogue_encoding(true);
        const auto train_all = [&](HierarchicalLanguageFabric& fabric) {
            for (std::size_t item = 0U; item < inputs.dialogues; ++item) {
                const std::string suffix = std::to_string(item);
                fabric.train_dialogue(
                    tokenizer,
                    "dialogue prompt " + suffix,
                    "reusable response",
                    "grounded context " + suffix
                );
            }
        };
        const auto run_redundant = [&]() {
            const auto start = Clock::now();
            train_all(redundant);
            report.redundant_seconds +=
                std::chrono::duration<double>(Clock::now() - start).count();
        };
        const auto run_fused = [&]() {
            const auto start = Clock::now();
            train_all(fused);
            report.fused_seconds +=
                std::chrono::duration<double>(Clock::now() - start).count();
        };
        if ((repetition & 1U) == 0U) {
            run_redundant();
            run_fused();
        } else {
            run_fused();
            run_redundant();
        }
        report.model_states_identical = report.model_states_identical &&
            redundant.deterministic_hash() == fused.deterministic_hash();
        report.model_hash = fused.deterministic_hash();
        const auto redundant_stats = redundant.training_operation_stats();
        const auto fused_stats = fused.training_operation_stats();
        report.redundant_encode_calls +=
            redundant_stats.dialogue_tokenizer_encode_calls;
        report.fused_encode_calls +=
            fused_stats.dialogue_tokenizer_encode_calls;
        report.encode_calls_avoided +=
            fused_stats.redundant_dialogue_encode_calls_avoided;
    }
    report.speedup = report.fused_seconds > 0.0
        ? report.redundant_seconds / report.fused_seconds
        : 0.0;
    return report;
}

void write_dialogue_encoding_ablation_json(
    std::ostream& output,
    const DialogueEncodingAblation& report
) {
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"rlf-dialogue-encoding-ablation-v1\",\n"
           << "  \"scope\": \"CPU preparation-host exact dialogue training with redundant versus fused prompt/grounding tokenization; not GPU-hour or external-quality evidence\",\n"
           << "  \"dialogues\": " << report.inputs.dialogues << ",\n"
           << "  \"repetitions\": " << report.inputs.repetitions << ",\n"
           << "  \"redundant_seconds\": " << report.redundant_seconds << ",\n"
           << "  \"fused_seconds\": " << report.fused_seconds << ",\n"
           << "  \"speedup\": " << report.speedup << ",\n"
           << "  \"model_states_identical\": "
           << (report.model_states_identical ? "true" : "false") << ",\n"
           << "  \"model_hash\": " << report.model_hash << ",\n"
           << "  \"redundant_encode_calls\": "
           << report.redundant_encode_calls << ",\n"
           << "  \"fused_encode_calls\": "
           << report.fused_encode_calls << ",\n"
           << "  \"encode_calls_avoided\": "
           << report.encode_calls_avoided << ",\n"
           << "  \"capacity_changed\": false,\n"
           << "  \"training_examples_changed\": false,\n"
           << "  \"learning_updates_changed\": false,\n"
           << "  \"checkpoint_format_changed\": false,\n"
           << "  \"external_quality_evidence_eligible\": false,\n"
           << "  \"gpu_efficiency_evidence_eligible\": false\n"
           << "}\n";
}

}  // namespace rlf::solstice
