#include "test_framework.hpp"

#include "rlf/solstice/checkpoint.hpp"
#include "rlf/solstice/frontier_gate.hpp"
#include "rlf/solstice/general_fabric.hpp"
#include "rlf/solstice/profile.hpp"
#include "rlf/solstice/solstice_model.hpp"

#include <array>
#include <filesystem>
#include <string>
#include <vector>

namespace {

[[nodiscard]] std::filesystem::path temporary_path(const std::string& name) {
    return std::filesystem::temp_directory_path() /
        ("rlf_general_frontier_" + name);
}

}  // namespace

RLF_TEST_CASE("General instruction fabric retrieves transferable solution patterns") {
    rlf::solstice::GeneralFabricConfig config;
    config.maximum_demonstrations = 32U;
    config.maximum_preferences = 32U;
    config.maximum_active_learning_items = 32U;
    config.maximum_retrieval_candidates = 32U;
    config.maximum_retrieved_demonstrations = 4U;
    rlf::solstice::GeneralInstructionFabric fabric(config);
    const std::uint64_t coding_id = fabric.train_instruction(
        "coding", "software",
        "A server crashes when an empty request reaches the parser.",
        "Reproduce the empty-input path, inspect the parser precondition, add a guard, and run regression tests.",
        "Validate input before parsing and add an empty-request regression test."
    );
    static_cast<void>(fabric.train_instruction(
        "mathematics", "math",
        "Find the area of a rectangle.",
        "Multiply width by height and preserve squared units.",
        "Use area = width times height."
    ));
    const auto matches = fabric.retrieve(
        "coding", "software",
        "The parser segfaults whenever the request body is empty."
    );
    RLF_CHECK(!matches.empty());
    RLF_CHECK(matches.front().demonstration_id == coding_id);
    const auto context = fabric.build_context(
        "coding", "software",
        "The parser segfaults whenever the request body is empty.", "", ""
    );
    RLF_CHECK(context.context.find("empty-input") != std::string::npos);
    RLF_CHECK(!context.plan_hint.empty());
}

RLF_TEST_CASE("General preference memory ranks chosen behavior above rejected behavior") {
    rlf::solstice::GeneralInstructionFabric fabric;
    fabric.train_preference(
        "Answer a question when the evidence is incomplete.",
        "State the uncertainty and identify the missing evidence.",
        "Invent a confident answer.",
        "Prefer calibrated uncertainty.",
        2.0
    );
    const std::vector<rlf::solstice::GeneralRetrievalMatch> no_matches;
    const double chosen = fabric.score_response(
        "Answer a question when the evidence is incomplete.",
        "State the uncertainty and identify the missing evidence.",
        no_matches
    );
    const double rejected = fabric.score_response(
        "Answer a question when the evidence is incomplete.",
        "Invent a confident answer.",
        no_matches
    );
    RLF_CHECK(chosen > rejected);
}

RLF_TEST_CASE("Indexed instruction duplicate prefilter preserves exact learning") {
    rlf::solstice::GeneralFabricConfig config;
    config.maximum_demonstrations = 256U;
    rlf::solstice::GeneralInstructionFabric reference(config);
    rlf::solstice::GeneralInstructionFabric indexed(config);
    reference.set_indexed_instruction_duplicates(false);
    indexed.set_indexed_instruction_duplicates(true);
    for (std::size_t item = 0U; item < 64U; ++item) {
        const std::string suffix = std::to_string(item);
        const auto reference_id = reference.train_instruction(
            "task", "domain", "unique prompt " + suffix,
            "rationale " + suffix, "response " + suffix, 1.0
        );
        const auto indexed_id = indexed.train_instruction(
            "task", "domain", "unique prompt " + suffix,
            "rationale " + suffix, "response " + suffix, 1.0
        );
        RLF_CHECK(reference_id == indexed_id);
        RLF_CHECK(reference.deterministic_hash() == indexed.deterministic_hash());
    }
    for (std::size_t item = 64U; item > 0U; --item) {
        const std::string suffix = std::to_string(item - 1U);
        const auto reference_id = reference.train_instruction(
            "task", "domain", "unique prompt " + suffix,
            "rationale " + suffix, "response " + suffix, 2.0
        );
        const auto indexed_id = indexed.train_instruction(
            "task", "domain", "unique prompt " + suffix,
            "rationale " + suffix, "response " + suffix, 2.0
        );
        RLF_CHECK(reference_id == indexed_id);
        RLF_CHECK(reference.deterministic_hash() == indexed.deterministic_hash());
    }

    auto resumed = rlf::solstice::GeneralInstructionFabric::from_snapshot(
        indexed.snapshot()
    );
    resumed.set_indexed_instruction_duplicates(true);
    indexed.train_instruction(
        "task", "domain", "unique prompt 63",
        "rationale 63", "response 63", 3.0
    );
    resumed.train_instruction(
        "task", "domain", "unique prompt 63",
        "rationale 63", "response 63", 3.0
    );
    RLF_CHECK(indexed.deterministic_hash() == resumed.deterministic_hash());

    const auto reference_stats = reference.training_operation_stats();
    const auto indexed_stats = indexed.training_operation_stats();
    RLF_CHECK(reference_stats.instruction_duplicate_prefilter_lookups == 128U);
    RLF_CHECK(reference_stats.instruction_duplicate_retrievals == 128U);
    RLF_CHECK(reference_stats.instruction_duplicate_retrievals_avoided == 0U);
    RLF_CHECK(indexed_stats.instruction_duplicate_prefilter_lookups == 129U);
    RLF_CHECK(indexed_stats.instruction_duplicate_retrievals == 65U);
    RLF_CHECK(indexed_stats.instruction_duplicate_retrievals_avoided == 64U);
    RLF_CHECK(indexed_stats.instruction_index_incremental_inserts ==
              indexed.demonstrations().size());
    RLF_CHECK(indexed_stats.instruction_index_incremental_inserts >= 64U);
}

RLF_TEST_CASE("Indexed preference duplicates preserve exact general learning") {
    rlf::solstice::GeneralFabricConfig config;
    config.maximum_preferences = 256U;
    rlf::solstice::GeneralInstructionFabric linear(config);
    rlf::solstice::GeneralInstructionFabric indexed(config);
    linear.set_indexed_preference_duplicates(false);
    indexed.set_indexed_preference_duplicates(true);
    for (std::size_t item = 0U; item < 64U; ++item) {
        const std::string suffix = std::to_string(item);
        linear.train_preference(
            "prompt " + suffix, "chosen " + suffix, "rejected " + suffix
        );
        indexed.train_preference(
            "prompt " + suffix, "chosen " + suffix, "rejected " + suffix
        );
        RLF_CHECK(linear.deterministic_hash() == indexed.deterministic_hash());
    }
    for (std::size_t item = 64U; item > 0U; --item) {
        const std::string suffix = std::to_string(item - 1U);
        linear.train_preference(
            "prompt " + suffix, "chosen " + suffix, "rejected " + suffix,
            "feedback", 0.5
        );
        indexed.train_preference(
            "prompt " + suffix, "chosen " + suffix, "rejected " + suffix,
            "feedback", 0.5
        );
        RLF_CHECK(linear.deterministic_hash() == indexed.deterministic_hash());
    }

    auto resumed = rlf::solstice::GeneralInstructionFabric::from_snapshot(
        indexed.snapshot()
    );
    resumed.set_indexed_preference_duplicates(true);
    indexed.train_preference("prompt 63", "chosen 63", "rejected 63", "", 2.0);
    resumed.train_preference("prompt 63", "chosen 63", "rejected 63", "", 2.0);
    RLF_CHECK(indexed.deterministic_hash() == resumed.deterministic_hash());

    auto duplicate_snapshot = indexed.snapshot();
    duplicate_snapshot.preferences.push_back(
        duplicate_snapshot.preferences.front()
    );
    duplicate_snapshot.preferences.back().id =
        duplicate_snapshot.next_preference_id++;
    auto duplicate_linear =
        rlf::solstice::GeneralInstructionFabric::from_snapshot(
            duplicate_snapshot
        );
    auto duplicate_indexed =
        rlf::solstice::GeneralInstructionFabric::from_snapshot(
            std::move(duplicate_snapshot)
        );
    duplicate_linear.set_indexed_preference_duplicates(false);
    duplicate_indexed.set_indexed_preference_duplicates(true);
    duplicate_linear.train_preference(
        "prompt 0", "chosen 0", "rejected 0", "", 3.0
    );
    duplicate_indexed.train_preference(
        "prompt 0", "chosen 0", "rejected 0", "", 3.0
    );
    RLF_CHECK(duplicate_linear.deterministic_hash() ==
              duplicate_indexed.deterministic_hash());

    const auto linear_stats = linear.training_operation_stats();
    const auto indexed_stats = indexed.training_operation_stats();
    RLF_CHECK(linear_stats.preference_duplicate_lookups == 128U);
    RLF_CHECK(linear_stats.linear_preference_comparisons > 0U);
    RLF_CHECK(linear_stats.indexed_preference_candidates == 0U);
    RLF_CHECK(indexed_stats.preference_duplicate_lookups > 128U);
    RLF_CHECK(indexed_stats.linear_preference_comparisons == 0U);
    RLF_CHECK(indexed_stats.indexed_preference_candidates > 0U);
    RLF_CHECK(indexed_stats.preference_index_incremental_inserts == 64U);
}

RLF_TEST_CASE("Indexed active learning duplicates preserve exact admission") {
    rlf::solstice::GeneralFabricConfig config;
    config.maximum_active_learning_items = 256U;
    config.active_learning_uncertainty = 0.0;
    rlf::solstice::GeneralInstructionFabric linear(config);
    rlf::solstice::GeneralInstructionFabric indexed(config);
    linear.set_indexed_active_learning_duplicates(false);
    indexed.set_indexed_active_learning_duplicates(true);
    for (std::size_t item = 0U; item < 64U; ++item) {
        const std::string suffix = std::to_string(item);
        RLF_CHECK(linear.observe_uncertain(
            "prompt " + suffix, "grounding " + suffix, 0.6
        ) == indexed.observe_uncertain(
            "prompt " + suffix, "grounding " + suffix, 0.6
        ));
        RLF_CHECK(linear.deterministic_hash() == indexed.deterministic_hash());
    }
    for (std::size_t item = 64U; item > 0U; --item) {
        const std::string suffix = std::to_string(item - 1U);
        RLF_CHECK(linear.observe_uncertain(
            "prompt " + suffix, "grounding " + suffix, 0.9
        ) == indexed.observe_uncertain(
            "prompt " + suffix, "grounding " + suffix, 0.9
        ));
        RLF_CHECK(linear.deterministic_hash() == indexed.deterministic_hash());
    }

    auto resumed = rlf::solstice::GeneralInstructionFabric::from_snapshot(
        indexed.snapshot()
    );
    resumed.set_indexed_active_learning_duplicates(true);
    indexed.observe_uncertain("prompt 63", "grounding 63", 1.0);
    resumed.observe_uncertain("prompt 63", "grounding 63", 1.0);
    RLF_CHECK(indexed.deterministic_hash() == resumed.deterministic_hash());

    auto duplicate_snapshot = indexed.snapshot();
    duplicate_snapshot.active_learning_items.push_back(
        duplicate_snapshot.active_learning_items.front()
    );
    duplicate_snapshot.active_learning_items.back().id =
        duplicate_snapshot.next_active_learning_id++;
    auto duplicate_linear =
        rlf::solstice::GeneralInstructionFabric::from_snapshot(
            duplicate_snapshot
        );
    auto duplicate_indexed =
        rlf::solstice::GeneralInstructionFabric::from_snapshot(
            std::move(duplicate_snapshot)
        );
    duplicate_linear.set_indexed_active_learning_duplicates(false);
    duplicate_indexed.set_indexed_active_learning_duplicates(true);
    duplicate_linear.observe_uncertain("prompt 0", "grounding 0", 1.0);
    duplicate_indexed.observe_uncertain("prompt 0", "grounding 0", 1.0);
    RLF_CHECK(duplicate_linear.deterministic_hash() ==
              duplicate_indexed.deterministic_hash());

    rlf::solstice::GeneralFabricConfig bounded_config;
    bounded_config.maximum_active_learning_items = 4U;
    bounded_config.active_learning_uncertainty = 0.0;
    rlf::solstice::GeneralInstructionFabric bounded_linear(bounded_config);
    rlf::solstice::GeneralInstructionFabric bounded_indexed(bounded_config);
    bounded_linear.set_indexed_active_learning_duplicates(false);
    bounded_indexed.set_indexed_active_learning_duplicates(true);
    for (std::size_t item = 0U; item < 4U; ++item) {
        const std::string suffix = std::to_string(item);
        const double uncertainty = 0.1 + 0.1 * static_cast<double>(item);
        bounded_linear.observe_uncertain(
            "bounded " + suffix, "grounding " + suffix, uncertainty
        );
        bounded_indexed.observe_uncertain(
            "bounded " + suffix, "grounding " + suffix, uncertainty
        );
    }
    bounded_linear.observe_uncertain("replacement", "new", 0.95);
    bounded_indexed.observe_uncertain("replacement", "new", 0.95);
    bounded_linear.observe_uncertain("replacement", "new", 1.0);
    bounded_indexed.observe_uncertain("replacement", "new", 1.0);
    RLF_CHECK(bounded_linear.deterministic_hash() ==
              bounded_indexed.deterministic_hash());

    const auto linear_stats = linear.training_operation_stats();
    const auto indexed_stats = indexed.training_operation_stats();
    RLF_CHECK(linear_stats.active_learning_duplicate_lookups == 128U);
    RLF_CHECK(linear_stats.linear_active_learning_comparisons > 0U);
    RLF_CHECK(linear_stats.indexed_active_learning_candidates == 0U);
    RLF_CHECK(indexed_stats.active_learning_duplicate_lookups > 128U);
    RLF_CHECK(indexed_stats.linear_active_learning_comparisons == 0U);
    RLF_CHECK(indexed_stats.indexed_active_learning_candidates > 0U);
    RLF_CHECK(indexed_stats.active_learning_index_incremental_inserts == 64U);
}

RLF_TEST_CASE("general H100 profile fits defensive checkpoint load ceilings") {
    const auto config = rlf::solstice::make_profile_config(
        rlf::solstice::SolsticeProfile::general_h100_80g
    );
    const auto capacity = rlf::solstice::estimate_profile_capacity(
        rlf::solstice::SolsticeProfile::general_h100_80g
    );
    const rlf::solstice::SolsticeCheckpointLimits limits;
    RLF_CHECK(capacity.checkpoint_ceiling_bytes <= limits.maximum_file_bytes);
    RLF_CHECK(config.language.maximum_contexts <= limits.maximum_collection_entries);
    RLF_CHECK(config.vision.maximum_examples <= limits.maximum_collection_entries);
    RLF_CHECK(config.abstraction.maximum_facts <= limits.maximum_collection_entries);
    RLF_CHECK(config.continual.maximum_prototypes <= limits.maximum_collection_entries);
    RLF_CHECK(config.grounding.maximum_links <= limits.maximum_collection_entries);
    RLF_CHECK(config.general.maximum_demonstrations <= limits.maximum_collection_entries);
}

RLF_TEST_CASE("profile-bound checkpoint inspection rejects a different profile") {
    using rlf::solstice::SolsticeProfile;
    const std::filesystem::path path = temporary_path("profile-bound.rlfsp");
    std::filesystem::remove(path);
    const rlf::solstice::SolsticeModel model(
        rlf::solstice::make_profile_config(SolsticeProfile::general_h100_80g),
        0x50524F46494C45ULL
    );
    rlf::solstice::save_solstice_checkpoint(path, model);
    const auto summary = rlf::solstice::inspect_solstice_checkpoint_for_profile(
        path, SolsticeProfile::general_h100_80g
    );
    RLF_CHECK(summary.format_version == 6U);
    RLF_CHECK_THROWS_AS(
        rlf::solstice::inspect_solstice_checkpoint_for_profile(
            path, SolsticeProfile::general_v100_32g_500m
        ),
        std::runtime_error
    );
    std::filesystem::remove(path);
}

RLF_TEST_CASE("General instruction and preference state persist in version six checkpoints") {
    const std::filesystem::path path = temporary_path("checkpoint.rlfsp");
    std::filesystem::remove(path);
    rlf::solstice::SolsticeModel model;
    model.bootstrap();
    model.train_instruction(
        "reasoning", "science",
        "How should a surprising experimental result be handled?",
        "Check measurement validity, reproduce the result, compare alternative hypotheses, and avoid premature conclusions.",
        "Verify the measurement, reproduce the observation, and compare competing explanations before drawing a conclusion."
    );
    model.train_preference(
        "How should a surprising experimental result be handled?",
        "Verify the measurement and reproduce the observation.",
        "Immediately announce a discovery.",
        "Prefer reproducibility over hype."
    );
    const std::uint64_t hash = model.deterministic_hash();
    rlf::solstice::save_solstice_checkpoint(path, model);
    const auto summary = rlf::solstice::inspect_solstice_checkpoint(path);
    RLF_CHECK(summary.format_version == 6U);
    const rlf::solstice::SolsticeModel restored =
        rlf::solstice::load_solstice_checkpoint(path);
    RLF_CHECK(restored.deterministic_hash() == hash);
    RLF_CHECK(restored.general().demonstrations().size() >= 5U);
    RLF_CHECK(restored.general().preferences().size() == 1U);
    const auto response = restored.respond(
        "How should a surprising experimental result be handled?",
        nullptr,
        nullptr,
        rlf::solstice::GenerationSettings{128U, 16U, 0.8, true, 9U}
    );
    RLF_CHECK(response.text.find("reproduce") != std::string::npos);
    std::filesystem::remove(path);
}

RLF_TEST_CASE("Frontier parity gate fails closed and passes complete external evidence") {
    const auto targets = rlf::solstice::leading_system_targets_2026();
    std::vector<rlf::solstice::FrontierBenchmarkEvidence> evidence;
    evidence.reserve(targets.size());
    for (const auto& target : targets) {
        evidence.emplace_back();
        auto& item = evidence.back();
        item.benchmark = target.benchmark;
        item.capability = target.capability;
        item.score = target.minimum_score;
        item.examples = target.minimum_examples;
        item.external_dataset = true;
        item.contamination_audited = true;
        item.independent_harness = true;
        item.evidence_artifact_present = true;
        item.evidence_path = "results/" + target.benchmark + ".json";
        item.dataset_version = "official-2026";
        item.harness_name = "official-harness";
        item.harness_version = "1.0.0";
        item.evaluator = "independent-lab";
        item.evidence_sha256 = std::string(64U, 'a');
        item.evidence_sha256_verified = true;
        item.contamination_report_path =
            "results/" + target.benchmark + "_contamination.json";
        item.contamination_report_sha256 = std::string(64U, 'b');
        item.contamination_report_present = true;
        item.contamination_sha256_verified = true;
        item.model_checkpoint_sha256 = std::string(64U, 'c');
        item.training_manifest_sha256 = std::string(64U, 'd');
        item.model_identity_verified = true;
    }
    const auto passed = rlf::solstice::evaluate_frontier_gate(evidence, targets);
    RLF_CHECK(passed.broad_frontier_parity_proven);
    RLF_CHECK(passed.passed_targets == targets.size());
    RLF_CHECK(passed.external_examples == passed.minimum_external_examples);
    evidence.front().contamination_audited = false;
    const auto failed = rlf::solstice::evaluate_frontier_gate(evidence, targets);
    RLF_CHECK(!failed.broad_frontier_parity_proven);
    RLF_CHECK(!failed.blocking_reasons.empty());
    evidence.front().contamination_audited = true;
    evidence.front().evidence_sha256_verified = false;
    const auto bad_hash = rlf::solstice::evaluate_frontier_gate(evidence, targets);
    RLF_CHECK(!bad_hash.broad_frontier_parity_proven);
    evidence.front().evidence_sha256_verified = true;
    evidence.front().model_identity_verified = false;
    const auto wrong_model = rlf::solstice::evaluate_frontier_gate(evidence, targets);
    RLF_CHECK(!wrong_model.broad_frontier_parity_proven);
    RLF_CHECK(
        wrong_model.items.front().reason.find("model checkpoint") != std::string::npos
    );
    evidence.front().model_identity_verified = true;
    evidence.push_back(evidence.front());
    const auto ambiguous = rlf::solstice::evaluate_frontier_gate(evidence, targets);
    RLF_CHECK(!ambiguous.broad_frontier_parity_proven);
}

RLF_TEST_CASE("General H100 profile exposes expanded instruction and deliberation capacity") {
    const auto config = rlf::solstice::make_profile_config(
        rlf::solstice::SolsticeProfile::general_h100_80g
    );
    const auto capacity = rlf::solstice::estimate_profile_capacity(
        rlf::solstice::SolsticeProfile::general_h100_80g
    );
    RLF_CHECK(config.general.maximum_demonstrations == 16'000'000U);
    RLF_CHECK(config.general.maximum_preferences == 8'000'000U);
    RLF_CHECK(config.general.deliberation_candidates == 8U);
    RLF_CHECK(config.language.maximum_episodes == 16'000'000U);
    RLF_CHECK(capacity.gpu_working_set_bytes == 74ULL * 1024ULL * 1024ULL * 1024ULL);
}

RLF_TEST_CASE("Constrained CUDA profiles preserve general ceilings with distinct VRAM budgets") {
    const auto profile_40g = rlf::solstice::parse_profile("general-40g");
    const auto profile_v100 = rlf::solstice::parse_profile("general-v100-32g");
    const auto config_40g = rlf::solstice::make_profile_config(profile_40g);
    const auto config_v100 = rlf::solstice::make_profile_config(profile_v100);
    const auto capacity_40g = rlf::solstice::estimate_profile_capacity(profile_40g);
    const auto capacity_v100 = rlf::solstice::estimate_profile_capacity(profile_v100);
    RLF_CHECK(rlf::solstice::to_string(profile_40g) == "general-40g");
    RLF_CHECK(rlf::solstice::to_string(profile_v100) == "general-v100-32g");
    RLF_CHECK(config_40g.general.maximum_demonstrations == 16'000'000U);
    RLF_CHECK(config_v100.grounding.maximum_links == 200'000'000U);
    RLF_CHECK(config_40g.vision.retrieval_candidate_batch == 4'096U);
    RLF_CHECK(config_v100.vision.training_patch_batch == 512U);
    RLF_CHECK(capacity_40g.gpu_working_set_bytes == 38ULL * 1024ULL * 1024ULL * 1024ULL);
    RLF_CHECK(capacity_v100.gpu_working_set_bytes == 30ULL * 1024ULL * 1024ULL * 1024ULL);
    RLF_CHECK(capacity_40g.checkpoint_ceiling_bytes ==
              capacity_v100.checkpoint_ceiling_bytes);

    const auto text_profile = rlf::solstice::parse_profile("general-v100-32g-text");
    const auto video_profile = rlf::solstice::parse_profile("video-rtx-pro-6000-96g");
    const auto v100_video_profile = rlf::solstice::parse_profile("video-v100-32g");
    const auto text_config = rlf::solstice::make_profile_config(text_profile);
    const auto video_config = rlf::solstice::make_profile_config(video_profile);
    const auto v100_video_config = rlf::solstice::make_profile_config(
        v100_video_profile
    );
    RLF_CHECK(!rlf::solstice::profile_allows_vision(text_profile));
    RLF_CHECK(!rlf::solstice::profile_allows_video(text_profile));
    RLF_CHECK(rlf::solstice::profile_allows_vision(video_profile));
    RLF_CHECK(rlf::solstice::profile_allows_video(video_profile));
    RLF_CHECK(text_config.vision.maximum_examples == 1U);
    RLF_CHECK(video_config.vision.maximum_examples == 32'000'000U);
    RLF_CHECK(rlf::solstice::to_string(v100_video_profile) == "video-v100-32g");
    RLF_CHECK(rlf::solstice::profile_allows_vision(v100_video_profile));
    RLF_CHECK(rlf::solstice::profile_allows_video(v100_video_profile));
    RLF_CHECK(v100_video_config.video.maximum_sequences == 8'000'000U);
    RLF_CHECK(v100_video_config.vision.maximum_examples ==
              video_config.vision.maximum_examples);
    RLF_CHECK(v100_video_config.vision.training_patch_batch == 512U);
    RLF_CHECK(rlf::solstice::profile_config_matches(
        v100_video_profile,
        v100_video_config
    ));
    RLF_CHECK(rlf::solstice::estimate_profile_capacity(
        v100_video_profile
    ).gpu_working_set_bytes == 30ULL * 1024ULL * 1024ULL * 1024ULL);
    RLF_CHECK(rlf::solstice::profile_config_matches(text_profile, text_config));
    RLF_CHECK(!rlf::solstice::profile_config_matches(text_profile, config_v100));
    RLF_CHECK(rlf::solstice::estimate_profile_capacity(video_profile).gpu_working_set_bytes ==
              90ULL * 1024ULL * 1024ULL * 1024ULL);
    const auto pro_profile = rlf::solstice::parse_profile(
        "general-rtx-pro-6000-96g"
    );
    const auto pro_text_profile = rlf::solstice::parse_profile(
        "general-rtx-pro-6000-96g-text"
    );
    RLF_CHECK(rlf::solstice::profile_allows_vision(pro_profile));
    RLF_CHECK(!rlf::solstice::profile_allows_vision(pro_text_profile));
    RLF_CHECK(!rlf::solstice::profile_allows_video(pro_profile));
    RLF_CHECK(rlf::solstice::make_profile_config(pro_profile).general.maximum_demonstrations ==
              32'000'000U);
    RLF_CHECK(rlf::solstice::make_profile_config(pro_text_profile).vision.maximum_examples ==
              1U);
    RLF_CHECK(rlf::solstice::estimate_profile_capacity(pro_profile).gpu_working_set_bytes ==
              90ULL * 1024ULL * 1024ULL * 1024ULL);
    const auto exact_target_profile = rlf::solstice::parse_profile(
        "rtx-pro-6000-96g"
    );
    RLF_CHECK(rlf::solstice::to_string(exact_target_profile) ==
              "rtx-pro-6000-96g");
    RLF_CHECK(rlf::solstice::profile_allows_vision(exact_target_profile));
    RLF_CHECK(!rlf::solstice::profile_allows_video(exact_target_profile));
    RLF_CHECK(rlf::solstice::profile_config_matches(
        exact_target_profile,
        rlf::solstice::make_profile_config(exact_target_profile)
    ));
    auto keyword_capacity_tamper = rlf::solstice::make_profile_config(exact_target_profile);
    --keyword_capacity_tamper.tool_router.maximum_keywords_per_tool;
    RLF_CHECK(!rlf::solstice::profile_config_matches(
        exact_target_profile,
        keyword_capacity_tamper
    ));
    RLF_CHECK(rlf::solstice::estimate_profile_capacity(
        exact_target_profile
    ).gpu_working_set_bytes == 88ULL * 1024ULL * 1024ULL * 1024ULL);
}

RLF_TEST_CASE("RTX PRO 6000 profile admits the fixed 50M nonvisual mix") {
    const auto rtx = rlf::solstice::make_profile_config(
        rlf::solstice::SolsticeProfile::general_rtx_pro_6000_96g
    );
    const auto h100 = rlf::solstice::make_profile_config(
        rlf::solstice::SolsticeProfile::general_h100_80g
    );
    constexpr std::size_t instruction_records = 24'500'001U;
    constexpr std::size_t preference_records = 8'166'666U;
    constexpr std::size_t dialogue_records =
        instruction_records + preference_records;
    RLF_CHECK(rtx.general.maximum_demonstrations >= instruction_records);
    RLF_CHECK(rtx.general.maximum_preferences >= preference_records);
    RLF_CHECK(rtx.language.maximum_episodes >= dialogue_records);
    RLF_CHECK(rtx.tool_router.maximum_keywords_per_tool >= 8'000'000U);
    RLF_CHECK(rtx.vision.maximum_examples >= 1'000'000U);
    RLF_CHECK(rtx.abstraction.maximum_facts >= 8'166'666U);
    RLF_CHECK(rlf::solstice::estimate_profile_capacity(
        rlf::solstice::SolsticeProfile::general_rtx_pro_6000_96g
    ).gpu_working_set_bytes == 90ULL * 1024ULL * 1024ULL * 1024ULL);
    RLF_CHECK(h100.general.maximum_demonstrations == 16'000'000U);
    RLF_CHECK(h100.general.maximum_preferences == 8'000'000U);
    RLF_CHECK(h100.language.maximum_episodes == 16'000'000U);
}

RLF_TEST_CASE("V100 500M profile expands only progressive host-backed admission ceilings") {
    using rlf::solstice::SolsticeProfile;
    const auto profile = rlf::solstice::parse_profile("general-v100-32g-500m");
    const auto alias = rlf::solstice::parse_profile("v100-32g-500m");
    const auto config = rlf::solstice::make_profile_config(profile);
    const auto v100 = rlf::solstice::make_profile_config(
        SolsticeProfile::general_v100_32g
    );
    const auto h100 = rlf::solstice::make_profile_config(
        SolsticeProfile::general_h100_80g
    );
    const auto capacity = rlf::solstice::estimate_profile_capacity(profile);

    RLF_CHECK(profile == SolsticeProfile::general_v100_32g_500m);
    RLF_CHECK(alias == profile);
    RLF_CHECK(rlf::solstice::to_string(profile) == "general-v100-32g-500m");
    RLF_CHECK(rlf::solstice::profile_allows_vision(profile));
    RLF_CHECK(!rlf::solstice::profile_allows_video(profile));
    RLF_CHECK(rlf::solstice::profile_config_matches(profile, config));

    struct CandidateMix final {
        std::size_t instructions;
        std::size_t preferences;
        std::size_t tools;
        std::size_t facts;
        std::size_t images;
    };
    constexpr std::array<CandidateMix, 3U> staged_mixes{{
        {24'500'001U, 8'166'666U, 8'166'667U, 8'166'666U, 1'000'000U},
        {98'000'001U, 32'666'666U, 32'666'667U, 32'666'666U, 4'000'000U},
        {245'000'001U, 81'666'666U, 81'666'667U, 81'666'666U, 10'000'000U},
    }};
    for (const CandidateMix& mix : staged_mixes) {
        RLF_CHECK(config.general.maximum_demonstrations >= mix.instructions);
        RLF_CHECK(config.general.maximum_preferences >= mix.preferences);
        RLF_CHECK(config.language.maximum_episodes >=
                  mix.instructions + mix.preferences);
        RLF_CHECK(config.tool_router.maximum_keywords_per_tool >= mix.tools);
        RLF_CHECK(config.abstraction.maximum_facts >= mix.facts);
        RLF_CHECK(config.vision.maximum_examples >= mix.images);
    }

    // Learning behavior and all device-facing batch sizes remain those of the
    // existing multimodal V100 profile; only logical host capacities differ.
    RLF_CHECK(config.language.context_orders == v100.language.context_orders);
    RLF_CHECK(config.language.smoothing == v100.language.smoothing);
    RLF_CHECK(config.language.long_context_weight == v100.language.long_context_weight);
    RLF_CHECK(config.vision.retrieval_query_batch == v100.vision.retrieval_query_batch);
    RLF_CHECK(config.vision.retrieval_candidate_batch ==
              v100.vision.retrieval_candidate_batch);
    RLF_CHECK(config.vision.training_patch_batch == v100.vision.training_patch_batch);
    RLF_CHECK(config.vision.local_learning_rate == v100.vision.local_learning_rate);
    RLF_CHECK(config.continual.replay_batch_size == v100.continual.replay_batch_size);
    RLF_CHECK(config.general.deliberation_candidates ==
              v100.general.deliberation_candidates);

    RLF_CHECK(capacity.gpu_working_set_bytes ==
              30ULL * 1024ULL * 1024ULL * 1024ULL);
    RLF_CHECK(capacity.cpu_resident_bytes ==
              5ULL * 1024ULL * 1024ULL * 1024ULL * 1024ULL);
    RLF_CHECK(capacity.checkpoint_ceiling_bytes ==
              10ULL * 1024ULL * 1024ULL * 1024ULL * 1024ULL);
    const auto large_checkpoint_limits =
        rlf::solstice::checkpoint_limits_for_profile(profile);
    const auto ordinary_checkpoint_limits =
        rlf::solstice::checkpoint_limits_for_profile(
            SolsticeProfile::general_v100_32g
        );
    RLF_CHECK(large_checkpoint_limits.maximum_file_bytes >=
              capacity.checkpoint_ceiling_bytes);
    RLF_CHECK(large_checkpoint_limits.maximum_collection_entries >=
              config.language.maximum_contexts);
    RLF_CHECK(large_checkpoint_limits.maximum_collection_entries >=
              config.language.maximum_episodes);
    RLF_CHECK(large_checkpoint_limits.maximum_collection_entries >=
              config.general.maximum_demonstrations);
    RLF_CHECK(ordinary_checkpoint_limits.maximum_file_bytes ==
              1'025ULL * 1024ULL * 1024ULL * 1024ULL);
    RLF_CHECK(ordinary_checkpoint_limits.maximum_collection_entries ==
              250'000'000U);

    // Existing H100 and V100 configurations are regression sentinels.
    RLF_CHECK(v100.general.maximum_demonstrations == 16'000'000U);
    RLF_CHECK(v100.language.maximum_episodes == 16'000'000U);
    RLF_CHECK(h100.general.maximum_demonstrations == 16'000'000U);
    RLF_CHECK(h100.language.maximum_episodes == 16'000'000U);

    auto tampered = config;
    --tampered.general.maximum_demonstrations;
    RLF_CHECK(!rlf::solstice::profile_config_matches(profile, tampered));
}

RLF_TEST_CASE("H200 30T profile is explicit large-state multimodal capacity") {
    using rlf::solstice::SolsticeProfile;
    const auto profile = rlf::solstice::parse_profile("general-h200-141g-30t");
    const auto alias = rlf::solstice::parse_profile("h200");
    const auto config = rlf::solstice::make_profile_config(profile);
    const auto capacity = rlf::solstice::estimate_profile_capacity(profile);
    const auto limits = rlf::solstice::checkpoint_limits_for_profile(profile);

    RLF_CHECK(profile == SolsticeProfile::general_h200_141g_30t);
    RLF_CHECK(alias == profile);
    RLF_CHECK(rlf::solstice::to_string(profile) == "general-h200-141g-30t");
    RLF_CHECK(rlf::solstice::profile_allows_vision(profile));
    RLF_CHECK(!rlf::solstice::profile_allows_video(profile));
    RLF_CHECK(rlf::solstice::profile_config_matches(profile, config));
    RLF_CHECK(config.language.maximum_contexts == 2'000'000'000U);
    RLF_CHECK(config.language.maximum_episodes == 500'000'000U);
    RLF_CHECK(config.vision.maximum_examples == 250'000'000U);
    RLF_CHECK(config.general.maximum_demonstrations == 500'000'000U);
    RLF_CHECK(config.general.maximum_preferences == 250'000'000U);
    RLF_CHECK(capacity.gpu_working_set_bytes ==
              132ULL * 1024ULL * 1024ULL * 1024ULL);
    RLF_CHECK(capacity.cpu_resident_bytes ==
              8ULL * 1024ULL * 1024ULL * 1024ULL * 1024ULL);
    RLF_CHECK(capacity.checkpoint_ceiling_bytes ==
              16ULL * 1024ULL * 1024ULL * 1024ULL * 1024ULL);
    RLF_CHECK(limits.maximum_file_bytes > capacity.checkpoint_ceiling_bytes);
    RLF_CHECK(limits.maximum_collection_entries >=
              config.grounding.maximum_links);

    auto tampered = config;
    --tampered.language.maximum_contexts;
    RLF_CHECK(!rlf::solstice::profile_config_matches(profile, tampered));
}
