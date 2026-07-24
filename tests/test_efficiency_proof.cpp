#include "test_framework.hpp"

#include "rlf/solstice/abstraction_fabric.hpp"
#include "rlf/solstice/efficiency_proof.hpp"
#include "rlf/solstice/checkpoint.hpp"
#include "rlf/solstice/profile.hpp"
#include "rlf/solstice/solstice_model.hpp"

#include <filesystem>
#include <string>

RLF_TEST_CASE("Frontier induces a reusable chain schema from one demonstration") {
    rlf::solstice::AbstractionFabric fabric;
    for (std::size_t index = 0U; index < 32U; ++index) {
        const std::string suffix = std::to_string(index);
        fabric.learn_fact("a_" + suffix, "left", "b_" + suffix);
        fabric.learn_fact("b_" + suffix, "right", "c_" + suffix);
    }
    const auto induction = fabric.induce_chain_rule(
        "learned chain", "a_0", "composed", "c_0", 2U
    );
    RLF_CHECK(induction.rule_id != 0U);
    RLF_CHECK(induction.path_hops == 2U);
    const auto result = fabric.infer_with_stats(
        {"?subject", "composed", "?object"}, 32U
    );
    RLF_CHECK(result.answers.size() == 32U);
    RLF_CHECK(result.stats.candidate_facts_examined <
        result.stats.naive_candidate_upper_bound);
}

RLF_TEST_CASE("Frontier quick proof keeps claims scoped and auditable") {
    rlf::solstice::EfficiencyProofConfig config;
    config.schema_instances = 129U;
    config.routing_vectors = 8'192U;
    config.routing_dimensions = 16U;
    config.routing_queries = 32U;
    config.routing_trials = 2U;
    config.continual_classes = 8U;
    config.continual_tasks = 4U;
    config.target_efficiency_ratio = 100.0;
    config.minimum_accuracy = 0.99;
    const auto report = rlf::solstice::run_efficiency_proofs(config);
    RLF_CHECK(!report.narrow_ten_thousand_x_proven);
    RLF_CHECK(report.all_internal_proofs_passed);
    RLF_CHECK(!report.general_learning_efficiency_proven);
    RLF_CHECK(!report.frontier_parity_proven);
}

RLF_TEST_CASE("Frontier induced schemas persist through Solstice checkpoints") {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        "rlf_frontier_induced_schema.rlfsp";
    std::filesystem::remove(path);
    rlf::solstice::SolsticeModel model(
        rlf::solstice::make_profile_config(
            rlf::solstice::SolsticeProfile::preview_6g
        )
    );
    model.learn_fact("a", "left", "b");
    model.learn_fact("b", "right", "c");
    model.learn_fact("x", "left", "y");
    model.learn_fact("y", "right", "z");
    const auto induction = model.induce_chain_rule(
        "persisted schema", "a", "composed", "c", 2U
    );
    RLF_CHECK(induction.rule_id != 0U);
    rlf::solstice::save_solstice_checkpoint(path, model);
    const auto restored = rlf::solstice::load_solstice_checkpoint(path);
    const auto answers = restored.reason({"x", "composed", "?answer"});
    RLF_CHECK(!answers.empty());
    RLF_CHECK(answers.front().value == "z");
    std::filesystem::remove(path);
}
