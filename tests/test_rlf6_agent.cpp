#include "test_framework.hpp"

#include "rlf/agent/agent_fabric.hpp"
#include "rlf/experiments/rlf6_agent.hpp"

#include <sstream>
#include <string>
#include <vector>

namespace {

rlf::agent::Goal goal_for(const std::string& key, const std::string& value) {
    rlf::agent::Goal goal;
    goal.specification = "achieve " + key;
    goal.completion_conditions = {{key, value, false}};
    goal.provenance = "test";
    return goal;
}

rlf::agent::Action action_for(
    const std::uint64_t id,
    const std::string& name,
    const std::vector<rlf::agent::Fact>& preconditions,
    const std::vector<rlf::agent::Fact>& effects
) {
    rlf::agent::Action action;
    action.stable_id = id;
    action.name = name;
    action.preconditions = preconditions;
    action.expected_effects = effects;
    action.confidence = 0.9;
    action.provenance = "test";
    return action;
}

rlf::experiments::Rlf6Config compact_config() {
    rlf::experiments::Rlf6Config config;
    config.seed = 0x524C463654455354ULL;
    config.training_episodes = 10U;
    config.evaluation_episodes = 10U;
    config.minimum_route_length = 5U;
    config.maximum_route_length = 22U;
    config.stress_episodes = 1U;
    config.stress_route_length = 101U;
    config.action_budget = 128U;
    config.planning_node_budget = 128U;
    config.tool_budget = 64U;
    config.memory_limit_records = 8'192U;
    config.tool_cost_budget = 2'000.0;
    config.risk_budget = 25.0;
    config.include_stress = false;
    config.experiment_name = "long_horizon_planning";
    return config;
}

}  // namespace

RLF_TEST_CASE("RLF-6 keeps observations beliefs verified facts and stale state distinct") {
    rlf::agent::AgentFabric fabric({}, 11U);
    rlf::agent::EvidenceRecord observation;
    observation.fact = {"door", "open", false};
    observation.kind = rlf::agent::EvidenceKind::observation;
    observation.confidence = 0.8;
    observation.source_reliability = 0.75;
    observation.provenance = "sensor";
    fabric.ingest_evidence(observation);
    RLF_CHECK(fabric.state().observation_state.size() == 1U);
    RLF_CHECK(fabric.state().belief_state.size() == 1U);
    RLF_CHECK(fabric.state().verified_facts.empty());

    rlf::agent::EvidenceRecord verified = observation;
    verified.kind = rlf::agent::EvidenceKind::verified_fact;
    verified.confidence = 1.0;
    verified.source_reliability = 1.0;
    verified.verified = true;
    fabric.ingest_evidence(verified);
    RLF_CHECK(!fabric.state().verified_facts.empty());
    fabric.mark_stale("door");
    const auto belief = fabric.best_belief("door");
    RLF_CHECK(!belief.has_value());
    RLF_CHECK(!fabric.state().belief_state.empty());
    RLF_CHECK(fabric.state().belief_state.front().stale);
}

RLF_TEST_CASE("RLF-6 detects conflicting persistent goals and graph-derived subgoals") {
    rlf::agent::AgentFabric fabric({}, 12U);
    const auto first = fabric.add_goal(goal_for("mode", "alpha"));
    static_cast<void>(first);
    const auto second = fabric.add_goal(goal_for("mode", "beta"));
    static_cast<void>(second);
    RLF_CHECK(fabric.detect_goal_conflicts());
    RLF_CHECK(fabric.state().goal_stack[0].status == rlf::agent::GoalStatus::conflicted);

    rlf::agent::AgentFabric planner({}, 13U);
    const auto goal_id = planner.add_goal(goal_for("goal", "done"));
    const std::vector<rlf::agent::Action> actions{
        action_for(1U, "prepare", {}, {{"prepared", "yes", false}}),
        action_for(2U, "finish", {{"prepared", "yes", false}}, {{"goal", "done", false}}),
    };
    const auto subgoals = planner.discover_subgoals(goal_id, actions);
    RLF_CHECK(!subgoals.empty());
    RLF_CHECK(!planner.state().subgoal_graph.empty());
}

RLF_TEST_CASE("RLF-6 validates typed tools learns reliability and rejects unsafe actions") {
    rlf::agent::AgentFabric fabric({}, 14U);
    rlf::agent::ToolDefinition tool;
    tool.stable_id = 100U;
    tool.name = "database";
    tool.input_schema = {"key"};
    tool.output_schema = {"value"};
    tool.declared_reliability = 0.8;
    fabric.register_tool(tool);

    rlf::agent::Action call;
    call.stable_id = 1U;
    call.name = "query";
    call.action_type = rlf::agent::ActionType::tool;
    call.tool_id = 100U;
    call.parameters = {{"key", "x"}};
    RLF_CHECK(fabric.validate_tool_arguments(call));
    call.parameters.clear();
    RLF_CHECK(!fabric.validate_tool_arguments(call));

    const double before = fabric.tool_reliability(100U, "ctx");
    fabric.update_tool_reliability(100U, "ctx", true);
    fabric.update_tool_reliability(100U, "ctx", true);
    RLF_CHECK(fabric.tool_reliability(100U, "ctx") > before);

    rlf::agent::Action prohibited;
    prohibited.stable_id = 8U;
    prohibited.name = "unsafe";
    prohibited.safety_class = rlf::agent::SafetyClass::prohibited;
    const auto decision = fabric.evaluate_safety(prohibited, {});
    RLF_CHECK(!decision.allowed);
    RLF_CHECK(decision.should_abstain);
}

RLF_TEST_CASE("RLF-6 bounded planning correction memory and skill consolidation are deterministic") {
    rlf::agent::AgentConfig config;
    config.maximum_candidate_actions = 32U;
    config.maximum_plan_depth = 16U;
    rlf::agent::AgentFabric first(config, 15U);
    const auto goal_id = first.add_goal(goal_for("goal", "done"));
    const std::vector<rlf::agent::Action> actions{
        action_for(1U, "prepare", {}, {{"prepared", "yes", false}}),
        action_for(2U, "finish", {{"prepared", "yes", false}}, {{"goal", "done", false}}),
        action_for(3U, "wrong", {}, {{"noise", "yes", false}}),
    };
    rlf::agent::PlanningRequest request;
    request.actions = actions;
    request.goal_id = goal_id;
    request.policy = rlf::agent::PlanningPolicy::bounded_astar;
    request.node_budget = 64U;
    request.depth_budget = 8U;
    const auto plan = first.plan(request);
    RLF_CHECK(plan.found);
    RLF_CHECK(plan.nodes_expanded <= request.node_budget);

    auto second = rlf::agent::AgentFabric::from_snapshot(first.snapshot());
    RLF_CHECK(second.deterministic_hash() == first.deterministic_hash());
    const auto alternatives = first.counterfactual_actions(actions, actions[2], goal_id, 2U);
    RLF_CHECK(!alternatives.empty());
    RLF_CHECK(alternatives.front() != actions[2].stable_id);

    rlf::agent::MemoryRecord memory;
    memory.memory_class = rlf::agent::MemoryClass::episodic;
    memory.key = "failure:door";
    memory.payload = "use alternate";
    memory.provenance = "test";
    const auto memory_id = first.insert_memory(memory);
    RLF_CHECK(first.retrieve_memory(rlf::agent::MemoryClass::episodic, "door", 4U).size() == 1U);
    first.invalidate_memory(memory_id);
    RLF_CHECK(first.retrieve_memory(rlf::agent::MemoryClass::episodic, "door", 4U).empty());

    RLF_CHECK(!first.consolidate_skill("goal", {1U, 2U}, 10.0, 4.0, true).has_value());
    RLF_CHECK(first.consolidate_skill("goal", {1U, 2U}, 10.0, 4.0, true).has_value());
}

RLF_TEST_CASE("RLF-6 controlled agent benchmark is deterministic bounded and leakage-audited") {
    const auto config = compact_config();
    const auto first = rlf::experiments::run_rlf6_agent(config);
    const auto second = rlf::experiments::run_rlf6_agent(config);
    RLF_CHECK(first.efficiency.deterministic_hash == second.efficiency.deterministic_hash);
    RLF_CHECK(first.leakage_audit.evaluation_routes_absent_from_memory);
    RLF_CHECK(first.leakage_audit.route_hash_overlap == 0U);
    RLF_CHECK(first.task.planning_nodes <= first.task.episodes * config.planning_node_budget);
    RLF_CHECK(first.safety.unsafe_actions_executed == 0U);
    RLF_CHECK(first.baselines.size() >= 20U);
    RLF_CHECK(first.scientific_decision == "Decision A — strong evidence" ||
        first.scientific_decision == "Decision B — partial evidence" ||
        first.scientific_decision == "Decision C — negative evidence");
    std::ostringstream json;
    rlf::experiments::write_rlf6_result_json(json, first);
    RLF_CHECK(json.str().find("\"architecture\": \"RLF-6\"") != std::string::npos);
}
