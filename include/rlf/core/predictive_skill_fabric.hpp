#pragma once

#include "rlf/core/deterministic_rng.hpp"
#include "rlf/core/latent_routing.hpp"
#include "rlf/core/phase_vector.hpp"
#include "rlf/core/transformation_operator.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rlf::core {

enum class Rlf2StopReason {
    successful_halt,
    learned_halt,
    planner_failure,
    cycle_limit,
    loop_detected,
    abstained,
    no_candidate,
};

[[nodiscard]] std::string_view to_string(Rlf2StopReason reason) noexcept;

struct PredictiveSkillConfig final {
    std::size_t dimension{24U};
    std::size_t maximum_cycles{24U};
    std::size_t maximum_route_depth{12U};
    std::size_t planner_node_budget{500'000U};
    std::size_t maximum_skills{8'192U};
    std::size_t maximum_subgoal_prototypes{200'000U};
    std::size_t maximum_skill_length{4U};
    std::size_t minimum_skill_support{2U};
    std::size_t nearest_prototypes{5U};
    double goal_similarity_threshold{0.9995};
    double prototype_merge_distance{0.015};
    double prototype_distance_scale{8.0};
    double learned_value_weight{1.25};
    double successor_value_weight{0.75};
    double direct_progress_weight{0.35};
    double causal_advantage_weight{0.50};
    double skill_cost_weight{0.025};
    double repetition_penalty{0.30};
    double action_temperature{0.20};
    double abstention_uncertainty_threshold{0.995};
    double abstention_value_threshold{0.05};
    bool enable_skill_consolidation{true};
    bool enable_intervention_credit{true};
};

struct PredictiveOperator final {
    std::uint64_t id{};
    std::string name;
    TransformationOperator forward{1U};
    TransformationOperator inverse{1U};
    double cost{1.0};
};

struct CausalSkill final {
    std::uint64_t id{};
    std::string name;
    std::vector<std::uint64_t> primitive_route;
    TransformationOperator forward{1U};
    TransformationOperator inverse{1U};
    std::size_t primitive_length{1U};
    std::uint64_t support{};
    std::uint64_t success_count{};
    std::uint64_t failure_count{};
    double utility{};
    double mean_causal_advantage{};
    bool accepted{false};
};

struct SubgoalPrototype final {
    std::uint64_t id{};
    std::vector<float> response_profile;
    std::uint64_t skill_id{};
    double remaining_steps{};
    double terminal_value{};
    double causal_advantage{};
    double confidence{0.5};
    std::uint64_t support{1ULL};
    std::uint64_t creation_step{};
    std::uint64_t last_used_step{};
};

struct Rlf2ActionCandidate final {
    std::uint64_t skill_id{};
    std::string skill_name;
    std::size_t primitive_length{};
    double direct_progress{};
    double learned_value{};
    double successor_value{};
    double causal_advantage{};
    double prototype_distance{};
    double score{};
    double normalized_weight{};
};

struct Rlf2TraceStep final {
    std::size_t cycle{};
    std::uint64_t state_hash{};
    std::uint64_t goal_hash{};
    std::uint64_t selected_skill_id{};
    std::string selected_skill_name;
    double goal_similarity_before{};
    double goal_similarity_after{};
    double uncertainty{};
    double predicted_remaining_steps{};
    bool bridge_subgoal{false};
    std::vector<Rlf2ActionCandidate> candidates;
};

struct Rlf2ExecutionResult final {
    PhaseVector final_state{std::vector<float>{0.0F}};
    std::vector<std::uint64_t> primitive_route;
    std::vector<std::uint64_t> skill_route;
    std::vector<Rlf2TraceStep> trace;
    Rlf2StopReason stop_reason{Rlf2StopReason::cycle_limit};
    bool success{false};
    bool abstained{false};
    std::size_t cycles{};
    std::size_t primitive_steps{};
    std::size_t planner_nodes{};
    std::size_t forward_nodes{};
    std::size_t backward_nodes{};
    std::size_t subgoals_considered{};
    double final_goal_similarity{};
    double mean_uncertainty{};
};

struct Rlf2TrainingStats final {
    std::uint64_t observed_routes{};
    std::uint64_t observed_transitions{};
    std::uint64_t intervention_tests{};
    std::uint64_t intervention_alternative_successes{};
    std::uint64_t prototypes_created{};
    std::uint64_t prototypes_merged{};
    std::uint64_t skills_proposed{};
    std::uint64_t skills_accepted{};
    std::uint64_t routes_segmented{};
};

struct PredictiveSkillSnapshot final {
    PredictiveSkillConfig config;
    std::uint64_t seed{};
    std::uint64_t training_step{};
    std::uint64_t next_operator_id{};
    std::uint64_t next_skill_id{};
    std::uint64_t next_prototype_id{};
    std::vector<PredictiveOperator> operators;
    std::vector<CausalSkill> skills;
    std::vector<SubgoalPrototype> prototypes;
    Rlf2TrainingStats training_stats;
};

class PredictiveSkillFabric final {
public:
    explicit PredictiveSkillFabric(
        PredictiveSkillConfig config,
        std::uint64_t seed
    );

    [[nodiscard]] const PredictiveSkillConfig& config() const noexcept;
    [[nodiscard]] std::uint64_t seed() const noexcept;
    [[nodiscard]] std::uint64_t training_step() const noexcept;
    [[nodiscard]] std::span<const PredictiveOperator> operators() const noexcept;
    [[nodiscard]] std::span<const CausalSkill> skills() const noexcept;
    [[nodiscard]] std::span<const SubgoalPrototype> prototypes() const noexcept;
    [[nodiscard]] const Rlf2TrainingStats& training_stats() const noexcept;

    std::uint64_t register_operator(
        std::string name,
        TransformationOperator transformation,
        double cost = 1.0
    );

    [[nodiscard]] const PredictiveOperator& operator_by_id(
        std::uint64_t operator_id
    ) const;
    [[nodiscard]] const CausalSkill& skill_by_id(
        std::uint64_t skill_id
    ) const;

    void observe_successful_route(
        const PhaseVector& start,
        const PhaseVector& goal,
        std::span<const std::uint64_t> primitive_route
    );

    [[nodiscard]] std::size_t consolidate_skills();
    [[nodiscard]] std::vector<std::uint64_t> segment_route(
        std::span<const std::uint64_t> primitive_route
    ) const;

    [[nodiscard]] Rlf2ExecutionResult execute_autonomous(
        const PhaseVector& start,
        const PhaseVector& goal
    );

    [[nodiscard]] Rlf2ExecutionResult execute_subgoal_bridge(
        const PhaseVector& start,
        const PhaseVector& goal,
        std::size_t maximum_primitive_depth = 0U,
        bool use_learned_ordering = true
    );

    [[nodiscard]] std::optional<std::vector<std::uint64_t>> plan_primitive_bridge(
        const PhaseVector& start,
        const PhaseVector& goal,
        std::size_t maximum_primitive_depth,
        std::size_t* explored_nodes = nullptr,
        std::size_t* forward_nodes = nullptr,
        std::size_t* backward_nodes = nullptr,
        bool use_learned_ordering = true
    ) const;

    [[nodiscard]] std::vector<float> response_profile(
        const PhaseVector& current,
        const PhaseVector& goal
    ) const;

    [[nodiscard]] PredictiveSkillSnapshot snapshot() const;
    [[nodiscard]] static PredictiveSkillFabric from_snapshot(
        PredictiveSkillSnapshot snapshot
    );

private:
    struct PrototypeEstimate final {
        double value{};
        double remaining_steps{};
        double causal_advantage{};
        double distance{1.0};
        double confidence{};
        bool found{false};
    };

    [[nodiscard]] std::vector<Rlf2ActionCandidate> score_actions(
        const PhaseVector& current,
        const PhaseVector& goal,
        std::span<const std::uint64_t> recent_skills
    ) const;
    [[nodiscard]] PrototypeEstimate estimate_for_skill(
        std::span<const float> profile,
        std::uint64_t skill_id
    ) const;
    [[nodiscard]] PrototypeEstimate estimate_state_value(
        std::span<const float> profile
    ) const;
    [[nodiscard]] double candidate_uncertainty(
        std::vector<Rlf2ActionCandidate>& candidates
    ) const;
    void update_prototype(
        std::span<const float> profile,
        std::uint64_t skill_id,
        double remaining_steps,
        double terminal_value,
        double causal_advantage
    );
    [[nodiscard]] double intervention_advantage(
        const PhaseVector& state,
        const PhaseVector& goal,
        std::uint64_t chosen_operator,
        std::size_t remaining_steps
    );
    [[nodiscard]] std::vector<std::uint64_t> ordered_operator_ids(
        const PhaseVector& state,
        const PhaseVector& goal,
        bool learned_ordering
    ) const;
    [[nodiscard]] TransformationOperator compose_route(
        std::span<const std::uint64_t> primitive_route
    ) const;
    [[nodiscard]] static std::uint64_t route_hash(
        std::span<const std::uint64_t> route
    ) noexcept;
    [[nodiscard]] static double profile_distance(
        std::span<const float> left,
        std::span<const float> right
    );
    [[nodiscard]] bool is_goal(
        const PhaseVector& state,
        const PhaseVector& goal
    ) const;

    PredictiveSkillConfig config_;
    std::uint64_t seed_{};
    DeterministicRng rng_;
    std::uint64_t training_step_{};
    std::uint64_t next_operator_id_{1ULL};
    std::uint64_t next_skill_id_{1ULL};
    std::uint64_t next_prototype_id_{1ULL};
    std::vector<PredictiveOperator> operators_;
    std::vector<CausalSkill> skills_;
    std::vector<SubgoalPrototype> prototypes_;
    Rlf2TrainingStats training_stats_;
    std::unordered_map<std::uint64_t, std::size_t> fragment_counts_;
    std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> fragment_routes_;
};

}  // namespace rlf::core
