#pragma once

#include "rlf/core/phase_vector.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rlf::core {

struct SparseWorldModelConfig final {
    std::size_t dimension{32U};
    std::size_t maximum_states{65'536U};
    std::size_t maximum_contexts{4'096U};
    std::size_t maximum_transitions{262'144U};
    std::size_t maximum_outcomes_per_transition{8U};
    std::size_t maximum_subgoals{262'144U};
    std::size_t hash_dimensions{4U};
    std::size_t phase_bins{8U};
    std::size_t maximum_bucket_candidates{256U};
    std::size_t nearest_subgoals{8U};
    std::size_t planner_node_budget{100'000U};
    std::size_t maximum_plan_depth{32U};
    std::size_t minimum_transition_support{1U};
    std::size_t minimum_subgoal_support{1U};
    double state_merge_distance{0.08};
    double context_merge_distance{0.08};
    double minimum_outcome_probability{0.01};
    double risk_penalty{0.20};
    double uncertainty_penalty{0.10};
    double heuristic_scale{1.0};
    std::size_t environment_layers{7U};
    std::size_t environment_lanes{5U};
    double environment_stochastic_probability{0.82};
    double environment_observation_noise{0.012};
};

struct WorldObservation final {
    PhaseVector visible{std::vector<float>{0.0F}};
    PhaseVector memory{std::vector<float>{0.0F}};
};

struct WorldAction final {
    std::uint64_t id{};
    std::string name;
    double cost{1.0};
};

struct WorldStatePrototype final {
    std::uint64_t id{};
    PhaseVector key{std::vector<float>{0.0F}};
    std::uint64_t support{1ULL};
    std::uint64_t creation_step{};
    std::uint64_t last_used_step{};
};

struct WorldContextPrototype final {
    std::uint64_t id{};
    PhaseVector key{std::vector<float>{0.0F}};
    std::uint64_t support{1ULL};
    std::uint64_t creation_step{};
    std::uint64_t last_used_step{};
};

struct WorldTransitionOutcome final {
    std::uint64_t next_state_id{};
    std::uint64_t next_context_id{};
    std::uint64_t count{};
    double mean_reward{};
    double terminal_probability{};
};

struct WorldTransition final {
    std::uint64_t id{};
    std::uint64_t state_id{};
    std::uint64_t context_id{};
    std::uint64_t action_id{};
    std::uint64_t support{};
    double mean_reward{};
    double mean_prediction_error{};
    std::vector<WorldTransitionOutcome> outcomes;
};

struct WorldTransitionExperience final {
    WorldObservation observation;
    std::uint64_t action_id{};
    WorldObservation next_observation;
    double reward{};
    bool terminal{false};
};

struct WorldPredictionOutcome final {
    std::uint64_t next_state_id{};
    std::uint64_t next_context_id{};
    double probability{};
    double expected_reward{};
    double terminal_probability{};
};

struct WorldPrediction final {
    std::uint64_t state_id{};
    std::uint64_t context_id{};
    std::uint64_t transition_id{};
    std::uint64_t action_id{};
    double context_distance{};
    double uncertainty{1.0};
    std::vector<WorldPredictionOutcome> outcomes;
};

struct SparseSubgoal final {
    std::uint64_t id{};
    std::uint64_t state_id{};
    std::uint64_t goal_state_id{};
    std::uint64_t preferred_action_id{};
    double remaining_steps{};
    double success_probability{1.0};
    std::uint64_t support{1ULL};
    std::uint64_t creation_step{};
    std::uint64_t last_used_step{};
};

struct WorldRouteExperience final {
    std::vector<WorldObservation> observations;
    std::vector<std::uint64_t> actions;
    bool success{true};
};

enum class Rlf3PlanStopReason {
    goal_reached,
    no_state_match,
    no_goal_match,
    no_transition,
    node_budget,
    depth_limit,
    exhausted,
};

[[nodiscard]] std::string_view to_string(Rlf3PlanStopReason reason) noexcept;

struct Rlf3PlanStep final {
    std::uint64_t state_id{};
    std::uint64_t context_id{};
    std::uint64_t action_id{};
    std::uint64_t predicted_next_state_id{};
    std::uint64_t predicted_next_context_id{};
    double outcome_probability{};
    double cumulative_cost{};
    double heuristic{};
};

struct Rlf3PlanResult final {
    bool success{false};
    Rlf3PlanStopReason stop_reason{Rlf3PlanStopReason::exhausted};
    std::vector<std::uint64_t> actions;
    std::vector<Rlf3PlanStep> steps;
    std::size_t nodes_expanded{};
    std::size_t transitions_evaluated{};
    std::size_t outcome_branches{};
    std::size_t subgoal_queries{};
    std::size_t subgoal_comparisons{};
    double predicted_success_probability{};
    double total_cost{};
};

struct SparseWorldModelStats final {
    std::uint64_t experiences_observed{};
    std::uint64_t states_created{};
    std::uint64_t states_merged{};
    std::uint64_t contexts_created{};
    std::uint64_t contexts_merged{};
    std::uint64_t transitions_created{};
    std::uint64_t outcomes_created{};
    std::uint64_t subgoals_created{};
    std::uint64_t subgoals_merged{};
    std::uint64_t state_index_queries{};
    std::uint64_t state_index_comparisons{};
    std::uint64_t state_index_fallbacks{};
};

struct SparseWorldModelSnapshot final {
    SparseWorldModelConfig config;
    std::uint64_t seed{};
    std::uint64_t training_step{};
    std::uint64_t next_action_id{};
    std::uint64_t next_state_id{};
    std::uint64_t next_context_id{};
    std::uint64_t next_transition_id{};
    std::uint64_t next_subgoal_id{};
    std::vector<WorldAction> actions;
    std::vector<WorldStatePrototype> states;
    std::vector<WorldContextPrototype> contexts;
    std::vector<WorldTransition> transitions;
    std::vector<SparseSubgoal> subgoals;
    SparseWorldModelStats stats;
};

class SparseWorldModel final {
public:
    explicit SparseWorldModel(
        SparseWorldModelConfig config,
        std::uint64_t seed
    );

    [[nodiscard]] const SparseWorldModelConfig& config() const noexcept;
    [[nodiscard]] std::uint64_t seed() const noexcept;
    [[nodiscard]] std::uint64_t training_step() const noexcept;
    [[nodiscard]] std::span<const WorldAction> actions() const noexcept;
    [[nodiscard]] std::span<const WorldStatePrototype> states() const noexcept;
    [[nodiscard]] std::span<const WorldContextPrototype> contexts() const noexcept;
    [[nodiscard]] std::span<const WorldTransition> transitions() const noexcept;
    [[nodiscard]] std::span<const SparseSubgoal> subgoals() const noexcept;
    [[nodiscard]] const SparseWorldModelStats& stats() const noexcept;

    std::uint64_t register_action(std::string name, double cost = 1.0);
    [[nodiscard]] const WorldAction& action_by_id(std::uint64_t action_id) const;
    [[nodiscard]] const WorldStatePrototype& state_by_id(
        std::uint64_t state_id
    ) const;
    [[nodiscard]] const WorldContextPrototype& context_by_id(
        std::uint64_t context_id
    ) const;

    void observe_transition(const WorldTransitionExperience& experience);
    void observe_successful_route(const WorldRouteExperience& route);

    [[nodiscard]] std::optional<std::uint64_t> match_state(
        const PhaseVector& observation,
        std::size_t* comparisons = nullptr
    ) const;
    [[nodiscard]] std::optional<std::uint64_t> match_context(
        const PhaseVector& context
    ) const;
    [[nodiscard]] std::optional<WorldPrediction> predict(
        const WorldObservation& observation,
        std::uint64_t action_id
    ) const;
    [[nodiscard]] std::optional<WorldPrediction> predict_by_id(
        std::uint64_t state_id,
        std::uint64_t context_id,
        std::uint64_t action_id
    ) const;

    [[nodiscard]] Rlf3PlanResult plan(
        const WorldObservation& start,
        const PhaseVector& goal,
        bool use_subgoal_index = true,
        std::size_t maximum_depth = 0U,
        std::size_t node_budget = 0U
    ) const;

    [[nodiscard]] double transition_top1_accuracy(
        std::span<const WorldTransitionExperience> experiences
    ) const;
    [[nodiscard]] double transition_negative_log_likelihood(
        std::span<const WorldTransitionExperience> experiences
    ) const;
    [[nodiscard]] double transition_brier_score(
        std::span<const WorldTransitionExperience> experiences
    ) const;

    [[nodiscard]] std::size_t estimated_bytes() const noexcept;
    [[nodiscard]] SparseWorldModelSnapshot snapshot() const;
    [[nodiscard]] static SparseWorldModel from_snapshot(
        SparseWorldModelSnapshot snapshot
    );

private:
    struct StateMatch final {
        std::uint64_t id{};
        double distance{};
        std::size_t comparisons{};
        bool found{false};
    };

    struct SubgoalEstimate final {
        double remaining_steps{};
        double success_probability{};
        std::uint64_t preferred_action_id{};
        std::size_t comparisons{};
        bool found{false};
    };

    [[nodiscard]] StateMatch find_state(
        const PhaseVector& observation,
        bool allow_fallback
    ) const;
    [[nodiscard]] StateMatch find_context(
        const PhaseVector& context
    ) const;
    [[nodiscard]] std::uint64_t match_or_create_state(
        const PhaseVector& observation
    );
    [[nodiscard]] std::uint64_t match_or_create_context(
        const PhaseVector& context
    );
    [[nodiscard]] std::uint64_t phase_bucket_hash(
        const PhaseVector& vector
    ) const noexcept;
    [[nodiscard]] std::vector<std::uint64_t> nearby_bucket_hashes(
        const PhaseVector& vector
    ) const;
    [[nodiscard]] SubgoalEstimate estimate_subgoal(
        std::uint64_t state_id,
        std::uint64_t goal_state_id
    ) const;
    void rebuild_indices();
    void validate_snapshot() const;

    SparseWorldModelConfig config_;
    std::uint64_t seed_{};
    std::uint64_t training_step_{};
    std::uint64_t next_action_id_{1ULL};
    std::uint64_t next_state_id_{1ULL};
    std::uint64_t next_context_id_{1ULL};
    std::uint64_t next_transition_id_{1ULL};
    std::uint64_t next_subgoal_id_{1ULL};
    std::vector<WorldAction> actions_;
    std::vector<WorldStatePrototype> states_;
    std::vector<WorldContextPrototype> contexts_;
    std::vector<WorldTransition> transitions_;
    std::vector<SparseSubgoal> subgoals_;
    SparseWorldModelStats stats_;
    std::vector<std::size_t> hash_coordinates_;
    std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> state_buckets_;
    std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> context_buckets_;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> transition_index_;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> subgoal_index_;
};

}  // namespace rlf::core
