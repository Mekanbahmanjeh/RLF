#pragma once

#include "rlf/core/deterministic_rng.hpp"
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

enum class LatentCreditStrategy {
    uniform_route,
    discounted_eligibility,
    progress_weighted,
    counterfactual_local,
};

enum class LatentHaltPolicy {
    goal_threshold,
    learned_resonant,
    combined_safe,
};

enum class LatentStopReason {
    successful_halt,
    learned_halt,
    premature_halt,
    cycle_limit,
    loop_detected,
    abstained,
    no_candidate,
};

[[nodiscard]] std::string_view to_string(
    LatentCreditStrategy strategy
) noexcept;
[[nodiscard]] std::string_view to_string(
    LatentHaltPolicy policy
) noexcept;
[[nodiscard]] std::string_view to_string(
    LatentStopReason reason
) noexcept;

struct LatentRouterConfig final {
    std::size_t dimension{48U};
    std::size_t maximum_cycles{32U};
    std::size_t search_node_budget{100'000U};
    std::size_t route_memory_capacity{4'096U};
    std::size_t maximum_modes{100'000U};
    std::size_t macro_minimum_occurrences{4U};
    std::size_t macro_maximum_length{6U};
    double goal_similarity_threshold{0.9995};
    double mode_creation_similarity{0.92};
    double mode_learning_rate{0.15};
    double utility_learning_rate{0.10};
    double eligibility_decay{0.85};
    double goal_progress_weight{0.20};
    double route_memory_weight{0.10};
    double successor_familiarity_weight{0.35};
    double route_repetition_penalty{1.0};
    double learned_halt_threshold{0.97};
    double learned_halt_goal_floor{0.90};
    double abstention_entropy_threshold{0.98};
    double minimum_action_score{-0.25};
    double action_temperature{0.25};
    double abstention_resonance_threshold{0.05};
    std::size_t search_beam_width{8U};
    std::size_t search_lookahead_depth{4U};
    bool enable_route_memory{true};
    bool enable_macro_operators{true};
    LatentCreditStrategy credit_strategy{
        LatentCreditStrategy::discounted_eligibility
    };
    LatentHaltPolicy halt_policy{LatentHaltPolicy::combined_safe};
};

struct RegisteredOperator final {
    std::uint64_t id{};
    std::string name;
    TransformationOperator transformation{1U};
    double cost{1.0};
    bool macro{false};
    std::vector<std::uint64_t> primitive_route;
};

struct LatentRoutingMode final {
    std::uint64_t id{};
    PhaseVector key{std::vector<float>{0.0F}};
    std::uint64_t operator_id{};
    double utility{};
    double confidence{0.25};
    double eligibility{};
    std::uint64_t activation_count{};
    std::uint64_t success_count{};
    std::uint64_t failure_count{};
    std::uint64_t creation_step{};
    std::uint64_t last_used_step{};
};

struct LatentHaltMode final {
    std::uint64_t id{};
    PhaseVector key{std::vector<float>{0.0F}};
    double confidence{0.5};
    double utility{};
    std::uint64_t activation_count{};
};

struct RouteMemoryRecord final {
    std::uint64_t id{};
    PhaseVector start_goal_signature{std::vector<float>{0.0F}};
    std::vector<std::uint64_t> route;
    double confidence{0.5};
    double utility{};
    std::uint64_t usage_count{};
    std::uint64_t observation_count{1ULL};
    std::uint64_t creation_step{};
    std::uint64_t last_used_step{};
    std::uint64_t route_hash{};
};

struct LatentActionCandidate final {
    std::uint64_t operator_id{};
    std::uint64_t routing_mode_id{};
    double resonance{};
    double utility{};
    double eligibility{};
    double immediate_progress{};
    double route_memory_bonus{};
    double successor_familiarity{};
    double score{};
    double normalized_weight{};
};

struct LatentEpisodeState final {
    PhaseVector current_state{std::vector<float>{0.0F}};
    PhaseVector goal_state{std::vector<float>{0.0F}};
    PhaseVector working_state{std::vector<float>{0.0F}};
    PhaseVector memory_summary{std::vector<float>{0.0F}};
    PhaseVector route_summary{std::vector<float>{0.0F}};
    double uncertainty_state{};
    double progress_state{};
    std::size_t step_index{};
};

struct LatentTraceStep final {
    std::size_t cycle{};
    std::uint64_t state_hash{};
    std::uint64_t working_state_hash{};
    std::uint64_t goal_state_hash{};
    std::uint64_t memory_summary_hash{};
    std::uint64_t route_summary_hash{};
    double goal_similarity_before{};
    double goal_similarity_after{};
    double progress{};
    double uncertainty{};
    double halt_score{};
    std::uint64_t selected_operator_id{};
    std::string selected_operator_name;
    std::uint64_t selected_mode_id{};
    std::vector<LatentActionCandidate> candidates;
};

struct LatentExecutionResult final {
    PhaseVector final_state{std::vector<float>{0.0F}};
    std::vector<std::uint64_t> route;
    std::vector<LatentTraceStep> trace;
    LatentStopReason stop_reason{LatentStopReason::cycle_limit};
    bool success{false};
    bool abstained{false};
    std::size_t cycles{};
    double final_goal_similarity{};
    double mean_uncertainty{};
    std::size_t exact_similarity_evaluations{};
    std::size_t active_mode_evaluations{};
    std::size_t search_nodes{};
};

struct LatentTrainingResult final {
    bool route_found{false};
    bool success{false};
    std::vector<std::uint64_t> discovered_route;
    std::size_t search_nodes{};
    double terminal_reward{};
    std::size_t modes_created{};
    std::size_t modes_updated{};
};

struct LatentRouterSnapshot final {
    LatentRouterConfig config;
    std::uint64_t seed{};
    std::uint64_t training_step{};
    std::uint64_t next_operator_id{};
    std::uint64_t next_mode_id{};
    std::uint64_t next_halt_mode_id{};
    std::uint64_t next_route_record_id{};
    std::vector<RegisteredOperator> operators;
    std::vector<LatentRoutingMode> modes;
    std::vector<LatentHaltMode> halt_modes;
    std::vector<RouteMemoryRecord> route_memory;
};

class LatentRouter final {
public:
    explicit LatentRouter(
        LatentRouterConfig config,
        std::uint64_t seed
    );

    [[nodiscard]] const LatentRouterConfig& config() const noexcept;
    [[nodiscard]] std::uint64_t seed() const noexcept;
    [[nodiscard]] std::uint64_t training_step() const noexcept;
    [[nodiscard]] std::span<const RegisteredOperator> operators() const noexcept;
    [[nodiscard]] std::span<const LatentRoutingMode> modes() const noexcept;
    [[nodiscard]] std::span<const LatentHaltMode> halt_modes() const noexcept;
    [[nodiscard]] std::span<const RouteMemoryRecord> route_memory() const noexcept;

    std::uint64_t register_operator(
        std::string name,
        TransformationOperator transformation,
        double cost = 1.0,
        bool macro = false,
        std::vector<std::uint64_t> primitive_route = {}
    );

    [[nodiscard]] const RegisteredOperator& operator_by_id(
        std::uint64_t operator_id
    ) const;

    [[nodiscard]] LatentExecutionResult execute(
        const PhaseVector& start,
        const PhaseVector& goal,
        std::optional<std::uint64_t> forced_first_operator = std::nullopt,
        bool allow_route_memory = true
    );

    [[nodiscard]] LatentExecutionResult execute_with_bounded_lookahead(
        const PhaseVector& start,
        const PhaseVector& goal,
        std::size_t lookahead_depth,
        std::size_t beam_width,
        bool allow_route_memory = true
    );

    [[nodiscard]] LatentTrainingResult train_episode(
        const PhaseVector& start,
        const PhaseVector& goal,
        std::size_t maximum_search_depth
    );

    [[nodiscard]] std::optional<std::vector<std::uint64_t>> discover_route(
        const PhaseVector& start,
        const PhaseVector& goal,
        std::size_t maximum_depth,
        std::size_t* explored_nodes = nullptr
    ) const;

    void reinforce_route(
        const PhaseVector& start,
        const PhaseVector& goal,
        std::span<const std::uint64_t> route,
        double terminal_reward
    );

    [[nodiscard]] std::size_t consolidate_macros();
    [[nodiscard]] LatentRouterSnapshot snapshot() const;
    [[nodiscard]] static LatentRouter from_snapshot(
        LatentRouterSnapshot snapshot
    );

    [[nodiscard]] static PhaseVector state_goal_signature(
        const PhaseVector& current,
        const PhaseVector& goal
    );
    [[nodiscard]] static std::uint64_t phase_state_hash(
        const PhaseVector& state
    ) noexcept;
    [[nodiscard]] static std::uint64_t episode_state_hash(
        const LatentEpisodeState& state
    ) noexcept;
    [[nodiscard]] static std::uint64_t route_hash(
        std::span<const std::uint64_t> route
    ) noexcept;

private:
    struct ModeSelection final {
        std::size_t mode_index{};
        double resonance{};
        bool found{false};
    };

    [[nodiscard]] ModeSelection best_mode_for_operator(
        const PhaseVector& signature,
        std::uint64_t operator_id,
        std::size_t* similarity_evaluations = nullptr
    ) const;
    [[nodiscard]] std::vector<LatentActionCandidate> score_actions(
        const PhaseVector& current,
        const PhaseVector& goal,
        bool allow_route_memory,
        bool record_memory_usage,
        std::size_t* similarity_evaluations
    );
    [[nodiscard]] double state_familiarity(
        const PhaseVector& state,
        const PhaseVector& goal,
        std::size_t* similarity_evaluations
    ) const;
    [[nodiscard]] std::optional<std::uint64_t> lookahead_action(
        const PhaseVector& current,
        const PhaseVector& goal,
        std::size_t depth,
        std::size_t beam_width,
        std::size_t* search_nodes,
        std::size_t* similarity_evaluations
    );
    [[nodiscard]] LatentExecutionResult execute_internal(
        const PhaseVector& start,
        const PhaseVector& goal,
        std::optional<std::uint64_t> forced_first_operator,
        bool allow_route_memory,
        bool use_lookahead,
        std::size_t lookahead_depth,
        std::size_t beam_width
    );
    [[nodiscard]] double halt_score(
        const PhaseVector& signature,
        std::size_t* similarity_evaluations
    ) const;
    [[nodiscard]] std::optional<std::uint64_t> route_memory_suggestion(
        const PhaseVector& signature,
        double* similarity,
        std::size_t* similarity_evaluations,
        bool record_usage
    );
    [[nodiscard]] std::size_t create_or_update_mode(
        const PhaseVector& signature,
        std::uint64_t operator_id,
        double local_progress,
        double terminal_credit,
        std::size_t* created_count
    );
    void create_or_update_halt_mode(const PhaseVector& signature);
    void remember_route(
        const PhaseVector& start,
        const PhaseVector& goal,
        std::span<const std::uint64_t> route,
        double reward
    );
    [[nodiscard]] double candidate_uncertainty(
        std::vector<LatentActionCandidate>& candidates
    ) const;
    [[nodiscard]] PhaseVector candidate_memory_summary(
        std::span<const LatentActionCandidate> candidates
    ) const;
    [[nodiscard]] static PhaseVector operator_route_code(
        std::size_t dimension,
        std::uint64_t operator_id
    );
    [[nodiscard]] bool is_goal(
        const PhaseVector& state,
        const PhaseVector& goal
    ) const;

    LatentRouterConfig config_;
    std::uint64_t seed_{};
    DeterministicRng rng_;
    std::uint64_t training_step_{};
    std::uint64_t next_operator_id_{1ULL};
    std::uint64_t next_mode_id_{1ULL};
    std::uint64_t next_halt_mode_id_{1ULL};
    std::uint64_t next_route_record_id_{1ULL};
    std::vector<RegisteredOperator> operators_;
    std::vector<LatentRoutingMode> modes_;
    std::vector<LatentHaltMode> halt_modes_;
    std::vector<RouteMemoryRecord> route_memory_;
    std::unordered_map<std::uint64_t, std::size_t> route_occurrences_;
};

}  // namespace rlf::core
