#include "rlf/experiments/associative_recall.hpp"
#include "rlf/experiments/capacity_scaling.hpp"
#include "rlf/experiments/checkpoint_workflow.hpp"
#include "rlf/experiments/compositional_generalization.hpp"
#include "rlf/experiments/continual_learning.hpp"
#include "rlf/experiments/operator_composition.hpp"
#include "rlf/experiments/optimization_benchmark.hpp"
#include "rlf/experiments/phase_vector_smoke.hpp"
#include "rlf/experiments/persistence_roundtrip.hpp"
#include "rlf/experiments/rlf1_latent_routing.hpp"
#include "rlf/experiments/rlf2_predictive_reasoning.hpp"
#include "rlf/experiments/rlf3_world_model.hpp"
#include "rlf/experiments/rlf4_self_supervised.hpp"
#include "rlf/experiments/rlf5_language.hpp"
#include "rlf/experiments/rlf6_agent.hpp"
#include "rlf/experiments/rlf7_frontier.hpp"
#include "rlf/frontier/frontier_trainer.hpp"
#include "rlf/frontier/multimodal.hpp"
#include "rlf/experiments/sequence_completion.hpp"
#include "rlf/experiments/structural_adaptation.hpp"
#include "rlf/experiments/transformation_learning.hpp"
#include "rlf/storage/checkpoint.hpp"
#include "rlf/storage/rlf1_checkpoint.hpp"
#include "rlf/storage/rlf2_checkpoint.hpp"
#include "rlf/storage/rlf3_checkpoint.hpp"
#include "rlf/storage/rlf4_checkpoint.hpp"
#include "rlf/storage/rlf5_checkpoint.hpp"
#include "rlf/storage/rlf6_checkpoint.hpp"
#include "rlf/storage/rlf7_checkpoint.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

struct CliOptions final {
    std::string command;
    std::string experiment_name;
    std::string suite_name;
    std::string task_name;
    std::filesystem::path config_path;
    std::filesystem::path output_path;
    std::filesystem::path checkpoint_path;
    std::filesystem::path manifest_path;
    std::filesystem::path generation_path;
    std::uint64_t seed{0x524C4630ULL};
    std::size_t dimension{1'024};
    std::size_t samples{64};
    std::size_t training_examples{256};
    std::size_t development_examples{24};
    std::size_t evaluation_examples{128};
    std::size_t training_examples_per_task{256};
    std::size_t evaluation_examples_per_task{128};
    std::size_t association_count{256};
    std::size_t symbol_count{8};
    std::size_t mode_count{8};
    std::size_t memory_records{32};
    std::size_t evaluation_queries{16};
    std::size_t candidate_count{256};
    std::size_t active_count{1};
    std::size_t context_dimensions{8U};
    std::size_t quantization_samples{256U};
    std::size_t training_min_route_length{1U};
    std::size_t training_max_route_length{4U};
    std::size_t evaluation_min_route_length{5U};
    std::size_t evaluation_max_route_length{8U};
    std::size_t maximum_cycles{16U};
    std::size_t operator_count{8U};
    double state_noise_radians{0.03};
    double goal_similarity_threshold{0.9995};
    std::size_t world_layers{7U};
    std::size_t world_lanes{5U};
    std::size_t transition_samples_per_case{12U};
    std::size_t training_routes{240U};
    std::size_t stochastic_rollouts{3U};
    std::size_t maximum_execution_steps{20U};
    std::size_t planner_node_budget{50'000U};
    std::size_t maximum_plan_depth{16U};
    double observation_noise_radians{0.012};
    std::size_t temporal_training_tokens{24'000U};
    std::size_t temporal_evaluation_tokens{6'000U};
    std::size_t temporal_adaptation_tokens{6'000U};
    std::size_t temporal_context_order{12U};
    std::size_t temporal_maximum_options{2'048U};
    std::size_t temporal_minimum_option_support{6U};
    std::size_t temporal_forecast_horizon{8U};
    std::size_t temporal_forecast_samples{256U};
    std::size_t temporal_change_tolerance{96U};
    double temporal_training_noise_radians{0.018};
    double temporal_evaluation_noise_radians{0.035};
    double temporal_prototype_merge_distance{0.10};
    double temporal_recent_decay{0.997};
    double temporal_recent_weight{0.70};
    std::size_t language_raw_training_sentences{12'000U};
    std::size_t language_supervised_training_examples{6'000U};
    std::size_t language_evaluation_examples{1'200U};
    std::size_t language_qa_episodes{400U};
    std::size_t language_free_generation_samples{128U};
    std::size_t language_maximum_lexemes{1'024U};
    std::size_t language_maximum_merges{512U};
    std::size_t language_minimum_pair_support{6U};
    std::size_t language_context_order{8U};
    std::size_t language_minimum_context_support{2U};
    std::size_t language_maximum_constructions{4'096U};
    std::size_t language_minimum_construction_support{3U};
    std::size_t language_maximum_generation_tokens{96U};
    std::size_t language_holdout_modulus{7U};
    std::size_t frontier_knowledge_records{10'000U};
    std::size_t frontier_knowledge_queries{1'000U};
    std::string frontier_backend{"optimized_cpu"};
    bool frontier_include_audio{true};
    bool frontier_run_agent_gate{};
    std::size_t rlf6_training_episodes{40U};
    std::size_t rlf6_evaluation_episodes{30U};
    std::size_t rlf6_minimum_route_length{5U};
    std::size_t rlf6_maximum_route_length{60U};
    std::size_t rlf6_stress_episodes{4U};
    std::size_t rlf6_stress_route_length{110U};
    std::size_t rlf6_action_budget{220U};
    std::size_t rlf6_tool_budget{96U};
    std::size_t rlf6_memory_limit_records{32'768U};
    double rlf6_tool_cost_budget{10'000.0};
    double rlf6_risk_budget{25.0};
    bool rlf6_include_stress{true};
    std::size_t sample_id{0};
    std::vector<std::size_t> mode_counts{
        1'024U,
        4'096U,
        16'384U,
        65'536U,
    };
    double context_noise_radians{0.05};
    double noise_radians{0.15};
    double corruption_radians{0.12};
    double dominant_probability{0.8};
    std::size_t threads{1};
    std::uint64_t max_memory_bytes{0ULL};
    std::string log_level{"info"};
    bool deterministic{true};
    bool show_help{false};
};

[[nodiscard]] std::string trim(const std::string_view input) {
    const auto first = std::find_if_not(
        input.begin(),
        input.end(),
        [](const unsigned char character) {
            return std::isspace(character) != 0;
        }
    );
    const auto last = std::find_if_not(
        input.rbegin(),
        input.rend(),
        [](const unsigned char character) {
            return std::isspace(character) != 0;
        }
    ).base();
    if (first >= last) {
        return {};
    }
    return std::string(first, last);
}

template <typename Integer>
[[nodiscard]] Integer parse_integer(
    const std::string_view text,
    const std::string_view option_name
) {
    Integer value{};
    const char* const begin = text.data();
    const char* const end = text.data() + text.size();
    const auto [parsed_end, error] = std::from_chars(begin, end, value);
    if (error != std::errc{} || parsed_end != end) {
        throw std::invalid_argument(
            "invalid integer for " + std::string(option_name) + ": " +
            std::string(text)
        );
    }
    return value;
}

[[nodiscard]] double parse_floating(
    const std::string_view text,
    const std::string_view option_name
) {
    double value{};
    const char* const begin = text.data();
    const char* const end = text.data() + text.size();
    const auto [parsed_end, error] = std::from_chars(begin, end, value);
    if (error != std::errc{} || parsed_end != end) {
        throw std::invalid_argument(
            "invalid floating-point value for " +
            std::string(option_name) + ": " + std::string(text)
        );
    }
    return value;
}

[[nodiscard]] std::vector<std::size_t> parse_size_list(
    const std::string_view text,
    const std::string_view option_name
) {
    std::vector<std::size_t> values;
    std::size_t start = 0U;
    while (start <= text.size()) {
        const std::size_t separator = text.find(',', start);
        const std::size_t length = separator == std::string_view::npos
            ? text.size() - start
            : separator - start;
        const std::string item = trim(text.substr(start, length));
        if (item.empty()) {
            throw std::invalid_argument(
                "empty value in " + std::string(option_name)
            );
        }
        values.push_back(
            parse_integer<std::size_t>(item, option_name)
        );
        if (separator == std::string_view::npos) {
            break;
        }
        start = separator + 1U;
    }
    if (values.empty()) {
        throw std::invalid_argument(
            std::string(option_name) + " must not be empty"
        );
    }
    return values;
}

[[nodiscard]] bool parse_boolean(
    const std::string_view text,
    const std::string_view option_name
) {
    if (text == "true" || text == "1" || text == "yes") {
        return true;
    }
    if (text == "false" || text == "0" || text == "no") {
        return false;
    }
    throw std::invalid_argument(
        "invalid boolean for " + std::string(option_name) + ": " +
        std::string(text)
    );
}

[[nodiscard]] std::uint64_t checked_multiply(
    const std::uint64_t left,
    const std::uint64_t right,
    const std::string_view option_name
) {
    if (right != 0ULL &&
        left > std::numeric_limits<std::uint64_t>::max() / right) {
        throw std::invalid_argument(
            std::string(option_name) + " exceeds the supported range"
        );
    }
    return left * right;
}

[[nodiscard]] std::uint64_t checked_add(
    const std::uint64_t left,
    const std::uint64_t right,
    const std::string_view option_name
) {
    if (left >
        std::numeric_limits<std::uint64_t>::max() - right) {
        throw std::invalid_argument(
            std::string(option_name) + " exceeds the supported range"
        );
    }
    return left + right;
}

[[nodiscard]] std::uint64_t parse_memory_size(
    const std::string_view text,
    const std::string_view option_name
) {
    std::size_t number_length = 0U;
    while (number_length < text.size() &&
           std::isdigit(static_cast<unsigned char>(text[number_length])) != 0) {
        ++number_length;
    }
    if (number_length == 0U) {
        throw std::invalid_argument(
            "invalid memory size for " + std::string(option_name)
        );
    }

    const std::uint64_t value = parse_integer<std::uint64_t>(
        text.substr(0U, number_length),
        option_name
    );
    const std::string suffix = trim(text.substr(number_length));
    std::uint64_t multiplier = 1ULL;
    if (suffix.empty() || suffix == "B") {
        multiplier = 1ULL;
    } else if (suffix == "KiB") {
        multiplier = 1'024ULL;
    } else if (suffix == "MiB") {
        multiplier = 1'024ULL * 1'024ULL;
    } else if (suffix == "GiB") {
        multiplier = 1'024ULL * 1'024ULL * 1'024ULL;
    } else if (suffix == "KB") {
        multiplier = 1'000ULL;
    } else if (suffix == "MB") {
        multiplier = 1'000ULL * 1'000ULL;
    } else if (suffix == "GB") {
        multiplier = 1'000ULL * 1'000ULL * 1'000ULL;
    } else {
        throw std::invalid_argument(
            "unsupported memory suffix for " + std::string(option_name)
        );
    }
    return checked_multiply(value, multiplier, option_name);
}

void validate_log_level(const std::string_view log_level) {
    if (log_level != "trace" && log_level != "debug" &&
        log_level != "info" && log_level != "warn" &&
        log_level != "error") {
        throw std::invalid_argument("invalid --log-level value");
    }
}

void apply_config_entry(
    CliOptions& options,
    const std::string_view key,
    const std::string_view value
) {
    if (key == "seed") {
        options.seed = parse_integer<std::uint64_t>(value, "seed");
    } else if (key == "dimension") {
        options.dimension = parse_integer<std::size_t>(value, "dimension");
    } else if (key == "samples") {
        options.samples = parse_integer<std::size_t>(value, "samples");
    } else if (key == "training_examples") {
        options.training_examples =
            parse_integer<std::size_t>(value, "training_examples");
    } else if (key == "evaluation_examples") {
        options.evaluation_examples =
            parse_integer<std::size_t>(value, "evaluation_examples");
    } else if (key == "training_examples_per_task") {
        options.training_examples_per_task = parse_integer<std::size_t>(
            value,
            "training_examples_per_task"
        );
    } else if (key == "evaluation_examples_per_task") {
        options.evaluation_examples_per_task = parse_integer<std::size_t>(
            value,
            "evaluation_examples_per_task"
        );
    } else if (key == "association_count") {
        options.association_count =
            parse_integer<std::size_t>(value, "association_count");
    } else if (key == "symbol_count") {
        options.symbol_count =
            parse_integer<std::size_t>(value, "symbol_count");
    } else if (key == "context_noise_radians") {
        options.context_noise_radians =
            parse_floating(value, "context_noise_radians");
    } else if (key == "noise_radians") {
        options.noise_radians =
            parse_floating(value, "noise_radians");
    } else if (key == "corruption_radians") {
        options.corruption_radians =
            parse_floating(value, "corruption_radians");
    } else if (key == "dominant_probability") {
        options.dominant_probability =
            parse_floating(value, "dominant_probability");
    } else if (key == "threads") {
        options.threads = parse_integer<std::size_t>(value, "threads");
    } else if (key == "output") {
        options.output_path = std::string(value);
    } else if (key == "checkpoint") {
        options.checkpoint_path = std::string(value);
    } else if (key == "manifest") {
        options.manifest_path = std::string(value);
    } else if (key == "generate_dir") {
        options.generation_path = std::string(value);
    } else if (key == "frontier_knowledge_records") {
        options.frontier_knowledge_records = parse_integer<std::size_t>(value, key);
    } else if (key == "frontier_knowledge_queries") {
        options.frontier_knowledge_queries = parse_integer<std::size_t>(value, key);
    } else if (key == "frontier_backend") {
        options.frontier_backend = std::string(value);
    } else if (key == "frontier_include_audio") {
        options.frontier_include_audio = parse_boolean(value, key);
    } else if (key == "frontier_run_agent_gate") {
        options.frontier_run_agent_gate = parse_boolean(value, key);
    } else if (key == "mode_count") {
        options.mode_count =
            parse_integer<std::size_t>(value, "mode_count");
    } else if (key == "memory_records") {
        options.memory_records =
            parse_integer<std::size_t>(value, "memory_records");
    } else if (key == "evaluation_queries") {
        options.evaluation_queries =
            parse_integer<std::size_t>(value, "evaluation_queries");
    } else if (key == "candidate_count") {
        options.candidate_count =
            parse_integer<std::size_t>(value, "candidate_count");
    } else if (key == "active_count") {
        options.active_count =
            parse_integer<std::size_t>(value, "active_count");
    } else if (key == "context_dimensions") {
        options.context_dimensions =
            parse_integer<std::size_t>(value, "context_dimensions");
    } else if (key == "quantization_samples") {
        options.quantization_samples =
            parse_integer<std::size_t>(value, "quantization_samples");
    } else if (key == "development_examples") {
        options.development_examples =
            parse_integer<std::size_t>(value, "development_examples");
    } else if (key == "training_min_route_length") {
        options.training_min_route_length =
            parse_integer<std::size_t>(value, "training_min_route_length");
    } else if (key == "training_max_route_length") {
        options.training_max_route_length =
            parse_integer<std::size_t>(value, "training_max_route_length");
    } else if (key == "evaluation_min_route_length") {
        options.evaluation_min_route_length =
            parse_integer<std::size_t>(value, "evaluation_min_route_length");
    } else if (key == "evaluation_max_route_length") {
        options.evaluation_max_route_length =
            parse_integer<std::size_t>(value, "evaluation_max_route_length");
    } else if (key == "maximum_cycles") {
        options.maximum_cycles =
            parse_integer<std::size_t>(value, "maximum_cycles");
    } else if (key == "operator_count") {
        options.operator_count =
            parse_integer<std::size_t>(value, "operator_count");
    } else if (key == "goal_similarity_threshold") {
        options.goal_similarity_threshold =
            parse_floating(value, "goal_similarity_threshold");
    } else if (key == "state_noise_radians") {
        options.state_noise_radians =
            parse_floating(value, "state_noise_radians");
    } else if (key == "world_layers") {
        options.world_layers = parse_integer<std::size_t>(value, "world_layers");
    } else if (key == "world_lanes") {
        options.world_lanes = parse_integer<std::size_t>(value, "world_lanes");
    } else if (key == "transition_samples_per_case") {
        options.transition_samples_per_case = parse_integer<std::size_t>(
            value, "transition_samples_per_case");
    } else if (key == "training_routes") {
        options.training_routes = parse_integer<std::size_t>(value, "training_routes");
    } else if (key == "stochastic_rollouts") {
        options.stochastic_rollouts = parse_integer<std::size_t>(value, "stochastic_rollouts");
    } else if (key == "maximum_execution_steps") {
        options.maximum_execution_steps = parse_integer<std::size_t>(
            value, "maximum_execution_steps");
    } else if (key == "planner_node_budget") {
        options.planner_node_budget = parse_integer<std::size_t>(value, "planner_node_budget");
    } else if (key == "maximum_plan_depth") {
        options.maximum_plan_depth = parse_integer<std::size_t>(value, "maximum_plan_depth");
    } else if (key == "observation_noise_radians") {
        options.observation_noise_radians = parse_floating(
            value, "observation_noise_radians");
    } else if (key == "temporal_training_tokens") {
        options.temporal_training_tokens = parse_integer<std::size_t>(value, key);
    } else if (key == "temporal_evaluation_tokens") {
        options.temporal_evaluation_tokens = parse_integer<std::size_t>(value, key);
    } else if (key == "temporal_adaptation_tokens") {
        options.temporal_adaptation_tokens = parse_integer<std::size_t>(value, key);
    } else if (key == "temporal_context_order") {
        options.temporal_context_order = parse_integer<std::size_t>(value, key);
    } else if (key == "temporal_maximum_options") {
        options.temporal_maximum_options = parse_integer<std::size_t>(value, key);
    } else if (key == "temporal_minimum_option_support") {
        options.temporal_minimum_option_support = parse_integer<std::size_t>(value, key);
    } else if (key == "temporal_forecast_horizon") {
        options.temporal_forecast_horizon = parse_integer<std::size_t>(value, key);
    } else if (key == "temporal_forecast_samples") {
        options.temporal_forecast_samples = parse_integer<std::size_t>(value, key);
    } else if (key == "temporal_change_tolerance") {
        options.temporal_change_tolerance = parse_integer<std::size_t>(value, key);
    } else if (key == "temporal_training_noise_radians") {
        options.temporal_training_noise_radians = parse_floating(value, key);
    } else if (key == "temporal_evaluation_noise_radians") {
        options.temporal_evaluation_noise_radians = parse_floating(value, key);
    } else if (key == "temporal_prototype_merge_distance") {
        options.temporal_prototype_merge_distance = parse_floating(value, key);
    } else if (key == "temporal_recent_decay") {
        options.temporal_recent_decay = parse_floating(value, key);
    } else if (key == "temporal_recent_weight") {
        options.temporal_recent_weight = parse_floating(value, key);
    } else if (key == "language_raw_training_sentences") {
        options.language_raw_training_sentences = parse_integer<std::size_t>(value, key);
    } else if (key == "language_supervised_training_examples") {
        options.language_supervised_training_examples = parse_integer<std::size_t>(value, key);
    } else if (key == "language_evaluation_examples") {
        options.language_evaluation_examples = parse_integer<std::size_t>(value, key);
    } else if (key == "language_qa_episodes") {
        options.language_qa_episodes = parse_integer<std::size_t>(value, key);
    } else if (key == "language_free_generation_samples") {
        options.language_free_generation_samples = parse_integer<std::size_t>(value, key);
    } else if (key == "language_maximum_lexemes") {
        options.language_maximum_lexemes = parse_integer<std::size_t>(value, key);
    } else if (key == "language_maximum_merges") {
        options.language_maximum_merges = parse_integer<std::size_t>(value, key);
    } else if (key == "language_minimum_pair_support") {
        options.language_minimum_pair_support = parse_integer<std::size_t>(value, key);
    } else if (key == "language_context_order") {
        options.language_context_order = parse_integer<std::size_t>(value, key);
    } else if (key == "language_minimum_context_support") {
        options.language_minimum_context_support = parse_integer<std::size_t>(value, key);
    } else if (key == "language_maximum_constructions") {
        options.language_maximum_constructions = parse_integer<std::size_t>(value, key);
    } else if (key == "language_minimum_construction_support") {
        options.language_minimum_construction_support = parse_integer<std::size_t>(value, key);
    } else if (key == "language_maximum_generation_tokens") {
        options.language_maximum_generation_tokens = parse_integer<std::size_t>(value, key);
    } else if (key == "language_holdout_modulus") {
        options.language_holdout_modulus = parse_integer<std::size_t>(value, key);
    } else if (key == "rlf6_training_episodes") {
        options.rlf6_training_episodes = parse_integer<std::size_t>(value, key);
    } else if (key == "rlf6_evaluation_episodes") {
        options.rlf6_evaluation_episodes = parse_integer<std::size_t>(value, key);
    } else if (key == "rlf6_minimum_route_length") {
        options.rlf6_minimum_route_length = parse_integer<std::size_t>(value, key);
    } else if (key == "rlf6_maximum_route_length") {
        options.rlf6_maximum_route_length = parse_integer<std::size_t>(value, key);
    } else if (key == "rlf6_stress_episodes") {
        options.rlf6_stress_episodes = parse_integer<std::size_t>(value, key);
    } else if (key == "rlf6_stress_route_length") {
        options.rlf6_stress_route_length = parse_integer<std::size_t>(value, key);
    } else if (key == "rlf6_action_budget") {
        options.rlf6_action_budget = parse_integer<std::size_t>(value, key);
    } else if (key == "rlf6_tool_budget") {
        options.rlf6_tool_budget = parse_integer<std::size_t>(value, key);
    } else if (key == "rlf6_memory_limit_records") {
        options.rlf6_memory_limit_records = parse_integer<std::size_t>(value, key);
    } else if (key == "rlf6_tool_cost_budget") {
        options.rlf6_tool_cost_budget = parse_floating(value, key);
    } else if (key == "rlf6_risk_budget") {
        options.rlf6_risk_budget = parse_floating(value, key);
    } else if (key == "rlf6_include_stress") {
        options.rlf6_include_stress = parse_boolean(value, key);
    } else if (key == "mode_counts") {
        options.mode_counts = parse_size_list(value, "mode_counts");
    } else if (key == "task") {
        options.task_name = std::string(value);
    } else if (key == "suite") {
        options.suite_name = std::string(value);
    } else if (key == "name") {
        options.experiment_name = std::string(value);
    } else if (key == "sample") {
        options.sample_id =
            parse_integer<std::size_t>(value, "sample");
    } else if (key == "log_level") {
        options.log_level = std::string(value);
        validate_log_level(options.log_level);
    } else if (key == "deterministic") {
        options.deterministic = parse_boolean(value, "deterministic");
    } else if (key == "max_memory") {
        options.max_memory_bytes = parse_memory_size(value, "max_memory");
    } else {
        throw std::invalid_argument(
            "unknown configuration key: " + std::string(key)
        );
    }
}

void load_config(CliOptions& options, const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
            "unable to open configuration file: " + path.string()
        );
    }

    std::string line;
    std::size_t line_number = 0U;
    while (std::getline(input, line)) {
        ++line_number;
        const std::size_t comment_position = line.find('#');
        const std::string content = trim(
            std::string_view(line).substr(0U, comment_position)
        );
        if (content.empty()) {
            continue;
        }

        const std::size_t separator = content.find('=');
        if (separator == std::string::npos) {
            throw std::runtime_error(
                "expected key=value at " + path.string() + ":" +
                std::to_string(line_number)
            );
        }

        const std::string key = trim(
            std::string_view(content).substr(0U, separator)
        );
        const std::string value = trim(
            std::string_view(content).substr(separator + 1U)
        );
        if (key.empty() || value.empty()) {
            throw std::runtime_error(
                "empty key or value at " + path.string() + ":" +
                std::to_string(line_number)
            );
        }
        apply_config_entry(options, key, value);
    }
    if (!input.eof()) {
        throw std::runtime_error(
            "failed while reading configuration file: " + path.string()
        );
    }
}

[[nodiscard]] std::string_view require_value(
    const int argument_count,
    char** argument_values,
    int& argument_index,
    const std::string_view option_name
) {
    if (argument_index + 1 >= argument_count) {
        throw std::invalid_argument(
            "missing value for " + std::string(option_name)
        );
    }
    ++argument_index;
    return argument_values[argument_index];
}

[[nodiscard]] std::optional<std::filesystem::path> find_config_path(
    const int argument_count,
    char** argument_values
) {
    std::optional<std::filesystem::path> config_path;
    for (int argument_index = 1;
         argument_index < argument_count;
         ++argument_index) {
        if (std::string_view(argument_values[argument_index]) == "--config") {
            const std::string_view value = require_value(
                argument_count,
                argument_values,
                argument_index,
                "--config"
            );
            if (config_path.has_value()) {
                throw std::invalid_argument("--config may only be provided once");
            }
            config_path = std::filesystem::path(value);
        }
    }
    return config_path;
}

[[nodiscard]] CliOptions parse_arguments(
    const int argument_count,
    char** argument_values
) {
    CliOptions options;
    const std::optional<std::filesystem::path> config_path =
        find_config_path(argument_count, argument_values);
    if (config_path.has_value()) {
        options.config_path = *config_path;
        load_config(options, *config_path);
    }

    for (int argument_index = 1;
         argument_index < argument_count;
         ++argument_index) {
        const std::string_view argument = argument_values[argument_index];
        if (argument == "--help" || argument == "-h") {
            options.show_help = true;
        } else if (argument == "experiment") {
            if (!options.command.empty()) {
                throw std::invalid_argument("multiple commands were provided");
            }
            options.command = "experiment";
        } else if (argument == "train") {
            if (!options.command.empty()) {
                throw std::invalid_argument("multiple commands were provided");
            }
            options.command = "train";
        } else if (argument == "evaluate") {
            if (!options.command.empty()) {
                throw std::invalid_argument("multiple commands were provided");
            }
            options.command = "evaluate";
        } else if (argument == "trace") {
            if (!options.command.empty()) {
                throw std::invalid_argument("multiple commands were provided");
            }
            options.command = "trace";
        } else if (argument == "benchmark") {
            if (!options.command.empty()) {
                throw std::invalid_argument("multiple commands were provided");
            }
            options.command = "benchmark";
        } else if (argument == "inspect") {
            if (!options.command.empty()) {
                throw std::invalid_argument("multiple commands were provided");
            }
            options.command = "inspect";
        } else if (argument == "verify-checkpoint") {
            if (!options.command.empty()) {
                throw std::invalid_argument("multiple commands were provided");
            }
            options.command = "verify-checkpoint";
        } else if (argument == "inspect-agent") {
            if (!options.command.empty()) {
                throw std::invalid_argument("multiple commands were provided");
            }
            options.command = "inspect-agent";
        } else if (argument == "generate") {
            if (!options.command.empty()) {
                throw std::invalid_argument("multiple commands were provided");
            }
            options.command = "generate";
        } else if (argument == "--name") {
            options.experiment_name = require_value(
                argument_count,
                argument_values,
                argument_index,
                argument
            );
        } else if (argument == "--task") {
            options.task_name = require_value(
                argument_count,
                argument_values,
                argument_index,
                argument
            );
        } else if (argument == "--suite") {
            options.suite_name = require_value(
                argument_count,
                argument_values,
                argument_index,
                argument
            );
        } else if (argument == "--seed") {
            options.seed = parse_integer<std::uint64_t>(
                require_value(
                    argument_count,
                    argument_values,
                    argument_index,
                    argument
                ),
                argument
            );
        } else if (argument == "--dimension") {
            options.dimension = parse_integer<std::size_t>(
                require_value(
                    argument_count,
                    argument_values,
                    argument_index,
                    argument
                ),
                argument
            );
        } else if (argument == "--samples") {
            options.samples = parse_integer<std::size_t>(
                require_value(
                    argument_count,
                    argument_values,
                    argument_index,
                    argument
                ),
                argument
            );
        } else if (argument == "--training-examples") {
            options.training_examples = parse_integer<std::size_t>(
                require_value(
                    argument_count,
                    argument_values,
                    argument_index,
                    argument
                ),
                argument
            );
        } else if (argument == "--evaluation-examples") {
            options.evaluation_examples = parse_integer<std::size_t>(
                require_value(
                    argument_count,
                    argument_values,
                    argument_index,
                    argument
                ),
                argument
            );
        } else if (argument == "--training-examples-per-task") {
            options.training_examples_per_task =
                parse_integer<std::size_t>(
                    require_value(
                        argument_count,
                        argument_values,
                        argument_index,
                        argument
                    ),
                    argument
                );
        } else if (argument == "--evaluation-examples-per-task") {
            options.evaluation_examples_per_task =
                parse_integer<std::size_t>(
                    require_value(
                        argument_count,
                        argument_values,
                        argument_index,
                        argument
                    ),
                    argument
                );
        } else if (argument == "--association-count") {
            options.association_count = parse_integer<std::size_t>(
                require_value(
                    argument_count,
                    argument_values,
                    argument_index,
                    argument
                ),
                argument
            );
        } else if (argument == "--symbol-count") {
            options.symbol_count = parse_integer<std::size_t>(
                require_value(
                    argument_count,
                    argument_values,
                    argument_index,
                    argument
                ),
                argument
            );
        } else if (argument == "--context-noise-radians") {
            options.context_noise_radians = parse_floating(
                require_value(
                    argument_count,
                    argument_values,
                    argument_index,
                    argument
                ),
                argument
            );
        } else if (argument == "--noise-radians") {
            options.noise_radians = parse_floating(
                require_value(
                    argument_count,
                    argument_values,
                    argument_index,
                    argument
                ),
                argument
            );
        } else if (argument == "--corruption-radians") {
            options.corruption_radians = parse_floating(
                require_value(
                    argument_count,
                    argument_values,
                    argument_index,
                    argument
                ),
                argument
            );
        } else if (argument == "--dominant-probability") {
            options.dominant_probability = parse_floating(
                require_value(
                    argument_count,
                    argument_values,
                    argument_index,
                    argument
                ),
                argument
            );
        } else if (argument == "--threads") {
            options.threads = parse_integer<std::size_t>(
                require_value(
                    argument_count,
                    argument_values,
                    argument_index,
                    argument
                ),
                argument
            );
        } else if (argument == "--output") {
            options.output_path = require_value(
                argument_count,
                argument_values,
                argument_index,
                argument
            );
        } else if (argument == "--checkpoint") {
            options.checkpoint_path = require_value(
                argument_count, argument_values, argument_index, argument
            );
        } else if (argument == "--manifest") {
            options.manifest_path = require_value(
                argument_count, argument_values, argument_index, argument
            );
        } else if (argument == "--generate-dir") {
            options.generation_path = require_value(
                argument_count, argument_values, argument_index, argument
            );
        } else if (argument == "--knowledge-records") {
            options.frontier_knowledge_records = parse_integer<std::size_t>(
                require_value(argument_count, argument_values, argument_index, argument), argument
            );
        } else if (argument == "--knowledge-queries") {
            options.frontier_knowledge_queries = parse_integer<std::size_t>(
                require_value(argument_count, argument_values, argument_index, argument), argument
            );
        } else if (argument == "--backend") {
            options.frontier_backend = require_value(
                argument_count, argument_values, argument_index, argument
            );
        } else if (argument == "--agent-gate") {
            options.frontier_run_agent_gate = true;
        } else if (argument == "--agent-evaluation-episodes") {
            options.rlf6_evaluation_episodes = parse_integer<std::size_t>(
                require_value(argument_count, argument_values, argument_index, argument), argument
            );
        } else if (argument == "--action-budget") {
            options.rlf6_action_budget = parse_integer<std::size_t>(
                require_value(argument_count, argument_values, argument_index, argument), argument);
        } else if (argument == "--planning-node-budget") {
            options.planner_node_budget = parse_integer<std::size_t>(
                require_value(argument_count, argument_values, argument_index, argument), argument);
        } else if (argument == "--tool-budget") {
            options.rlf6_tool_budget = parse_integer<std::size_t>(
                require_value(argument_count, argument_values, argument_index, argument), argument);
        } else if (argument == "--risk-budget") {
            options.rlf6_risk_budget = parse_floating(
                require_value(argument_count, argument_values, argument_index, argument), argument);
        } else if (argument == "--mode-count") {
            options.mode_count = parse_integer<std::size_t>(
                require_value(
                    argument_count,
                    argument_values,
                    argument_index,
                    argument
                ),
                argument
            );
        } else if (argument == "--memory-records") {
            options.memory_records = parse_integer<std::size_t>(
                require_value(
                    argument_count,
                    argument_values,
                    argument_index,
                    argument
                ),
                argument
            );
        } else if (argument == "--evaluation-queries") {
            options.evaluation_queries = parse_integer<std::size_t>(
                require_value(
                    argument_count,
                    argument_values,
                    argument_index,
                    argument
                ),
                argument
            );
        } else if (argument == "--candidate-count") {
            options.candidate_count = parse_integer<std::size_t>(
                require_value(
                    argument_count,
                    argument_values,
                    argument_index,
                    argument
                ),
                argument
            );
        } else if (argument == "--active-count") {
            options.active_count = parse_integer<std::size_t>(
                require_value(
                    argument_count,
                    argument_values,
                    argument_index,
                    argument
                ),
                argument
            );
        } else if (argument == "--context-dimensions") {
            options.context_dimensions = parse_integer<std::size_t>(
                require_value(
                    argument_count,
                    argument_values,
                    argument_index,
                    argument
                ),
                argument
            );
        } else if (argument == "--quantization-samples") {
            options.quantization_samples =
                parse_integer<std::size_t>(
                    require_value(
                        argument_count,
                        argument_values,
                        argument_index,
                        argument
                    ),
                    argument
                );
        } else if (argument == "--development-examples") {
            options.development_examples = parse_integer<std::size_t>(
                require_value(argument_count, argument_values, argument_index, argument),
                argument
            );
        } else if (argument == "--training-min-route-length") {
            options.training_min_route_length = parse_integer<std::size_t>(
                require_value(argument_count, argument_values, argument_index, argument),
                argument
            );
        } else if (argument == "--training-max-route-length") {
            options.training_max_route_length = parse_integer<std::size_t>(
                require_value(argument_count, argument_values, argument_index, argument),
                argument
            );
        } else if (argument == "--evaluation-min-route-length") {
            options.evaluation_min_route_length = parse_integer<std::size_t>(
                require_value(argument_count, argument_values, argument_index, argument),
                argument
            );
        } else if (argument == "--evaluation-max-route-length") {
            options.evaluation_max_route_length = parse_integer<std::size_t>(
                require_value(argument_count, argument_values, argument_index, argument),
                argument
            );
        } else if (argument == "--maximum-cycles") {
            options.maximum_cycles = parse_integer<std::size_t>(
                require_value(argument_count, argument_values, argument_index, argument),
                argument
            );
        } else if (argument == "--state-noise-radians") {
            options.state_noise_radians = parse_floating(
                require_value(argument_count, argument_values, argument_index, argument),
                argument
            );
        } else if (argument == "--mode-counts") {
            options.mode_counts = parse_size_list(
                require_value(
                    argument_count,
                    argument_values,
                    argument_index,
                    argument
                ),
                argument
            );
        } else if (argument == "--sample") {
            options.sample_id = parse_integer<std::size_t>(
                require_value(
                    argument_count,
                    argument_values,
                    argument_index,
                    argument
                ),
                argument
            );
        } else if (argument == "--log-level") {
            options.log_level = require_value(
                argument_count,
                argument_values,
                argument_index,
                argument
            );
            validate_log_level(options.log_level);
        } else if (argument == "--deterministic") {
            options.deterministic = true;
        } else if (argument == "--max-memory") {
            options.max_memory_bytes = parse_memory_size(
                require_value(
                    argument_count,
                    argument_values,
                    argument_index,
                    argument
                ),
                argument
            );
        } else if (argument == "--config") {
            static_cast<void>(require_value(
                argument_count,
                argument_values,
                argument_index,
                argument
            ));
        } else {
            throw std::invalid_argument(
                "unknown command or option: " + std::string(argument)
            );
        }
    }

    if (options.threads == 0U) {
        throw std::invalid_argument("--threads must be positive");
    }
    if (!options.deterministic) {
        throw std::invalid_argument(
            "RLF-0 currently requires deterministic execution"
        );
    }
    return options;
}

void print_help(std::ostream& output) {
    output
        << "RLF-Frontier single-GPU non-neural trainer (RLF-0 through Frontier)\n\n"
        << "Usage:\n"
        << "  rlf train --config PATH\n"
        << "  rlf evaluate --checkpoint PATH --task NAME [options]\n"
        << "  rlf benchmark --suite core [options]\n"
        << "  rlf experiment --name associative_recall [options]\n"
        << "  rlf experiment --name capacity_scaling [options]\n"
        << "  rlf experiment --name compositional_generalization [options]\n"
        << "  rlf experiment --name operator_composition [options]\n"
        << "  rlf experiment --name latent_routing [options]\n"
        << "  rlf experiment --name predictive_reasoning [options]\n"
        << "  rlf experiment --name sparse_world_model [options]\n"
        << "  rlf experiment --name self_supervised_temporal [options]\n"
        << "  rlf experiment --name compositional_language [options]\n"
        << "  rlf experiment --name successor_prediction [options]\n"
        << "  rlf experiment --name causal_subgoals [options]\n"
        << "  rlf experiment --name intervention_credit [options]\n"
        << "  rlf experiment --name skill_consolidation [options]\n"
        << "  rlf experiment --name withheld_routes [options]\n"
        << "  rlf experiment --name adaptive_halting [options]\n"
        << "  rlf experiment --name delayed_credit [options]\n"
        << "  rlf experiment --name route_recovery [options]\n"
        << "  rlf experiment --name route_continual_learning [options]\n"
        << "  rlf experiment --name macro_consolidation [options]\n"
        << "  rlf experiment --name optimization_benchmark [options]\n"
        << "  rlf experiment --name continual_learning [options]\n"
        << "  rlf experiment --name phase_vector_smoke [options]\n"
        << "  rlf experiment --name persistence_roundtrip [options]\n"
        << "  rlf experiment --name sequence_completion [options]\n"
        << "  rlf experiment --name structural_adaptation [options]\n"
        << "  rlf experiment --name transformation_learning [options]\n"
        << "  rlf experiment --name long_horizon_planning [options]\n"
        << "  rlf experiment --name tool_use [options]\n"
        << "  rlf experiment --name tool_reliability [options]\n"
        << "  rlf experiment --name self_correction [options]\n"
        << "  rlf experiment --name uncertainty_calibration [options]\n"
        << "  rlf experiment --name information_seeking [options]\n"
        << "  rlf experiment --name changing_world [options]\n"
        << "  rlf experiment --name failure_recovery [options]\n"
        << "  rlf experiment --name continual_agent_learning [options]\n"
        << "  rlf experiment --name skill_consolidation [options]\n"
        << "  rlf experiment --name memory_scaling [options]\n"
        << "  rlf experiment --name resource_bounded_agent [options]\n"
        << "  rlf experiment --name adversarial_robustness [options]\n"
        << "  rlf experiment --name hundred_step_agent [options]\n"
        << "  rlf experiment --name agent_scaling [options]\n"
        << "  rlf train --task agent --config PATH [options]\n"
        << "  rlf evaluate --task agent --checkpoint PATH [options]\n"
        << "  rlf trace --task agent --checkpoint PATH [options]\n"
        << "  rlf inspect --checkpoint PATH\n"
        << "  rlf trace --checkpoint PATH --task NAME --sample ID\n"
        << "  rlf verify-checkpoint --checkpoint PATH\n"
        << "  rlf inspect-agent --checkpoint PATH\n"
        << "  rlf train --task rlf7|frontier --checkpoint PATH [--manifest PATH]\n"
        << "  rlf evaluate --task rlf7|frontier --checkpoint PATH --manifest PATH\n"
        << "  rlf generate --task frontier --checkpoint PATH --generate-dir PATH\n\n"
        << "Options:\n"
        << "  --config PATH       Read key=value configuration\n"
        << "  --seed UINT64       Deterministic master seed\n"
        << "  --task NAME         Controlled checkpoint task\n"
        << "  --suite NAME        Benchmark suite name\n"
        << "  --dimension COUNT   Phase-vector dimension\n"
        << "  --samples COUNT     Smoke-experiment sample count\n"
        << "  --training-examples COUNT   Training examples\n"
        << "  --evaluation-examples COUNT Evaluation examples\n"
        << "  --training-examples-per-task COUNT Continual examples/task\n"
        << "  --evaluation-examples-per-task COUNT Continual eval/task\n"
        << "  --association-count COUNT   Recall associations\n"
        << "  --symbol-count COUNT        Sequence symbols\n"
        << "  --context-noise-radians VALUE Context perturbation\n"
        << "  --noise-radians VALUE Recall-key perturbation\n"
        << "  --corruption-radians VALUE Sequence corruption\n"
        << "  --dominant-probability VALUE Probabilistic transition mass\n"
        << "  --threads COUNT     Deterministic worker count\n"
        << "  --backend NAME     Frontier backend: scalar_cpu|optimized_cpu|cuda\n"
        << "  --agent-evaluation-episodes COUNT Frontier inherited agency gate episodes\n"
        << "  --action-budget COUNT RLF-6 action budget\n"
        << "  --planning-node-budget COUNT RLF-6 planning-node budget\n"
        << "  --tool-budget COUNT RLF-6 tool-call budget\n"
        << "  --risk-budget VALUE RLF-6 risk budget\n"
        << "  --output PATH       Write JSON result to PATH\n"
        << "  --checkpoint PATH   Read or write an RLF checkpoint\n"
        << "  --manifest PATH     Native multimodal TSV dataset manifest\n"
        << "  --generate-dir PATH Prototype image/video/audio output directory\n"
        << "  --knowledge-records COUNT Synthetic knowledge scale for RLF-7/Frontier\n"
        << "  --knowledge-queries COUNT Synthetic retrieval queries\n"
        << "  --agent-gate       Run inherited long-horizon agency promotion gate\n"
        << "  --mode-count COUNT  Persistence experiment modes\n"
        << "  --memory-records COUNT Persistence experiment records\n"
        << "  --evaluation-queries COUNT Capacity retrieval queries\n"
        << "  --candidate-count COUNT Capacity top-K candidates\n"
        << "  --active-count COUNT Capacity active proposals\n"
        << "  --context-dimensions COUNT Operator routing prefix\n"
        << "  --quantization-samples COUNT Quantization benchmark samples\n"
        << "  --development-examples COUNT RLF-1 development episodes\n"
        << "  --training-min-route-length COUNT RLF-1 minimum train route\n"
        << "  --training-max-route-length COUNT RLF-1 maximum train route\n"
        << "  --evaluation-min-route-length COUNT RLF-1 minimum test route\n"
        << "  --evaluation-max-route-length COUNT RLF-1 maximum test route\n"
        << "  --maximum-cycles COUNT RLF-1 reasoning budget\n"
        << "  --state-noise-radians VALUE RLF-1 state noise\n"
        << "  RLF-3/RLF-4/RLF-5/RLF-6 settings are configurable through key=value config files.\n"
        << "  --mode-counts CSV   Capacity scales, e.g. 1024,4096\n"
        << "  --sample ID         Deterministic trace sample ID\n"
        << "  --log-level LEVEL   trace, debug, info, warn, or error\n"
        << "  --deterministic     Require deterministic execution\n"
        << "  --max-memory SIZE   Memory cap, for example 64MiB\n"
        << "  --help              Show this help\n";
}

[[nodiscard]] std::uint64_t estimated_smoke_memory(
    const std::size_t dimension
) {
    constexpr std::uint64_t resident_phase_vectors = 6ULL;
    constexpr std::uint64_t bytes_per_angle = sizeof(float);
    const auto dimension_u64 = static_cast<std::uint64_t>(dimension);
    return checked_multiply(
        checked_multiply(
            dimension_u64,
            bytes_per_angle,
            "--dimension"
        ),
        resident_phase_vectors,
        "--dimension"
    );
}

[[nodiscard]] std::uint64_t estimated_phase_storage(
    const std::size_t vector_count,
    const std::size_t dimension
) {
    return checked_multiply(
        checked_multiply(
            static_cast<std::uint64_t>(vector_count),
            static_cast<std::uint64_t>(dimension),
            "--max-memory"
        ),
        sizeof(float),
        "--max-memory"
    );
}

[[nodiscard]] std::uint64_t estimated_optimization_memory(
    const CliOptions& options
) {
    const std::uint64_t mode_vectors = checked_multiply(
        static_cast<std::uint64_t>(options.mode_count),
        3ULL,
        "--mode-count"
    );
    const std::uint64_t query_vectors =
        static_cast<std::uint64_t>(options.evaluation_queries);
    const std::uint64_t quantization_vectors = checked_multiply(
        static_cast<std::uint64_t>(options.quantization_samples),
        10ULL,
        "--quantization-samples"
    );
    const std::uint64_t vector_count = checked_add(
        checked_add(
            mode_vectors,
            query_vectors,
            "--max-memory"
        ),
        quantization_vectors,
        "--max-memory"
    );
    return checked_multiply(
        checked_multiply(
            vector_count,
            static_cast<std::uint64_t>(options.dimension),
            "--max-memory"
        ),
        sizeof(float),
        "--max-memory"
    );
}

[[nodiscard]] std::uint64_t estimated_rlf4_memory(
    const CliOptions& options
) {
    const std::uint64_t dimension = static_cast<std::uint64_t>(
        options.config_path.empty() && options.dimension == 1'024U
            ? 48U
            : options.dimension
    );
    const std::uint64_t observation_count = checked_add(
        checked_add(
            static_cast<std::uint64_t>(options.temporal_training_tokens),
            static_cast<std::uint64_t>(options.temporal_evaluation_tokens),
            "RLF-4 observation count"
        ),
        checked_add(
            static_cast<std::uint64_t>(options.temporal_adaptation_tokens),
            static_cast<std::uint64_t>(
                std::max<std::size_t>(18U, options.symbol_count)
            ),
            "RLF-4 observation count"
        ),
        "RLF-4 observation count"
    );
    const std::uint64_t phase_bytes = checked_multiply(
        checked_multiply(
            observation_count, dimension, "RLF-4 phase storage"
        ),
        sizeof(float),
        "RLF-4 phase storage"
    );
    const std::uint64_t context_entries = checked_multiply(
        static_cast<std::uint64_t>(options.temporal_training_tokens),
        static_cast<std::uint64_t>(options.temporal_context_order),
        "RLF-4 context storage"
    );
    const std::uint64_t context_bytes = checked_multiply(
        context_entries, 64ULL, "RLF-4 context storage"
    );
    const std::uint64_t option_bytes = checked_multiply(
        static_cast<std::uint64_t>(options.temporal_maximum_options),
        checked_add(
            checked_multiply(
                static_cast<std::uint64_t>(options.temporal_context_order),
                sizeof(std::uint64_t),
                "RLF-4 option storage"
            ),
            96ULL,
            "RLF-4 option storage"
        ),
        "RLF-4 option storage"
    );
    return checked_add(
        checked_add(phase_bytes, context_bytes, "RLF-4 memory estimate"),
        option_bytes,
        "RLF-4 memory estimate"
    );
}

[[nodiscard]] std::uint64_t estimated_rlf5_memory(
    const CliOptions& options
) {
    const std::uint64_t dimension = static_cast<std::uint64_t>(
        options.config_path.empty() && options.dimension == 1'024U
            ? 64U
            : options.dimension
    );
    const std::uint64_t corpus_bytes = checked_multiply(
        static_cast<std::uint64_t>(options.language_raw_training_sentences),
        96ULL,
        "RLF-5 corpus estimate"
    );
    const std::uint64_t supervised_bytes = checked_multiply(
        static_cast<std::uint64_t>(options.language_supervised_training_examples),
        320ULL,
        "RLF-5 supervised estimate"
    );
    const std::uint64_t lexeme_bytes = checked_multiply(
        static_cast<std::uint64_t>(options.language_maximum_lexemes),
        checked_add(
            checked_multiply(dimension, sizeof(float), "RLF-5 lexeme phases"),
            128ULL,
            "RLF-5 lexeme estimate"
        ),
        "RLF-5 lexeme estimate"
    );
    const std::uint64_t context_bytes = checked_multiply(
        corpus_bytes,
        checked_add(
            checked_multiply(
                static_cast<std::uint64_t>(options.language_context_order),
                4ULL,
                "RLF-5 context estimate"
            ),
            16ULL,
            "RLF-5 context estimate"
        ),
        "RLF-5 context estimate"
    );
    const std::uint64_t construction_bytes = checked_multiply(
        static_cast<std::uint64_t>(options.language_maximum_constructions),
        checked_add(
            checked_multiply(
                static_cast<std::uint64_t>(options.language_maximum_generation_tokens),
                24ULL,
                "RLF-5 construction estimate"
            ),
            128ULL,
            "RLF-5 construction estimate"
        ),
        "RLF-5 construction estimate"
    );
    return checked_add(
        checked_add(corpus_bytes, supervised_bytes, "RLF-5 memory estimate"),
        checked_add(
            checked_add(lexeme_bytes, context_bytes, "RLF-5 memory estimate"),
            construction_bytes,
            "RLF-5 memory estimate"
        ),
        "RLF-5 memory estimate"
    );
}

[[nodiscard]] std::uint64_t estimated_experiment_memory(
    const CliOptions& options
) {
    if (options.experiment_name == "rlf7_full" || options.experiment_name == "frontier_full") {
        return checked_multiply(
            static_cast<std::uint64_t>(options.frontier_knowledge_records),
            512ULL,
            "RLF-7/Frontier memory estimate"
        );
    }

    if (rlf::experiments::is_rlf6_experiment_name(options.experiment_name)) {
        return checked_multiply(
            static_cast<std::uint64_t>(options.rlf6_memory_limit_records),
            512ULL,
            "RLF-6 memory estimate"
        );
    }
    if (options.experiment_name == "compositional_language" ||
        options.experiment_name == "language_understanding" ||
        options.experiment_name == "language_generation" ||
        options.experiment_name == "semantic_qa") {
        return estimated_rlf5_memory(options);
    }
    if (options.experiment_name == "self_supervised_temporal" ||
        options.experiment_name == "temporal_prediction" ||
        options.experiment_name == "temporal_abstraction" ||
        options.experiment_name == "regime_adaptation") {
        return estimated_rlf4_memory(options);
    }
    if (options.experiment_name == "sparse_world_model" ||
        options.experiment_name == "learned_transition_model" ||
        options.experiment_name == "sparse_subgoal_index" ||
        options.experiment_name == "partial_observation" ||
        options.experiment_name == "stochastic_planning") {
        const std::uint64_t physical_states = checked_add(
            checked_multiply(
                static_cast<std::uint64_t>(options.world_layers),
                static_cast<std::uint64_t>(options.world_lanes),
                "world dimensions"
            ),
            1ULL,
            "world dimensions"
        );
        constexpr std::uint64_t hidden_modes = 2ULL;
        constexpr std::uint64_t training_contexts = 3ULL;
        constexpr std::uint64_t actions = 6ULL;
        constexpr std::uint64_t vectors_per_transition = 4ULL;
        const std::uint64_t transition_vectors = checked_multiply(
            checked_multiply(
                checked_multiply(
                    checked_multiply(
                        physical_states,
                        hidden_modes,
                        "RLF-3 memory estimate"
                    ),
                    training_contexts,
                    "RLF-3 memory estimate"
                ),
                actions,
                "RLF-3 memory estimate"
            ),
            checked_multiply(
                static_cast<std::uint64_t>(
                    options.transition_samples_per_case
                ),
                vectors_per_transition,
                "RLF-3 memory estimate"
            ),
            "RLF-3 memory estimate"
        );
        const std::uint64_t route_vectors = checked_multiply(
            checked_multiply(
                static_cast<std::uint64_t>(options.training_routes),
                static_cast<std::uint64_t>(options.world_layers + 2U),
                "RLF-3 memory estimate"
            ),
            2ULL,
            "RLF-3 memory estimate"
        );
        const std::uint64_t evaluation_vectors = checked_multiply(
            checked_multiply(
                static_cast<std::uint64_t>(options.evaluation_examples),
                static_cast<std::uint64_t>(options.stochastic_rollouts),
                "RLF-3 memory estimate"
            ),
            4ULL,
            "RLF-3 memory estimate"
        );
        const std::uint64_t vector_count = checked_add(
            checked_add(
                transition_vectors,
                route_vectors,
                "RLF-3 memory estimate"
            ),
            checked_add(
                evaluation_vectors,
                checked_multiply(
                    physical_states,
                    8ULL,
                    "RLF-3 memory estimate"
                ),
                "RLF-3 memory estimate"
            ),
            "RLF-3 memory estimate"
        );
        return checked_multiply(
            checked_multiply(
                vector_count,
                static_cast<std::uint64_t>(
                    options.config_path.empty() && options.dimension == 1'024U
                        ? 32U
                        : options.dimension
                ),
                "RLF-3 memory estimate"
            ),
            sizeof(float),
            "RLF-3 memory estimate"
        );
    }
    if (options.experiment_name == "latent_routing" ||
        options.experiment_name == "withheld_routes" ||
        options.experiment_name == "adaptive_halting" ||
        options.experiment_name == "delayed_credit" ||
        options.experiment_name == "route_recovery" ||
        options.experiment_name == "route_continual_learning" ||
        options.experiment_name == "macro_consolidation" ||
        options.experiment_name == "predictive_reasoning" ||
        options.experiment_name == "successor_prediction" ||
        options.experiment_name == "causal_subgoals" ||
        options.experiment_name == "intervention_credit" ||
        options.experiment_name == "skill_consolidation") {
        const std::size_t routed_vectors =
            options.training_examples * 16U +
            options.development_examples * 8U +
            options.evaluation_examples * 8U + 4'096U;
        return estimated_phase_storage(
            routed_vectors,
            options.config_path.empty() && options.dimension == 1'024U
                ? 24U
                : options.dimension
        );
    }
    if (options.experiment_name == "phase_vector_smoke") {
        return estimated_smoke_memory(options.dimension);
    }
    if (options.experiment_name == "associative_recall") {
        return estimated_phase_storage(
            options.association_count * 4U,
            options.dimension
        );
    }
    if (options.experiment_name == "capacity_scaling") {
        const std::size_t largest_mode_count =
            *std::max_element(
                options.mode_counts.begin(),
                options.mode_counts.end()
            );
        return estimated_phase_storage(
            largest_mode_count * 2U,
            options.dimension
        );
    }
    if (options.experiment_name == "compositional_generalization") {
        return estimated_phase_storage(
            options.training_examples * 8U,
            options.dimension
        );
    }
    if (options.experiment_name == "operator_composition") {
        constexpr std::size_t task_count = 6U;
        constexpr std::size_t hypotheses_per_task = 6U;
        return estimated_phase_storage(
            task_count * hypotheses_per_task *
                options.training_examples * 2U,
            options.dimension
        );
    }
    if (options.experiment_name == "continual_learning") {
        return estimated_phase_storage(
            options.training_examples_per_task * 8U,
            options.dimension
        );
    }
    if (options.experiment_name == "sequence_completion") {
        return estimated_phase_storage(
            options.training_examples * 4U,
            options.dimension
        );
    }
    if (options.experiment_name == "persistence_roundtrip") {
        return estimated_phase_storage(
            (options.mode_count * 2U) +
                (options.memory_records * 2U),
            options.dimension
        );
    }
    if (options.experiment_name == "optimization_benchmark") {
        return estimated_optimization_memory(options);
    }
    const bool rlf1_experiment =
        options.experiment_name == "latent_routing" ||
        options.experiment_name == "withheld_routes" ||
        options.experiment_name == "adaptive_halting" ||
        options.experiment_name == "delayed_credit" ||
        options.experiment_name == "route_recovery" ||
        options.experiment_name == "route_continual_learning" ||
        options.experiment_name == "macro_consolidation";
    if (rlf1_experiment) {
        return estimated_phase_storage(
            (options.training_examples + options.evaluation_examples) * 6U,
            options.dimension
        );
    }
    if (options.experiment_name == "transformation_learning" ||
        options.experiment_name == "structural_adaptation") {
        return estimated_phase_storage(
            options.training_examples * 4U,
            options.dimension
        );
    }
    return estimated_smoke_memory(options.dimension);
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::PhaseVectorSmokeResult& result
) {
    if (path.empty()) {
        return;
    }
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::filesystem::path temporary_path = path;
    temporary_path += ".tmp";
    {
        std::ofstream output(
            temporary_path,
            std::ios::binary | std::ios::trunc
        );
        if (!output) {
            throw std::runtime_error(
                "unable to open result file: " + temporary_path.string()
            );
        }
        rlf::experiments::write_phase_vector_smoke_json(output, result);
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "failed to flush result file: " + temporary_path.string()
            );
        }
    }

    std::error_code rename_error;
    std::filesystem::rename(temporary_path, path, rename_error);
    if (rename_error) {
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        rename_error.clear();
        std::filesystem::rename(temporary_path, path, rename_error);
    }
    if (rename_error) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary_path, cleanup_error);
        throw std::runtime_error(
            "failed to publish result file: " + rename_error.message()
        );
    }
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::OperatorCompositionResult& result
) {
    if (path.empty()) {
        return;
    }
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::filesystem::path temporary_path = path;
    temporary_path += ".tmp";
    {
        std::ofstream output(
            temporary_path,
            std::ios::binary | std::ios::trunc
        );
        if (!output) {
            throw std::runtime_error(
                "unable to open result file: " + temporary_path.string()
            );
        }
        rlf::experiments::write_operator_composition_json(output, result);
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "failed to flush result file: " + temporary_path.string()
            );
        }
    }
    std::error_code rename_error;
    std::filesystem::rename(temporary_path, path, rename_error);
    if (rename_error) {
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        rename_error.clear();
        std::filesystem::rename(temporary_path, path, rename_error);
    }
    if (rename_error) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary_path, cleanup_error);
        throw std::runtime_error(
            "failed to publish result file: " + rename_error.message()
        );
    }
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::OptimizationBenchmarkResult& result
) {
    if (path.empty()) {
        return;
    }
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::filesystem::path temporary_path = path;
    temporary_path += ".tmp";
    {
        std::ofstream output(
            temporary_path,
            std::ios::binary | std::ios::trunc
        );
        if (!output) {
            throw std::runtime_error(
                "unable to open result file: " + temporary_path.string()
            );
        }
        rlf::experiments::write_optimization_benchmark_json(
            output,
            result
        );
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "failed to flush result file: " + temporary_path.string()
            );
        }
    }
    std::error_code rename_error;
    std::filesystem::rename(temporary_path, path, rename_error);
    if (rename_error) {
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        rename_error.clear();
        std::filesystem::rename(temporary_path, path, rename_error);
    }
    if (rename_error) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary_path, cleanup_error);
        throw std::runtime_error(
            "failed to publish result file: " + rename_error.message()
        );
    }
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::AssociativeRecallResult& result
) {
    if (path.empty()) {
        return;
    }
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::filesystem::path temporary_path = path;
    temporary_path += ".tmp";
    {
        std::ofstream output(
            temporary_path,
            std::ios::binary | std::ios::trunc
        );
        if (!output) {
            throw std::runtime_error(
                "unable to open result file: " + temporary_path.string()
            );
        }
        rlf::experiments::write_associative_recall_json(output, result);
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "failed to flush result file: " + temporary_path.string()
            );
        }
    }

    std::error_code rename_error;
    std::filesystem::rename(temporary_path, path, rename_error);
    if (rename_error) {
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        rename_error.clear();
        std::filesystem::rename(temporary_path, path, rename_error);
    }
    if (rename_error) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary_path, cleanup_error);
        throw std::runtime_error(
            "failed to publish result file: " + rename_error.message()
        );
    }
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::CapacityScalingResult& result
) {
    if (path.empty()) {
        return;
    }
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::filesystem::path temporary_path = path;
    temporary_path += ".tmp";
    {
        std::ofstream output(
            temporary_path,
            std::ios::binary | std::ios::trunc
        );
        if (!output) {
            throw std::runtime_error(
                "unable to open result file: " + temporary_path.string()
            );
        }
        rlf::experiments::write_capacity_scaling_json(output, result);
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "failed to flush result file: " + temporary_path.string()
            );
        }
    }

    std::error_code rename_error;
    std::filesystem::rename(temporary_path, path, rename_error);
    if (rename_error) {
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        rename_error.clear();
        std::filesystem::rename(temporary_path, path, rename_error);
    }
    if (rename_error) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary_path, cleanup_error);
        throw std::runtime_error(
            "failed to publish result file: " + rename_error.message()
        );
    }
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::CheckpointTrainingResult& result
) {
    if (path.empty()) {
        return;
    }
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::filesystem::path temporary_path = path;
    temporary_path += ".tmp";
    {
        std::ofstream output(
            temporary_path,
            std::ios::binary | std::ios::trunc
        );
        if (!output) {
            throw std::runtime_error(
                "unable to open result file: " + temporary_path.string()
            );
        }
        rlf::experiments::write_checkpoint_training_json(output, result);
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "failed to flush result file: " + temporary_path.string()
            );
        }
    }
    std::error_code rename_error;
    std::filesystem::rename(temporary_path, path, rename_error);
    if (rename_error) {
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        rename_error.clear();
        std::filesystem::rename(temporary_path, path, rename_error);
    }
    if (rename_error) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary_path, cleanup_error);
        throw std::runtime_error(
            "failed to publish result file: " + rename_error.message()
        );
    }
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::CheckpointEvaluationResult& result
) {
    if (path.empty()) {
        return;
    }
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::filesystem::path temporary_path = path;
    temporary_path += ".tmp";
    {
        std::ofstream output(
            temporary_path,
            std::ios::binary | std::ios::trunc
        );
        if (!output) {
            throw std::runtime_error(
                "unable to open result file: " + temporary_path.string()
            );
        }
        rlf::experiments::write_checkpoint_evaluation_json(
            output,
            result
        );
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "failed to flush result file: " + temporary_path.string()
            );
        }
    }
    std::error_code rename_error;
    std::filesystem::rename(temporary_path, path, rename_error);
    if (rename_error) {
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        rename_error.clear();
        std::filesystem::rename(temporary_path, path, rename_error);
    }
    if (rename_error) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary_path, cleanup_error);
        throw std::runtime_error(
            "failed to publish result file: " + rename_error.message()
        );
    }
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::CheckpointTraceResult& result
) {
    if (path.empty()) {
        return;
    }
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::filesystem::path temporary_path = path;
    temporary_path += ".tmp";
    {
        std::ofstream output(
            temporary_path,
            std::ios::binary | std::ios::trunc
        );
        if (!output) {
            throw std::runtime_error(
                "unable to open result file: " + temporary_path.string()
            );
        }
        rlf::experiments::write_checkpoint_trace_json(output, result);
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "failed to flush result file: " + temporary_path.string()
            );
        }
    }
    std::error_code rename_error;
    std::filesystem::rename(temporary_path, path, rename_error);
    if (rename_error) {
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        rename_error.clear();
        std::filesystem::rename(temporary_path, path, rename_error);
    }
    if (rename_error) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary_path, cleanup_error);
        throw std::runtime_error(
            "failed to publish result file: " + rename_error.message()
        );
    }
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::CompositionalGeneralizationResult& result
) {
    if (path.empty()) {
        return;
    }
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::filesystem::path temporary_path = path;
    temporary_path += ".tmp";
    {
        std::ofstream output(
            temporary_path,
            std::ios::binary | std::ios::trunc
        );
        if (!output) {
            throw std::runtime_error(
                "unable to open result file: " + temporary_path.string()
            );
        }
        rlf::experiments::write_compositional_generalization_json(
            output,
            result
        );
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "failed to flush result file: " + temporary_path.string()
            );
        }
    }

    std::error_code rename_error;
    std::filesystem::rename(temporary_path, path, rename_error);
    if (rename_error) {
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        rename_error.clear();
        std::filesystem::rename(temporary_path, path, rename_error);
    }
    if (rename_error) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary_path, cleanup_error);
        throw std::runtime_error(
            "failed to publish result file: " + rename_error.message()
        );
    }
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::ContinualLearningResult& result
) {
    if (path.empty()) {
        return;
    }
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::filesystem::path temporary_path = path;
    temporary_path += ".tmp";
    {
        std::ofstream output(
            temporary_path,
            std::ios::binary | std::ios::trunc
        );
        if (!output) {
            throw std::runtime_error(
                "unable to open result file: " + temporary_path.string()
            );
        }
        rlf::experiments::write_continual_learning_json(output, result);
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "failed to flush result file: " + temporary_path.string()
            );
        }
    }

    std::error_code rename_error;
    std::filesystem::rename(temporary_path, path, rename_error);
    if (rename_error) {
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        rename_error.clear();
        std::filesystem::rename(temporary_path, path, rename_error);
    }
    if (rename_error) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary_path, cleanup_error);
        throw std::runtime_error(
            "failed to publish result file: " + rename_error.message()
        );
    }
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::SequenceCompletionResult& result
) {
    if (path.empty()) {
        return;
    }
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::filesystem::path temporary_path = path;
    temporary_path += ".tmp";
    {
        std::ofstream output(
            temporary_path,
            std::ios::binary | std::ios::trunc
        );
        if (!output) {
            throw std::runtime_error(
                "unable to open result file: " + temporary_path.string()
            );
        }
        rlf::experiments::write_sequence_completion_json(output, result);
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "failed to flush result file: " + temporary_path.string()
            );
        }
    }

    std::error_code rename_error;
    std::filesystem::rename(temporary_path, path, rename_error);
    if (rename_error) {
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        rename_error.clear();
        std::filesystem::rename(temporary_path, path, rename_error);
    }
    if (rename_error) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary_path, cleanup_error);
        throw std::runtime_error(
            "failed to publish result file: " + rename_error.message()
        );
    }
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::StructuralAdaptationResult& result
) {
    if (path.empty()) {
        return;
    }
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::filesystem::path temporary_path = path;
    temporary_path += ".tmp";
    {
        std::ofstream output(
            temporary_path,
            std::ios::binary | std::ios::trunc
        );
        if (!output) {
            throw std::runtime_error(
                "unable to open result file: " + temporary_path.string()
            );
        }
        rlf::experiments::write_structural_adaptation_json(output, result);
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "failed to flush result file: " + temporary_path.string()
            );
        }
    }

    std::error_code rename_error;
    std::filesystem::rename(temporary_path, path, rename_error);
    if (rename_error) {
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        rename_error.clear();
        std::filesystem::rename(temporary_path, path, rename_error);
    }
    if (rename_error) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary_path, cleanup_error);
        throw std::runtime_error(
            "failed to publish result file: " + rename_error.message()
        );
    }
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::TransformationLearningResult& result
) {
    if (path.empty()) {
        return;
    }
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::filesystem::path temporary_path = path;
    temporary_path += ".tmp";
    {
        std::ofstream output(
            temporary_path,
            std::ios::binary | std::ios::trunc
        );
        if (!output) {
            throw std::runtime_error(
                "unable to open result file: " + temporary_path.string()
            );
        }
        rlf::experiments::write_transformation_learning_json(output, result);
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "failed to flush result file: " + temporary_path.string()
            );
        }
    }

    std::error_code rename_error;
    std::filesystem::rename(temporary_path, path, rename_error);
    if (rename_error) {
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        rename_error.clear();
        std::filesystem::rename(temporary_path, path, rename_error);
    }
    if (rename_error) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary_path, cleanup_error);
        throw std::runtime_error(
            "failed to publish result file: " + rename_error.message()
        );
    }
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::PersistenceRoundtripResult& result
) {
    if (path.empty()) {
        return;
    }
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::filesystem::path temporary_path = path;
    temporary_path += ".tmp";
    {
        std::ofstream output(
            temporary_path,
            std::ios::binary | std::ios::trunc
        );
        if (!output) {
            throw std::runtime_error(
                "unable to open result file: " + temporary_path.string()
            );
        }
        rlf::experiments::write_persistence_roundtrip_json(output, result);
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "failed to flush result file: " + temporary_path.string()
            );
        }
    }

    std::error_code rename_error;
    std::filesystem::rename(temporary_path, path, rename_error);
    if (rename_error) {
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        rename_error.clear();
        std::filesystem::rename(temporary_path, path, rename_error);
    }
    if (rename_error) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary_path, cleanup_error);
        throw std::runtime_error(
            "failed to publish result file: " + rename_error.message()
        );
    }
}


template <typename Result, typename Writer>
void write_rlf1_result_transactionally(
    const std::filesystem::path& path,
    const Result& result,
    Writer writer
) {
    if (path.empty()) {
        return;
    }
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::filesystem::path temporary = path;
    temporary += ".tmp";
    {
        std::ofstream output(
            temporary,
            std::ios::binary | std::ios::trunc
        );
        if (!output) {
            throw std::runtime_error(
                "unable to open RLF-1 result file: " + temporary.string()
            );
        }
        writer(output, result);
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "failed to flush RLF-1 result file: " + temporary.string()
            );
        }
    }
    std::error_code rename_error;
    std::filesystem::rename(temporary, path, rename_error);
    if (rename_error) {
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        rename_error.clear();
        std::filesystem::rename(temporary, path, rename_error);
    }
    if (rename_error) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary, cleanup_error);
        throw std::runtime_error(
            "failed to publish RLF-1 result file: " + rename_error.message()
        );
    }
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::Rlf1Result& result
) {
    write_rlf1_result_transactionally(
        path,
        result,
        rlf::experiments::write_rlf1_latent_routing_json
    );
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::Rlf1TrainingWorkflowResult& result
) {
    write_rlf1_result_transactionally(
        path,
        result,
        rlf::experiments::write_rlf1_training_json
    );
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::Rlf1EvaluationWorkflowResult& result
) {
    write_rlf1_result_transactionally(
        path,
        result,
        rlf::experiments::write_rlf1_evaluation_json
    );
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::Rlf1TraceWorkflowResult& result
) {
    write_rlf1_result_transactionally(
        path,
        result,
        rlf::experiments::write_rlf1_trace_json
    );
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::Rlf2Result& result
) {
    write_rlf1_result_transactionally(
        path,
        result,
        rlf::experiments::write_rlf2_predictive_reasoning_json
    );
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::Rlf2TrainingWorkflowResult& result
) {
    write_rlf1_result_transactionally(
        path,
        result,
        rlf::experiments::write_rlf2_training_json
    );
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::Rlf2EvaluationWorkflowResult& result
) {
    write_rlf1_result_transactionally(
        path,
        result,
        rlf::experiments::write_rlf2_evaluation_json
    );
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::Rlf2TraceWorkflowResult& result
) {
    write_rlf1_result_transactionally(
        path,
        result,
        rlf::experiments::write_rlf2_trace_json
    );
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::Rlf3Result& result
) {
    write_rlf1_result_transactionally(
        path,
        result,
        rlf::experiments::write_rlf3_world_model_json
    );
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::Rlf3TrainingWorkflowResult& result
) {
    write_rlf1_result_transactionally(
        path,
        result,
        rlf::experiments::write_rlf3_training_json
    );
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::Rlf3EvaluationWorkflowResult& result
) {
    write_rlf1_result_transactionally(
        path,
        result,
        rlf::experiments::write_rlf3_evaluation_json
    );
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::Rlf3TraceWorkflowResult& result
) {
    write_rlf1_result_transactionally(
        path,
        result,
        rlf::experiments::write_rlf3_trace_json
    );
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::Rlf4Result& result
) {
    write_rlf1_result_transactionally(
        path, result, rlf::experiments::write_rlf4_result_json
    );
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::Rlf4TrainingWorkflowResult& result
) {
    write_rlf1_result_transactionally(
        path, result, rlf::experiments::write_rlf4_training_json
    );
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::Rlf4EvaluationWorkflowResult& result
) {
    write_rlf1_result_transactionally(
        path, result, rlf::experiments::write_rlf4_evaluation_json
    );
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::Rlf4TraceWorkflowResult& result
) {
    write_rlf1_result_transactionally(
        path, result, rlf::experiments::write_rlf4_trace_json
    );
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::Rlf5Result& result
) {
    write_rlf1_result_transactionally(
        path, result, rlf::experiments::write_rlf5_result_json
    );
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::Rlf5TrainingWorkflowResult& result
) {
    write_rlf1_result_transactionally(
        path, result, rlf::experiments::write_rlf5_training_json
    );
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::Rlf5EvaluationWorkflowResult& result
) {
    write_rlf1_result_transactionally(
        path, result, rlf::experiments::write_rlf5_evaluation_json
    );
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::Rlf5TraceWorkflowResult& result
) {
    write_rlf1_result_transactionally(
        path, result, rlf::experiments::write_rlf5_trace_json
    );
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::Rlf6Result& result
) {
    write_rlf1_result_transactionally(
        path, result, rlf::experiments::write_rlf6_result_json
    );
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::Rlf6TrainingWorkflowResult& result
) {
    write_rlf1_result_transactionally(
        path, result, rlf::experiments::write_rlf6_training_json
    );
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::Rlf6EvaluationWorkflowResult& result
) {
    write_rlf1_result_transactionally(
        path, result, rlf::experiments::write_rlf6_evaluation_json
    );
}

void write_result_transactionally(
    const std::filesystem::path& path,
    const rlf::experiments::Rlf6TraceWorkflowResult& result
) {
    write_rlf1_result_transactionally(
        path, result, rlf::experiments::write_rlf6_trace_json
    );
}

[[nodiscard]] bool is_rlf6_checkpoint_file(
    const std::filesystem::path& path
) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("unable to open checkpoint: " + path.string());
    }
    char bytes[8]{};
    input.read(bytes, 8);
    return input.gcount() == 8 && std::string_view(bytes, 8) == "RLF6CKP8";
}

void write_rlf6_checkpoint_summary_json(
    std::ostream& output,
    const rlf::storage::Rlf6CheckpointSummary& summary
) {
    output << "{\n"
        << "  \"checkpoint_status\": \"valid\",\n"
        << "  \"architecture\": \"RLF-6\",\n"
        << "  \"format_version\": " << summary.format_version << ",\n"
        << "  \"seed\": " << summary.seed << ",\n"
        << "  \"episode_id\": " << summary.episode_id << ",\n"
        << "  \"step_index\": " << summary.step_index << ",\n"
        << "  \"observation_count\": " << summary.observation_count << ",\n"
        << "  \"belief_count\": " << summary.belief_count << ",\n"
        << "  \"goal_count\": " << summary.goal_count << ",\n"
        << "  \"tool_count\": " << summary.tool_count << ",\n"
        << "  \"transition_count\": " << summary.transition_count << ",\n"
        << "  \"memory_count\": " << summary.memory_count << ",\n"
        << "  \"skill_count\": " << summary.skill_count << ",\n"
        << "  \"error_count\": " << summary.error_count << ",\n"
        << "  \"deterministic_hash\": \""
        << rlf::experiments::format_run_hash(summary.deterministic_hash) << "\",\n"
        << "  \"file_bytes\": " << summary.file_bytes << ",\n"
        << "  \"payload_checksum\": \""
        << rlf::experiments::format_run_hash(summary.payload_checksum)
        << "\"\n}\n";
}

[[nodiscard]] bool is_rlf5_checkpoint_file(
    const std::filesystem::path& path
) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("unable to open checkpoint: " + path.string());
    }
    char bytes[8]{};
    input.read(bytes, 8);
    return input.gcount() == 8 && std::string_view(bytes, 8) == "RLF5CKP7";
}

void write_rlf5_checkpoint_summary_json(
    std::ostream& output,
    const rlf::storage::Rlf5CheckpointSummary& summary
) {
    output << "{\n"
        << "  \"checkpoint_status\": \"valid\",\n"
        << "  \"architecture\": \"RLF-5\",\n"
        << "  \"format_version\": " << summary.format_version << ",\n"
        << "  \"seed\": " << summary.seed << ",\n"
        << "  \"dimension\": " << summary.phase_dimension << ",\n"
        << "  \"lexeme_count\": " << summary.lexeme_count << ",\n"
        << "  \"merge_count\": " << summary.merge_count << ",\n"
        << "  \"context_count\": " << summary.context_count << ",\n"
        << "  \"outcome_count\": " << summary.outcome_count << ",\n"
        << "  \"concept_count\": " << summary.concept_count << ",\n"
        << "  \"construction_count\": " << summary.construction_count << ",\n"
        << "  \"file_bytes\": " << summary.file_bytes << ",\n"
        << "  \"payload_checksum\": \""
        << rlf::experiments::format_run_hash(summary.payload_checksum)
        << "\"\n}\n";
}

[[nodiscard]] bool is_rlf4_checkpoint_file(
    const std::filesystem::path& path
) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("unable to open checkpoint: " + path.string());
    }
    char bytes[8]{};
    input.read(bytes, 8);
    if (input.gcount() != 8) {
        return false;
    }
    return std::string_view(bytes, 8) == "RLF4CKP6";
}

void write_rlf4_checkpoint_summary_json(
    std::ostream& output,
    const rlf::storage::Rlf4CheckpointSummary& summary
) {
    output << "{\n"
        << "  \"checkpoint_status\": \"valid\",\n"
        << "  \"architecture\": \"RLF-4\",\n"
        << "  \"format_version\": " << summary.format_version << ",\n"
        << "  \"seed\": " << summary.seed << ",\n"
        << "  \"training_step\": " << summary.training_step << ",\n"
        << "  \"dimension\": " << summary.dimension << ",\n"
        << "  \"prototype_count\": " << summary.prototype_count << ",\n"
        << "  \"context_count\": " << summary.context_count << ",\n"
        << "  \"outcome_count\": " << summary.outcome_count << ",\n"
        << "  \"option_count\": " << summary.option_count << ",\n"
        << "  \"file_bytes\": " << summary.file_bytes << ",\n"
        << "  \"payload_checksum\": \""
        << rlf::experiments::format_run_hash(summary.payload_checksum)
        << "\"\n}\n";
}

[[nodiscard]] bool is_rlf3_checkpoint_file(
    const std::filesystem::path& path
) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "unable to open checkpoint: " + path.string()
        );
    }
    char bytes[8]{};
    input.read(bytes, 8);
    if (input.gcount() != 8) {
        return false;
    }
    return std::string_view(bytes, 8) == "RLF3CKP5";
}

void write_rlf3_checkpoint_summary_json(
    std::ostream& output,
    const rlf::storage::Rlf3CheckpointSummary& summary
) {
    output
        << "{\n"
        << "  \"checkpoint_status\": \"valid\",\n"
        << "  \"architecture\": \"RLF-3\",\n"
        << "  \"format_version\": " << summary.format_version << ",\n"
        << "  \"seed\": " << summary.seed << ",\n"
        << "  \"training_step\": " << summary.training_step << ",\n"
        << "  \"dimension\": " << summary.dimension << ",\n"
        << "  \"action_count\": " << summary.action_count << ",\n"
        << "  \"state_count\": " << summary.state_count << ",\n"
        << "  \"context_count\": " << summary.context_count << ",\n"
        << "  \"transition_count\": " << summary.transition_count << ",\n"
        << "  \"outcome_count\": " << summary.outcome_count << ",\n"
        << "  \"subgoal_count\": " << summary.subgoal_count << ",\n"
        << "  \"file_bytes\": " << summary.file_bytes << ",\n"
        << "  \"payload_checksum\": \""
        << rlf::experiments::format_run_hash(summary.payload_checksum)
        << "\"\n}\n";
}

[[nodiscard]] bool is_rlf2_checkpoint_file(
    const std::filesystem::path& path
) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "unable to open checkpoint: " + path.string()
        );
    }
    char bytes[8]{};
    input.read(bytes, 8);
    if (input.gcount() != 8) {
        return false;
    }
    return std::string_view(bytes, 8) == "RLF2CKP4";
}

void write_rlf2_checkpoint_summary_json(
    std::ostream& output,
    const rlf::storage::Rlf2CheckpointSummary& summary
) {
    output
        << "{\n"
        << "  \"checkpoint_status\": \"valid\",\n"
        << "  \"architecture\": \"RLF-2\",\n"
        << "  \"format_version\": " << summary.format_version << ",\n"
        << "  \"seed\": " << summary.seed << ",\n"
        << "  \"training_step\": " << summary.training_step << ",\n"
        << "  \"dimension\": " << summary.dimension << ",\n"
        << "  \"operator_count\": " << summary.operator_count << ",\n"
        << "  \"skill_count\": " << summary.skill_count << ",\n"
        << "  \"compound_skill_count\": " << summary.compound_skill_count << ",\n"
        << "  \"prototype_count\": " << summary.prototype_count << ",\n"
        << "  \"file_bytes\": " << summary.file_bytes << ",\n"
        << "  \"payload_checksum\": \""
        << rlf::experiments::format_run_hash(summary.payload_checksum)
        << "\"\n}\n";
}

[[nodiscard]] bool is_rlf1_checkpoint_file(
    const std::filesystem::path& path
) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "unable to open checkpoint: " + path.string()
        );
    }
    char bytes[8]{};
    input.read(bytes, 8);
    if (input.gcount() != 8) {
        return false;
    }
    return std::string_view(bytes, 8) == "RLF1CKP3";
}

void write_rlf1_checkpoint_summary_json(
    std::ostream& output,
    const rlf::storage::Rlf1CheckpointSummary& summary
) {
    output
        << "{\n"
        << "  \"checkpoint_status\": \"valid\",\n"
        << "  \"architecture\": \"RLF-1\",\n"
        << "  \"format_version\": " << summary.format_version << ",\n"
        << "  \"seed\": " << summary.seed << ",\n"
        << "  \"training_step\": " << summary.training_step << ",\n"
        << "  \"dimension\": " << summary.dimension << ",\n"
        << "  \"operator_count\": " << summary.operator_count << ",\n"
        << "  \"macro_operator_count\": " << summary.macro_operator_count << ",\n"
        << "  \"routing_mode_count\": " << summary.routing_mode_count << ",\n"
        << "  \"halt_mode_count\": " << summary.halt_mode_count << ",\n"
        << "  \"route_memory_count\": " << summary.route_memory_count << ",\n"
        << "  \"file_bytes\": " << summary.file_bytes << ",\n"
        << "  \"payload_checksum\": \""
        << rlf::experiments::format_run_hash(summary.payload_checksum)
        << "\"\n}\n";
}

[[nodiscard]] rlf::experiments::Rlf1Config rlf1_config_from_options(
    const CliOptions& options
) {
    const std::size_t dimension =
        options.config_path.empty() && options.dimension == 1'024U
        ? 24U
        : options.dimension;
    return {
        .seed = options.seed,
        .dimension = dimension,
        .training_episodes = options.training_examples,
        .development_episodes = options.development_examples,
        .evaluation_episodes = options.evaluation_examples,
        .training_min_route_length = options.training_min_route_length,
        .training_max_route_length = options.training_max_route_length,
        .evaluation_min_route_length = options.evaluation_min_route_length,
        .evaluation_max_route_length = options.evaluation_max_route_length,
        .maximum_cycles = options.maximum_cycles,
        .operator_count = options.operator_count,
        .state_noise_radians = options.state_noise_radians,
        .goal_similarity_threshold = options.goal_similarity_threshold,
    };
}

[[nodiscard]] rlf::experiments::Rlf2Config rlf2_config_from_options(
    const CliOptions& options
) {
    const std::size_t dimension =
        options.config_path.empty() && options.dimension == 1'024U
        ? 24U
        : options.dimension;
    return {
        .seed = options.seed,
        .dimension = dimension,
        .training_episodes = options.training_examples,
        .development_episodes = options.development_examples,
        .evaluation_episodes = options.evaluation_examples,
        .training_min_route_length = options.training_min_route_length,
        .training_max_route_length = options.training_max_route_length,
        .evaluation_min_route_length = options.evaluation_min_route_length,
        .evaluation_max_route_length = options.evaluation_max_route_length,
        .maximum_cycles = options.maximum_cycles,
        .operator_count = options.operator_count,
        .state_noise_radians = options.state_noise_radians,
        .goal_similarity_threshold = options.goal_similarity_threshold,
    };
}

[[nodiscard]] rlf::experiments::Rlf6Config rlf6_config_from_options(
    const CliOptions& options
) {
    return {
        .seed = options.seed,
        .training_episodes = options.rlf6_training_episodes,
        .evaluation_episodes = options.rlf6_evaluation_episodes,
        .minimum_route_length = options.rlf6_minimum_route_length,
        .maximum_route_length = options.rlf6_maximum_route_length,
        .stress_episodes = options.rlf6_stress_episodes,
        .stress_route_length = options.rlf6_stress_route_length,
        .action_budget = options.rlf6_action_budget,
        .planning_node_budget = options.planner_node_budget,
        .tool_budget = options.rlf6_tool_budget,
        .memory_limit_records = options.rlf6_memory_limit_records,
        .tool_cost_budget = options.rlf6_tool_cost_budget,
        .risk_budget = options.rlf6_risk_budget,
        .threads = options.threads,
        .deterministic = options.deterministic,
        .include_stress = options.rlf6_include_stress,
        .experiment_name = options.experiment_name.empty()
            ? std::string("long_horizon_planning")
            : options.experiment_name,
    };
}

[[nodiscard]] rlf::experiments::Rlf5Config rlf5_config_from_options(
    const CliOptions& options
) {
    const std::size_t dimension =
        options.config_path.empty() && options.dimension == 1'024U
        ? 64U
        : options.dimension;
    return {
        .seed = options.seed,
        .phase_dimension = dimension,
        .raw_training_sentences = options.language_raw_training_sentences,
        .supervised_training_examples = options.language_supervised_training_examples,
        .evaluation_examples = options.language_evaluation_examples,
        .qa_episodes = options.language_qa_episodes,
        .free_generation_samples = options.language_free_generation_samples,
        .maximum_lexemes = options.language_maximum_lexemes,
        .maximum_merges = options.language_maximum_merges,
        .minimum_pair_support = options.language_minimum_pair_support,
        .maximum_context_order = options.language_context_order,
        .minimum_context_support = options.language_minimum_context_support,
        .maximum_constructions = options.language_maximum_constructions,
        .minimum_construction_support = options.language_minimum_construction_support,
        .maximum_generation_tokens = options.language_maximum_generation_tokens,
        .holdout_modulus = options.language_holdout_modulus,
    };
}

[[nodiscard]] bool is_rlf5_experiment_name(const std::string_view name) noexcept {
    return name == "compositional_language" ||
        name == "language_understanding" ||
        name == "language_generation" ||
        name == "semantic_qa";
}

[[nodiscard]] rlf::experiments::Rlf4Config rlf4_config_from_options(
    const CliOptions& options
) {
    const std::size_t dimension =
        options.config_path.empty() && options.dimension == 1'024U
        ? 48U
        : options.dimension;
    return {
        .seed = options.seed,
        .dimension = dimension,
        .symbol_count = std::max<std::size_t>(18U, options.symbol_count),
        .training_tokens = options.temporal_training_tokens,
        .evaluation_tokens = options.temporal_evaluation_tokens,
        .adaptation_tokens = options.temporal_adaptation_tokens,
        .maximum_context_order = options.temporal_context_order,
        .minimum_context_support = 2U,
        .maximum_options = options.temporal_maximum_options,
        .minimum_option_support = options.temporal_minimum_option_support,
        .forecast_horizon = options.temporal_forecast_horizon,
        .forecast_samples = options.temporal_forecast_samples,
        .change_tolerance = options.temporal_change_tolerance,
        .training_noise_radians = options.temporal_training_noise_radians,
        .evaluation_noise_radians = options.temporal_evaluation_noise_radians,
        .dominant_motif_probability = options.dominant_probability,
        .prototype_merge_distance = options.temporal_prototype_merge_distance,
        .recent_decay = options.temporal_recent_decay,
        .recent_weight = options.temporal_recent_weight,
    };
}

[[nodiscard]] bool is_rlf4_experiment_name(
    const std::string_view name
) noexcept {
    return name == "self_supervised_temporal" ||
        name == "temporal_prediction" ||
        name == "temporal_abstraction" ||
        name == "regime_adaptation";
}

[[nodiscard]] rlf::experiments::Rlf3Config rlf3_config_from_options(
    const CliOptions& options
) {
    const std::size_t dimension =
        options.config_path.empty() && options.dimension == 1'024U
        ? 32U
        : options.dimension;
    return {
        .seed = options.seed,
        .dimension = dimension,
        .layers = options.world_layers,
        .lanes = options.world_lanes,
        .transition_samples_per_case = options.transition_samples_per_case,
        .training_routes = options.training_routes,
        .evaluation_episodes = options.evaluation_examples,
        .stochastic_rollouts_per_episode = options.stochastic_rollouts,
        .maximum_execution_steps = options.maximum_execution_steps,
        .planner_node_budget = options.planner_node_budget,
        .maximum_plan_depth = options.maximum_plan_depth,
        .observation_noise_radians = options.observation_noise_radians,
        .stochastic_dominant_probability = options.dominant_probability,
        .goal_similarity_threshold = options.goal_similarity_threshold,
    };
}

[[nodiscard]] bool is_rlf3_experiment_name(
    const std::string_view name
) noexcept {
    return name == "sparse_world_model" ||
        name == "learned_transition_model" ||
        name == "sparse_subgoal_index" ||
        name == "partial_observation" ||
        name == "stochastic_planning";
}

[[nodiscard]] bool is_rlf2_experiment_name(
    const std::string_view name
) noexcept {
    return name == "predictive_reasoning" ||
        name == "successor_prediction" ||
        name == "causal_subgoals" ||
        name == "intervention_credit" ||
        name == "skill_consolidation";
}

[[nodiscard]] bool is_rlf1_experiment_name(
    const std::string_view name
) noexcept {
    return name == "latent_routing" ||
        name == "withheld_routes" ||
        name == "adaptive_halting" ||
        name == "delayed_credit" ||
        name == "route_recovery" ||
        name == "route_continual_learning" ||
        name == "macro_consolidation";
}

void write_checkpoint_summary_json(
    std::ostream& output,
    const rlf::storage::CheckpointSummary& summary
) {
    output
        << "{\n"
        << "  \"checkpoint_status\": \"valid\",\n"
        << "  \"format_version\": " << summary.format_version << ",\n"
        << "  \"master_seed\": " << summary.master_seed << ",\n"
        << "  \"training_step\": " << summary.training_step << ",\n"
        << "  \"dimension\": " << summary.dimension << ",\n"
        << "  \"mode_count\": " << summary.mode_count << ",\n"
        << "  \"enabled_mode_count\": "
        << summary.enabled_mode_count << ",\n"
        << "  \"associative_record_count\": "
        << summary.associative_record_count << ",\n"
        << "  \"associative_capacity\": "
        << summary.associative_capacity << ",\n"
        << "  \"update_strategy\": \""
        << summary.update_strategy << "\",\n"
        << "  \"modes_created\": "
        << summary.structural_statistics.modes_created << ",\n"
        << "  \"modes_split\": "
        << summary.structural_statistics.modes_split << ",\n"
        << "  \"modes_merged\": "
        << summary.structural_statistics.modes_merged << ",\n"
        << "  \"modes_pruned\": "
        << summary.structural_statistics.modes_pruned << ",\n"
        << "  \"file_bytes\": " << summary.file_bytes << ",\n"
        << "  \"payload_checksum\": \""
        << rlf::experiments::format_run_hash(summary.payload_checksum)
        << "\",\n"
        << "  \"metadata\": {";
    bool first_entry = true;
    for (const auto& [key, value] : summary.experiment_metadata) {
        if (!first_entry) {
            output << ',';
        }
        output << "\n    \"" << key << "\": \"" << value << "\"";
        first_entry = false;
    }
    if (!summary.experiment_metadata.empty()) {
        output << '\n';
    }
    output << "  }\n}\n";
}

int run(const int argument_count, char** argument_values) {
    const CliOptions options = parse_arguments(argument_count, argument_values);
    if (options.show_help || argument_count == 1) {
        print_help(std::cout);
        return 0;
    }
    const bool supports_multiple_threads =
        options.command == "benchmark" ||
        options.task_name == "agent" ||
        (options.command == "experiment" &&
         (options.experiment_name == "optimization_benchmark" ||
          rlf::experiments::is_rlf6_experiment_name(options.experiment_name) ||
          options.experiment_name == "rlf7_full" ||
          options.experiment_name == "frontier_full"));
    if (options.threads != 1U && !supports_multiple_threads) {
        throw std::invalid_argument(
            "--threads greater than 1 is only supported by "
            "optimization benchmarks"
        );
    }
    if (options.command == "train") {
        if (options.checkpoint_path.empty()) {
            throw std::invalid_argument(
                "--checkpoint is required for train"
            );
        }
        const std::string task = options.task_name.empty()
            ? rlf::experiments::checkpoint_workflow_task
            : options.task_name;
        if (task == "rlf7" || task == "frontier") {
            rlf::frontier::FrontierModel model(options.seed);
            rlf::frontier::TrainingReport report;
            const auto backend = rlf::frontier::parse_frontier_backend(options.frontier_backend);
            if (!options.manifest_path.empty()) {
                const auto manifest = rlf::frontier::ManifestLoader::load(options.manifest_path);
                rlf::frontier::FrontierTrainer trainer(std::move(model), backend);
                report = trainer.train(manifest, true);
                model = std::move(trainer.model());
            } else {
                rlf::experiments::Rlf7FrontierConfig config;
                config.seed = options.seed;
                config.knowledge_records = options.frontier_knowledge_records;
                config.knowledge_queries = options.frontier_knowledge_queries;
                config.threads = options.threads;
                config.include_audio = options.frontier_include_audio;
                config.frontier_mode = task == "frontier";
                config.backend = backend;
                config.agent_evaluation_episodes = options.rlf6_evaluation_episodes;
                static_cast<void>(rlf::experiments::run_rlf7_frontier(config, &model));
                report.entries_seen = static_cast<std::size_t>(model.training_examples);
                report.knowledge_records_added = model.fabric.records().size();
                report.modes_learned = model.fabric.modes().size();
                report.deterministic_hash = model.fabric.deterministic_hash();
            }
            if (task == "frontier") {
                rlf::storage::save_frontier_checkpoint(
                    options.checkpoint_path,
                    model,
                    std::string(rlf::frontier::to_string(backend))
                );
            } else {
                rlf::storage::save_rlf7_checkpoint(options.checkpoint_path, model);
            }
            rlf::frontier::write_training_report_json(std::cout, report);
            if (!options.output_path.empty()) {
                std::ofstream output(options.output_path, std::ios::trunc);
                if (!output) throw std::runtime_error("unable to open result output");
                rlf::frontier::write_training_report_json(output, report);
            }
            return 0;
        }
        if (task == "agent") {
            const rlf::experiments::Rlf6TrainingWorkflowResult result =
                rlf::experiments::train_rlf6_checkpoint(
                    rlf6_config_from_options(options), options.checkpoint_path
                );
            rlf::experiments::write_rlf6_training_json(std::cout, result);
            write_result_transactionally(options.output_path, result);
            return 0;
        }
        if (task == "compositional_language") {
            const std::uint64_t estimated_memory = estimated_rlf5_memory(options);
            if (options.max_memory_bytes != 0ULL &&
                estimated_memory > options.max_memory_bytes) {
                throw std::runtime_error(
                    "estimated RLF-5 training storage exceeds --max-memory"
                );
            }
            const rlf::experiments::Rlf5TrainingWorkflowResult result =
                rlf::experiments::train_rlf5_checkpoint(
                    rlf5_config_from_options(options), options.checkpoint_path
                );
            rlf::experiments::write_rlf5_training_json(std::cout, result);
            write_result_transactionally(options.output_path, result);
            return 0;
        }
        if (task == "self_supervised_temporal") {
            const std::uint64_t estimated_memory = estimated_rlf4_memory(options);
            if (options.max_memory_bytes != 0ULL &&
                estimated_memory > options.max_memory_bytes) {
                throw std::runtime_error(
                    "estimated RLF-4 training storage exceeds --max-memory"
                );
            }
            const rlf::experiments::Rlf4TrainingWorkflowResult result =
                rlf::experiments::train_rlf4_checkpoint(
                    rlf4_config_from_options(options), options.checkpoint_path
                );
            rlf::experiments::write_rlf4_training_json(std::cout, result);
            write_result_transactionally(options.output_path, result);
            return 0;
        }
        if (task == "sparse_world_model") {
            const rlf::experiments::Rlf3TrainingWorkflowResult result =
                rlf::experiments::train_rlf3_checkpoint(
                    rlf3_config_from_options(options),
                    options.checkpoint_path
                );
            rlf::experiments::write_rlf3_training_json(std::cout, result);
            write_result_transactionally(options.output_path, result);
            return 0;
        }
        if (task == "predictive_reasoning") {
            const rlf::experiments::Rlf2TrainingWorkflowResult result =
                rlf::experiments::train_rlf2_checkpoint(
                    rlf2_config_from_options(options),
                    options.checkpoint_path
                );
            rlf::experiments::write_rlf2_training_json(
                std::cout,
                result
            );
            write_result_transactionally(options.output_path, result);
            return 0;
        }
        if (task == "latent_routing") {
            const rlf::experiments::Rlf1TrainingWorkflowResult result =
                rlf::experiments::train_rlf1_checkpoint(
                    rlf1_config_from_options(options),
                    options.checkpoint_path
                );
            rlf::experiments::write_rlf1_training_json(
                std::cout,
                result
            );
            write_result_transactionally(options.output_path, result);
            return 0;
        }
        if (task != rlf::experiments::checkpoint_workflow_task) {
            throw std::invalid_argument(
                "unsupported training task: " + task
            );
        }
        const rlf::experiments::CheckpointTrainingResult result =
            rlf::experiments::train_checkpoint_workflow({
                .seed = options.seed,
                .dimension = options.dimension,
                .training_examples = options.training_examples,
                .evaluation_examples = options.evaluation_examples,
                .checkpoint_path = options.checkpoint_path,
            });
        rlf::experiments::write_checkpoint_training_json(
            std::cout,
            result
        );
        write_result_transactionally(options.output_path, result);
        return 0;
    }
    if (options.command == "evaluate") {
        if (options.checkpoint_path.empty() ||
            options.task_name.empty()) {
            throw std::invalid_argument(
                "--checkpoint and --task are required for evaluate"
            );
        }
        if (options.task_name == "rlf7" || options.task_name == "frontier") {
            if (options.manifest_path.empty()) {
                throw std::invalid_argument("--manifest is required for RLF-7/Frontier evaluation");
            }
            rlf::frontier::FrontierModel model = options.task_name == "frontier"
                ? rlf::storage::load_frontier_checkpoint(options.checkpoint_path)
                : rlf::storage::load_rlf7_checkpoint(options.checkpoint_path);
            const auto backend = rlf::frontier::parse_frontier_backend(options.frontier_backend);
            rlf::frontier::FrontierTrainer trainer(std::move(model), backend);
            const auto manifest = rlf::frontier::ManifestLoader::load(options.manifest_path);
            const auto report = trainer.evaluate(manifest);
            rlf::frontier::write_evaluation_report_json(std::cout, report);
            if (!options.output_path.empty()) {
                std::ofstream output(options.output_path, std::ios::trunc);
                if (!output) throw std::runtime_error("unable to open result output");
                rlf::frontier::write_evaluation_report_json(output, report);
            }
            return 0;
        }
        if (options.task_name == "agent") {
            const rlf::experiments::Rlf6EvaluationWorkflowResult result =
                rlf::experiments::evaluate_rlf6_checkpoint(
                    options.checkpoint_path, rlf6_config_from_options(options)
                );
            rlf::experiments::write_rlf6_evaluation_json(std::cout, result);
            write_result_transactionally(options.output_path, result);
            return 0;
        }
        if (options.task_name == "compositional_language") {
            const rlf::experiments::Rlf5EvaluationWorkflowResult result =
                rlf::experiments::evaluate_rlf5_checkpoint(
                    options.checkpoint_path, options.seed,
                    options.language_evaluation_examples
                );
            rlf::experiments::write_rlf5_evaluation_json(std::cout, result);
            write_result_transactionally(options.output_path, result);
            return 0;
        }
        if (options.task_name == "self_supervised_temporal") {
            const rlf::experiments::Rlf4EvaluationWorkflowResult result =
                rlf::experiments::evaluate_rlf4_checkpoint(
                    options.checkpoint_path, options.seed,
                    options.temporal_evaluation_tokens
                );
            rlf::experiments::write_rlf4_evaluation_json(std::cout, result);
            write_result_transactionally(options.output_path, result);
            return 0;
        }
        if (options.task_name == "sparse_world_model") {
            const rlf::experiments::Rlf3EvaluationWorkflowResult result =
                rlf::experiments::evaluate_rlf3_checkpoint(
                    options.checkpoint_path,
                    options.seed,
                    options.evaluation_examples
                );
            rlf::experiments::write_rlf3_evaluation_json(std::cout, result);
            write_result_transactionally(options.output_path, result);
            return 0;
        }
        if (options.task_name == "predictive_reasoning") {
            const rlf::experiments::Rlf2EvaluationWorkflowResult result =
                rlf::experiments::evaluate_rlf2_checkpoint(
                    options.checkpoint_path,
                    options.seed,
                    options.evaluation_examples
                );
            rlf::experiments::write_rlf2_evaluation_json(
                std::cout,
                result
            );
            write_result_transactionally(options.output_path, result);
            return 0;
        }
        if (options.task_name == "latent_routing") {
            const rlf::experiments::Rlf1EvaluationWorkflowResult result =
                rlf::experiments::evaluate_rlf1_checkpoint(
                    options.checkpoint_path,
                    options.seed,
                    options.evaluation_examples
                );
            rlf::experiments::write_rlf1_evaluation_json(
                std::cout,
                result
            );
            write_result_transactionally(options.output_path, result);
            return 0;
        }
        const rlf::experiments::CheckpointEvaluationResult result =
            rlf::experiments::evaluate_checkpoint_workflow(
                options.checkpoint_path,
                options.task_name,
                options.evaluation_examples
            );
        rlf::experiments::write_checkpoint_evaluation_json(
            std::cout,
            result
        );
        write_result_transactionally(options.output_path, result);
        return 0;
    }
    if (options.command == "trace") {
        if (options.checkpoint_path.empty() ||
            options.task_name.empty()) {
            throw std::invalid_argument(
                "--checkpoint and --task are required for trace"
            );
        }
        if (options.task_name == "agent") {
            const rlf::experiments::Rlf6TraceWorkflowResult result =
                rlf::experiments::trace_rlf6_checkpoint(
                    options.checkpoint_path, rlf6_config_from_options(options),
                    options.sample_id
                );
            rlf::experiments::write_rlf6_trace_json(std::cout, result);
            write_result_transactionally(options.output_path, result);
            return 0;
        }
        if (options.task_name == "compositional_language") {
            const rlf::experiments::Rlf5TraceWorkflowResult result =
                rlf::experiments::trace_rlf5_checkpoint(
                    options.checkpoint_path, options.seed, options.sample_id
                );
            rlf::experiments::write_rlf5_trace_json(std::cout, result);
            write_result_transactionally(options.output_path, result);
            return 0;
        }
        if (options.task_name == "self_supervised_temporal") {
            const rlf::experiments::Rlf4TraceWorkflowResult result =
                rlf::experiments::trace_rlf4_checkpoint(
                    options.checkpoint_path, options.seed, options.sample_id
                );
            rlf::experiments::write_rlf4_trace_json(std::cout, result);
            write_result_transactionally(options.output_path, result);
            return 0;
        }
        if (options.task_name == "sparse_world_model") {
            const rlf::experiments::Rlf3TraceWorkflowResult result =
                rlf::experiments::trace_rlf3_checkpoint(
                    options.checkpoint_path,
                    options.seed,
                    options.sample_id
                );
            rlf::experiments::write_rlf3_trace_json(std::cout, result);
            write_result_transactionally(options.output_path, result);
            return 0;
        }
        if (options.task_name == "predictive_reasoning") {
            const rlf::experiments::Rlf2TraceWorkflowResult result =
                rlf::experiments::trace_rlf2_checkpoint(
                    options.checkpoint_path,
                    options.seed,
                    options.sample_id
                );
            rlf::experiments::write_rlf2_trace_json(
                std::cout,
                result
            );
            write_result_transactionally(options.output_path, result);
            return 0;
        }
        if (options.task_name == "latent_routing") {
            const rlf::experiments::Rlf1TraceWorkflowResult result =
                rlf::experiments::trace_rlf1_checkpoint(
                    options.checkpoint_path,
                    options.seed,
                    options.sample_id
                );
            rlf::experiments::write_rlf1_trace_json(
                std::cout,
                result
            );
            write_result_transactionally(options.output_path, result);
            return 0;
        }
        const rlf::experiments::CheckpointTraceResult result =
            rlf::experiments::trace_checkpoint_workflow(
                options.checkpoint_path,
                options.task_name,
                options.sample_id
            );
        rlf::experiments::write_checkpoint_trace_json(
            std::cout,
            result
        );
        write_result_transactionally(options.output_path, result);
        return 0;
    }
    if (options.command == "benchmark") {
        if (options.suite_name != "core") {
            throw std::invalid_argument(
                "--suite core is required for benchmark"
            );
        }
        const std::size_t largest_mode_count =
            *std::max_element(
                options.mode_counts.begin(),
                options.mode_counts.end()
            );
        const std::uint64_t capacity_memory =
            estimated_phase_storage(
                largest_mode_count * 2U,
                options.dimension
            );
        const std::uint64_t optimization_memory =
            estimated_optimization_memory(options);
        const std::uint64_t estimated_memory =
            std::max(capacity_memory, optimization_memory);
        if (options.max_memory_bytes != 0ULL &&
            estimated_memory > options.max_memory_bytes) {
            throw std::runtime_error(
                "estimated benchmark phase storage exceeds --max-memory"
            );
        }
        const rlf::experiments::CapacityScalingResult capacity =
            rlf::experiments::run_capacity_scaling({
                .seed = options.seed,
                .dimension = options.dimension,
                .evaluation_queries = options.evaluation_queries,
                .candidate_count = options.candidate_count,
                .active_count = options.active_count,
                .noise_radians = options.noise_radians,
                .mode_counts = options.mode_counts,
            });
        const rlf::experiments::OperatorCompositionResult operators =
            rlf::experiments::run_operator_composition({
                .seed = options.seed ^ 0x4F50455241544F52ULL,
                .dimension = options.dimension,
                .context_dimensions = options.context_dimensions,
                .training_examples = options.training_examples,
                .evaluation_examples = options.evaluation_examples,
                .noise_radians = options.noise_radians,
            });
        const rlf::experiments::OptimizationBenchmarkResult optimization =
            rlf::experiments::run_optimization_benchmark({
                .seed = options.seed ^ 0x42454E43484D4152ULL,
                .dimension = options.dimension,
                .mode_count = options.mode_count,
                .query_count = options.evaluation_queries,
                .candidate_count = options.candidate_count,
                .thread_count = options.threads,
                .similarity_iterations = options.samples,
                .quantization_samples =
                    options.quantization_samples,
            });
        if (!options.output_path.empty()) {
            if (options.output_path.has_parent_path()) {
                std::filesystem::create_directories(
                    options.output_path.parent_path()
                );
            }
            std::filesystem::path temporary = options.output_path;
            temporary += ".tmp";
            std::ofstream output(
                temporary,
                std::ios::binary | std::ios::trunc
            );
            if (!output) {
                throw std::runtime_error(
                    "unable to open benchmark result file"
                );
            }
            output
                << "{\n"
                << "  \"suite\": \"core\",\n"
                << "  \"capacity_scaling\": ";
            rlf::experiments::write_capacity_scaling_json(
                output,
                capacity
            );
            output
                << ",\n  \"operator_composition\": ";
            rlf::experiments::write_operator_composition_json(
                output,
                operators
            );
            output
                << ",\n  \"optimization_benchmark\": ";
            rlf::experiments::write_optimization_benchmark_json(
                output,
                optimization
            );
            output << "}\n";
            output.close();
            std::filesystem::rename(temporary, options.output_path);
        }
        std::cout
            << "{\n"
            << "  \"suite\": \"core\",\n"
            << "  \"largest_mode_count\": "
            << capacity.scales.back().mode_count << ",\n"
            << "  \"operator_unseen_composition_accuracy\": "
            << operators.operator_unseen_composition_accuracy << ",\n"
            << "  \"similarity_speedup\": "
            << optimization.similarity_speedup << ",\n"
            << "  \"cuda_scientifically_justified\": "
            << (optimization.cuda_scientifically_justified
                    ? "true"
                    : "false")
            << ",\n"
            << "  \"scientific_decision\": \""
            << operators.scientific_decision << "\"\n"
            << "}\n";
        return 0;
    }
    if (options.command == "generate") {
        if (options.checkpoint_path.empty() || options.generation_path.empty()) {
            throw std::invalid_argument("--checkpoint and --generate-dir are required for generate");
        }
        rlf::frontier::FrontierModel model;
        const auto summary = rlf::storage::inspect_frontier_checkpoint(options.checkpoint_path);
        if (summary.architecture == "RLF-Frontier") {
            model = rlf::storage::load_frontier_checkpoint(options.checkpoint_path);
        } else {
            model = rlf::storage::load_rlf7_checkpoint(options.checkpoint_path);
        }
        std::filesystem::create_directories(options.generation_path);
        bool image_written = false;
        bool video_written = false;
        bool audio_written = false;
        for (const auto& [id, mode] : model.fabric.modes()) {
            static_cast<void>(id);
            if (!image_written && mode.modality == rlf::frontier::Modality::image) {
                rlf::frontier::PrototypeGenerator::generate_visual_mode(options.generation_path / "image_prototype.ppm", mode);
                image_written = true;
            } else if (!video_written && mode.modality == rlf::frontier::Modality::video) {
                rlf::frontier::PrototypeGenerator::generate_video_mode(options.generation_path / "video_prototype", mode);
                video_written = true;
            } else if (!audio_written && mode.modality == rlf::frontier::Modality::audio) {
                rlf::frontier::PrototypeGenerator::generate_audio_mode(options.generation_path / "audio_prototype.wav", mode);
                audio_written = true;
            }
        }
        std::cout << "{\n  \"generation_kind\": \"prototype_only\",\n"
            << "  \"image_written\": " << (image_written ? "true" : "false") << ",\n"
            << "  \"video_written\": " << (video_written ? "true" : "false") << ",\n"
            << "  \"audio_written\": " << (audio_written ? "true" : "false") << "\n}\n";
        return 0;
    }
    if (options.command == "inspect" ||
        options.command == "inspect-agent" ||
        options.command == "verify-checkpoint") {
        if (options.checkpoint_path.empty()) {
            throw std::invalid_argument(
                "--checkpoint is required for checkpoint commands"
            );
        }
        {
            std::ifstream input(options.checkpoint_path, std::ios::binary);
            char magic[8]{};
            input.read(magic, 8);
            const std::string_view checkpoint_magic(magic, static_cast<std::size_t>(input.gcount()));
            if (checkpoint_magic == "RLF7CKP9" || checkpoint_magic == "RLFFRT10") {
                const auto summary = rlf::storage::inspect_frontier_checkpoint(options.checkpoint_path);
                std::cout << "{\n  \"checkpoint_status\": \"valid\",\n"
                    << "  \"architecture\": \"" << summary.architecture << "\",\n"
                    << "  \"format_version\": " << summary.format_version << ",\n"
                    << "  \"knowledge_records\": " << summary.knowledge_records << ",\n"
                    << "  \"modes\": " << summary.modes << ",\n"
                    << "  \"deterministic_hash\": \""
                    << rlf::experiments::format_run_hash(summary.deterministic_hash) << "\",\n"
                    << "  \"payload_checksum\": \""
                    << rlf::experiments::format_run_hash(summary.payload_checksum) << "\"\n}\n";
                return 0;
            }
        }
        if (is_rlf6_checkpoint_file(options.checkpoint_path)) {
            const rlf::storage::Rlf6CheckpointSummary summary =
                rlf::storage::inspect_rlf6_checkpoint(options.checkpoint_path);
            if (options.command == "verify-checkpoint") {
                std::cout << "{\n"
                    << "  \"checkpoint_status\": \"valid\",\n"
                    << "  \"architecture\": \"RLF-6\",\n"
                    << "  \"format_version\": " << summary.format_version << ",\n"
                    << "  \"deterministic_hash\": \""
                    << rlf::experiments::format_run_hash(summary.deterministic_hash)
                    << "\",\n"
                    << "  \"payload_checksum\": \""
                    << rlf::experiments::format_run_hash(summary.payload_checksum)
                    << "\"\n}\n";
            } else {
                write_rlf6_checkpoint_summary_json(std::cout, summary);
            }
            return 0;
        }
        if (is_rlf5_checkpoint_file(options.checkpoint_path)) {
            const rlf::storage::Rlf5CheckpointSummary summary =
                rlf::storage::inspect_rlf5_checkpoint(options.checkpoint_path);
            if (options.command == "verify-checkpoint") {
                std::cout << "{\n"
                    << "  \"checkpoint_status\": \"valid\",\n"
                    << "  \"architecture\": \"RLF-5\",\n"
                    << "  \"format_version\": " << summary.format_version << ",\n"
                    << "  \"dimension\": " << summary.phase_dimension << ",\n"
                    << "  \"payload_checksum\": \""
                    << rlf::experiments::format_run_hash(summary.payload_checksum)
                    << "\"\n}\n";
            } else {
                write_rlf5_checkpoint_summary_json(std::cout, summary);
            }
            return 0;
        }
        if (is_rlf4_checkpoint_file(options.checkpoint_path)) {
            const rlf::storage::Rlf4CheckpointSummary summary =
                rlf::storage::inspect_rlf4_checkpoint(options.checkpoint_path);
            if (options.command == "verify-checkpoint") {
                std::cout << "{\n"
                    << "  \"checkpoint_status\": \"valid\",\n"
                    << "  \"architecture\": \"RLF-4\",\n"
                    << "  \"format_version\": " << summary.format_version << ",\n"
                    << "  \"dimension\": " << summary.dimension << ",\n"
                    << "  \"payload_checksum\": \""
                    << rlf::experiments::format_run_hash(summary.payload_checksum)
                    << "\"\n}\n";
            } else {
                write_rlf4_checkpoint_summary_json(std::cout, summary);
            }
            return 0;
        }
        if (is_rlf3_checkpoint_file(options.checkpoint_path)) {
            const rlf::storage::Rlf3CheckpointSummary summary =
                rlf::storage::inspect_rlf3_checkpoint(options.checkpoint_path);
            if (options.command == "verify-checkpoint") {
                std::cout
                    << "{\n"
                    << "  \"checkpoint_status\": \"valid\",\n"
                    << "  \"architecture\": \"RLF-3\",\n"
                    << "  \"format_version\": " << summary.format_version << ",\n"
                    << "  \"dimension\": " << summary.dimension << ",\n"
                    << "  \"payload_checksum\": \""
                    << rlf::experiments::format_run_hash(summary.payload_checksum)
                    << "\"\n}\n";
            } else {
                write_rlf3_checkpoint_summary_json(std::cout, summary);
            }
            return 0;
        }
        if (is_rlf2_checkpoint_file(options.checkpoint_path)) {
            const rlf::storage::Rlf2CheckpointSummary summary =
                rlf::storage::inspect_rlf2_checkpoint(
                    options.checkpoint_path
                );
            if (options.command == "verify-checkpoint") {
                std::cout
                    << "{\n"
                    << "  \"checkpoint_status\": \"valid\",\n"
                    << "  \"architecture\": \"RLF-2\",\n"
                    << "  \"format_version\": "
                    << summary.format_version << ",\n"
                    << "  \"dimension\": "
                    << summary.dimension << ",\n"
                    << "  \"payload_checksum\": \""
                    << rlf::experiments::format_run_hash(
                           summary.payload_checksum
                       )
                    << "\"\n"
                    << "}\n";
            } else {
                write_rlf2_checkpoint_summary_json(std::cout, summary);
            }
            return 0;
        }
        if (is_rlf1_checkpoint_file(options.checkpoint_path)) {
            const rlf::storage::Rlf1CheckpointSummary summary =
                rlf::storage::inspect_rlf1_checkpoint(
                    options.checkpoint_path
                );
            if (options.command == "verify-checkpoint") {
                std::cout
                    << "{\n"
                    << "  \"checkpoint_status\": \"valid\",\n"
                    << "  \"architecture\": \"RLF-1\",\n"
                    << "  \"format_version\": "
                    << summary.format_version << ",\n"
                    << "  \"dimension\": "
                    << summary.dimension << ",\n"
                    << "  \"payload_checksum\": \""
                    << rlf::experiments::format_run_hash(
                           summary.payload_checksum
                       )
                    << "\"\n"
                    << "}\n";
            } else {
                write_rlf1_checkpoint_summary_json(std::cout, summary);
            }
            return 0;
        }
        const rlf::storage::CheckpointSummary summary =
            rlf::storage::inspect_checkpoint(options.checkpoint_path);
        if (options.command == "verify-checkpoint") {
            std::cout
                << "{\n"
                << "  \"checkpoint_status\": \"valid\",\n"
                << "  \"format_version\": "
                << summary.format_version << ",\n"
                << "  \"dimension\": " << summary.dimension << ",\n"
                << "  \"payload_checksum\": \""
                << rlf::experiments::format_run_hash(
                       summary.payload_checksum
                   )
                << "\"\n"
                << "}\n";
        } else {
            write_checkpoint_summary_json(std::cout, summary);
        }
        return 0;
    }
    if (options.command != "experiment") {
        throw std::invalid_argument("a supported command is required");
    }
    const std::uint64_t estimated_memory =
        estimated_experiment_memory(options);
    if (options.max_memory_bytes != 0ULL &&
        estimated_memory > options.max_memory_bytes) {
        throw std::runtime_error(
            "estimated experiment phase storage exceeds --max-memory"
        );
    }

    if (options.experiment_name == "rlf7_full" || options.experiment_name == "frontier_full") {
        rlf::experiments::Rlf7FrontierConfig config;
        config.seed = options.seed;
        config.knowledge_records = options.frontier_knowledge_records;
        config.knowledge_queries = options.frontier_knowledge_queries;
        config.threads = options.threads;
        config.include_audio = options.frontier_include_audio;
        config.run_agent_gate = options.frontier_run_agent_gate;
        config.frontier_mode = options.experiment_name == "frontier_full";
        config.backend = rlf::frontier::parse_frontier_backend(options.frontier_backend);
        config.agent_evaluation_episodes = options.rlf6_evaluation_episodes;
        const auto result = rlf::experiments::run_rlf7_frontier(config);
        rlf::experiments::write_rlf7_frontier_json(std::cout, result);
        if (!options.output_path.empty()) {
            std::ofstream output(options.output_path, std::ios::trunc);
            if (!output) throw std::runtime_error("unable to open result output");
            rlf::experiments::write_rlf7_frontier_json(output, result);
        }
        return 0;
    }

    if (rlf::experiments::is_rlf6_experiment_name(options.experiment_name)) {
        const rlf::experiments::Rlf6Result result =
            rlf::experiments::run_rlf6_agent(rlf6_config_from_options(options));
        rlf::experiments::write_rlf6_result_json(std::cout, result);
        write_result_transactionally(options.output_path, result);
        return 0;
    }

    if (is_rlf5_experiment_name(options.experiment_name)) {
        const rlf::experiments::Rlf5Result result =
            rlf::experiments::run_rlf5_language(
                rlf5_config_from_options(options)
            );
        rlf::experiments::write_rlf5_result_json(std::cout, result);
        write_result_transactionally(options.output_path, result);
        return 0;
    }

    if (is_rlf4_experiment_name(options.experiment_name)) {
        const rlf::experiments::Rlf4Result result =
            rlf::experiments::run_rlf4_self_supervised(
                rlf4_config_from_options(options)
            );
        rlf::experiments::write_rlf4_result_json(std::cout, result);
        write_result_transactionally(options.output_path, result);
        return 0;
    }

    if (is_rlf3_experiment_name(options.experiment_name)) {
        const rlf::experiments::Rlf3Result result =
            rlf::experiments::run_rlf3_world_model(
                rlf3_config_from_options(options)
            );
        rlf::experiments::write_rlf3_world_model_json(std::cout, result);
        write_result_transactionally(options.output_path, result);
        return 0;
    }

    if (is_rlf2_experiment_name(options.experiment_name)) {
        const rlf::experiments::Rlf2Result result =
            rlf::experiments::run_rlf2_predictive_reasoning(
                rlf2_config_from_options(options)
            );
        rlf::experiments::write_rlf2_predictive_reasoning_json(
            std::cout,
            result
        );
        write_result_transactionally(options.output_path, result);
        return 0;
    }

    if (is_rlf1_experiment_name(options.experiment_name)) {
        const rlf::experiments::Rlf1Result result =
            rlf::experiments::run_rlf1_latent_routing(
                rlf1_config_from_options(options)
            );
        rlf::experiments::write_rlf1_latent_routing_json(
            std::cout,
            result
        );
        write_result_transactionally(options.output_path, result);
        return 0;
    }

    if (options.experiment_name == "phase_vector_smoke") {
        const rlf::experiments::PhaseVectorSmokeResult result =
            rlf::experiments::run_phase_vector_smoke({
                .seed = options.seed,
                .dimension = options.dimension,
                .samples = options.samples,
            });
        rlf::experiments::write_phase_vector_smoke_json(std::cout, result);
        write_result_transactionally(options.output_path, result);
        return 0;
    }
    if (options.experiment_name == "associative_recall") {
        const rlf::experiments::AssociativeRecallResult result =
            rlf::experiments::run_associative_recall({
                .seed = options.seed,
                .dimension = options.dimension,
                .association_count = options.association_count,
                .noise_radians = options.noise_radians,
            });
        rlf::experiments::write_associative_recall_json(
            std::cout,
            result
        );
        write_result_transactionally(options.output_path, result);
        return 0;
    }
    if (options.experiment_name == "capacity_scaling") {
        const rlf::experiments::CapacityScalingResult result =
            rlf::experiments::run_capacity_scaling({
                .seed = options.seed,
                .dimension = options.dimension,
                .evaluation_queries = options.evaluation_queries,
                .candidate_count = options.candidate_count,
                .active_count = options.active_count,
                .noise_radians = options.noise_radians,
                .mode_counts = options.mode_counts,
            });
        rlf::experiments::write_capacity_scaling_json(
            std::cout,
            result
        );
        write_result_transactionally(options.output_path, result);
        return 0;
    }
    if (options.experiment_name == "compositional_generalization") {
        const rlf::experiments::CompositionalGeneralizationResult result =
            rlf::experiments::run_compositional_generalization({
                .seed = options.seed,
                .dimension = options.dimension,
                .training_examples = options.training_examples,
                .evaluation_examples = options.evaluation_examples,
            });
        rlf::experiments::write_compositional_generalization_json(
            std::cout,
            result
        );
        write_result_transactionally(options.output_path, result);
        return 0;
    }
    if (options.experiment_name == "continual_learning") {
        const rlf::experiments::ContinualLearningResult result =
            rlf::experiments::run_continual_learning({
                .seed = options.seed,
                .dimension = options.dimension,
                .training_examples_per_task =
                    options.training_examples_per_task,
                .evaluation_examples_per_task =
                    options.evaluation_examples_per_task,
                .context_noise_radians =
                    options.context_noise_radians,
            });
        rlf::experiments::write_continual_learning_json(
            std::cout,
            result
        );
        write_result_transactionally(options.output_path, result);
        return 0;
    }
    if (options.experiment_name == "operator_composition") {
        const rlf::experiments::OperatorCompositionResult result =
            rlf::experiments::run_operator_composition({
                .seed = options.seed,
                .dimension = options.dimension,
                .context_dimensions = options.context_dimensions,
                .training_examples = options.training_examples,
                .evaluation_examples = options.evaluation_examples,
                .noise_radians = options.noise_radians,
            });
        rlf::experiments::write_operator_composition_json(
            std::cout,
            result
        );
        write_result_transactionally(options.output_path, result);
        return 0;
    }
    if (options.experiment_name == "optimization_benchmark") {
        const rlf::experiments::OptimizationBenchmarkResult result =
            rlf::experiments::run_optimization_benchmark({
                .seed = options.seed,
                .dimension = options.dimension,
                .mode_count = options.mode_count,
                .query_count = options.evaluation_queries,
                .candidate_count = options.candidate_count,
                .thread_count = options.threads,
                .similarity_iterations = options.samples,
                .quantization_samples =
                    options.quantization_samples,
            });
        rlf::experiments::write_optimization_benchmark_json(
            std::cout,
            result
        );
        write_result_transactionally(options.output_path, result);
        return 0;
    }
    if (options.experiment_name == "sequence_completion") {
        const rlf::experiments::SequenceCompletionResult result =
            rlf::experiments::run_sequence_completion({
                .seed = options.seed,
                .dimension = options.dimension,
                .symbol_count = options.symbol_count,
                .training_examples = options.training_examples,
                .evaluation_examples = options.evaluation_examples,
                .corruption_radians = options.corruption_radians,
                .dominant_probability =
                    options.dominant_probability,
            });
        rlf::experiments::write_sequence_completion_json(
            std::cout,
            result
        );
        write_result_transactionally(options.output_path, result);
        return 0;
    }
    if (options.experiment_name == "persistence_roundtrip") {
        const std::filesystem::path checkpoint_path =
            options.checkpoint_path.empty()
            ? std::filesystem::path(
                  "results/milestone4_checkpoint.rlf"
              )
            : options.checkpoint_path;
        const rlf::experiments::PersistenceRoundtripResult result =
            rlf::experiments::run_persistence_roundtrip({
                .seed = options.seed,
                .dimension = options.dimension,
                .mode_count = options.mode_count,
                .memory_records = options.memory_records,
                .checkpoint_path = checkpoint_path,
            });
        rlf::experiments::write_persistence_roundtrip_json(
            std::cout,
            result
        );
        write_result_transactionally(options.output_path, result);
        return 0;
    }
    if (options.experiment_name == "transformation_learning") {
        const rlf::experiments::TransformationLearningResult result =
            rlf::experiments::run_transformation_learning({
                .seed = options.seed,
                .dimension = options.dimension,
                .training_examples = options.training_examples,
                .evaluation_examples = options.evaluation_examples,
            });
        rlf::experiments::write_transformation_learning_json(
            std::cout,
            result
        );
        write_result_transactionally(options.output_path, result);
        return 0;
    }
    if (options.experiment_name == "structural_adaptation") {
        const rlf::experiments::StructuralAdaptationResult result =
            rlf::experiments::run_structural_adaptation({
                .seed = options.seed,
                .dimension = options.dimension,
                .training_examples = options.training_examples,
                .evaluation_examples = options.evaluation_examples,
                .context_noise_radians =
                    options.context_noise_radians,
            });
        rlf::experiments::write_structural_adaptation_json(
            std::cout,
            result
        );
        write_result_transactionally(options.output_path, result);
        return 0;
    }
    throw std::invalid_argument(
        "unsupported experiment: " + options.experiment_name
    );
}

}  // namespace

int main(const int argument_count, char** argument_values) {
    try {
        return run(argument_count, argument_values);
    } catch (const std::exception& error) {
        std::cerr << "rlf: " << error.what() << '\n';
        return 2;
    }
}
