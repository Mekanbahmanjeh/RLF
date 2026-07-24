#include "rlf/solstice/abstraction_fabric.hpp"
#include "rlf/solstice/continual_learning.hpp"
#include "rlf/solstice/grounding_fabric.hpp"
#include "rlf/solstice/sparse_router.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct BenchmarkResult final {
    double compositional_accuracy{};
    double transfer_accuracy{};
    double continual_retention{};
    double grounding_accuracy{};
    double sparse_recall{};
    double operation_reduction{};
};

[[nodiscard]] std::vector<float> synthetic_vector(
    const std::size_t item,
    const std::size_t dimensions
) {
    std::vector<float> values(dimensions);
    for (std::size_t dimension = 0U; dimension < dimensions; ++dimension) {
        std::uint64_t value =
            (static_cast<std::uint64_t>(item) + 1ULL) *
            (static_cast<std::uint64_t>(dimension) + 17ULL) *
            0x9e3779b97f4a7c15ULL;
        value ^= value >> 29U;
        values[dimension] = static_cast<float>(
            static_cast<double>(value % 65'521ULL) / 32'760.5 - 1.0
        );
    }
    return values;
}

[[nodiscard]] double run_composition() {
    rlf::solstice::AbstractionFabric fabric;
    const std::array<rlf::solstice::RelationalPattern, 2U> premises{{
        {"?x", "edge", "?y"},
        {"?y", "edge", "?z"},
    }};
    fabric.learn_rule("two hop", premises, {"?x", "two_hop", "?z"});
    constexpr std::size_t cases = 128U;
    for (std::size_t index = 0U; index < cases + 1U; ++index) {
        fabric.learn_fact(
            "node_" + std::to_string(index),
            "edge",
            "node_" + std::to_string(index + 1U)
        );
    }
    std::size_t correct = 0U;
    for (std::size_t index = 0U; index < cases; ++index) {
        const auto answers = fabric.answer(
            "node_" + std::to_string(index), "two_hop"
        );
        const std::string expected = "node_" + std::to_string(index + 2U);
        if (!answers.empty() && answers.front().value == expected) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(cases);
}

[[nodiscard]] double run_transfer() {
    rlf::solstice::AbstractionFabric fabric;
    const std::array<rlf::solstice::RelationalPattern, 2U> source{{
        {"?x", "parent", "?y"},
        {"?y", "parent", "?z"},
    }};
    const std::uint64_t rule = fabric.learn_rule(
        "source", source, {"?x", "grandparent", "?z"}
    );
    constexpr std::size_t domains = 64U;
    std::size_t correct = 0U;
    for (std::size_t domain = 0U; domain < domains; ++domain) {
        const std::string link = "link_" + std::to_string(domain);
        const std::string composed = "composed_" + std::to_string(domain);
        fabric.transfer_rule(
            rule,
            "transfer_" + std::to_string(domain),
            {{"parent", link}, {"grandparent", composed}}
        );
        const std::string left = "a_" + std::to_string(domain);
        const std::string middle = "b_" + std::to_string(domain);
        const std::string right = "c_" + std::to_string(domain);
        fabric.learn_fact(left, link, middle);
        fabric.learn_fact(middle, link, right);
        const auto answers = fabric.answer(left, composed);
        if (!answers.empty() && answers.front().value == right) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(domains);
}

[[nodiscard]] double run_continual() {
    rlf::solstice::ContinualLearningConfig config;
    config.feature_dimensions = 32U;
    config.maximum_prototypes = 4'096U;
    config.replay_capacity = 4'096U;
    config.consolidation_interval = 32U;
    config.replay_batch_size = 128U;
    config.router.signature_bits = 14U;
    config.router.maximum_candidates = 128U;
    rlf::solstice::ContinualLearningFabric fabric(config);

    constexpr std::size_t old_classes = 16U;
    std::vector<std::vector<float>> old_examples;
    for (std::size_t label = 0U; label < old_classes; ++label) {
        old_examples.push_back(synthetic_vector(label, config.feature_dimensions));
        for (std::size_t repeat = 0U; repeat < 12U; ++repeat) {
            fabric.learn(
                "old",
                "old_" + std::to_string(label),
                old_examples.back()
            );
        }
    }
    for (std::size_t task = 0U; task < 8U; ++task) {
        for (std::size_t label = 0U; label < 32U; ++label) {
            const auto example = synthetic_vector(
                10'000U + task * 100U + label,
                config.feature_dimensions
            );
            for (std::size_t repeat = 0U; repeat < 4U; ++repeat) {
                fabric.learn(
                    "new_" + std::to_string(task),
                    "label_" + std::to_string(label),
                    example
                );
            }
        }
    }
    fabric.consolidate();
    std::size_t retained = 0U;
    for (std::size_t label = 0U; label < old_classes; ++label) {
        const auto prediction = fabric.predict("old", old_examples[label]);
        if (prediction.label == "old_" + std::to_string(label)) {
            ++retained;
        }
    }
    return static_cast<double>(retained) /
        static_cast<double>(old_classes);
}

[[nodiscard]] double run_grounding() {
    rlf::solstice::CrossModalGroundingFabric fabric;
    constexpr std::size_t concepts = 64U;
    for (std::size_t index = 0U; index < concepts; ++index) {
        const std::array<std::uint64_t, 2U> modes{
            static_cast<std::uint64_t>(index * 2U + 1U),
            static_cast<std::uint64_t>(index * 2U + 2U),
        };
        const std::array<std::string, 1U> positive{
            "concept_" + std::to_string(index)
        };
        const std::array<std::string, 1U> negative{
            "concept_" + std::to_string((index + 1U) % concepts)
        };
        for (std::size_t repeat = 0U; repeat < 8U; ++repeat) {
            fabric.observe(modes, positive, negative);
        }
    }
    std::size_t correct = 0U;
    for (std::size_t index = 0U; index < concepts; ++index) {
        const auto hits = fabric.modes_for_concept(
            "concept_" + std::to_string(index)
        );
        if (!hits.empty() &&
            (hits.front().visual_mode_id == index * 2U + 1U ||
             hits.front().visual_mode_id == index * 2U + 2U)) {
            ++correct;
        }
    }
    return static_cast<double>(correct) /
        static_cast<double>(concepts);
}

[[nodiscard]] std::pair<double, double> run_sparse_routing() {
    constexpr std::size_t dimensions = 64U;
    constexpr std::size_t count = 32'768U;
    std::vector<float> matrix;
    matrix.reserve(count * dimensions);
    for (std::size_t item = 0U; item < count; ++item) {
        const auto vector = synthetic_vector(item, dimensions);
        matrix.insert(matrix.end(), vector.begin(), vector.end());
    }
    rlf::solstice::SparseRouterConfig config;
    config.signature_bits = 20U;
    config.maximum_candidates = 512U;
    config.probe_radius = 2U;
    rlf::solstice::SparseRoutingIndex router(config);
    router.rebuild(matrix, count, dimensions);

    constexpr std::size_t queries = 256U;
    std::size_t recalled = 0U;
    std::uint64_t examined = 0U;
    for (std::size_t query_index = 0U; query_index < queries; ++query_index) {
        const std::size_t target = (query_index * 127U + 19U) % count;
        const auto query = std::span<const float>(
            matrix.data() + target * dimensions, dimensions
        );
        const auto route = router.route(query);
        examined += route.candidates_examined;
        if (std::find(
                route.candidate_indices.begin(),
                route.candidate_indices.end(),
                target
            ) != route.candidate_indices.end()) {
            ++recalled;
        }
    }
    const double recall = static_cast<double>(recalled) /
        static_cast<double>(queries);
    const double average_examined = static_cast<double>(examined) /
        static_cast<double>(queries);
    const double reduction = static_cast<double>(count) /
        std::max(average_examined, 1.0);
    return {recall, reduction};
}

[[nodiscard]] BenchmarkResult run_benchmark() {
    const auto sparse = run_sparse_routing();
    return {
        run_composition(),
        run_transfer(),
        run_continual(),
        run_grounding(),
        sparse.first,
        sparse.second,
    };
}

void write_json(std::ostream& output, const BenchmarkResult& result) {
    output << std::fixed << std::setprecision(6);
    output << "{\n"
           << "  \"suite\": \"rlf_frontier_mechanism_benchmark_v1\",\n"
           << "  \"claim_status\": \"experimental_not_frontier_validated\",\n"
           << "  \"compositional_accuracy\": " << result.compositional_accuracy << ",\n"
           << "  \"few_shot_transfer_accuracy\": " << result.transfer_accuracy << ",\n"
           << "  \"continual_retention\": " << result.continual_retention << ",\n"
           << "  \"multimodal_grounding_accuracy\": " << result.grounding_accuracy << ",\n"
           << "  \"sparse_candidate_recall\": " << result.sparse_recall << ",\n"
           << "  \"candidate_operation_reduction\": " << result.operation_reduction << "\n"
           << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string output_path;
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument = argv[index];
            if (argument == "--output" && index + 1 < argc) {
                output_path = argv[++index];
            } else if (argument == "--help") {
                std::cout << "Usage: rlf_frontier_benchmark [--output result.json]\n";
                return 0;
            } else {
                throw std::invalid_argument(
                    "unknown benchmark argument: " + std::string(argument)
                );
            }
        }
        const BenchmarkResult result = run_benchmark();
        write_json(std::cout, result);
        if (!output_path.empty()) {
            std::ofstream output(output_path);
            if (!output) {
                throw std::runtime_error("unable to open benchmark output");
            }
            write_json(output, result);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "frontier benchmark error: " << error.what() << '\n';
        return 1;
    }
}
