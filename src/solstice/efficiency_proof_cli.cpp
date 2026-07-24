#include "rlf/solstice/efficiency_proof.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

int main(int argc, char** argv) {
    try {
        rlf::solstice::EfficiencyProofConfig config;
        std::string output_path;
        bool quick = false;
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument = argv[index];
            if (argument == "--output" && index + 1 < argc) {
                output_path = argv[++index];
            } else if (argument == "--quick") {
                quick = true;
            } else if (argument == "--help") {
                std::cout
                    << "Usage: rlf_efficiency_proof [--quick] [--output report.json]\n"
                    << "  full: 10,000 schema instances and 524,288 routing vectors\n"
                    << "  quick: 128 schema instances and 8,192 routing vectors\n";
                return 0;
            } else {
                throw std::invalid_argument(
                    "unknown proof argument: " + std::string(argument)
                );
            }
        }
        if (quick) {
            config.schema_instances = 129U;
            config.routing_vectors = 8'192U;
            config.routing_queries = 32U;
            config.routing_trials = 2U;
            config.continual_classes = 8U;
            config.continual_tasks = 4U;
            config.target_efficiency_ratio = 100.0;
            config.minimum_accuracy = 0.99;
        }
        const auto report = rlf::solstice::run_efficiency_proofs(config);
        rlf::solstice::write_efficiency_proof_json(std::cout, report);
        if (!output_path.empty()) {
            std::ofstream output(output_path);
            if (!output) {
                throw std::runtime_error("unable to open proof output path");
            }
            rlf::solstice::write_efficiency_proof_json(output, report);
        }
        return report.all_internal_proofs_passed ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "efficiency proof error: " << error.what() << '\n';
        return 1;
    }
}
