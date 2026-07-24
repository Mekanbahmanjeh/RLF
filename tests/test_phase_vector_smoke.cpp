#include "test_framework.hpp"

#include "rlf/experiments/phase_vector_smoke.hpp"

#include <sstream>
#include <string>

RLF_TEST_CASE("phase-vector smoke experiment is deterministic") {
    const rlf::experiments::PhaseVectorSmokeConfig config{
        .seed = 123'456ULL,
        .dimension = 128U,
        .samples = 8U,
    };
    const rlf::experiments::PhaseVectorSmokeResult first =
        rlf::experiments::run_phase_vector_smoke(config);
    const rlf::experiments::PhaseVectorSmokeResult second =
        rlf::experiments::run_phase_vector_smoke(config);

    RLF_CHECK(
        first.deterministic_run_hash == second.deterministic_run_hash
    );
    RLF_CHECK(
        first.minimum_reconstruction_similarity ==
        second.minimum_reconstruction_similarity
    );
    RLF_CHECK(first.minimum_reconstruction_similarity > 0.999999);
    RLF_CHECK(first.minimum_serialization_similarity == 1.0);
    RLF_CHECK(first.maximum_reconstruction_error_radians < 1.0e-6);
}

RLF_TEST_CASE("phase-vector smoke result is valid JSON-shaped output") {
    const rlf::experiments::PhaseVectorSmokeResult result =
        rlf::experiments::run_phase_vector_smoke({
            .seed = 7ULL,
            .dimension = 32U,
            .samples = 2U,
        });
    std::ostringstream output;
    rlf::experiments::write_phase_vector_smoke_json(output, result);
    const std::string json = output.str();

    RLF_CHECK(json.find("\"experiment\": \"phase_vector_smoke\"") !=
              std::string::npos);
    RLF_CHECK(json.find("\"deterministic_run_hash\": \"") !=
              std::string::npos);
}
