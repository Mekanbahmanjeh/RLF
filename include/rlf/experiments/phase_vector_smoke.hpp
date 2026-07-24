#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>

namespace rlf::experiments {

struct PhaseVectorSmokeConfig final {
    std::uint64_t seed{0x524C4630ULL};
    std::size_t dimension{1'024};
    std::size_t samples{64};
};

struct PhaseVectorSmokeResult final {
    std::uint64_t seed{};
    std::size_t dimension{};
    std::size_t samples{};
    double minimum_reconstruction_similarity{};
    double minimum_serialization_similarity{};
    double maximum_reconstruction_error_radians{};
    double mean_unrelated_similarity{};
    std::uint64_t deterministic_run_hash{};
};

[[nodiscard]] PhaseVectorSmokeResult run_phase_vector_smoke(
    const PhaseVectorSmokeConfig& config
);
void write_phase_vector_smoke_json(
    std::ostream& output,
    const PhaseVectorSmokeResult& result
);
[[nodiscard]] std::string format_run_hash(std::uint64_t run_hash);

}  // namespace rlf::experiments
