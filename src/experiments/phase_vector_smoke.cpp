#include "rlf/experiments/phase_vector_smoke.hpp"

#include "rlf/core/deterministic_rng.hpp"
#include "rlf/core/phase_vector.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace rlf::experiments {
namespace {

constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;

void hash_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (unsigned int byte_index = 0U; byte_index < 8U; ++byte_index) {
        hash ^= (value >> (byte_index * 8U)) & 0xFFULL;
        hash *= fnv_prime;
    }
}

void hash_phase_vector(
    std::uint64_t& hash,
    const core::PhaseVector& phase_vector
) noexcept {
    for (const float angle : phase_vector.angles()) {
        hash_u64(
            hash,
            static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(angle))
        );
    }
}

}  // namespace

PhaseVectorSmokeResult run_phase_vector_smoke(
    const PhaseVectorSmokeConfig& config
) {
    if (config.dimension == 0U) {
        throw std::invalid_argument("phase-vector smoke dimension must be positive");
    }
    if (config.samples == 0U) {
        throw std::invalid_argument("phase-vector smoke sample count must be positive");
    }

    core::DeterministicRng random_number_generator(config.seed);
    PhaseVectorSmokeResult result{
        .seed = config.seed,
        .dimension = config.dimension,
        .samples = config.samples,
        .minimum_reconstruction_similarity = 1.0,
        .minimum_serialization_similarity = 1.0,
        .maximum_reconstruction_error_radians = 0.0,
        .mean_unrelated_similarity = 0.0,
        .deterministic_run_hash = fnv_offset_basis,
    };

    double unrelated_similarity_total = 0.0;
    for (std::size_t sample_index = 0U;
         sample_index < config.samples;
         ++sample_index) {
        const core::PhaseVector source = core::PhaseVector::random(
            config.dimension,
            random_number_generator
        );
        const core::PhaseVector target = core::PhaseVector::random(
            config.dimension,
            random_number_generator
        );
        const core::PhaseVector transformation =
            core::PhaseVector::phase_difference(source, target);
        const core::PhaseVector reconstructed =
            source.composed(transformation);

        const double reconstruction_similarity =
            reconstructed.similarity(target);
        const double reconstruction_error =
            reconstructed.mean_angular_error(target);
        result.minimum_reconstruction_similarity = std::min(
            result.minimum_reconstruction_similarity,
            reconstruction_similarity
        );
        result.maximum_reconstruction_error_radians = std::max(
            result.maximum_reconstruction_error_radians,
            reconstruction_error
        );
        unrelated_similarity_total += source.similarity(target);

        std::stringstream serialized(
            std::ios::in | std::ios::out | std::ios::binary
        );
        source.serialize(serialized);
        serialized.seekg(0);
        const core::PhaseVector restored =
            core::PhaseVector::deserialize(serialized, config.dimension);
        result.minimum_serialization_similarity = std::min(
            result.minimum_serialization_similarity,
            source.similarity(restored)
        );

        hash_u64(
            result.deterministic_run_hash,
            static_cast<std::uint64_t>(sample_index)
        );
        hash_phase_vector(result.deterministic_run_hash, source);
        hash_phase_vector(result.deterministic_run_hash, target);
        hash_phase_vector(result.deterministic_run_hash, transformation);
        hash_phase_vector(result.deterministic_run_hash, reconstructed);
    }

    result.mean_unrelated_similarity =
        unrelated_similarity_total / static_cast<double>(config.samples);
    hash_u64(result.deterministic_run_hash, config.seed);
    hash_u64(
        result.deterministic_run_hash,
        static_cast<std::uint64_t>(config.dimension)
    );
    hash_u64(
        result.deterministic_run_hash,
        static_cast<std::uint64_t>(config.samples)
    );
    return result;
}

void write_phase_vector_smoke_json(
    std::ostream& output,
    const PhaseVectorSmokeResult& result
) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "{\n"
           << "  \"experiment\": \"phase_vector_smoke\",\n"
           << "  \"status\": \"observed\",\n"
           << "  \"seed\": " << result.seed << ",\n"
           << "  \"dimension\": " << result.dimension << ",\n"
           << "  \"samples\": " << result.samples << ",\n"
           << "  \"minimum_reconstruction_similarity\": "
           << result.minimum_reconstruction_similarity << ",\n"
           << "  \"minimum_serialization_similarity\": "
           << result.minimum_serialization_similarity << ",\n"
           << "  \"maximum_reconstruction_error_radians\": "
           << result.maximum_reconstruction_error_radians << ",\n"
           << "  \"mean_unrelated_similarity\": "
           << result.mean_unrelated_similarity << ",\n"
           << "  \"deterministic_run_hash\": \""
           << format_run_hash(result.deterministic_run_hash) << "\"\n"
           << "}\n";
    if (!output) {
        throw std::runtime_error("failed to write phase-vector smoke result");
    }
}

std::string format_run_hash(const std::uint64_t run_hash) {
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << run_hash;
    return output.str();
}

}  // namespace rlf::experiments
