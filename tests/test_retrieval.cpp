#include "test_framework.hpp"

#include "rlf/core/phase_vector.hpp"
#include "rlf/core/resonant_mode.hpp"
#include "rlf/backend/compute_backend.hpp"
#include "rlf/core/deterministic_rng.hpp"
#include "rlf/retrieval/mode_retriever.hpp"
#include "rlf/retrieval/contiguous_mode_index.hpp"

#include <cstddef>
#include <vector>

RLF_TEST_CASE("exact retrieval returns strongest enabled modes") {
    const rlf::core::PhaseVector query({0.0F, 0.0F, 0.0F, 0.0F});
    std::vector<rlf::core::ResonantMode> modes;
    modes.emplace_back(
        10ULL,
        rlf::core::PhaseVector({0.0F, 0.0F, 0.0F, 0.0F}),
        rlf::core::PhaseVector::zeros(4U)
    );
    modes.emplace_back(
        20ULL,
        rlf::core::PhaseVector({0.0F, 0.0F, 3.0F, 3.0F}),
        rlf::core::PhaseVector::zeros(4U)
    );
    modes.emplace_back(
        30ULL,
        rlf::core::PhaseVector({0.1F, 0.1F, 0.1F, 0.1F}),
        rlf::core::PhaseVector::zeros(4U)
    );
    modes[0U].enabled = false;

    const rlf::retrieval::ExactModeRetriever retriever;
    const std::vector<rlf::retrieval::RetrievedMode> result =
        retriever.retrieve(query, modes, 2U);

    RLF_CHECK(result.size() == 2U);
    RLF_CHECK(result[0U].mode_id == 30ULL);
    RLF_CHECK(result[1U].mode_id == 20ULL);
}

RLF_TEST_CASE("exact retrieval resolves score ties by stable mode ID") {
    const rlf::core::PhaseVector query({0.0F, 0.0F});
    std::vector<rlf::core::ResonantMode> modes;
    modes.emplace_back(
        9ULL,
        query,
        rlf::core::PhaseVector::zeros(2U)
    );
    modes.emplace_back(
        3ULL,
        query,
        rlf::core::PhaseVector::zeros(2U)
    );
    modes.emplace_back(
        7ULL,
        query,
        rlf::core::PhaseVector::zeros(2U)
    );

    const rlf::retrieval::ExactModeRetriever retriever;
    const std::vector<rlf::retrieval::RetrievedMode> result =
        retriever.retrieve(query, modes, 2U);

    RLF_CHECK(result.size() == 2U);
    RLF_CHECK(result[0U].mode_id == 3ULL);
    RLF_CHECK(result[1U].mode_id == 7ULL);
}

RLF_TEST_CASE("optimized and parallel retrieval match scalar top K") {
    constexpr std::size_t dimension = 64U;
    rlf::core::DeterministicRng rng(0x7E71E0A1ULL);
    std::vector<rlf::core::ResonantMode> modes;
    for (std::size_t index = 0U; index < 257U; ++index) {
        modes.emplace_back(
            static_cast<std::uint64_t>(index + 1U),
            rlf::core::PhaseVector::random(dimension, rng),
            rlf::core::PhaseVector::zeros(dimension)
        );
    }
    const rlf::core::PhaseVector query = modes[91U].context_key;
    const rlf::retrieval::ExactModeRetriever scalar;
    const rlf::retrieval::ExactModeRetriever optimized(
        rlf::backend::make_backend(
            rlf::backend::BackendKind::optimized_cpu
        )
    );
    const rlf::retrieval::ParallelExactModeRetriever parallel(
        4U,
        rlf::backend::make_backend(
            rlf::backend::BackendKind::optimized_cpu
        )
    );
    const auto scalar_result = scalar.retrieve(query, modes, 32U);
    const auto optimized_result = optimized.retrieve(query, modes, 32U);
    const auto parallel_result = parallel.retrieve(query, modes, 32U);
    const rlf::retrieval::ContiguousModeIndex contiguous_index(
        modes,
        rlf::backend::make_backend(
            rlf::backend::BackendKind::optimized_cpu
        )
    );
    const auto contiguous_result = contiguous_index.retrieve(
        query,
        32U,
        4U
    );
    RLF_CHECK(scalar_result.size() == optimized_result.size());
    RLF_CHECK(scalar_result.size() == parallel_result.size());
    RLF_CHECK(scalar_result.size() == contiguous_result.size());
    for (std::size_t index = 0U;
         index < scalar_result.size();
         ++index) {
        RLF_CHECK(
            scalar_result[index].mode_id ==
            optimized_result[index].mode_id
        );
        RLF_CHECK(
            scalar_result[index].mode_id ==
            parallel_result[index].mode_id
        );
        RLF_CHECK(
            scalar_result[index].mode_id ==
            contiguous_result[index].mode_id
        );
    }
}

RLF_TEST_CASE("optimized retrieval preserves scalar ordering at scale") {
    constexpr std::size_t dimension = 128U;
    constexpr std::size_t mode_count = 4'096U;
    constexpr std::size_t candidate_count = 256U;
    rlf::core::DeterministicRng rng(0x5CA1AB1EULL);
    std::vector<rlf::core::ResonantMode> modes;
    modes.reserve(mode_count);
    for (std::size_t index = 0U; index < mode_count; ++index) {
        modes.emplace_back(
            static_cast<std::uint64_t>(index + 1U),
            rlf::core::PhaseVector::random(dimension, rng),
            rlf::core::PhaseVector::zeros(dimension)
        );
    }
    const rlf::retrieval::ExactModeRetriever scalar;
    const rlf::retrieval::ExactModeRetriever optimized(
        rlf::backend::make_backend(
            rlf::backend::BackendKind::optimized_cpu
        )
    );
    for (const std::size_t query_index : {
             0U,
             1'023U,
             2'047U,
             4'095U,
         }) {
        const auto scalar_result = scalar.retrieve(
            modes[query_index].context_key,
            modes,
            candidate_count
        );
        const auto optimized_result = optimized.retrieve(
            modes[query_index].context_key,
            modes,
            candidate_count
        );
        RLF_CHECK(scalar_result.size() == optimized_result.size());
        for (std::size_t index = 0U;
             index < scalar_result.size();
             ++index) {
            RLF_CHECK(
                scalar_result[index].mode_id ==
                optimized_result[index].mode_id
            );
        }
    }
}
