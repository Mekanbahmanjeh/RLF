#include "test_framework.hpp"

#include "rlf/core/encoding.hpp"
#include "rlf/core/phase_vector.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

RLF_TEST_CASE("symbol encoding is deterministic and exactly decodable") {
    const std::vector<std::string> symbols{"alpha", "beta", "gamma"};
    const rlf::core::SymbolEncoder first(128U, 91ULL, symbols);
    const rlf::core::SymbolEncoder second(128U, 91ULL, symbols);

    for (const std::string& symbol : symbols) {
        RLF_CHECK(
            first.encode(symbol).similarity(second.encode(symbol)) ==
            1.0
        );
        const rlf::core::DecodedSymbol decoded =
            first.decode(first.encode(symbol));
        RLF_CHECK(decoded.symbol == symbol);
        RLF_CHECK(decoded.similarity == 1.0);
    }
}

RLF_TEST_CASE("bounded integer encoding round trips every claimed value") {
    const rlf::core::IntegerEncoder encoder(
        128U,
        92ULL,
        -12,
        12
    );
    for (std::int64_t value = encoder.minimum();
         value <= encoder.maximum();
         ++value) {
        RLF_CHECK(encoder.decode(encoder.encode(value)) == value);
    }
}

RLF_TEST_CASE("deterministic permutations invert exactly") {
    const rlf::core::PermutationFamily permutations(64U, 93ULL, 8U);
    const rlf::core::SymbolEncoder symbols(
        64U,
        94ULL,
        {"x"}
    );
    const rlf::core::PhaseVector original = symbols.encode("x");

    for (std::size_t index = 0U;
         index < permutations.size();
         ++index) {
        const rlf::core::PhaseVector restored =
            permutations.invert(
                permutations.apply(original, index),
                index
            );
        RLF_CHECK(restored.similarity(original) == 1.0);
        RLF_CHECK(
            restored.mean_angular_error(original) == 0.0
        );
    }
}

RLF_TEST_CASE("role value binding is invertible") {
    const rlf::core::SymbolEncoder encoder(
        128U,
        95ULL,
        {"role", "value"}
    );
    const rlf::core::PhaseVector bound = rlf::core::bind(
        encoder.encode("role"),
        encoder.encode("value")
    );
    const rlf::core::PhaseVector recovered = rlf::core::unbind(
        bound,
        encoder.encode("role")
    );

    RLF_CHECK(
        recovered.similarity(encoder.encode("value")) > 0.999999
    );
    RLF_CHECK(
        recovered.mean_angular_error(encoder.encode("value")) <
        1.0e-6
    );
}

RLF_TEST_CASE("bundled sets decode all members above unrelated symbols") {
    const rlf::core::SymbolEncoder encoder(
        512U,
        96ULL,
        {"a", "b", "c", "d"}
    );
    const std::vector<rlf::core::PhaseVector> members{
        encoder.encode("a"),
        encoder.encode("b"),
        encoder.encode("c"),
    };
    const rlf::core::PhaseVector bundled = rlf::core::bundle(members);

    const double unrelated = bundled.similarity(encoder.encode("d"));
    RLF_CHECK(bundled.similarity(encoder.encode("a")) > unrelated);
    RLF_CHECK(bundled.similarity(encoder.encode("b")) > unrelated);
    RLF_CHECK(bundled.similarity(encoder.encode("c")) > unrelated);
}

RLF_TEST_CASE("ordered sequence preserves position information") {
    const rlf::core::SymbolEncoder encoder(
        1'024U,
        97ULL,
        {"a", "b", "c"}
    );
    const rlf::core::PermutationFamily positions(
        1'024U,
        98ULL,
        3U
    );
    const std::vector<rlf::core::PhaseVector> elements{
        encoder.encode("a"),
        encoder.encode("b"),
        encoder.encode("c"),
    };
    const rlf::core::PhaseVector sequence =
        rlf::core::encode_ordered_sequence(elements, positions);

    for (std::size_t position = 0U;
         position < elements.size();
         ++position) {
        const rlf::core::PhaseVector recovered =
            rlf::core::recover_sequence_element(
                sequence,
                positions,
                position
            );
        const rlf::core::DecodedSymbol decoded =
            encoder.decode(recovered);
        RLF_CHECK(decoded.symbol == encoder.symbols()[position]);
    }
}
