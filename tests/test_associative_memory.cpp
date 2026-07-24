#include "test_framework.hpp"

#include "rlf/core/phase_vector.hpp"
#include "rlf/memory/associative_memory.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <variant>
#include <vector>

namespace {

[[nodiscard]] std::filesystem::path temporary_path(
    const std::string& name
) {
    return std::filesystem::temp_directory_path() /
        ("rlf-" + name + ".bin");
}

}  // namespace

RLF_TEST_CASE("associative memory retrieves exact phase and byte records") {
    rlf::memory::AssociativeMemory memory(4U, 4U);
    const rlf::core::PhaseVector first_key({0.1F, 0.2F, 0.3F, 0.4F});
    const rlf::core::PhaseVector second_key({1.1F, 1.2F, 1.3F, 1.4F});
    const rlf::core::PhaseVector phase_value({2.1F, 2.2F, 2.3F, 2.4F});
    const std::uint64_t first_id = memory.insert(
        first_key,
        phase_value,
        0.9F
    );
    const std::uint64_t second_id = memory.insert(
        second_key,
        rlf::memory::BytePayload{1U, 2U, 3U},
        0.8F
    );

    const auto first_matches = memory.retrieve(first_key, 1U);
    const auto second_matches = memory.retrieve(second_key, 1U);
    RLF_CHECK(first_matches[0U].record_id == first_id);
    RLF_CHECK(second_matches[0U].record_id == second_id);
    RLF_CHECK(first_matches[0U].similarity > 0.999999);
    RLF_CHECK(
        std::holds_alternative<rlf::core::PhaseVector>(
            memory.records()[first_matches[0U].record_index].value
        )
    );
    RLF_CHECK(
        std::holds_alternative<rlf::memory::BytePayload>(
            memory.records()[second_matches[0U].record_index].value
        )
    );
}

RLF_TEST_CASE("associative top-K ties and eviction are deterministic") {
    rlf::memory::AssociativeMemory memory(2U, 2U);
    const rlf::core::PhaseVector key({0.0F, 0.0F});
    const std::uint64_t first_id = memory.insert(
        key,
        rlf::memory::BytePayload{1U},
        0.5F,
        1ULL
    );
    const std::uint64_t second_id = memory.insert(
        key,
        rlf::memory::BytePayload{2U},
        0.5F,
        2ULL
    );

    const auto matches = memory.retrieve(key, 2U);
    RLF_CHECK(matches[0U].record_id == first_id);
    RLF_CHECK(matches[1U].record_id == second_id);

    const std::uint64_t third_id = memory.insert(
        rlf::core::PhaseVector({1.0F, 1.0F}),
        rlf::memory::BytePayload{3U},
        0.9F,
        3ULL
    );
    RLF_CHECK(memory.size() == 2U);
    RLF_CHECK(memory.records()[0U].id == second_id);
    RLF_CHECK(memory.records()[1U].id == third_id);
}

RLF_TEST_CASE("associative upsert replaces a matching record") {
    rlf::memory::AssociativeMemory memory(2U, 2U);
    const rlf::core::PhaseVector key({0.2F, 0.4F});
    const std::uint64_t id = memory.insert(
        key,
        rlf::memory::BytePayload{1U}
    );
    const std::uint64_t updated_id = memory.upsert(
        key,
        rlf::memory::BytePayload{9U, 8U},
        0.7F
    );

    RLF_CHECK(updated_id == id);
    RLF_CHECK(memory.size() == 1U);
    const auto& payload = std::get<rlf::memory::BytePayload>(
        memory.records()[0U].value
    );
    RLF_CHECK(payload.size() == 2U);
    RLF_CHECK(payload[0U] == 9U);
}

RLF_TEST_CASE("associative memory persists mixed records") {
    const std::filesystem::path path =
        temporary_path("associative-roundtrip");
    std::filesystem::remove(path);

    rlf::memory::AssociativeMemory memory(4U, 8U);
    const rlf::core::PhaseVector key({0.1F, 0.2F, 0.3F, 0.4F});
    static_cast<void>(memory.insert(
        key,
        rlf::core::PhaseVector({1.1F, 1.2F, 1.3F, 1.4F}),
        0.9F,
        11ULL
    ));
    static_cast<void>(memory.insert(
        rlf::core::PhaseVector({2.1F, 2.2F, 2.3F, 2.4F}),
        rlf::memory::BytePayload{4U, 5U, 6U},
        0.7F,
        12ULL
    ));
    static_cast<void>(memory.retrieve(key, 1U));
    memory.save(path);

    const rlf::memory::AssociativeMemory loaded =
        rlf::memory::AssociativeMemory::load(path);
    RLF_CHECK(loaded.dimension() == 4U);
    RLF_CHECK(loaded.capacity() == 8U);
    RLF_CHECK(loaded.size() == 2U);
    RLF_CHECK(loaded.records()[0U].access_count == 1ULL);
    RLF_CHECK(loaded.records()[0U].key.similarity(key) == 1.0);
    std::filesystem::remove(path);
}

RLF_TEST_CASE("associative memory rejects corruption and truncation") {
    const std::filesystem::path path =
        temporary_path("associative-corrupt");
    const std::filesystem::path corrupt_path =
        temporary_path("associative-corrupt-mutated");
    const std::filesystem::path truncated_path =
        temporary_path("associative-truncated");

    rlf::memory::AssociativeMemory memory(2U, 2U);
    static_cast<void>(memory.insert(
        rlf::core::PhaseVector::zeros(2U),
        rlf::memory::BytePayload{1U, 2U}
    ));
    memory.save(path);

    std::ifstream input(path, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>()
    );
    input.close();
    bytes.back() = static_cast<char>(bytes.back() ^ 0x01);
    {
        std::ofstream output(
            corrupt_path,
            std::ios::binary | std::ios::trunc
        );
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    bytes.resize(bytes.size() - 1U);
    {
        std::ofstream output(
            truncated_path,
            std::ios::binary | std::ios::trunc
        );
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    RLF_CHECK_THROWS_AS(
        rlf::memory::AssociativeMemory::load(corrupt_path),
        std::runtime_error
    );
    RLF_CHECK_THROWS_AS(
        rlf::memory::AssociativeMemory::load(truncated_path),
        std::runtime_error
    );
    std::filesystem::remove(path);
    std::filesystem::remove(corrupt_path);
    std::filesystem::remove(truncated_path);
}
