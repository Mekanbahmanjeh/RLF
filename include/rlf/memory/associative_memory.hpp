#pragma once

#include "rlf/core/phase_vector.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <variant>
#include <vector>

namespace rlf::memory {

using BytePayload = std::vector<std::uint8_t>;
using AssociativeValue = std::variant<core::PhaseVector, BytePayload>;

struct AssociativeRecord final {
    std::uint64_t id;
    core::PhaseVector key;
    AssociativeValue value;
    float confidence;
    std::uint64_t timestamp;
    std::uint64_t access_count;
    std::uint64_t last_access_sequence;
};

struct AssociativeMatch final {
    std::size_t record_index;
    std::uint64_t record_id;
    double similarity;
    float confidence;
};

struct AssociativeMemoryLoadOptions final {
    std::size_t maximum_file_bytes{1U << 30U};
    std::size_t maximum_dimension{
        core::PhaseVector::default_max_serialized_dimension
    };
    std::size_t maximum_capacity{1'000'000U};
    std::size_t maximum_records{1'000'000U};
    std::size_t maximum_payload_bytes{1U << 28U};
};

class AssociativeMemory final {
public:
    AssociativeMemory(std::size_t dimension, std::size_t capacity);

    [[nodiscard]] static AssociativeMemory restore(
        std::size_t dimension,
        std::size_t capacity,
        std::vector<AssociativeRecord> records,
        std::uint64_t next_record_id,
        std::uint64_t sequence
    );

    [[nodiscard]] std::size_t dimension() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] std::uint64_t next_record_id() const noexcept;
    [[nodiscard]] std::uint64_t sequence() const noexcept;
    [[nodiscard]] std::span<const AssociativeRecord> records() const noexcept;

    [[nodiscard]] std::uint64_t insert(
        core::PhaseVector key,
        AssociativeValue value,
        float confidence = 1.0F,
        std::uint64_t timestamp = 0ULL
    );
    [[nodiscard]] std::uint64_t upsert(
        core::PhaseVector key,
        AssociativeValue value,
        float confidence = 1.0F,
        std::uint64_t timestamp = 0ULL,
        double match_threshold = 0.999999
    );
    [[nodiscard]] bool erase(std::uint64_t record_id);
    [[nodiscard]] std::vector<AssociativeMatch> retrieve(
        const core::PhaseVector& query,
        std::size_t count
    );

    [[nodiscard]] std::size_t bytes_stored() const noexcept;

    void save(const std::filesystem::path& path) const;
    [[nodiscard]] static AssociativeMemory load(
        const std::filesystem::path& path,
        const AssociativeMemoryLoadOptions& options = {}
    );

private:
    std::size_t dimension_;
    std::size_t capacity_;
    std::vector<AssociativeRecord> records_;
    std::uint64_t next_record_id_{1ULL};
    std::uint64_t sequence_{0ULL};

    void validate_key_and_value(
        const core::PhaseVector& key,
        const AssociativeValue& value,
        float confidence
    ) const;
    void evict_one();
};

}  // namespace rlf::memory
