#include "rlf/memory/associative_memory.hpp"

#include "storage/binary_codec.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace rlf::memory {
namespace {

constexpr std::array<std::uint8_t, 8> memory_magic{
    'R', 'L', 'F', 'A', 'M', '0', '0', '1'
};
constexpr std::uint32_t memory_format_version = 1U;
constexpr std::size_t header_size = 8U + 4U + 8U + 8U;

void write_phase_vector(
    storage::detail::BufferWriter& writer,
    const core::PhaseVector& value
) {
    writer.write_u64(static_cast<std::uint64_t>(value.size()));
    for (const float angle : value.angles()) {
        writer.write_float(angle);
    }
}

[[nodiscard]] core::PhaseVector read_phase_vector(
    storage::detail::BufferReader& reader,
    const std::size_t expected_dimension,
    const std::size_t maximum_dimension
) {
    const std::size_t dimension = storage::detail::checked_size(
        reader.read_u64(),
        maximum_dimension,
        "phase-vector dimension"
    );
    if (dimension != expected_dimension) {
        throw std::runtime_error(
            "serialized phase-vector dimension is incompatible"
        );
    }
    std::vector<float> angles;
    angles.reserve(dimension);
    constexpr float tau = 2.0F * std::numbers::pi_v<float>;
    for (std::size_t index = 0U; index < dimension; ++index) {
        const float angle = reader.read_float();
        if (!std::isfinite(angle) || angle < 0.0F || angle >= tau) {
            throw std::runtime_error("invalid serialized phase angle");
        }
        angles.push_back(angle);
    }
    return core::PhaseVector(std::move(angles));
}

void write_payload(
    storage::detail::BufferWriter& writer,
    const AssociativeValue& value
) {
    if (std::holds_alternative<core::PhaseVector>(value)) {
        writer.write_u8(0U);
        write_phase_vector(
            writer,
            std::get<core::PhaseVector>(value)
        );
        return;
    }
    writer.write_u8(1U);
    const BytePayload& bytes = std::get<BytePayload>(value);
    writer.write_u64(static_cast<std::uint64_t>(bytes.size()));
    writer.write_bytes(bytes);
}

[[nodiscard]] AssociativeValue read_payload(
    storage::detail::BufferReader& reader,
    const std::size_t dimension,
    const AssociativeMemoryLoadOptions& options
) {
    const std::uint8_t type = reader.read_u8();
    if (type == 0U) {
        return read_phase_vector(
            reader,
            dimension,
            options.maximum_dimension
        );
    }
    if (type == 1U) {
        const std::size_t size = storage::detail::checked_size(
            reader.read_u64(),
            options.maximum_payload_bytes,
            "associative byte payload"
        );
        return reader.read_bytes(size);
    }
    throw std::runtime_error("unsupported associative payload type");
}

[[nodiscard]] std::vector<std::uint8_t> make_file(
    const std::span<const std::uint8_t> payload
) {
    storage::detail::BufferWriter writer;
    writer.write_bytes(memory_magic);
    writer.write_u32(memory_format_version);
    writer.write_u64(static_cast<std::uint64_t>(payload.size()));
    writer.write_u64(storage::detail::checksum(payload));
    writer.write_bytes(payload);
    return writer.take();
}

[[nodiscard]] std::span<const std::uint8_t> parse_header(
    const std::span<const std::uint8_t> file_bytes
) {
    if (file_bytes.size() < header_size) {
        throw std::runtime_error("truncated associative-memory header");
    }
    storage::detail::BufferReader reader(file_bytes);
    for (const std::uint8_t expected : memory_magic) {
        if (reader.read_u8() != expected) {
            throw std::runtime_error(
                "invalid associative-memory file magic"
            );
        }
    }
    const std::uint32_t version = reader.read_u32();
    if (version != memory_format_version) {
        throw std::runtime_error(
            "unsupported associative-memory file version"
        );
    }
    const std::size_t payload_size = storage::detail::checked_size(
        reader.read_u64(),
        file_bytes.size() - header_size,
        "associative-memory payload"
    );
    const std::uint64_t expected_checksum = reader.read_u64();
    if (payload_size != reader.remaining()) {
        throw std::runtime_error(
            "associative-memory payload size mismatch"
        );
    }
    const std::span<const std::uint8_t> payload =
        file_bytes.subspan(header_size, payload_size);
    if (storage::detail::checksum(payload) != expected_checksum) {
        throw std::runtime_error(
            "associative-memory checksum mismatch"
        );
    }
    return payload;
}

}  // namespace

AssociativeMemory::AssociativeMemory(
    const std::size_t dimension,
    const std::size_t capacity
)
    : dimension_(dimension), capacity_(capacity) {
    if (dimension_ == 0U || capacity_ == 0U) {
        throw std::invalid_argument(
            "associative memory dimension and capacity must be positive"
        );
    }
    records_.reserve(capacity_);
}

AssociativeMemory AssociativeMemory::restore(
    const std::size_t dimension,
    const std::size_t capacity,
    std::vector<AssociativeRecord> records,
    const std::uint64_t next_record_id,
    const std::uint64_t sequence
) {
    AssociativeMemory memory(dimension, capacity);
    if (records.size() > capacity) {
        throw std::invalid_argument(
            "restored associative records exceed capacity"
        );
    }
    if (next_record_id == 0ULL) {
        throw std::invalid_argument(
            "restored associative next record ID must be non-zero"
        );
    }

    std::vector<std::uint64_t> ids;
    ids.reserve(records.size());
    for (const AssociativeRecord& record : records) {
        memory.validate_key_and_value(
            record.key,
            record.value,
            record.confidence
        );
        if (record.id == 0ULL || record.id >= next_record_id) {
            throw std::invalid_argument(
                "restored associative record ID is invalid"
            );
        }
        if (record.last_access_sequence > sequence) {
            throw std::invalid_argument(
                "restored access sequence exceeds memory sequence"
            );
        }
        ids.push_back(record.id);
    }
    std::sort(ids.begin(), ids.end());
    if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
        throw std::invalid_argument(
            "restored associative record IDs are not unique"
        );
    }

    memory.records_ = std::move(records);
    memory.next_record_id_ = next_record_id;
    memory.sequence_ = sequence;
    return memory;
}

std::size_t AssociativeMemory::dimension() const noexcept {
    return dimension_;
}

std::size_t AssociativeMemory::capacity() const noexcept {
    return capacity_;
}

std::size_t AssociativeMemory::size() const noexcept {
    return records_.size();
}

bool AssociativeMemory::empty() const noexcept {
    return records_.empty();
}

std::uint64_t AssociativeMemory::next_record_id() const noexcept {
    return next_record_id_;
}

std::uint64_t AssociativeMemory::sequence() const noexcept {
    return sequence_;
}

std::span<const AssociativeRecord> AssociativeMemory::records() const noexcept {
    return records_;
}

std::uint64_t AssociativeMemory::insert(
    core::PhaseVector key,
    AssociativeValue value,
    const float confidence,
    std::uint64_t timestamp
) {
    validate_key_and_value(key, value, confidence);
    if (next_record_id_ == 0ULL) {
        throw std::overflow_error(
            "associative record ID space exhausted"
        );
    }
    if (records_.size() >= capacity_) {
        evict_one();
    }
    if (timestamp == 0ULL) {
        if (sequence_ == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error(
                "associative sequence space exhausted"
            );
        }
        timestamp = ++sequence_;
    } else {
        sequence_ = std::max(sequence_, timestamp);
    }

    const std::uint64_t record_id = next_record_id_;
    if (next_record_id_ == std::numeric_limits<std::uint64_t>::max()) {
        next_record_id_ = 0ULL;
    } else {
        ++next_record_id_;
    }
    records_.push_back({
        .id = record_id,
        .key = std::move(key),
        .value = std::move(value),
        .confidence = confidence,
        .timestamp = timestamp,
        .access_count = 0ULL,
        .last_access_sequence = 0ULL,
    });
    return record_id;
}

std::uint64_t AssociativeMemory::upsert(
    core::PhaseVector key,
    AssociativeValue value,
    const float confidence,
    std::uint64_t timestamp,
    const double match_threshold
) {
    validate_key_and_value(key, value, confidence);
    if (!std::isfinite(match_threshold) ||
        match_threshold < 0.0 ||
        match_threshold > 1.0) {
        throw std::invalid_argument(
            "associative upsert threshold must be in [0, 1]"
        );
    }

    std::size_t best_index = records_.size();
    double best_similarity = -1.0;
    for (std::size_t index = 0U; index < records_.size(); ++index) {
        const double similarity = key.similarity(records_[index].key);
        if (similarity > best_similarity ||
            (similarity == best_similarity &&
             best_index != records_.size() &&
             records_[index].id < records_[best_index].id)) {
            best_similarity = similarity;
            best_index = index;
        }
    }
    if (best_index == records_.size() ||
        best_similarity < match_threshold) {
        return insert(
            std::move(key),
            std::move(value),
            confidence,
            timestamp
        );
    }

    if (timestamp == 0ULL) {
        if (sequence_ == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error(
                "associative sequence space exhausted"
            );
        }
        timestamp = ++sequence_;
    } else {
        sequence_ = std::max(sequence_, timestamp);
    }
    AssociativeRecord& record = records_[best_index];
    record.key = std::move(key);
    record.value = std::move(value);
    record.confidence = confidence;
    record.timestamp = timestamp;
    return record.id;
}

bool AssociativeMemory::erase(const std::uint64_t record_id) {
    const auto record = std::find_if(
        records_.begin(),
        records_.end(),
        [record_id](const AssociativeRecord& candidate) {
            return candidate.id == record_id;
        }
    );
    if (record == records_.end()) {
        return false;
    }
    records_.erase(record);
    return true;
}

std::vector<AssociativeMatch> AssociativeMemory::retrieve(
    const core::PhaseVector& query,
    const std::size_t count
) {
    if (query.size() != dimension_) {
        throw std::invalid_argument(
            "associative query dimension must match memory"
        );
    }
    if (count == 0U || records_.empty()) {
        return {};
    }

    std::vector<AssociativeMatch> matches;
    matches.reserve(records_.size());
    for (std::size_t index = 0U; index < records_.size(); ++index) {
        matches.push_back({
            .record_index = index,
            .record_id = records_[index].id,
            .similarity = query.similarity(records_[index].key),
            .confidence = records_[index].confidence,
        });
    }
    const auto strongest_first = [](
        const AssociativeMatch& left,
        const AssociativeMatch& right
    ) {
        if (left.similarity != right.similarity) {
            return left.similarity > right.similarity;
        }
        if (left.confidence != right.confidence) {
            return left.confidence > right.confidence;
        }
        return left.record_id < right.record_id;
    };
    const std::size_t selected_count = std::min(count, matches.size());
    std::partial_sort(
        matches.begin(),
        matches.begin() +
            static_cast<std::ptrdiff_t>(selected_count),
        matches.end(),
        strongest_first
    );
    matches.resize(selected_count);

    if (sequence_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(
            "associative sequence space exhausted"
        );
    }
    const std::uint64_t access_sequence = ++sequence_;
    for (const AssociativeMatch& match : matches) {
        AssociativeRecord& record = records_[match.record_index];
        ++record.access_count;
        record.last_access_sequence = access_sequence;
    }
    return matches;
}

std::size_t AssociativeMemory::bytes_stored() const noexcept {
    std::size_t bytes = sizeof(*this);
    for (const AssociativeRecord& record : records_) {
        bytes += sizeof(record);
        bytes += record.key.size() * sizeof(core::PhaseVector::Angle);
        if (std::holds_alternative<core::PhaseVector>(record.value)) {
            bytes +=
                std::get<core::PhaseVector>(record.value).size() *
                sizeof(core::PhaseVector::Angle);
        } else {
            bytes += std::get<BytePayload>(record.value).size();
        }
    }
    return bytes;
}

void AssociativeMemory::save(const std::filesystem::path& path) const {
    storage::detail::BufferWriter payload;
    payload.write_u64(static_cast<std::uint64_t>(dimension_));
    payload.write_u64(static_cast<std::uint64_t>(capacity_));
    payload.write_u64(next_record_id_);
    payload.write_u64(sequence_);
    payload.write_u64(static_cast<std::uint64_t>(records_.size()));
    for (const AssociativeRecord& record : records_) {
        payload.write_u64(record.id);
        write_phase_vector(payload, record.key);
        write_payload(payload, record.value);
        payload.write_float(record.confidence);
        payload.write_u64(record.timestamp);
        payload.write_u64(record.access_count);
        payload.write_u64(record.last_access_sequence);
    }
    const std::vector<std::uint8_t> file = make_file(payload.bytes());
    storage::detail::write_file_transactionally(path, file);
}

AssociativeMemory AssociativeMemory::load(
    const std::filesystem::path& path,
    const AssociativeMemoryLoadOptions& options
) {
    const std::vector<std::uint8_t> file = storage::detail::read_file(
        path,
        options.maximum_file_bytes
    );
    const std::span<const std::uint8_t> payload = parse_header(file);
    storage::detail::BufferReader reader(payload);

    const std::size_t dimension = storage::detail::checked_size(
        reader.read_u64(),
        options.maximum_dimension,
        "associative-memory dimension"
    );
    if (dimension == 0U) {
        throw std::runtime_error(
            "associative-memory dimension must be positive"
        );
    }
    const std::size_t capacity = storage::detail::checked_size(
        reader.read_u64(),
        options.maximum_capacity,
        "associative-memory capacity"
    );
    if (capacity == 0U) {
        throw std::runtime_error(
            "associative-memory capacity must be positive"
        );
    }
    const std::uint64_t next_record_id = reader.read_u64();
    const std::uint64_t sequence = reader.read_u64();
    const std::size_t record_count = storage::detail::checked_size(
        reader.read_u64(),
        std::min(options.maximum_records, capacity),
        "associative-memory record count"
    );

    std::vector<AssociativeRecord> records;
    records.reserve(record_count);
    for (std::size_t record_index = 0U;
         record_index < record_count;
         ++record_index) {
        const std::uint64_t id = reader.read_u64();
        core::PhaseVector key = read_phase_vector(
            reader,
            dimension,
            options.maximum_dimension
        );
        AssociativeValue value =
            read_payload(reader, dimension, options);
        const float confidence = reader.read_float();
        const std::uint64_t timestamp = reader.read_u64();
        const std::uint64_t access_count = reader.read_u64();
        const std::uint64_t last_access_sequence = reader.read_u64();
        records.push_back({
            .id = id,
            .key = std::move(key),
            .value = std::move(value),
            .confidence = confidence,
            .timestamp = timestamp,
            .access_count = access_count,
            .last_access_sequence = last_access_sequence,
        });
    }
    if (!reader.empty()) {
        throw std::runtime_error(
            "associative-memory file contains trailing data"
        );
    }
    return restore(
        dimension,
        capacity,
        std::move(records),
        next_record_id,
        sequence
    );
}

void AssociativeMemory::validate_key_and_value(
    const core::PhaseVector& key,
    const AssociativeValue& value,
    const float confidence
) const {
    if (key.size() != dimension_) {
        throw std::invalid_argument(
            "associative key dimension must match memory"
        );
    }
    if (!std::isfinite(confidence) ||
        confidence < 0.0F ||
        confidence > 1.0F) {
        throw std::invalid_argument(
            "associative confidence must be in [0, 1]"
        );
    }
    if (std::holds_alternative<core::PhaseVector>(value) &&
        std::get<core::PhaseVector>(value).size() != dimension_) {
        throw std::invalid_argument(
            "associative phase value dimension must match memory"
        );
    }
}

void AssociativeMemory::evict_one() {
    if (records_.empty()) {
        throw std::logic_error(
            "cannot evict from an empty associative memory"
        );
    }
    const auto lower_priority = [](
        const AssociativeRecord& left,
        const AssociativeRecord& right
    ) {
        if (left.confidence != right.confidence) {
            return left.confidence < right.confidence;
        }
        if (left.access_count != right.access_count) {
            return left.access_count < right.access_count;
        }
        if (left.last_access_sequence != right.last_access_sequence) {
            return left.last_access_sequence < right.last_access_sequence;
        }
        if (left.timestamp != right.timestamp) {
            return left.timestamp < right.timestamp;
        }
        return left.id < right.id;
    };
    const auto victim = std::min_element(
        records_.begin(),
        records_.end(),
        lower_priority
    );
    records_.erase(victim);
}

}  // namespace rlf::memory
