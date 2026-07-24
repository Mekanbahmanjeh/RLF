#include "rlf/storage/rlf7_checkpoint.hpp"

#include "binary_codec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rlf::storage {
namespace {

constexpr std::array<char, 8U> rlf7_magic{'R','L','F','7','C','K','P','9'};
constexpr std::array<char, 8U> frontier_magic{'R','L','F','F','R','T','1','0'};
constexpr std::uint32_t rlf7_version = 9U;
constexpr std::uint32_t frontier_version = 10U;

[[nodiscard]] bool valid_backend_name(const std::string_view backend) noexcept {
    return backend == "cpu_reference" || backend == "scalar_cpu" ||
        backend == "optimized_cpu" || backend == "cuda" || backend == "auto";
}

void write_record(detail::BufferWriter& writer, const frontier::KnowledgeRecord& record) {
    writer.write_u64(record.stable_id);
    writer.write_u8(static_cast<std::uint8_t>(record.kind));
    writer.write_string(record.subject);
    writer.write_string(record.predicate);
    writer.write_string(record.object);
    writer.write_string(record.source);
    writer.write_double(record.confidence);
    writer.write_double(record.utility);
    writer.write_u64(record.creation_step);
    writer.write_u64(record.last_used_step);
    writer.write_u64(record.valid_from);
    writer.write_u64(record.valid_until);
    writer.write_u64(record.version);
    writer.write_u64(static_cast<std::uint64_t>(record.use_count));
    writer.write_u64(static_cast<std::uint64_t>(record.contradiction_count));
    writer.write_bool(record.verified);
    writer.write_bool(record.stale);
    writer.write_bool(record.invalidated);
    writer.write_u8(static_cast<std::uint8_t>(record.tier));
}

frontier::KnowledgeRecord read_record(
    detail::BufferReader& reader,
    const FrontierCheckpointLoadOptions& options
) {
    frontier::KnowledgeRecord record;
    record.stable_id = reader.read_u64();
    const std::uint8_t kind = reader.read_u8();
    if (kind > static_cast<std::uint8_t>(frontier::KnowledgeKind::concept_record)) {
        throw std::runtime_error("invalid checkpoint knowledge kind");
    }
    record.kind = static_cast<frontier::KnowledgeKind>(kind);
    record.subject = reader.read_string(options.maximum_string_bytes);
    record.predicate = reader.read_string(options.maximum_string_bytes);
    record.object = reader.read_string(options.maximum_string_bytes);
    record.source = reader.read_string(options.maximum_string_bytes);
    record.confidence = reader.read_double();
    record.utility = reader.read_double();
    record.creation_step = reader.read_u64();
    record.last_used_step = reader.read_u64();
    record.valid_from = reader.read_u64();
    record.valid_until = reader.read_u64();
    record.version = reader.read_u64();
    record.use_count = detail::checked_size(reader.read_u64(), options.maximum_records, "record use count");
    record.contradiction_count = detail::checked_size(reader.read_u64(), options.maximum_records, "record contradiction count");
    record.verified = reader.read_bool();
    record.stale = reader.read_bool();
    record.invalidated = reader.read_bool();
    const std::uint8_t tier = reader.read_u8();
    if (tier > static_cast<std::uint8_t>(frontier::MemoryTier::cold)) {
        throw std::runtime_error("invalid checkpoint memory tier");
    }
    record.tier = static_cast<frontier::MemoryTier>(tier);
    return record;
}

void write_mode(detail::BufferWriter& writer, const frontier::ModeRecord& mode) {
    writer.write_u64(mode.stable_id);
    writer.write_u8(static_cast<std::uint8_t>(mode.modality));
    writer.write_string(mode.label);
    writer.write_u64(static_cast<std::uint64_t>(mode.prototype.size()));
    for (const float value : mode.prototype) writer.write_float(value);
    writer.write_bool(mode.parent_id.has_value());
    if (mode.parent_id.has_value()) writer.write_u64(*mode.parent_id);
    writer.write_u64(static_cast<std::uint64_t>(mode.child_ids.size()));
    for (const auto value : mode.child_ids) writer.write_u64(value);
    writer.write_u64(static_cast<std::uint64_t>(mode.linked_modes.size()));
    for (const auto value : mode.linked_modes) writer.write_u64(value);
    writer.write_double(mode.confidence);
    writer.write_double(mode.utility);
    writer.write_u64(static_cast<std::uint64_t>(mode.support));
    writer.write_u64(mode.creation_step);
    writer.write_u64(mode.last_used_step);
    writer.write_bool(mode.enabled);
}

frontier::ModeRecord read_mode(
    detail::BufferReader& reader,
    const FrontierCheckpointLoadOptions& options
) {
    frontier::ModeRecord mode;
    mode.stable_id = reader.read_u64();
    const std::uint8_t modality = reader.read_u8();
    if (modality > static_cast<std::uint8_t>(frontier::Modality::action)) {
        throw std::runtime_error("invalid checkpoint modality");
    }
    mode.modality = static_cast<frontier::Modality>(modality);
    mode.label = reader.read_string(options.maximum_string_bytes);
    const std::size_t prototype_size = detail::checked_size(
        reader.read_u64(), options.maximum_prototype_values, "mode prototype size"
    );
    mode.prototype.reserve(prototype_size);
    for (std::size_t index = 0U; index < prototype_size; ++index) {
        const float value = reader.read_float();
        if (!std::isfinite(static_cast<double>(value))) {
            throw std::runtime_error("checkpoint mode prototype contains non-finite value");
        }
        mode.prototype.push_back(value);
    }
    if (reader.read_bool()) mode.parent_id = reader.read_u64();
    const std::size_t child_count = detail::checked_size(
        reader.read_u64(), options.maximum_modes, "mode child count"
    );
    for (std::size_t index = 0U; index < child_count; ++index) {
        if (!mode.child_ids.insert(reader.read_u64()).second) {
            throw std::runtime_error("duplicate mode child ID");
        }
    }
    const std::size_t link_count = detail::checked_size(
        reader.read_u64(), options.maximum_modes, "mode link count"
    );
    for (std::size_t index = 0U; index < link_count; ++index) {
        if (!mode.linked_modes.insert(reader.read_u64()).second) {
            throw std::runtime_error("duplicate linked mode ID");
        }
    }
    mode.confidence = reader.read_double();
    mode.utility = reader.read_double();
    mode.support = detail::checked_size(reader.read_u64(), options.maximum_records, "mode support");
    mode.creation_step = reader.read_u64();
    mode.last_used_step = reader.read_u64();
    mode.enabled = reader.read_bool();
    return mode;
}

std::vector<std::uint8_t> serialize(
    const frontier::FrontierModel& model,
    const bool frontier_format,
    const std::string_view backend
) {
    detail::BufferWriter payload;
    payload.write_u32(frontier_format ? frontier_version : rlf7_version);
    payload.write_u64(model.seed);
    payload.write_u64(model.training_step);
    payload.write_u64(model.training_examples);
    payload.write_u64(model.evaluation_examples);
    payload.write_u64(model.media_bytes_read);
    payload.write_string(backend);
    payload.write_u64(static_cast<std::uint64_t>(model.fabric.records().size()));
    for (const auto& [id, record] : model.fabric.records()) {
        static_cast<void>(id);
        write_record(payload, record);
    }
    payload.write_u64(static_cast<std::uint64_t>(model.fabric.modes().size()));
    for (const auto& [id, mode] : model.fabric.modes()) {
        static_cast<void>(id);
        write_mode(payload, mode);
    }
    const std::vector<std::uint8_t> payload_bytes = payload.take();
    detail::BufferWriter file;
    const auto& magic = frontier_format ? frontier_magic : rlf7_magic;
    file.write_bytes(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(magic.data()), magic.size()
    ));
    file.write_u64(static_cast<std::uint64_t>(payload_bytes.size()));
    file.write_u64(detail::checksum(payload_bytes));
    file.write_bytes(payload_bytes);
    return file.take();
}

struct ParsedCheckpoint final {
    frontier::FrontierModel model;
    FrontierCheckpointSummary summary;
};

ParsedCheckpoint parse(
    const std::filesystem::path& path,
    const FrontierCheckpointLoadOptions& options
) {
    const auto bytes = detail::read_file(path, options.maximum_file_bytes);
    if (bytes.size() < 24U) throw std::runtime_error("checkpoint is truncated");
    const bool is_rlf7 = std::memcmp(bytes.data(), rlf7_magic.data(), rlf7_magic.size()) == 0;
    const bool is_frontier = std::memcmp(bytes.data(), frontier_magic.data(), frontier_magic.size()) == 0;
    if (!is_rlf7 && !is_frontier) throw std::runtime_error("unsupported RLF-7/Frontier checkpoint magic");
    detail::BufferReader file_reader(bytes);
    static_cast<void>(file_reader.read_bytes(8U));
    const std::size_t payload_size = detail::checked_size(
        file_reader.read_u64(), options.maximum_file_bytes, "checkpoint payload size"
    );
    const std::uint64_t expected_checksum = file_reader.read_u64();
    if (payload_size != file_reader.remaining()) throw std::runtime_error("checkpoint payload size mismatch");
    const auto payload = file_reader.read_bytes(payload_size);
    if (detail::checksum(payload) != expected_checksum) throw std::runtime_error("checkpoint checksum mismatch");
    detail::BufferReader reader(payload);
    const std::uint32_t version = reader.read_u32();
    if ((is_rlf7 && version != rlf7_version) || (is_frontier && version != frontier_version)) {
        throw std::runtime_error("unsupported checkpoint format version");
    }
    const std::uint64_t seed = reader.read_u64();
    frontier::FrontierModel model(seed);
    model.training_step = reader.read_u64();
    model.training_examples = reader.read_u64();
    model.evaluation_examples = reader.read_u64();
    model.media_bytes_read = reader.read_u64();
    const std::string backend = reader.read_string(options.maximum_string_bytes);
    if (!valid_backend_name(backend)) {
        throw std::runtime_error("checkpoint contains an invalid backend identifier");
    }
    model.fabric.set_step(model.training_step);
    const std::size_t record_count = detail::checked_size(
        reader.read_u64(), options.maximum_records, "knowledge record count"
    );
    model.fabric.reserve_records(record_count, 3U);
    std::set<std::uint64_t> record_ids;
    for (std::size_t index = 0U; index < record_count; ++index) {
        auto record = read_record(reader, options);
        if (record.stable_id == 0U || !record_ids.insert(record.stable_id).second) {
            throw std::runtime_error("invalid or duplicate checkpoint record ID");
        }
        model.fabric.import_record(std::move(record));
    }
    const std::size_t mode_count = detail::checked_size(
        reader.read_u64(), options.maximum_modes, "mode count"
    );
    std::vector<frontier::ModeRecord> pending;
    pending.reserve(mode_count);
    std::set<std::uint64_t> mode_ids;
    for (std::size_t index = 0U; index < mode_count; ++index) {
        auto mode = read_mode(reader, options);
        if (mode.stable_id == 0U || !mode_ids.insert(mode.stable_id).second) {
            throw std::runtime_error("invalid or duplicate checkpoint mode ID");
        }
        pending.push_back(std::move(mode));
    }
    if (!reader.empty()) throw std::runtime_error("checkpoint has trailing payload bytes");
    std::size_t imported = 0U;
    while (!pending.empty()) {
        bool progressed = false;
        for (auto iterator = pending.begin(); iterator != pending.end();) {
            if (!iterator->parent_id.has_value() || model.fabric.find_mode(*iterator->parent_id) != nullptr) {
                model.fabric.import_mode(std::move(*iterator));
                iterator = pending.erase(iterator);
                ++imported;
                progressed = true;
            } else {
                ++iterator;
            }
        }
        if (!progressed) throw std::runtime_error("checkpoint mode graph is cyclic or references a missing parent");
    }
    for (const auto& [id, mode] : model.fabric.modes()) {
        for (const auto linked : mode.linked_modes) {
            if (linked == id || model.fabric.find_mode(linked) == nullptr) {
                throw std::runtime_error("checkpoint contains invalid mode link");
            }
        }
    }
    if (imported != mode_count) throw std::runtime_error("checkpoint mode import failed");
    FrontierCheckpointSummary summary;
    summary.architecture = is_frontier ? "RLF-Frontier" : "RLF-7";
    summary.format_version = version;
    summary.seed = seed;
    summary.training_step = model.training_step;
    summary.training_examples = model.training_examples;
    summary.evaluation_examples = model.evaluation_examples;
    summary.knowledge_records = record_count;
    summary.modes = mode_count;
    summary.deterministic_hash = model.fabric.deterministic_hash();
    summary.payload_checksum = expected_checksum;
    summary.file_bytes = bytes.size();
    summary.backend = backend;
    return {.model = std::move(model), .summary = std::move(summary)};
}

}  // namespace

void save_rlf7_checkpoint(
    const std::filesystem::path& path,
    const frontier::FrontierModel& model
) {
    detail::write_file_transactionally(path, serialize(model, false, "cpu_reference"));
}

frontier::FrontierModel load_rlf7_checkpoint(
    const std::filesystem::path& path,
    const FrontierCheckpointLoadOptions& options
) {
    ParsedCheckpoint parsed = parse(path, options);
    if (parsed.summary.architecture != "RLF-7") throw std::runtime_error("checkpoint is not RLF7CKP9");
    return std::move(parsed.model);
}

void save_frontier_checkpoint(
    const std::filesystem::path& path,
    const frontier::FrontierModel& model,
    std::string backend
) {
    if (!valid_backend_name(backend) || backend == "cpu_reference") {
        throw std::invalid_argument("invalid RLF-Frontier checkpoint backend identifier");
    }
    detail::write_file_transactionally(path, serialize(model, true, backend));
}

frontier::FrontierModel load_frontier_checkpoint(
    const std::filesystem::path& path,
    const FrontierCheckpointLoadOptions& options
) {
    ParsedCheckpoint parsed = parse(path, options);
    if (parsed.summary.architecture != "RLF-Frontier") throw std::runtime_error("checkpoint is not RLFFRT10");
    return std::move(parsed.model);
}

FrontierCheckpointSummary inspect_frontier_checkpoint(
    const std::filesystem::path& path,
    const FrontierCheckpointLoadOptions& options
) {
    return parse(path, options).summary;
}

}  // namespace rlf::storage
