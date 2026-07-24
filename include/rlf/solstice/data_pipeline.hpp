#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace rlf::solstice {

enum class DataShardKind : std::uint8_t {
    text,
    dialogue,
    instruction,
    preference,
    vision,
    video,
    tools,
    facts,
    rules,
};

enum class DataShardSplit : std::uint8_t { train, development, evaluation };
enum class DataRecordFormat : std::uint8_t {
    text_lines,
    tsv,
    vision_tsv,
    video_frames_tsv,
    binary,
};

struct DataShard final {
    std::string shard_id;
    DataShardKind kind{DataShardKind::text};
    DataShardSplit split{DataShardSplit::train};
    std::string modality;
    std::string language;
    std::string domain;
    DataRecordFormat format{DataRecordFormat::text_lines};
    std::filesystem::path path;
    std::string source_uri;
    std::string license;
    std::string created_utc;
    std::string expected_sha256;
    std::string preprocessing_version;
    std::string teacher;
    std::string evaluation_family;
    bool approved{};
    std::size_t ledger_line{};
};

struct DataLedger final {
    std::filesystem::path source_path;
    std::string sha256;
    std::vector<DataShard> shards;
};

struct DataAuditOptions final {
    std::size_t maximum_records{10'000'000U};
    std::size_t maximum_record_bytes{16U * 1024U * 1024U};
    std::uint64_t maximum_text_shard_bytes{2ULL * 1024ULL * 1024ULL * 1024ULL};
    std::uint64_t maximum_train_shard_bytes{4ULL * 1024ULL * 1024ULL * 1024ULL};
    unsigned int near_duplicate_hamming_distance{3U};
    std::size_t maximum_token_hash_cache_entries{};
    std::size_t maximum_reported_issues{200U};
    std::size_t maximum_exact_bucket_records{5'000'000U};
    std::size_t maximum_near_reference_records{2'000'000U};
    bool require_approved{true};
    bool require_media_sha256{};
    bool single_read_media_audit{true};
};

struct AuditedShard final {
    std::string shard_id;
    std::string actual_sha256;
    std::uint64_t file_bytes{};
    std::uint64_t referenced_media_bytes{};
    std::size_t records{};
    bool checksum_matches{};
    bool provenance_complete{};
    bool approved{};
};

struct DataAuditReport final {
    std::string audit_strategy{"in_memory_full_fingerprint"};
    std::string ledger_sha256;
    std::size_t shards{};
    std::size_t records{};
    std::uint64_t shard_bytes{};
    std::uint64_t referenced_media_bytes{};
    std::size_t missing_files{};
    std::size_t checksum_mismatches{};
    std::size_t incomplete_provenance{};
    std::size_t unapproved_shards{};
    std::size_t oversized_train_shards{};
    std::size_t within_split_duplicates{};
    std::size_t cross_split_exact_collisions{};
    std::size_t train_evaluation_exact_collisions{};
    std::size_t train_evaluation_near_collisions{};
    std::size_t within_split_near_duplicate_records{};
    std::size_t within_split_solution_template_groups{};
    std::size_t train_evaluation_solution_template_groups{};
    std::size_t cross_split_source_families{};
    std::size_t train_evaluation_source_families{};
    std::size_t within_split_perceptual_image_duplicates{};
    std::size_t train_evaluation_perceptual_image_collisions{};
    double shard_scan_seconds{};
    double shard_checksum_seconds{};
    double record_fingerprint_seconds{};
    double media_read_seconds{};
    double media_checksum_seconds{};
    double media_decode_seconds{};
    double exact_dedup_seconds{};
    double train_evaluation_near_seconds{};
    double hierarchical_dedup_seconds{};
    double perceptual_dedup_seconds{};
    double total_audit_seconds{};
    std::size_t token_hash_cache_entries{};
    std::size_t token_hash_cache_maximum_entries{};
    std::uint64_t token_hash_cache_hits{};
    std::uint64_t token_hash_cache_misses{};
    std::uint64_t exact_dedup_scratch_bytes{};
    std::size_t peak_exact_bucket_records{};
    std::size_t near_reference_records{};
    bool single_read_media_audit{true};
    bool exact_deduplication_passed{};
    bool contamination_audit_passed{};
    bool valid{};
    std::vector<AuditedShard> audited_shards;
    std::vector<std::string> issues;
};

[[nodiscard]] DataLedger load_data_ledger(const std::filesystem::path& path);
[[nodiscard]] DataAuditReport audit_data_ledger(
    const DataLedger& ledger,
    DataAuditOptions options = {}
);

// Text-only audit intended for very large prompt-language ledgers. Exact
// fingerprints are partitioned on disk, while only held-out near-duplicate
// references remain in memory. It preserves the DataAuditReport contract.
[[nodiscard]] DataAuditReport audit_prompt_language_ledger_scalable(
    const DataLedger& ledger,
    DataAuditOptions options = {}
);

[[nodiscard]] std::string_view to_string(DataShardKind value) noexcept;
[[nodiscard]] std::string_view to_string(DataShardSplit value) noexcept;
[[nodiscard]] std::string_view to_string(DataRecordFormat value) noexcept;
void write_data_audit_json(std::ostream& output, const DataAuditReport& report);

}  // namespace rlf::solstice
