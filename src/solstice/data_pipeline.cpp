#include "rlf/solstice/data_pipeline.hpp"

#include "rlf/core/sha256.hpp"
#include "rlf/solstice/vision_fabric.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <charconv>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace rlf::solstice {
namespace {

struct RecordFingerprint final {
    std::string digest;
    std::uint64_t simhash{};
    std::array<std::uint64_t, 8U> minhash{};
    DataShardSplit split{DataShardSplit::train};
    std::string location;
    bool text_like{};
    std::uint64_t perceptual_hash{};
    std::uint32_t perceptual_color_signature{};
    bool image_like{};
    std::string solution_template_digest;
    std::string source_family;
};

struct CompactExactFingerprint final {
    core::Sha256Digest digest{};
    DataShardSplit split{DataShardSplit::train};
};

struct NearTextFingerprint final {
    core::Sha256Digest digest{};
    std::uint64_t simhash{};
    std::array<std::uint64_t, 8U> minhash{};
    std::string location;
};

class AuditScratchDirectory final {
public:
    AuditScratchDirectory() {
        const auto base = std::filesystem::temp_directory_path();
        const auto stamp = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count()
        );
        for (std::uint64_t attempt = 0U; attempt < 1'024U; ++attempt) {
            path_ = base / (
                "rlf-prompt-audit-" + std::to_string(stamp) + "-" +
                std::to_string(attempt)
            );
            std::error_code error;
            if (std::filesystem::create_directory(path_, error)) {
                return;
            }
            if (error && error != std::errc::file_exists) {
                throw std::runtime_error(
                    "unable to create prompt-audit scratch directory"
                );
            }
        }
        throw std::runtime_error(
            "unable to allocate unique prompt-audit scratch directory"
        );
    }

    AuditScratchDirectory(const AuditScratchDirectory&) = delete;
    AuditScratchDirectory& operator=(const AuditScratchDirectory&) = delete;

    ~AuditScratchDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

struct AuditStageTiming final {
    double shard_checksum_seconds{};
    double record_fingerprint_seconds{};
    double media_read_seconds{};
    double media_checksum_seconds{};
    double media_decode_seconds{};
};

using AuditClock = std::chrono::steady_clock;

[[nodiscard]] double audit_elapsed_seconds(const AuditClock::time_point start) {
    return std::chrono::duration<double>(AuditClock::now() - start).count();
}

[[nodiscard]] std::string trim(const std::string_view value) {
    std::size_t begin = 0U;
    while (begin < value.size() &&
           std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
        ++begin;
    }
    std::size_t end = value.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(value[end - 1U])) != 0) {
        --end;
    }
    return std::string(value.substr(begin, end - begin));
}

[[nodiscard]] std::vector<std::string> split_tabs(const std::string_view line) {
    std::vector<std::string> fields;
    std::size_t start = 0U;
    while (start <= line.size()) {
        const std::size_t tab = line.find('\t', start);
        const std::size_t end = tab == std::string_view::npos ? line.size() : tab;
        fields.push_back(trim(line.substr(start, end - start)));
        if (tab == std::string_view::npos) break;
        start = tab + 1U;
    }
    return fields;
}

[[nodiscard]] bool parse_bool(const std::string_view value) {
    if (value == "true" || value == "1" || value == "yes") return true;
    if (value == "false" || value == "0" || value == "no") return false;
    throw std::invalid_argument("invalid ledger boolean: " + std::string(value));
}

[[nodiscard]] DataShardKind parse_kind(const std::string_view value) {
    if (value == "text") return DataShardKind::text;
    if (value == "dialogue") return DataShardKind::dialogue;
    if (value == "instruction") return DataShardKind::instruction;
    if (value == "preference") return DataShardKind::preference;
    if (value == "vision") return DataShardKind::vision;
    if (value == "video") return DataShardKind::video;
    if (value == "tools") return DataShardKind::tools;
    if (value == "facts") return DataShardKind::facts;
    if (value == "rules") return DataShardKind::rules;
    throw std::invalid_argument("unsupported data shard kind: " + std::string(value));
}

[[nodiscard]] DataShardSplit parse_split(const std::string_view value) {
    if (value == "train") return DataShardSplit::train;
    if (value == "development" || value == "dev") return DataShardSplit::development;
    if (value == "evaluation" || value == "eval") return DataShardSplit::evaluation;
    throw std::invalid_argument("unsupported data shard split: " + std::string(value));
}

[[nodiscard]] DataRecordFormat parse_format(const std::string_view value) {
    if (value == "text_lines") return DataRecordFormat::text_lines;
    if (value == "tsv") return DataRecordFormat::tsv;
    if (value == "vision_tsv") return DataRecordFormat::vision_tsv;
    if (value == "video_frames_tsv") return DataRecordFormat::video_frames_tsv;
    if (value == "binary") return DataRecordFormat::binary;
    throw std::invalid_argument("unsupported data record format: " + std::string(value));
}

[[nodiscard]] std::filesystem::path resolve_path(
    const std::filesystem::path& base,
    const std::filesystem::path& path
) {
    return path.is_absolute() ? path : base / path;
}

[[nodiscard]] std::vector<std::uint8_t> read_media_bytes(
    const std::filesystem::path& path,
    const std::uintmax_t size
) {
    const ImageLimits limits;
    if (size > limits.maximum_file_bytes ||
        size > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max()) ||
        size > static_cast<std::uintmax_t>(
            std::numeric_limits<std::streamsize>::max()
        )) {
        throw std::runtime_error("audited image exceeds configured decode byte limit");
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("unable to open audited media: " + path.string());
    if (!bytes.empty()) {
        input.read(
            reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size())
        );
    }
    if (!input) throw std::runtime_error("failed while reading audited media: " + path.string());
    return bytes;
}

struct AuditedMedia final {
    std::string sha256;
    std::vector<std::uint8_t> encoded;
    bool single_read{};
};

[[nodiscard]] AuditedMedia load_audited_media(
    const std::filesystem::path& path,
    const std::uintmax_t size,
    const bool single_read,
    AuditStageTiming& timing
) {
    if (!single_read) {
        const auto checksum_start = AuditClock::now();
        std::string sha256 = core::sha256_hex(core::sha256_file(path));
        timing.media_checksum_seconds += audit_elapsed_seconds(checksum_start);
        return {std::move(sha256), {}, false};
    }
    const auto read_start = AuditClock::now();
    std::vector<std::uint8_t> bytes = read_media_bytes(path, size);
    timing.media_read_seconds += audit_elapsed_seconds(read_start);
    const auto checksum_start = AuditClock::now();
    std::string sha256 = core::sha256_hex(core::sha256(bytes));
    timing.media_checksum_seconds += audit_elapsed_seconds(checksum_start);
    return {std::move(sha256), std::move(bytes), true};
}

[[nodiscard]] ImageData decode_audited_media(
    const AuditedMedia& media,
    const std::filesystem::path& path,
    AuditStageTiming& timing
) {
    const auto decode_start = AuditClock::now();
    ImageData image = media.single_read
        ? decode_image(media.encoded, path.extension().string())
        : load_image(path);
    timing.media_decode_seconds += audit_elapsed_seconds(decode_start);
    return image;
}

[[nodiscard]] std::string lowercase_hex(std::string value) {
    for (char& character : value) {
        character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    }
    return value;
}

[[nodiscard]] std::string normalize_text(const std::string_view value) {
    std::string normalized;
    normalized.reserve(value.size());
    bool pending_space = false;
    for (const char character : value) {
        const unsigned char byte = static_cast<unsigned char>(character);
        if (std::isspace(byte) != 0) {
            pending_space = !normalized.empty();
            continue;
        }
        if (pending_space) {
            normalized.push_back(' ');
            pending_space = false;
        }
        normalized.push_back(byte < 128U
            ? static_cast<char>(std::tolower(byte))
            : character);
    }
    return normalized;
}

[[nodiscard]] std::string normalize_solution_template(const std::string_view value) {
    const std::string normalized = normalize_text(value);
    std::string result;
    result.reserve(normalized.size());
    bool in_number = false;
    for (const char character : normalized) {
        const bool digit = std::isdigit(static_cast<unsigned char>(character)) != 0;
        if (digit) {
            if (!in_number) result += "<number>";
            in_number = true;
            continue;
        }
        in_number = false;
        result.push_back(character);
    }
    return result;
}

[[nodiscard]] std::uint64_t digest_prefix(const core::Sha256Digest& digest) noexcept {
    std::uint64_t value = 0U;
    for (std::size_t index = 0U; index < 8U; ++index) {
        value = (value << 8U) | static_cast<std::uint64_t>(digest[index]);
    }
    return value;
}

[[nodiscard]] std::uint64_t mix_hash(std::uint64_t value) noexcept {
    value += 0x9E37'79B9'7F4A'7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58'476D'1CE4'E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D0'49BB'1331'11EBULL;
    return value ^ (value >> 31U);
}

struct TextSketch final {
    std::uint64_t simhash{};
    std::array<std::uint64_t, 8U> minhash{};
};

class TextSketchCache final {
public:
    explicit TextSketchCache(const std::size_t maximum_entries)
        : maximum_entries_(maximum_entries) {
        token_hashes_.reserve(std::min<std::size_t>(maximum_entries, 65'536U));
    }

    [[nodiscard]] TextSketch sketch(const std::string_view value) {
        std::array<int, 64U> votes{};
        TextSketch result;
        result.minhash.fill(std::numeric_limits<std::uint64_t>::max());
        std::string token;
        const auto commit = [this, &votes, &result, &token]() {
            if (token.empty()) return;
            const std::uint64_t base = token_hash(token);
            for (std::size_t bit = 0U; bit < votes.size(); ++bit) {
                votes[bit] += ((base >> bit) & 1ULL) != 0U ? 1 : -1;
            }
            for (std::size_t index = 0U; index < result.minhash.size(); ++index) {
                const std::uint64_t candidate = mix_hash(
                    base ^ (0xD6E8'FEB8'6659'FD93ULL * (index + 1U))
                );
                result.minhash[index] = std::min(result.minhash[index], candidate);
            }
            token.clear();
        };
        for (const char character : value) {
            const unsigned char byte = static_cast<unsigned char>(character);
            if (std::isalnum(byte) != 0 || byte >= 128U || byte == '_') {
                token.push_back(byte < 128U
                    ? static_cast<char>(std::tolower(byte))
                    : character);
            } else {
                commit();
            }
        }
        commit();
        for (std::size_t bit = 0U; bit < votes.size(); ++bit) {
            if (votes[bit] >= 0) result.simhash |= 1ULL << bit;
        }
        return result;
    }

    [[nodiscard]] std::size_t entries() const noexcept { return token_hashes_.size(); }
    [[nodiscard]] std::uint64_t hits() const noexcept { return hits_; }
    [[nodiscard]] std::uint64_t misses() const noexcept { return misses_; }

private:
    [[nodiscard]] std::uint64_t token_hash(const std::string& token) {
        const auto found = token_hashes_.find(token);
        if (found != token_hashes_.end()) {
            ++hits_;
            return found->second;
        }
        ++misses_;
        const std::uint64_t hash = digest_prefix(core::sha256(token));
        if (token_hashes_.size() < maximum_entries_) {
            token_hashes_.emplace(token, hash);
        }
        return hash;
    }

    std::size_t maximum_entries_{};
    std::unordered_map<std::string, std::uint64_t> token_hashes_;
    std::uint64_t hits_{};
    std::uint64_t misses_{};
};

[[nodiscard]] std::uint64_t image_difference_hash(const ImageData& image) {
    if (image.width == 0U || image.height == 0U ||
        image.rgb.size() != image.width * image.height * 3U) {
        throw std::runtime_error("invalid image while computing perceptual hash");
    }
    const auto luminance = [&image](const std::size_t x, const std::size_t y) {
        const std::size_t offset = (y * image.width + x) * 3U;
        return 299U * static_cast<unsigned int>(image.rgb[offset]) +
            587U * static_cast<unsigned int>(image.rgb[offset + 1U]) +
            114U * static_cast<unsigned int>(image.rgb[offset + 2U]);
    };
    std::uint64_t hash = 0U;
    for (std::size_t row = 0U; row < 8U; ++row) {
        const std::size_t y = image.height == 1U
            ? 0U : row * (image.height - 1U) / 7U;
        for (std::size_t column = 0U; column < 8U; ++column) {
            const std::size_t left_x = image.width == 1U
                ? 0U : column * (image.width - 1U) / 8U;
            const std::size_t right_x = image.width == 1U
                ? 0U : (column + 1U) * (image.width - 1U) / 8U;
            if (luminance(left_x, y) > luminance(right_x, y)) {
                hash |= 1ULL << (row * 8U + column);
            }
        }
    }
    return hash;
}

[[nodiscard]] std::uint32_t image_color_signature(const ImageData& image) {
    if (image.width == 0U || image.height == 0U ||
        image.rgb.size() != image.width * image.height * 3U) {
        throw std::runtime_error("invalid image while computing color signature");
    }
    std::uint64_t red = 0U;
    std::uint64_t green = 0U;
    std::uint64_t blue = 0U;
    const std::size_t pixels = image.width * image.height;
    for (std::size_t index = 0U; index < pixels; ++index) {
        red += image.rgb[index * 3U];
        green += image.rgb[index * 3U + 1U];
        blue += image.rgb[index * 3U + 2U];
    }
    const auto quantized = [pixels](const std::uint64_t sum) {
        return static_cast<std::uint32_t>((sum / pixels) >> 3U);
    };
    const std::uint32_t aspect = image.width == image.height
        ? 0U : (image.width > image.height ? 1U : 2U);
    return quantized(red) | (quantized(green) << 5U) |
        (quantized(blue) << 10U) | (aspect << 15U);
}

void add_issue(
    DataAuditReport& report,
    const DataAuditOptions& options,
    std::string issue
) {
    if (report.issues.size() < options.maximum_reported_issues) {
        report.issues.push_back(std::move(issue));
    }
}

[[nodiscard]] bool provenance_complete(const DataShard& shard) noexcept {
    const bool date_shape = shard.created_utc.size() >= 10U &&
        shard.created_utc[4U] == '-' && shard.created_utc[7U] == '-';
    const bool license_known = !shard.license.empty() && shard.license != "UNKNOWN" &&
        shard.license != "unknown" && shard.license != "unlicensed";
    return !shard.shard_id.empty() && !shard.modality.empty() &&
        !shard.language.empty() && !shard.domain.empty() &&
        !shard.source_uri.empty() && shard.source_uri != "UNKNOWN" &&
        license_known && date_shape &&
        !shard.preprocessing_version.empty() && !shard.teacher.empty() &&
        !shard.evaluation_family.empty();
}

[[nodiscard]] bool format_matches_kind(const DataShard& shard) noexcept {
    if (shard.format == DataRecordFormat::binary) {
        return shard.split != DataShardSplit::train;
    }
    if (shard.kind == DataShardKind::text) {
        return shard.format == DataRecordFormat::text_lines;
    }
    if (shard.kind == DataShardKind::vision) {
        return shard.format == DataRecordFormat::vision_tsv;
    }
    if (shard.kind == DataShardKind::video) {
        return shard.format == DataRecordFormat::video_frames_tsv;
    }
    return shard.format == DataRecordFormat::tsv;
}

void add_text_record(
    std::vector<RecordFingerprint>& records,
    const DataShard& shard,
    const std::string_view record,
    const std::string& location,
    TextSketchCache& sketch_cache,
    AuditStageTiming& timing
) {
    const auto fingerprint_start = AuditClock::now();
    const std::string normalized = normalize_text(record);
    if (normalized.empty()) {
        timing.record_fingerprint_seconds += audit_elapsed_seconds(fingerprint_start);
        return;
    }
    const TextSketch sketch = sketch_cache.sketch(normalized);
    records.push_back(RecordFingerprint{
        core::sha256_hex(core::sha256(normalized)),
        sketch.simhash,
        sketch.minhash,
        shard.split,
        location,
        true,
        0U,
        0U,
        false,
        {},
        {},
    });
    records.back().solution_template_digest = core::sha256_hex(
        core::sha256(normalize_solution_template(record))
    );
    records.back().source_family = shard.source_uri;
    timing.record_fingerprint_seconds += audit_elapsed_seconds(fingerprint_start);
}

[[nodiscard]] std::uint64_t add_records(
    const DataShard& shard,
    const std::filesystem::path& path,
    const DataAuditOptions& options,
    TextSketchCache& sketch_cache,
    AuditStageTiming& timing,
    std::vector<RecordFingerprint>& records,
    std::size_t& record_count,
    DataAuditReport& report
) {
    if (shard.format == DataRecordFormat::binary) {
        const auto fingerprint_start = AuditClock::now();
        records.push_back(RecordFingerprint{
            core::sha256_hex(core::sha256_file(path)), 0U, {}, shard.split,
            shard.shard_id + ":1", false, 0U, 0U, false, {}, {},
        });
        records.back().source_family = shard.source_uri;
        timing.record_fingerprint_seconds += audit_elapsed_seconds(fingerprint_start);
        ++record_count;
        return 0U;
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("unable to open audited shard: " + path.string());
    std::string line;
    std::size_t line_number = 0U;
    std::uint64_t media_bytes = 0U;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.size() > options.maximum_record_bytes) {
            throw std::runtime_error("record exceeds audit byte limit at " + path.string() +
                                     ":" + std::to_string(line_number));
        }
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.front() == '#') continue;
        if (record_count >= options.maximum_records) {
            throw std::runtime_error("data audit record limit exceeded");
        }
        const std::string location = shard.shard_id + ":" + std::to_string(line_number);
        if (shard.format == DataRecordFormat::vision_tsv) {
            const std::vector<std::string> fields = split_tabs(line);
            if ((fields.size() != 2U && fields.size() != 3U) ||
                fields[0U].empty() || fields.back().empty()) {
                add_issue(report, options, location + ": invalid vision TSV row");
                continue;
            }
            if (options.require_media_sha256 && fields.size() != 3U) {
                add_issue(report, options, location + ": media SHA-256 is required");
                continue;
            }
            if (fields.size() == 3U && !core::is_sha256_hex(fields[1U])) {
                add_issue(report, options, location + ": invalid media SHA-256");
                continue;
            }
            const std::filesystem::path media_path = resolve_path(
                path.parent_path(), std::filesystem::path(fields[0U])
            );
            std::error_code size_error;
            const std::uintmax_t size = std::filesystem::file_size(media_path, size_error);
            if (size_error) {
                ++report.missing_files;
                add_issue(report, options, location + ": missing media file " + media_path.string());
                continue;
            }
            if (size > std::numeric_limits<std::uint64_t>::max() - media_bytes) {
                throw std::runtime_error("referenced media byte count overflow");
            }
            media_bytes += static_cast<std::uint64_t>(size);
            AuditedMedia audited_media = load_audited_media(
                media_path, size, options.single_read_media_audit, timing
            );
            const std::string& media_sha = audited_media.sha256;
            if (fields.size() == 3U && media_sha != lowercase_hex(fields[1U])) {
                ++report.checksum_mismatches;
                add_issue(report, options, location + ": image SHA-256 mismatch");
                continue;
            }
            const std::string caption = normalize_text(fields.back());
            const std::string combined = media_sha +
                "\t" + caption;
            const ImageData audited_image = decode_audited_media(
                audited_media, media_path, timing
            );
            const auto fingerprint_start = AuditClock::now();
            const TextSketch sketch = sketch_cache.sketch(caption);
            records.push_back(RecordFingerprint{
                core::sha256_hex(core::sha256(combined)),
                sketch.simhash, sketch.minhash,
                shard.split, location, true,
                image_difference_hash(audited_image),
                image_color_signature(audited_image), true,
                {}, {},
            });
            records.back().solution_template_digest = core::sha256_hex(
                core::sha256(normalize_solution_template(caption))
            );
            records.back().source_family = shard.source_uri;
            timing.record_fingerprint_seconds += audit_elapsed_seconds(fingerprint_start);
        } else if (shard.format == DataRecordFormat::video_frames_tsv) {
            const std::vector<std::string> fields = split_tabs(line);
            if (fields.size() != 7U || fields[0U].empty() || fields[3U].empty() ||
                !core::is_sha256_hex(fields[4U]) || fields[5U].empty() ||
                fields[6U].empty()) {
                add_issue(report, options, location + ": invalid video frame TSV row");
                continue;
            }
            std::uint64_t frame_index = 0U;
            double frames_per_second = 0.0;
            const auto [index_end, index_error] = std::from_chars(
                fields[1U].data(), fields[1U].data() + fields[1U].size(), frame_index
            );
            const auto [fps_end, fps_error] = std::from_chars(
                fields[2U].data(), fields[2U].data() + fields[2U].size(),
                frames_per_second
            );
            if (index_error != std::errc{} ||
                index_end != fields[1U].data() + fields[1U].size() ||
                fps_error != std::errc{} ||
                fps_end != fields[2U].data() + fields[2U].size() ||
                frames_per_second <= 0.0) {
                add_issue(report, options, location + ": invalid frame index or fps");
                continue;
            }
            const std::filesystem::path media_path = resolve_path(
                path.parent_path(), std::filesystem::path(fields[3U])
            );
            std::error_code size_error;
            const std::uintmax_t size = std::filesystem::file_size(media_path, size_error);
            if (size_error) {
                ++report.missing_files;
                add_issue(report, options, location + ": missing video frame " +
                    media_path.string());
                continue;
            }
            if (size > std::numeric_limits<std::uint64_t>::max() - media_bytes) {
                throw std::runtime_error("referenced media byte count overflow");
            }
            media_bytes += static_cast<std::uint64_t>(size);
            AuditedMedia audited_media = load_audited_media(
                media_path, size, options.single_read_media_audit, timing
            );
            const std::string& actual_media_sha = audited_media.sha256;
            if (actual_media_sha != lowercase_hex(fields[4U])) {
                ++report.checksum_mismatches;
                add_issue(report, options, location + ": video frame SHA-256 mismatch");
                continue;
            }
            const std::string prompt = normalize_text(fields[5U]);
            const std::string caption = normalize_text(fields[6U]);
            const std::string combined = fields[0U] + "\t" + fields[1U] + "\t" +
                fields[2U] + "\t" + actual_media_sha +
                "\t" + prompt + "\t" + caption;
            const ImageData audited_image = decode_audited_media(
                audited_media, media_path, timing
            );
            const auto fingerprint_start = AuditClock::now();
            const TextSketch sketch = sketch_cache.sketch(prompt + " " + caption);
            records.push_back(RecordFingerprint{
                core::sha256_hex(core::sha256(combined)),
                sketch.simhash,
                sketch.minhash,
                shard.split, location, true,
                image_difference_hash(audited_image),
                image_color_signature(audited_image), true,
                {}, {},
            });
            records.back().solution_template_digest = core::sha256_hex(
                core::sha256(normalize_solution_template(prompt + " " + caption))
            );
            records.back().source_family = shard.source_uri;
            timing.record_fingerprint_seconds += audit_elapsed_seconds(fingerprint_start);
        } else {
            add_text_record(records, shard, line, location, sketch_cache, timing);
        }
        ++record_count;
    }
    if (!input.eof()) throw std::runtime_error("failed while auditing shard: " + path.string());
    return media_bytes;
}

[[nodiscard]] std::string escape_json(const std::string_view value) {
    std::string output;
    output.reserve(value.size() + 8U);
    for (const char character : value) {
        switch (character) {
        case '\\': output += "\\\\"; break;
        case '"': output += "\\\""; break;
        case '\n': output += "\\n"; break;
        case '\r': output += "\\r"; break;
        case '\t': output += "\\t"; break;
        default: output.push_back(character); break;
        }
    }
    return output;
}

}  // namespace

std::string_view to_string(const DataShardKind value) noexcept {
    switch (value) {
    case DataShardKind::text: return "text";
    case DataShardKind::dialogue: return "dialogue";
    case DataShardKind::instruction: return "instruction";
    case DataShardKind::preference: return "preference";
    case DataShardKind::vision: return "vision";
    case DataShardKind::video: return "video";
    case DataShardKind::tools: return "tools";
    case DataShardKind::facts: return "facts";
    case DataShardKind::rules: return "rules";
    }
    return "unknown";
}

std::string_view to_string(const DataShardSplit value) noexcept {
    switch (value) {
    case DataShardSplit::train: return "train";
    case DataShardSplit::development: return "development";
    case DataShardSplit::evaluation: return "evaluation";
    }
    return "unknown";
}

std::string_view to_string(const DataRecordFormat value) noexcept {
    switch (value) {
    case DataRecordFormat::text_lines: return "text_lines";
    case DataRecordFormat::tsv: return "tsv";
    case DataRecordFormat::vision_tsv: return "vision_tsv";
    case DataRecordFormat::video_frames_tsv: return "video_frames_tsv";
    case DataRecordFormat::binary: return "binary";
    }
    return "unknown";
}

DataLedger load_data_ledger(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("unable to open data ledger: " + path.string());
    DataLedger ledger;
    ledger.source_path = path;
    ledger.sha256 = core::sha256_hex(core::sha256_file(path));
    std::set<std::string> shard_ids;
    std::string line;
    std::size_t line_number = 0U;
    while (std::getline(input, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.front() == '#') continue;
        const std::vector<std::string> fields = split_tabs(line);
        if (!fields.empty() && fields.front() == "shard_id") continue;
        if (fields.size() != 16U) {
            throw std::runtime_error("data ledger requires 16 TSV fields at " +
                                     path.string() + ":" + std::to_string(line_number));
        }
        if (!core::is_sha256_hex(fields[11U])) {
            throw std::runtime_error("invalid SHA-256 at " + path.string() + ":" +
                                     std::to_string(line_number));
        }
        if (!shard_ids.insert(fields[0U]).second) {
            throw std::runtime_error("duplicate shard_id in data ledger: " + fields[0U]);
        }
        ledger.shards.push_back(DataShard{
            fields[0U], parse_kind(fields[1U]), parse_split(fields[2U]),
            fields[3U], fields[4U], fields[5U], parse_format(fields[6U]),
            std::filesystem::path(fields[7U]), fields[8U], fields[9U],
            fields[10U], lowercase_hex(fields[11U]), fields[12U], fields[13U],
            fields[14U], parse_bool(fields[15U]), line_number,
        });
    }
    if (!input.eof()) throw std::runtime_error("failed while reading data ledger");
    if (ledger.shards.empty()) throw std::runtime_error("data ledger is empty");
    return ledger;
}

DataAuditReport audit_data_ledger(
    const DataLedger& ledger,
    const DataAuditOptions options
) {
    const auto audit_start = AuditClock::now();
    const auto seconds_since = [](const AuditClock::time_point start) {
        return std::chrono::duration<double>(AuditClock::now() - start).count();
    };
    if (options.maximum_records == 0U || options.maximum_record_bytes == 0U ||
        options.maximum_text_shard_bytes == 0U ||
        options.maximum_train_shard_bytes == 0U ||
        options.maximum_text_shard_bytes > options.maximum_train_shard_bytes ||
        options.near_duplicate_hamming_distance > 63U) {
        throw std::invalid_argument("invalid data audit options");
    }
    DataAuditReport report;
    report.ledger_sha256 = ledger.sha256;
    report.shards = ledger.shards.size();
    report.audited_shards.reserve(ledger.shards.size());
    std::vector<RecordFingerprint> records;
    records.reserve(std::min<std::size_t>(options.maximum_records, 1'000'000U));
    TextSketchCache sketch_cache(options.maximum_token_hash_cache_entries);
    AuditStageTiming stage_timing;
    const std::filesystem::path base = ledger.source_path.parent_path();
    const auto shard_scan_start = AuditClock::now();
    for (const DataShard& shard : ledger.shards) {
        AuditedShard audited;
        audited.shard_id = shard.shard_id;
        audited.approved = shard.approved;
        audited.provenance_complete = provenance_complete(shard);
        if (!audited.provenance_complete) {
            ++report.incomplete_provenance;
            add_issue(report, options, shard.shard_id + ": incomplete provenance");
        }
        if (options.require_approved && !shard.approved) {
            ++report.unapproved_shards;
            add_issue(report, options, shard.shard_id + ": shard is not approved");
        }
        if (!format_matches_kind(shard)) {
            add_issue(report, options, shard.shard_id + ": format does not match shard kind");
        }
        const std::filesystem::path shard_path = resolve_path(base, shard.path);
        std::error_code size_error;
        const std::uintmax_t file_size = std::filesystem::file_size(shard_path, size_error);
        if (size_error) {
            ++report.missing_files;
            add_issue(report, options, shard.shard_id + ": missing shard file " +
                      shard_path.string());
            report.audited_shards.push_back(std::move(audited));
            continue;
        }
        audited.file_bytes = static_cast<std::uint64_t>(file_size);
        if (audited.file_bytes > std::numeric_limits<std::uint64_t>::max() -
            report.shard_bytes) {
            throw std::runtime_error("audited shard byte count overflow");
        }
        report.shard_bytes += audited.file_bytes;
        const auto shard_checksum_start = AuditClock::now();
        audited.actual_sha256 = core::sha256_hex(core::sha256_file(shard_path));
        stage_timing.shard_checksum_seconds +=
            audit_elapsed_seconds(shard_checksum_start);
        audited.checksum_matches = audited.actual_sha256 == shard.expected_sha256;
        if (!audited.checksum_matches) {
            ++report.checksum_mismatches;
            add_issue(report, options, shard.shard_id + ": SHA-256 mismatch");
        }
        const std::size_t before = records.size();
        audited.referenced_media_bytes = add_records(
            shard, shard_path, options, sketch_cache, stage_timing, records,
            report.records, report
        );
        audited.records = records.size() - before;
        if (audited.referenced_media_bytes > std::numeric_limits<std::uint64_t>::max() -
            report.referenced_media_bytes) {
            throw std::runtime_error("audited media byte count overflow");
        }
        report.referenced_media_bytes += audited.referenced_media_bytes;
        if (shard.split == DataShardSplit::train) {
            const bool total_overflow = audited.file_bytes >
                std::numeric_limits<std::uint64_t>::max() -
                    audited.referenced_media_bytes;
            const std::uint64_t training_bytes = total_overflow
                ? std::numeric_limits<std::uint64_t>::max()
                : audited.file_bytes + audited.referenced_media_bytes;
            const bool text_oversized = shard.kind == DataShardKind::text &&
                audited.file_bytes > options.maximum_text_shard_bytes;
            if (total_overflow || text_oversized ||
                training_bytes > options.maximum_train_shard_bytes) {
                ++report.oversized_train_shards;
                add_issue(
                    report, options,
                    shard.shard_id + ": training shard exceeds the configured "
                    "checkpoint-cadence byte limit"
                );
            }
        }
        report.audited_shards.push_back(std::move(audited));
    }
    report.shard_scan_seconds = seconds_since(shard_scan_start);
    report.shard_checksum_seconds = stage_timing.shard_checksum_seconds;
    report.record_fingerprint_seconds = stage_timing.record_fingerprint_seconds;
    report.media_read_seconds = stage_timing.media_read_seconds;
    report.media_checksum_seconds = stage_timing.media_checksum_seconds;
    report.media_decode_seconds = stage_timing.media_decode_seconds;

    const auto exact_dedup_start = AuditClock::now();
    std::map<std::string, std::vector<const RecordFingerprint*>> by_digest;
    for (const RecordFingerprint& record : records) by_digest[record.digest].push_back(&record);
    for (const auto& [digest, group] : by_digest) {
        static_cast<void>(digest);
        if (group.size() < 2U) continue;
        std::set<DataShardSplit> splits;
        std::map<DataShardSplit, std::size_t> per_split;
        for (const RecordFingerprint* record : group) {
            splits.insert(record->split);
            ++per_split[record->split];
        }
        for (const auto& [split, count] : per_split) {
            static_cast<void>(split);
            if (count > 1U) ++report.within_split_duplicates;
        }
        if (splits.size() > 1U) ++report.cross_split_exact_collisions;
        if (splits.contains(DataShardSplit::train) &&
            splits.contains(DataShardSplit::evaluation)) {
            ++report.train_evaluation_exact_collisions;
            add_issue(report, options, "exact train/evaluation collision: " +
                      group.front()->location);
        }
    }
    report.exact_dedup_seconds = seconds_since(exact_dedup_start);

    const auto train_evaluation_near_start = AuditClock::now();
    std::unordered_map<std::uint32_t, std::vector<std::size_t>> evaluation_bands;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> evaluation_minhash;
    for (std::size_t index = 0U; index < records.size(); ++index) {
        const RecordFingerprint& record = records[index];
        if (!record.text_like || record.split != DataShardSplit::evaluation) continue;
        for (std::uint32_t band = 0U; band < 4U; ++band) {
            const std::uint32_t part = static_cast<std::uint32_t>(
                (record.simhash >> (band * 16U)) & 0xFFFFULL
            );
            evaluation_bands[(band << 16U) | part].push_back(index);
        }
        for (std::size_t seed = 0U; seed < record.minhash.size(); ++seed) {
            evaluation_minhash[record.minhash[seed] ^
                (0x9E37'79B9'7F4A'7C15ULL * (seed + 1U))].push_back(index);
        }
    }
    std::set<std::pair<std::size_t, std::size_t>> near_pairs;
    for (std::size_t train_index = 0U; train_index < records.size(); ++train_index) {
        const RecordFingerprint& train = records[train_index];
        if (!train.text_like || train.split != DataShardSplit::train) continue;
        std::unordered_set<std::size_t> candidates;
        for (std::uint32_t band = 0U; band < 4U; ++band) {
            const std::uint32_t part = static_cast<std::uint32_t>(
                (train.simhash >> (band * 16U)) & 0xFFFFULL
            );
            const auto posting = evaluation_bands.find((band << 16U) | part);
            if (posting == evaluation_bands.end()) continue;
            candidates.insert(posting->second.begin(), posting->second.end());
        }
        for (std::size_t seed = 0U; seed < train.minhash.size(); ++seed) {
            const auto posting = evaluation_minhash.find(train.minhash[seed] ^
                (0x9E37'79B9'7F4A'7C15ULL * (seed + 1U)));
            if (posting == evaluation_minhash.end()) continue;
            candidates.insert(posting->second.begin(), posting->second.end());
        }
        for (const std::size_t evaluation_index : candidates) {
            const RecordFingerprint& evaluation = records[evaluation_index];
            if (train.digest == evaluation.digest) continue;
            std::size_t minhash_matches = 0U;
            for (std::size_t seed = 0U; seed < train.minhash.size(); ++seed) {
                if (train.minhash[seed] == evaluation.minhash[seed]) {
                    ++minhash_matches;
                }
            }
            const bool simhash_near = std::popcount(train.simhash ^ evaluation.simhash) <=
                static_cast<int>(options.near_duplicate_hamming_distance);
            const bool minhash_near = minhash_matches >= 6U;
            if ((simhash_near || minhash_near) &&
                near_pairs.emplace(train_index, evaluation_index).second) {
                ++report.train_evaluation_near_collisions;
                add_issue(report, options, "near train/evaluation collision: " +
                          train.location + " vs " + evaluation.location);
            }
        }
    }
    report.train_evaluation_near_seconds = seconds_since(train_evaluation_near_start);

    const auto hierarchical_dedup_start = AuditClock::now();
    // Scalable within-split near-duplicate observability: retain one stable
    // representative per SimHash band and MinHash component. Each record checks
    // at most twelve representatives, so this remains O(records) in ordinary
    // operation. It is an approximate audit metric and does not delete records.
    std::map<std::pair<DataShardSplit, std::uint32_t>, std::size_t>
        within_split_band_representatives;
    std::map<std::pair<DataShardSplit, std::uint64_t>, std::size_t>
        within_split_minhash_representatives;
    for (std::size_t index = 0U; index < records.size(); ++index) {
        const RecordFingerprint& record = records[index];
        if (!record.text_like) continue;
        std::set<std::size_t> candidates;
        for (std::uint32_t band = 0U; band < 4U; ++band) {
            const std::uint32_t part = static_cast<std::uint32_t>(
                (record.simhash >> (band * 16U)) & 0xFFFFULL
            );
            const auto key = std::pair{record.split, (band << 16U) | part};
            const auto representative = within_split_band_representatives.find(key);
            if (representative != within_split_band_representatives.end()) {
                candidates.insert(representative->second);
            } else {
                within_split_band_representatives.emplace(key, index);
            }
        }
        for (std::size_t seed = 0U; seed < record.minhash.size(); ++seed) {
            const std::uint64_t component = record.minhash[seed] ^
                (0x9E37'79B9'7F4A'7C15ULL * (seed + 1U));
            const auto key = std::pair{record.split, component};
            const auto representative = within_split_minhash_representatives.find(key);
            if (representative != within_split_minhash_representatives.end()) {
                candidates.insert(representative->second);
            } else {
                within_split_minhash_representatives.emplace(key, index);
            }
        }
        bool near_duplicate = false;
        for (const std::size_t candidate_index : candidates) {
            const RecordFingerprint& candidate = records[candidate_index];
            if (candidate.digest == record.digest) continue;
            std::size_t minhash_matches = 0U;
            for (std::size_t seed = 0U; seed < record.minhash.size(); ++seed) {
                if (candidate.minhash[seed] == record.minhash[seed]) ++minhash_matches;
            }
            if (std::popcount(candidate.simhash ^ record.simhash) <=
                    static_cast<int>(options.near_duplicate_hamming_distance) ||
                minhash_matches >= 6U) {
                near_duplicate = true;
                break;
            }
        }
        if (near_duplicate) ++report.within_split_near_duplicate_records;
    }

    std::map<std::string, std::vector<const RecordFingerprint*>> by_solution_template;
    std::map<std::string, std::set<DataShardSplit>> source_family_splits;
    for (const RecordFingerprint& record : records) {
        if (!record.solution_template_digest.empty()) {
            by_solution_template[record.solution_template_digest].push_back(&record);
        }
        if (!record.source_family.empty()) {
            source_family_splits[record.source_family].insert(record.split);
        }
    }
    for (const auto& [template_digest, group] : by_solution_template) {
        static_cast<void>(template_digest);
        std::map<DataShardSplit, std::set<std::string>> digests_by_split;
        for (const RecordFingerprint* record : group) {
            digests_by_split[record->split].insert(record->digest);
        }
        for (const auto& [split, digests] : digests_by_split) {
            static_cast<void>(split);
            if (digests.size() > 1U) ++report.within_split_solution_template_groups;
        }
        if (digests_by_split.contains(DataShardSplit::train) &&
            digests_by_split.contains(DataShardSplit::evaluation)) {
            ++report.train_evaluation_solution_template_groups;
        }
    }
    for (const auto& [source_family, splits] : source_family_splits) {
        static_cast<void>(source_family);
        if (splits.size() > 1U) ++report.cross_split_source_families;
        if (splits.contains(DataShardSplit::train) &&
            splits.contains(DataShardSplit::evaluation)) {
            ++report.train_evaluation_source_families;
        }
    }
    report.hierarchical_dedup_seconds = seconds_since(hierarchical_dedup_start);
    report.token_hash_cache_entries = sketch_cache.entries();
    report.token_hash_cache_maximum_entries = options.maximum_token_hash_cache_entries;
    report.token_hash_cache_hits = sketch_cache.hits();
    report.token_hash_cache_misses = sketch_cache.misses();
    report.single_read_media_audit = options.single_read_media_audit;
    const auto perceptual_dedup_start = AuditClock::now();
    std::map<std::pair<std::uint64_t, std::uint32_t>,
             std::vector<const RecordFingerprint*>> by_perceptual_hash;
    for (const RecordFingerprint& record : records) {
        if (record.image_like) {
            by_perceptual_hash[{record.perceptual_hash,
                                record.perceptual_color_signature}].push_back(&record);
        }
    }
    for (const auto& [hash, group] : by_perceptual_hash) {
        static_cast<void>(hash);
        if (group.size() < 2U) continue;
        std::map<DataShardSplit, std::size_t> per_split;
        for (const RecordFingerprint* record : group) ++per_split[record->split];
        for (const auto& [split, count] : per_split) {
            static_cast<void>(split);
            if (count > 1U) ++report.within_split_perceptual_image_duplicates;
        }
        if (per_split.contains(DataShardSplit::train) &&
            per_split.contains(DataShardSplit::evaluation)) {
            ++report.train_evaluation_perceptual_image_collisions;
            add_issue(report, options, "perceptual image train/evaluation collision: " +
                group.front()->location);
        }
    }
    report.perceptual_dedup_seconds = seconds_since(perceptual_dedup_start);
    report.exact_deduplication_passed = report.within_split_duplicates == 0U &&
        report.cross_split_exact_collisions == 0U;
    report.contamination_audit_passed = report.train_evaluation_exact_collisions == 0U &&
        report.train_evaluation_near_collisions == 0U &&
        report.train_evaluation_perceptual_image_collisions == 0U;
    report.valid = report.missing_files == 0U && report.checksum_mismatches == 0U &&
        report.incomplete_provenance == 0U && report.unapproved_shards == 0U &&
        report.oversized_train_shards == 0U &&
        report.exact_deduplication_passed && report.contamination_audit_passed &&
        report.issues.empty();
    report.total_audit_seconds = seconds_since(audit_start);
    return report;
}

DataAuditReport audit_prompt_language_ledger_scalable(
    const DataLedger& ledger,
    DataAuditOptions options
) {
    constexpr std::size_t bucket_count = 256U;
    constexpr std::uint64_t exact_record_bytes = 33U;
    const auto audit_start = AuditClock::now();
    const auto seconds_since = [](const AuditClock::time_point start) {
        return std::chrono::duration<double>(
            AuditClock::now() - start
        ).count();
    };
    if (options.maximum_records == 0U ||
        options.maximum_record_bytes == 0U ||
        options.maximum_text_shard_bytes == 0U ||
        options.maximum_train_shard_bytes == 0U ||
        options.maximum_text_shard_bytes > options.maximum_train_shard_bytes ||
        options.near_duplicate_hamming_distance > 63U ||
        options.maximum_exact_bucket_records == 0U ||
        options.maximum_near_reference_records == 0U) {
        throw std::invalid_argument(
            "invalid scalable prompt-language audit options"
        );
    }

    DataAuditReport report;
    report.audit_strategy = "disk_partitioned_prompt_language";
    report.ledger_sha256 = ledger.sha256;
    report.shards = ledger.shards.size();
    report.audited_shards.reserve(ledger.shards.size());
    TextSketchCache sketch_cache(options.maximum_token_hash_cache_entries);
    AuditScratchDirectory scratch;
    std::array<std::ofstream, bucket_count> bucket_outputs;
    for (std::size_t bucket = 0U; bucket < bucket_count; ++bucket) {
        bucket_outputs[bucket].open(
            scratch.path() / ("exact-" + std::to_string(bucket) + ".bin"),
            std::ios::binary | std::ios::trunc
        );
        if (!bucket_outputs[bucket]) {
            throw std::runtime_error(
                "unable to create prompt-audit exact bucket"
            );
        }
    }

    std::vector<NearTextFingerprint> evaluation_records;
    evaluation_records.reserve(std::min<std::size_t>(
        options.maximum_near_reference_records, 100'000U
    ));
    std::map<std::string, std::set<DataShardSplit>> source_splits;
    const auto scan_start = AuditClock::now();
    const auto base = ledger.source_path.parent_path();
    for (const DataShard& shard : ledger.shards) {
        AuditedShard audited;
        audited.shard_id = shard.shard_id;
        audited.approved = shard.approved;
        audited.provenance_complete = provenance_complete(shard);
        if (!audited.provenance_complete) {
            ++report.incomplete_provenance;
            add_issue(
                report, options, shard.shard_id + ": incomplete provenance"
            );
        }
        if (options.require_approved && !shard.approved) {
            ++report.unapproved_shards;
            add_issue(report, options, shard.shard_id + ": shard is not approved");
        }
        if (shard.kind != DataShardKind::text ||
            shard.format != DataRecordFormat::text_lines) {
            add_issue(
                report,
                options,
                shard.shard_id +
                    ": scalable prompt audit requires text/text_lines"
            );
            report.audited_shards.push_back(std::move(audited));
            continue;
        }
        source_splits[shard.source_uri].insert(shard.split);
        const auto path = resolve_path(base, shard.path);
        std::error_code status_error;
        const auto status = std::filesystem::symlink_status(path, status_error);
        if (status_error || std::filesystem::is_symlink(status) ||
            !std::filesystem::is_regular_file(status)) {
            ++report.missing_files;
            add_issue(
                report, options,
                shard.shard_id + ": missing or unsafe shard file"
            );
            report.audited_shards.push_back(std::move(audited));
            continue;
        }
        const std::uintmax_t raw_size = std::filesystem::file_size(path);
        if (raw_size > std::numeric_limits<std::uint64_t>::max()) {
            throw std::runtime_error("prompt shard size exceeds uint64 range");
        }
        audited.file_bytes = static_cast<std::uint64_t>(raw_size);
        if (audited.file_bytes >
            std::numeric_limits<std::uint64_t>::max() - report.shard_bytes) {
            throw std::overflow_error("prompt audit byte count overflow");
        }
        report.shard_bytes += audited.file_bytes;
        const auto checksum_start = AuditClock::now();
        audited.actual_sha256 = core::sha256_hex(core::sha256_file(path));
        report.shard_checksum_seconds += seconds_since(checksum_start);
        audited.checksum_matches =
            audited.actual_sha256 == shard.expected_sha256;
        if (!audited.checksum_matches) {
            ++report.checksum_mismatches;
            add_issue(report, options, shard.shard_id + ": SHA-256 mismatch");
        }
        if (shard.split == DataShardSplit::train &&
            (audited.file_bytes > options.maximum_text_shard_bytes ||
             audited.file_bytes > options.maximum_train_shard_bytes)) {
            ++report.oversized_train_shards;
            add_issue(
                report, options,
                shard.shard_id +
                    ": training shard exceeds checkpoint cadence limit"
            );
        }

        std::ifstream input(path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("unable to open prompt-language shard");
        }
        std::string line;
        std::size_t line_number = 0U;
        while (std::getline(input, line)) {
            ++line_number;
            if (line.size() > options.maximum_record_bytes) {
                throw std::runtime_error(
                    "prompt-language record exceeds audit byte limit at " +
                    path.string() + ":" + std::to_string(line_number)
                );
            }
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty() || line.front() == '#') {
                continue;
            }
            if (report.records >= options.maximum_records) {
                throw std::runtime_error(
                    "scalable prompt audit record limit exceeded"
                );
            }
            ++report.records;
            ++audited.records;
            const auto fingerprint_start = AuditClock::now();
            const std::string normalized = normalize_text(line);
            if (normalized.empty()) {
                add_issue(
                    report,
                    options,
                    shard.shard_id + ":" + std::to_string(line_number) +
                        ": empty normalized prompt record"
                );
                report.record_fingerprint_seconds +=
                    seconds_since(fingerprint_start);
                continue;
            }
            const core::Sha256Digest digest = core::sha256(normalized);
            auto& bucket = bucket_outputs[digest.front()];
            bucket.put(static_cast<char>(shard.split));
            bucket.write(
                reinterpret_cast<const char*>(digest.data()),
                static_cast<std::streamsize>(digest.size())
            );
            if (!bucket) {
                throw std::runtime_error(
                    "failed while writing prompt-audit exact bucket"
                );
            }
            if (report.exact_dedup_scratch_bytes >
                std::numeric_limits<std::uint64_t>::max() - exact_record_bytes) {
                throw std::overflow_error(
                    "prompt-audit scratch byte count overflow"
                );
            }
            report.exact_dedup_scratch_bytes += exact_record_bytes;
            if (shard.split == DataShardSplit::evaluation) {
                if (evaluation_records.size() >=
                    options.maximum_near_reference_records) {
                    throw std::runtime_error(
                        "held-out prompt near-reference limit exceeded"
                    );
                }
                const TextSketch sketch = sketch_cache.sketch(normalized);
                evaluation_records.push_back({
                    .digest = digest,
                    .simhash = sketch.simhash,
                    .minhash = sketch.minhash,
                    .location = shard.shard_id + ":" +
                        std::to_string(line_number),
                });
            }
            report.record_fingerprint_seconds +=
                seconds_since(fingerprint_start);
        }
        if (!input.eof()) {
            throw std::runtime_error(
                "failed while reading prompt-language shard"
            );
        }
        report.audited_shards.push_back(std::move(audited));
    }
    for (auto& output : bucket_outputs) {
        output.close();
        if (!output) {
            throw std::runtime_error(
                "unable to finalize prompt-audit exact bucket"
            );
        }
    }
    report.shard_scan_seconds = seconds_since(scan_start);

    const auto exact_start = AuditClock::now();
    for (std::size_t bucket = 0U; bucket < bucket_count; ++bucket) {
        const auto path =
            scratch.path() / ("exact-" + std::to_string(bucket) + ".bin");
        const std::uintmax_t bytes = std::filesystem::file_size(path);
        if (bytes % exact_record_bytes != 0U) {
            throw std::runtime_error("corrupt prompt-audit exact bucket size");
        }
        const std::uint64_t count_u64 = bytes / exact_record_bytes;
        if (count_u64 > options.maximum_exact_bucket_records ||
            count_u64 > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max()
            )) {
            throw std::runtime_error(
                "prompt-audit exact bucket exceeds bounded memory limit"
            );
        }
        const std::size_t count = static_cast<std::size_t>(count_u64);
        report.peak_exact_bucket_records = std::max(
            report.peak_exact_bucket_records, count
        );
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("unable to read prompt-audit exact bucket");
        }
        std::vector<CompactExactFingerprint> fingerprints;
        fingerprints.reserve(count);
        for (std::size_t index = 0U; index < count; ++index) {
            const int split_byte = input.get();
            CompactExactFingerprint value;
            input.read(
                reinterpret_cast<char*>(value.digest.data()),
                static_cast<std::streamsize>(value.digest.size())
            );
            if (split_byte < 0 || !input ||
                split_byte > static_cast<int>(DataShardSplit::evaluation)) {
                throw std::runtime_error(
                    "corrupt prompt-audit exact bucket record"
                );
            }
            value.split = static_cast<DataShardSplit>(split_byte);
            fingerprints.push_back(value);
        }
        std::sort(
            fingerprints.begin(), fingerprints.end(),
            [](const auto& left, const auto& right) {
                if (left.digest != right.digest) {
                    return left.digest < right.digest;
                }
                return left.split < right.split;
            }
        );
        std::size_t begin = 0U;
        while (begin < fingerprints.size()) {
            std::size_t end = begin + 1U;
            std::array<std::size_t, 3U> per_split{};
            ++per_split[static_cast<std::size_t>(fingerprints[begin].split)];
            while (end < fingerprints.size() &&
                   fingerprints[end].digest == fingerprints[begin].digest) {
                ++per_split[static_cast<std::size_t>(fingerprints[end].split)];
                ++end;
            }
            std::size_t distinct_splits = 0U;
            for (const auto split_count : per_split) {
                if (split_count > 0U) {
                    ++distinct_splits;
                }
                if (split_count > 1U) {
                    ++report.within_split_duplicates;
                }
            }
            if (distinct_splits > 1U) {
                ++report.cross_split_exact_collisions;
            }
            if (per_split[static_cast<std::size_t>(DataShardSplit::train)] > 0U &&
                per_split[static_cast<std::size_t>(DataShardSplit::evaluation)] > 0U) {
                ++report.train_evaluation_exact_collisions;
                add_issue(
                    report,
                    options,
                    "exact prompt train/evaluation collision: " +
                        core::sha256_hex(fingerprints[begin].digest)
                );
            }
            begin = end;
        }
    }
    report.exact_dedup_seconds = seconds_since(exact_start);

    const auto near_start = AuditClock::now();
    std::unordered_map<std::uint32_t, std::vector<std::size_t>> evaluation_bands;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> evaluation_minhash;
    for (std::size_t index = 0U; index < evaluation_records.size(); ++index) {
        const auto& record = evaluation_records[index];
        for (std::uint32_t band = 0U; band < 4U; ++band) {
            const auto part = static_cast<std::uint32_t>(
                (record.simhash >> (band * 16U)) & 0xFFFFULL
            );
            evaluation_bands[(band << 16U) | part].push_back(index);
        }
        for (std::size_t seed = 0U; seed < record.minhash.size(); ++seed) {
            evaluation_minhash[
                record.minhash[seed] ^
                (0x9E37'79B9'7F4A'7C15ULL * (seed + 1U))
            ].push_back(index);
        }
    }
    for (const DataShard& shard : ledger.shards) {
        if (shard.kind != DataShardKind::text ||
            shard.format != DataRecordFormat::text_lines ||
            shard.split != DataShardSplit::train) {
            continue;
        }
        const auto path = resolve_path(base, shard.path);
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            continue;
        }
        std::string line;
        std::size_t line_number = 0U;
        while (std::getline(input, line)) {
            ++line_number;
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty() || line.front() == '#') {
                continue;
            }
            const std::string normalized = normalize_text(line);
            if (normalized.empty()) {
                continue;
            }
            const core::Sha256Digest digest = core::sha256(normalized);
            const TextSketch sketch = sketch_cache.sketch(normalized);
            std::unordered_set<std::size_t> candidates;
            for (std::uint32_t band = 0U; band < 4U; ++band) {
                const auto part = static_cast<std::uint32_t>(
                    (sketch.simhash >> (band * 16U)) & 0xFFFFULL
                );
                const auto found = evaluation_bands.find(
                    (band << 16U) | part
                );
                if (found != evaluation_bands.end()) {
                    candidates.insert(found->second.begin(), found->second.end());
                }
            }
            for (std::size_t seed = 0U; seed < sketch.minhash.size(); ++seed) {
                const auto found = evaluation_minhash.find(
                    sketch.minhash[seed] ^
                    (0x9E37'79B9'7F4A'7C15ULL * (seed + 1U))
                );
                if (found != evaluation_minhash.end()) {
                    candidates.insert(found->second.begin(), found->second.end());
                }
            }
            for (const auto candidate_index : candidates) {
                const auto& evaluation = evaluation_records[candidate_index];
                if (digest == evaluation.digest) {
                    continue;
                }
                std::size_t minhash_matches = 0U;
                for (std::size_t seed = 0U; seed < sketch.minhash.size(); ++seed) {
                    minhash_matches +=
                        sketch.minhash[seed] == evaluation.minhash[seed] ? 1U : 0U;
                }
                if (std::popcount(sketch.simhash ^ evaluation.simhash) <=
                        static_cast<int>(options.near_duplicate_hamming_distance) ||
                    minhash_matches >= 6U) {
                    ++report.train_evaluation_near_collisions;
                    add_issue(
                        report,
                        options,
                        "near prompt train/evaluation collision: " +
                            shard.shard_id + ":" +
                            std::to_string(line_number) + " vs " +
                            evaluation.location
                    );
                    break;
                }
            }
        }
    }
    report.train_evaluation_near_seconds = seconds_since(near_start);

    const auto hierarchy_start = AuditClock::now();
    for (const auto& [source, splits] : source_splits) {
        static_cast<void>(source);
        if (splits.size() > 1U) {
            ++report.cross_split_source_families;
        }
        if (splits.contains(DataShardSplit::train) &&
            splits.contains(DataShardSplit::evaluation)) {
            ++report.train_evaluation_source_families;
        }
    }
    report.hierarchical_dedup_seconds = seconds_since(hierarchy_start);
    report.token_hash_cache_entries = sketch_cache.entries();
    report.token_hash_cache_maximum_entries =
        options.maximum_token_hash_cache_entries;
    report.token_hash_cache_hits = sketch_cache.hits();
    report.token_hash_cache_misses = sketch_cache.misses();
    report.near_reference_records = evaluation_records.size();
    report.single_read_media_audit = true;
    report.exact_deduplication_passed =
        report.within_split_duplicates == 0U &&
        report.cross_split_exact_collisions == 0U;
    report.contamination_audit_passed =
        report.train_evaluation_exact_collisions == 0U &&
        report.train_evaluation_near_collisions == 0U;
    report.valid = report.missing_files == 0U &&
        report.checksum_mismatches == 0U &&
        report.incomplete_provenance == 0U &&
        report.unapproved_shards == 0U &&
        report.oversized_train_shards == 0U &&
        report.exact_deduplication_passed &&
        report.contamination_audit_passed && report.issues.empty();
    report.total_audit_seconds = seconds_since(audit_start);
    return report;
}

void write_data_audit_json(std::ostream& output, const DataAuditReport& report) {
    output << std::boolalpha
        << "{\n  \"valid\": " << report.valid
        << ",\n  \"audit_strategy\": \"" << escape_json(report.audit_strategy)
        << "\""
        << ",\n  \"ledger_sha256\": \"" << escape_json(report.ledger_sha256)
        << "\",\n  \"shards\": " << report.shards
        << ",\n  \"records\": " << report.records
        << ",\n  \"shard_bytes\": " << report.shard_bytes
        << ",\n  \"referenced_media_bytes\": " << report.referenced_media_bytes
        << ",\n  \"missing_files\": " << report.missing_files
        << ",\n  \"checksum_mismatches\": " << report.checksum_mismatches
        << ",\n  \"incomplete_provenance\": " << report.incomplete_provenance
        << ",\n  \"unapproved_shards\": " << report.unapproved_shards
        << ",\n  \"oversized_train_shards\": " << report.oversized_train_shards
        << ",\n  \"within_split_duplicates\": " << report.within_split_duplicates
        << ",\n  \"cross_split_exact_collisions\": " << report.cross_split_exact_collisions
        << ",\n  \"train_evaluation_exact_collisions\": "
        << report.train_evaluation_exact_collisions
        << ",\n  \"train_evaluation_near_collisions\": "
        << report.train_evaluation_near_collisions
        << ",\n  \"within_split_near_duplicate_records\": "
        << report.within_split_near_duplicate_records
        << ",\n  \"within_split_near_duplicate_method\": "
           "\"bounded deterministic SimHash/MinHash LSH representatives; approximate, informational, no automatic deletion\""
        << ",\n  \"within_split_solution_template_groups\": "
        << report.within_split_solution_template_groups
        << ",\n  \"train_evaluation_solution_template_groups\": "
        << report.train_evaluation_solution_template_groups
        << ",\n  \"cross_split_source_families\": "
        << report.cross_split_source_families
        << ",\n  \"train_evaluation_source_families\": "
        << report.train_evaluation_source_families
        << ",\n  \"within_split_perceptual_image_duplicates\": "
        << report.within_split_perceptual_image_duplicates
        << ",\n  \"train_evaluation_perceptual_image_collisions\": "
        << report.train_evaluation_perceptual_image_collisions
        << ",\n  \"timing_seconds\": {"
        << "\"shard_scan\": " << report.shard_scan_seconds
        << ", \"shard_checksum\": " << report.shard_checksum_seconds
        << ", \"record_fingerprint\": " << report.record_fingerprint_seconds
        << ", \"media_read\": " << report.media_read_seconds
        << ", \"media_checksum\": " << report.media_checksum_seconds
        << ", \"media_decode\": " << report.media_decode_seconds
        << ", \"exact_dedup\": " << report.exact_dedup_seconds
        << ", \"train_evaluation_near\": "
        << report.train_evaluation_near_seconds
        << ", \"hierarchical_dedup\": " << report.hierarchical_dedup_seconds
        << ", \"perceptual_dedup\": " << report.perceptual_dedup_seconds
        << ", \"total\": " << report.total_audit_seconds << "}"
        << ",\n  \"token_hash_cache\": {\"entries\": "
        << report.token_hash_cache_entries
        << ", \"maximum_entries\": " << report.token_hash_cache_maximum_entries
        << ", \"hits\": " << report.token_hash_cache_hits
        << ", \"misses\": " << report.token_hash_cache_misses
        << ", \"scope\": \"one audit invocation; bounded by audit options\"}"
        << ",\n  \"disk_partitioned_exact_dedup\": {\"scratch_bytes\": "
        << report.exact_dedup_scratch_bytes
        << ", \"peak_bucket_records\": " << report.peak_exact_bucket_records
        << ", \"near_reference_records\": " << report.near_reference_records
        << "}"
        << ",\n  \"single_read_media_audit\": " << report.single_read_media_audit
        << ",\n  \"exact_deduplication_passed\": " << report.exact_deduplication_passed
        << ",\n  \"contamination_audit_passed\": " << report.contamination_audit_passed
        << ",\n  \"audited_shards\": [\n";
    for (std::size_t index = 0U; index < report.audited_shards.size(); ++index) {
        const AuditedShard& shard = report.audited_shards[index];
        output << "    {\"shard_id\": \"" << escape_json(shard.shard_id)
            << "\", \"actual_sha256\": \"" << escape_json(shard.actual_sha256)
            << "\", \"file_bytes\": " << shard.file_bytes
            << ", \"referenced_media_bytes\": " << shard.referenced_media_bytes
            << ", \"records\": " << shard.records
            << ", \"checksum_matches\": " << shard.checksum_matches
            << ", \"provenance_complete\": " << shard.provenance_complete
            << ", \"approved\": " << shard.approved << "}"
            << (index + 1U == report.audited_shards.size() ? "\n" : ",\n");
    }
    output << "  ],\n  \"issues\": [";
    for (std::size_t index = 0U; index < report.issues.size(); ++index) {
        if (index != 0U) output << ", ";
        output << '"' << escape_json(report.issues[index]) << '"';
    }
    output << "]\n}\n";
}

}  // namespace rlf::solstice
