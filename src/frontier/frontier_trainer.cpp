#include "rlf/frontier/frontier_trainer.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace rlf::frontier {
namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

void hash_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (unsigned int index = 0U; index < 8U; ++index) {
        hash ^= (value >> (index * 8U)) & 0xFFULL;
        hash *= fnv_prime;
    }
}

void hash_string(std::uint64_t& hash, const std::string_view value) noexcept {
    hash_u64(hash, static_cast<std::uint64_t>(value.size()));
    for (const char character : value) {
        hash ^= static_cast<unsigned char>(character);
        hash *= fnv_prime;
    }
}

[[nodiscard]] std::string trim(const std::string_view input) {
    std::size_t begin = 0U;
    while (begin < input.size() && std::isspace(static_cast<unsigned char>(input[begin])) != 0) ++begin;
    std::size_t end = input.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(input[end - 1U])) != 0) --end;
    return std::string(input.substr(begin, end - begin));
}

[[nodiscard]] std::vector<std::string> split(
    const std::string_view value,
    const char delimiter
) {
    std::vector<std::string> parts;
    std::size_t start = 0U;
    while (start <= value.size()) {
        const std::size_t position = value.find(delimiter, start);
        const std::size_t length = position == std::string_view::npos
            ? value.size() - start
            : position - start;
        parts.push_back(trim(value.substr(start, length)));
        if (position == std::string_view::npos) break;
        start = position + 1U;
    }
    return parts;
}

[[nodiscard]] std::vector<std::uint8_t> read_all_bytes(
    const std::filesystem::path& path
) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("unable to open dataset payload: " + path.string());
    return std::vector<std::uint8_t>(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>()
    );
}

[[nodiscard]] std::filesystem::path resolve_path(
    const std::filesystem::path& directory,
    const std::string& value
) {
    const std::filesystem::path path(value);
    return path.is_absolute() ? path : directory / path;
}

struct EntryHashes final {
    std::uint64_t aggregate{};
    std::vector<std::uint64_t> components;
};

[[nodiscard]] EntryHashes entry_content_hashes(
    const DatasetEntry& entry,
    const std::filesystem::path& directory
) {
    EntryHashes result;
    if (entry.modality == Modality::structured || entry.modality == Modality::text ||
        entry.modality == Modality::sensor || entry.modality == Modality::action) {
        result.aggregate = hash_bytes(std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(entry.payload.data()),
            entry.payload.size()
        ));
        result.components.push_back(result.aggregate);
        return result;
    }
    std::uint64_t aggregate = fnv_offset;
    const char delimiter = entry.modality == Modality::video ? ';' : '\0';
    const std::vector<std::string> paths = delimiter == '\0'
        ? std::vector<std::string>{entry.payload}
        : split(entry.payload, delimiter);
    if (paths.empty()) throw std::runtime_error("dataset media entry has no payload files");
    result.components.reserve(paths.size());
    for (const auto& path_text : paths) {
        const auto bytes = read_all_bytes(resolve_path(directory, path_text));
        const std::uint64_t component = hash_bytes(bytes);
        result.components.push_back(component);
        hash_u64(aggregate, component);
        hash_u64(aggregate, static_cast<std::uint64_t>(bytes.size()));
    }
    result.aggregate = aggregate;
    return result;
}

[[nodiscard]] std::vector<float> structured_descriptor(const std::string_view payload) {
    std::vector<float> descriptor(32U, 0.0F);
    std::uint64_t rolling = fnv_offset;
    for (std::size_t index = 0U; index < payload.size(); ++index) {
        const unsigned char character = static_cast<unsigned char>(payload[index]);
        rolling ^= character;
        rolling *= fnv_prime;
        const std::size_t bucket = static_cast<std::size_t>((rolling ^ index) % descriptor.size());
        descriptor[bucket] += 1.0F;
    }
    double norm = 0.0;
    for (const float value : descriptor) norm += static_cast<double>(value) * static_cast<double>(value);
    norm = std::sqrt(norm);
    if (norm > 0.0) {
        for (float& value : descriptor) value = static_cast<float>(static_cast<double>(value) / norm);
    }
    return descriptor;
}

[[nodiscard]] std::uint64_t report_hash(
    const std::uint64_t model_hash,
    const std::size_t left,
    const std::size_t right,
    const double value
) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, model_hash);
    hash_u64(hash, static_cast<std::uint64_t>(left));
    hash_u64(hash, static_cast<std::uint64_t>(right));
    hash_u64(hash, std::bit_cast<std::uint64_t>(value));
    return hash;
}

}  // namespace

std::string_view to_string(const DatasetSplit split_value) noexcept {
    switch (split_value) {
    case DatasetSplit::train: return "train";
    case DatasetSplit::development: return "development";
    case DatasetSplit::evaluation: return "evaluation";
    }
    return "unknown";
}

DatasetSplit parse_dataset_split(const std::string_view value) {
    if (value == "train") return DatasetSplit::train;
    if (value == "development" || value == "dev") return DatasetSplit::development;
    if (value == "evaluation" || value == "eval") return DatasetSplit::evaluation;
    throw std::invalid_argument("unsupported dataset split: " + std::string(value));
}

Modality parse_modality(const std::string_view value) {
    if (value == "text") return Modality::text;
    if (value == "structured" || value == "knowledge") return Modality::structured;
    if (value == "image") return Modality::image;
    if (value == "video") return Modality::video;
    if (value == "audio") return Modality::audio;
    if (value == "sensor") return Modality::sensor;
    if (value == "action") return Modality::action;
    throw std::invalid_argument("unsupported modality: " + std::string(value));
}

FrontierBackendKind parse_frontier_backend(const std::string_view value) {
    if (value == "scalar" || value == "scalar_cpu") return FrontierBackendKind::scalar_cpu;
    if (value == "optimized" || value == "optimized_cpu" || value == "cpu") {
        return FrontierBackendKind::optimized_cpu;
    }
    if (value == "cuda" || value == "gpu") return FrontierBackendKind::cuda;
    throw std::invalid_argument("unsupported Frontier backend: " + std::string(value));
}

FrontierModel::FrontierModel(const std::uint64_t seed_value)
    : seed(seed_value), fabric(seed_value) {}

DatasetManifest ManifestLoader::load(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("unable to open dataset manifest: " + path.string());
    DatasetManifest manifest;
    manifest.source_path = path;
    std::string line;
    std::size_t line_number = 0U;
    std::uint64_t manifest_hash = fnv_offset;
    const std::filesystem::path directory = path.parent_path();
    while (std::getline(input, line)) {
        ++line_number;
        const std::string stripped = trim(line);
        if (stripped.empty() || stripped.starts_with('#')) continue;
        const auto fields = split(stripped, '\t');
        if (fields.size() != 4U) {
            throw std::runtime_error(
                "manifest line " + std::to_string(line_number) +
                " must contain split, modality, label, and payload"
            );
        }
        DatasetEntry entry;
        entry.split = parse_dataset_split(fields[0]);
        entry.modality = parse_modality(fields[1]);
        entry.label = fields[2];
        entry.payload = fields[3];
        entry.line_number = line_number;
        if (entry.label.empty() || entry.payload.empty()) {
            throw std::runtime_error("manifest label and payload must be non-empty");
        }
        EntryHashes hashes = entry_content_hashes(entry, directory);
        entry.content_hash = hashes.aggregate;
        entry.component_hashes = std::move(hashes.components);
        hash_string(manifest_hash, stripped);
        hash_u64(manifest_hash, entry.content_hash);
        for (const std::uint64_t component_hash : entry.component_hashes) {
            hash_u64(manifest_hash, component_hash);
        }
        manifest.entries.push_back(std::move(entry));
    }
    if (!input.eof()) throw std::runtime_error("failed while reading dataset manifest");
    if (manifest.entries.empty()) throw std::runtime_error("dataset manifest is empty");
    manifest.manifest_hash = manifest_hash;
    return manifest;
}

std::size_t ManifestLoader::leakage_collisions(const DatasetManifest& manifest) {
    std::map<std::uint64_t, std::set<DatasetSplit>> splits_by_hash;
    for (const auto& entry : manifest.entries) {
        splits_by_hash[entry.content_hash].insert(entry.split);
        if (entry.modality == Modality::video) {
            for (const std::uint64_t component_hash : entry.component_hashes) {
                splits_by_hash[component_hash].insert(entry.split);
            }
        }
    }
    std::size_t collisions = 0U;
    for (const auto& [hash, splits] : splits_by_hash) {
        static_cast<void>(hash);
        if (splits.size() > 1U) ++collisions;
    }
    return collisions;
}

FrontierTrainer::FrontierTrainer(
    FrontierModel model,
    const FrontierBackendKind backend
) : FrontierTrainer(std::move(model), make_frontier_backend(backend)) {}

FrontierTrainer::FrontierTrainer(
    FrontierModel model,
    std::unique_ptr<FrontierComputeBackend> backend
) : model_(std::move(model)), backend_(std::move(backend)) {
    if (backend_ == nullptr) throw std::invalid_argument("Frontier backend must not be null");
    if (!backend_->capabilities().available) {
        throw std::runtime_error(
            "requested Frontier backend is unavailable: " + std::string(backend_->name())
        );
    }
}
FrontierModel& FrontierTrainer::model() noexcept { return model_; }
const FrontierModel& FrontierTrainer::model() const noexcept { return model_; }
FrontierBackendKind FrontierTrainer::backend_kind() const noexcept { return backend_->kind(); }
std::string_view FrontierTrainer::backend_name() const noexcept { return backend_->name(); }
BackendCapabilities FrontierTrainer::backend_capabilities() const noexcept {
    return backend_->capabilities();
}

void FrontierTrainer::learn_descriptor(
    const Modality modality,
    const std::string& label,
    const std::span<const float> descriptor,
    const double confidence
) {
    if (descriptor.empty()) throw std::invalid_argument("training descriptor must not be empty");
    for (const auto& [id, existing] : model_.fabric.modes()) {
        if (!existing.enabled || existing.modality != modality || existing.label != label) continue;
        ModeRecord* mode = model_.fabric.find_mode(id);
        if (mode == nullptr) throw std::logic_error("mode index became inconsistent");
        if (mode->prototype.size() != descriptor.size()) {
            throw std::invalid_argument("mode prototype dimension changed");
        }
        const double old_support = static_cast<double>(mode->support);
        const double new_support = old_support + 1.0;
        backend_->local_average_update(
            mode->prototype,
            descriptor,
            static_cast<float>(1.0 / new_support)
        );
        mode->confidence = std::clamp(
            (mode->confidence * old_support + confidence) / new_support,
            0.0,
            1.0
        );
        ++mode->support;
        mode->last_used_step = model_.training_step;
        return;
    }
    model_.fabric.learn_mode(modality, label, descriptor, std::nullopt, confidence);
}

std::vector<float> FrontierTrainer::descriptor_for(
    const DatasetEntry& entry,
    const std::filesystem::path& manifest_directory,
    std::uint64_t* bytes_read
) const {
    if (entry.modality == Modality::image) {
        const auto path = resolve_path(manifest_directory, entry.payload);
        const auto file_bytes = read_all_bytes(path);
        if (bytes_read != nullptr) *bytes_read += file_bytes.size();
        return ImageUnderstanding::analyze(ImageUnderstanding::load_pnm(path)).descriptor;
    }
    if (entry.modality == Modality::video) {
        const auto frame_paths = split(entry.payload, ';');
        if (frame_paths.empty()) throw std::runtime_error("video manifest entry has no frames");
        std::vector<Image> frames;
        frames.reserve(frame_paths.size());
        for (const auto& frame_path : frame_paths) {
            const auto path = resolve_path(manifest_directory, frame_path);
            const auto file_bytes = read_all_bytes(path);
            if (bytes_read != nullptr) *bytes_read += file_bytes.size();
            frames.push_back(ImageUnderstanding::load_pnm(path));
        }
        return VideoUnderstanding::analyze(frames).descriptor;
    }
    if (entry.modality == Modality::audio) {
        const auto path = resolve_path(manifest_directory, entry.payload);
        const auto file_bytes = read_all_bytes(path);
        if (bytes_read != nullptr) *bytes_read += file_bytes.size();
        return AudioUnderstanding::analyze_wav(path).descriptor;
    }
    return structured_descriptor(entry.payload);
}

TrainingReport FrontierTrainer::train(
    const DatasetManifest& manifest,
    const bool include_development
) {
    TrainingReport report;
    report.leakage_collisions = ManifestLoader::leakage_collisions(manifest);
    if (report.leakage_collisions != 0U) {
        throw std::runtime_error("dataset leakage audit failed");
    }
    const std::filesystem::path directory = manifest.source_path.parent_path();
    const std::size_t modes_before = model_.fabric.modes().size();
    for (const auto& entry : manifest.entries) {
        if (entry.split != DatasetSplit::train &&
            !(include_development && entry.split == DatasetSplit::development)) continue;
        ++report.entries_seen;
        ++model_.training_examples;
        ++model_.training_step;
        model_.fabric.set_step(model_.training_step);
        if (entry.modality == Modality::structured || entry.modality == Modality::text) {
            const auto fields = split(entry.payload, '|');
            if (fields.size() >= 3U) {
                KnowledgeRecord record;
                record.kind = entry.modality == Modality::text
                    ? KnowledgeKind::claim
                    : KnowledgeKind::observed_fact;
                record.subject = fields[0];
                record.predicate = fields[1];
                record.object = fields[2];
                record.source = fields.size() >= 4U ? fields[3] : "manifest";
                record.confidence = 0.8;
                record.creation_step = model_.training_step;
                model_.fabric.insert(std::move(record));
                ++report.knowledge_records_added;
            }
        }
        std::uint64_t bytes_read = 0U;
        const std::vector<float> descriptor = descriptor_for(entry, directory, &bytes_read);
        model_.media_bytes_read += bytes_read;
        learn_descriptor(entry.modality, entry.label, descriptor, 0.8);
        switch (entry.modality) {
        case Modality::image: ++report.image_examples; break;
        case Modality::video: ++report.video_examples; break;
        case Modality::audio: ++report.audio_examples; break;
        default: ++report.structured_examples; break;
        }
    }
    report.modes_learned = model_.fabric.modes().size() - modes_before;
    model_.fabric.consolidate(1'024U, 128U);
    report.deterministic_hash = report_hash(
        model_.fabric.deterministic_hash(),
        report.entries_seen,
        report.modes_learned,
        static_cast<double>(report.knowledge_records_added)
    );
    return report;
}

std::string FrontierTrainer::classify(
    const Modality modality,
    const std::vector<float>& descriptor,
    double* confidence
) const {
    if (descriptor.empty()) {
        if (confidence != nullptr) *confidence = 0.0;
        return {};
    }
    std::vector<const ModeRecord*> candidates;
    std::vector<float> prototypes;
    for (const auto& [id, mode] : model_.fabric.modes()) {
        static_cast<void>(id);
        if (!mode.enabled || mode.modality != modality || mode.prototype.size() != descriptor.size()) {
            continue;
        }
        candidates.push_back(&mode);
        prototypes.insert(prototypes.end(), mode.prototype.begin(), mode.prototype.end());
    }
    if (candidates.empty()) {
        if (confidence != nullptr) *confidence = 0.0;
        return {};
    }
    const auto similarities = backend_->batch_cosine(
        descriptor,
        1U,
        prototypes,
        candidates.size(),
        descriptor.size()
    );
    struct ScoredMode final {
        const ModeRecord* mode{};
        double score{};
    };
    std::vector<ScoredMode> scored;
    scored.reserve(candidates.size());
    for (std::size_t index = 0U; index < candidates.size(); ++index) {
        scored.push_back({
            .mode = candidates[index],
            .score = static_cast<double>(similarities[index]) *
                (0.75 + 0.25 * candidates[index]->confidence),
        });
    }
    std::sort(scored.begin(), scored.end(), [](const ScoredMode& left, const ScoredMode& right) {
        if (left.score != right.score) return left.score > right.score;
        return left.mode->stable_id > right.mode->stable_id;
    });
    double calibrated = std::clamp((scored.front().score + 1.0) * 0.5, 0.0, 1.0);
    if (scored.size() > 1U) {
        const double margin = std::clamp(scored.front().score - scored[1U].score, 0.0, 2.0) * 0.5;
        calibrated = std::clamp(0.65 * calibrated + 0.35 * margin, 0.0, 1.0);
    }
    if (confidence != nullptr) *confidence = calibrated;
    return scored.front().mode->label;
}

void FrontierTrainer::update_metrics(
    ModalityMetrics& metrics,
    const bool correct,
    const bool abstained,
    const double confidence
) {
    ++metrics.examples;
    if (correct) ++metrics.correct;
    if (abstained) ++metrics.abstentions;
    metrics.mean_confidence += confidence;
}

EvaluationReport FrontierTrainer::evaluate(
    const DatasetManifest& manifest,
    const DatasetSplit split_value,
    const double abstention_threshold
) {
    if (!std::isfinite(abstention_threshold) || abstention_threshold < 0.0 || abstention_threshold > 1.0) {
        throw std::invalid_argument("abstention threshold must be in [0,1]");
    }
    EvaluationReport report;
    report.leakage_collisions = ManifestLoader::leakage_collisions(manifest);
    if (report.leakage_collisions != 0U) {
        throw std::runtime_error("dataset leakage audit failed");
    }
    const std::filesystem::path directory = manifest.source_path.parent_path();
    for (const auto& entry : manifest.entries) {
        if (entry.split != split_value) continue;
        ++model_.evaluation_examples;
        const std::vector<float> descriptor = descriptor_for(entry, directory, nullptr);
        double confidence = 0.0;
        const std::string prediction = classify(entry.modality, descriptor, &confidence);
        const bool abstained = prediction.empty() || confidence < abstention_threshold;
        const bool correct = !abstained && prediction == entry.label;
        ModalityMetrics* metrics = &report.structured;
        if (entry.modality == Modality::image) metrics = &report.image;
        else if (entry.modality == Modality::video) metrics = &report.video;
        else if (entry.modality == Modality::audio) metrics = &report.audio;
        update_metrics(*metrics, correct, abstained, confidence);
    }
    std::size_t examples = 0U;
    std::size_t correct = 0U;
    double confidence_sum = 0.0;
    for (ModalityMetrics* metrics : {&report.image, &report.video, &report.audio, &report.structured}) {
        if (metrics->examples != 0U) {
            metrics->accuracy = static_cast<double>(metrics->correct) /
                static_cast<double>(metrics->examples);
            metrics->mean_confidence /= static_cast<double>(metrics->examples);
        }
        examples += metrics->examples;
        correct += metrics->correct;
        confidence_sum += metrics->mean_confidence * static_cast<double>(metrics->examples);
    }
    report.aggregate_accuracy = examples == 0U ? 0.0 :
        static_cast<double>(correct) / static_cast<double>(examples);
    report.deterministic_hash = report_hash(
        model_.fabric.deterministic_hash(),
        examples,
        correct,
        examples == 0U ? 0.0 : confidence_sum / static_cast<double>(examples)
    );
    return report;
}


void write_training_report_json(std::ostream& output, const TrainingReport& report) {
    output << std::setprecision(10)
        << "{\n"
        << "  \"workflow\": \"frontier_train\",\n"
        << "  \"entries_seen\": " << report.entries_seen << ",\n"
        << "  \"knowledge_records_added\": " << report.knowledge_records_added << ",\n"
        << "  \"modes_learned\": " << report.modes_learned << ",\n"
        << "  \"image_examples\": " << report.image_examples << ",\n"
        << "  \"video_examples\": " << report.video_examples << ",\n"
        << "  \"audio_examples\": " << report.audio_examples << ",\n"
        << "  \"structured_examples\": " << report.structured_examples << ",\n"
        << "  \"leakage_collisions\": " << report.leakage_collisions << ",\n"
        << "  \"deterministic_hash\": \"" << std::hex << std::setw(16) << std::setfill('0')
        << report.deterministic_hash << std::dec << "\"\n}\n";
}

void write_evaluation_report_json(std::ostream& output, const EvaluationReport& report) {
    const auto write_modality = [&output](const char* name, const ModalityMetrics& metrics, const bool comma) {
        output << "    \"" << name << "\": {\"examples\": " << metrics.examples
            << ", \"correct\": " << metrics.correct
            << ", \"abstentions\": " << metrics.abstentions
            << ", \"accuracy\": " << metrics.accuracy
            << ", \"mean_confidence\": " << metrics.mean_confidence << "}"
            << (comma ? ",\n" : "\n");
    };
    output << std::setprecision(10)
        << "{\n  \"workflow\": \"frontier_evaluate\",\n"
        << "  \"aggregate_accuracy\": " << report.aggregate_accuracy << ",\n"
        << "  \"leakage_collisions\": " << report.leakage_collisions << ",\n"
        << "  \"modalities\": {\n";
    write_modality("image", report.image, true);
    write_modality("video", report.video, true);
    write_modality("audio", report.audio, true);
    write_modality("structured", report.structured, false);
    output << "  },\n  \"deterministic_hash\": \"" << std::hex << std::setw(16) << std::setfill('0')
        << report.deterministic_hash << std::dec << "\"\n}\n";
}

}  // namespace rlf::frontier
