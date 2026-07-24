#pragma once

#include "rlf/frontier/frontier_backend.hpp"
#include "rlf/frontier/knowledge_fabric.hpp"
#include "rlf/frontier/multimodal.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace rlf::frontier {

enum class DatasetSplit : std::uint8_t { train, development, evaluation };

struct DatasetEntry final {
    DatasetSplit split{DatasetSplit::train};
    Modality modality{Modality::structured};
    std::string label;
    std::string payload;
    std::uint64_t content_hash{};
    std::vector<std::uint64_t> component_hashes;
    std::size_t line_number{};
};

struct DatasetManifest final {
    std::filesystem::path source_path;
    std::vector<DatasetEntry> entries;
    std::uint64_t manifest_hash{};
};

struct FrontierModel final {
    explicit FrontierModel(std::uint64_t seed = 0x524C4638ULL);

    std::uint64_t seed{};
    std::uint64_t training_step{};
    std::uint64_t training_examples{};
    std::uint64_t evaluation_examples{};
    std::uint64_t media_bytes_read{};
    KnowledgeFabric fabric;
};

struct ModalityMetrics final {
    std::size_t examples{};
    std::size_t correct{};
    std::size_t abstentions{};
    double accuracy{};
    double mean_confidence{};
};

struct TrainingReport final {
    std::size_t entries_seen{};
    std::size_t knowledge_records_added{};
    std::size_t modes_learned{};
    std::size_t image_examples{};
    std::size_t video_examples{};
    std::size_t audio_examples{};
    std::size_t structured_examples{};
    std::size_t leakage_collisions{};
    std::uint64_t deterministic_hash{};
};

struct EvaluationReport final {
    ModalityMetrics image;
    ModalityMetrics video;
    ModalityMetrics audio;
    ModalityMetrics structured;
    std::size_t leakage_collisions{};
    double aggregate_accuracy{};
    std::uint64_t deterministic_hash{};
};

class ManifestLoader final {
public:
    [[nodiscard]] static DatasetManifest load(
        const std::filesystem::path& path
    );
    [[nodiscard]] static std::size_t leakage_collisions(
        const DatasetManifest& manifest
    );
};

class FrontierTrainer final {
public:
    explicit FrontierTrainer(
        FrontierModel model = FrontierModel{},
        FrontierBackendKind backend = FrontierBackendKind::optimized_cpu
    );
    FrontierTrainer(
        FrontierModel model,
        std::unique_ptr<FrontierComputeBackend> backend
    );

    [[nodiscard]] FrontierModel& model() noexcept;
    [[nodiscard]] const FrontierModel& model() const noexcept;
    [[nodiscard]] FrontierBackendKind backend_kind() const noexcept;
    [[nodiscard]] std::string_view backend_name() const noexcept;
    [[nodiscard]] BackendCapabilities backend_capabilities() const noexcept;

    [[nodiscard]] TrainingReport train(
        const DatasetManifest& manifest,
        bool include_development = false
    );
    [[nodiscard]] EvaluationReport evaluate(
        const DatasetManifest& manifest,
        DatasetSplit split = DatasetSplit::evaluation,
        double abstention_threshold = 0.25
    );

    [[nodiscard]] std::string classify(
        Modality modality,
        const std::vector<float>& descriptor,
        double* confidence = nullptr
    ) const;

private:
    FrontierModel model_;
    std::unique_ptr<FrontierComputeBackend> backend_;

    [[nodiscard]] std::vector<float> descriptor_for(
        const DatasetEntry& entry,
        const std::filesystem::path& manifest_directory,
        std::uint64_t* bytes_read = nullptr
    ) const;
    static void update_metrics(
        ModalityMetrics& metrics,
        bool correct,
        bool abstained,
        double confidence
    );
    void learn_descriptor(
        Modality modality,
        const std::string& label,
        std::span<const float> descriptor,
        double confidence
    );
};

[[nodiscard]] std::string_view to_string(DatasetSplit split) noexcept;
[[nodiscard]] DatasetSplit parse_dataset_split(std::string_view value);
[[nodiscard]] Modality parse_modality(std::string_view value);
[[nodiscard]] FrontierBackendKind parse_frontier_backend(std::string_view value);
void write_training_report_json(std::ostream& output, const TrainingReport& report);
void write_evaluation_report_json(std::ostream& output, const EvaluationReport& report);

}  // namespace rlf::frontier
