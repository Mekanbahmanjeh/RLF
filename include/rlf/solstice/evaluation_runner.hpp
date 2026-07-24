#pragma once

#include "rlf/solstice/language_fabric.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace rlf::solstice {

class SolsticeModel;
class ToolRuntime;

struct EvaluationBatchOptions final {
    std::filesystem::path manifest_path;
    std::filesystem::path output_directory;
    std::string checkpoint_sha256;
    std::string backend_name{"unspecified"};
    GenerationSettings generation;
    std::size_t maximum_prompt_bytes{16U * 1024U * 1024U};
    bool allow_images{true};
};

struct EvaluationBatchReport final {
    std::string manifest_sha256;
    std::string checkpoint_sha256;
    std::uint64_t model_hash{};
    std::size_t total_examples{};
    std::size_t produced_examples{};
    std::size_t resumed_examples{};
    std::uint64_t inference_microseconds{};
    std::uint64_t new_inference_microseconds{};
    std::filesystem::path output_directory;
};

[[nodiscard]] EvaluationBatchReport run_evaluation_batch(
    const SolsticeModel& model,
    const EvaluationBatchOptions& options,
    ToolRuntime* tools = nullptr
);

[[nodiscard]] std::string evaluation_batch_report_json(
    const EvaluationBatchReport& report
);

}  // namespace rlf::solstice
