#pragma once

#include "rlf/sdk/pipeline.hpp"

#include <filesystem>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace rlf::sdk {

struct TextTrainingRecord final {
    std::string corpus;
};

struct DialogueTrainingRecord final {
    std::string prompt;
    std::string response;
    std::string grounding;
};

struct ImageTrainingRecord final {
    std::filesystem::path image;
    std::string caption;
    solstice::ImageLimits limits;
};

struct InstructionTrainingRecord final {
    std::string task;
    std::string domain;
    std::string prompt;
    std::string rationale;
    std::string response;
    double quality{1.0};
};

struct PreferenceTrainingRecord final {
    std::string prompt;
    std::string chosen;
    std::string rejected;
    std::string feedback;
    double weight{1.0};
};

struct ToolRouteTrainingRecord final {
    std::string request;
    std::string tool_name;
};

struct FactTrainingRecord final {
    std::string subject;
    std::string relation;
    std::string object;
    double confidence{1.0};
    std::string provenance;
};

struct RuleTrainingRecord final {
    std::string name;
    std::vector<solstice::RelationalPattern> premises;
    solstice::RelationalPattern conclusion;
    double confidence{1.0};
};

using TrainingRecord = std::variant<
    TextTrainingRecord,
    DialogueTrainingRecord,
    ImageTrainingRecord,
    InstructionTrainingRecord,
    PreferenceTrainingRecord,
    ToolRouteTrainingRecord,
    FactTrainingRecord,
    RuleTrainingRecord
>;

struct TrainerStats final {
    std::size_t records{};
    std::size_t text_records{};
    std::size_t dialogue_records{};
    std::size_t image_records{};
    std::size_t instruction_records{};
    std::size_t preference_records{};
    std::size_t tool_route_records{};
    std::size_t fact_records{};
    std::size_t rule_records{};
    solstice::SolsticeStats model;
};

class Trainer final {
public:
    explicit Trainer(AutoModel model);

    void train(const TrainingRecord& record);
    void train(std::span<const TrainingRecord> records);
    void save_pretrained(
        const std::filesystem::path& directory,
        SaveOptions options = {}
    );

    [[nodiscard]] const AutoModel& model() const noexcept;
    [[nodiscard]] TrainerStats stats() const noexcept;

private:
    AutoModel model_;
    TrainerStats stats_;
};

}  // namespace rlf::sdk
