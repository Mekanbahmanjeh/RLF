#include "rlf/sdk/trainer.hpp"

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace rlf::sdk {

Trainer::Trainer(AutoModel model)
    : model_(std::move(model)) {
    stats_.model = model_.model().stats();
}

void Trainer::train(const TrainingRecord& record) {
    solstice::SolsticeModel& model = model_.mutable_model();
    std::visit(
        [this, &model](const auto& value) {
            using Record = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Record, TextTrainingRecord>) {
                if (value.corpus.empty()) {
                    throw std::invalid_argument("text corpus must not be empty");
                }
                model.train_text_corpus(value.corpus);
                ++stats_.text_records;
            } else if constexpr (std::is_same_v<Record, DialogueTrainingRecord>) {
                if (value.prompt.empty() || value.response.empty()) {
                    throw std::invalid_argument(
                        "dialogue prompt and response must not be empty"
                    );
                }
                model.train_dialogue(
                    value.prompt, value.response, value.grounding
                );
                ++stats_.dialogue_records;
            } else if constexpr (std::is_same_v<Record, ImageTrainingRecord>) {
                if (value.image.empty() || value.caption.empty()) {
                    throw std::invalid_argument(
                        "image path and caption must not be empty"
                    );
                }
                model.train_image_file(value.image, value.caption, value.limits);
                ++stats_.image_records;
            } else if constexpr (
                std::is_same_v<Record, InstructionTrainingRecord>
            ) {
                model.train_instruction(
                    value.task, value.domain, value.prompt, value.rationale,
                    value.response, value.quality
                );
                ++stats_.instruction_records;
            } else if constexpr (
                std::is_same_v<Record, PreferenceTrainingRecord>
            ) {
                model.train_preference(
                    value.prompt, value.chosen, value.rejected,
                    value.feedback, value.weight
                );
                ++stats_.preference_records;
            } else if constexpr (
                std::is_same_v<Record, ToolRouteTrainingRecord>
            ) {
                model.train_tool_route(value.request, value.tool_name);
                ++stats_.tool_route_records;
            } else if constexpr (std::is_same_v<Record, FactTrainingRecord>) {
                static_cast<void>(model.learn_fact(
                    value.subject, value.relation, value.object,
                    value.confidence, value.provenance
                ));
                ++stats_.fact_records;
            } else if constexpr (std::is_same_v<Record, RuleTrainingRecord>) {
                static_cast<void>(model.learn_rule(
                    value.name, value.premises, value.conclusion,
                    value.confidence
                ));
                ++stats_.rule_records;
            }
        },
        record
    );
    ++stats_.records;
    model_.refresh_info();
    stats_.model = model_.model().stats();
}

void Trainer::train(const std::span<const TrainingRecord> records) {
    for (const TrainingRecord& record : records) {
        train(record);
    }
}

void Trainer::save_pretrained(
    const std::filesystem::path& directory,
    SaveOptions options
) {
    model_.save_pretrained(directory, std::move(options));
    stats_.model = model_.model().stats();
}

const AutoModel& Trainer::model() const noexcept {
    return model_;
}

TrainerStats Trainer::stats() const noexcept {
    return stats_;
}

}  // namespace rlf::sdk
