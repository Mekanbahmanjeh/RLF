#include "test_framework.hpp"

#include "rlf/core/sha256.hpp"
#include "rlf/sdk/pipeline.hpp"
#include "rlf/sdk/session.hpp"
#include "rlf/sdk/trainer.hpp"
#include "rlf/solstice/checkpoint.hpp"
#include "rlf/solstice/profile.hpp"

#include <filesystem>
#include <fstream>
#include <string>

namespace {

class TemporaryBundle final {
public:
    explicit TemporaryBundle(const std::string& suffix)
        : path_(std::filesystem::temp_directory_path() /
            ("rlf_sdk_" + suffix)) {
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }

    ~TemporaryBundle() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

void create_preview_bundle(const std::filesystem::path& directory) {
    const std::filesystem::path checkpoint = directory / "model.rlfsp";
    rlf::solstice::SolsticeModel model(
        rlf::solstice::make_profile_config(
            rlf::solstice::SolsticeProfile::preview_6g
        )
    );
    model.bootstrap();
    rlf::solstice::save_solstice_checkpoint(checkpoint, model);

    rlf::sdk::BundleManifest manifest;
    manifest.name = "sdk-preview";
    manifest.checkpoint = "model.rlfsp";
    manifest.checkpoint_sha256 = rlf::core::sha256_hex(
        rlf::core::sha256_file(checkpoint)
    );
    manifest.profile = rlf::solstice::SolsticeProfile::preview_6g;
    manifest.tasks = {
        rlf::sdk::PipelineTask::text_generation,
        rlf::sdk::PipelineTask::image_text_to_text,
    };
    manifest.license = "MIT";
    rlf::sdk::save_bundle_manifest(directory, manifest);
}

}  // namespace

RLF_TEST_CASE("RLF SDK loads a verified bundle and runs a text pipeline") {
    TemporaryBundle bundle("verified");
    create_preview_bundle(bundle.path());

    const rlf::sdk::AutoModel model =
        rlf::sdk::AutoModel::from_pretrained(bundle.path());
    RLF_CHECK(model.info().name == "sdk-preview");
    RLF_CHECK(model.info().profile ==
              rlf::solstice::SolsticeProfile::preview_6g);
    RLF_CHECK(model.supports(rlf::sdk::PipelineTask::text_generation));
    RLF_CHECK(!model.supports(rlf::sdk::PipelineTask::tool_use));

    const rlf::sdk::Pipeline pipeline(
        rlf::sdk::PipelineTask::text_generation,
        model
    );
    const rlf::sdk::PipelineOutput output =
        pipeline("What can you do?");
    RLF_CHECK(output.text.find("visual patterns") != std::string::npos);

    const rlf::sdk::Pipeline one_call = rlf::sdk::make_pipeline(
        rlf::sdk::PipelineTask::text_generation,
        bundle.path()
    );
    RLF_CHECK(!one_call("Hello").text.empty());
}

RLF_TEST_CASE("RLF SDK fails closed on a bundle hash mismatch") {
    TemporaryBundle bundle("bad_hash");
    create_preview_bundle(bundle.path());
    const std::filesystem::path manifest =
        bundle.path() / rlf::sdk::bundle_manifest_filename;
    rlf::sdk::BundleManifest altered =
        rlf::sdk::load_bundle_manifest(bundle.path());
    altered.checkpoint_sha256 = std::string(64U, '0');
    std::filesystem::remove(manifest);
    rlf::sdk::save_bundle_manifest(bundle.path(), altered);
    RLF_CHECK_THROWS_AS(
        rlf::sdk::AutoModel::from_pretrained(bundle.path()),
        std::runtime_error
    );
}

RLF_TEST_CASE("RLF SDK enforces declared pipeline tasks") {
    TemporaryBundle bundle("tasks");
    create_preview_bundle(bundle.path());
    const rlf::sdk::AutoModel model =
        rlf::sdk::AutoModel::from_pretrained(bundle.path());
    RLF_CHECK_THROWS_AS(
        rlf::sdk::Pipeline(rlf::sdk::PipelineTask::tool_use, model),
        std::invalid_argument
    );
}

RLF_TEST_CASE("RLF SDK loads raw checkpoints without task restrictions") {
    TemporaryBundle bundle("raw");
    const std::filesystem::path checkpoint = bundle.path() / "raw.rlfsp";
    rlf::solstice::SolsticeModel source;
    source.bootstrap();
    rlf::solstice::save_solstice_checkpoint(checkpoint, source);

    const rlf::sdk::AutoModel model =
        rlf::sdk::AutoModel::from_pretrained(checkpoint);
    RLF_CHECK(model.supports(rlf::sdk::PipelineTask::text_generation));
    RLF_CHECK(model.supports(rlf::sdk::PipelineTask::tool_use));
    RLF_CHECK(model.info().checkpoint_sha256.size() == 64U);
}

RLF_TEST_CASE("RLF SDK reports distinct context capabilities") {
    const rlf::sdk::AutoModel model = rlf::sdk::AutoModel::from_profile(
        rlf::solstice::SolsticeProfile::preview_6g
    );
    const auto context = model.info().context;
    RLF_CHECK(context.maximum_predictive_context_tokens == 64U);
    RLF_CHECK(context.maximum_episode_cue_tokens == 256U);
    RLF_CHECK(context.maximum_generation_tokens == 256U);
    RLF_CHECK(context.maximum_retrieval_context_characters == 32'768U);
}

RLF_TEST_CASE("RLF SDK chat session bounds and evicts multi-turn context") {
    rlf::sdk::AutoModel model = rlf::sdk::AutoModel::from_profile(
        rlf::solstice::SolsticeProfile::preview_6g
    );
    rlf::sdk::PipelineOptions pipeline_options;
    pipeline_options.generation.maximum_tokens = 24U;
    rlf::sdk::ChatSession session(
        rlf::sdk::Pipeline(
            rlf::sdk::PipelineTask::text_generation,
            std::move(model),
            pipeline_options
        ),
        rlf::sdk::ContextWindowConfig{
            96U,
            2U,
            "Answer using only learned evidence.",
        }
    );

    for (std::size_t turn = 0U; turn < 5U; ++turn) {
        const auto output = session.send(
            "What can you do? This is conversation turn " +
            std::to_string(turn)
        );
        RLF_CHECK(!output.text.empty());
        RLF_CHECK(
            session.context_stats().input_tokens <=
            session.config().maximum_context_tokens
        );
    }
    RLF_CHECK(session.history().size() <= 4U);
    RLF_CHECK(session.context_stats().evicted_messages > 0U);
}

RLF_TEST_CASE("RLF SDK trainer creates and saves a reloadable model bundle") {
    TemporaryBundle bundle("trainer");
    rlf::sdk::Trainer trainer(rlf::sdk::AutoModel::from_profile(
        rlf::solstice::SolsticeProfile::preview_6g,
        {},
        123U,
        false
    ));
    const std::vector<rlf::sdk::TrainingRecord> records{
        rlf::sdk::DialogueTrainingRecord{
            "What is the project codename?",
            "The project codename is Aurora.",
            {},
        },
        rlf::sdk::InstructionTrainingRecord{
            "question_answering",
            "project",
            "State the project codename.",
            "Retrieve the explicitly learned project identity.",
            "The project codename is Aurora.",
            1.0,
        },
        rlf::sdk::FactTrainingRecord{
            "project",
            "codename",
            "aurora",
            1.0,
            "sdk-test",
        },
    };
    trainer.train(records);
    RLF_CHECK(trainer.stats().records == 3U);
    RLF_CHECK(trainer.stats().dialogue_records == 1U);
    RLF_CHECK(trainer.stats().instruction_records == 1U);
    RLF_CHECK(trainer.stats().fact_records == 1U);

    rlf::sdk::SaveOptions save_options;
    save_options.name = "aurora-preview";
    save_options.profile = rlf::solstice::SolsticeProfile::preview_6g;
    save_options.tasks = {rlf::sdk::PipelineTask::text_generation};
    trainer.save_pretrained(bundle.path(), save_options);

    const rlf::sdk::AutoModel reloaded =
        rlf::sdk::AutoModel::from_pretrained(bundle.path());
    RLF_CHECK(reloaded.info().name == "aurora-preview");
    RLF_CHECK(reloaded.model().language().episodes().size() == 2U);
    RLF_CHECK(reloaded.model().general().demonstrations().size() == 1U);
    RLF_CHECK(reloaded.model().abstraction().facts().size() == 1U);
}
