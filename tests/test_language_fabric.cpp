#include "test_framework.hpp"

#include "rlf/core/language_fabric.hpp"

#include <string>
#include <vector>

namespace {

rlf::core::LanguageFabricConfig test_config() {
    rlf::core::LanguageFabricConfig config;
    config.phase_dimension = 16U;
    config.maximum_lexemes = 512U;
    config.maximum_merges = 200U;
    config.minimum_pair_support = 2U;
    config.maximum_context_order = 6U;
    config.minimum_context_support = 1U;
    config.minimum_construction_support = 1U;
    config.maximum_surfaces_per_concept = 6U;
    config.minimum_lexical_score = 0.10;
    return config;
}

std::vector<rlf::core::LanguageSupervisedExample> semantic_examples() {
    using rlf::core::LanguageAct;
    using rlf::core::LanguageFrame;
    using rlf::core::LanguageSupervisedExample;
    std::vector<LanguageSupervisedExample> examples;
    const std::vector<std::string> agents{"fox", "raven", "otter"};
    const std::vector<std::string> patients{"key", "book", "lamp"};
    const std::vector<std::string> colors{"red", "blue", "golden"};
    const std::vector<std::string> locations{"garden", "tower", "harbor"};
    const std::size_t combinations = agents.size() * patients.size() *
        colors.size() * colors.size() * locations.size();
    for (std::size_t index = 0U; index < combinations; ++index) {
        std::size_t cursor = index;
        const auto& agent = agents[cursor % agents.size()];
        cursor /= agents.size();
        const auto& patient = patients[cursor % patients.size()];
        cursor /= patients.size();
        const auto& agent_color = colors[cursor % colors.size()];
        cursor /= colors.size();
        const auto& patient_color = colors[cursor % colors.size()];
        cursor /= colors.size();
        const auto& location = locations[cursor % locations.size()];
        LanguageFrame frame{
            LanguageAct::statement,
            "carry",
            agent,
            patient,
            agent_color,
            patient_color,
            location,
        };
        examples.push_back({
            "the " + agent_color + " " + agent + " carries the " +
                patient_color + " " + patient + " in the " + location + ".\n",
            frame,
        });
        LanguageFrame query = frame;
        query.act = LanguageAct::query_patient;
        query.patient.clear();
        query.patient_attribute.clear();
        examples.push_back({
            "what does the " + agent_color + " " + agent +
                " carry in the " + location + "?\n",
            query,
        });
        LanguageFrame answer;
        answer.act = LanguageAct::answer_patient;
        answer.patient = patient;
        answer.patient_attribute = patient_color;
        examples.push_back({
            "the " + patient_color + " " + patient + ".\n",
            answer,
        });
    }
    return examples;
}

std::string corpus_from(const std::vector<rlf::core::LanguageSupervisedExample>& examples) {
    std::string corpus;
    for (const auto& example : examples) {
        corpus += example.text;
    }
    for (std::size_t repeat = 0U; repeat < 4U; ++repeat) {
        for (const auto& example : examples) {
            corpus += example.text;
        }
    }
    return corpus;
}

}  // namespace

RLF_TEST_CASE("RLF-5 byte lexicon round trips and compresses repeated words") {
    const auto examples = semantic_examples();
    const auto corpus = corpus_from(examples);
    rlf::core::LanguageFabric fabric(test_config(), 0x524C46354C4558ULL);
    fabric.learn_lexicon(corpus);
    const std::string text = "the red fox carries the blue key.\n";
    const auto encoded = fabric.encode(text);
    RLF_CHECK(fabric.decode(encoded) == text);
    RLF_CHECK(fabric.merges().size() > 20U);
    RLF_CHECK(encoded.size() < text.size());
}

RLF_TEST_CASE("RLF-5 language model predicts and generates lexeme sequences") {
    const auto examples = semantic_examples();
    const auto corpus = corpus_from(examples);
    rlf::core::LanguageFabric fabric(test_config(), 0x524C46354C4DULL);
    fabric.learn_lexicon(corpus);
    fabric.train_language_model(corpus);
    RLF_CHECK(fabric.sequence_nll(corpus.substr(0U, 512U)) < 8.0);
    const auto generated = fabric.generate("the red fox", 16U, true);
    RLF_CHECK(generated.starts_with("the red fox"));
    RLF_CHECK(generated.size() > std::string("the red fox").size());
}

RLF_TEST_CASE("RLF-5 semantic grammar composes unseen frames and answers queries") {
    const auto examples = semantic_examples();
    const auto corpus = corpus_from(examples);
    rlf::core::LanguageFabric fabric(test_config(), 0x524C463553454DULL);
    fabric.learn_lexicon(corpus);
    fabric.train_language_model(corpus);
    fabric.train_semantics(examples);

    const std::string statement =
        "the golden raven carries the red lamp in the tower.\n";
    const auto parsed = fabric.parse(statement);
    RLF_CHECK(parsed.success);
    RLF_CHECK(parsed.frame.act == rlf::core::LanguageAct::statement);
    RLF_CHECK(parsed.frame.agent == "raven");
    RLF_CHECK(parsed.frame.patient == "lamp");
    RLF_CHECK(parsed.frame.location == "tower");

    const std::vector<std::string> context{statement};
    const auto answer = fabric.answer(
        context, "what does the golden raven carry in the tower?\n"
    );
    RLF_CHECK(answer.success);
    RLF_CHECK(answer.answer_frame.patient == "lamp");
    RLF_CHECK(answer.answer_frame.patient_attribute == "red");
    const auto answer_parse = fabric.parse(answer.text);
    RLF_CHECK(answer_parse.success);
    RLF_CHECK(answer_parse.frame == answer.answer_frame);
}

RLF_TEST_CASE("RLF-5 snapshot rejects duplicate semantic and context structures") {
    const auto examples = semantic_examples();
    const auto corpus = corpus_from(examples);
    rlf::core::LanguageFabric fabric(test_config(), 0x524C463556414CULL);
    fabric.learn_lexicon(corpus);
    fabric.train_language_model(corpus);
    fabric.train_semantics(examples);

    auto duplicate_context = fabric.snapshot();
    RLF_CHECK(!duplicate_context.contexts.empty());
    auto copied_context = duplicate_context.contexts.front();
    copied_context.id = duplicate_context.next_context_id++;
    duplicate_context.contexts.push_back(std::move(copied_context));
    RLF_CHECK_THROWS_AS(
        rlf::core::LanguageFabric::from_snapshot(std::move(duplicate_context)),
        std::runtime_error
    );

    auto duplicate_concept = fabric.snapshot();
    RLF_CHECK(!duplicate_concept.concepts.empty());
    auto copied_concept = duplicate_concept.concepts.front();
    copied_concept.id = duplicate_concept.next_concept_id++;
    duplicate_concept.concepts.push_back(std::move(copied_concept));
    RLF_CHECK_THROWS_AS(
        rlf::core::LanguageFabric::from_snapshot(std::move(duplicate_concept)),
        std::runtime_error
    );
}
