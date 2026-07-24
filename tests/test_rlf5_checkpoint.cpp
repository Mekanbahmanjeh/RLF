#include "test_framework.hpp"

#include "rlf/core/language_fabric.hpp"
#include "rlf/storage/rlf5_checkpoint.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

RLF_TEST_CASE("RLF-5 checkpoint round trip and corruption rejection") {
    rlf::core::LanguageFabricConfig config;
    config.phase_dimension = 8U;
    config.maximum_lexemes = 384U;
    config.maximum_merges = 100U;
    config.minimum_pair_support = 2U;
    config.maximum_context_order = 4U;
    config.minimum_context_support = 1U;
    config.minimum_construction_support = 1U;
    rlf::core::LanguageFabric fabric(config, 0x524C4635434B50ULL);
    const std::string corpus =
        "the red fox carries the blue key.\n"
        "the blue raven carries the red book.\n"
        "the red fox carries the blue key.\n"
        "the blue raven carries the red book.\n";
    fabric.learn_lexicon(corpus);
    fabric.train_language_model(corpus);
    const std::vector<rlf::core::LanguageSupervisedExample> examples{
        {
            "the red fox carries the blue key.\n",
            {rlf::core::LanguageAct::statement, "carry", "fox", "key",
             "red", "blue", ""},
        },
        {
            "the blue raven carries the red book.\n",
            {rlf::core::LanguageAct::statement, "carry", "raven", "book",
             "blue", "red", ""},
        },
    };
    fabric.train_semantics(examples);

    const auto path = std::filesystem::temp_directory_path() /
        "rlf5_checkpoint_test.rlf";
    const auto corrupt = std::filesystem::temp_directory_path() /
        "rlf5_checkpoint_corrupt.rlf";
    const auto truncated = std::filesystem::temp_directory_path() /
        "rlf5_checkpoint_truncated.rlf";
    rlf::storage::save_rlf5_checkpoint(path, fabric);
    const auto restored = rlf::storage::load_rlf5_checkpoint(path);
    RLF_CHECK(restored.deterministic_hash() == fabric.deterministic_hash());
    const auto summary = rlf::storage::inspect_rlf5_checkpoint(path);
    RLF_CHECK(summary.format_version == 7U);
    RLF_CHECK(summary.lexeme_count == fabric.lexemes().size());

    std::ifstream input(path, std::ios::binary);
    std::vector<char> bytes{
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()
    };
    input.close();
    RLF_CHECK(bytes.size() > 32U);
    bytes.back() = static_cast<char>(bytes.back() ^ 0x01);
    std::ofstream output(corrupt, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    output.close();
    RLF_CHECK_THROWS_AS(
        rlf::storage::load_rlf5_checkpoint(corrupt), std::runtime_error
    );
    bytes.resize(bytes.size() - 11U);
    std::ofstream truncated_output(
        truncated, std::ios::binary | std::ios::trunc
    );
    truncated_output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    truncated_output.close();
    RLF_CHECK_THROWS_AS(
        rlf::storage::load_rlf5_checkpoint(truncated), std::runtime_error
    );
    rlf::storage::Rlf5CheckpointLoadOptions tight;
    tight.maximum_lexemes = 100U;
    RLF_CHECK_THROWS_AS(
        rlf::storage::load_rlf5_checkpoint(path, tight), std::runtime_error
    );
    std::filesystem::remove(path);
    std::filesystem::remove(corrupt);
    std::filesystem::remove(truncated);
}
