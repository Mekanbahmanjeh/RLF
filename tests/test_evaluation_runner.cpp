#include "test_framework.hpp"

#include "rlf/core/sha256.hpp"
#include "rlf/solstice/evaluation_runner.hpp"
#include "rlf/solstice/solstice_model.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

[[nodiscard]] std::filesystem::path evaluation_test_root() {
    return std::filesystem::temp_directory_path() / "rlf_evaluation_runner_test";
}

void write_text(const std::filesystem::path& path, const std::string& value) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << value;
    if (!output) {
        throw std::runtime_error("unable to create evaluation fixture");
    }
}

}  // namespace

RLF_TEST_CASE("Evaluation batch produces hash-linked artifacts and resumes safely") {
    const std::filesystem::path root = evaluation_test_root();
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const std::filesystem::path first = root / "first.txt";
    const std::filesystem::path second = root / "second.txt";
    const std::filesystem::path image_path = root / "red.ppm";
    write_text(first, "What can you do?");
    write_text(second, "Who are you?");
    std::string image = "P6\n8 8\n255\n";
    for (std::size_t pixel = 0U; pixel < 64U; ++pixel) {
        image.push_back(static_cast<char>(240));
        image.push_back(static_cast<char>(20));
        image.push_back(static_cast<char>(20));
    }
    write_text(image_path, image);
    const std::string first_hash = rlf::core::sha256_hex(rlf::core::sha256_file(first));
    const std::string second_hash = rlf::core::sha256_hex(rlf::core::sha256_file(second));
    const std::string image_hash = rlf::core::sha256_hex(
        rlf::core::sha256_file(image_path)
    );
    write_text(
        root / "manifest.tsv",
        "# id\tprompt_path\tprompt_sha256\timage_path\timage_sha256\n"
        "first\tfirst.txt\t" + first_hash + "\t-\t-\n"
        "second\tsecond.txt\t" + second_hash + "\t-\t-\n"
        "visual\tfirst.txt\t" + first_hash + "\tred.ppm\t" + image_hash + "\n"
    );

    rlf::solstice::SolsticeModel model;
    model.bootstrap();
    model.train_image_file(image_path, "a red square");
    rlf::solstice::EvaluationBatchOptions options;
    options.manifest_path = root / "manifest.tsv";
    options.output_directory = root / "results";
    options.checkpoint_sha256 = std::string(64U, 'a');
    options.generation = {64U, 8U, 0.8, true, 7U};
    options.allow_images = false;
    RLF_CHECK_THROWS_AS(
        rlf::solstice::run_evaluation_batch(model, options),
        std::runtime_error
    );
    RLF_CHECK(!std::filesystem::exists(options.output_directory));
    options.allow_images = true;
    const auto first_run = rlf::solstice::run_evaluation_batch(model, options);
    RLF_CHECK(first_run.total_examples == 3U);
    RLF_CHECK(first_run.produced_examples == 3U);
    RLF_CHECK(first_run.resumed_examples == 0U);
    RLF_CHECK(std::filesystem::is_regular_file(root / "results/first.json"));
    RLF_CHECK(std::filesystem::is_regular_file(root / "results/visual.json"));
    RLF_CHECK(std::filesystem::is_regular_file(root / "results/run_identity.tsv"));
    RLF_CHECK(std::filesystem::is_regular_file(root / "results/run_summary.json"));

    const auto resumed = rlf::solstice::run_evaluation_batch(model, options);
    RLF_CHECK(resumed.produced_examples == 0U);
    RLF_CHECK(resumed.resumed_examples == 3U);

    write_text(
        root / "manifest_changed.tsv",
        "first\tfirst.txt\t" + first_hash + "\t-\t-\n"
    );
    options.manifest_path = root / "manifest_changed.tsv";
    bool identity_rejected = false;
    try {
        static_cast<void>(rlf::solstice::run_evaluation_batch(model, options));
    } catch (const std::runtime_error&) {
        identity_rejected = true;
    }
    RLF_CHECK(identity_rejected);
    options.manifest_path = root / "manifest.tsv";

    write_text(root / "results/first.json", "tampered\n");
    bool rejected = false;
    try {
        static_cast<void>(rlf::solstice::run_evaluation_batch(model, options));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    RLF_CHECK(rejected);
    std::filesystem::remove_all(root);
}

RLF_TEST_CASE("Evaluation batch audits every input before creating output") {
    const std::filesystem::path root = evaluation_test_root();
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    write_text(root / "prompt.txt", "held-out prompt");
    write_text(
        root / "manifest.tsv",
        "example\tprompt.txt\t" + std::string(64U, '0') + "\t-\t-\n"
    );
    rlf::solstice::SolsticeModel model;
    rlf::solstice::EvaluationBatchOptions options;
    options.manifest_path = root / "manifest.tsv";
    options.output_directory = root / "results";
    options.checkpoint_sha256 = std::string(64U, 'a');
    bool rejected = false;
    try {
        static_cast<void>(rlf::solstice::run_evaluation_batch(model, options));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    RLF_CHECK(rejected);
    RLF_CHECK(!std::filesystem::exists(options.output_directory));
    std::filesystem::remove_all(root);
}
