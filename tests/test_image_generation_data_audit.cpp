#include "test_framework.hpp"

#include "rlf/core/sha256.hpp"
#include "rlf/solstice/image_generation_data_audit.hpp"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

void write_ppm(
    const std::filesystem::path& path,
    const bool horizontal
) {
    std::ofstream output(path, std::ios::binary);
    output << "P6\n16 16\n255\n";
    for (std::size_t y = 0U; y < 16U; ++y) {
        for (std::size_t x = 0U; x < 16U; ++x) {
            const auto value = static_cast<unsigned char>(
                (horizontal ? x : 15U - x) * 16U
            );
            const std::vector<unsigned char> pixel{value, value, value};
            output.write(
                reinterpret_cast<const char*>(pixel.data()),
                static_cast<std::streamsize>(pixel.size())
            );
        }
    }
}

void write_manifest(
    const std::filesystem::path& path,
    const std::string& id,
    const std::string& target_name,
    const std::string& target_hash
) {
    constexpr std::string_view marker = "@neutral-gray128-target-size-v1";
    std::ofstream output(path);
    output << id << '\t' << marker << '\t'
           << rlf::core::sha256_hex(rlf::core::sha256(marker)) << '\t'
           << target_name << '\t' << target_hash
           << "\tmake a gradient\tfixture:image-audit\tCC0-1.0\n";
}

void write_source_manifest(
    const std::filesystem::path& path,
    const std::string& source_hash,
    const std::string& target_hash
) {
    std::ofstream output(path);
    output << "eval-source-copy\ttrain.ppm\t" << source_hash
           << "\teval.ppm\t" << target_hash
           << "\ttransform the source\tfixture:image-audit\tCC0-1.0\n";
}

}  // namespace

RLF_TEST_CASE("Image-generation native audit verifies media and frozen split") {
    const auto directory = std::filesystem::temp_directory_path() /
        "rlf_image_generation_native_audit";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    write_ppm(directory / "train.ppm", true);
    write_ppm(directory / "eval.ppm", false);
    const auto train_hash = rlf::core::sha256_hex(
        rlf::core::sha256_file(directory / "train.ppm")
    );
    const auto eval_hash = rlf::core::sha256_hex(
        rlf::core::sha256_file(directory / "eval.ppm")
    );
    write_manifest(directory / "train.tsv", "train-1", "train.ppm", train_hash);
    write_manifest(directory / "eval.tsv", "eval-1", "eval.ppm", eval_hash);
    {
        std::ofstream policy(directory / "licenses.txt");
        policy << "CC0-1.0\n";
    }
    const auto report = rlf::solstice::audit_image_generation_data(
        directory / "train.tsv",
        directory / "eval.tsv",
        directory / "licenses.txt",
        directory / "reports",
        {.maximum_records = 10U, .near_duplicate_hamming_distance = 0U}
    );
    RLF_CHECK(report.passed());
    RLF_CHECK(report.records_audited == 1U);
    RLF_CHECK(report.evaluation_records_audited == 1U);
    RLF_CHECK(std::filesystem::is_regular_file(
        directory / "reports" / "contamination_report.json"
    ));

    write_manifest(
        directory / "contaminated.tsv", "eval-copy", "train.ppm", train_hash
    );
    const auto contaminated = rlf::solstice::audit_image_generation_data(
        directory / "train.tsv",
        directory / "contaminated.tsv",
        directory / "licenses.txt",
        directory / "contaminated-reports",
        {.maximum_records = 10U, .near_duplicate_hamming_distance = 0U}
    );
    RLF_CHECK(!contaminated.passed());
    RLF_CHECK(contaminated.overlap_records == 1U);
    write_source_manifest(
        directory / "source-contaminated.tsv", train_hash, eval_hash
    );
    const auto source_contaminated =
        rlf::solstice::audit_image_generation_data(
            directory / "train.tsv",
            directory / "source-contaminated.tsv",
            directory / "licenses.txt",
            directory / "source-contaminated-reports",
            {.maximum_records = 10U, .near_duplicate_hamming_distance = 0U}
        );
    RLF_CHECK(!source_contaminated.passed());
    RLF_CHECK(source_contaminated.overlap_records == 1U);
    std::filesystem::remove_all(directory);
}
