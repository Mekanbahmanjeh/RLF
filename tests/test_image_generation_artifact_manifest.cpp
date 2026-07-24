#include "test_framework.hpp"

#include "rlf/core/sha256.hpp"
#include "rlf/solstice/image_generation_artifact_manifest.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct FixtureArtifact final {
    std::string kind;
    std::filesystem::path relative_path;
};

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("unable to create manifest test fixture");
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!output) throw std::runtime_error("unable to write manifest test fixture");
}

[[nodiscard]] rlf::solstice::ImageData fixture_image() {
    rlf::solstice::ImageData image;
    image.width = 16U;
    image.height = 16U;
    image.rgb.resize(16U * 16U * 3U, std::uint8_t{42U});
    return image;
}

[[nodiscard]] std::filesystem::path fixture_manifest_path(
    const std::filesystem::path& directory
) {
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    rlf::solstice::ImageGenerationCheckpointState checkpoint;
    checkpoint.master_seed = 17U;
    checkpoint.training_step = 1U;
    checkpoint.fabric.train(fixture_image(), "gray integrity fixture");
    const auto checkpoint_path = directory / "model.rlfimg";
    rlf::solstice::save_image_generation_checkpoint(checkpoint_path, checkpoint);

    std::vector<FixtureArtifact> artifacts;
    artifacts.reserve(rlf::solstice::required_image_generation_artifact_kinds.size());
    for (const std::string_view kind :
         rlf::solstice::required_image_generation_artifact_kinds) {
        std::filesystem::path relative_path;
        if (kind == "checkpoint") {
            relative_path = "model.rlfimg";
        } else {
            relative_path = std::string(kind) + ".txt";
            write_text(directory / relative_path, std::string(kind) + "\n");
        }
        artifacts.push_back({std::string(kind), std::move(relative_path)});
    }

    std::ostringstream manifest;
    manifest << "RLF_IMAGEGEN_ARTIFACT_MANIFEST\t1\n";
    for (const FixtureArtifact& artifact : artifacts) {
        const auto full_path = directory / artifact.relative_path;
        manifest << artifact.kind << '\t'
                 << artifact.relative_path.generic_string() << '\t'
                 << std::filesystem::file_size(full_path) << '\t'
                 << rlf::core::sha256_hex(rlf::core::sha256_file(full_path))
                 << '\n';
    }
    const auto manifest_path = directory / "artifacts.tsv";
    write_text(manifest_path, manifest.str());
    const std::string manifest_hash = rlf::core::sha256_hex(
        rlf::core::sha256_file(manifest_path)
    );
    write_text(
        rlf::solstice::image_generation_manifest_sidecar_path(manifest_path),
        "RLF_IMAGEGEN_MANIFEST_SHA256\t1\nsha256\t" + manifest_hash + "\n"
    );
    return manifest_path;
}

void refresh_sidecar(const std::filesystem::path& manifest_path) {
    write_text(
        rlf::solstice::image_generation_manifest_sidecar_path(manifest_path),
        "RLF_IMAGEGEN_MANIFEST_SHA256\t1\nsha256\t" +
            rlf::core::sha256_hex(rlf::core::sha256_file(manifest_path)) + "\n"
    );
}

[[nodiscard]] std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("unable to read manifest fixture");
    const auto raw_size = std::filesystem::file_size(path);
    if (raw_size > static_cast<std::uintmax_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("manifest fixture is too large");
    }
    std::string contents(static_cast<std::size_t>(raw_size), '\0');
    input.read(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (input.gcount() != static_cast<std::streamsize>(contents.size())) {
        throw std::runtime_error("unable to read complete manifest fixture");
    }
    return contents;
}

}  // namespace

RLF_TEST_CASE("Image-generation artifact manifest verifies all sixteen files") {
    const auto directory = std::filesystem::temp_directory_path() /
        "rlf_imagegen_artifact_manifest_valid";
    const auto manifest = fixture_manifest_path(directory);
    const auto report =
        rlf::solstice::verify_image_generation_artifact_manifest(manifest);
    RLF_CHECK(report.format_version == 1U);
    RLF_CHECK(report.schema_valid);
    RLF_CHECK(report.manifest_self_hash_verified);
    RLF_CHECK(report.artifact_set_complete);
    RLF_CHECK(report.artifact_integrity_verified);
    RLF_CHECK(report.checkpoint_verified);
    RLF_CHECK(report.bundle_integrity_verified);
    RLF_CHECK(!report.origin_authenticated);
    RLF_CHECK(!report.state_of_art_claim_proven);
    RLF_CHECK(report.artifacts.size() == 16U);
    RLF_CHECK(report.checkpoint_summary.has_value());
    RLF_CHECK(report.failures.empty());
    std::filesystem::remove_all(directory);
}

RLF_TEST_CASE("Image-generation artifact manifest rejects artifact tampering") {
    const auto directory = std::filesystem::temp_directory_path() /
        "rlf_imagegen_artifact_manifest_tamper";
    const auto manifest = fixture_manifest_path(directory);
    write_text(directory / "ledger.txt", "Ledger\n");
    const auto report =
        rlf::solstice::verify_image_generation_artifact_manifest(manifest);
    RLF_CHECK(report.manifest_self_hash_verified);
    RLF_CHECK(!report.artifact_integrity_verified);
    RLF_CHECK(!report.bundle_integrity_verified);
    RLF_CHECK(!report.state_of_art_claim_proven);
    RLF_CHECK(!report.failures.empty());
    std::filesystem::remove_all(directory);
}

RLF_TEST_CASE("Image-generation artifact manifest rejects a missing artifact") {
    const auto directory = std::filesystem::temp_directory_path() /
        "rlf_imagegen_artifact_manifest_missing";
    const auto manifest = fixture_manifest_path(directory);
    std::filesystem::remove(directory / "data_audit.txt");
    const auto report =
        rlf::solstice::verify_image_generation_artifact_manifest(manifest);
    RLF_CHECK(!report.artifact_integrity_verified);
    RLF_CHECK(!report.bundle_integrity_verified);
    RLF_CHECK(!report.failures.empty());
    std::filesystem::remove_all(directory);
}

RLF_TEST_CASE("Image-generation artifact manifest rejects duplicate kinds") {
    const auto directory = std::filesystem::temp_directory_path() /
        "rlf_imagegen_artifact_manifest_duplicate";
    const auto manifest = fixture_manifest_path(directory);
    std::string contents = read_text(manifest);
    const std::string expected = "data_audit\tdata_audit.txt";
    const std::size_t offset = contents.find(expected);
    RLF_CHECK(offset != std::string::npos);
    contents.replace(offset, std::string("data_audit").size(), "ledger");
    write_text(manifest, contents);
    refresh_sidecar(manifest);
    const auto report =
        rlf::solstice::verify_image_generation_artifact_manifest(manifest);
    RLF_CHECK(report.manifest_self_hash_verified);
    RLF_CHECK(!report.schema_valid);
    RLF_CHECK(!report.artifact_set_complete);
    RLF_CHECK(!report.bundle_integrity_verified);
    RLF_CHECK(!report.failures.empty());
    std::filesystem::remove_all(directory);
}

RLF_TEST_CASE("Image-generation artifact manifest rejects unexpected rows and duplicate paths") {
    const auto directory = std::filesystem::temp_directory_path() /
        "rlf_imagegen_artifact_manifest_unexpected";
    const auto manifest = fixture_manifest_path(directory);
    write_text(directory / "unexpected.txt", "unexpected\n");
    std::string contents = read_text(manifest);
    contents += "unexpected\tunexpected.txt\t11\t" +
        rlf::core::sha256_hex(
            rlf::core::sha256_file(directory / "unexpected.txt")
        ) + "\n";
    write_text(manifest, contents);
    refresh_sidecar(manifest);
    auto report = rlf::solstice::verify_image_generation_artifact_manifest(manifest);
    RLF_CHECK(!report.schema_valid);
    RLF_CHECK(!report.artifact_set_complete);
    RLF_CHECK(!report.bundle_integrity_verified);
    RLF_CHECK(!report.failures.empty());

    (void)fixture_manifest_path(directory);
    contents = read_text(manifest);
    const std::string expected = "data_audit\tdata_audit.txt";
    const std::size_t offset = contents.find(expected);
    RLF_CHECK(offset != std::string::npos);
    contents.replace(offset, expected.size(), "data_audit\tledger.txt");
    write_text(manifest, contents);
    refresh_sidecar(manifest);
    report = rlf::solstice::verify_image_generation_artifact_manifest(manifest);
    RLF_CHECK(!report.schema_valid);
    RLF_CHECK(!report.artifact_set_complete);
    RLF_CHECK(!report.bundle_integrity_verified);
    RLF_CHECK(!report.failures.empty());
    std::filesystem::remove_all(directory);
}

RLF_TEST_CASE("Image-generation artifact manifest rejects symlink artifacts") {
    const auto directory = std::filesystem::temp_directory_path() /
        "rlf_imagegen_artifact_manifest_symlink";
    const auto manifest = fixture_manifest_path(directory);
    const auto target = directory / "ledger-target.txt";
    std::filesystem::rename(directory / "ledger.txt", target);
    std::error_code error;
    std::filesystem::create_symlink(target.filename(), directory / "ledger.txt", error);
#if defined(_WIN32)
    if (error) {
        std::filesystem::remove_all(directory);
        return;
    }
#else
    RLF_CHECK(!error);
#endif
    const auto report =
        rlf::solstice::verify_image_generation_artifact_manifest(manifest);
    RLF_CHECK(!report.artifact_integrity_verified);
    RLF_CHECK(!report.bundle_integrity_verified);
    RLF_CHECK(!report.failures.empty());
    std::filesystem::remove_all(directory);
}

RLF_TEST_CASE("Image-generation artifact manifest rejects directories and empty files") {
    const auto directory = std::filesystem::temp_directory_path() /
        "rlf_imagegen_artifact_manifest_nonregular";
    const auto manifest = fixture_manifest_path(directory);
    std::filesystem::remove(directory / "environment.txt");
    std::filesystem::create_directory(directory / "environment.txt");
    auto report = rlf::solstice::verify_image_generation_artifact_manifest(manifest);
    RLF_CHECK(!report.bundle_integrity_verified);
    RLF_CHECK(!report.failures.empty());

    std::filesystem::remove_all(directory / "environment.txt");
    write_text(directory / "environment.txt", "");
    report = rlf::solstice::verify_image_generation_artifact_manifest(manifest);
    RLF_CHECK(!report.bundle_integrity_verified);
    RLF_CHECK(!report.failures.empty());
    std::filesystem::remove_all(directory);
}

RLF_TEST_CASE("Image-generation artifact manifest requires its exact self-hash") {
    const auto directory = std::filesystem::temp_directory_path() /
        "rlf_imagegen_artifact_manifest_self_hash";
    const auto manifest = fixture_manifest_path(directory);
    std::string contents = read_text(manifest);
    contents.replace(contents.find("ledger.txt"), 6U, "Ledger");
    write_text(manifest, contents);
    const auto report =
        rlf::solstice::verify_image_generation_artifact_manifest(manifest);
    RLF_CHECK(!report.manifest_self_hash_verified);
    RLF_CHECK(!report.schema_valid);
    RLF_CHECK(!report.bundle_integrity_verified);
    RLF_CHECK(!report.failures.empty());
    std::filesystem::remove_all(directory);
}

RLF_TEST_CASE("Image-generation artifact manifest fully inspects its checkpoint") {
    const auto directory = std::filesystem::temp_directory_path() /
        "rlf_imagegen_artifact_manifest_bad_checkpoint";
    const auto manifest = fixture_manifest_path(directory);
    const auto checkpoint = directory / "model.rlfimg";
    write_text(checkpoint, "not-an-image-generation-checkpoint\n");

    std::string contents = read_text(manifest);
    const std::size_t row_begin = contents.find("checkpoint\t");
    RLF_CHECK(row_begin != std::string::npos);
    const std::size_t row_end = contents.find('\n', row_begin);
    RLF_CHECK(row_end != std::string::npos);
    const std::string replacement = "checkpoint\tmodel.rlfimg\t" +
        std::to_string(std::filesystem::file_size(checkpoint)) + "\t" +
        rlf::core::sha256_hex(rlf::core::sha256_file(checkpoint));
    contents.replace(row_begin, row_end - row_begin, replacement);
    write_text(manifest, contents);
    refresh_sidecar(manifest);

    const auto report =
        rlf::solstice::verify_image_generation_artifact_manifest(manifest);
    RLF_CHECK(report.schema_valid);
    RLF_CHECK(report.manifest_self_hash_verified);
    RLF_CHECK(!report.checkpoint_verified);
    RLF_CHECK(!report.bundle_integrity_verified);
    RLF_CHECK(!report.origin_authenticated);
    RLF_CHECK(!report.state_of_art_claim_proven);
    RLF_CHECK(!report.failures.empty());
    std::filesystem::remove_all(directory);
}
