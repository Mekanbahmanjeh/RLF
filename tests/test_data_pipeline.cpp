#include "test_framework.hpp"

#include "rlf/core/sha256.hpp"
#include "rlf/solstice/checkpoint.hpp"
#include "rlf/solstice/data_pipeline.hpp"
#include "rlf/solstice/solstice_model.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("unable to create test data file");
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    output.close();
}

[[nodiscard]] std::string ledger_row(
    const std::string& shard_id,
    const std::string& kind,
    const std::string& split,
    const std::string& format,
    const std::string& path,
    const std::string& sha256,
    const std::string& evaluation_family = "none"
) {
    return shard_id + '\t' + kind + '\t' + split +
        "\ttext\ten\ttest\t" + format + '\t' + path +
        "\tlocal:test-fixture\tCC0-1.0\t2026-07-20\t" + sha256 +
        "\ttest-v1\tnone\t" + evaluation_family + "\ttrue\n";
}

}  // namespace

RLF_TEST_CASE("SHA-256 matches standard empty and abc vectors") {
    RLF_CHECK(rlf::core::sha256_hex(rlf::core::sha256("")) ==
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    RLF_CHECK(rlf::core::sha256_hex(rlf::core::sha256("abc")) ==
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

RLF_TEST_CASE("Ranged SHA-256 streams an exact bounded file region") {
    const auto path = std::filesystem::temp_directory_path() /
        "rlf_sha256_range_test.bin";
    write_text(path, "xabcx");
    RLF_CHECK(rlf::core::sha256_file_range(path, 1U, 3U) ==
              rlf::core::sha256("abc"));
    RLF_CHECK(rlf::core::sha256_file_range(path, 5U, 0U) ==
              rlf::core::sha256(""));
    bool rejected = false;
    try {
        static_cast<void>(rlf::core::sha256_file_range(path, 4U, 2U));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    RLF_CHECK(rejected);
    std::filesystem::remove(path);
}

RLF_TEST_CASE("data ledger verifies provenance checksums and disjoint records") {
    const auto directory = std::filesystem::temp_directory_path() /
        "rlf_solstice_data_audit_valid";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const auto train = directory / "train.tsv";
    const auto evaluation = directory / "evaluation.tsv";
    const auto ledger_path = directory / "ledger.tsv";
    write_text(train, "coding\tsoftware\talpha parser fault\treproduce isolate patch\tguard input\t1.0\n");
    write_text(evaluation, "science\tbiology\tzephyr genome question\tmeasure compare verify\treport evidence\t1.0\n");
    write_text(
        ledger_path,
        ledger_row("train_instructions", "instruction", "train", "tsv", "train.tsv",
                   rlf::core::sha256_hex(rlf::core::sha256_file(train))) +
        ledger_row("eval_instructions", "instruction", "evaluation", "tsv", "evaluation.tsv",
                   rlf::core::sha256_hex(rlf::core::sha256_file(evaluation)), "private_holdout")
    );
    const auto ledger = rlf::solstice::load_data_ledger(ledger_path);
    rlf::solstice::DataAuditOptions options;
    options.near_duplicate_hamming_distance = 0U;
    const auto report = rlf::solstice::audit_data_ledger(ledger, options);
    RLF_CHECK(report.valid);
    RLF_CHECK(report.shards == 2U);
    RLF_CHECK(report.records == 2U);
    RLF_CHECK(report.checksum_mismatches == 0U);
    RLF_CHECK(report.train_evaluation_exact_collisions == 0U);
    rlf::solstice::DataAuditOptions bounded_options;
    bounded_options.maximum_text_shard_bytes = 1U;
    bounded_options.maximum_train_shard_bytes = 1U;
    const auto oversized = rlf::solstice::audit_data_ledger(
        ledger, bounded_options
    );
    RLF_CHECK(!oversized.valid);
    RLF_CHECK(oversized.oversized_train_shards == 1U);
    std::filesystem::remove_all(directory);
}

RLF_TEST_CASE("scalable prompt audit partitions exact fingerprints on disk") {
    const auto directory = std::filesystem::temp_directory_path() /
        "rlf_scalable_prompt_audit";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const auto train = directory / "train.txt";
    const auto evaluation = directory / "evaluation.txt";
    const auto ledger_path = directory / "ledger.tsv";
    write_text(train, "crimson pigment on illuminated canvas\n");
    write_text(evaluation, "orbital mechanics predicts satellite motion\n");
    write_text(
        ledger_path,
        ledger_row(
            "prompt_train", "text", "train", "text_lines", "train.txt",
            rlf::core::sha256_hex(rlf::core::sha256_file(train))
        ) +
        ledger_row(
            "prompt_eval", "text", "evaluation", "text_lines",
            "evaluation.txt",
            rlf::core::sha256_hex(rlf::core::sha256_file(evaluation)),
            "private_prompt_holdout"
        )
    );
    auto ledger = rlf::solstice::load_data_ledger(ledger_path);
    auto report = rlf::solstice::audit_prompt_language_ledger_scalable(ledger);
    RLF_CHECK(report.valid);
    RLF_CHECK(report.audit_strategy == "disk_partitioned_prompt_language");
    RLF_CHECK(report.records == 2U);
    RLF_CHECK(report.exact_dedup_scratch_bytes == 66U);
    RLF_CHECK(report.peak_exact_bucket_records >= 1U);
    RLF_CHECK(report.near_reference_records == 1U);
    RLF_CHECK(report.audited_shards.size() == 2U);
    RLF_CHECK(report.train_evaluation_exact_collisions == 0U);

    write_text(evaluation, "crimson pigment on illuminated canvas\n");
    write_text(
        ledger_path,
        ledger_row(
            "prompt_train", "text", "train", "text_lines", "train.txt",
            rlf::core::sha256_hex(rlf::core::sha256_file(train))
        ) +
        ledger_row(
            "prompt_eval", "text", "evaluation", "text_lines",
            "evaluation.txt",
            rlf::core::sha256_hex(rlf::core::sha256_file(evaluation)),
            "private_prompt_holdout"
        )
    );
    ledger = rlf::solstice::load_data_ledger(ledger_path);
    report = rlf::solstice::audit_prompt_language_ledger_scalable(ledger);
    RLF_CHECK(!report.valid);
    RLF_CHECK(report.train_evaluation_exact_collisions == 1U);

    std::string many_records;
    for (std::size_t index = 0U; index < 300U; ++index) {
        many_records += "unique prompt audit record " +
            std::to_string(index) + " token\n";
    }
    write_text(train, many_records);
    write_text(evaluation, "heldout unrelated phrase\n");
    write_text(
        ledger_path,
        ledger_row(
            "prompt_train", "text", "train", "text_lines", "train.txt",
            rlf::core::sha256_hex(rlf::core::sha256_file(train))
        ) +
        ledger_row(
            "prompt_eval", "text", "evaluation", "text_lines",
            "evaluation.txt",
            rlf::core::sha256_hex(rlf::core::sha256_file(evaluation)),
            "private_prompt_holdout"
        )
    );
    ledger = rlf::solstice::load_data_ledger(ledger_path);
    rlf::solstice::DataAuditOptions bounded;
    bounded.maximum_records = 400U;
    bounded.maximum_exact_bucket_records = 1U;
    RLF_CHECK_THROWS_AS(
        rlf::solstice::audit_prompt_language_ledger_scalable(ledger, bounded),
        std::runtime_error
    );
    std::filesystem::remove_all(directory);
}

RLF_TEST_CASE("data ledger fails closed on checksum and near-duplicate contamination") {
    const auto directory = std::filesystem::temp_directory_path() /
        "rlf_solstice_data_audit_reject";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const auto train = directory / "train.txt";
    const auto evaluation = directory / "evaluation.txt";
    const auto ledger_path = directory / "ledger.tsv";
    std::string common;
    for (std::size_t index = 0U; index < 80U; ++index) {
        common += "token" + std::to_string(index) + ' ';
    }
    write_text(train, common + "training_tail\n");
    write_text(evaluation, common + "evaluation_tail\n");
    write_text(
        ledger_path,
        ledger_row("train_text", "text", "train", "text_lines", "train.txt",
                   rlf::core::sha256_hex(rlf::core::sha256_file(train))) +
        ledger_row("eval_text", "text", "evaluation", "text_lines", "evaluation.txt",
                   rlf::core::sha256_hex(rlf::core::sha256_file(evaluation)), "private_holdout")
    );
    const auto ledger = rlf::solstice::load_data_ledger(ledger_path);
    const auto contaminated = rlf::solstice::audit_data_ledger(ledger);
    RLF_CHECK(!contaminated.valid);
    RLF_CHECK(contaminated.train_evaluation_near_collisions >= 1U);
    write_text(train, common + "tampered_tail\n");
    const auto tampered = rlf::solstice::audit_data_ledger(ledger);
    RLF_CHECK(!tampered.valid);
    RLF_CHECK(tampered.checksum_mismatches == 1U);
    std::filesystem::remove_all(directory);
}

RLF_TEST_CASE("data audit reports bounded near and solution-template redundancy") {
    const auto directory = std::filesystem::temp_directory_path() /
        "rlf_solstice_hierarchical_dedup_metrics";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const auto train = directory / "train.txt";
    const auto ledger_path = directory / "ledger.tsv";
    std::string common;
    for (std::size_t index = 0U; index < 80U; ++index) {
        common += "shared" + std::to_string(index) + ' ';
    }
    write_text(
        train,
        common + "solve item 100 with guarded addition\n" +
        common + "solve item 200 with guarded addition\n"
    );
    write_text(ledger_path, ledger_row(
        "train_text", "text", "train", "text_lines", "train.txt",
        rlf::core::sha256_hex(rlf::core::sha256_file(train))
    ));
    rlf::solstice::DataAuditOptions options;
    options.maximum_token_hash_cache_entries = 4U;
    const auto report = rlf::solstice::audit_data_ledger(
        rlf::solstice::load_data_ledger(ledger_path), options
    );
    RLF_CHECK(report.valid);
    RLF_CHECK(report.within_split_duplicates == 0U);
    RLF_CHECK(report.within_split_near_duplicate_records == 1U);
    RLF_CHECK(report.within_split_solution_template_groups == 1U);
    RLF_CHECK(report.train_evaluation_solution_template_groups == 0U);
    RLF_CHECK(report.token_hash_cache_entries == 4U);
    RLF_CHECK(report.token_hash_cache_maximum_entries == 4U);
    RLF_CHECK(report.token_hash_cache_hits > 0U);
    RLF_CHECK(report.token_hash_cache_misses > report.token_hash_cache_entries);
    std::ostringstream json;
    rlf::solstice::write_data_audit_json(json, report);
    RLF_CHECK(json.str().find("approximate, informational, no automatic deletion") !=
        std::string::npos);
    options.maximum_token_hash_cache_entries = 0U;
    const auto uncached = rlf::solstice::audit_data_ledger(
        rlf::solstice::load_data_ledger(ledger_path), options
    );
    RLF_CHECK(uncached.valid);
    RLF_CHECK(uncached.within_split_near_duplicate_records ==
        report.within_split_near_duplicate_records);
    RLF_CHECK(uncached.within_split_solution_template_groups ==
        report.within_split_solution_template_groups);
    RLF_CHECK(uncached.token_hash_cache_entries == 0U);
    RLF_CHECK(uncached.token_hash_cache_hits == 0U);
    RLF_CHECK(uncached.token_hash_cache_misses > 0U);
    std::filesystem::remove_all(directory);
}

RLF_TEST_CASE("video frame shards audit every referenced frame and temporal field") {
    const auto directory = std::filesystem::temp_directory_path() /
        "rlf_solstice_video_data_audit";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory / "frames");
    const auto frame = directory / "frames" / "000001.ppm";
    const auto manifest = directory / "video.tsv";
    const auto ledger_path = directory / "ledger.tsv";
    write_text(frame, "P6\n1 1\n255\n\x10\x20\x30");
    const std::string frame_sha = rlf::core::sha256_hex(
        rlf::core::sha256_file(frame)
    );
    write_text(manifest,
        "clip-001\t0\t24\tframes/000001.ppm\t" + frame_sha +
        "\ta red cube rotates\tthe cube begins facing forward\n");
    write_text(
        ledger_path,
        ledger_row("video_train", "video", "train", "video_frames_tsv",
                   "video.tsv", rlf::core::sha256_hex(
                       rlf::core::sha256_file(manifest)))
    );
    const auto ledger = rlf::solstice::load_data_ledger(ledger_path);
    const auto report = rlf::solstice::audit_data_ledger(ledger);
    RLF_CHECK(report.valid);
    RLF_CHECK(report.records == 1U);
    RLF_CHECK(report.referenced_media_bytes == std::filesystem::file_size(frame));
    write_text(frame, "changed");
    const auto changed = rlf::solstice::audit_data_ledger(ledger);
    RLF_CHECK(!changed.valid);
    RLF_CHECK(changed.checksum_mismatches == 1U);
    std::filesystem::remove(frame);
    const auto missing = rlf::solstice::audit_data_ledger(ledger);
    RLF_CHECK(!missing.valid);
    RLF_CHECK(missing.missing_files == 1U);
    std::filesystem::remove_all(directory);
}

RLF_TEST_CASE("strict media audit requires and verifies image hashes") {
    const auto directory = std::filesystem::temp_directory_path() /
        "rlf_solstice_strict_image_hash";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const auto image = directory / "image.ppm";
    const auto manifest = directory / "vision.tsv";
    const auto ledger_path = directory / "ledger.tsv";
    write_text(image, "P6\n1 1\n255\n\x01\x02\x03");
    write_text(manifest, "image.ppm\ta tiny image\n");
    write_text(ledger_path, ledger_row(
        "vision_train", "vision", "train", "vision_tsv", "vision.tsv",
        rlf::core::sha256_hex(rlf::core::sha256_file(manifest))
    ));
    rlf::solstice::DataAuditOptions strict;
    strict.require_media_sha256 = true;
    RLF_CHECK(!rlf::solstice::audit_data_ledger(
        rlf::solstice::load_data_ledger(ledger_path), strict
    ).valid);
    write_text(
        manifest,
        "image.ppm\t" + rlf::core::sha256_hex(rlf::core::sha256_file(image)) +
            "\ta tiny image\n"
    );
    write_text(ledger_path, ledger_row(
        "vision_train", "vision", "train", "vision_tsv", "vision.tsv",
        rlf::core::sha256_hex(rlf::core::sha256_file(manifest))
    ));
    RLF_CHECK(rlf::solstice::audit_data_ledger(
        rlf::solstice::load_data_ledger(ledger_path), strict
    ).valid);
    write_text(image, "changed");
    const auto changed = rlf::solstice::audit_data_ledger(
        rlf::solstice::load_data_ledger(ledger_path), strict
    );
    RLF_CHECK(!changed.valid);
    RLF_CHECK(changed.checksum_mismatches == 1U);
    std::filesystem::remove_all(directory);
}

RLF_TEST_CASE("data audit rejects perceptually identical train and evaluation images") {
    const auto directory = std::filesystem::temp_directory_path() /
        "rlf_solstice_perceptual_contamination";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const auto train_image = directory / "train.ppm";
    const auto eval_image = directory / "eval.ppm";
    const auto train_manifest = directory / "train.tsv";
    const auto eval_manifest = directory / "eval.tsv";
    const auto ledger_path = directory / "ledger.tsv";
    std::string pixels = "P6\n3 2\n255\n";
    pixels.append(
        "\x00\x10\x20\x30\x40\x50\x60\x70\x80"
        "\x90\xa0\xb0\xc0\xd0\xe0\xf0\x20\x40",
        18U
    );
    write_text(train_image, pixels);
    write_text(eval_image, pixels);
    write_text(train_manifest,
        "train.ppm\t" + rlf::core::sha256_hex(rlf::core::sha256_file(train_image)) +
        "\ttraining color relation\n");
    write_text(eval_manifest,
        "eval.ppm\t" + rlf::core::sha256_hex(rlf::core::sha256_file(eval_image)) +
        "\theldout visual question\n");
    write_text(ledger_path,
        ledger_row("vision_train", "vision", "train", "vision_tsv", "train.tsv",
                   rlf::core::sha256_hex(rlf::core::sha256_file(train_manifest))) +
        ledger_row("vision_eval", "vision", "evaluation", "vision_tsv", "eval.tsv",
                   rlf::core::sha256_hex(rlf::core::sha256_file(eval_manifest)),
                   "private_visual_holdout"));
    rlf::solstice::DataAuditOptions options;
    options.require_media_sha256 = true;
    const auto report = rlf::solstice::audit_data_ledger(
        rlf::solstice::load_data_ledger(ledger_path), options
    );
    RLF_CHECK(!report.valid);
    RLF_CHECK(report.train_evaluation_perceptual_image_collisions == 1U);
    RLF_CHECK(!report.contamination_audit_passed);
    options.single_read_media_audit = false;
    const auto double_read = rlf::solstice::audit_data_ledger(
        rlf::solstice::load_data_ledger(ledger_path), options
    );
    RLF_CHECK(!double_read.valid);
    RLF_CHECK(!double_read.single_read_media_audit);
    RLF_CHECK(double_read.train_evaluation_perceptual_image_collisions ==
        report.train_evaluation_perceptual_image_collisions);
    std::filesystem::remove_all(directory);
}

RLF_TEST_CASE("Solstice version six checkpoint persists resumable shard provenance") {
    const auto directory = std::filesystem::temp_directory_path() /
        "rlf_solstice_training_resume";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const auto checkpoint = directory / "model.rlfsp";
    rlf::solstice::SolsticeModel model;
    model.record_completed_training_shard({
        "shard-001", "instruction",
        rlf::core::sha256_hex(rlf::core::sha256("shard")),
        rlf::core::sha256_hex(rlf::core::sha256("ledger")),
        "local:test", "CC0-1.0", 7U, 123U,
    });
    rlf::solstice::save_solstice_checkpoint(checkpoint, model);
    const auto summary = rlf::solstice::inspect_solstice_checkpoint(checkpoint);
    RLF_CHECK(summary.format_version == 6U);
    RLF_CHECK(summary.stats.completed_training_shards == 1U);
    auto restored = rlf::solstice::load_solstice_checkpoint(checkpoint);
    RLF_CHECK(restored.has_completed_training_shard(
        rlf::core::sha256_hex(rlf::core::sha256("shard"))
    ));
    RLF_CHECK(restored.stats().audited_training_records == 7U);
    RLF_CHECK(restored.stats().audited_training_bytes == 123U);
    RLF_CHECK_THROWS_AS(
        restored.record_completed_training_shard({
            "shard-001", "instruction",
            rlf::core::sha256_hex(rlf::core::sha256("changed")),
            rlf::core::sha256_hex(rlf::core::sha256("ledger")),
            "local:test", "CC0-1.0", 1U, 1U,
        }),
        std::invalid_argument
    );
    std::filesystem::remove_all(directory);
}

RLF_TEST_CASE("Solstice checkpoint refuses to replace a directory target") {
    const auto directory = std::filesystem::temp_directory_path() /
        "rlf_solstice_checkpoint_directory_target";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    rlf::solstice::SolsticeModel model;
    RLF_CHECK_THROWS_AS(
        rlf::solstice::save_solstice_checkpoint(directory, model),
        std::invalid_argument
    );
    RLF_CHECK(std::filesystem::is_directory(directory));
    std::filesystem::remove_all(directory);
}

RLF_TEST_CASE("streamed Solstice checkpoints are deterministic and reject corruption") {
    const auto directory = std::filesystem::temp_directory_path() /
        "rlf_solstice_streamed_checkpoint";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const auto first = directory / "first.rlfsp";
    const auto second = directory / "second.rlfsp";
    rlf::solstice::SolsticeModel model;
    model.bootstrap();
    model.train_dialogue("stream this state", "without a full payload buffer");
    rlf::solstice::save_solstice_checkpoint(first, model);
    rlf::solstice::save_solstice_checkpoint(second, model);
    RLF_CHECK(rlf::core::sha256_file(first) == rlf::core::sha256_file(second));

    rlf::solstice::SolsticeCheckpointLimits tight_limits;
    tight_limits.maximum_file_bytes = std::filesystem::file_size(first) - 1U;
    RLF_CHECK_THROWS_AS(
        rlf::solstice::load_solstice_checkpoint(first, tight_limits),
        std::runtime_error
    );

    std::fstream corrupt(first, std::ios::binary | std::ios::in | std::ios::out);
    corrupt.seekg(-1, std::ios::end);
    const int original = corrupt.get();
    if (original == std::char_traits<char>::eof()) {
        throw std::runtime_error("unable to read checkpoint corruption byte");
    }
    corrupt.seekp(-1, std::ios::end);
    corrupt.put(static_cast<char>(static_cast<unsigned char>(original) ^ 0x01U));
    corrupt.close();
    RLF_CHECK_THROWS_AS(
        rlf::solstice::load_solstice_checkpoint(first),
        std::runtime_error
    );
    std::filesystem::remove_all(directory);
}
