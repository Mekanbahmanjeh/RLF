#include "rlf/solstice/resonant_image_fabric.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rlf::solstice {
namespace {

constexpr double tau = 2.0 * std::numbers::pi_v<double>;
constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;
constexpr std::uint64_t golden_ratio = 0x9E3779B97F4A7C15ULL;

[[nodiscard]] std::uint64_t mix64(std::uint64_t value) noexcept {
    value ^= value >> 30U;
    value *= 0xBF58476D1CE4E5B9ULL;
    value ^= value >> 27U;
    value *= 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

void hash_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (unsigned int byte_index = 0U; byte_index < 8U; ++byte_index) {
        hash ^= (value >> (byte_index * 8U)) & 0xFFULL;
        hash *= fnv_prime;
    }
}

void hash_string(std::uint64_t& hash, const std::string_view value) noexcept {
    hash_u64(hash, static_cast<std::uint64_t>(value.size()));
    for (const char character : value) {
        hash ^= static_cast<std::uint64_t>(
            static_cast<unsigned char>(character)
        );
        hash *= fnv_prime;
    }
}

void hash_double(std::uint64_t& hash, const double value) noexcept {
    hash_u64(hash, std::bit_cast<std::uint64_t>(value));
}

void hash_float(std::uint64_t& hash, const float value) noexcept {
    hash_u64(
        hash,
        static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(value))
    );
}

[[nodiscard]] double unit_from_hash(const std::uint64_t value) noexcept {
    constexpr double inverse_two_to_the_53 =
        1.0 / static_cast<double>(1ULL << 53U);
    return static_cast<double>(value >> 11U) * inverse_two_to_the_53;
}

[[nodiscard]] std::uint64_t stable_text_hash(
    const std::string_view text
) noexcept {
    std::uint64_t hash = fnv_offset_basis;
    for (const char character : text) {
        hash ^= static_cast<std::uint64_t>(
            static_cast<unsigned char>(character)
        );
        hash *= fnv_prime;
    }
    return hash;
}

[[nodiscard]] std::vector<std::uint64_t> prompt_concept_hashes(
    const std::string_view prompt,
    const std::size_t maximum_concepts
) {
    std::vector<std::string> words;
    std::string current;
    const auto flush = [&]() {
        if (!current.empty()) {
            words.push_back(std::move(current));
            current.clear();
        }
    };
    for (const char raw_character : prompt) {
        const auto byte = static_cast<unsigned char>(raw_character);
        if (std::isalnum(byte) != 0 || byte >= 0x80U) {
            current.push_back(static_cast<char>(
                byte < 0x80U
                    ? static_cast<unsigned char>(std::tolower(byte))
                    : byte
            ));
        } else {
            flush();
        }
    }
    flush();

    static const std::unordered_set<std::string> stop_words{
        "a", "an", "and", "as", "at", "be", "by", "for", "from",
        "in", "into", "is", "it", "of", "on", "or", "please", "the",
        "then", "to", "with",
    };
    std::vector<std::uint64_t> concepts;
    concepts.reserve(std::min(maximum_concepts, words.size() * 2U));
    for (std::size_t index = 0U;
         index < words.size() && concepts.size() < maximum_concepts;
         ++index) {
        if (words[index].size() >= 2U && !stop_words.contains(words[index])) {
            concepts.push_back(stable_text_hash("word:" + words[index]));
        }
        if (index != 0U && concepts.size() < maximum_concepts &&
            !stop_words.contains(words[index - 1U]) &&
            !stop_words.contains(words[index])) {
            std::string bigram = words[index - 1U];
            bigram.push_back(' ');
            bigram += words[index];
            concepts.push_back(stable_text_hash("bigram:" + bigram));
        }
    }
    // Character n-grams keep retrieval open-vocabulary for inflection,
    // spelling variants, and minor label noise without an external neural
    // text encoder. They do not manufacture synonym knowledge: semantic
    // aliases still have to be grounded by the training corpus.
    for (const auto& word : words) {
        if (concepts.size() >= maximum_concepts) {
            break;
        }
        if (word.size() < 4U || stop_words.contains(word)) {
            continue;
        }
        for (std::size_t begin = 0U;
             begin + 3U <= word.size() && concepts.size() < maximum_concepts;
             ++begin) {
            concepts.push_back(stable_text_hash(
                "char3:" + word.substr(begin, 3U)
            ));
        }
    }
    std::sort(concepts.begin(), concepts.end());
    concepts.erase(std::unique(concepts.begin(), concepts.end()), concepts.end());
    if (concepts.size() > maximum_concepts) {
        concepts.resize(maximum_concepts);
    }
    return concepts;
}

[[nodiscard]] double concept_similarity(
    const std::span<const std::uint64_t> left,
    const std::span<const std::uint64_t> right
) noexcept {
    std::size_t left_index = 0U;
    std::size_t right_index = 0U;
    std::size_t intersection = 0U;
    while (left_index < left.size() && right_index < right.size()) {
        if (left[left_index] == right[right_index]) {
            ++intersection;
            ++left_index;
            ++right_index;
        } else if (left[left_index] < right[right_index]) {
            ++left_index;
        } else {
            ++right_index;
        }
    }
    const std::size_t union_count = left.size() + right.size() - intersection;
    return union_count == 0U
        ? 0.0
        : static_cast<double>(intersection) /
            static_cast<double>(union_count);
}

[[nodiscard]] std::uint64_t semantic_cell_key(
    const std::uint64_t concept_hash,
    const std::uint16_t x_bin,
    const std::uint16_t y_bin
) noexcept {
    return mix64(
        concept_hash ^ (static_cast<std::uint64_t>(x_bin) << 16U) ^
        static_cast<std::uint64_t>(y_bin) ^ 0x53454D414E544943ULL
    );
}

[[nodiscard]] std::size_t checked_patch_area(
    const ResonantImageConfig& config
) {
    if (config.patch_size == 0U ||
        config.patch_size >
            std::numeric_limits<std::size_t>::max() / config.patch_size) {
        throw std::invalid_argument("invalid resonant-image patch size");
    }
    return config.patch_size * config.patch_size;
}

void validate_config(const ResonantImageConfig& config) {
    const std::size_t patch_area = checked_patch_area(config);
    if (config.phase_redundancy == 0U ||
        config.coordinate_bins == 0U ||
        config.coordinate_bins >
            static_cast<std::size_t>(
                std::numeric_limits<std::uint16_t>::max()
            ) ||
        config.maximum_modes == 0U ||
        config.maximum_concept_bytes == 0U ||
        config.maximum_image_side == 0U ||
        config.maximum_image_pixels == 0U ||
        config.maximum_image_side >
            std::numeric_limits<std::size_t>::max() /
                config.coordinate_bins ||
        config.candidate_count == 0U ||
        config.active_count == 0U ||
        config.active_count > config.candidate_count ||
        config.maximum_settling_cycles == 0U ||
        config.maximum_trace_entries == 0U ||
        config.maximum_prompt_concepts == 0U ||
        config.maximum_semantic_candidates == 0U ||
        patch_area > std::numeric_limits<std::size_t>::max() / 3U ||
        patch_area * 3U >
            std::numeric_limits<std::size_t>::max() /
                config.phase_redundancy ||
        !std::isfinite(config.minimum_resonance) ||
        config.minimum_resonance < 0.0 || config.minimum_resonance > 1.0 ||
        !std::isfinite(config.minimum_semantic_similarity) ||
        config.minimum_semantic_similarity < 0.0 ||
        config.minimum_semantic_similarity > 1.0 ||
        !std::isfinite(config.semantic_resonance_weight) ||
        config.semantic_resonance_weight < 0.0 ||
        config.semantic_resonance_weight > 1.0 ||
        !std::isfinite(config.convergence_tolerance_radians) ||
        config.convergence_tolerance_radians < 0.0 ||
        !std::isfinite(config.settling_relaxation) ||
        config.settling_relaxation <= 0.0 ||
        config.settling_relaxation > 1.0 ||
        !std::isfinite(config.transformation_learning_rate) ||
        config.transformation_learning_rate <= 0.0 ||
        config.transformation_learning_rate > 1.0 ||
        !std::isfinite(config.context_learning_rate) ||
        config.context_learning_rate < 0.0 ||
        config.context_learning_rate > 1.0 ||
        !std::isfinite(config.confidence_learning_rate) ||
        config.confidence_learning_rate < 0.0 ||
        config.confidence_learning_rate > 1.0) {
        throw std::invalid_argument("invalid resonant-image configuration");
    }
}

[[nodiscard]] bool prompt_semantic_configs_equal(
    const PromptSemanticConfig& left,
    const PromptSemanticConfig& right
) noexcept {
    return left.phase_dimension == right.phase_dimension &&
        left.maximum_words == right.maximum_words &&
        left.maximum_word_bytes == right.maximum_word_bytes &&
        left.maximum_words_per_record == right.maximum_words_per_record &&
        left.context_window == right.context_window &&
        left.bucket_bits == right.bucket_bits &&
        left.maximum_expansions_per_word ==
            right.maximum_expansions_per_word &&
        left.minimum_support == right.minimum_support &&
        left.learning_rate == right.learning_rate &&
        left.seed == right.seed;
}

[[nodiscard]] ResonantImageOperationStats subtract_stats(
    const ResonantImageOperationStats& after,
    const ResonantImageOperationStats& before
) noexcept {
    return {
        .training_examples = after.training_examples - before.training_examples,
        .training_patches = after.training_patches - before.training_patches,
        .modes_created = after.modes_created - before.modes_created,
        .local_mode_updates =
            after.local_mode_updates - before.local_mode_updates,
        .generation_calls = after.generation_calls - before.generation_calls,
        .generated_patches =
            after.generated_patches - before.generated_patches,
        .sparse_bucket_lookups =
            after.sparse_bucket_lookups - before.sparse_bucket_lookups,
        .resonance_evaluations =
            after.resonance_evaluations - before.resonance_evaluations,
        .active_mode_applications =
            after.active_mode_applications - before.active_mode_applications,
        .settling_cycles = after.settling_cycles - before.settling_cycles,
        .unresolved_patch_transformations =
            after.unresolved_patch_transformations -
            before.unresolved_patch_transformations,
        .decoded_channels = after.decoded_channels - before.decoded_channels,
        .semantic_bucket_lookups =
            after.semantic_bucket_lookups - before.semantic_bucket_lookups,
        .semantic_candidates_scored =
            after.semantic_candidates_scored - before.semantic_candidates_scored,
        .semantic_matches = after.semantic_matches - before.semantic_matches,
    };
}

[[nodiscard]] core::PhaseVector circular_interpolate(
    const core::PhaseVector& current,
    const core::PhaseVector& target,
    const double amount,
    frontier::FrontierComputeBackend* backend = nullptr
) {
    if (amount <= 0.0) {
        return current;
    }
    if (amount >= 1.0) {
        return target;
    }
    if (backend != nullptr &&
        backend->kind() == frontier::FrontierBackendKind::cuda) {
        std::vector<float> current_cartesian;
        std::vector<float> target_cartesian;
        current_cartesian.reserve(current.size() * 2U);
        target_cartesian.reserve(target.size() * 2U);
        for (std::size_t index = 0U; index < current.size(); ++index) {
            current_cartesian.push_back(static_cast<float>(
                std::cos(static_cast<double>(current[index]))
            ));
            current_cartesian.push_back(static_cast<float>(
                std::sin(static_cast<double>(current[index]))
            ));
            target_cartesian.push_back(static_cast<float>(
                std::cos(static_cast<double>(target[index]))
            ));
            target_cartesian.push_back(static_cast<float>(
                std::sin(static_cast<double>(target[index]))
            ));
        }
        backend->local_average_update(
            current_cartesian,
            target_cartesian,
            static_cast<float>(amount)
        );
        std::vector<float> angles;
        angles.reserve(current.size());
        for (std::size_t index = 0U; index < current.size(); ++index) {
            angles.push_back(core::PhaseVector::normalize_angle(
                static_cast<float>(std::atan2(
                    static_cast<double>(current_cartesian[index * 2U + 1U]),
                    static_cast<double>(current_cartesian[index * 2U])
                ))
            ));
        }
        return core::PhaseVector(std::move(angles));
    }
    const std::vector<core::PhaseVector> vectors{current, target};
    const std::vector<float> weights{
        static_cast<float>(1.0 - amount),
        static_cast<float>(amount),
    };
    return core::PhaseVector::weighted_circular_average(vectors, weights);
}

[[nodiscard]] std::size_t checked_rgb_size(
    const std::size_t width,
    const std::size_t height
) {
    if (width == 0U || height == 0U ||
        width > std::numeric_limits<std::size_t>::max() / height ||
        width * height > std::numeric_limits<std::size_t>::max() / 3U) {
        throw std::invalid_argument("invalid resonant-image dimensions");
    }
    return width * height * 3U;
}

}  // namespace

ResonantImageConfig make_resonant_image_profile_config(
    const ImageGenerationProfile profile
) {
    ResonantImageConfig config;
    if (profile == ImageGenerationProfile::a100_80g ||
        profile == ImageGenerationProfile::v100_32g) {
        config.patch_size = 8U;
        config.phase_redundancy = 4U;
        config.coordinate_bins = 32U;
        config.maximum_modes = 48'000'000U;
        config.maximum_concept_bytes = 16'384U;
        config.maximum_image_side = 4'096U;
        config.maximum_image_pixels = 16U * 1024U * 1024U;
        config.candidate_count = 64U;
        config.active_count = 8U;
        config.maximum_settling_cycles = 16U;
        config.maximum_trace_entries = 4'000'000U;
    }
    validate_config(config);
    return config;
}

std::vector<std::string> parse_resonant_image_prompt(
    const std::string_view prompt
) {
    std::string normalized(prompt);
    for (char& character : normalized) {
        const auto byte = static_cast<unsigned char>(character);
        if (byte < 0x80U) {
            character = static_cast<char>(std::tolower(byte));
        }
    }
    static constexpr std::array<std::string_view, 4U> separators{
        " and then ", " followed by ", " then ", ";",
    };
    std::vector<std::string> operations;
    std::size_t begin = 0U;
    while (begin < prompt.size()) {
        std::size_t next = std::string_view::npos;
        std::size_t separator_size = 0U;
        for (const std::string_view separator : separators) {
            const std::size_t found = normalized.find(separator, begin);
            if (found < next) {
                next = found;
                separator_size = separator.size();
            }
        }
        const std::size_t end = next == std::string_view::npos
            ? prompt.size()
            : next;
        std::size_t trimmed_begin = begin;
        std::size_t trimmed_end = end;
        while (trimmed_begin < trimmed_end &&
               std::isspace(static_cast<unsigned char>(
                   prompt[trimmed_begin]
               )) != 0) {
            ++trimmed_begin;
        }
        while (trimmed_end > trimmed_begin &&
               std::isspace(static_cast<unsigned char>(
                   prompt[trimmed_end - 1U]
               )) != 0) {
            --trimmed_end;
        }
        if (trimmed_begin < trimmed_end) {
            operations.emplace_back(prompt.substr(
                trimmed_begin,
                trimmed_end - trimmed_begin
            ));
        }
        if (next == std::string_view::npos) {
            break;
        }
        begin = next + separator_size;
    }
    if (operations.empty() && !prompt.empty()) {
        operations.emplace_back(prompt);
    }
    return operations;
}

bool resonant_image_profile_config_matches(
    const ImageGenerationProfile profile,
    const ResonantImageConfig& config
) noexcept {
    try {
        const ResonantImageConfig expected =
            make_resonant_image_profile_config(profile);
        return config.patch_size == expected.patch_size &&
            config.phase_redundancy == expected.phase_redundancy &&
            config.coordinate_bins == expected.coordinate_bins &&
            config.maximum_modes == expected.maximum_modes &&
            config.maximum_concept_bytes == expected.maximum_concept_bytes &&
            config.maximum_image_side == expected.maximum_image_side &&
            config.maximum_image_pixels == expected.maximum_image_pixels &&
            config.candidate_count == expected.candidate_count &&
            config.active_count == expected.active_count &&
            config.maximum_settling_cycles == expected.maximum_settling_cycles &&
            config.maximum_trace_entries == expected.maximum_trace_entries &&
            config.maximum_prompt_concepts == expected.maximum_prompt_concepts &&
            config.maximum_semantic_candidates ==
                expected.maximum_semantic_candidates &&
            config.minimum_resonance == expected.minimum_resonance &&
            config.minimum_semantic_similarity ==
                expected.minimum_semantic_similarity &&
            config.semantic_resonance_weight ==
                expected.semantic_resonance_weight &&
            config.convergence_tolerance_radians ==
                expected.convergence_tolerance_radians &&
            config.settling_relaxation == expected.settling_relaxation &&
            config.transformation_learning_rate ==
                expected.transformation_learning_rate &&
            config.context_learning_rate == expected.context_learning_rate &&
            config.confidence_learning_rate ==
                expected.confidence_learning_rate &&
            prompt_semantic_configs_equal(
                config.prompt_semantics, expected.prompt_semantics
            );
    } catch (...) {
        return false;
    }
}

bool ResonantImageFabric::CellKey::operator<(
    const CellKey& other
) const noexcept {
    return std::tie(semantic_label, y_bin, x_bin) <
        std::tie(other.semantic_label, other.y_bin, other.x_bin);
}

ResonantImageFabric::ResonantImageFabric(ResonantImageConfig config)
    : config_(std::move(config)),
      prompt_semantics_(config_.prompt_semantics),
      backend_(frontier::make_frontier_backend(
          frontier::FrontierBackendKind::optimized_cpu
      )) {
    validate_config(config_);
    const std::size_t patch_area = checked_patch_area(config_);
    phase_dimension_ =
        patch_area * 3U * config_.phase_redundancy;
    carriers_.reserve(phase_dimension_);
    carrier_signs_.reserve(phase_dimension_);
    for (std::size_t index = 0U; index < phase_dimension_; ++index) {
        const std::uint64_t mixed = mix64(
            config_.seed +
            (golden_ratio * (static_cast<std::uint64_t>(index) + 1ULL))
        );
        carriers_.push_back(static_cast<float>(unit_from_hash(mixed) * tau));
        carrier_signs_.push_back((mixed & 1ULL) == 0ULL ? 1 : -1);
    }
}

const ResonantImageConfig& ResonantImageFabric::config() const noexcept {
    return config_;
}

std::span<const ResonantImageMode> ResonantImageFabric::modes() const noexcept {
    return modes_;
}

ResonantImageOperationStats
ResonantImageFabric::operation_stats() const noexcept {
    return operation_stats_;
}

void ResonantImageFabric::train_prompt_language_record(
    const std::string_view text
) {
    prompt_semantics_.train_record(text);
    semantic_index_dirty_ = true;
}

const PromptSemanticFabric& ResonantImageFabric::prompt_semantics() const
    noexcept {
    return prompt_semantics_;
}

std::vector<std::uint64_t> ResonantImageFabric::prompt_concepts(
    const std::string_view prompt
) const {
    std::vector<std::uint64_t> concepts = prompt_concept_hashes(
        prompt, config_.maximum_prompt_concepts
    );
    const auto learned = prompt_semantics_.semantic_concept_hashes(
        prompt, config_.maximum_prompt_concepts
    );
    std::unordered_set<std::uint64_t> seen(concepts.begin(), concepts.end());
    for (const auto value : learned) {
        if (concepts.size() >= config_.maximum_prompt_concepts) {
            break;
        }
        if (seen.insert(value).second) {
            concepts.push_back(value);
        }
    }
    std::sort(concepts.begin(), concepts.end());
    return concepts;
}

void ResonantImageFabric::rebuild_semantic_index() {
    semantic_cell_index_.clear();
    for (std::size_t index = 0U; index < modes_.size(); ++index) {
        index_semantic_mode(index);
    }
    semantic_index_dirty_ = false;
}

void ResonantImageFabric::ensure_semantic_index() {
    if (semantic_index_dirty_) {
        rebuild_semantic_index();
    }
}

void ResonantImageFabric::set_backend(
    const frontier::FrontierBackendKind kind
) {
    auto backend = frontier::make_frontier_backend(kind);
    if (!backend->capabilities().available ||
        !backend->capabilities().supports_local_update) {
        throw std::runtime_error(
            "requested resonant-image backend is unavailable"
        );
    }
    backend_ = std::move(backend);
}

frontier::FrontierBackendKind ResonantImageFabric::backend_kind() const noexcept {
    return backend_->kind();
}

frontier::BackendOperationStats
ResonantImageFabric::backend_operation_stats() const noexcept {
    return backend_->operation_stats();
}

void ResonantImageFabric::reset_operation_stats() noexcept {
    operation_stats_ = {};
}

void ResonantImageFabric::validate_image(const ImageData& image) const {
    const std::size_t rgb_size = checked_rgb_size(image.width, image.height);
    if (image.width > config_.maximum_image_side ||
        image.height > config_.maximum_image_side ||
        image.width * image.height > config_.maximum_image_pixels ||
        image.rgb.size() != rgb_size ||
        image.width % config_.patch_size != 0U ||
        image.height % config_.patch_size != 0U) {
        throw std::invalid_argument(
            "image violates resonant-image dimensions or patch alignment"
        );
    }
}

void ResonantImageFabric::validate_concept(
    const std::string_view semantic_label
) const {
    if (semantic_label.empty() ||
        semantic_label.size() > config_.maximum_concept_bytes) {
        throw std::invalid_argument(
            "resonant-image concept must be non-empty and within its byte limit"
        );
    }
}

core::PhaseVector ResonantImageFabric::encode_patch(
    const ImageData& image,
    const std::size_t patch_x,
    const std::size_t patch_y
) const {
    validate_image(image);
    const std::size_t patch_columns = image.width / config_.patch_size;
    const std::size_t patch_rows = image.height / config_.patch_size;
    if (patch_x >= patch_columns || patch_y >= patch_rows) {
        throw std::out_of_range("resonant-image patch coordinate is out of range");
    }

    std::vector<float> angles;
    angles.reserve(phase_dimension_);
    std::size_t phase_index = 0U;
    for (std::size_t local_y = 0U;
         local_y < config_.patch_size;
         ++local_y) {
        const std::size_t pixel_y = patch_y * config_.patch_size + local_y;
        for (std::size_t local_x = 0U;
             local_x < config_.patch_size;
             ++local_x) {
            const std::size_t pixel_x =
                patch_x * config_.patch_size + local_x;
            const std::size_t rgb_offset =
                (pixel_y * image.width + pixel_x) * 3U;
            for (std::size_t channel = 0U; channel < 3U; ++channel) {
                const double color_angle =
                    static_cast<double>(image.rgb[rgb_offset + channel]) *
                    tau / 256.0;
                for (std::size_t lane = 0U;
                     lane < config_.phase_redundancy;
                     ++lane) {
                    const double signed_color =
                        static_cast<double>(carrier_signs_[phase_index]) *
                        color_angle;
                    angles.push_back(core::PhaseVector::normalize_angle(
                        static_cast<float>(
                            static_cast<double>(carriers_[phase_index]) +
                            signed_color
                        )
                    ));
                    ++phase_index;
                }
            }
        }
    }
    return core::PhaseVector(std::move(angles));
}

ImageData ResonantImageFabric::decode_patch(
    const core::PhaseVector& state
) const {
    if (state.size() != phase_dimension_) {
        throw std::invalid_argument(
            "resonant-image state dimension does not match the patch encoder"
        );
    }
    const std::size_t patch_area = checked_patch_area(config_);
    ImageData image{
        .width = config_.patch_size,
        .height = config_.patch_size,
        .rgb = std::vector<std::uint8_t>(patch_area * 3U),
    };

    std::size_t phase_index = 0U;
    for (std::size_t component = 0U;
         component < image.rgb.size();
         ++component) {
        double cosine_sum = 0.0;
        double sine_sum = 0.0;
        for (std::size_t lane = 0U;
             lane < config_.phase_redundancy;
             ++lane) {
            const double raw_angle = carrier_signs_[phase_index] > 0
                ? static_cast<double>(state[phase_index]) -
                    static_cast<double>(carriers_[phase_index])
                : static_cast<double>(carriers_[phase_index]) -
                    static_cast<double>(state[phase_index]);
            cosine_sum += std::cos(raw_angle);
            sine_sum += std::sin(raw_angle);
            ++phase_index;
        }
        double mean_angle = std::atan2(sine_sum, cosine_sum);
        if (mean_angle < 0.0) {
            mean_angle += tau;
        }
        const std::uint64_t quantized = static_cast<std::uint64_t>(
            std::llround(mean_angle * 256.0 / tau)
        ) % 256ULL;
        image.rgb[component] = static_cast<std::uint8_t>(quantized);
    }
    return image;
}

std::pair<std::uint16_t, std::uint16_t>
ResonantImageFabric::coordinate_bin(
    const std::size_t patch_x,
    const std::size_t patch_y,
    const std::size_t patch_columns,
    const std::size_t patch_rows
) const {
    if (patch_columns == 0U || patch_rows == 0U ||
        patch_x >= patch_columns || patch_y >= patch_rows) {
        throw std::invalid_argument("invalid resonant-image patch grid");
    }
    const std::size_t x_bin = std::min(
        config_.coordinate_bins - 1U,
        patch_x * config_.coordinate_bins / patch_columns
    );
    const std::size_t y_bin = std::min(
        config_.coordinate_bins - 1U,
        patch_y * config_.coordinate_bins / patch_rows
    );
    return {
        static_cast<std::uint16_t>(x_bin),
        static_cast<std::uint16_t>(y_bin),
    };
}

core::PhaseVector ResonantImageFabric::position_context(
    const std::uint16_t x_bin,
    const std::uint16_t y_bin
) const {
    const std::uint64_t packed =
        (static_cast<std::uint64_t>(x_bin) << 16U) |
        static_cast<std::uint64_t>(y_bin);
    std::vector<float> angles;
    angles.reserve(phase_dimension_);
    for (std::size_t dimension = 0U;
         dimension < phase_dimension_;
         ++dimension) {
        const std::uint64_t mixed = mix64(
            config_.seed ^ packed ^
            (golden_ratio * (static_cast<std::uint64_t>(dimension) + 1ULL))
        );
        angles.push_back(static_cast<float>(unit_from_hash(mixed) * tau));
    }
    return core::PhaseVector(std::move(angles));
}

std::vector<ResonantImageFabric::RetrievedMode>
ResonantImageFabric::retrieve(
    const std::string_view semantic_label,
    const std::uint16_t x_bin,
    const std::uint16_t y_bin,
    const core::PhaseVector& context
) {
    ensure_semantic_index();
    ++operation_stats_.sparse_bucket_lookups;
    const auto cell = cell_index_.find(CellKey{
        std::string(semantic_label),
        x_bin,
        y_bin,
    });
    std::vector<RetrievedMode> retrieved;
    if (cell != cell_index_.end()) {
        const std::size_t mode_index = cell->second;
        const ResonantImageMode& mode = modes_.at(mode_index);
        if (mode.resonant_mode.enabled) {
            ++operation_stats_.resonance_evaluations;
            retrieved.push_back({
                .mode_index = mode_index,
                .resonance = mode.resonant_mode.resonance(context),
                .semantic_similarity = 1.0,
            });
        }
        return retrieved;
    }

    const std::vector<std::uint64_t> query_concepts = prompt_concepts(
        semantic_label
    );
    std::unordered_set<std::size_t> candidate_indices;
    candidate_indices.reserve(config_.maximum_semantic_candidates * 2U);
    for (const std::uint64_t concept_hash : query_concepts) {
        ++operation_stats_.semantic_bucket_lookups;
        const auto bucket = semantic_cell_index_.find(
            semantic_cell_key(concept_hash, x_bin, y_bin)
        );
        if (bucket == semantic_cell_index_.end()) {
            continue;
        }
        for (const std::size_t mode_index : bucket->second) {
            candidate_indices.insert(mode_index);
            if (candidate_indices.size() >=
                config_.maximum_semantic_candidates) {
                break;
            }
        }
        if (candidate_indices.size() >= config_.maximum_semantic_candidates) {
            break;
        }
    }
    retrieved.reserve(candidate_indices.size());
    for (const std::size_t mode_index : candidate_indices) {
        const ResonantImageMode& mode = modes_.at(mode_index);
        if (!mode.resonant_mode.enabled) {
            continue;
        }
        ++operation_stats_.semantic_candidates_scored;
        const std::vector<std::uint64_t> mode_concepts = prompt_concepts(
            mode.semantic_label
        );
        const double semantic = concept_similarity(
            query_concepts,
            mode_concepts
        );
        if (semantic < config_.minimum_semantic_similarity) {
            continue;
        }
        ++operation_stats_.resonance_evaluations;
        ++operation_stats_.semantic_matches;
        retrieved.push_back({
            .mode_index = mode_index,
            .resonance = mode.resonant_mode.resonance(context),
            .semantic_similarity = semantic,
        });
    }
    std::sort(
        retrieved.begin(),
        retrieved.end(),
        [this](const RetrievedMode& left, const RetrievedMode& right) {
            const double left_score =
                config_.semantic_resonance_weight * left.semantic_similarity +
                (1.0 - config_.semantic_resonance_weight) * left.resonance;
            const double right_score =
                config_.semantic_resonance_weight * right.semantic_similarity +
                (1.0 - config_.semantic_resonance_weight) * right.resonance;
            if (left_score != right_score) {
                return left_score > right_score;
            }
            return modes_[left.mode_index].resonant_mode.id <
                modes_[right.mode_index].resonant_mode.id;
        }
    );
    if (retrieved.size() > config_.candidate_count) {
        retrieved.resize(config_.candidate_count);
    }
    return retrieved;
}

void ResonantImageFabric::index_semantic_mode(const std::size_t mode_index) {
    const ResonantImageMode& mode = modes_.at(mode_index);
    for (const std::uint64_t concept_hash : prompt_concepts(
             mode.semantic_label
         )) {
        semantic_cell_index_[semantic_cell_key(
            concept_hash,
            mode.x_bin,
            mode.y_bin
        )].push_back(mode_index);
    }
}

ResonantImageTrainingResult ResonantImageFabric::train(
    const ResonantImageTrainingPair& example
) {
    validate_image(example.source);
    validate_image(example.target);
    validate_concept(example.semantic_label);
    if (example.source.width != example.target.width ||
        example.source.height != example.target.height) {
        throw std::invalid_argument(
            "resonant-image source and target dimensions must match"
        );
    }

    const std::size_t patch_columns =
        example.source.width / config_.patch_size;
    const std::size_t patch_rows =
        example.source.height / config_.patch_size;
    const std::size_t patch_count = patch_columns * patch_rows;

    std::set<CellKey> missing_cells;
    for (std::size_t patch_y = 0U; patch_y < patch_rows; ++patch_y) {
        for (std::size_t patch_x = 0U; patch_x < patch_columns; ++patch_x) {
            const auto [x_bin, y_bin] = coordinate_bin(
                patch_x,
                patch_y,
                patch_columns,
                patch_rows
            );
            CellKey key{example.semantic_label, x_bin, y_bin};
            if (!cell_index_.contains(key)) {
                missing_cells.insert(std::move(key));
            }
        }
    }
    if (missing_cells.size() > config_.maximum_modes - modes_.size()) {
        throw std::runtime_error("resonant-image maximum mode count reached");
    }
    if (missing_cells.size() >=
        static_cast<std::size_t>(
            std::numeric_limits<std::uint64_t>::max() - next_mode_id_
        )) {
        throw std::overflow_error("resonant-image mode ID space exhausted");
    }

    const ResonantImageOperationStats before = operation_stats_;
    std::size_t created = 0U;
    std::size_t updated = 0U;
    ++operation_stats_.training_examples;
    operation_stats_.training_patches +=
        static_cast<std::uint64_t>(patch_count);

    for (std::size_t patch_y = 0U; patch_y < patch_rows; ++patch_y) {
        for (std::size_t patch_x = 0U; patch_x < patch_columns; ++patch_x) {
            const auto [x_bin, y_bin] = coordinate_bin(
                patch_x,
                patch_y,
                patch_columns,
                patch_rows
            );
            const CellKey key{example.semantic_label, x_bin, y_bin};
            const core::PhaseVector source_state =
                encode_patch(example.source, patch_x, patch_y);
            const core::PhaseVector target_state =
                encode_patch(example.target, patch_x, patch_y);
            const core::PhaseVector desired_transformation =
                core::PhaseVector::phase_difference(
                    source_state,
                    target_state
                );
            const core::PhaseVector context = position_context(x_bin, y_bin);

            const auto existing = cell_index_.find(key);
            if (existing == cell_index_.end()) {
                const std::size_t mode_index = modes_.size();
                modes_.push_back({
                    .semantic_label = example.semantic_label,
                    .x_bin = x_bin,
                    .y_bin = y_bin,
                    .resonant_mode = core::ResonantMode(
                        next_mode_id_,
                        context,
                        desired_transformation,
                        1.0F,
                        0.5F,
                        0.0F,
                        operation_stats_.training_examples
                    ),
                    .example_count = 1ULL,
                });
                cell_index_.emplace(key, mode_index);
                index_semantic_mode(mode_index);
                ++next_mode_id_;
                ++created;
                ++operation_stats_.modes_created;
                continue;
            }

            ResonantImageMode& image_mode = modes_.at(existing->second);
            core::ResonantMode& mode = image_mode.resonant_mode;
            const double before_quality =
                source_state.composed(mode.transformation).similarity(
                    target_state
                );
            mode.transformation = circular_interpolate(
                mode.transformation,
                desired_transformation,
                config_.transformation_learning_rate,
                backend_.get()
            );
            mode.context_key = circular_interpolate(
                mode.context_key,
                context,
                config_.context_learning_rate,
                backend_.get()
            );
            const double after_quality =
                source_state.composed(mode.transformation).similarity(
                    target_state
                );
            mode.confidence = static_cast<float>(std::clamp(
                static_cast<double>(mode.confidence) +
                    config_.confidence_learning_rate *
                    (after_quality - static_cast<double>(mode.confidence)),
                0.0,
                1.0
            ));
            const double quality_gain = after_quality - before_quality;
            mode.utility = static_cast<float>(
                static_cast<double>(mode.utility) +
                config_.confidence_learning_rate *
                (quality_gain - static_cast<double>(mode.utility))
            );
            ++mode.activation_count;
            mode.last_used_step = operation_stats_.training_examples;
            if (after_quality + 1.0e-12 >= before_quality) {
                ++mode.successful_update_count;
            } else {
                ++mode.unsuccessful_update_count;
            }
            ++image_mode.example_count;
            ++updated;
            ++operation_stats_.local_mode_updates;
        }
    }

    return {
        .patches = patch_count,
        .modes_created = created,
        .modes_updated = updated,
        .operation_delta = subtract_stats(operation_stats_, before),
    };
}

core::PhaseVector ResonantImageFabric::settle_patch(
    const core::PhaseVector& input,
    const std::string_view semantic_label,
    const std::size_t patch_x,
    const std::size_t patch_y,
    const std::size_t patch_columns,
    const std::size_t patch_rows,
    const bool capture_trace,
    std::vector<ResonantImageTraceEntry>& trace,
    std::vector<std::uint64_t>& selected_mode_ids
) {
    const auto [x_bin, y_bin] = coordinate_bin(
        patch_x,
        patch_y,
        patch_columns,
        patch_rows
    );
    const core::PhaseVector context = position_context(x_bin, y_bin);
    core::PhaseVector previous = input;

    for (std::size_t cycle = 0U;
         cycle < config_.maximum_settling_cycles;
         ++cycle) {
        const std::vector<RetrievedMode> retrieved = retrieve(
            semantic_label,
            x_bin,
            y_bin,
            context
        );
        std::vector<core::PhaseVector> proposals;
        std::vector<float> proposal_weights;
        std::vector<std::uint64_t> active_ids;
        proposals.reserve(config_.active_count);
        proposal_weights.reserve(config_.active_count);
        active_ids.reserve(config_.active_count);
        double strongest_resonance = 0.0;

        for (const RetrievedMode& candidate : retrieved) {
            if (candidate.resonance < config_.minimum_resonance ||
                proposals.size() >= config_.active_count) {
                continue;
            }
            ResonantImageMode& image_mode = modes_.at(candidate.mode_index);
            core::ResonantMode& mode = image_mode.resonant_mode;
            proposals.push_back(input.composed(mode.transformation));
            proposal_weights.push_back(static_cast<float>(
                candidate.resonance *
                candidate.semantic_similarity *
                std::max(static_cast<double>(mode.confidence), 1.0e-6)
            ));
            active_ids.push_back(mode.id);
            strongest_resonance = std::max(
                strongest_resonance,
                candidate.resonance
            );
            ++mode.activation_count;
            mode.last_used_step = operation_stats_.generation_calls;
        }

        if (proposals.empty()) {
            ++operation_stats_.unresolved_patch_transformations;
            if (capture_trace) {
                trace.push_back({
                    .semantic_label = std::string(semantic_label),
                    .patch_x = patch_x,
                    .patch_y = patch_y,
                    .cycle = cycle,
                    .active_mode_ids = {},
                    .strongest_resonance = 0.0,
                    .state_change_radians = 0.0,
                });
            }
            return input;
        }

        const core::PhaseVector proposal =
            core::PhaseVector::weighted_circular_average(
                proposals,
                proposal_weights
            );
        const core::PhaseVector next = circular_interpolate(
            previous,
            proposal,
            config_.settling_relaxation,
            backend_.get()
        );
        const double change = next.mean_angular_error(previous);
        operation_stats_.active_mode_applications +=
            static_cast<std::uint64_t>(proposals.size());
        ++operation_stats_.settling_cycles;
        if (cycle == 0U) {
            selected_mode_ids.insert(
                selected_mode_ids.end(),
                active_ids.begin(),
                active_ids.end()
            );
        }
        if (capture_trace) {
            trace.push_back({
                .semantic_label = std::string(semantic_label),
                .patch_x = patch_x,
                .patch_y = patch_y,
                .cycle = cycle,
                .active_mode_ids = std::move(active_ids),
                .strongest_resonance = strongest_resonance,
                .state_change_radians = change,
            });
        }
        previous = next;
        if (change <= config_.convergence_tolerance_radians) {
            break;
        }
    }
    return previous;
}

ResonantGeneratedImage ResonantImageFabric::generate(
    const ResonantImageGenerateRequest& request
) {
    validate_image(request.base_image);
    for (const std::string& semantic_label : request.transformations) {
        validate_concept(semantic_label);
    }
    const std::size_t patch_columns =
        request.base_image.width / config_.patch_size;
    const std::size_t patch_rows =
        request.base_image.height / config_.patch_size;
    const std::size_t patch_count = patch_columns * patch_rows;
    if (request.capture_trace && !request.transformations.empty()) {
        if (patch_count >
                config_.maximum_trace_entries /
                    request.transformations.size() ||
            patch_count * request.transformations.size() >
                config_.maximum_trace_entries /
                    config_.maximum_settling_cycles) {
            throw std::invalid_argument(
                "requested resonant-image trace exceeds its configured limit"
            );
        }
    }

    const ResonantImageOperationStats before = operation_stats_;
    ++operation_stats_.generation_calls;
    operation_stats_.generated_patches +=
        static_cast<std::uint64_t>(patch_count);

    std::vector<core::PhaseVector> states;
    states.reserve(patch_count);
    for (std::size_t patch_y = 0U; patch_y < patch_rows; ++patch_y) {
        for (std::size_t patch_x = 0U; patch_x < patch_columns; ++patch_x) {
            states.push_back(encode_patch(
                request.base_image,
                patch_x,
                patch_y
            ));
        }
    }

    std::vector<std::uint64_t> selected_mode_ids;
    std::vector<ResonantImageTraceEntry> trace;
    if (request.capture_trace) {
        trace.reserve(
            patch_count * request.transformations.size() *
            config_.maximum_settling_cycles
        );
    }
    for (const std::string& semantic_label : request.transformations) {
        for (std::size_t patch_y = 0U; patch_y < patch_rows; ++patch_y) {
            for (std::size_t patch_x = 0U; patch_x < patch_columns; ++patch_x) {
                const std::size_t patch_index =
                    patch_y * patch_columns + patch_x;
                states[patch_index] = settle_patch(
                    states[patch_index],
                    semantic_label,
                    patch_x,
                    patch_y,
                    patch_columns,
                    patch_rows,
                    request.capture_trace,
                    trace,
                    selected_mode_ids
                );
            }
        }
    }

    ImageData output{
        .width = request.base_image.width,
        .height = request.base_image.height,
        .rgb = std::vector<std::uint8_t>(request.base_image.rgb.size()),
    };
    for (std::size_t patch_y = 0U; patch_y < patch_rows; ++patch_y) {
        for (std::size_t patch_x = 0U; patch_x < patch_columns; ++patch_x) {
            const std::size_t patch_index =
                patch_y * patch_columns + patch_x;
            const ImageData patch = decode_patch(states[patch_index]);
            for (std::size_t local_y = 0U;
                 local_y < config_.patch_size;
                 ++local_y) {
                for (std::size_t local_x = 0U;
                     local_x < config_.patch_size;
                     ++local_x) {
                    const std::size_t output_x =
                        patch_x * config_.patch_size + local_x;
                    const std::size_t output_y =
                        patch_y * config_.patch_size + local_y;
                    const std::size_t output_offset =
                        (output_y * output.width + output_x) * 3U;
                    const std::size_t patch_offset =
                        (local_y * config_.patch_size + local_x) * 3U;
                    for (std::size_t channel = 0U;
                         channel < 3U;
                         ++channel) {
                        output.rgb[output_offset + channel] =
                            patch.rgb[patch_offset + channel];
                    }
                }
            }
        }
    }
    operation_stats_.decoded_channels +=
        static_cast<std::uint64_t>(output.rgb.size());

    std::sort(selected_mode_ids.begin(), selected_mode_ids.end());
    selected_mode_ids.erase(
        std::unique(selected_mode_ids.begin(), selected_mode_ids.end()),
        selected_mode_ids.end()
    );
    const ResonantImageOperationStats operation_delta =
        subtract_stats(operation_stats_, before);

    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, deterministic_hash());
    hash_u64(hash, static_cast<std::uint64_t>(output.width));
    hash_u64(hash, static_cast<std::uint64_t>(output.height));
    for (const std::uint8_t byte : output.rgb) {
        hash ^= static_cast<std::uint64_t>(byte);
        hash *= fnv_prime;
    }
    for (const std::string& semantic_label : request.transformations) {
        hash_string(hash, semantic_label);
    }
    for (const std::uint64_t mode_id : selected_mode_ids) {
        hash_u64(hash, mode_id);
    }

    return {
        .image = std::move(output),
        .selected_mode_ids = std::move(selected_mode_ids),
        .trace = std::move(trace),
        .operation_delta = operation_delta,
        .deterministic_hash = hash,
    };
}

ResonantImageQuality evaluate_resonant_image_quality(
    const ImageData& candidate,
    const ImageData& reference
) {
    const std::size_t candidate_size =
        checked_rgb_size(candidate.width, candidate.height);
    const std::size_t reference_size =
        checked_rgb_size(reference.width, reference.height);
    if (candidate.width != reference.width ||
        candidate.height != reference.height ||
        candidate.rgb.size() != candidate_size ||
        reference.rgb.size() != reference_size) {
        throw std::invalid_argument(
            "resonant-image quality comparison requires matching valid images"
        );
    }

    double absolute_error_sum = 0.0;
    double squared_error_sum = 0.0;
    std::size_t exact_channels = 0U;
    std::size_t exact_pixels = 0U;
    const std::size_t pixels = candidate.width * candidate.height;
    for (std::size_t pixel = 0U; pixel < pixels; ++pixel) {
        bool exact_pixel = true;
        for (std::size_t channel = 0U; channel < 3U; ++channel) {
            const std::size_t index = pixel * 3U + channel;
            const double difference =
                static_cast<double>(candidate.rgb[index]) -
                static_cast<double>(reference.rgb[index]);
            absolute_error_sum += std::abs(difference);
            squared_error_sum += difference * difference;
            if (difference == 0.0) {
                ++exact_channels;
            } else {
                exact_pixel = false;
            }
        }
        if (exact_pixel) {
            ++exact_pixels;
        }
    }
    const double channels = static_cast<double>(candidate.rgb.size());
    const double mean_squared_error = squared_error_sum / channels;
    return {
        .mean_absolute_error = absolute_error_sum / channels,
        .mean_squared_error = mean_squared_error,
        .peak_signal_to_noise_db = mean_squared_error == 0.0
            ? std::numeric_limits<double>::infinity()
            : 10.0 * std::log10(
                (255.0 * 255.0) / mean_squared_error
            ),
        .exact_channel_fraction =
            static_cast<double>(exact_channels) / channels,
        .exact_pixel_fraction =
            static_cast<double>(exact_pixels) /
            static_cast<double>(pixels),
    };
}

ResonantImageComparison ResonantImageFabric::compare(
    const ResonantGeneratedImage& generated,
    const ImageData& reference
) const {
    return {
        .quality = evaluate_resonant_image_quality(
            generated.image,
            reference
        ),
        .generation_operations = generated.operation_delta,
        .learned_modes = modes_.size(),
        .estimated_model_bytes = estimated_model_bytes(),
    };
}

std::size_t ResonantImageFabric::estimated_model_bytes() const noexcept {
    std::size_t bytes = sizeof(*this);
    bytes += carriers_.capacity() * sizeof(float);
    bytes += carrier_signs_.capacity() * sizeof(std::int8_t);
    bytes += modes_.capacity() * sizeof(ResonantImageMode);
    for (const ResonantImageMode& mode : modes_) {
        bytes += mode.semantic_label.capacity();
        bytes += mode.resonant_mode.context_key.size() * sizeof(float);
        bytes += mode.resonant_mode.transformation.size() * sizeof(float);
    }
    for (const auto& [key, mode_index] : cell_index_) {
        static_cast<void>(mode_index);
        bytes += key.semantic_label.capacity();
    }
    for (const auto& [key, postings] : semantic_cell_index_) {
        static_cast<void>(key);
        bytes += postings.capacity() * sizeof(std::size_t);
    }
    bytes += prompt_semantics_.estimated_bytes();
    return bytes;
}

std::uint64_t ResonantImageFabric::deterministic_hash() const noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, static_cast<std::uint64_t>(config_.patch_size));
    hash_u64(hash, static_cast<std::uint64_t>(config_.phase_redundancy));
    hash_u64(hash, static_cast<std::uint64_t>(config_.coordinate_bins));
    hash_u64(hash, static_cast<std::uint64_t>(config_.maximum_modes));
    hash_u64(hash, static_cast<std::uint64_t>(config_.maximum_concept_bytes));
    hash_u64(hash, static_cast<std::uint64_t>(config_.maximum_image_side));
    hash_u64(hash, static_cast<std::uint64_t>(config_.maximum_image_pixels));
    hash_u64(hash, static_cast<std::uint64_t>(config_.candidate_count));
    hash_u64(hash, static_cast<std::uint64_t>(config_.active_count));
    hash_u64(
        hash,
        static_cast<std::uint64_t>(config_.maximum_settling_cycles)
    );
    hash_u64(hash, static_cast<std::uint64_t>(config_.maximum_trace_entries));
    hash_u64(hash, static_cast<std::uint64_t>(config_.maximum_prompt_concepts));
    hash_u64(
        hash,
        static_cast<std::uint64_t>(config_.maximum_semantic_candidates)
    );
    hash_double(hash, config_.minimum_resonance);
    hash_double(hash, config_.minimum_semantic_similarity);
    hash_double(hash, config_.semantic_resonance_weight);
    hash_double(hash, config_.convergence_tolerance_radians);
    hash_double(hash, config_.settling_relaxation);
    hash_double(hash, config_.transformation_learning_rate);
    hash_double(hash, config_.context_learning_rate);
    hash_double(hash, config_.confidence_learning_rate);
    hash_u64(hash, config_.seed);
    hash_u64(hash, prompt_semantics_.deterministic_hash());
    hash_u64(hash, next_mode_id_);
    hash_u64(hash, static_cast<std::uint64_t>(modes_.size()));
    for (const ResonantImageMode& image_mode : modes_) {
        hash_string(hash, image_mode.semantic_label);
        hash_u64(hash, static_cast<std::uint64_t>(image_mode.x_bin));
        hash_u64(hash, static_cast<std::uint64_t>(image_mode.y_bin));
        const core::ResonantMode& mode = image_mode.resonant_mode;
        hash_u64(hash, mode.id);
        for (const float angle : mode.context_key.angles()) {
            hash_float(hash, angle);
        }
        for (const float angle : mode.transformation.angles()) {
            hash_float(hash, angle);
        }
        hash_float(hash, mode.selectivity);
        hash_float(hash, mode.confidence);
        hash_float(hash, mode.utility);
        hash_u64(hash, mode.successful_update_count);
        hash_u64(hash, mode.unsuccessful_update_count);
        hash_u64(hash, mode.creation_step);
        hash_u64(hash, mode.enabled ? 1ULL : 0ULL);
        hash_u64(hash, image_mode.example_count);
    }
    return hash;
}

ResonantImageSnapshot ResonantImageFabric::snapshot() const {
    return {
        .config = config_,
        .next_mode_id = next_mode_id_,
        .modes = modes_,
        .operation_stats = operation_stats_,
        .prompt_semantics = prompt_semantics_.snapshot(),
    };
}

ResonantImageFabric ResonantImageFabric::from_snapshot(
    ResonantImageSnapshot snapshot_value
) {
    ResonantImageFabric fabric(snapshot_value.config);
    fabric.prompt_semantics_ = PromptSemanticFabric::from_snapshot(
        std::move(snapshot_value.prompt_semantics)
    );
    if (!prompt_semantic_configs_equal(
            fabric.prompt_semantics_.config(),
            snapshot_value.config.prompt_semantics
        )) {
        throw std::invalid_argument(
            "prompt-semantic snapshot configuration mismatch"
        );
    }
    if (snapshot_value.next_mode_id == 0ULL ||
        snapshot_value.modes.size() > snapshot_value.config.maximum_modes) {
        throw std::invalid_argument("invalid resonant-image snapshot bounds");
    }

    std::unordered_set<std::uint64_t> mode_ids;
    mode_ids.reserve(snapshot_value.modes.size());
    std::map<CellKey, std::size_t> rebuilt_index;
    std::uint64_t maximum_id = 0ULL;
    for (std::size_t index = 0U;
         index < snapshot_value.modes.size();
         ++index) {
        const ResonantImageMode& image_mode = snapshot_value.modes[index];
        fabric.validate_concept(image_mode.semantic_label);
        if (image_mode.x_bin >= snapshot_value.config.coordinate_bins ||
            image_mode.y_bin >= snapshot_value.config.coordinate_bins ||
            image_mode.example_count == 0ULL ||
            image_mode.resonant_mode.id == 0ULL ||
            !mode_ids.insert(image_mode.resonant_mode.id).second ||
            image_mode.resonant_mode.context_key.size() !=
                fabric.phase_dimension_ ||
            image_mode.resonant_mode.transformation.size() !=
                fabric.phase_dimension_ ||
            !std::isfinite(image_mode.resonant_mode.selectivity) ||
            image_mode.resonant_mode.selectivity < 0.0F ||
            !std::isfinite(image_mode.resonant_mode.confidence) ||
            image_mode.resonant_mode.confidence < 0.0F ||
            image_mode.resonant_mode.confidence > 1.0F ||
            !std::isfinite(image_mode.resonant_mode.utility)) {
            throw std::invalid_argument("invalid resonant-image snapshot mode");
        }
        for (const core::CorrectionSummary& correction :
             image_mode.resonant_mode.recent_corrections) {
            if (correction.context.size() != fabric.phase_dimension_ ||
                correction.desired_transformation.size() !=
                    fabric.phase_dimension_ ||
                !std::isfinite(correction.proposal_quality)) {
                throw std::invalid_argument(
                    "invalid resonant-image snapshot correction"
                );
            }
        }
        if (!rebuilt_index.emplace(
                CellKey{
                    image_mode.semantic_label,
                    image_mode.x_bin,
                    image_mode.y_bin,
                },
                index
            ).second) {
            throw std::invalid_argument(
                "duplicate resonant-image snapshot cell"
            );
        }
        maximum_id = std::max(maximum_id, image_mode.resonant_mode.id);
    }
    if (!snapshot_value.modes.empty() &&
        snapshot_value.next_mode_id <= maximum_id) {
        throw std::invalid_argument(
            "resonant-image snapshot next mode ID is not monotonic"
        );
    }

    fabric.next_mode_id_ = snapshot_value.next_mode_id;
    fabric.modes_ = std::move(snapshot_value.modes);
    fabric.cell_index_ = std::move(rebuilt_index);
    fabric.rebuild_semantic_index();
    fabric.operation_stats_ = snapshot_value.operation_stats;
    return fabric;
}

}  // namespace rlf::solstice
