#include "rlf/solstice/profile.hpp"

#include <stdexcept>
#include <string>

namespace rlf::solstice {

SolsticeProfile parse_profile(const std::string_view name) {
    if (name == "preview-6g" || name == "preview" || name == "6g") {
        return SolsticeProfile::preview_6g;
    }
    if (name == "frontier-24g" || name == "frontier" || name == "3090" ||
        name == "24g") {
        return SolsticeProfile::frontier_24g;
    }
    if (name == "general-h100" || name == "frontier-general" ||
        name == "general-frontier" || name == "general-80g") {
        return SolsticeProfile::general_h100_80g;
    }
    if (name == "general-40g" || name == "general-cuda-40g" ||
        name == "frontier-40g" || name == "40g") {
        return SolsticeProfile::general_cuda_40g;
    }
    if (name == "general-v100-32g" || name == "v100-32g" ||
        name == "general-32g" || name == "32g") {
        return SolsticeProfile::general_v100_32g;
    }
    if (name == "general-v100-32g-text" || name == "v100-32g-text" ||
        name == "general-32g-text" || name == "32g-text") {
        return SolsticeProfile::general_v100_32g_text;
    }
    if (name == "general-v100-32g-500m" || name == "v100-32g-500m") {
        return SolsticeProfile::general_v100_32g_500m;
    }
    if (name == "general-h200-141g-30t" || name == "general-h200" ||
        name == "h200-141g" || name == "h200") {
        return SolsticeProfile::general_h200_141g_30t;
    }
    if (name == "video-v100-32g" || name == "v100-32g-video" ||
        name == "video-32g") {
        return SolsticeProfile::video_v100_32g;
    }
    if (name == "video-rtx-pro-6000-96g" || name == "video-pro6000-96g" ||
        name == "video-96g") {
        return SolsticeProfile::video_rtx_pro_6000_96g;
    }
    if (name == "general-rtx-pro-6000-96g" || name == "general-pro6000-96g" ||
        name == "general-96g") {
        return SolsticeProfile::general_rtx_pro_6000_96g;
    }
    if (name == "rtx-pro-6000-96g") {
        return SolsticeProfile::rtx_pro_6000_96g;
    }
    if (name == "general-rtx-pro-6000-96g-text" ||
        name == "general-pro6000-96g-text" || name == "general-96g-text") {
        return SolsticeProfile::general_rtx_pro_6000_96g_text;
    }
    if (name == "frontier-h100" || name == "h100" || name == "80g" ||
        name == "frontier-80g") {
        return SolsticeProfile::frontier_h100_80g;
    }
    throw std::invalid_argument("unknown Solstice profile: " + std::string(name));
}

std::string_view to_string(const SolsticeProfile profile) noexcept {
    switch (profile) {
        case SolsticeProfile::preview_6g: return "preview-6g";
        case SolsticeProfile::frontier_24g: return "frontier-24g";
        case SolsticeProfile::frontier_h100_80g: return "frontier-h100";
        case SolsticeProfile::general_h100_80g: return "general-h100";
        case SolsticeProfile::general_cuda_40g: return "general-40g";
        case SolsticeProfile::general_v100_32g: return "general-v100-32g";
        case SolsticeProfile::general_v100_32g_text: return "general-v100-32g-text";
        case SolsticeProfile::video_rtx_pro_6000_96g: return "video-rtx-pro-6000-96g";
        case SolsticeProfile::general_rtx_pro_6000_96g: return "general-rtx-pro-6000-96g";
        case SolsticeProfile::general_rtx_pro_6000_96g_text: return "general-rtx-pro-6000-96g-text";
        case SolsticeProfile::rtx_pro_6000_96g: return "rtx-pro-6000-96g";
        case SolsticeProfile::general_v100_32g_500m: return "general-v100-32g-500m";
        case SolsticeProfile::video_v100_32g: return "video-v100-32g";
        case SolsticeProfile::general_h200_141g_30t: return "general-h200-141g-30t";
    }
    return "unknown";
}

SolsticeConfig make_profile_config(const SolsticeProfile profile) {
    SolsticeConfig config;
    if (profile == SolsticeProfile::preview_6g) {
        return config;
    }
    const bool general = profile == SolsticeProfile::general_h100_80g ||
        profile == SolsticeProfile::general_cuda_40g ||
        profile == SolsticeProfile::general_v100_32g ||
        profile == SolsticeProfile::general_v100_32g_text ||
        profile == SolsticeProfile::general_v100_32g_500m ||
        profile == SolsticeProfile::general_h200_141g_30t ||
        profile == SolsticeProfile::video_v100_32g ||
        profile == SolsticeProfile::video_rtx_pro_6000_96g ||
        profile == SolsticeProfile::general_rtx_pro_6000_96g ||
        profile == SolsticeProfile::general_rtx_pro_6000_96g_text ||
        profile == SolsticeProfile::rtx_pro_6000_96g;
    const bool large = profile == SolsticeProfile::frontier_h100_80g || general;
    const bool constrained_general = profile == SolsticeProfile::general_cuda_40g ||
        profile == SolsticeProfile::general_v100_32g ||
        profile == SolsticeProfile::general_v100_32g_text ||
        profile == SolsticeProfile::general_v100_32g_500m ||
        profile == SolsticeProfile::video_v100_32g;
    const bool text_only = profile == SolsticeProfile::general_v100_32g_text ||
        profile == SolsticeProfile::general_rtx_pro_6000_96g_text;
    const bool video = profile == SolsticeProfile::video_rtx_pro_6000_96g ||
        profile == SolsticeProfile::video_v100_32g ||
        profile == SolsticeProfile::preview_6g;

    const bool rtx_pro_50m =
        profile == SolsticeProfile::general_rtx_pro_6000_96g ||
        profile == SolsticeProfile::rtx_pro_6000_96g;
    const bool v100_500m = profile == SolsticeProfile::general_v100_32g_500m;
    const bool h200_30t = profile == SolsticeProfile::general_h200_141g_30t;

    config.tokenizer.maximum_vocabulary = large ? 131'072U : 65'536U;
    config.tokenizer.maximum_merges = large ? 120'000U : 60'000U;
    config.tokenizer.minimum_pair_support = 3U;
    config.tokenizer.maximum_piece_bytes = 128U;
    config.tokenizer.maximum_training_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;

    config.language.context_orders = {
        0U, 1U, 2U, 4U, 8U, 16U, 32U, 64U,
        128U, 256U, 512U, 1'024U, 2'048U
    };
    config.language.maximum_contexts = general ? 120'000'000U : (large ? 80'000'000U : 20'000'000U);
    config.language.maximum_episodes = general ? 16'000'000U : (large ? 8'000'000U : 2'000'000U);
    config.language.maximum_episode_cue_tokens = general ? 8'192U : 4'096U;
    config.language.maximum_episode_response_tokens = general ? 8'192U : 4'096U;
    config.language.maximum_generation_tokens = general ? 4'096U : 2'048U;
    config.language.prediction_candidate_limit = general ? 4'096U : 2'048U;
    config.language.smoothing = 0.02;
    config.language.long_context_weight = 0.48;
    config.language.episode_conditioning_weight = 12.0;
    config.language.repetition_penalty = 1.22;

    config.vision.patch_size = 8U;
    config.vision.patch_sizes = {8U, 16U, 32U, 64U};
    config.vision.descriptor_dimensions = 32U;
    config.vision.maximum_input_side = 1'024U;
    config.vision.maximum_patches = 24'576U;
    config.vision.retrieval_query_batch = constrained_general ? 512U : 1'024U;
    config.vision.retrieval_candidate_batch = constrained_general ? 4'096U : 8'192U;
    config.vision.training_patch_batch = constrained_general ? 512U : 1'024U;
    config.vision.sparse_routing_minimum_modes = large ? 4'096U : 8'192U;
    config.vision.sparse_router.signature_bits = large ? 22U : 18U;
    config.vision.sparse_router.maximum_candidates = constrained_general
        ? 4'096U : (large ? 8'192U : 2'048U);
    config.vision.maximum_modes = general ? 2'097'152U : (large ? 1'048'576U : 262'144U);
    config.vision.maximum_examples = general ? 16'000'000U : (large ? 8'000'000U : 2'000'000U);
    config.vision.maximum_regions = 256U;
    config.vision.maximum_concepts_per_mode = 128U;
    config.vision.mode_creation_similarity = 0.90;
    config.vision.example_match_similarity = 0.76;
    config.vision.local_learning_rate = 0.06;

    config.tool_router.maximum_tools = 1'024U;
    config.tool_router.maximum_keywords_per_tool = 4'096U;
    config.tool_router.minimum_confidence = 0.52;
    config.abstraction.maximum_facts = general ? 50'000'000U : (large ? 25'000'000U : 5'000'000U);
    config.abstraction.maximum_rules = general ? 2'000'000U : (large ? 1'000'000U : 250'000U);
    config.abstraction.maximum_inference_depth = large ? 16U : 10U;
    config.abstraction.maximum_derivations_per_query = large ? 1'000'000U : 250'000U;

    config.continual.feature_dimensions = 64U;
    config.continual.maximum_prototypes = general ? 16'000'000U : (large ? 8'000'000U : 2'000'000U);
    config.continual.replay_capacity = large ? 2'000'000U : 500'000U;
    config.continual.replay_batch_size = constrained_general ? 1'024U : (large ? 2'048U : 512U);
    config.continual.router.signature_bits = large ? 22U : 18U;
    config.continual.router.maximum_candidates = constrained_general
        ? 2'048U : (large ? 4'096U : 1'024U);

    config.grounding.maximum_links = general ? 200'000'000U : (large ? 100'000'000U : 20'000'000U);
    config.grounding.maximum_concepts = general ? 16'000'000U : (large ? 8'000'000U : 2'000'000U);

    config.general.maximum_demonstrations = general
        ? 16'000'000U
        : (large ? 4'000'000U : 1'000'000U);
    config.general.maximum_preferences = general
        ? 8'000'000U
        : (large ? 2'000'000U : 500'000U);
    config.general.maximum_active_learning_items = general ? 1'000'000U : 250'000U;
    config.general.maximum_retrieval_candidates = general ? 65'536U : 16'384U;
    config.general.maximum_retrieved_demonstrations = general ? 32U : 16U;
    config.general.maximum_context_characters = general ? 131'072U : 65'536U;
    config.general.deliberation_candidates = general ? 8U : 4U;
    config.general.minimum_retrieval_similarity = 0.025;
    config.general.direct_recall_threshold = 0.97;

    if (rtx_pro_50m) {
        // The fixed 50M efficiency corpus contains 24,500,001 instruction
        // demonstrations, 8,166,666 preferences, and 32,666,667 language
        // dialogues. These are logical, progressively allocated host-backed
        // ceilings; they do not enlarge the GPU working set or change updates.
        config.language.maximum_episodes = 40'000'000U;
        config.general.maximum_demonstrations = 32'000'000U;
        config.general.maximum_preferences = 12'000'000U;
        config.tool_router.maximum_keywords_per_tool = 8'388'608U;
    }
    if (v100_500m) {
        // This isolated profile admits a fixed 10x expansion of the exact 50M
        // campaign mix without changing V100 learning rules or device batches.
        // The 500M generator ceiling contains 245,000,001 instruction rows,
        // 81,666,666 preferences, 326,666,667 resulting dialogue episodes,
        // 81,666,667 tool rows, 81,666,666 facts, and 10,000,000 images.
        // Containers grow
        // progressively (their constructors retain bounded initial reserves),
        // so these are host-backed admission ceilings rather than allocations.
        config.language.maximum_contexts = 1'200'000'000U;
        config.language.maximum_episodes = 400'000'000U;
        config.general.maximum_demonstrations = 320'000'000U;
        config.general.maximum_preferences = 120'000'000U;
        config.tool_router.maximum_keywords_per_tool = 100'000'000U;
        config.abstraction.maximum_facts = 100'000'000U;
    }
    if (h200_30t) {
        config.language.maximum_contexts = 2'000'000'000U;
        config.language.maximum_episodes = 500'000'000U;
        config.vision.maximum_modes = 16'777'216U;
        config.vision.maximum_examples = 250'000'000U;
        config.abstraction.maximum_facts = 500'000'000U;
        config.abstraction.maximum_rules = 16'000'000U;
        config.continual.maximum_prototypes = 250'000'000U;
        config.continual.replay_capacity = 16'000'000U;
        config.grounding.maximum_links = 4'000'000'000ULL;
        config.grounding.maximum_concepts = 250'000'000U;
        config.general.maximum_demonstrations = 500'000'000U;
        config.general.maximum_preferences = 250'000'000U;
        config.general.maximum_active_learning_items = 32'000'000U;
        config.tool_router.maximum_keywords_per_tool = 250'000'000U;
        config.general.maximum_context_characters = 262'144U;
    }

    config.maximum_tool_result_characters = large
        ? 4U * 1024U * 1024U
        : 1U * 1024U * 1024U;
    if (text_only) {
        config.vision.maximum_modes = 1U;
        config.vision.maximum_examples = 1U;
        config.vision.maximum_patches = 1U;
        config.vision.retrieval_query_batch = 1U;
        config.vision.retrieval_candidate_batch = 1U;
        config.vision.training_patch_batch = 1U;
        config.vision.sparse_routing_minimum_modes = 1U;
        config.vision.sparse_router.maximum_candidates = 1U;
        config.vision.maximum_regions = 1U;
        config.vision.maximum_concepts_per_mode = 1U;
    }
    if (video) {
        config.language.maximum_contexts = 160'000'000U;
        config.language.maximum_episodes = 24'000'000U;
        config.vision.maximum_modes = 4'194'304U;
        config.vision.maximum_examples = 32'000'000U;
        config.vision.maximum_patches = 32'768U;
        config.continual.maximum_prototypes = 24'000'000U;
        config.grounding.maximum_links = 300'000'000U;
        config.grounding.maximum_concepts = 24'000'000U;
        config.general.maximum_demonstrations = 24'000'000U;
        config.general.maximum_preferences = 12'000'000U;
        config.general.maximum_context_characters = 262'144U;
        config.video.maximum_sequences = 8'000'000U;
        config.video.maximum_frames_per_sequence = 256U;
        config.video.maximum_generation_frames = 1'024U;
        config.video.maximum_prompt_bytes = 262'144U;
        config.video.output_width = 256U;
        config.video.output_height = 256U;
        config.video.minimum_prompt_similarity = 0.08;
        config.video.local_learning_rate = 0.20;
    }
    return config;
}

ProfileCapacityEstimate estimate_profile_capacity(
    const SolsticeProfile profile
) noexcept {
    if (profile == SolsticeProfile::general_h100_80g) {
        return {
            74ULL * 1024ULL * 1024ULL * 1024ULL,
            512ULL * 1024ULL * 1024ULL * 1024ULL,
            1'024ULL * 1024ULL * 1024ULL * 1024ULL,
        };
    }
    if (profile == SolsticeProfile::general_h200_141g_30t) {
        return {
            132ULL * 1024ULL * 1024ULL * 1024ULL,
            8ULL * 1024ULL * 1024ULL * 1024ULL * 1024ULL,
            16ULL * 1024ULL * 1024ULL * 1024ULL * 1024ULL,
        };
    }
    if (profile == SolsticeProfile::general_cuda_40g) {
        return {
            38ULL * 1024ULL * 1024ULL * 1024ULL,
            512ULL * 1024ULL * 1024ULL * 1024ULL,
            1'024ULL * 1024ULL * 1024ULL * 1024ULL,
        };
    }
    if (profile == SolsticeProfile::general_v100_32g) {
        return {
            30ULL * 1024ULL * 1024ULL * 1024ULL,
            512ULL * 1024ULL * 1024ULL * 1024ULL,
            1'024ULL * 1024ULL * 1024ULL * 1024ULL,
        };
    }
    if (profile == SolsticeProfile::general_v100_32g_text) {
        return {
            30ULL * 1024ULL * 1024ULL * 1024ULL,
            512ULL * 1024ULL * 1024ULL * 1024ULL,
            1'024ULL * 1024ULL * 1024ULL * 1024ULL,
        };
    }
    if (profile == SolsticeProfile::general_v100_32g_500m) {
        return {
            30ULL * 1024ULL * 1024ULL * 1024ULL,
            5ULL * 1024ULL * 1024ULL * 1024ULL * 1024ULL,
            10ULL * 1024ULL * 1024ULL * 1024ULL * 1024ULL,
        };
    }
    if (profile == SolsticeProfile::video_v100_32g) {
        return {
            30ULL * 1024ULL * 1024ULL * 1024ULL,
            768ULL * 1024ULL * 1024ULL * 1024ULL,
            1'024ULL * 1024ULL * 1024ULL * 1024ULL,
        };
    }
    if (profile == SolsticeProfile::video_rtx_pro_6000_96g) {
        return {
            90ULL * 1024ULL * 1024ULL * 1024ULL,
            768ULL * 1024ULL * 1024ULL * 1024ULL,
            1'024ULL * 1024ULL * 1024ULL * 1024ULL,
        };
    }
    if (profile == SolsticeProfile::general_rtx_pro_6000_96g ||
        profile == SolsticeProfile::general_rtx_pro_6000_96g_text) {
        return {
            90ULL * 1024ULL * 1024ULL * 1024ULL,
            512ULL * 1024ULL * 1024ULL * 1024ULL,
            1'024ULL * 1024ULL * 1024ULL * 1024ULL,
        };
    }
    if (profile == SolsticeProfile::rtx_pro_6000_96g) {
        return {
            88ULL * 1024ULL * 1024ULL * 1024ULL,
            512ULL * 1024ULL * 1024ULL * 1024ULL,
            1'024ULL * 1024ULL * 1024ULL * 1024ULL,
        };
    }
    if (profile == SolsticeProfile::frontier_h100_80g) {
        return {
            72ULL * 1024ULL * 1024ULL * 1024ULL,
            256ULL * 1024ULL * 1024ULL * 1024ULL,
            512ULL * 1024ULL * 1024ULL * 1024ULL,
        };
    }
    if (profile == SolsticeProfile::frontier_24g) {
        return {
            21ULL * 1024ULL * 1024ULL * 1024ULL,
            64ULL * 1024ULL * 1024ULL * 1024ULL,
            128ULL * 1024ULL * 1024ULL * 1024ULL,
        };
    }
    return {
        5ULL * 1024ULL * 1024ULL * 1024ULL,
        12ULL * 1024ULL * 1024ULL * 1024ULL,
        24ULL * 1024ULL * 1024ULL * 1024ULL,
    };
}

bool profile_config_matches(
    const SolsticeProfile profile,
    const SolsticeConfig& config
) noexcept {
    try {
        const SolsticeConfig expected = make_profile_config(profile);
        return config.tokenizer.maximum_vocabulary == expected.tokenizer.maximum_vocabulary &&
            config.tokenizer.maximum_merges == expected.tokenizer.maximum_merges &&
            config.language.maximum_contexts == expected.language.maximum_contexts &&
            config.language.maximum_episodes == expected.language.maximum_episodes &&
            config.language.maximum_generation_tokens == expected.language.maximum_generation_tokens &&
            config.vision.maximum_patches == expected.vision.maximum_patches &&
            config.vision.retrieval_query_batch == expected.vision.retrieval_query_batch &&
            config.vision.retrieval_candidate_batch == expected.vision.retrieval_candidate_batch &&
            config.vision.training_patch_batch == expected.vision.training_patch_batch &&
            config.vision.maximum_modes == expected.vision.maximum_modes &&
            config.vision.maximum_examples == expected.vision.maximum_examples &&
            config.abstraction.maximum_facts == expected.abstraction.maximum_facts &&
            config.abstraction.maximum_rules == expected.abstraction.maximum_rules &&
            config.continual.maximum_prototypes == expected.continual.maximum_prototypes &&
            config.continual.replay_capacity == expected.continual.replay_capacity &&
            config.grounding.maximum_links == expected.grounding.maximum_links &&
            config.grounding.maximum_concepts == expected.grounding.maximum_concepts &&
            config.tool_router.maximum_tools == expected.tool_router.maximum_tools &&
            config.tool_router.maximum_keywords_per_tool ==
                expected.tool_router.maximum_keywords_per_tool &&
            config.general.maximum_demonstrations == expected.general.maximum_demonstrations &&
            config.general.maximum_preferences == expected.general.maximum_preferences &&
            config.general.maximum_context_characters == expected.general.maximum_context_characters &&
            config.video.maximum_sequences == expected.video.maximum_sequences &&
            config.video.maximum_frames_per_sequence == expected.video.maximum_frames_per_sequence &&
            config.video.maximum_generation_frames == expected.video.maximum_generation_frames &&
            config.video.maximum_prompt_bytes == expected.video.maximum_prompt_bytes &&
            config.video.output_width == expected.video.output_width &&
            config.video.output_height == expected.video.output_height &&
            config.maximum_tool_result_characters == expected.maximum_tool_result_characters;
    } catch (...) {
        return false;
    }
}

bool profile_allows_vision(const SolsticeProfile profile) noexcept {
    return profile != SolsticeProfile::general_v100_32g_text &&
        profile != SolsticeProfile::general_rtx_pro_6000_96g_text;
}

bool profile_allows_video(const SolsticeProfile profile) noexcept {
    return profile == SolsticeProfile::video_rtx_pro_6000_96g ||
        profile == SolsticeProfile::video_v100_32g ||
        profile == SolsticeProfile::preview_6g;
}


}  // namespace rlf::solstice
