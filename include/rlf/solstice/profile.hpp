#pragma once

#include "rlf/solstice/solstice_model.hpp"

#include <cstdint>
#include <string_view>

namespace rlf::solstice {

enum class SolsticeProfile : std::uint8_t {
    preview_6g = 0U,
    frontier_24g = 1U,
    frontier_h100_80g = 2U,
    general_h100_80g = 3U,
    general_cuda_40g = 4U,
    general_v100_32g = 5U,
    general_v100_32g_text = 6U,
    video_rtx_pro_6000_96g = 7U,
    general_rtx_pro_6000_96g = 8U,
    general_rtx_pro_6000_96g_text = 9U,
    rtx_pro_6000_96g = 10U,
    general_v100_32g_500m = 11U,
    video_v100_32g = 12U,
    general_h200_141g_30t = 13U,
};

struct ProfileCapacityEstimate final {
    std::uint64_t gpu_working_set_bytes{};
    std::uint64_t cpu_resident_bytes{};
    std::uint64_t checkpoint_ceiling_bytes{};
};

[[nodiscard]] SolsticeProfile parse_profile(std::string_view name);
[[nodiscard]] std::string_view to_string(SolsticeProfile profile) noexcept;
[[nodiscard]] SolsticeConfig make_profile_config(SolsticeProfile profile);
[[nodiscard]] ProfileCapacityEstimate estimate_profile_capacity(
    SolsticeProfile profile
) noexcept;
[[nodiscard]] bool profile_config_matches(
    SolsticeProfile profile,
    const SolsticeConfig& config
) noexcept;
[[nodiscard]] bool profile_allows_vision(SolsticeProfile profile) noexcept;
[[nodiscard]] bool profile_allows_video(SolsticeProfile profile) noexcept;

}  // namespace rlf::solstice
