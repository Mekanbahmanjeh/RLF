#pragma once

#include "rlf/sdk/pipeline.hpp"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rlf::sdk {

enum class ChatRole : std::uint8_t {
    user = 0U,
    assistant = 1U,
};

struct ChatMessage final {
    ChatRole role{ChatRole::user};
    std::string text;
};

struct ContextWindowConfig final {
    std::size_t maximum_context_tokens{};
    std::size_t maximum_turns{32U};
    std::string system_prompt;
};

struct ContextWindowStats final {
    std::size_t maximum_context_tokens{};
    std::size_t input_tokens{};
    std::size_t retained_messages{};
    std::size_t evicted_messages{};
    bool system_prompt_evicted{};
    bool current_prompt_truncated{};
};

class ChatSession final {
public:
    ChatSession(Pipeline pipeline, ContextWindowConfig config = {});

    [[nodiscard]] PipelineOutput send(
        std::string_view prompt,
        const std::optional<std::filesystem::path>& image = std::nullopt
    );
    void clear() noexcept;

    [[nodiscard]] std::span<const ChatMessage> history() const noexcept;
    [[nodiscard]] const ContextWindowConfig& config() const noexcept;
    [[nodiscard]] ContextWindowStats context_stats() const noexcept;

private:
    [[nodiscard]] std::string build_prompt(
        std::string_view current_prompt,
        ContextWindowStats& stats
    );
    void enforce_turn_limit();

    Pipeline pipeline_;
    ContextWindowConfig config_;
    std::vector<ChatMessage> history_;
    std::size_t evicted_messages_{};
    ContextWindowStats last_stats_;
};

}  // namespace rlf::sdk
