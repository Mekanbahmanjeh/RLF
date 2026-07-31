#include "rlf/sdk/session.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace rlf::sdk {
namespace {

[[nodiscard]] std::string role_prefix(const ChatRole role) {
    return role == ChatRole::user ? "User: " : "Assistant: ";
}

}  // namespace

ChatSession::ChatSession(Pipeline pipeline, ContextWindowConfig config)
    : pipeline_(std::move(pipeline)),
      config_(std::move(config)) {
    if (config_.maximum_turns == 0U) {
        throw std::invalid_argument("chat maximum_turns must be positive");
    }
    if (pipeline_.task() == PipelineTask::tool_use) {
        throw std::invalid_argument(
            "stateful ChatSession does not support tool-use pipelines"
        );
    }
    if (config_.maximum_context_tokens == 0U) {
        config_.maximum_context_tokens = std::max<std::size_t>(
            1U,
            pipeline_.model_info().context.maximum_episode_cue_tokens
        );
    }
    const std::size_t minimum_context_tokens =
        pipeline_.model_.token_count("User: \nAssistant:") + 1U;
    if (config_.maximum_context_tokens < minimum_context_tokens) {
        throw std::invalid_argument(
            "chat context window is too small for the conversation framing"
        );
    }
    last_stats_.maximum_context_tokens = config_.maximum_context_tokens;
}

std::string ChatSession::build_prompt(
    const std::string_view current_prompt,
    ContextWindowStats& stats
) {
    const auto& tokenizer = pipeline_.model_.model().tokenizer();
    const auto token_count = [&tokenizer](const std::string_view text) {
        return tokenizer.encode(text).size();
    };
    const auto render = [this](
        const std::size_t begin,
        const std::string_view current,
        const bool include_system
    ) {
        std::string result;
        if (include_system && !config_.system_prompt.empty()) {
            result += "System: ";
            result += config_.system_prompt;
            result += '\n';
        }
        for (std::size_t index = begin; index < history_.size(); ++index) {
            result += role_prefix(history_[index].role);
            result += history_[index].text;
            result += '\n';
        }
        result += "User: ";
        result += current;
        result += "\nAssistant:";
        return result;
    };

    bool include_system = !config_.system_prompt.empty();
    std::size_t begin = 0U;
    std::string result = render(begin, current_prompt, include_system);
    while (token_count(result) > config_.maximum_context_tokens &&
           begin < history_.size()) {
        const std::size_t remaining = history_.size() - begin;
        const std::size_t removed = std::min<std::size_t>(2U, remaining);
        begin += removed;
        evicted_messages_ += removed;
        result = render(begin, current_prompt, include_system);
    }
    if (token_count(result) > config_.maximum_context_tokens && include_system) {
        include_system = false;
        stats.system_prompt_evicted = true;
        result = render(begin, current_prompt, false);
    }
    if (token_count(result) > config_.maximum_context_tokens) {
        std::vector<solstice::TokenId> tokens = tokenizer.encode(current_prompt);
        const std::string suffix = "\nAssistant:";
        const std::size_t overhead = token_count("User: " + suffix);
        const std::size_t available = config_.maximum_context_tokens > overhead
            ? config_.maximum_context_tokens - overhead
            : 1U;
        if (tokens.size() > available) {
            tokens.erase(
                tokens.begin(),
                tokens.end() - static_cast<std::ptrdiff_t>(available)
            );
            stats.current_prompt_truncated = true;
        }
        result = "User: " + tokenizer.decode(tokens) + suffix;
    }

    if (begin != 0U) {
        history_.erase(
            history_.begin(),
            history_.begin() + static_cast<std::ptrdiff_t>(begin)
        );
    }
    stats.maximum_context_tokens = config_.maximum_context_tokens;
    stats.input_tokens = token_count(result);
    stats.retained_messages = history_.size();
    stats.evicted_messages = evicted_messages_;
    return result;
}

void ChatSession::enforce_turn_limit() {
    const std::size_t maximum_messages = config_.maximum_turns * 2U;
    if (history_.size() <= maximum_messages) {
        return;
    }
    const std::size_t remove_count = history_.size() - maximum_messages;
    history_.erase(
        history_.begin(),
        history_.begin() + static_cast<std::ptrdiff_t>(remove_count)
    );
    evicted_messages_ += remove_count;
}

PipelineOutput ChatSession::send(
    const std::string_view prompt,
    const std::optional<std::filesystem::path>& image
) {
    if (prompt.empty()) {
        throw std::invalid_argument("chat prompt must not be empty");
    }
    ContextWindowStats stats;
    const std::string contextual_prompt = build_prompt(prompt, stats);
    PipelineOutput output = pipeline_(
        PipelineRequest{contextual_prompt, image}
    );
    history_.push_back(ChatMessage{ChatRole::user, std::string(prompt)});
    history_.push_back(ChatMessage{ChatRole::assistant, output.text});
    enforce_turn_limit();
    stats.retained_messages = history_.size();
    stats.evicted_messages = evicted_messages_;
    last_stats_ = stats;
    return output;
}

void ChatSession::clear() noexcept {
    history_.clear();
    evicted_messages_ = 0U;
    last_stats_ = ContextWindowStats{
        config_.maximum_context_tokens, 0U, 0U, 0U, false, false,
    };
}

std::span<const ChatMessage> ChatSession::history() const noexcept {
    return history_;
}

const ContextWindowConfig& ChatSession::config() const noexcept {
    return config_;
}

ContextWindowStats ChatSession::context_stats() const noexcept {
    return last_stats_;
}

}  // namespace rlf::sdk
