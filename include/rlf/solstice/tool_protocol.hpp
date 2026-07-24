#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rlf::solstice {

enum class ToolArgumentType : std::uint8_t {
    string = 0U,
    number = 1U,
    boolean = 2U,
    path = 3U,
};

struct ToolArgumentSpec final {
    std::string name;
    ToolArgumentType type{ToolArgumentType::string};
    bool required{true};
    std::size_t maximum_length{4'096U};
};

struct ToolDefinition final {
    std::string name;
    std::string description;
    std::vector<ToolArgumentSpec> arguments;
    bool read_only{true};
    double cost{1.0};
};

struct ToolCall final {
    std::string name;
    std::map<std::string, std::string> arguments;
};

struct ToolResult final {
    bool success{};
    std::string output;
    std::string error;
};

struct ToolPolicy final {
    std::filesystem::path sandbox_root;
    std::size_t maximum_calls_per_turn{4U};
    std::size_t maximum_read_bytes{1U * 1024U * 1024U};
    std::size_t maximum_directory_entries{256U};
    bool allow_file_reads{false};
};

class ToolRuntime final {
public:
    using Handler = std::function<ToolResult(const ToolCall&, const ToolPolicy&)>;

    explicit ToolRuntime(ToolPolicy policy = {});

    void register_tool(ToolDefinition definition, Handler handler);
    void register_safe_builtins();

    [[nodiscard]] std::span<const ToolDefinition> definitions() const noexcept;
    [[nodiscard]] const ToolPolicy& policy() const noexcept;
    void set_policy(ToolPolicy policy);

    [[nodiscard]] ToolResult execute(const ToolCall& call) const;
    [[nodiscard]] std::optional<std::string> validate(const ToolCall& call) const;

    [[nodiscard]] static std::string serialize_call(const ToolCall& call);
    [[nodiscard]] static std::string serialize_result(const ToolResult& result);
    [[nodiscard]] static ToolCall parse_call(std::string_view json);

private:
    std::vector<ToolDefinition> definitions_;
    std::unordered_map<std::string, Handler> handlers_;
    ToolPolicy policy_;
};

struct ToolRouterConfig final {
    std::size_t maximum_tools{128U};
    std::size_t maximum_keywords_per_tool{512U};
    double minimum_confidence{0.58};
};

struct ToolKeywordCount final {
    std::string keyword;
    std::uint64_t count{};
};

struct ToolRoute final {
    std::string tool_name;
    std::uint64_t examples{};
    std::vector<ToolKeywordCount> keywords;
};

struct ToolProposal final {
    bool should_call{};
    ToolCall call;
    double confidence{};
    std::string reason;
};

struct ToolRouterSnapshot final {
    ToolRouterConfig config;
    std::vector<ToolRoute> routes;
};

struct ToolRouterTrainingOperationStats final {
    std::uint64_t training_rows{};
    std::uint64_t route_insert_attempts{};
    std::uint64_t route_inserts{};
    std::uint64_t route_capacity_failures{};
    std::uint64_t keyword_update_lookups{};
    std::uint64_t linear_keyword_comparisons{};
    std::uint64_t indexed_keyword_lookups{};
    std::uint64_t keyword_insert_attempts{};
    std::uint64_t keyword_inserts{};
    std::uint64_t keyword_capacity_skips{};
    std::uint64_t keyword_index_rebuilds{};
    std::uint64_t keyword_index_entries_built{};
    std::uint64_t keyword_index_incremental_inserts{};
};

class ToolRouter final {
public:
    explicit ToolRouter(ToolRouterConfig config = {});

    void train(std::string_view request, std::string_view tool_name);
    [[nodiscard]] ToolProposal propose(
        std::string_view request,
        std::span<const ToolDefinition> available_tools
    ) const;

    [[nodiscard]] std::span<const ToolRoute> routes() const noexcept;
    [[nodiscard]] const ToolRouterConfig& config() const noexcept;
    [[nodiscard]] ToolRouterSnapshot snapshot() const;
    [[nodiscard]] static ToolRouter from_snapshot(ToolRouterSnapshot snapshot);
    [[nodiscard]] std::uint64_t deterministic_hash() const noexcept;
    void set_indexed_keyword_updates(bool enabled);
    [[nodiscard]] ToolRouterTrainingOperationStats training_operation_stats() const noexcept;

private:
    [[nodiscard]] static std::vector<std::string> keywords(std::string_view text);
    [[nodiscard]] static std::string extract_quoted_or_path(std::string_view request);
    [[nodiscard]] static std::string extract_expression(std::string_view request);
    void rebuild_keyword_indexes();

    ToolRouterConfig config_;
    std::vector<ToolRoute> routes_;
    std::vector<std::unordered_map<std::string, std::size_t>> keyword_indexes_;
    bool indexed_keyword_updates_{true};
    ToolRouterTrainingOperationStats training_operation_stats_;
};

}  // namespace rlf::solstice
