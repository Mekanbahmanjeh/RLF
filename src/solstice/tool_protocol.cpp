#include "rlf/solstice/tool_protocol.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unordered_set>
#include <utility>

namespace rlf::solstice {
namespace {

constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;

[[nodiscard]] bool indexed_tool_keyword_updates_from_environment() {
    const char* const value = std::getenv("RLF_TOOL_KEYWORD_UPDATE_POLICY");
    if (value == nullptr || std::string_view(value).empty() ||
        std::string_view(value) == "indexed") {
        return true;
    }
    if (std::string_view(value) == "linear") {
        return false;
    }
    throw std::invalid_argument(
        "RLF_TOOL_KEYWORD_UPDATE_POLICY must be indexed or linear"
    );
}

void hash_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (unsigned int index = 0U; index < 8U; ++index) {
        hash ^= (value >> (index * 8U)) & 0xFFULL;
        hash *= fnv_prime;
    }
}

void hash_string(std::uint64_t& hash, const std::string_view value) noexcept {
    hash_u64(hash, value.size());
    for (const char character : value) {
        hash ^= static_cast<unsigned char>(character);
        hash *= fnv_prime;
    }
}

[[nodiscard]] std::string lowercase(const std::string_view input) {
    std::string output;
    output.reserve(input.size());
    for (const char raw_character : input) {
        const unsigned char character = static_cast<unsigned char>(raw_character);
        output.push_back(static_cast<char>(std::tolower(character)));
    }
    return output;
}

[[nodiscard]] std::string json_escape(const std::string_view input) {
    std::ostringstream output;
    for (const char raw_character : input) {
        const unsigned char character = static_cast<unsigned char>(raw_character);
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << static_cast<unsigned int>(character)
                           << std::dec << std::setfill(' ');
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
}

class JsonCursor final {
public:
    explicit JsonCursor(const std::string_view input) : input_(input) {}

    void whitespace() {
        while (position_ < input_.size() &&
               std::isspace(static_cast<unsigned char>(input_[position_])) != 0) {
            ++position_;
        }
    }

    void expect(const char expected) {
        whitespace();
        if (position_ >= input_.size() || input_[position_] != expected) {
            throw std::invalid_argument("invalid tool-call JSON");
        }
        ++position_;
    }

    [[nodiscard]] bool consume(const char expected) {
        whitespace();
        if (position_ < input_.size() && input_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    [[nodiscard]] std::string string() {
        whitespace();
        expect('"');
        std::string result;
        while (position_ < input_.size()) {
            const char character = input_[position_++];
            if (character == '"') {
                return result;
            }
            if (character != '\\') {
                result.push_back(character);
                continue;
            }
            if (position_ >= input_.size()) {
                throw std::invalid_argument("invalid JSON string escape");
            }
            const char escape = input_[position_++];
            switch (escape) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                default:
                    throw std::invalid_argument("unsupported JSON string escape");
            }
        }
        throw std::invalid_argument("unterminated JSON string");
    }

    [[nodiscard]] std::string scalar() {
        whitespace();
        if (position_ < input_.size() && input_[position_] == '"') {
            return string();
        }
        const std::size_t start = position_;
        while (position_ < input_.size() && input_[position_] != ',' &&
               input_[position_] != '}' &&
               std::isspace(static_cast<unsigned char>(input_[position_])) == 0) {
            ++position_;
        }
        if (start == position_) {
            throw std::invalid_argument("expected JSON scalar");
        }
        return std::string(input_.substr(start, position_ - start));
    }

    [[nodiscard]] bool finished() {
        whitespace();
        return position_ == input_.size();
    }

private:
    std::string_view input_;
    std::size_t position_{};
};

class ExpressionParser final {
public:
    explicit ExpressionParser(const std::string_view expression)
        : expression_(expression) {}

    [[nodiscard]] double parse() {
        const double result = expression();
        whitespace();
        if (position_ != expression_.size() || !std::isfinite(result)) {
            throw std::invalid_argument("invalid calculator expression");
        }
        return result;
    }

private:
    void whitespace() {
        while (position_ < expression_.size() &&
               std::isspace(static_cast<unsigned char>(expression_[position_])) != 0) {
            ++position_;
        }
    }

    [[nodiscard]] bool consume(const char character) {
        whitespace();
        if (position_ < expression_.size() && expression_[position_] == character) {
            ++position_;
            return true;
        }
        return false;
    }

    [[nodiscard]] double expression() {
        double value = term();
        while (true) {
            if (consume('+')) {
                value += term();
            } else if (consume('-')) {
                value -= term();
            } else {
                break;
            }
        }
        return value;
    }

    [[nodiscard]] double term() {
        double value = factor();
        while (true) {
            if (consume('*')) {
                value *= factor();
            } else if (consume('/')) {
                const double divisor = factor();
                if (std::abs(divisor) < 1.0e-15) {
                    throw std::invalid_argument("division by zero");
                }
                value /= divisor;
            } else {
                break;
            }
        }
        return value;
    }

    [[nodiscard]] double factor() {
        whitespace();
        if (consume('+')) {
            return factor();
        }
        if (consume('-')) {
            return -factor();
        }
        if (consume('(')) {
            const double value = expression();
            if (!consume(')')) {
                throw std::invalid_argument("missing closing parenthesis");
            }
            return value;
        }
        return number();
    }

    [[nodiscard]] double number() {
        whitespace();
        const char* const begin = expression_.data() + position_;
        const char* const end = expression_.data() + expression_.size();
        double value{};
        const auto [parsed_end, error] = std::from_chars(begin, end, value);
        if (error != std::errc{} || parsed_end == begin) {
            throw std::invalid_argument("expected number in calculator expression");
        }
        position_ += static_cast<std::size_t>(parsed_end - begin);
        return value;
    }

    std::string_view expression_;
    std::size_t position_{};
};

[[nodiscard]] std::filesystem::path resolve_sandbox_path(
    const ToolPolicy& policy,
    const std::string_view requested
) {
    if (!policy.allow_file_reads || policy.sandbox_root.empty()) {
        throw std::runtime_error("file tools are disabled; configure --tool-root");
    }
    const std::filesystem::path relative(requested);
    if (relative.empty() || relative.is_absolute()) {
        throw std::runtime_error("tool paths must be non-empty and relative to the sandbox root");
    }
    const std::filesystem::path root = std::filesystem::weakly_canonical(policy.sandbox_root);
    const std::filesystem::path candidate = std::filesystem::weakly_canonical(root / relative);
    const std::filesystem::path relation = candidate.lexically_relative(root);
    if (relation.empty() || relation.is_absolute()) {
        throw std::runtime_error("tool path is outside the configured sandbox root");
    }
    for (const std::filesystem::path& component : relation) {
        if (component == "..") {
            throw std::runtime_error("tool path is outside the configured sandbox root");
        }
    }
    return candidate;
}

[[nodiscard]] bool parse_boolean_text(const std::string_view value) {
    return value == "true" || value == "false" || value == "1" || value == "0";
}

[[nodiscard]] bool parse_number_text(const std::string_view value) {
    double parsed{};
    const auto [end, error] = std::from_chars(
        value.data(), value.data() + value.size(), parsed
    );
    return error == std::errc{} && end == value.data() + value.size() &&
        std::isfinite(parsed);
}

[[nodiscard]] std::string local_time_text() {
    const std::time_t now = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now()
    );
    std::tm local{};
#if defined(_WIN32)
    if (localtime_s(&local, &now) != 0) {
        throw std::runtime_error("unable to determine local time");
    }
#else
    if (localtime_r(&now, &local) == nullptr) {
        throw std::runtime_error("unable to determine local time");
    }
#endif
    std::ostringstream output;
    output << std::put_time(&local, "%Y-%m-%d %H:%M:%S %Z");
    return output.str();
}

}  // namespace

ToolRuntime::ToolRuntime(ToolPolicy policy) : policy_(std::move(policy)) {
    if (policy_.maximum_calls_per_turn == 0U ||
        policy_.maximum_read_bytes == 0U ||
        policy_.maximum_directory_entries == 0U) {
        throw std::invalid_argument("invalid Solstice tool policy");
    }
}

void ToolRuntime::register_tool(ToolDefinition definition, Handler handler) {
    if (definition.name.empty() || !handler ||
        !std::isfinite(definition.cost) || definition.cost < 0.0) {
        throw std::invalid_argument("invalid Solstice tool definition");
    }
    if (handlers_.contains(definition.name)) {
        throw std::invalid_argument("duplicate Solstice tool name: " + definition.name);
    }
    for (const ToolArgumentSpec& argument : definition.arguments) {
        if (argument.name.empty() || argument.maximum_length == 0U) {
            throw std::invalid_argument("invalid Solstice tool argument definition");
        }
    }
    handlers_.emplace(definition.name, std::move(handler));
    definitions_.push_back(std::move(definition));
}

void ToolRuntime::register_safe_builtins() {
    register_tool(
        ToolDefinition{
            "calculator",
            "Evaluate a finite arithmetic expression using +, -, *, / and parentheses.",
            {ToolArgumentSpec{"expression", ToolArgumentType::string, true, 1'024U}},
            true,
            0.1,
        },
        [](const ToolCall& call, const ToolPolicy&) -> ToolResult {
            try {
                const double value = ExpressionParser(
                    call.arguments.at("expression")
                ).parse();
                std::ostringstream output;
                output << std::setprecision(15) << value;
                return ToolResult{true, output.str(), {}};
            } catch (const std::exception& error) {
                return ToolResult{false, {}, error.what()};
            }
        }
    );
    register_tool(
        ToolDefinition{
            "current_time",
            "Return the current local date and time.",
            {},
            true,
            0.1,
        },
        [](const ToolCall&, const ToolPolicy&) -> ToolResult {
            try {
                return ToolResult{true, local_time_text(), {}};
            } catch (const std::exception& error) {
                return ToolResult{false, {}, error.what()};
            }
        }
    );
    register_tool(
        ToolDefinition{
            "read_text_file",
            "Read a UTF-8 or plain-text file below the configured tool root.",
            {ToolArgumentSpec{"path", ToolArgumentType::path, true, 4'096U}},
            true,
            0.5,
        },
        [](const ToolCall& call, const ToolPolicy& policy) -> ToolResult {
            try {
                const std::filesystem::path path = resolve_sandbox_path(
                    policy, call.arguments.at("path")
                );
                if (!std::filesystem::is_regular_file(path)) {
                    return ToolResult{false, {}, "requested path is not a regular file"};
                }
                const std::uintmax_t size = std::filesystem::file_size(path);
                if (size > policy.maximum_read_bytes) {
                    return ToolResult{false, {}, "file exceeds tool read limit"};
                }
                std::ifstream input(path, std::ios::binary);
                if (!input) {
                    return ToolResult{false, {}, "unable to open requested file"};
                }
                std::string contents(
                    static_cast<std::size_t>(size), '\0'
                );
                if (!contents.empty()) {
                    input.read(contents.data(), static_cast<std::streamsize>(contents.size()));
                }
                if (!input) {
                    return ToolResult{false, {}, "failed while reading requested file"};
                }
                return ToolResult{true, std::move(contents), {}};
            } catch (const std::exception& error) {
                return ToolResult{false, {}, error.what()};
            }
        }
    );
    register_tool(
        ToolDefinition{
            "list_directory",
            "List entries in a directory below the configured tool root.",
            {ToolArgumentSpec{"path", ToolArgumentType::path, false, 4'096U}},
            true,
            0.3,
        },
        [](const ToolCall& call, const ToolPolicy& policy) -> ToolResult {
            try {
                const auto found = call.arguments.find("path");
                const std::string requested = found == call.arguments.end()
                    ? "."
                    : found->second;
                const std::filesystem::path path = resolve_sandbox_path(policy, requested);
                if (!std::filesystem::is_directory(path)) {
                    return ToolResult{false, {}, "requested path is not a directory"};
                }
                std::vector<std::string> entries;
                for (const std::filesystem::directory_entry& entry :
                     std::filesystem::directory_iterator(path)) {
                    std::string name = entry.path().filename().string();
                    if (entry.is_directory()) {
                        name += '/';
                    }
                    entries.push_back(std::move(name));
                    if (entries.size() >= policy.maximum_directory_entries) {
                        break;
                    }
                }
                std::sort(entries.begin(), entries.end());
                std::ostringstream output;
                for (const std::string& entry : entries) {
                    output << entry << '\n';
                }
                return ToolResult{true, output.str(), {}};
            } catch (const std::exception& error) {
                return ToolResult{false, {}, error.what()};
            }
        }
    );
}

std::span<const ToolDefinition> ToolRuntime::definitions() const noexcept {
    return definitions_;
}

const ToolPolicy& ToolRuntime::policy() const noexcept { return policy_; }

void ToolRuntime::set_policy(ToolPolicy policy) {
    if (policy.maximum_calls_per_turn == 0U || policy.maximum_read_bytes == 0U ||
        policy.maximum_directory_entries == 0U) {
        throw std::invalid_argument("invalid Solstice tool policy");
    }
    policy_ = std::move(policy);
}

std::optional<std::string> ToolRuntime::validate(const ToolCall& call) const {
    const auto definition = std::find_if(
        definitions_.begin(), definitions_.end(),
        [&call](const ToolDefinition& value) { return value.name == call.name; }
    );
    if (definition == definitions_.end()) {
        return "unknown tool: " + call.name;
    }
    for (const ToolArgumentSpec& specification : definition->arguments) {
        const auto found = call.arguments.find(specification.name);
        if (found == call.arguments.end()) {
            if (specification.required) {
                return "missing required argument: " + specification.name;
            }
            continue;
        }
        if (found->second.size() > specification.maximum_length) {
            return "argument exceeds maximum length: " + specification.name;
        }
        if (specification.type == ToolArgumentType::number &&
            !parse_number_text(found->second)) {
            return "argument is not a finite number: " + specification.name;
        }
        if (specification.type == ToolArgumentType::boolean &&
            !parse_boolean_text(found->second)) {
            return "argument is not a boolean: " + specification.name;
        }
        if (specification.type == ToolArgumentType::path) {
            const std::filesystem::path path(found->second);
            if (path.empty() || path.is_absolute()) {
                return "path arguments must be relative: " + specification.name;
            }
        }
    }
    for (const auto& [name, value] : call.arguments) {
        static_cast<void>(value);
        const bool known = std::any_of(
            definition->arguments.begin(), definition->arguments.end(),
            [&name](const ToolArgumentSpec& specification) {
                return specification.name == name;
            }
        );
        if (!known) {
            return "unknown tool argument: " + name;
        }
    }
    return std::nullopt;
}

ToolResult ToolRuntime::execute(const ToolCall& call) const {
    if (const auto error = validate(call); error.has_value()) {
        return ToolResult{false, {}, *error};
    }
    const auto handler = handlers_.find(call.name);
    if (handler == handlers_.end()) {
        return ToolResult{false, {}, "tool implementation is unavailable"};
    }
    return handler->second(call, policy_);
}

std::string ToolRuntime::serialize_call(const ToolCall& call) {
    std::ostringstream output;
    output << "{\"name\":\"" << json_escape(call.name) << "\",\"arguments\":{";
    bool first = true;
    for (const auto& [name, value] : call.arguments) {
        if (!first) {
            output << ',';
        }
        output << "\"" << json_escape(name) << "\":\""
               << json_escape(value) << "\"";
        first = false;
    }
    output << "}}";
    return output.str();
}

std::string ToolRuntime::serialize_result(const ToolResult& result) {
    std::ostringstream output;
    output << "{\"status\":\"" << (result.success ? "success" : "error") << "\"";
    if (result.success) {
        output << ",\"output\":\"" << json_escape(result.output) << "\"";
    } else {
        output << ",\"error\":\"" << json_escape(result.error) << "\"";
    }
    output << '}';
    return output.str();
}

ToolCall ToolRuntime::parse_call(const std::string_view json) {
    JsonCursor cursor(json);
    cursor.expect('{');
    ToolCall call;
    bool have_name = false;
    bool have_arguments = false;
    while (!cursor.consume('}')) {
        const std::string key = cursor.string();
        cursor.expect(':');
        if (key == "name") {
            call.name = cursor.string();
            have_name = true;
        } else if (key == "arguments") {
            cursor.expect('{');
            while (!cursor.consume('}')) {
                const std::string argument_name = cursor.string();
                cursor.expect(':');
                const std::string value = cursor.scalar();
                if (!call.arguments.emplace(argument_name, value).second) {
                    throw std::invalid_argument("duplicate tool-call argument");
                }
                if (!cursor.consume(',')) {
                    cursor.expect('}');
                    break;
                }
            }
            have_arguments = true;
        } else {
            throw std::invalid_argument("unknown tool-call JSON field");
        }
        if (!cursor.consume(',')) {
            cursor.expect('}');
            break;
        }
    }
    if (!cursor.finished() || !have_name || !have_arguments || call.name.empty()) {
        throw std::invalid_argument("incomplete tool-call JSON");
    }
    return call;
}

ToolRouter::ToolRouter(ToolRouterConfig config)
    : config_(config),
      indexed_keyword_updates_(indexed_tool_keyword_updates_from_environment()) {
    if (config_.maximum_tools == 0U || config_.maximum_keywords_per_tool == 0U ||
        !std::isfinite(config_.minimum_confidence) ||
        config_.minimum_confidence < 0.0 || config_.minimum_confidence > 1.0) {
        throw std::invalid_argument("invalid Solstice tool-router configuration");
    }
}

std::vector<std::string> ToolRouter::keywords(const std::string_view text) {
    static const std::unordered_set<std::string> stop_words{
        "a", "an", "the", "please", "can", "could", "you", "me", "my",
        "to", "for", "of", "and", "or", "is", "are", "what", "would",
    };
    std::vector<std::string> output;
    std::string word;
    const auto flush = [&output, &word]() {
        if (word.size() >= 2U && !stop_words.contains(word) &&
            std::find(output.begin(), output.end(), word) == output.end()) {
            output.push_back(word);
        }
        word.clear();
    };
    for (const char raw_character : text) {
        const unsigned char character = static_cast<unsigned char>(raw_character);
        if (std::isalnum(character) != 0 || character == static_cast<unsigned char>('_')) {
            word.push_back(static_cast<char>(std::tolower(character)));
        } else {
            flush();
        }
    }
    flush();
    return output;
}

void ToolRouter::train(
    const std::string_view request,
    const std::string_view tool_name
) {
    ++training_operation_stats_.training_rows;
    if (tool_name.empty()) {
        throw std::invalid_argument("tool-router training requires a tool name");
    }
    auto route = std::find_if(
        routes_.begin(), routes_.end(),
        [tool_name](const ToolRoute& value) { return value.tool_name == tool_name; }
    );
    if (route == routes_.end()) {
        ++training_operation_stats_.route_insert_attempts;
        if (routes_.size() >= config_.maximum_tools) {
            ++training_operation_stats_.route_capacity_failures;
            throw std::runtime_error("Solstice tool-router capacity exceeded");
        }
        routes_.push_back(ToolRoute{std::string(tool_name), 0U, {}});
        keyword_indexes_.emplace_back();
        ++training_operation_stats_.route_inserts;
        route = routes_.end() - 1;
    }
    const std::size_t route_index = static_cast<std::size_t>(
        std::distance(routes_.begin(), route)
    );
    ++route->examples;
    for (const std::string& keyword : keywords(request)) {
        ++training_operation_stats_.keyword_update_lookups;
        std::size_t keyword_position = route->keywords.size();
        if (indexed_keyword_updates_) {
            ++training_operation_stats_.indexed_keyword_lookups;
            const auto found = keyword_indexes_[route_index].find(keyword);
            if (found != keyword_indexes_[route_index].end()) {
                keyword_position = found->second;
            }
        } else {
            for (std::size_t index = 0U; index < route->keywords.size(); ++index) {
                ++training_operation_stats_.linear_keyword_comparisons;
                if (route->keywords[index].keyword == keyword) {
                    keyword_position = index;
                    break;
                }
            }
        }
        if (keyword_position == route->keywords.size()) {
            ++training_operation_stats_.keyword_insert_attempts;
            if (route->keywords.size() >= config_.maximum_keywords_per_tool) {
                ++training_operation_stats_.keyword_capacity_skips;
                continue;
            }
            route->keywords.push_back(ToolKeywordCount{keyword, 1U});
            keyword_position = route->keywords.size() - 1U;
            ++training_operation_stats_.keyword_inserts;
            if (indexed_keyword_updates_) {
                const auto inserted = keyword_indexes_[route_index].emplace(
                    route->keywords.back().keyword, keyword_position
                );
                if (!inserted.second) {
                    throw std::logic_error("failed to index Solstice tool keyword");
                }
                ++training_operation_stats_.keyword_index_incremental_inserts;
            }
        } else {
            ++route->keywords[keyword_position].count;
        }
    }
}

std::string ToolRouter::extract_quoted_or_path(const std::string_view request) {
    const std::size_t quote = request.find_first_of("\"'");
    if (quote != std::string_view::npos) {
        const char delimiter = request[quote];
        const std::size_t end = request.find(delimiter, quote + 1U);
        if (end != std::string_view::npos && end > quote + 1U) {
            return std::string(request.substr(quote + 1U, end - quote - 1U));
        }
    }
    const std::string lower = lowercase(request);
    for (const std::string_view marker : {" file ", " directory ", " folder ", " path "}) {
        const std::size_t position = lower.find(marker);
        if (position != std::string::npos) {
            std::string result(request.substr(position + marker.size()));
            while (!result.empty() &&
                   std::isspace(static_cast<unsigned char>(result.front())) != 0) {
                result.erase(result.begin());
            }
            while (!result.empty() &&
                   (std::isspace(static_cast<unsigned char>(result.back())) != 0 ||
                    result.back() == '.' || result.back() == '?')) {
                result.pop_back();
            }
            return result.empty() ? "." : result;
        }
    }
    return ".";
}

std::string ToolRouter::extract_expression(const std::string_view request) {
    std::string expression;
    bool started = false;
    for (const char character : request) {
        const bool accepted = std::isdigit(static_cast<unsigned char>(character)) != 0 ||
            character == '.' || character == '+' || character == '-' ||
            character == '*' || character == '/' || character == '(' ||
            character == ')' || std::isspace(static_cast<unsigned char>(character)) != 0;
        if (accepted) {
            if (std::isdigit(static_cast<unsigned char>(character)) != 0 ||
                character == '(' || character == '+' || character == '-') {
                started = true;
            }
            if (started) {
                expression.push_back(character);
            }
        } else if (started) {
            if (!expression.empty() &&
                std::isspace(static_cast<unsigned char>(expression.back())) == 0) {
                expression.push_back(' ');
            }
        }
    }
    while (!expression.empty() &&
           std::isspace(static_cast<unsigned char>(expression.back())) != 0) {
        expression.pop_back();
    }
    return expression;
}

ToolProposal ToolRouter::propose(
    const std::string_view request,
    const std::span<const ToolDefinition> available_tools
) const {
    const std::string lower = lowercase(request);
    const std::vector<std::string> request_keywords = keywords(request);
    const auto available = [&available_tools](const std::string_view name) {
        return std::any_of(
            available_tools.begin(), available_tools.end(),
            [name](const ToolDefinition& definition) {
                return definition.name == name;
            }
        );
    };

    std::string selected;
    double confidence = 0.0;
    std::string reason;
    if (available("calculator") &&
        (lower.find("calculate") != std::string::npos ||
         lower.find("compute") != std::string::npos ||
         lower.find("evaluate") != std::string::npos ||
         lower.find('*') != std::string::npos || lower.find('/') != std::string::npos)) {
        selected = "calculator";
        confidence = 0.96;
        reason = "arithmetic request";
    } else if (available("current_time") &&
               (lower.find("what time") != std::string::npos ||
                lower.find("current time") != std::string::npos ||
                lower.find("today's date") != std::string::npos ||
                lower.find("current date") != std::string::npos)) {
        selected = "current_time";
        confidence = 0.95;
        reason = "current date/time request";
    } else if (available("list_directory") &&
               (lower.find("list files") != std::string::npos ||
                lower.find("list directory") != std::string::npos ||
                lower.find("show files") != std::string::npos ||
                lower.find("folder contents") != std::string::npos)) {
        selected = "list_directory";
        confidence = 0.94;
        reason = "directory listing request";
    } else if (available("read_text_file") &&
               (lower.find("read file") != std::string::npos ||
                lower.find("open file") != std::string::npos ||
                lower.find("file contents") != std::string::npos ||
                lower.find("show me the file") != std::string::npos)) {
        selected = "read_text_file";
        confidence = 0.94;
        reason = "file reading request";
    }

    if (selected.empty()) {
        double best_score = 0.0;
        for (const ToolRoute& route : routes_) {
            if (!available(route.tool_name) || route.examples == 0U) {
                continue;
            }
            double score = 0.0;
            for (const std::string& keyword : request_keywords) {
                const auto found = std::find_if(
                    route.keywords.begin(), route.keywords.end(),
                    [&keyword](const ToolKeywordCount& value) {
                        return value.keyword == keyword;
                    }
                );
                if (found != route.keywords.end()) {
                    score += std::log1p(static_cast<double>(found->count));
                }
            }
            score /= std::max(1.0, std::sqrt(static_cast<double>(route.examples)));
            if (score > best_score ||
                (score == best_score && !selected.empty() && route.tool_name < selected)) {
                best_score = score;
                selected = route.tool_name;
            }
        }
        if (!selected.empty()) {
            confidence = 1.0 - std::exp(-best_score / 3.0);
            reason = "learned keyword resonance";
        }
    }

    ToolProposal proposal;
    proposal.should_call = !selected.empty() && confidence >= config_.minimum_confidence;
    proposal.confidence = confidence;
    proposal.reason = std::move(reason);
    if (!proposal.should_call) {
        return proposal;
    }
    proposal.call.name = selected;
    if (selected == "calculator") {
        proposal.call.arguments.emplace("expression", extract_expression(request));
    } else if (selected == "read_text_file" || selected == "list_directory") {
        proposal.call.arguments.emplace("path", extract_quoted_or_path(request));
    }
    return proposal;
}

std::span<const ToolRoute> ToolRouter::routes() const noexcept { return routes_; }
const ToolRouterConfig& ToolRouter::config() const noexcept { return config_; }

void ToolRouter::rebuild_keyword_indexes() {
    keyword_indexes_.clear();
    keyword_indexes_.resize(routes_.size());
    if (!indexed_keyword_updates_) {
        return;
    }
    ++training_operation_stats_.keyword_index_rebuilds;
    for (std::size_t route_index = 0U; route_index < routes_.size(); ++route_index) {
        auto& index = keyword_indexes_[route_index];
        index.reserve(routes_[route_index].keywords.size());
        for (std::size_t keyword_index = 0U;
             keyword_index < routes_[route_index].keywords.size();
             ++keyword_index) {
            const auto inserted = index.emplace(
                routes_[route_index].keywords[keyword_index].keyword,
                keyword_index
            );
            if (!inserted.second) {
                throw std::invalid_argument("duplicate keyword in Solstice tool route");
            }
            ++training_operation_stats_.keyword_index_entries_built;
        }
    }
}

void ToolRouter::set_indexed_keyword_updates(const bool enabled) {
    if (indexed_keyword_updates_ == enabled) {
        return;
    }
    indexed_keyword_updates_ = enabled;
    rebuild_keyword_indexes();
}

ToolRouterTrainingOperationStats
ToolRouter::training_operation_stats() const noexcept {
    return training_operation_stats_;
}

ToolRouterSnapshot ToolRouter::snapshot() const {
    return ToolRouterSnapshot{config_, routes_};
}

ToolRouter ToolRouter::from_snapshot(ToolRouterSnapshot snapshot) {
    ToolRouter router(snapshot.config);
    if (snapshot.routes.size() > snapshot.config.maximum_tools) {
        throw std::invalid_argument("invalid Solstice tool-router snapshot dimensions");
    }
    for (const ToolRoute& route : snapshot.routes) {
        if (route.tool_name.empty() ||
            route.keywords.size() > snapshot.config.maximum_keywords_per_tool) {
            throw std::invalid_argument("invalid Solstice tool-router snapshot route");
        }
    }
    router.routes_ = std::move(snapshot.routes);
    router.rebuild_keyword_indexes();
    return router;
}

std::uint64_t ToolRouter::deterministic_hash() const noexcept {
    std::uint64_t hash = fnv_offset_basis;
    std::vector<ToolRoute> routes = routes_;
    std::sort(
        routes.begin(), routes.end(),
        [](const ToolRoute& left, const ToolRoute& right) {
            return left.tool_name < right.tool_name;
        }
    );
    for (ToolRoute& route : routes) {
        hash_string(hash, route.tool_name);
        hash_u64(hash, route.examples);
        std::sort(
            route.keywords.begin(), route.keywords.end(),
            [](const ToolKeywordCount& left, const ToolKeywordCount& right) {
                return left.keyword < right.keyword;
            }
        );
        for (const ToolKeywordCount& keyword : route.keywords) {
            hash_string(hash, keyword.keyword);
            hash_u64(hash, keyword.count);
        }
    }
    return hash;
}

}  // namespace rlf::solstice
