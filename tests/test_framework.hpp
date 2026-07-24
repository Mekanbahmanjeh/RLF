#pragma once

#include <cmath>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rlf::test {

using TestFunction = void (*)();

struct TestCase final {
    std::string name;
    TestFunction function;
};

[[nodiscard]] inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> test_cases;
    return test_cases;
}

class Registrar final {
public:
    Registrar(std::string name, const TestFunction function) {
        registry().push_back({std::move(name), function});
    }
};

inline void fail(
    const std::string_view expression,
    const std::string_view file,
    const int line,
    const std::string_view detail = {}
) {
    std::ostringstream message;
    message << file << ':' << line << ": check failed: " << expression;
    if (!detail.empty()) {
        message << " (" << detail << ')';
    }
    throw std::runtime_error(message.str());
}

inline void check(
    const bool condition,
    const std::string_view expression,
    const std::string_view file,
    const int line
) {
    if (!condition) {
        fail(expression, file, line);
    }
}

template <typename Actual, typename Expected, typename Tolerance>
void check_near(
    const Actual actual,
    const Expected expected,
    const Tolerance tolerance,
    const std::string_view expression,
    const std::string_view file,
    const int line
) {
    const auto difference = std::abs(actual - expected);
    if (difference > tolerance) {
        std::ostringstream detail;
        detail << "actual=" << actual << ", expected=" << expected
               << ", tolerance=" << tolerance;
        fail(expression, file, line, detail.str());
    }
}

inline int run_all() {
    std::size_t passed = 0U;
    std::size_t failed = 0U;
    std::string filter_storage;
#ifdef _WIN32
    char* filter_value = nullptr;
    std::size_t filter_size = 0U;
    if (_dupenv_s(&filter_value, &filter_size, "RLF_TEST_FILTER") == 0 &&
        filter_value != nullptr) {
        filter_storage.assign(filter_value);
    }
    std::free(filter_value);
#else
    const char* filter_value = std::getenv("RLF_TEST_FILTER");
    if (filter_value != nullptr) {
        filter_storage.assign(filter_value);
    }
#endif
    const std::string_view filter(filter_storage);
    for (const TestCase& test_case : registry()) {
        if (!filter.empty() && test_case.name.find(filter) == std::string::npos) {
            continue;
        }
        try {
            test_case.function();
            ++passed;
            std::cout << "[PASS] " << test_case.name << '\n' << std::flush;
        } catch (const std::exception& error) {
            ++failed;
            std::cerr << "[FAIL] " << test_case.name << ": "
                      << error.what() << '\n' << std::flush;
        } catch (...) {
            ++failed;
            std::cerr << "[FAIL] " << test_case.name
                      << ": unknown exception\n" << std::flush;
        }
    }
    std::cout << passed << " passed, " << failed << " failed\n" << std::flush;
    return failed == 0U ? 0 : 1;
}

}  // namespace rlf::test

#define RLF_TEST_CONCAT_IMPL(left, right) left##right
#define RLF_TEST_CONCAT(left, right) RLF_TEST_CONCAT_IMPL(left, right)
#define RLF_TEST_CASE(name)                                                   \
    static void RLF_TEST_CONCAT(rlf_test_function_, __LINE__)();              \
    [[maybe_unused]] static const ::rlf::test::Registrar                      \
        RLF_TEST_CONCAT(rlf_test_registrar_, __LINE__){                       \
            name, &RLF_TEST_CONCAT(rlf_test_function_, __LINE__)};            \
    static void RLF_TEST_CONCAT(rlf_test_function_, __LINE__)()

#define RLF_CHECK(expression)                                                 \
    ::rlf::test::check(                                                       \
        static_cast<bool>(expression), #expression, __FILE__, __LINE__        \
    )

#define RLF_CHECK_NEAR(actual, expected, tolerance)                           \
    ::rlf::test::check_near(                                                  \
        (actual), (expected), (tolerance),                                    \
        #actual " ~= " #expected, __FILE__, __LINE__                          \
    )

#define RLF_CHECK_THROWS_AS(expression, exception_type)                       \
    do {                                                                      \
        bool rlf_expected_exception_thrown = false;                           \
        try {                                                                 \
            static_cast<void>(expression);                                    \
        } catch (const exception_type&) {                                     \
            rlf_expected_exception_thrown = true;                             \
        }                                                                     \
        ::rlf::test::check(                                                   \
            rlf_expected_exception_thrown,                                    \
            #expression " throws " #exception_type, __FILE__, __LINE__        \
        );                                                                    \
    } while (false)
