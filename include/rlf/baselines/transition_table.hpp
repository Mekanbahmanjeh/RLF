#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace rlf::baselines {

class TransitionTablePredictor final {
public:
    void observe(std::uint64_t current, std::uint64_t next);
    [[nodiscard]] std::optional<std::uint64_t> predict(
        std::uint64_t current
    ) const;
    [[nodiscard]] std::vector<
        std::pair<std::uint64_t, std::uint64_t>
    > counts(std::uint64_t current) const;
    [[nodiscard]] std::size_t contexts() const noexcept;
    [[nodiscard]] std::size_t transitions() const noexcept;
    [[nodiscard]] std::size_t bytes_stored() const noexcept;

private:
    std::map<
        std::uint64_t,
        std::map<std::uint64_t, std::uint64_t>
    > counts_;
    std::size_t observations_{0U};
};

}  // namespace rlf::baselines
