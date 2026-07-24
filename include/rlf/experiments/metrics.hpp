#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>

namespace rlf::experiments {

struct ExperimentMetrics final {
    double task_accuracy{0.0};
    double one_shot_recall{0.0};
    double retained_accuracy{0.0};
    double catastrophic_forgetting_score{0.0};
    double compositional_generalization_score{0.0};
    double prediction_error{0.0};
    double average_settling_cycles{0.0};
    double average_modes_retrieved{0.0};
    double average_modes_activated{0.0};
    std::uint64_t modes_created{0ULL};
    std::uint64_t modes_split{0ULL};
    std::uint64_t modes_merged{0ULL};
    std::uint64_t modes_pruned{0ULL};
    std::size_t bytes_stored{0U};
    std::size_t peak_resident_bytes{0U};
    double training_seconds{0.0};
    double inference_seconds{0.0};
    double training_examples_per_second{0.0};
    double inference_examples_per_second{0.0};
    double update_operations_per_example{0.0};
    double active_operations_per_inference{0.0};
    double efficiency_score{0.0};
    std::uint64_t deterministic_run_hash{0ULL};
};

[[nodiscard]] std::size_t peak_resident_memory_bytes() noexcept;
[[nodiscard]] double provisional_efficiency_score(
    double useful_retained_transformations,
    std::size_t stored_bytes,
    double active_operations,
    double training_seconds
) noexcept;
void write_metrics_json(
    std::ostream& output,
    const ExperimentMetrics& metrics,
    std::size_t indentation = 0U
);

}  // namespace rlf::experiments
