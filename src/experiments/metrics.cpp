#include "rlf/experiments/metrics.hpp"

#include "rlf/experiments/phase_vector_smoke.hpp"

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <ostream>
#include <string>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace rlf::experiments {

std::size_t peak_resident_memory_bytes() noexcept {
#if defined(__linux__) || defined(__APPLE__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0U;
    }
#if defined(__APPLE__)
    return static_cast<std::size_t>(usage.ru_maxrss);
#else
    constexpr std::size_t bytes_per_kibibyte = 1'024U;
    return static_cast<std::size_t>(usage.ru_maxrss) *
        bytes_per_kibibyte;
#endif
#else
    return 0U;
#endif
}

double provisional_efficiency_score(
    const double useful_retained_transformations,
    const std::size_t stored_bytes,
    const double active_operations,
    const double training_seconds
) noexcept {
    if (!std::isfinite(useful_retained_transformations) ||
        useful_retained_transformations <= 0.0 ||
        stored_bytes == 0U ||
        !std::isfinite(active_operations) ||
        active_operations <= 0.0 ||
        !std::isfinite(training_seconds) ||
        training_seconds <= 0.0) {
        return 0.0;
    }
    return useful_retained_transformations /
        (static_cast<double>(stored_bytes) *
         active_operations *
         training_seconds);
}

void write_metrics_json(
    std::ostream& output,
    const ExperimentMetrics& metrics,
    const std::size_t indentation
) {
    const std::string indent(indentation, ' ');
    const std::string field_indent(indentation + 2U, ' ');
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << indent << "{\n"
           << field_indent << "\"task_accuracy\": "
           << metrics.task_accuracy << ",\n"
           << field_indent << "\"one_shot_recall\": "
           << metrics.one_shot_recall << ",\n"
           << field_indent << "\"retained_accuracy\": "
           << metrics.retained_accuracy << ",\n"
           << field_indent << "\"catastrophic_forgetting_score\": "
           << metrics.catastrophic_forgetting_score << ",\n"
           << field_indent
           << "\"compositional_generalization_score\": "
           << metrics.compositional_generalization_score << ",\n"
           << field_indent << "\"prediction_error\": "
           << metrics.prediction_error << ",\n"
           << field_indent << "\"average_settling_cycles\": "
           << metrics.average_settling_cycles << ",\n"
           << field_indent << "\"average_modes_retrieved\": "
           << metrics.average_modes_retrieved << ",\n"
           << field_indent << "\"average_modes_activated\": "
           << metrics.average_modes_activated << ",\n"
           << field_indent << "\"modes_created\": "
           << metrics.modes_created << ",\n"
           << field_indent << "\"modes_split\": "
           << metrics.modes_split << ",\n"
           << field_indent << "\"modes_merged\": "
           << metrics.modes_merged << ",\n"
           << field_indent << "\"modes_pruned\": "
           << metrics.modes_pruned << ",\n"
           << field_indent << "\"bytes_stored\": "
           << metrics.bytes_stored << ",\n"
           << field_indent << "\"peak_resident_bytes\": "
           << metrics.peak_resident_bytes << ",\n"
           << field_indent << "\"training_seconds\": "
           << metrics.training_seconds << ",\n"
           << field_indent << "\"inference_seconds\": "
           << metrics.inference_seconds << ",\n"
           << field_indent << "\"training_examples_per_second\": "
           << metrics.training_examples_per_second << ",\n"
           << field_indent << "\"inference_examples_per_second\": "
           << metrics.inference_examples_per_second << ",\n"
           << field_indent << "\"update_operations_per_example\": "
           << metrics.update_operations_per_example << ",\n"
           << field_indent << "\"active_operations_per_inference\": "
           << metrics.active_operations_per_inference << ",\n"
           << field_indent << "\"efficiency_score\": "
           << metrics.efficiency_score << ",\n"
           << field_indent << "\"deterministic_run_hash\": \""
           << format_run_hash(metrics.deterministic_run_hash)
           << "\"\n"
           << indent << '}';
}

}  // namespace rlf::experiments
