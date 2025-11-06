#pragma once

#include <functional>

namespace util {
float measure_time(const std::function<void()>& func);
void perf_single_threaded(const std::function<void()>& func,
                          int iterations,
                          float& avg_time_ms,
                          float& throughput_ops_per_sec,
                          int warmup_iterations = 5);
void perf_multi_threaded(const std::function<void()>& func,
                         int iterations,
                         int num_threads,
                         float& avg_time_ms,
                         float& throughput_ops_per_sec,
                         int warmup_iterations = 5);
}  // namespace util
