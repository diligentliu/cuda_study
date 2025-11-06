#include "cuda_study/util/perf_util.h"

#include <stdexcept>
#include <thread>
#include <vector>

namespace util {
void perf_single_threaded(const std::function<void()>& func,
                          int iterations,
                          float& avg_time_ms,
                          float& throughput_ops_per_sec,
                          int warmup_iterations) {
    if (iterations <= 0) {
        throw std::invalid_argument("Iterations must be positive.");
    }

    // Warm-up run
    for (int i = 0; i < warmup_iterations; ++i) {
        func();
    }

    // Measure time
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> total_duration = end - start;
    avg_time_ms = static_cast<float>(total_duration.count()) / iterations;
    throughput_ops_per_sec = static_cast<float>(iterations) / (total_duration.count() / 1000.0f);
}

void perf_multi_threaded(const std::function<void()>& func,
                         int iterations,
                         int num_threads,
                         float& avg_time_ms,
                         float& throughput_ops_per_sec,
                         int warmup_iterations) {
    if (iterations <= 0 || num_threads <= 0) {
        throw std::invalid_argument("Iterations and number of threads must be positive.");
    }

    // Warm-up run
    for (int i = 0; i < warmup_iterations; ++i) {
        func();
    }

    // Measure time
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    int iterations_per_thread = iterations / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            int start_iter = t * iterations_per_thread;
            int end_iter = (t == num_threads - 1) ? iterations : start_iter + iterations_per_thread;
            for (int i = start_iter; i < end_iter; ++i) {
                func();
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> total_duration = end - start;
    avg_time_ms = static_cast<float>(total_duration.count()) / iterations;
    throughput_ops_per_sec = static_cast<float>(iterations) / (total_duration.count() / 1000.0f);
}
}  // namespace util
