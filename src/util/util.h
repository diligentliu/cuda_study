#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <iomanip>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr,                                                \
                    "CUDA error at %s:%d: %s (%d)\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err), err);     \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

namespace util {

template <typename T>
void show_matrix(const std::vector<T>& matrix, int M, int N, int n = 3) {
    std::cout << std::fixed << std::setprecision(5);
    if (M <= 0 || N <= 0 || matrix.size() != M * N) {
        throw std::invalid_argument("Invalid matrix dimensions or size.");
    }

    bool show_full_col = (2 * n >= N);
    bool show_full_row = (2 * n >= M);

    for (int i = 0; i < M; i++) {
        if (!show_full_row && i == n) {
            std::cout << " ...\n";
            std::cout << " ...\n";
            std::cout << " ...\n";
            i = M - n - 1;
            continue;
        }

        for (int j = 0; j < N; j++) {
            if (!show_full_col && j == n) {
                std::cout << " ... ";
                j = N - n - 1;
                continue;
            }
            std::cout << std::setw(6) << matrix[i * N + j];
            if (j != N - 1) {
                std::cout << " ";
            }
        }
        std::cout << "\n";
    }
}

template <typename T>
bool compare_diff(const std::vector<T>& a,
                  const std::vector<T>& b,
                  int M, int N,
                  T tol = static_cast<T>(1e-5)) {
    if (a.size() != b.size() || a.size() != M * N) {
        throw std::invalid_argument("Vectors must have the same size and match dimensions.");
    }

    // int、int64、int32 类型比较
    if (std::is_integral<T>::value) {
        for (int i = 0; i < M * N; i++) {
            if (a[i] != b[i]) {
                std::cout << "Difference found at index " << i
                          << ": a = " << a[i]
                          << ", b = " << b[i] << "\n";
                return false;
            }
        }
        return true;
    } else if (std::is_floating_point<T>::value) {
        // 浮点数类型比较有效位数
        for (int i = 0; i < M * N; i++) {
            float diff = std::fabs(a[i] - b[i]);
            float rel_diff = diff / (std::fabs(b[i]) + 1e-8f); // 防止除以零

            if (rel_diff > tol) {
                std::cout << "Difference found at index " << i
                          << ": a = " << a[i]
                          << ", b = " << b[i]
                          << ", abs diff = " << diff
                          << ", rel diff = " << rel_diff << "\n";
                return false;
            }
        }
        return true;
    } else {
        throw std::invalid_argument("Unsupported data type for comparison.");
    }
}

}  // namespace util
