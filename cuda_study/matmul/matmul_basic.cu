#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>
#include <glog/logging.h>

#include "cuda_study/util/util.h"
#include "cuda_study/util/perf_util.h"

// C[M,N] = A[M,K] * B[K,N]
__global__ void MatMul(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float value = 0;
        for (int k = 0; k < K; k++) {
            value += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

int main() {
    int M = 50;
    int N = 1024;
    int K = 1024;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);
    std::vector<float> h_C_host(M * N);

    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 0.2f);
    for (int i = 0; i < M * K; i++) {
        h_A[i] = dist(gen);
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = dist(gen);
    }

    auto host_test_func = [&]() {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float value = 0;
                for (int k = 0; k < K; k++) {
                    value += h_A[i * K + k] * h_B[k * N + j];
                }
                h_C_host[i * N + j] = value;
            }
        }
    };
    float avg_time, throughput;
    util::perf_single_threaded(host_test_func, 10, avg_time, throughput, 10);
    LOG(INFO) << "Host matrix multiplication time: " << avg_time << " ms";
    LOG(INFO) << "Throughput: " << throughput << " GFLOPS";

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice);

    constexpr int BLOCK_SIZE = 32;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    auto device_test_func = [&]() {
        MatMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    float device_avg_time, device_throughput;
    util::perf_single_threaded(device_test_func, 10, device_avg_time, device_throughput, 10);
    LOG(INFO) << "Device matrix multiplication time: " << device_avg_time << " ms";
    LOG(INFO) << "Throughput: " << device_throughput << " GFLOPS";

    cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost);

    if (!util::compare_diff(h_C_host, h_C, M, N, 1e-2f)) {
        LOG(ERROR) << "Result verification failed!";
        goto cleanup;
    }
    LOG(INFO) << "Host time: " << avg_time << " ms";
    LOG(INFO) << "Device time: " << device_avg_time << " ms";

cleanup:
    util::show_matrix(h_C, M, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}