#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

#include "src/util/util.h"
#include "src/util/perf_util.h"

// CUDA kernel：每个线程计算一个元素
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

int main() {
    int N = 1 << 20;
    size_t size = N * sizeof(float);

    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_C(N);

    // 初始化输入数据
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 0.2f);
    for (int i = 0; i < N; i++) {
        h_A[i] = dist(gen);
        h_B[i] = dist(gen);
    }

    // host time cost test
    std::vector<float> h_C_host(N);
    auto host_test_func = [&]() {
        for (int i = 0; i < N; i++) {
            h_C_host[i] = h_A[i] + h_B[i];
        }
    };
    float avg_time_host, throughput_host;
    util::perf_single_threaded(host_test_func, 10, avg_time_host, throughput_host, 10);
    std::cout << "Host vector addition time: " << avg_time_host
              << " ms\nThroughput: " << throughput_host << " GFLOPS" << std::endl;

    // 在设备端分配内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // 将数据拷贝到 GPU
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    // 设置 CUDA kernel 启动参数
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 启动 kernel
    auto device_test_func = [&]() {
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    float device_avg_time, device_throughput;
    util::perf_single_threaded(device_test_func, 10, device_avg_time, device_throughput, 10);
    std::cout << "Device vector addition time: " << device_avg_time
              << " ms\nThroughput: " << device_throughput << " GFLOPS" << std::endl;
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    // 验证结果
    util::show_matrix(h_C_host, 1, N, 5);
    util::show_matrix(h_C, 1, N, 5);

    // show time comparison
    std::cout << "Host time: " << avg_time_host
              << " ms,\nDevice time: " << device_avg_time << " ms" << std::endl;
    // 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << "Vector addition completed successfully!" << std::endl;
    return 0;
}
