#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

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
    for (int i = 0; i < N; i++) {
        h_A[i] = random() % 100 / 10.0f;
        h_B[i] = random() % 100 / 10.0f;
    }

    // host time cost test
    std::vector<float> h_C_host(N);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        h_C_host[i] = h_A[i] + h_B[i];
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> host_duration = end - start;
    std::cout << "Host vector addition time: " << host_duration.count() << " ms" << std::endl;
    for (int i = 0; i < 20; i++) {
        std::cout << "Host C[" << i << "] = " << h_C_host[i] << std::endl;
    }

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
    start = std::chrono::high_resolution_clock::now();
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 等待 GPU 执行完毕
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> device_duration = end - start;
    std::cout << "Device vector addition time: " << device_duration.count() << " ms" << std::endl;
    // 从 GPU 拷贝结果回主机
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    // 验证结果
    for (int i = 0; i < 20; i++) {
        std::cout << "C[" << i << "] = " << h_C[i] << std::endl;
    }

    // test difference
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_C_host[i] - h_C[i]) > 1e-5) {
            correct = false;
            std::cout << "Mismatch at index " << i << ": host " << h_C_host[i] << " vs device " << h_C[i] << std::endl;
            break;
        }
    }
    if (correct) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }
    // show time comparison
    std::cout << "Host time: " << host_duration.count() << " ms,\nDevice time: " << device_duration.count() << " ms" << std::endl;

    // 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << "Vector addition completed successfully!" << std::endl;
    return 0;
}
