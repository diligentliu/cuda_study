#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

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
    int M = 50; // 行数
    int N = 128; // 列数
    int K = 256; // 共享维度

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);
    std::vector<float> h_C_host(M * N);

    // 初始化输入数据
    for (int i = 0; i < M * K; i++) {
        h_A[i] = 1.0f;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = 1.0f;
    }

    // host time cost test
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float value = 0;
            for (int k = 0; k < K; k++) {
                value += h_A[i * K + k] * h_B[k * N + j];
            }
            h_C_host[i * N + j] = value;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> host_duration = end - start;
    std::cout << "Host matrix multiplication time: " << host_duration.count() << " ms" << std::endl;

    // 在设备端分配内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // 将数据拷贝到 GPU
    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice);

    // 设置 CUDA kernel 启动参数
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    start = std::chrono::high_resolution_clock::now();
    // 启动 kernel
    MatMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> device_duration = end - start;
    std::cout << "Device matrix multiplication time: " << device_duration.count() << " ms" << std::endl;
    // 将结果拷贝回主机
    cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(h_C_host[i * N + j] - h_C[i * N + j]) > 1e-3) {
                std::cout << "Mismatch at C[" << i << "," << j << "]: host " << h_C_host[i * N + j]
                          << " vs device " << h_C[i * N + j] << std::endl;
            }
        }
    }
    // show time comparison
    std::cout << "Host time: " << host_duration.count() << " ms,\nDevice time: " << device_duration.count() << " ms" << std::endl;
    // 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}