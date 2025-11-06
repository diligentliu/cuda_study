#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <glog/logging.h>

#include "cuda_study/util/util.h"
#include "cuda_study/util/perf_util.h"

template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void MatMul_Basic(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float sA[BLOCK_M][BLOCK_K];
    __shared__ float sB[BLOCK_K][BLOCK_N];

    int row = blockIdx.y * BLOCK_M + threadIdx.y;
    int col = blockIdx.x * BLOCK_N + threadIdx.x;

    float value = 0;

    const int load_sA_per_thread = (BLOCK_K + BLOCK_N - 1) / BLOCK_N;
    const int load_sB_per_thread = (BLOCK_K + BLOCK_M - 1) / BLOCK_M;

    for (int k = 0; k < (K + BLOCK_K - 1) / BLOCK_K; k++) {
        #pragma unroll
        for (int i = 0; i < load_sA_per_thread; i++) {
            int k_index = threadIdx.x + i * BLOCK_N;
            int a_col = k * BLOCK_K + k_index;
            if (k_index < BLOCK_K) {
                sA[threadIdx.y][k_index] = (row < M && a_col < K) ?
                    A[row * K + a_col] : 0.0f;
            }
        }

        #pragma unroll
        for (int i = 0; i < load_sB_per_thread; i++) {
            int k_index = threadIdx.y + i * BLOCK_M;
            int b_row = k * BLOCK_K + k_index;
            if (k_index < BLOCK_K) {
                sB[k_index][threadIdx.x] = (col < N && b_row < K) ?
                    B[b_row * N + col] : 0.0f;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int n = 0; n < BLOCK_K; ++n)
            value += sA[threadIdx.y][n] * sB[n][threadIdx.x];

        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N>
__global__ void MatMul(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float sA[BLOCK_M][BLOCK_K];
    __shared__ float sB[BLOCK_K][BLOCK_N];

    int row = blockIdx.y * BLOCK_M + threadIdx.y * THREAD_M;
    int col = blockIdx.x * BLOCK_N + threadIdx.x * THREAD_N;

    float values[THREAD_M][THREAD_N] = {0};

    const int THREADS_PER_BLOCK_M = BLOCK_M / THREAD_M;
    const int THREADS_PER_BLOCK_N = BLOCK_N / THREAD_N;
    const int load_sA_per_thread = (BLOCK_K + THREADS_PER_BLOCK_N - 1) / THREADS_PER_BLOCK_N;
    const int load_sB_per_thread = (BLOCK_K + THREADS_PER_BLOCK_M - 1) / THREADS_PER_BLOCK_M;

    for (int k = 0; k < (K + BLOCK_K - 1) / BLOCK_K; k++) {
        #pragma unroll
        for (int i = 0; i < THREAD_M; i++) {
            #pragma unroll
            for (int j = 0; j < load_sA_per_thread; j++) {
                int k_index = threadIdx.x * load_sA_per_thread + j;
                int a_col = k * BLOCK_K + k_index;
                if (k_index < BLOCK_K) {
                    sA[threadIdx.y * THREAD_M + i][k_index] = (row + i < M && a_col < K) ?
                        A[(row + i) * K + a_col] : 0.0f;
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < THREAD_N; i++) {
            #pragma unroll
            for (int j = 0; j < load_sB_per_thread; j++) {
                int k_index = threadIdx.y * load_sB_per_thread + j;
                int b_row = k * BLOCK_K + k_index;
                if (k_index < BLOCK_K) {
                    sB[k_index][threadIdx.x * THREAD_N + i] = (col + i < N && b_row < K) ?
                        B[b_row * N + (col + i)] : 0.0f;
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int n = 0; n < BLOCK_K; ++n) {
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_N; j++) {
                    values[i][j] += sA[threadIdx.y * THREAD_M + i][n] * sB[n][threadIdx.x * THREAD_N + j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            if (row + i < M && col + j < N) {
                C[(row + i) * N + (col + j)] = values[i][j];
            }
        }
    }
}

int main() {
    int M = 1024;
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

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice);

    float* d_C_basic;
    cudaMalloc((void**)&d_C_basic, size_C);
    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;
    constexpr int BLOCK_K = 64;
    dim3 threadsPerBlock(BLOCK_N, BLOCK_M);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    auto basic_perf_func = [&]() {
        MatMul_Basic<BLOCK_M, BLOCK_N, BLOCK_K><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C_basic, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    float avg_time_basic, throughput_basic;
    util::perf_single_threaded(basic_perf_func, 10, avg_time_basic, throughput_basic, 10);
    cudaMemcpy(h_C_host.data(), d_C_basic, size_C, cudaMemcpyDeviceToHost);
    LOG(INFO) << "Basic matrix multiplication time: " << avg_time_basic << " ms";
    LOG(INFO) << "Throughput: " << throughput_basic << " GFLOPS";

    constexpr int TILED_BLOCK_M = 128;
    constexpr int TILED_BLOCK_N = 128;
    constexpr int TILED_BLOCK_K = 16;
    constexpr int THREAD_M = 8;
    constexpr int THREAD_N = 8;
    threadsPerBlock = dim3(TILED_BLOCK_N / THREAD_N, TILED_BLOCK_M / THREAD_M);
    blocksPerGrid = dim3((N + TILED_BLOCK_N - 1) / TILED_BLOCK_N,
                         (M + TILED_BLOCK_M - 1) / TILED_BLOCK_M);
    auto tiled_perf_func = [&]() {
        MatMul<TILED_BLOCK_M, TILED_BLOCK_N, TILED_BLOCK_K, THREAD_M, THREAD_N><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    float avg_time_shared, throughput_shared;
    util::perf_single_threaded(tiled_perf_func, 10, avg_time_shared, throughput_shared, 10);

    LOG(INFO) << "Tiled matrix multiplication time: " << avg_time_shared << " ms";
    LOG(INFO) << "Throughput: " << throughput_shared << " GFLOPS";

    cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost);

    LOG(INFO) << "Basic time: " << avg_time_basic << " ms";
    LOG(INFO) << "Tiled time: " << avg_time_shared << " ms";

    if (!util::compare_diff(h_C_host, h_C, M, N, 1e-3f)) {
        LOG(ERROR) << "Result verification failed!";
        goto cleanup;
    }

cleanup:
    util::show_matrix(h_C, M, N);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_basic);
    cudaFree(d_C);
    return 0;
}