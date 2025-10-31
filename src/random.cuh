#pragma once

#include <curand_kernel.h>

__forceinline__ __device__ int randint(int a, int b, curandState &state) {
    return a + curand(&state) % (b - a + 1);
}

__forceinline__ __device__ void randomPermutation(int *array, int n, curandState &state) {
    #pragma unroll
    for (int i = 0; i < n; i++)
        array[i] = i;

    #pragma unroll
    for (int i = n - 1; i > 0; i--) {
        int j = curand(&state) % (i + 1);

        int tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }
}

__forceinline__ __device__ void randomMatrixZ2(int n, int *matrix, curandState &state) {
    #pragma unroll
    for (int i = 0; i < n; i++) {
        unsigned int rnd = curand(&state);

        for (int j = 0; j < n; j++)
            matrix[i * n + j] = (rnd >> j) & 1;
    }
}
