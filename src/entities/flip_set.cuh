#pragma once

#include <cstdint>
#include "../config.cuh"

struct FlipSet {
    size_t size;
    uint32_t pairs[MAX_PAIRS];

    __device__ __host__ FlipSet();

    __device__ __host__ void add(uint32_t index1, uint32_t index2);
    __device__ __host__ void remove(uint32_t index1, uint32_t index2);
    __device__ __host__ void remove(uint32_t index);
    __device__ __host__ void clear();

    __device__ __host__ uint32_t index1(size_t i) const;
    __device__ __host__ uint32_t index2(size_t i) const;
};
