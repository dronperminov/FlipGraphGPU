#include "flip_set.cuh"

__device__ __host__ FlipSet::FlipSet() {
    size = 0;
}

__device__ __host__ void FlipSet::add(uint32_t index1, uint32_t index2) {
    uint32_t pair = (index1 << 16) | index2;
    pairs[size++] = pair;
}

__device__ __host__ void FlipSet::remove(uint32_t index1, uint32_t index2) {
    uint32_t pair1 = (index1 << 16) | index2;
    uint32_t pair2 = (index2 << 16) | index1;

    for (size_t i = 0; i < size; i++) {
        if (pairs[i] == pair1 || pairs[i] == pair2) {
            pairs[i] = pairs[--size];
            return;
        }
    }
}

__device__ __host__ void FlipSet::remove(uint32_t index) {
    for (size_t i = 0; i < size; i++)
        if (index1(i) == index || index2(i) == index)
            pairs[i--] = pairs[--size];
}

__device__ __host__ void FlipSet::clear() {
    size = 0;
}

__device__ __host__ uint32_t FlipSet::index1(size_t i) const {
    return pairs[i] >> 16;
}

__device__ __host__ uint32_t FlipSet::index2(size_t i) const {
    return pairs[i] & 0xFFFF;
}
