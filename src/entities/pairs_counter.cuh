#pragma once

#include <iostream>
#include <curand_kernel.h>

#include "../config.cuh"

struct Pair {
    int i;
    int j;
    int count;
};

template <size_t maxPairs>
class PairsCounter {
    Pair pairs[maxPairs];
    int size;
    int topSize;
public:
    __device__ __host__ PairsCounter();

    __device__ __host__ void insert(int i, int j);
    __device__ __host__ void sort();
    __device__ __host__ void clear();
    __device__ __host__ void copyFrom(const PairsCounter<maxPairs> &counter);

    __device__ __host__ Pair getTop() const;
    __device__ Pair getGreedyRandom(curandState &state) const;
    __device__ Pair getRandom(curandState &state) const;
    __device__ Pair getTopRandom(curandState &state) const;

    __device__ __host__ int count() const;
    __device__ __host__ operator bool() const;
};

template <size_t maxPairs>
__device__ __host__ PairsCounter<maxPairs>::PairsCounter() {
    size = 0;
    topSize = 0;
}

template <size_t maxPairs>
__device__ __host__ void PairsCounter<maxPairs>::insert(int i, int j) {
    if (abs(i) > abs(j)) {
        int tmp = i;
        i = j;
        j = tmp;
    }

    if (i < 0) {
        i = -i;
        j = -j;
    }

    int index = 0;

    while (index < size && !(pairs[index].i == i && pairs[index].j == j))
        index++;

    if (index < size) {
        pairs[index].count++;
    }
    else {
        pairs[size].i = i;
        pairs[size].j = j;
        pairs[size++].count = 1;
    }
}

template <size_t maxPairs>
__device__ __host__ void PairsCounter<maxPairs>::sort() {
    int gaps[] = {701, 301, 132, 57, 23, 10, 4, 1};

    for (int g = 0; g < 8; g++) {
        int gap = gaps[g];

        if (gap > size)
            continue;

        for (int i = gap; i < size; i++) {
            Pair tmp = pairs[i];
            int j = i;

            while (j >= gap && pairs[j - gap].count < tmp.count) {
                pairs[j] = pairs[j - gap];
                j -= gap;
            }

            pairs[j] = tmp;
        }
    }

    topSize = 0;

    while (topSize < size && pairs[topSize].count == pairs[0].count)
        topSize++;
}

template <size_t maxPairs>
__device__ __host__ void PairsCounter<maxPairs>::clear() {
    size = 0;
    topSize = 0;
}

template <size_t maxPairs>
__device__ __host__ void PairsCounter<maxPairs>::copyFrom(const PairsCounter<maxPairs> &counter) {
    size = counter.size;
    topSize = counter.topSize;

    for (int index = 0; index < size; index++) {
        pairs[index].i = counter.pairs[index].i;
        pairs[index].j = counter.pairs[index].j;
        pairs[index].count = counter.pairs[index].count;
    }
}

template <size_t maxPairs>
__device__ __host__ Pair PairsCounter<maxPairs>::getTop() const {
    return pairs[0];
}

template <size_t maxPairs>
__device__ Pair PairsCounter<maxPairs>::getRandom(curandState &state) const {
    return pairs[curand(&state) % size];
}

template <size_t maxPairs>
__device__ Pair PairsCounter<maxPairs>::getGreedyRandom(curandState &state) const {
    return pairs[curand(&state) % topSize];
}

template <size_t maxPairs>
__device__ Pair PairsCounter<maxPairs>::getTopRandom(curandState &state) const {
    if (curand_uniform(&state) < 0.8)
        return pairs[curand(&state) % topSize];

    return pairs[curand(&state) % size];
}

template <size_t maxPairs>
__device__ __host__ int PairsCounter<maxPairs>::count() const {
    return size;
}

template <size_t maxPairs>
__device__ __host__ PairsCounter<maxPairs>::operator bool() const {
    return size > 0 && pairs[0].count > 1;
}
