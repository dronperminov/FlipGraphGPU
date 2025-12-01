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
    int maxCount;
    int hashTable[maxPairs * 2];

    __device__ __host__ void canonizePair(int &i, int &j) const;
    __device__ __host__ unsigned int getHash(int i, int j) const;
    __device__ __host__ int findOrCreateSlot(int i, int j);
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
    __device__ Pair getWeightedRandom(curandState &state) const;

    __device__ __host__ int count() const;
    __device__ __host__ operator bool() const;
};

template <size_t maxPairs>
__device__ __host__ PairsCounter<maxPairs>::PairsCounter() {
    size = 0;
    topSize = 0;
    maxCount = 0;

    for (int i = 0; i < maxPairs * 2; i++)
        hashTable[i] = -1;
}

template <size_t maxPairs>
__device__ __host__ void PairsCounter<maxPairs>::insert(int i, int j) {
    int index = findOrCreateSlot(i, j);
    if (index == -1)
        return;

    pairs[index].count++;

    if (pairs[index].count > maxCount)
        maxCount = pairs[index].count;
}

template <size_t maxPairs>
__device__ __host__ void PairsCounter<maxPairs>::sort() {
    if (maxCount < 2) {
        topSize = size;
        return;
    }

    topSize = 0;

    for (int i = 0; i < size; i++) {
        if (pairs[i].count != maxCount)
            continue;

        if (i != topSize) {
            Pair tmp = pairs[i];
            pairs[i] = pairs[topSize];
            pairs[topSize] = tmp;
        }

        topSize++;
    }

    int j = topSize;

    for (int i = topSize; i < size; i++)
        if (pairs[i].count > 1)
            pairs[j++] = pairs[i];

    size = j;
}

template <size_t maxPairs>
__device__ __host__ void PairsCounter<maxPairs>::clear() {
    size = 0;
    topSize = 0;
    maxCount = 0;

    for (int i = 0; i < maxPairs * 2; i++)
        hashTable[i] = -1;
}

template <size_t maxPairs>
__device__ __host__ void PairsCounter<maxPairs>::copyFrom(const PairsCounter<maxPairs> &counter) {
    size = counter.size;
    topSize = counter.topSize;
    maxCount = counter.maxCount;

    for (int index = 0; index < size; index++) {
        pairs[index].i = counter.pairs[index].i;
        pairs[index].j = counter.pairs[index].j;
        pairs[index].count = counter.pairs[index].count;
    }

    for (int i = 0; i < maxPairs * 2; i++)
        hashTable[i] = counter.hashTable[i];
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
__device__ Pair PairsCounter<maxPairs>::getWeightedRandom(curandState &state) const {
    float total = 0;

    for (int i = 0; i < size; i++)
        total += pairs[i].count;

    float p = curand_uniform(&state) * total;
    float sum = 0;

    for (int i = 0; i < size; i++) {
        sum += pairs[i].count;

        if (p <= sum)
            return pairs[i];
    }

    return pairs[size - 1];
}

template <size_t maxPairs>
__device__ __host__ int PairsCounter<maxPairs>::count() const {
    return size;
}

template <size_t maxPairs>
__device__ __host__ PairsCounter<maxPairs>::operator bool() const {
    return size > 0 && maxCount > 1;
}

template <size_t maxPairs>
__device__ __host__ void PairsCounter<maxPairs>::canonizePair(int &i, int &j) const {
    if (abs(i) > abs(j)) {
        int tmp = i;
        i = j;
        j = tmp;
    }

    if (i < 0) {
        i = -i;
        j = -j;
    }
}

template <size_t maxPairs>
__device__ __host__ unsigned int PairsCounter<maxPairs>::getHash(int i, int j) const {
    unsigned int hash = (static_cast<unsigned int>(i) * 2654435761u) ^ (static_cast<unsigned int>(j) * 2246822519u);
    return hash % (maxPairs * 2);
}

template <size_t maxPairs>
__device__ __host__ int PairsCounter<maxPairs>::findOrCreateSlot(int i, int j) {
    canonizePair(i, j);
    unsigned int hash = getHash(i, j);

    for (int attempts = 0; attempts < maxPairs * 2; attempts++) {
        int index = hashTable[hash];

        if (index == -1) {
            if (size >= maxPairs)
                return -1;

            pairs[size].i = i;
            pairs[size].j = j;
            pairs[size].count = 0;
            hashTable[hash] = size;
            return size++;
        }

        if (pairs[index].i == i && pairs[index].j == j)
            return index;

        hash = (hash + 1) % (maxPairs * 2);
    }

    return -1;
}
