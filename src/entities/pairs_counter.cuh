#pragma once

#include <iostream>
#include <curand_kernel.h>

#include "../config.cuh"

struct Pair {
    int i;
    int j;
    int count;

    __device__ __host__ bool intersects(const Pair &pair) const {
        return i == pair.i || i == abs(pair.j) || abs(j) == pair.i || abs(j) == abs(pair.j);
    }
};

template <size_t maxPairs>
class PairsCounter {
    static constexpr int capacity = maxPairs * 2 + 1;

    Pair pairs[maxPairs];
    int size;
    int topSize;
    int maxCount;
    int hashTable[capacity];

    __device__ __host__ void canonizePair(int &i, int &j) const;
    __device__ __host__ unsigned int getHash(int i, int j) const;
    __device__ __host__ int findOrCreateSlot(int i, int j);
public:
    __device__ __host__ PairsCounter();

    __device__ __host__ void insert(int i, int j);
    __device__ __host__ void sort();
    __device__ __host__ void clear();
    __device__ __host__ void copyFrom(const PairsCounter<maxPairs> &counter);

    __device__ __host__ Pair getGreedy() const;
    __device__ Pair getGreedyAlternative(curandState &state) const;
    __device__ Pair getGreedyRandom(curandState &state, float scale) const;
    __device__ Pair getGreedyIntersections(curandState &state, float scale) const;
    __device__ Pair getWeightedRandom(curandState &state) const;
    __device__ Pair getRandom(curandState &state) const;

    __device__ __host__ int count() const;
    __device__ __host__ operator bool() const;
};

template <size_t maxPairs>
__device__ __host__ PairsCounter<maxPairs>::PairsCounter() {
    size = 0;
    topSize = 0;
    maxCount = 0;

    for (int i = 0; i < capacity; i++)
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

    for (int i = 0; i < capacity; i++)
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

    for (int i = 0; i < capacity; i++)
        hashTable[i] = counter.hashTable[i];
}

template <size_t maxPairs>
__device__ __host__ Pair PairsCounter<maxPairs>::getGreedy() const {
    return pairs[0];
}

template <size_t maxPairs>
__device__ Pair PairsCounter<maxPairs>::getGreedyAlternative(curandState &state) const {
    return pairs[curand(&state) % topSize];
}

template <size_t maxPairs>
__device__ Pair PairsCounter<maxPairs>::getGreedyRandom(curandState &state, float scale) const {
    if (curand_uniform(&state) < scale)
        return pairs[curand(&state) % size];

    return pairs[curand(&state) % topSize];
}

template <size_t maxPairs>
__device__ Pair PairsCounter<maxPairs>::getGreedyIntersections(curandState &state, float scale) const {
    int imax = 0;
    float maxScore = 0;

    for (int i = 0; i < size; i++) {
        float intersections = 0;

        for (int j = 0; j < size; j++) {
            if (i == j)
                continue;

            if (!pairs[i].intersects(pairs[j])) {
                intersections += pairs[j].count - 1;
            }
            else {
                intersections += curand(&state) % 2 ? 0 : 0.69 * (pairs[j].count);
            }
        }

        float score = pairs[i].count - 1 + scale * intersections;

        if (score > maxScore) {
            maxScore = score;
            imax = i;
        }
    }

    return pairs[imax];
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
__device__ Pair PairsCounter<maxPairs>::getRandom(curandState &state) const {
    return pairs[curand(&state) % size];
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
    return hash % capacity;
}

template <size_t maxPairs>
__device__ __host__ int PairsCounter<maxPairs>::findOrCreateSlot(int i, int j) {
    canonizePair(i, j);
    unsigned int hash = getHash(i, j);

    for (int attempts = 0; attempts < capacity; attempts++) {
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

        hash = (hash + 1) % capacity;
    }

    return -1;
}
