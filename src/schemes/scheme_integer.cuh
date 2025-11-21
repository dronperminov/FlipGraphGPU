#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "../config.cuh"
#include "../utils/random.cuh"
#include "../entities/addition.cuh"
#include "../entities/flip_set.cuh"

struct SchemeInteger {
    int n[3];
    int nn[3];
    int m;
    Addition uvw[3][MAX_RANK];
    FlipSet flips[3];

    __device__ __host__ bool validate() const;
    __device__ __host__ void initializeNaive(int n1, int n2, int n3);
    __device__ __host__ void copyTo(SchemeInteger &target, bool withFlips = true) const;
    __host__ bool read(std::istream &is, bool checkValidity = true);
    __host__ bool read(std::istream &is, int n1, int n2, int n3, int m, bool checkValidity = true);

    __device__ __host__ int getComplexity() const;
    __device__ __host__ bool isValidProject(int i, int minN) const;
    __device__ __host__ bool isValidExtension(int i, int maxN) const;
    __device__ __host__ bool isValidProduct(int i, int maxN) const;

    __device__ bool tryFlip(curandState &state, bool checkReduce = true);
    __device__ bool tryPlus(curandState &state);
    __device__ bool trySplit(curandState &state);
    __device__ bool trySplitExisted(curandState &state);
    __device__ bool tryExpand(int count, curandState &state);
    __device__ __host__ bool tryReduce();
    __device__ bool tryProject(curandState &state, int minN = MIN_PROJECT_N);
    __device__ bool tryExtend(curandState &state, int maxN = MAX_EXTENSION_N);
    __device__ bool tryProduct(curandState &state, int maxN = MAX_EXTENSION_N);
    __device__ bool tryMerge(const SchemeInteger &scheme, curandState &state);
    __device__ bool tryProduct(const SchemeInteger &scheme);
    __device__ void sandwiching(curandState &state);
    __device__ void swapBasis(curandState &state);
    __device__ void swapSize(curandState &state);

    void save(const std::string &path);
    void show() const;
private:
    __device__ __host__ bool validateEquation(int i, int j, int k) const;

    __device__ __host__ void initFlips();
    __device__ __host__ void removeZeroes();
    __device__ __host__ void removeAt(int startIndex);
    __device__ __host__ void addTriplet(int i, int j, int k, const Addition &u, const Addition &v, const Addition &w);
    __device__ __host__ void excludeColumn(int matrix, int column);
    __device__ __host__ void excludeRow(int matrix, int row);
    __device__ __host__ void addColumn(int matrix);
    __device__ __host__ void addRow(int matrix);
    __device__ __host__ bool fixSigns();

    __device__ __host__ void flip(int i, int j, int k, int index1, int index2, bool checkReduce);
    __device__ __host__ void plus(int i, int j, int k, int index1, int index2, int variant);
    __device__ __host__ void split(int i, int j, int k, int index, const Addition& addition);
    __device__ __host__ void reduceAdd(int i, int index1, int index2);
    __device__ __host__ void reduceSub(int i, int index1, int index2);
    __device__ __host__ void project(int p, int q);
    __device__ __host__ void extend(int p);
    __device__ __host__ void product(int p);
    __device__ __host__ void merge(const SchemeInteger &scheme, int p);
    __device__ __host__ void product(const SchemeInteger &scheme);
    __device__ __host__ void swapBasisRows(int i1, int i2);
    __device__ __host__ void swapBasisColumns(int j1, int j2);
    __device__ __host__ void swapSize(int p1, int p2);

    void saveMatrix(std::ofstream &f, std::string name, int m, const Addition *additions) const;
    void showTensor(const Addition &addition, int n1, int n2, std::string name, bool transpose) const;
};
