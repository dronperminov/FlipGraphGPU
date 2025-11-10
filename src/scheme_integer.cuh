#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "random.cuh"
#include "addition.cuh"
#include "flip_set.cuh"

const int MAX_RANK = 150;

struct Scheme {
    int n[3];
    int nn[3];
    int m;
    Addition uvw[3][MAX_RANK];
    FlipSet flips[3];

    __device__ __host__ bool validate() const;
    __device__ __host__ void initializeNaive(int n1, int n2, int n3);
    __device__ __host__ void initializeFrom(int n1, int n2, int n3, int m, int scheme[3][MAX_RANK][MAX_MATRIX_ELEMENTS]);
    __device__ __host__ void copyTo(Scheme &target);

    __device__ bool tryFlip(curandState &state);
    __device__ bool tryPlus(curandState &state);
    __device__ bool trySplitExisted(curandState &state);
    __device__ bool tryExpand(int count, curandState &state);
    __device__ bool tryReduce(curandState &state);
    __device__ bool tryProject(curandState &state, int n1 = 2, int n2 = 2, int n3 = 2);
    __device__ bool tryExtend(curandState &state, int n1 = 7, int n2 = 7, int n3 = 7);
    __device__ void sandwiching(curandState &state);

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

    __device__ __host__ void flip(int i, int j, int k, int index1, int index2, bool checkReduce = true);
    __device__ __host__ void plus(int i, int j, int k, int index1, int index2, int variant);
    __device__ __host__ void split(int i, int j, int k, int index, const Addition& addition);
    __device__ __host__ void reduce(int i, int index1, int index2);
    __device__ __host__ void project(int p, int q);
    __device__ __host__ void extend(int p);

    void saveMatrix(std::ofstream &f, std::string name, int m, const Addition *additions) const;
    void showTensor(const Addition &addition, int n1, int n2, std::string name, bool transpose) const;
};
