#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "random.cuh"

const int MAX_RANK = 128;
const int MAX_MATRIX_SIZE = 8;

typedef uint32_t T;

struct Scheme {
    int n;
    int m;
    int nn;
    T uvw[3][MAX_RANK];
};

struct FlipCandidate {
    int first;
    int second;
    int index1;
    int index2;
};

struct ReduceCandidate {
    int i;
    int index1;
    int index2;
};

struct ReduceGaussCandidate {
    int i;
    int combination[MAX_RANK];
    int size;
};

__device__ __host__ bool validateEquation(const Scheme &scheme, int i, int j, int k);
__device__ __host__ bool validateScheme(const Scheme &scheme);

__device__ void initializeNaive(Scheme &scheme, int n);
__device__ void copyScheme(const Scheme &scheme, Scheme &target);

__device__ void removeZeroes(Scheme &scheme);
__device__ void removeAt(Scheme &scheme, int startIndex);
__device__ void addTriplet(Scheme &scheme, int i, int j, int k, const T u, const T v, const T w);

/*************************************************** helpers ***************************************************/
__device__ FlipCandidate getFlipCandidate(const Scheme &scheme, curandState &state);
__device__ ReduceCandidate getReduceCandidate(const Scheme &scheme, curandState &state);
__device__ ReduceGaussCandidate getReduceGaussCandidate(const Scheme &scheme, curandState &state);
__device__ int findXorCombination(const Scheme &scheme, int uvwIndex, int *indices, int size, int *combination);
__device__ void shellSort(int *indices, const T *values, int n);
__device__ bool inverseMatrixZ2(int n, int *matrix, int *inverse);
__device__ void invertibleMatrixZ2(int n, int *matrix, int *inverse, curandState &state);
__device__ T matmul(const T matrix, int *left, int *right, int n);

/************************************************** operators **************************************************/
__device__ void flip(Scheme &scheme, int first, int second, int index1, int index2);
__device__ void plus(Scheme &scheme, int i, int j, int k, int index1, int index2, int variant);
__device__ void split(Scheme &scheme, int i, int j, int k, int index, const T a1);
__device__ void reduceGauss(Scheme &scheme, int i, int *combination, int combinationSize);
__device__ void reduce(Scheme &scheme, int i, int index1, int index2);

/********************************************** random operators ***********************************************/
__device__ bool tryPlus(Scheme &scheme, curandState &state);
__device__ bool tryFlip(Scheme &scheme, curandState &state);
__device__ bool trySplit(Scheme &scheme, curandState &state);
__device__ bool trySplitExisted(Scheme &scheme, curandState &state);
__device__ bool tryReduceGauss(Scheme &scheme, curandState &state);
__device__ bool tryReduce(Scheme &scheme, curandState &state);
__device__ void expand(Scheme &scheme, int count, curandState &state);
__device__ void sandwiching(Scheme &scheme, curandState &state);

/**************************************************** save *****************************************************/
void saveMatrix(std::ofstream &f, std::string name, int n, int m, const T *matrix);
void saveScheme(const Scheme& scheme, const std::string &path);
