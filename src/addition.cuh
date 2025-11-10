#pragma once

#include <cstdint>
#include <iostream>

const int MAX_MATRIX_SIZE = 7;
const int MAX_MATRIX_ELEMENTS = MAX_MATRIX_SIZE * MAX_MATRIX_SIZE;
const int MAX_SIZE = MAX_MATRIX_ELEMENTS;

const int LOWER_BOUND = -1;
const int UPPER_BOUND = 1;

struct Addition {
    int n;
    int8_t values[MAX_MATRIX_ELEMENTS];

    __device__ __host__ Addition();
    __device__ __host__ Addition(int n);
    __device__ __host__ Addition(int n, int index);
    __device__ __host__ Addition(int n, int *values);

    __device__ __host__ void copyTo(Addition &target) const;

    __device__ __host__ bool operator==(const Addition &addition) const;
    __device__ __host__ bool operator!=(const Addition &addition) const;
    __device__ __host__ int8_t operator[](int index) const;

    __device__ __host__ Addition operator+(const Addition &addition) const;
    __device__ __host__ Addition operator-(const Addition &addition) const;

    __device__ __host__ Addition& operator+=(const Addition &addition);
    __device__ __host__ Addition& operator-=(const Addition &addition);

    __device__ __host__ bool limit(bool firstPositiveNonZero) const;
    __device__ __host__ bool limitSum(const Addition &addition, bool firstPositiveNonZero) const;
    __device__ __host__ bool limitSub(const Addition &addition) const;

    __device__ __host__ bool positiveFirstNonZero() const;
    __device__ __host__ bool positiveFirstNonZeroSub(const Addition &addition) const;

    __device__ __host__ operator bool() const;
};
