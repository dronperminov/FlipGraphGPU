#pragma once

#include <cstdint>
#include <iostream>
#include "../config.cuh"
#include "../utils/random.cuh"

const int ADDITION_LOWER_BOUND = -7;
const int ADDITION_UPPER_BOUND = 7;

struct AdditionArray {
    int n;
    int8_t values[MAX_MATRIX_ELEMENTS];
    bool valid;

    __device__ __host__ AdditionArray();
    __device__ __host__ AdditionArray(int n);
    __device__ __host__ AdditionArray(int n, int index);
    __device__ __host__ AdditionArray(int n, int *values);

    __device__ __host__ void copyTo(AdditionArray &target) const;
    __device__ __host__ void set(int index, int value);
    __device__ __host__ void inverse();
    __device__ void random(curandState &state);

    __device__ __host__ int nonZeroCount() const;

    __device__ __host__ int compare(const AdditionArray &addition) const;
    __device__ __host__ bool operator==(const AdditionArray &addition) const;
    __device__ __host__ bool operator!=(const AdditionArray &addition) const;
    __device__ __host__ int operator[](int index) const;

    __device__ __host__ AdditionArray operator+(const AdditionArray &addition) const;
    __device__ __host__ AdditionArray operator-(const AdditionArray &addition) const;
    __device__ __host__ AdditionArray operator-() const;

    __device__ __host__ AdditionArray& operator+=(const AdditionArray &addition);
    __device__ __host__ AdditionArray& operator-=(const AdditionArray &addition);

    __device__ __host__ bool limit(bool firstPositiveNonZero) const;
    __device__ __host__ bool limitSum(const AdditionArray &addition, bool firstPositiveNonZero) const;
    __device__ __host__ bool limitSub(const AdditionArray &addition, bool firstPositiveNonZero = false) const;

    __device__ __host__ bool positiveFirstNonZero() const;
    __device__ __host__ bool positiveFirstNonZeroSub(const AdditionArray &addition) const;

    __device__ __host__ operator bool() const;

    friend std::ostream& operator<<(std::ostream &os, const AdditionArray &addition);
};
