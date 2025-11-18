#pragma once

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>

#include "../config.cuh"
#include "../schemes/scheme_integer.cuh"
#include "../schemes/scheme_z2.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

std::string getKey(int n1, int n2, int n3);
std::string getKey(const Scheme &scheme);
std::string prettyTime(double elapsed);
