#pragma once

#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <vector>

#include "additions_reducer.cuh"
#include "../config.cuh"
#include "../utils/utils.cuh"

class SchemeAdditionsReducer {
    int n1, n2, n3;
    int m;
    int count;
    int seed;
    int blockSize;
    int numBlocks;
    std::string outputPath;

    int topCount;
    int naiveAdditions;
    int reducedAdditions;
    int bestAdditions[3];
    std::vector<int> indices[3];

    AdditionsReducer<MAX_UV_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_REAL_UV_VARIABLES> *reducersU;
    AdditionsReducer<MAX_UV_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_REAL_UV_VARIABLES> *reducersV;
    AdditionsReducer<MAX_W_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_REAL_W_VARIABLES> *reducersW;
    curandState *states;

    void initialize();
    void reduceIteration();
    bool updateBest();
    void report(std::chrono::high_resolution_clock::time_point startTime, int iteration, const std::vector<double> &elapsedTimes);
    void save() const;

    std::string getSavePath() const;
public:
    SchemeAdditionsReducer(int count, int seed, int blockSize, const std::string &outputPath, int topCount = 10);

    bool read(std::ifstream &f);
    void reduce(int iterations);

    ~SchemeAdditionsReducer();
};

__global__ void initializeRandomKernel(curandState *states, int count, int seed);

__global__ void runReducersKernel(
    AdditionsReducer<MAX_UV_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_REAL_UV_VARIABLES> *reducersU,
    AdditionsReducer<MAX_UV_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_REAL_UV_VARIABLES> *reducersV,
    AdditionsReducer<MAX_W_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_REAL_W_VARIABLES> *reducersW,
    curandState *states,
    int count
);
