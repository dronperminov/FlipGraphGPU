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
#include "../schemes/scheme_integer.cuh"

class SchemeAdditionsReducer {
    int n1, n2, n3;
    int m;
    int count;
    int schemesCount;
    int maxFlips;
    int seed;
    int blockSize;
    int numBlocks;
    std::string outputPath;

    int topCount;
    int reducedAdditions;
    int reducedFreshVars;
    int bestAdditions[3];
    int bestFreshVars[3];
    std::vector<int> indices[3];

    AdditionsReducer<MAX_U_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_U_REAL_VARIABLES, MAX_U_SUBEXPRESSIONS> *reducersU;
    AdditionsReducer<MAX_V_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_V_REAL_VARIABLES, MAX_V_SUBEXPRESSIONS> *reducersV;
    AdditionsReducer<MAX_W_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_W_REAL_VARIABLES, MAX_W_SUBEXPRESSIONS> *reducersW;
    SchemeInteger *schemes;
    curandState *states;

    void initialize();
    void reduceIteration(int iteration);
    bool updateBest(int startAdditions);
    bool updateBestIndependent();
    bool updateBestTogether();
    void report(std::chrono::high_resolution_clock::time_point startTime, int iteration, const std::vector<double> &elapsedTimes);
    void save() const;

    std::string getSavePath() const;
    std::string getDimensions() const;
public:
    SchemeAdditionsReducer(int count, int schemesCount, int maxFlips, int seed, int blockSize, const std::string &outputPath, int topCount = 10);

    bool read(std::ifstream &f);
    void reduce(int maxNoImprovements, int startAdditions);

    ~SchemeAdditionsReducer();
};

__global__ void initializeKernel(
    AdditionsReducer<MAX_U_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_U_REAL_VARIABLES, MAX_U_SUBEXPRESSIONS> *reducersU,
    AdditionsReducer<MAX_V_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_V_REAL_VARIABLES, MAX_V_SUBEXPRESSIONS> *reducersV,
    AdditionsReducer<MAX_W_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_W_REAL_VARIABLES, MAX_W_SUBEXPRESSIONS> *reducersW,
    SchemeInteger *schemes,
    curandState *states,
    int count,
    int schemesCount,
    int seed
);

__global__ void flipSchemesKernel(SchemeInteger *schemes, curandState *states, int schemesCount, int maxFlips);

__device__ void copySchemeToReducers(
    AdditionsReducer<MAX_U_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_U_REAL_VARIABLES, MAX_U_SUBEXPRESSIONS> *reducersU,
    AdditionsReducer<MAX_V_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_V_REAL_VARIABLES, MAX_V_SUBEXPRESSIONS> *reducersV,
    AdditionsReducer<MAX_W_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_W_REAL_VARIABLES, MAX_W_SUBEXPRESSIONS> *reducersW,
    int idx,
    const SchemeInteger &scheme
);

__global__ void runReducersKernel(
    AdditionsReducer<MAX_U_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_U_REAL_VARIABLES, MAX_U_SUBEXPRESSIONS> *reducersU,
    AdditionsReducer<MAX_V_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_V_REAL_VARIABLES, MAX_V_SUBEXPRESSIONS> *reducersV,
    AdditionsReducer<MAX_W_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_W_REAL_VARIABLES, MAX_W_SUBEXPRESSIONS> *reducersW,
    SchemeInteger *schemes,
    curandState *states,
    int count,
    int schemesCount,
    bool independent
);
