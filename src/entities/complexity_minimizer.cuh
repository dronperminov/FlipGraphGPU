#pragma once

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "../config.cuh"
#include "../utils/random.cuh"
#include "../utils/utils.cuh"
#include "../schemes/scheme_integer.cuh"
#include "../schemes/scheme_z2.cuh"

class ComplexityMinimizer {
    int initialCount;
    int schemesCount;
    std::string path;

    int maxIterations;
    int seed;
    int topCount;
    int blockSize;
    int numBlocks;

    Scheme *schemes;
    Scheme *schemesBest;
    int *bestComplexities;
    int bestComplexity;
    curandState *states;

    std::vector<int> indices;
public:
    ComplexityMinimizer(int schemesCount, int blockSize, int maxIterations, const std::string &path, int seed, int topCount = 10);

    bool read(std::istream &f);
    void minimize(int targetComplexity, int maxNoImprovements);

    ~ComplexityMinimizer();
private:
    void initialize();
    void minimizeIteration();
    bool updateBest(int iteration);
    void report(std::chrono::high_resolution_clock::time_point startTime, int iteration, const std::vector<double> &elapsedTimes);

    std::string getSavePath(const Scheme &scheme, int iteration, int runId) const;
};

__global__ void initializeKernel(Scheme *schemes, Scheme *schemesBest, int *bestComplexities, curandState *states, int schemesCount, int initialCount, int complexity, int seed);
__global__ void minimizeKernel(Scheme *schemes, Scheme *schemesBest, int *bestComplexities, curandState *states, int schemesCount, int iterations);