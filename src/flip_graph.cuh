#pragma once

#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "scheme.cuh"
#include "random.cuh"

class FlipGraph {
    int n1;
    int n2;
    int n3;

    int initialRank;
    int targetRank;
    int schemesCount;
    int maxIterations;
    std::string path;
    int reduceStart;
    int seed;

    int blockSize;
    int numBlocks;

    Scheme *schemes;
    Scheme *schemesBest;
    int *bestRanks;
    int *flips;
    curandState *states;
    int bestRank;
public:
    FlipGraph(int n1, int n2, int n3, int initialRank, int targetRank, int schemesCount, int blockSize, int maxIterations, const std::string &path, int reduceStart, int seed);

    void run();

    ~FlipGraph();
private:
    void initialize();
    void optimize();
    void report(std::chrono::high_resolution_clock::time_point startTime, int iteration, const std::vector<double> &elapsedTimes, int count = 5);

    std::string prettyFlips(int flips) const;
    std::string prettyTime(double elapsed) const;
    std::string getSavePath(const Scheme &scheme, int iteration, int runId) const;
    std::vector<int> getSortedIndices(int count) const;
};

__global__ void initializeSchemesKernel(Scheme *schemes, Scheme *schemesBest, int *bestRanks, int *flips, curandState *states, int n1, int n2, int n3, int m, int schemesCount, int seed);
__global__ void randomWalkKernel(Scheme *schemes, Scheme *schemesBest, int *bestRanks, int *flips, curandState *states, int schemesCount, int maxIterations, int reduceStart);
