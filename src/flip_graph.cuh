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
#include <unordered_map>

#include "scheme.cuh"
#include "random.cuh"

class FlipGraph {
    int n1;
    int n2;
    int n3;

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

    std::unordered_map<std::string, int> n2bestRank;
public:
    FlipGraph(int n1, int n2, int n3, int schemesCount, int blockSize, int maxIterations, const std::string &path, int seed);

    void run();

    ~FlipGraph();
private:
    void initialize();
    void optimize();
    void projectExpand(int iteration);
    void report(std::chrono::high_resolution_clock::time_point startTime, int iteration, const std::vector<double> &elapsedTimes, int count = 3);

    std::string prettyFlips(int flips) const;
    std::string prettyTime(double elapsed) const;
    std::string getSavePath(const Scheme &scheme, int iteration, int runId) const;
    std::string getKey(int n1, int n2, int n3) const;
    std::string getKey(const Scheme &scheme) const;
    std::unordered_map<std::string, std::vector<int>> getSortedIndices(int count) const;
};

__global__ void initializeSchemesKernel(Scheme *schemes, Scheme *schemesBest, int *bestRanks, int *flips, curandState *states, int n1, int n2, int n3, int schemesCount, int seed);
__global__ void randomWalkKernel(Scheme *schemes, Scheme *schemesBest, int *bestRanks, int *flips, curandState *states, int schemesCount, int maxIterations);
__global__ void projectExpandKernel(Scheme *schemes, int schemesCount, curandState *states);