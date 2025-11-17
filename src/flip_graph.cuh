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

#include "config.cuh"
#include "random.cuh"
#include "utils.cuh"
#include "scheme_integer.cuh"
#include "scheme_z2.cuh"

struct FlipGraphProbabilities {
    double extend;
    double project;

    double expand;
    double sandwiching;
    double basis;
    double reduce;
};

class FlipGraph {
    int n1;
    int n2;
    int n3;

    int schemesCount;
    int maxIterations;
    std::string path;

    FlipGraphProbabilities probabilities;
    int seed;

    int blockSize;
    int numBlocks;

    Scheme *schemes;
    Scheme *schemesBest;
    int *bestRanks;
    int *flips;
    curandState *states;

    std::unordered_map<std::string, int> n2bestRank;
    std::unordered_map<std::string, int> n2knownRanks;
public:
    FlipGraph(int n1, int n2, int n3, int schemesCount, int blockSize, int maxIterations, const std::string &path, const FlipGraphProbabilities &probabilities, int seed);

    bool initializeFromFile(std::istream &f);
    void initializeNaive();
    void run();

    ~FlipGraph();
private:
    void initialize();
    void optimize();
    void projectExtend();
    void updateRanks(int iteration);
    void report(std::chrono::high_resolution_clock::time_point startTime, int iteration, const std::vector<double> &elapsedTimes, int count = 3);

    std::string prettyFlips(int flips) const;
    std::string getSavePath(const Scheme &scheme, int iteration, int runId) const;
    std::unordered_map<std::string, std::vector<int>> getSortedIndices(int count) const;
};

__global__ void initializeNaiveKernel(Scheme *schemes, int schemesCount, int n1, int n2, int n3);
__global__ void initializeCopyKernel(Scheme *schemes, int schemesCount, int count);
__global__ void initializeSchemesKernel(Scheme *schemes, Scheme *schemesBest, int *bestRanks, int *flips, curandState *states, int n1, int n2, int n3, int schemesCount, int seed);
__global__ void randomWalkKernel(Scheme *schemes, Scheme *schemesBest, int *bestRanks, int *flips, curandState *states, int schemesCount, int maxIterations, double reduceProbability, double expandProbability, double sandwichingProbability, double basisProbability);
__global__ void projectExtendKernel(Scheme *schemes, int schemesCount, curandState *states, double extendProbability, double projectProbability);