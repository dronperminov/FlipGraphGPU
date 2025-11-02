#include "flip_graph.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


FlipGraph::FlipGraph(int n, int initialRank, int targetRank, int schemesCount, int blockSize, int maxIterations, const std::string &path, int reduceStart, int seed) {
    this->n = n;

    this->initialRank = initialRank;
    this->targetRank = targetRank;

    this->schemesCount = schemesCount;
    this->maxIterations = maxIterations;
    this->path = path;
    this->reduceStart = reduceStart;

    this->seed = seed;
    this->bestRank = n*n*n;

    this->blockSize = blockSize;
    this->numBlocks = (schemesCount + blockSize - 1) / blockSize;

    CUDA_CHECK(cudaMallocManaged(&schemes, schemesCount * sizeof(Scheme)));
    CUDA_CHECK(cudaMallocManaged(&schemesBest, schemesCount * sizeof(Scheme)));
    CUDA_CHECK(cudaMallocManaged(&bestRanks, schemesCount * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&flips, schemesCount * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&states, schemesCount * sizeof(curandState)));
}

void FlipGraph::initialize() {
    initializeSchemesKernel<<<numBlocks, blockSize>>>(schemes, schemesBest, bestRanks, flips, states, n, initialRank, schemesCount, seed);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void FlipGraph::optimize() {
    randomWalkKernel<<<numBlocks, blockSize>>>(schemes, schemesBest, bestRanks, flips, states, schemesCount, maxIterations, reduceStart);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void FlipGraph::run() {
    initialize();

    auto startTime = std::chrono::high_resolution_clock::now();
    std::vector<double> elapsedTimes;

    for (int iteration = 0; bestRank > targetRank; iteration++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        optimize();
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        elapsedTimes.push_back(duration.count() / 1000.0);

        report(startTime, iteration + 1, elapsedTimes);
    }
}

void FlipGraph::report(std::chrono::high_resolution_clock::time_point startTime, int iteration, const std::vector<double> &elapsedTimes, int count) {

    std::vector<int> indices = getSortedIndices(std::min(count, schemesCount));

    if (bestRanks[indices[0]] < bestRank) {
        bestRank = bestRanks[indices[0]];

        Scheme best = schemesBest[indices[0]];
        std::string savePath = getSavePath(best, iteration, indices[0]);
        saveScheme(best, savePath);
        std::cout << "Best rank was improved to " << bestRank << "! Scheme saved to \"" << savePath << "\"" << std::endl;
    }

    double minTime = *std::min_element(elapsedTimes.begin(), elapsedTimes.end());
    double maxTime = *std::max_element(elapsedTimes.begin(), elapsedTimes.end());
    double meanTime = std::accumulate(elapsedTimes.begin(), elapsedTimes.end(), 0.0) / elapsedTimes.size();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

    std::cout << "+-----------+-----------+--------+------+------+-------------+" << std::endl;
    std::cout << "|  elapsed  | iteration | run id | best | curr | flips count |" << std::endl;
    std::cout << "+-----------+-----------+--------+------+------+-------------+" << std::endl;

    for (int i = 0; i < std::min(count, schemesCount); i++) {
        if (!validateScheme(schemes[indices[i]]))
            throw std::runtime_error("Invalid scheme");

        std::cout << "| ";
        std::cout << std::setw(9) << prettyTime(elapsed) << " | ";
        std::cout << std::setw(9) << iteration << " | ";
        std::cout << std::setw(6) << (indices[i] + 1) << " | ";
        std::cout << std::setw(4) << bestRanks[indices[i]] << " | ";
        std::cout << std::setw(4) << schemes[indices[i]].m << " | ";
        std::cout << std::setw(11) << prettyFlips(flips[indices[i]]) << " |";
        std::cout << std::endl;
    }

    std::cout << "+-----------+-----------+--------+------+------+-------------+" << std::endl;
    std::cout << "- best rank: " << bestRank << std::endl;
    std::cout << "- iteration time (min / max / mean): " << prettyTime(minTime) << " / " << prettyTime(maxTime) << " / " << prettyTime(meanTime) << std::endl;
    std::cout << std::endl;
}

std::string FlipGraph::prettyFlips(int flips) const {
    std::stringstream ss;

    if (flips < 1000)
        ss << flips;
    else if (flips < 1000000)
        ss << std::setprecision(2) << std::fixed <<(flips / 1000.0) << "K";
    else
        ss << std::setprecision(2) << std::fixed << (flips / 1000000.0) << "M";

    return ss.str();
}

std::string FlipGraph::prettyTime(double elapsed) const {
    std::stringstream ss;

    if (elapsed < 60) {
        ss << std::setprecision(3) << std::fixed << elapsed;
    }
    else {
        int seconds = int(elapsed + 0.5);
        int hours = seconds / 3600;
        int minutes = (seconds % 3600) / 60;

        ss << std::setw(2) << std::setfill('0') << hours << ":";
        ss << std::setw(2) << std::setfill('0') << minutes << ":";
        ss << std::setw(2) << std::setfill('0') << (seconds % 60);
    }

    return ss.str();
}

std::string FlipGraph::getSavePath(const Scheme &scheme, int iteration, int runId) const {
    std::stringstream ss;

    ss << path << "/";
    ss << "n" << scheme.n;
    ss << "_m" << scheme.m;
    ss << "_iteration" << iteration;
    ss << "_run" << runId;
    ss << "_scheme.json";

    return ss.str();
}

std::vector<int> FlipGraph::getSortedIndices(int count) const {
    std::vector<int> indices(schemesCount);

    for (int i = 0; i < schemesCount; i++)
        indices[i] = i;

    std::partial_sort(indices.begin(), indices.begin() + count, indices.end(), [this](int index1, int index2) { return schemesBest[index1].m < schemesBest[index2].m; });
    return indices;
}

FlipGraph::~FlipGraph() {
    if (schemes) {
        cudaFree(schemes);
        schemes = nullptr;
    }

    if (states) {
        cudaFree(states);
        states = nullptr;
    }
    
    if (bestRanks) {
        cudaFree(bestRanks);
        bestRanks = nullptr;
    }

    if (flips) {
        cudaFree(flips);
        flips = nullptr;
    }
}

/************************************************************************************ kernels ************************************************************************************/
__global__ void initializeSchemesKernel(Scheme *schemes, Scheme *schemesBest, int *bestRanks, int *flips, curandState *states, int n, int m, int schemesCount, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= schemesCount)
        return;

    initializeNaive(schemes[idx], n);
    copyScheme(schemes[idx], schemesBest[idx]);
    curand_init(seed, idx, 0, &states[idx]);

    bestRanks[idx] = m;
    flips[idx] = 0;
}

__global__ void randomWalkKernel(Scheme *schemes, Scheme *schemesBest, int *bestRanks, int *flips, curandState *states, int schemesCount, int maxIterations, int reduceStart) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= schemesCount)
        return;

    Scheme& scheme = schemes[idx];
    curandState& state = states[idx];
    int flipsCount = flips[idx];
    int bestRank = bestRanks[idx];

    for (int iteration = 0; iteration < maxIterations; iteration++) {
        if (!tryFlip(scheme, state)) {
            expand(scheme, randint(1, 3, state), state);
            continue;
        }

        flipsCount++;

        if (scheme.m < bestRank) {
            bestRank = scheme.m;
            copyScheme(scheme, schemesBest[idx]);
            continue;
        }

        if (flipsCount >= reduceStart && curand_uniform(&state) * maxIterations < 1.0) {
            tryReduce(scheme, state);
            tryReduceGauss(scheme, state);
        }

        if (curand_uniform(&state) * maxIterations < 1.0)
            expand(scheme, randint(1, 3, state), state);

        if (curand_uniform(&state) * maxIterations < 0.1)
            sandwiching(scheme, state);
    }

    flips[idx] = flipsCount;
    bestRanks[idx] = bestRank;
}
