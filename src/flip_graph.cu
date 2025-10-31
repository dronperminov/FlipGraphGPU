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

    std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();

    for (int iteration = 0; bestRank > targetRank; iteration++) {
        optimize();
        report(startTime, iteration + 1);
    }
}

void FlipGraph::report(std::chrono::high_resolution_clock::time_point startTime, int iteration, int count) {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime);

    std::vector<int> indices = getSortedIndices(std::min(count, schemesCount));

    if (bestRanks[indices[0]] < bestRank) {
        bestRank = bestRanks[indices[0]];

        Scheme best = schemesBest[indices[0]];
        std::string savePath = getSavePath(best, iteration, indices[0]);
        saveScheme(best, savePath);
        std::cout << "Best rank was improved to " << bestRank << "! Scheme saved to \"" << savePath << "\"" << std::endl;
    }

    std::cout << "+-----------+-----------+--------+------+------+-------------+" << std::endl;
    std::cout << "|  elapsed  | iteration | run id | best | curr | flips count |" << std::endl;
    std::cout << "+-----------+-----------+--------+------+------+-------------+" << std::endl;

    for (int i = 0; i < std::min(count, schemesCount); i++) {
        if (!validateScheme(schemes[indices[i]])) {
            std::cout << "invalid scheme" << std::endl;
            throw std::runtime_error("Invalid scheme");
        }

        std::cout << "| ";
        std::cout << std::setw(9) << prettyTime(duration.count() / 1000.0) << " | ";
        std::cout << std::setw(9) << iteration << " | ";
        std::cout << std::setw(6) << (indices[i] + 1) << " | ";
        std::cout << std::setw(4) << bestRanks[indices[i]] << " | ";
        std::cout << std::setw(4) << schemes[indices[i]].m << " | ";
        std::cout << std::setw(11) << prettyFlips(flips[indices[i]]) << " |";
        std::cout << std::endl;
    }

    std::cout << "+-----------+-----------+--------+------+------+-------------+" << std::endl;
    std::cout << "- best rank: " << bestRank << std::endl;
    std::cout << "- iteration time: " << prettyTime(duration.count() / 1000.0 / iteration) << std::endl;
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
        ss << std::setprecision(2) << std::fixed << elapsed << " sec";
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

    if (n == 5) {
        // const T u[] = {4225028, 524804, 68450, 10, 26240025, 832, 4194308, 16404, 1024, 864, 10485760, 524824, 262400, 64, 327680, 885610, 29388828, 524288, 27033600, 16777552, 65607, 161796, 656000, 67650, 131204, 68546, 26624, 27033600, 16793616, 65601, 17, 5248005, 16777216, 27033600, 16793936, 7, 75850, 159748, 656320, 4, 844825, 4198404, 30277632, 16384, 70722, 11371360, 16777240, 524289, 65602, 5406720, 541184, 168965, 11338560, 8, 525120, 65610, 26406940, 10250, 159876, 960, 2162688, 10, 169125, 524800, 800, 31488030, 10551360, 524312, 557920, 524292, 895850, 4222980, 2163552, 16777242, 524801, 131076, 994250, 24576, 557600, 2162752, 524314, 16394, 136324, 2163648, 16777220, 16401, 4199428, 136196, 11469760, 16392, 16777217, 262208, 4096};
        // const T v[] = {68450, 4225028, 524804, 832, 10, 26240025, 1024, 4194308, 16404, 524824, 864, 10485760, 327680, 262400, 64, 524288, 885610, 29388828, 65607, 27033600, 16777552, 67650, 161796, 656000, 26624, 131204, 68546, 65601, 27033600, 16793616, 16777216, 17, 5248005, 7, 27033600, 16793936, 656320, 75850, 159748, 4198404, 4, 844825, 70722, 30277632, 16384, 524289, 11371360, 16777240, 541184, 65602, 5406720, 8, 168965, 11338560, 26406940, 525120, 65610, 960, 10250, 159876, 169125, 2162688, 10, 31488030, 524800, 800, 557920, 10551360, 524312, 4222980, 524292, 895850, 524801, 2163552, 16777242, 24576, 131076, 994250, 524314, 557600, 2162752, 2163648, 16394, 136324, 4199428, 16777220, 16401, 16392, 136196, 11469760, 16777217, 262208, 4096};
        // const T w[] = {524804, 68450, 4225028, 26240025, 832, 10, 16404, 1024, 4194308, 10485760, 524824, 864, 64, 327680, 262400, 29388828, 524288, 885610, 16777552, 65607, 27033600, 656000, 67650, 161796, 68546, 26624, 131204, 16793616, 65601, 27033600, 5248005, 16777216, 17, 16793936, 7, 27033600, 159748, 656320, 75850, 844825, 4198404, 4, 16384, 70722, 30277632, 16777240, 524289, 11371360, 5406720, 541184, 65602, 11338560, 8, 168965, 65610, 26406940, 525120, 159876, 960, 10250, 10, 169125, 2162688, 800, 31488030, 524800, 524312, 557920, 10551360, 895850, 4222980, 524292, 16777242, 524801, 2163552, 994250, 24576, 131076, 2162752, 524314, 557600, 136324, 2163648, 16394, 16401, 4199428, 16777220, 11469760, 16392, 136196, 16777217, 262208, 4096};

        // schemes[idx].n = n;
        // schemes[idx].nn = n * n;
        // schemes[idx].m = 93;

        // for (int index = 0; index < schemes[idx].m; index++) {
        //     schemes[idx].uvw[0][index] = u[index];
        //     schemes[idx].uvw[1][index] = v[index];
        //     schemes[idx].uvw[2][index] = w[index];
        // }
    }
    else {
        initializeNaive(schemes[idx], n);
    }

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
