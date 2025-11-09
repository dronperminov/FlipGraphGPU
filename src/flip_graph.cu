#include "flip_graph.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


FlipGraph::FlipGraph(int n1, int n2, int n3, int schemesCount, int blockSize, int maxIterations, const std::string &path, const FlipGraphProbabilities &probabilities, int seed) {
    this->n1 = n1;
    this->n2 = n2;
    this->n3 = n3;

    this->schemesCount = schemesCount;
    this->maxIterations = maxIterations;
    this->path = path;
    this->probabilities = probabilities;
    this->seed = seed;

    this->blockSize = blockSize;
    this->numBlocks = (schemesCount + blockSize - 1) / blockSize;

    n2bestRank[getKey(n1, n2, n3)] = n1 * n2 * n3;

    CUDA_CHECK(cudaMallocManaged(&schemes, schemesCount * sizeof(Scheme)));
    CUDA_CHECK(cudaMallocManaged(&schemesBest, schemesCount * sizeof(Scheme)));
    CUDA_CHECK(cudaMallocManaged(&bestRanks, schemesCount * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&flips, schemesCount * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&states, schemesCount * sizeof(curandState)));
}

void FlipGraph::initialize() {
    initializeSchemesKernel<<<numBlocks, blockSize>>>(schemes, schemesBest, bestRanks, flips, states, n1, n2, n3, schemesCount, seed);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void FlipGraph::optimize() {
    randomWalkKernel<<<numBlocks, blockSize>>>(schemes, schemesBest, bestRanks, flips, states, schemesCount, maxIterations, probabilities.reduce, probabilities.expand, probabilities.sandwiching);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void FlipGraph::projectExtend() {
    projectExtendKernel<<<numBlocks, blockSize>>>(schemes, schemesCount, states, probabilities.extend, probabilities.project);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void FlipGraph::updateRanks(int iteration) {
    std::vector<std::string> keys(schemesCount);
    std::unordered_map<std::string, int> n2bestIndex;

    for (int i = 0; i < schemesCount; i++) {
        keys[i] = getKey(schemes[i]);

        auto result = n2bestRank.find(keys[i]);
        if (result == n2bestRank.end() || schemes[i].m < result->second) {
            n2bestRank[keys[i]] = schemes[i].m;
            n2bestIndex[keys[i]] = i;
        }
    }

    for (int i = 0; i < schemesCount; i++)
        bestRanks[i] = n2bestRank[keys[i]];

    for (auto pair : n2bestIndex) {
        std::string savePath = getSavePath(schemes[pair.second], iteration, pair.second);
        saveScheme(schemes[pair.second], savePath);
        std::cout << "Best rank of " << pair.first << " was improved to " << bestRanks[pair.second] << "! Scheme saved to \"" << savePath << "\"" << std::endl;
    }

    if (n2bestIndex.size())
        std::cout << std::endl;
}

void FlipGraph::run() {
    initialize();
    updateRanks(0);

    auto startTime = std::chrono::high_resolution_clock::now();
    std::vector<double> elapsedTimes;

    for (int iteration = 0; 1; iteration++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        optimize();
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        elapsedTimes.push_back(duration.count() / 1000.0);

        report(startTime, iteration + 1, elapsedTimes);

        if (probabilities.extend > 0 || probabilities.project > 0) {
            projectExtend();
            updateRanks(iteration);
        }
    }
}

void FlipGraph::report(std::chrono::high_resolution_clock::time_point startTime, int iteration, const std::vector<double> &elapsedTimes, int count) {
    double lastTime = elapsedTimes[elapsedTimes.size() - 1];
    double minTime = *std::min_element(elapsedTimes.begin(), elapsedTimes.end());
    double maxTime = *std::max_element(elapsedTimes.begin(), elapsedTimes.end());
    double meanTime = std::accumulate(elapsedTimes.begin(), elapsedTimes.end(), 0.0) / elapsedTimes.size();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

    std::unordered_map<std::string, std::vector<int>> n2indices = getSortedIndices(std::min(count, schemesCount));
    std::vector<std::string> keys;
    keys.reserve(n2indices.size());

    for (auto const &pair : n2indices) {
        const std::string &n = pair.first;
        int index = pair.second[0];
        keys.push_back(n);

        if (bestRanks[index] < n2bestRank[n]) {
            std::string savePath = getSavePath(schemesBest[index], iteration, index);
            saveScheme(schemesBest[index], savePath);

            std::cout << "Best rank of " << n << " was improved from " << n2bestRank[n] << " to " << bestRanks[index] << "! Scheme saved to \"" << savePath << "\"" << std::endl;
            n2bestRank[n] = bestRanks[index];
        }
    }

    std::cout << "+-----------+-----------+--------+------+------+------+------+-------------+" << std::endl;
    std::cout << "|  elapsed  | iteration | run id | size | real | best | curr | flips count |" << std::endl;
    std::cout << "+-----------+-----------+--------+------+------+------+------+-------------+" << std::endl;

    std::sort(keys.begin(), keys.end());

    for (auto key : keys) {
        const std::vector<int> &indices = n2indices[key];

        for (int i = 0; i < count && i < indices.size(); i++) {
            Scheme &scheme = schemes[indices[i]];

            if (!validateScheme(scheme))
                throw std::runtime_error("Invalid scheme");

            std::cout << "| ";
            std::cout << std::setw(9) << prettyTime(elapsed) << " | ";
            std::cout << std::setw(9) << iteration << " | ";
            std::cout << std::setw(6) << (indices[i] + 1) << " | ";
            std::cout << std::setw(4) << key << " | ";
            std::cout << " " << scheme.n[0] << scheme.n[1] << scheme.n[2] << " | ";
            std::cout << std::setw(4) << bestRanks[indices[i]] << " | ";
            std::cout << std::setw(4) << scheme.m << " | ";
            std::cout << std::setw(11) << prettyFlips(flips[indices[i]]) << " |";

            if (i == 0)
                std::cout << " total: " << indices.size();

            std::cout << std::endl;
        }

        std::cout << "+-----------+-----------+--------+------+------+------+------+-------------+" << std::endl;
    }

    std::cout << "- iteration time (last / min / max / mean): " << prettyTime(lastTime) << " / " << prettyTime(minTime) << " / " << prettyTime(maxTime) << " / " << prettyTime(meanTime) << std::endl;
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
    ss << getKey(scheme);
    ss << "_m" << scheme.m;
    ss << "_iteration" << iteration;
    ss << "_run" << runId;
    ss << "_" << scheme.n[0] << scheme.n[1] << scheme.n[2];
    ss << "_scheme.json";

    return ss.str();
}

std::string FlipGraph::getKey(int n1, int n2, int n3) const {
    std::vector<int> n = {n1, n2, n3};
    std::sort(n.begin(), n.end());

    std::stringstream ss;
    ss << n[0] << n[1] << n[2];
    return ss.str();
}

std::string FlipGraph::getKey(const Scheme &scheme) const {
    return getKey(scheme.n[0], scheme.n[1], scheme.n[2]);
}

std::unordered_map<std::string, std::vector<int>> FlipGraph::getSortedIndices(int count) const {
    std::unordered_map<std::string, std::vector<int>> n2indices;

    for (int i = 0; i < schemesCount; i++)
        n2indices[getKey(schemes[i])].push_back(i);

    for (auto pair : n2indices) {
        if (pair.second.size() < count)
            count = pair.second.size();

        std::partial_sort(n2indices[pair.first].begin(), n2indices[pair.first].begin() + count, n2indices[pair.first].end(), [this](int index1, int index2) { return schemesBest[index1].m < schemesBest[index2].m; });
    }

    return n2indices;
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
__global__ void initializeSchemesKernel(Scheme *schemes, Scheme *schemesBest, int *bestRanks, int *flips, curandState *states, int n1, int n2, int n3, int schemesCount, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= schemesCount)
        return;

    initializeNaive(schemes[idx], n1, n2, n3);

    if (!validateScheme(schemes[idx]))
        printf("not valid initialized scheme %d\n", idx);

    copyScheme(schemes[idx], schemesBest[idx]);
    curand_init(seed, idx, 0, &states[idx]);

    bestRanks[idx] = n1 * n2 * n3;
    flips[idx] = 0;
}

__global__ void randomWalkKernel(Scheme *schemes, Scheme *schemesBest, int *bestRanks, int *flips, curandState *states, int schemesCount, int maxIterations, double reduceProbability, double expandProbability, double sandwichingProbability) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= schemesCount)
        return;

    Scheme& scheme = schemes[idx];
    curandState& state = states[idx];
    int flipsCount = flips[idx];
    int bestRank = bestRanks[idx];

    for (int iteration = 0; iteration < maxIterations; iteration++) {
        if (!tryFlip(scheme, state)) {
            expand(scheme, randint(1, 2, state), state);
            continue;

        flipsCount++;

        if (scheme.m < bestRank) {
            bestRank = scheme.m;
            copyScheme(scheme, schemesBest[idx]);
        }

        if (curand_uniform(&state) * maxIterations < reduceProbability) {
            tryReduce(scheme, state);
            tryReduceGauss(scheme, state);
        }

        if (curand_uniform(&state) * maxIterations < expandProbability)
            expand(scheme, randint(1, 2, state), state);

        if (curand_uniform(&state) * maxIterations < sandwichingProbability)
            sandwiching(scheme, state);
    }

    flips[idx] = flipsCount;
    bestRanks[idx] = bestRank;
}

__global__ void projectExtendKernel(Scheme *schemes, int schemesCount, curandState *states, double extendProbability, double projectProbability) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= schemesCount)
        return;

    Scheme &scheme = schemes[idx];
    curandState &state = states[idx];

    if (curand_uniform(&state) < extendProbability)
        tryExtend(scheme, state);

    if (curand_uniform(&state) < projectProbability)
        tryProject(scheme, state);
}
