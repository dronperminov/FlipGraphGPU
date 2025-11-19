#include "flip_graph.cuh"


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

    n2knownRanks = {
        {"2x2x2", 7}, {"2x2x3", 11}, {"2x2x4", 14}, {"2x2x5", 18}, {"2x2x6", 21}, {"2x2x7", 25}, {"2x2x8", 28}, {"2x2x9", 32}, {"2x2x10", 35},
        {"2x3x3", 15}, {"2x3x4", 20}, {"2x3x5", 25}, {"2x3x6", 30}, {"2x3x7", 35}, {"2x3x8", 40}, {"2x3x9", 45}, {"2x3x10", 50},
        {"2x4x4", 26}, {"2x4x5", 32}, {"2x4x6", 39}, {"2x4x7", 45}, {"2x4x8", 51}, {"2x4x9", 59}, {"2x4x10", 64},
        {"2x5x5", 40}, {"2x5x6", 47}, {"2x5x7", 55}, {"2x5x8", 63}, {"2x5x9", 72}, {"2x5x10", 79},
        {"2x6x6", 56}, {"2x6x7", 66}, {"2x6x8", 75}, {"2x6x9", 86}, {"2x6x10", 94},
        {"2x7x7", 76}, {"2x7x8", 88}, {"2x7x9", 99},
        {"2x8x8", 100},
        {"3x3x3", 23}, {"3x3x4", 29}, {"3x3x5", 36}, {"3x3x6", 40}, {"3x3x7", 49}, {"3x3x8", 55}, {"3x3x9", 63}, {"3x3x10", 69},
        {"3x4x4", 38}, {"3x4x5", 47}, {"3x4x6", 54}, {"3x4x7", 63}, {"3x4x8", 73}, {"3x4x9", 83}, {"3x4x10", 92},
        {"3x5x5", 58}, {"3x5x6", 68}, {"3x5x7", 79}, {"3x5x8", 90}, {"3x5x9", 104}, {"3x5x10", 115},
        {"3x6x6", 80}, {"3x6x7", 94}, {"3x6x8", 108}, {"3x6x9", 120}, {"3x6x10", 134},
        {"3x7x7", 111}, {"3x7x8", 126}, {"3x7x9", 142},
        {"3x8x8", 145},
        {"4x4x4", 48}, {"4x4x5", 61}, {"4x4x6", 73}, {"4x4x7", 85}, {"4x4x8", 96}, {"4x4x9", 104}, {"4x4x10", 120},
        {"4x5x5", 76}, {"4x5x6", 90}, {"4x5x7", 104}, {"4x5x8", 118}, {"4x5x9", 136}, {"4x5x10", 151},
        {"4x6x6", 105}, {"4x6x7", 123}, {"4x6x8", 140}, {"4x6x9", 159}, {"4x6x10", 175},
        {"4x7x7", 144}, {"4x7x8", 164}, {"4x7x9", 186},
        {"4x8x8", 182},
        {"5x5x5", 93}, {"5x5x6", 110}, {"5x5x7", 127}, {"5x5x8", 144}, {"5x5x9", 167}, {"5x5x10", 184},
        {"5x6x6", 130}, {"5x6x7", 150}, {"5x6x8", 170}, {"5x6x9", 197}, {"5x6x10", 218},
        {"5x7x7", 176}, {"5x7x8", 205}, {"5x7x9", 229},
        {"5x8x8", 230},
        {"6x6x6", 153}, {"6x6x7", 183}, {"6x6x8", 203}, {"6x6x9", 225}, {"6x6x10", 247},
        {"6x7x7", 215}, {"6x7x8", 239}, {"6x7x9", 270},
        {"6x8x8", 266},
        {"7x7x7", 249}, {"7x7x8", 277}, {"7x7x9", 315},
        {"7x8x8", 306},
        {"8x8x8", 336}
    };

#ifdef SCHEME_INTEGER
    n2knownRanks["2x4x5"] = 33;

    n2knownRanks["2x5x7"] = 57;
    n2knownRanks["2x5x8"] = 65;

    n2knownRanks["2x6x6"] = 57;
    n2knownRanks["2x6x7"] = 68;
    n2knownRanks["2x6x8"] = 77;
    n2knownRanks["2x6x9"] = 86;

    n2knownRanks["2x7x7"] = 77;
    n2knownRanks["2x7x8"] = 90;
    n2knownRanks["2x7x9"] = 102;

    n2knownRanks["3x3x6"] = 44;
    n2knownRanks["3x3x7"] = 51;
    n2knownRanks["3x3x8"] = 58;
    n2knownRanks["3x3x9"] = 65;

    n2knownRanks["3x4x6"] = 57;
    n2knownRanks["3x4x7"] = 66;
    n2knownRanks["3x4x8"] = 74;
    n2knownRanks["3x4x9"] = 85;

    n2knownRanks["3x5x6"] = 70;
    n2knownRanks["3x5x7"] = 83;
    n2knownRanks["3x5x8"] = 94;
    n2knownRanks["3x5x9"] = 105;

    n2knownRanks["3x6x6"] = 85;
    n2knownRanks["3x6x7"] = 100;
    n2knownRanks["3x6x8"] = 113;
    n2knownRanks["3x6x9"] = 127;

    n2knownRanks["3x7x7"] = 117;
    n2knownRanks["3x7x8"] = 132;
    n2knownRanks["3x7x9"] = 149;

    n2knownRanks["3x8x8"] = 148;

    n2knownRanks["4x4x4"] = 49;
    n2knownRanks["4x4x9"] = 110;

    n2knownRanks["4x5x9"] = 137;

    n2knownRanks["4x6x9"] = 162;

    n2knownRanks["4x7x7"] = 148;
    n2knownRanks["4x7x9"] = 189;

    n2knownRanks["5x6x8"] = 176;

    n2knownRanks["5x7x7"] = 184;
    n2knownRanks["5x7x8"] = 207;

    n2knownRanks["6x6x7"] = 185;
    n2knownRanks["6x6x9"] = 225;
    n2knownRanks["6x7x9"] = 270;

    n2knownRanks["7x7x7"] = 261;
    n2knownRanks["7x7x8"] = 292;
#else
    n2knownRanks["2x4x5"] = 33;
    n2knownRanks["2x5x7"] = 57; // ?
    n2knownRanks["2x5x8"] = 65; // ?
    n2knownRanks["2x6x8"] = 77; // ?
    n2knownRanks["2x7x7"] = 77; // ?
    n2knownRanks["2x7x9"] = 101; // ?

    n2knownRanks["3x3x6"] = 42; // ?
    n2knownRanks["3x3x8"] = 57; // ?
    n2knownRanks["3x3x9"] = 64; // ?

    n2knownRanks["3x4x7"] = 64; // ?
    n2knownRanks["3x4x8"] = 74; // ?

    n2knownRanks["3x6x6"] = 84; // ?
    n2knownRanks["3x6x7"] = 96; // ?
    n2knownRanks["3x6x9"] = 122; // ?

    n2knownRanks["3x7x7"] = 113; // ?
    n2knownRanks["3x7x8"] = 128; // ?
    n2knownRanks["3x7x9"] = 143; // ?

    n2knownRanks["4x4x4"] = 47;
    n2knownRanks["4x4x5"] = 60;
    n2knownRanks["4x4x8"] = 94;
    n2knownRanks["4x4x9"] = 107;

    n2knownRanks["4x5x5"] = 73;
    n2knownRanks["4x5x6"] = 89;
    n2knownRanks["4x5x9"] = 133;

    n2knownRanks["4x7x9"] = 187;

    n2knownRanks["5x5x9"] = 166; // ?
    n2knownRanks["5x6x10"] = 217; // ?
    n2knownRanks["5x7x8"] = 206; // ?

    n2knownRanks["7x7x7"] = 261; // ?
    n2knownRanks["7x7x8"] = 292; // ?
#endif

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

bool FlipGraph::initializeFromFile(std::istream &f) {
    int count;
    f >> count;
    std::cout << "Start reading " << std::min(count, schemesCount) << " schemes" << std::endl;

    for (int i = 0; i < count && i < schemesCount; i++)
        if (!schemes[i].read(f))
            return false;

    initializeCopyKernel<<<numBlocks, blockSize>>>(schemes, schemesCount, count);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}

void FlipGraph::initializeNaive() {
    std::cout << "Start initializing with naive schemes" << std::endl;

    initializeNaiveKernel<<<numBlocks, blockSize>>>(schemes, schemesCount, n1, n2, n3);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void FlipGraph::optimize() {
    randomWalkKernel<<<numBlocks, blockSize>>>(schemes, schemesBest, bestRanks, flips, states, schemesCount, maxIterations, probabilities.reduce, probabilities.expand, probabilities.sandwiching, probabilities.basis);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void FlipGraph::projectExtend() {
    projectExtendKernel<<<numBlocks, blockSize>>>(schemes, schemesBest, schemesCount, states, probabilities.extend, probabilities.project);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void FlipGraph::updateRanks(int iteration, bool save) {
    std::vector<std::string> keys(schemesCount);
    std::unordered_map<std::string, int> n2bestIndex;

    for (int i = 0; i < schemesCount; i++) {
        keys[i] = getKey(schemes[i]);

        auto result = n2bestRank.find(keys[i]);
        if (result == n2bestRank.end() || schemes[i].m < result->second) {
            n2bestRank[keys[i]] = schemes[i].m;
            n2bestIndex[keys[i]] = i;
            schemes[i].copyTo(schemesBest[i]);
        }
    }

    for (int i = 0; i < schemesCount; i++)
        bestRanks[i] = n2bestRank[keys[i]];

    if (save) {
        for (auto pair : n2bestIndex) {
            std::string savePath = getSavePath(schemes[pair.second], iteration, pair.second);
            schemes[pair.second].save(savePath);
            std::cout << "Best rank of " << pair.first << " was improved to " << bestRanks[pair.second] << " (known: " << n2knownRanks[pair.first] << ")! Scheme saved to \"" << savePath << "\"" << std::endl;
        }
    }

    if (n2bestIndex.size())
        std::cout << std::endl;
}

void FlipGraph::run() {
    initialize();
    updateRanks(0, false);

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
            updateRanks(iteration, true);
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
            schemesBest[index].save(savePath);

            std::cout << "Best rank of " << n << " was improved from " << n2bestRank[n] << " to " << bestRanks[index] << " (known: " << n2knownRanks[n] << ")! Scheme saved to \"" << savePath << "\"" << std::endl;
            n2bestRank[n] = bestRanks[index];
        }
    }

    std::cout << "+-----------+-----------+--------+--------+--------+-------+------+------+-------------+" << std::endl;
    std::cout << "|  elapsed  | iteration | run id |  size  |  real  | known | best | curr | flips count |" << std::endl;
    std::cout << "+-----------+-----------+--------+--------+--------+-------+------+------+-------------+" << std::endl;

    std::sort(keys.begin(), keys.end());

    for (auto key : keys) {
        const std::vector<int> &indices = n2indices[key];

        for (int i = 0; i < count && i < indices.size(); i++) {
            Scheme &scheme = schemes[indices[i]];

            std::cout << "| ";
            std::cout << std::setw(9) << prettyTime(elapsed) << " | ";
            std::cout << std::setw(9) << iteration << " | ";
            std::cout << std::setw(6) << (indices[i] + 1) << " | ";
            std::cout << std::setw(6) << key << " | ";
            std::cout << std::setw(6) << getKey(scheme, false) << " | ";
            std::cout << std::setw(5) << n2knownRanks[key] << " | ";
            std::cout << std::setw(4) << bestRanks[indices[i]] << " | ";
            std::cout << std::setw(4) << scheme.m << " | ";
            std::cout << std::setw(11) << prettyFlips(flips[indices[i]]) << " |";

            if (i == 0) {
                std::cout << " total: " << indices.size();

                if (bestRanks[indices[0]] <= n2knownRanks[key])
                    std::cout << ", " << (bestRanks[indices[0]] == n2knownRanks[key] ? "equal" : "BETTER!!!");
            }

            std::cout << std::endl;
        }

        std::cout << "+-----------+-----------+--------+--------+--------+-------+------+------+-------------+" << std::endl;

        int period = 1 + rand() % 10;
        for (size_t i = 0; i < indices.size(); i++)
            if (i % (iteration % period + 1) == 0)
                schemesBest[indices[0]].copyTo(schemes[indices[i]]);
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

std::string FlipGraph::getSavePath(const Scheme &scheme, int iteration, int runId) const {
    std::stringstream ss;

    ss << path << "/";
    ss << getKey(scheme, true);
    ss << "_m" << scheme.m;
    ss << "_c" << scheme.getComplexity();
    ss << "_iteration" << iteration;
    ss << "_run" << runId;
    ss << "_" << getKey(scheme, false);
    ss << "_" << ring << ".json";

    return ss.str();
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

    if (schemesBest) {
        cudaFree(schemesBest);
        schemesBest = nullptr;
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
__global__ void initializeNaiveKernel(Scheme *schemes, int schemesCount, int n1, int n2, int n3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < schemesCount)
        schemes[idx].initializeNaive(n1, n2, n3);
}

__global__ void initializeCopyKernel(Scheme *schemes, int schemesCount, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count || idx >= schemesCount)
        return;

    schemes[idx % count].copyTo(schemes[idx]);
}

__global__ void initializeSchemesKernel(Scheme *schemes, Scheme *schemesBest, int *bestRanks, int *flips, curandState *states, int n1, int n2, int n3, int schemesCount, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= schemesCount)
        return;

    if (!schemes[idx].validate())
        printf("not valid initialized scheme %d\n", idx);

    schemes[idx].copyTo(schemesBest[idx]);
    curand_init(seed, idx, 0, &states[idx]);

    bestRanks[idx] = n1 * n2 * n3;
    flips[idx] = 0;
}

__global__ void randomWalkKernel(Scheme *schemes, Scheme *schemesBest, int *bestRanks, int *flips, curandState *states, int schemesCount, int maxIterations, double reduceProbability, double expandProbability, double sandwichingProbability, double basisProbability) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= schemesCount)
        return;

    Scheme& scheme = schemes[idx];
    curandState& state = states[idx];
    int flipsCount = flips[idx];
    int bestRank = bestRanks[idx];
    int iterations = randint(1, maxIterations, state);

    for (int iteration = 0; iteration < iterations; iteration++) {
        if (!scheme.tryFlip(state)) {
            scheme.tryExpand(randint(1, 2, state), state);
            continue;
        }

        flipsCount++;

        if (scheme.m < bestRank || (scheme.m == bestRank && curand_uniform(&state) < 0.5)) {
            bestRank = scheme.m;
            scheme.copyTo(schemesBest[idx]);
        }

        if (curand_uniform(&state) * maxIterations < reduceProbability)
            scheme.tryReduce();

        if (curand_uniform(&state) * maxIterations < expandProbability)
            scheme.tryExpand(randint(1, 2, state), state);

        if (curand_uniform(&state) * maxIterations < sandwichingProbability)
            scheme.sandwiching(state);

        if (curand_uniform(&state) * maxIterations < basisProbability)
            scheme.swapBasis(state);
    }

    flips[idx] = flipsCount;
    bestRanks[idx] = bestRank;

    if (!scheme.validate())
        printf("invalid (%d) scheme (random walk)\n", idx);
}

__global__ void projectExtendKernel(Scheme *schemes, Scheme *schemesBest, int schemesCount, curandState *states, double extendProbability, double projectProbability) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= schemesCount)
        return;

    Scheme &scheme = schemes[idx];
    curandState &state = states[idx];

    if (curand_uniform(&state) < 0.5)
        scheme.swapSize(state);

    int index = curand(&state) % schemesCount;
    scheme.tryMerge(schemesBest[index], state);

    int n1 = min(max(scheme.n[0], MIN_PROJECT_N1), MAX_EXTENSION_N1);
    int n2 = min(max(scheme.n[1], MIN_PROJECT_N2), MAX_EXTENSION_N2);
    int n3 = min(max(scheme.n[2], MIN_PROJECT_N3), MAX_EXTENSION_N3);

    double d1 = double(n1 - MIN_PROJECT_N1) / (MAX_EXTENSION_N1 - MIN_PROJECT_N1);
    double d2 = double(n2 - MIN_PROJECT_N2) / (MAX_EXTENSION_N2 - MIN_PROJECT_N2);
    double d3 = double(n3 - MIN_PROJECT_N3) / (MAX_EXTENSION_N3 - MIN_PROJECT_N3);
    double d = (d1 + d2 + d3) / 3.0;

    if (curand_uniform(&state) < extendProbability * (1 - d)) {
        if (curand(&state) & 1) {
            scheme.tryExtend(state);
        }
        else {
            scheme.tryProduct(state);
        }
    }

    if (curand_uniform(&state) < projectProbability * d)
        scheme.tryProject(state);

    if (!scheme.validate())
        printf("invalid (%d) scheme (project extend)\n", idx);
}
