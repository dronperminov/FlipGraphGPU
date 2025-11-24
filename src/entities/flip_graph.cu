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
        {"2x2x2", 7}, {"2x2x3", 11}, {"2x2x4", 14}, {"2x2x5", 18}, {"2x2x6", 21}, {"2x2x7", 25}, {"2x2x8", 28}, {"2x2x9", 32}, {"2x2x10", 35}, {"2x2x11", 39}, {"2x2x12", 42}, {"2x2x13", 46}, {"2x2x14", 49}, {"2x2x15", 53}, {"2x2x16", 56},
        {"2x3x3", 15}, {"2x3x4", 20}, {"2x3x5", 25}, {"2x3x6", 30}, {"2x3x7", 35}, {"2x3x8", 40}, {"2x3x9", 45}, {"2x3x10", 50}, {"2x3x11", 55}, {"2x3x12", 60}, {"2x3x13", 65}, {"2x3x14", 70}, {"2x3x15", 75}, {"2x3x16", 80},
        {"2x4x4", 26}, {"2x4x5", 32}, {"2x4x6", 39}, {"2x4x7", 45}, {"2x4x8", 51}, {"2x4x9", 59}, {"2x4x10", 64}, {"2x4x11", 71}, {"2x4x12", 77}, {"2x4x13", 83}, {"2x4x14", 90}, {"2x4x15", 96}, {"2x4x16", 102},
        {"2x5x5", 40}, {"2x5x6", 47}, {"2x5x7", 55}, {"2x5x8", 63}, {"2x5x9", 72}, {"2x5x10", 79}, {"2x5x11", 87}, {"2x5x12", 94},
        {"2x6x6", 56}, {"2x6x7", 66}, {"2x6x8", 75}, {"2x6x9", 86}, {"2x6x10", 94},
        {"2x7x7", 76}, {"2x7x8", 88}, {"2x7x9", 99},
        {"2x8x8", 100},
        {"3x3x3", 23}, {"3x3x4", 29}, {"3x3x5", 36}, {"3x3x6", 40}, {"3x3x7", 49}, {"3x3x8", 55}, {"3x3x9", 63}, {"3x3x10", 69}, {"3x3x11", 76}, {"3x3x12", 80}, {"3x3x13", 89}, {"3x3x14", 95}, {"3x3x15", 103}, {"3x3x16", 109},
        {"3x4x4", 38}, {"3x4x5", 47}, {"3x4x6", 54}, {"3x4x7", 63}, {"3x4x8", 73}, {"3x4x9", 83}, {"3x4x10", 92}, {"3x4x11", 101}, {"3x4x12", 108}, {"3x4x13", 117}, {"3x4x14", 126}, {"3x4x15", 136}, {"3x4x16", 146},
        {"3x5x5", 58}, {"3x5x6", 68}, {"3x5x7", 79}, {"3x5x8", 90}, {"3x5x9", 104}, {"3x5x10", 115}, {"3x5x11", 126}, {"3x5x12", 136},
        {"3x6x6", 80}, {"3x6x7", 94}, {"3x6x8", 108}, {"3x6x9", 120}, {"3x6x10", 134},
        {"3x7x7", 111}, {"3x7x8", 126}, {"3x7x9", 142},
        {"3x8x8", 145},
        {"4x4x4", 48}, {"4x4x5", 61}, {"4x4x6", 73}, {"4x4x7", 85}, {"4x4x8", 96}, {"4x4x9", 104}, {"4x4x10", 120}, {"4x4x11", 130}, {"4x4x12", 142}, {"4x4x13", 152}, {"4x4x14", 165}, {"4x4x15", 177}, {"4x4x16", 189},
        {"4x5x5", 76}, {"4x5x6", 90}, {"4x5x7", 104}, {"4x5x8", 118}, {"4x5x9", 136}, {"4x5x10", 151}, {"4x5x11", 165}, {"4x5x12", 179},
        {"4x6x6", 105}, {"4x6x7", 123}, {"4x6x8", 140}, {"4x6x9", 159}, {"4x6x10", 175},
        {"4x7x7", 144}, {"4x7x8", 164}, {"4x7x9", 186},
        {"4x8x8", 182},
        {"5x5x5", 93}, {"5x5x6", 110}, {"5x5x7", 127}, {"5x5x8", 144}, {"5x5x9", 167}, {"5x5x10", 184}, {"5x5x11", 202}, {"5x5x12", 220},
        {"5x6x6", 130}, {"5x6x7", 150}, {"5x6x8", 170}, {"5x6x9", 197}, {"5x6x10", 217},
        {"5x7x7", 176}, {"5x7x8", 205}, {"5x7x9", 229},
        {"5x8x8", 230},
        {"6x6x6", 153}, {"6x6x7", 183}, {"6x6x8", 203}, {"6x6x9", 225}, {"6x6x10", 247},
        {"6x7x7", 215}, {"6x7x8", 239}, {"6x7x9", 269},
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

    n2knownRanks["4x6x9"] = 160;

    n2knownRanks["4x7x7"] = 147;
    n2knownRanks["4x7x9"] = 188;

    n2knownRanks["5x7x8"] = 206;

    n2knownRanks["6x6x7"] = 185;
    n2knownRanks["6x6x9"] = 225;

    n2knownRanks["7x7x7"] = 250;
    n2knownRanks["7x7x8"] = 279;
    n2knownRanks["7x7x9"] = 316;
    n2knownRanks["7x8x8"] = 310;
    n2knownRanks["8x8x8"] = 343;
#else
    n2knownRanks["2x4x5"] = 33;
    n2knownRanks["2x4x10"] = 65;

    n2knownRanks["2x5x7"] = 57; // ?
    n2knownRanks["2x5x8"] = 65; // ?
    n2knownRanks["2x5x10"] = 80; // ?

    n2knownRanks["2x6x8"] = 77; // ?
    n2knownRanks["2x7x7"] = 77; // ?
    n2knownRanks["2x7x9"] = 101; // ?

    n2knownRanks["3x3x6"] = 42; // ?
    n2knownRanks["3x3x8"] = 56; // ?
    n2knownRanks["3x3x9"] = 64; // ?
    n2knownRanks["3x3x10"] = 71; // ?
    n2knownRanks["3x3x15"] = 105; // ?
    n2knownRanks["3x3x16"] = 112; // ?

    n2knownRanks["3x4x7"] = 64; // ?
    n2knownRanks["3x4x8"] = 74; // ?

    n2knownRanks["3x6x6"] = 84; // ?
    n2knownRanks["3x6x7"] = 96; // ?
    n2knownRanks["3x6x9"] = 122; // ?
    n2knownRanks["3x6x10"] = 136; // ?

    n2knownRanks["3x7x7"] = 113; // ?
    n2knownRanks["3x7x8"] = 128; // ?
    n2knownRanks["3x7x9"] = 143; // ?

    n2knownRanks["4x4x4"] = 47;
    n2knownRanks["4x4x5"] = 60;
    n2knownRanks["4x4x8"] = 94;
    n2knownRanks["4x4x9"] = 107;
    n2knownRanks["4x4x12"] = 141;
    n2knownRanks["4x4x16"] = 188;

    n2knownRanks["4x5x5"] = 73;
    n2knownRanks["4x5x6"] = 89;
    n2knownRanks["4x5x9"] = 133;
    n2knownRanks["4x5x10"] = 146;
    n2knownRanks["4x5x11"] = 162;
    n2knownRanks["4x5x12"] = 177;

    n2knownRanks["4x7x9"] = 187;

    n2knownRanks["5x5x9"] = 166; // ?
    n2knownRanks["5x5x10"] = 183; // ?
    n2knownRanks["5x5x11"] = 200; // ?
    n2knownRanks["5x5x12"] = 217; // ?
    n2knownRanks["5x7x8"] = 206; // ?

    n2knownRanks["6x6x10"] = 252; // ?

    n2knownRanks["7x7x7"] = 248; // ?
    n2knownRanks["7x7x8"] = 275; // ?
    n2knownRanks["7x7x9"] = 313; // ?
    n2knownRanks["7x8x8"] = 302; // ?
    n2knownRanks["8x8x8"] = 329; // ?
#endif

    CUDA_CHECK(cudaMallocManaged(&schemes, schemesCount * sizeof(Scheme)));
    CUDA_CHECK(cudaMallocManaged(&schemesBest, schemesCount * sizeof(Scheme)));
    CUDA_CHECK(cudaMallocManaged(&bestRanks, schemesCount * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&flips, schemesCount * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&states, schemesCount * sizeof(curandState)));

    initializeRandomStatesKernel<<<numBlocks, blockSize>>>(states, schemesCount, seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void FlipGraph::initialize() {
    std::cout << "Start main initialization" << std::endl;
    initializeSchemesKernel<<<numBlocks, blockSize>>>(schemes, schemesBest, bestRanks, flips, n1, n2, n3, schemesCount);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    updateRanks(0, false);
    std::cout << "Initialized" << std::endl;
}

bool FlipGraph::initializeFromFile(std::istream &f) {
    int count;
    f >> count;
    std::cout << "Start reading " << std::min(count, schemesCount) << " schemes" << std::endl;

    for (int i = 0; i < count && i < schemesCount; i++) {
        if (!schemes[i].read(f, false))
            return false;

        while (schemes[i].tryReduce())
            ;
    }

    std::cout << "Start copying " << std::min(count, schemesCount) << " readed schemes" << std::endl;
    initializeCopyKernel<<<numBlocks, blockSize>>>(schemes, schemesCount, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // std::cout << "Start resizing " << schemesCount << " schemes to " << n1 << "x" << n2 << "x" << n3 << std::endl;
    // initializeResizeKernel<<<numBlocks, blockSize>>>(schemes, schemesCount, n1, n2, n3, states);
    // CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}

void FlipGraph::initializeNaive() {
    std::cout << "Start initializing with naive schemes" << std::endl;

    initializeNaiveKernel<<<numBlocks, blockSize>>>(schemes, schemesCount, n1, n2, n3);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void FlipGraph::randomWalk() {
    randomWalkKernel<<<numBlocks, blockSize>>>(schemes, schemesBest, bestRanks, flips, states, schemesCount, maxIterations, probabilities.reduce, probabilities.expand, probabilities.sandwiching, probabilities.basis);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void FlipGraph::resize() {
    resizeKernel<<<numBlocks, blockSize>>>(schemes, schemesBest, schemesCount, states, probabilities.resize);

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

        if (n2bestIndex.size())
            std::cout << std::endl;
    }
}

void FlipGraph::run(int logPeriod) {
    initialize();

    auto startTime = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<double> elapsedTimes;

    for (int iteration = 0; 1; iteration++) {
        randomWalk();
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        elapsedTimes.push_back(duration.count() / 1000.0);

        report(startTime, iteration + 1, elapsedTimes, logPeriod);

        t1 = std::chrono::high_resolution_clock::now();

        if (probabilities.resize > 0) {
            resize();
            updateRanks(iteration, true);
        }
    }
}

void FlipGraph::report(std::chrono::high_resolution_clock::time_point startTime, int iteration, const std::vector<double> &elapsedTimes, int logPeriod, int count) {
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

    if (logPeriod == 0 || iteration % logPeriod == 0) {
        std::cout << "+-----------+-----------+--------+--------+--------+-------+------+------+-------------+" << std::endl;
        std::cout << "|  elapsed  | iteration | run id |  size  |  real  | known | best | curr | flips count |" << std::endl;
        std::cout << "+-----------+-----------+--------+--------+--------+-------+------+------+-------------+" << std::endl;

        std::sort(keys.begin(), keys.end(), [n2indices, this](std::string &s1, std::string &s2){
            return compareKeys(schemes[n2indices.at(s1)[0]], schemes[n2indices.at(s2)[0]]);
        });

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
                std::cout << std::setw(11) << prettyInt(flips[indices[i]]) << " |";

                if (i == 0) {
                    std::cout << " total: " << indices.size();

                    if (bestRanks[indices[0]] <= n2knownRanks[key])
                        std::cout << ", " << (bestRanks[indices[0]] == n2knownRanks[key] ? "equal" : "BETTER!!!");
                }

                std::cout << std::endl;
            }

            std::cout << "+-----------+-----------+--------+--------+--------+-------+------+------+-------------+" << std::endl;

            // int period = 1 + rand() % 10;
            // for (size_t i = 0; i < indices.size(); i++)
            //     if (i % (iteration % period + 1) == 0)
            //         schemesBest[indices[0]].copyTo(schemes[indices[i]]);
        }

        std::cout << "- iteration time (last / min / max / mean): " << prettyTime(lastTime) << " / " << prettyTime(minTime) << " / " << prettyTime(maxTime) << " / " << prettyTime(meanTime) << std::endl;
        std::cout << std::endl;
    }
}

bool FlipGraph::compareKeys(const Scheme &s1, const Scheme &s2) const {
    return getKey(s1, true, true) < getKey(s2, true, true);
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
__global__ void initializeRandomStatesKernel(curandState *states, int schemesCount, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= schemesCount)
        return;

    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void initializeNaiveKernel(Scheme *schemes, int schemesCount, int n1, int n2, int n3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < schemesCount)
        schemes[idx].initializeNaive(n1, n2, n3);
}

__global__ void initializeCopyKernel(Scheme *schemes, int schemesCount, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= schemesCount)
        return;

    if (idx >= count) {
        schemes[idx % count].copyTo(schemes[idx]);
    }
    else if (!schemes[idx].validate()) {
        printf("not valid initialized scheme %d\n", idx);
    }
}

__global__ void initializeResizeKernel(Scheme *schemes, int schemesCount, int n1, int n2, int n3, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= schemesCount)
        return;

    Scheme &scheme = schemes[idx];
    curandState& state = states[idx];

    int n[3] = {n1, n2, n3};

    while (scheme.n[0] != n[0] || scheme.n[1] != n[1] || scheme.n[2] != n[2]) {
        int pmax = 0;

        if (scheme.n[1] > scheme.n[pmax])
            pmax = 1;

        if (scheme.n[2] > scheme.n[pmax])
            pmax = 2;

        int p = pmax;

        if (scheme.n[pmax] <= n[pmax]) {
            do {
                p = curand(&state) % 3;
            } while (scheme.n[p] == n[p]);
        }

        if (scheme.n[p] > n[p])
            scheme.project(p, curand(&state) % scheme.n[p]);
        else if (scheme.n[p] * 2 <= n[p] && curand(&state) & 1 && scheme.isValidProduct(p))
            scheme.product(p);
        else if (scheme.isValidExtension(p))
            scheme.extend(p);
        else
            break;
    }
}

__global__ void initializeSchemesKernel(Scheme *schemes, Scheme *schemesBest, int *bestRanks, int *flips, int n1, int n2, int n3, int schemesCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= schemesCount)
        return;

    schemes[idx].copyTo(schemesBest[idx], false);
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

        if (scheme.m < bestRank || (scheme.m == bestRank && curand_uniform(&state) < 0.01)) {
            bestRank = scheme.m;
            scheme.copyTo(schemesBest[idx], false);
        }

        if (curand_uniform(&state) * maxIterations < reduceProbability)
            scheme.tryReduce();

        if (curand_uniform(&state) * maxIterations < expandProbability && scheme.m <= schemesBest[idx].m + 2)
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

__global__ void resizeKernel(Scheme *schemes, Scheme *schemesBest, int schemesCount, curandState *states, double resizeProbability) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= schemesCount)
        return;

    Scheme &scheme = schemes[idx];
    curandState &state = states[idx];

    if (curand_uniform(&state) < 0.5)
        scheme.swapSize(state);

    bool merged = false;

    for (int i = 0; i < 3 && !merged; i++) {
        int index = curand(&state) % schemesCount;
        merged |= scheme.tryMerge(schemesBest[index], state);
    }

    if (!merged && curand_uniform(&state) < resizeProbability) {
        float p = curand_uniform(&state);
        if (p < 0.05) {
            scheme.tryProject(state);
        }
        else if (p < 0.55) {
            int index = curand(&state) % schemesCount;
            scheme.tryProduct(schemesBest[index]);
        }
        else if (p < 0.85) {
            scheme.tryProduct(state);
        }
        else {
            scheme.tryExtend(state);
        }
    }

    if (!scheme.validate())
        printf("invalid (%d) scheme (resize)\n", idx);
}
