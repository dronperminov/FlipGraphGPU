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

    n2knownRanks = {
        {"222", 7}, {"223", 11}, {"224", 14}, {"225", 18}, {"226", 21}, {"227", 25}, {"228", 28},
        {"233", 15}, {"234", 20}, {"235", 25}, {"236", 30}, {"237", 35}, {"238", 40},
        {"244", 26}, {"245", 32}, {"246", 39}, {"247", 45}, {"248", 51},
        {"255", 40}, {"256", 47}, {"257", 55}, {"258", 63},
        {"266", 56}, {"267", 66}, {"268", 75},
        {"277", 76}, {"278", 88},
        {"288", 100},
        {"333", 23}, {"334", 29}, {"335", 36}, {"336", 40}, {"337", 49}, {"338", 55},
        {"344", 38}, {"345", 47}, {"346", 54}, {"347", 63}, {"348", 73},
        {"355", 58}, {"356", 68}, {"357", 79}, {"358", 90},
        {"366", 80}, {"367", 94}, {"368", 108},
        {"377", 111}, {"378", 126},
        {"388", 145},
        {"444", 48}, {"445", 61}, {"446", 73}, {"447", 85}, {"448", 96},
        {"455", 76}, {"456", 90}, {"457", 104}, {"458", 118},
        {"466", 105}, {"467", 123}, {"468", 140},
        {"477", 144}, {"478", 164},
        {"488", 182},
        {"555", 93}, {"556", 110}, {"557", 127}, {"558", 144},
        {"566", 130}, {"567", 150}, {"568", 170},
        {"577", 176}, {"578", 205},
        {"588", 230},
        {"666", 153}, {"667", 183}, {"668", 203},
        {"677", 215}, {"678", 239},
        {"688", 266},
        {"777", 249}, {"778", 277},
        {"788", 306},
        {"888", 336}
    };

#ifdef SCHEME_INTEGER
    n2knownRanks["245"] = 33;
    n2knownRanks["257"] = 57;
    n2knownRanks["258"] = 66;
    n2knownRanks["266"] = 57;
    n2knownRanks["267"] = 69;
    n2knownRanks["268"] = 78;
    n2knownRanks["277"] = 77;
    n2knownRanks["278"] = 90;
    n2knownRanks["336"] = 44;
    n2knownRanks["337"] = 51;
    n2knownRanks["338"] = 58;
    n2knownRanks["346"] = 57;
    n2knownRanks["347"] = 66;
    n2knownRanks["348"] = 74;
    n2knownRanks["356"] = 70;
    n2knownRanks["357"] = 84;
    n2knownRanks["358"] = 94;
    n2knownRanks["366"] = 85;
    n2knownRanks["367"] = 101;
    n2knownRanks["368"] = 114;
    n2knownRanks["377"] = 119;
    n2knownRanks["378"] = 132;
    n2knownRanks["388"] = 148;
    n2knownRanks["444"] = 49;
    n2knownRanks["445"] = 61;
    n2knownRanks["446"] = 73;
    n2knownRanks["448"] = 96;
    n2knownRanks["477"] = 148;
    n2knownRanks["568"] = 176;
    n2knownRanks["577"] = 185;
    n2knownRanks["578"] = 208;
    n2knownRanks["667"] = 185;
    n2knownRanks["777"] = 281;
    n2knownRanks["778"] = 302;
#else
    n2knownRanks["444"] = 47;
    n2knownRanks["445"] = 60;
    n2knownRanks["455"] = 73;
    n2knownRanks["456"] = 89;
    n2knownRanks["448"] = 94;

    // maybe
    n2knownRanks["245"] = 33;
    n2knownRanks["257"] = 57; // ?
    n2knownRanks["258"] = 66; // ?
    n2knownRanks["268"] = 78; // ?
    n2knownRanks["277"] = 77; // ?
    n2knownRanks["336"] = 42; // ?
    n2knownRanks["338"] = 58; // ?
    n2knownRanks["347"] = 64; // ?
    n2knownRanks["348"] = 74; // ?
    n2knownRanks["366"] = 84; // ?
    n2knownRanks["367"] = 98; // ?
    n2knownRanks["377"] = 116; // ?
    n2knownRanks["378"] = 128; // ?
    n2knownRanks["578"] = 207; // ?
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
            schemes[i].copyTo(schemesBest[i]);
        }
    }

    for (int i = 0; i < schemesCount; i++)
        bestRanks[i] = n2bestRank[keys[i]];

    for (auto pair : n2bestIndex) {
        std::string savePath = getSavePath(schemes[pair.second], iteration, pair.second);
        schemes[pair.second].save(savePath);
        std::cout << "Best rank of " << pair.first << " was improved to " << bestRanks[pair.second] << " (known: " << n2knownRanks[pair.first] << ")! Scheme saved to \"" << savePath << "\"" << std::endl;
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
            schemesBest[index].save(savePath);

            std::cout << "Best rank of " << n << " was improved from " << n2bestRank[n] << " to " << bestRanks[index] << " (known: " << n2knownRanks[n] << ")! Scheme saved to \"" << savePath << "\"" << std::endl;
            n2bestRank[n] = bestRanks[index];
        }
    }

    std::cout << "+-----------+-----------+--------+------+------+-------+------+------+-------------+" << std::endl;
    std::cout << "|  elapsed  | iteration | run id | size | real | known | best | curr | flips count |" << std::endl;
    std::cout << "+-----------+-----------+--------+------+------+-------+------+------+-------------+" << std::endl;

    std::sort(keys.begin(), keys.end());

    for (auto key : keys) {
        const std::vector<int> &indices = n2indices[key];

        for (int i = 0; i < count && i < indices.size(); i++) {
            Scheme &scheme = schemes[indices[i]];

            std::cout << "| ";
            std::cout << std::setw(9) << prettyTime(elapsed) << " | ";
            std::cout << std::setw(9) << iteration << " | ";
            std::cout << std::setw(6) << (indices[i] + 1) << " | ";
            std::cout << std::setw(4) << key << " | ";
            std::cout << " " << scheme.n[0] << scheme.n[1] << scheme.n[2] << " | ";
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

        std::cout << "+-----------+-----------+--------+------+------+-------+------+------+-------------+" << std::endl;

        for (size_t i = 0; i < indices.size(); i++)
            if (i % (iteration % 10 + 1) == 0)
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
    ss << "_" << mod << ".json";

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

        if (scheme.m < bestRank) {
            bestRank = scheme.m;
            scheme.copyTo(schemesBest[idx]);
        }

        if (curand_uniform(&state) * maxIterations < reduceProbability) {
            scheme.tryReduce();
        }

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

__global__ void projectExtendKernel(Scheme *schemes, int schemesCount, curandState *states, double extendProbability, double projectProbability) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= schemesCount)
        return;

    Scheme &scheme = schemes[idx];
    curandState &state = states[idx];

    double d1 = double(scheme.n[0] - MIN_PROJECT_N1) / (MAX_EXTENSION_N1 - MIN_PROJECT_N1);
    double d2 = double(scheme.n[1] - MIN_PROJECT_N2) / (MAX_EXTENSION_N2 - MIN_PROJECT_N2);
    double d3 = double(scheme.n[2] - MIN_PROJECT_N3) / (MAX_EXTENSION_N3 - MIN_PROJECT_N3);
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
