#include "complexity_minimizer.cuh"

ComplexityMinimizer::ComplexityMinimizer(int schemesCount, int blockSize, int maxIterations, const std::string &path, int seed, int topCount) {
    this->schemesCount = schemesCount;
    this->maxIterations = maxIterations;
    this->path = path;
    this->seed = seed;
    this->topCount = topCount;

    this->blockSize = blockSize;
    this->numBlocks = (schemesCount + blockSize - 1) / blockSize;

    this->indices.reserve(schemesCount);
    for (int i = 0; i < schemesCount; i++)
        this->indices.push_back(i);

    CUDA_CHECK(cudaMallocManaged(&schemes, schemesCount * sizeof(Scheme)));
    CUDA_CHECK(cudaMallocManaged(&schemesBest, schemesCount * sizeof(Scheme)));
    CUDA_CHECK(cudaMallocManaged(&bestComplexities, schemesCount * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&states, schemesCount * sizeof(curandState)));
}

bool ComplexityMinimizer::read(std::istream &f) {
    int n1, n2, n3, m;
    f >> n1 >> n2 >> n3 >> m >> initialCount;

    std::cout << "Start reading " << initialCount << " schemes (" << n1 << "x" << n2 << "x" << n3 << ": " << m << ")" << std::endl;

    for (int i = 0; i < initialCount; i++)
        if (!schemes[i].read(f, n1, n2, n3, m, false))
            return false;

    std::cout << "Successfully read " << initialCount << " schemes" << std::endl;
    return true;
}

void ComplexityMinimizer::minimize(int targetComplexity, int maxNoImprovements) {
    initialize();

    auto startTime = std::chrono::high_resolution_clock::now();
    std::vector<double> elapsedTimes;

    int noImprovements = 0;

    for (int iteration = 1; bestComplexity > targetComplexity; iteration++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        minimizeIteration();
        bool improved = updateBest(iteration);
        auto t2 = std::chrono::high_resolution_clock::now();

        elapsedTimes.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 1000.0);
        report(startTime, iteration, elapsedTimes);

        if (improved) {
            noImprovements = 0;
        }
        else {
            noImprovements++;
            std::cout << "No improvements for " << noImprovements << " iterations" << std::endl;

            if (noImprovements >= maxNoImprovements)
                break;
        }
    }
}

void ComplexityMinimizer::initialize() {
    bestComplexity = schemes[0].getComplexity();

    for (int i = 1; i < initialCount; i++)
        bestComplexity = std::min(bestComplexity,schemes[i].getComplexity());

    initializeKernel<<<numBlocks, blockSize>>>(schemes, schemesBest, bestComplexities, states, schemesCount, initialCount, bestComplexity, seed);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void ComplexityMinimizer::minimizeIteration() {
    minimizeKernel<<<numBlocks, blockSize>>>(schemes, schemesBest, bestComplexities, states, schemesCount, maxIterations);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

bool ComplexityMinimizer::updateBest(int iteration) {
    std::partial_sort(indices.begin(), indices.begin() + topCount, indices.end(), [this](int index1, int index2) { return bestComplexities[index1] < bestComplexities[index2]; });
    int topIndex = indices[0];

    if (bestComplexities[topIndex] >= bestComplexity)
        return false;

    std::string savePath = getSavePath(schemesBest[topIndex], iteration, topIndex);
    schemesBest[topIndex].save(savePath);

    std::cout << "Best complexity improved from " << bestComplexity << " to " << bestComplexities[topIndex] << "! Scheme saved to \"" << savePath << "\"" << std::endl;
    bestComplexity = bestComplexities[topIndex];
    return true;
}

void ComplexityMinimizer::report(std::chrono::high_resolution_clock::time_point startTime, int iteration, const std::vector<double> &elapsedTimes) {
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

    double lastTime = elapsedTimes[elapsedTimes.size() - 1];
    double minTime = *std::min_element(elapsedTimes.begin(), elapsedTimes.end());
    double maxTime = *std::max_element(elapsedTimes.begin(), elapsedTimes.end());
    double meanTime = std::accumulate(elapsedTimes.begin(), elapsedTimes.end(), 0.0) / elapsedTimes.size();

    std::cout << "+-----------+-----------+--------+--------+--------+" << std::endl;
    std::cout << "|  elapsed  | iteration | run id |  best  |  curr  |" << std::endl;
    std::cout << "+-----------+-----------+--------+--------+--------+" << std::endl;

    for (int i = 0; i < topCount && i < schemesCount; i++) {
        Scheme &scheme = schemes[indices[i]];
        int complexity = scheme.getComplexity();

        std::cout << "| ";
        std::cout << std::setw(9) << prettyTime(elapsed) << " | ";
        std::cout << std::setw(9) << iteration << " | ";
        std::cout << std::setw(6) << (indices[i] + 1) << " | ";
        std::cout << std::setw(6) << bestComplexities[indices[i]] << " | ";
        std::cout << std::setw(6) << complexity << " |";
        std::cout << std::endl;
    }

    std::cout << "+-----------+-----------+--------+--------+--------+" << std::endl;
    std::cout << "- iteration time (last / min / max / mean): " << prettyTime(lastTime) << " / " << prettyTime(minTime) << " / " << prettyTime(maxTime) << " / " << prettyTime(meanTime) << std::endl;
    std::cout << std::endl;

    int period = 1 + rand() % 10;
    for (size_t i = 0; i < indices.size(); i++)
        if (i % (iteration % period + 1) == 0)
            schemesBest[indices[0]].copyTo(schemes[indices[i]]);
}

std::string ComplexityMinimizer::getSavePath(const Scheme &scheme, int iteration, int runId) const {
    std::stringstream ss;

    ss << path << "/";
    ss << getKey(scheme);
    ss << "_m" << scheme.m;
    ss << "_c" << scheme.getComplexity();
    ss << "_iteration" << iteration;
    ss << "_run" << runId;
    ss << "_" << getKey(scheme, false);
    ss << "_" << ring << ".json";

    return ss.str();
}

ComplexityMinimizer::~ComplexityMinimizer() {
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
    
    if (bestComplexities) {
        cudaFree(bestComplexities);
        bestComplexities = nullptr;
    }
}

__global__ void initializeKernel(Scheme *schemes, Scheme *schemesBest, int *bestComplexities, curandState *states, int schemesCount, int initialCount, int complexity, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= schemesCount)
        return;

    if (idx >= initialCount) {
        schemes[idx % initialCount].copyTo(schemes[idx]);
    }
    else if (!schemes[idx].validate()) {
        printf("Invalid scheme %d\n", idx);
    }

    schemes[idx].copyTo(schemesBest[idx]);
    bestComplexities[idx] = complexity;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void minimizeKernel(Scheme *schemes, Scheme *schemesBest, int *bestComplexities, curandState *states, int schemesCount, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= schemesCount)
        return;

    Scheme& scheme = schemes[idx];
    curandState& state = states[idx];
    int bestComplexity = bestComplexities[idx];

    for (int iteration = 0; iteration < iterations; iteration++) {
        if (!scheme.tryFlip(state, false))
            break;

        int complexity = scheme.getComplexity();
        if (complexity < bestComplexity) {
            bestComplexity = complexity;
            scheme.copyTo(schemesBest[idx]);
        }
    }

    bestComplexities[idx] = bestComplexity;

    if (!scheme.validate())
        printf("invalid (%d) scheme (minimize)\n", idx);
}
