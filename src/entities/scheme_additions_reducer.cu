#include "scheme_additions_reducer.cuh"

SchemeAdditionsReducer::SchemeAdditionsReducer(int count, int seed, int blockSize, const std::string &outputPath, int topCount) {
    this->count = count;
    this->seed = seed;
    this->blockSize = blockSize;
    this->numBlocks = (count + blockSize - 1) / blockSize;
    this->outputPath = outputPath;
    this->topCount = topCount;

    for (int i = 0; i < 3; i++) {
        this->indices[i].reserve(count);
        this->bestAdditions[i] = 0;

        for (int j = 0; j < count; j++)
            this->indices[i].push_back(j);
    }

    CUDA_CHECK(cudaMallocManaged(&reducersU, (count + 2) * sizeof(AdditionsReducer<MAX_RANK, MAX_FRESH_VARIABLES, MAX_MATRIX_ELEMENTS>)));
    CUDA_CHECK(cudaMallocManaged(&reducersV, (count + 2) * sizeof(AdditionsReducer<MAX_RANK, MAX_FRESH_VARIABLES, MAX_MATRIX_ELEMENTS>)));
    CUDA_CHECK(cudaMallocManaged(&reducersW, (count + 2) * sizeof(AdditionsReducer<MAX_MATRIX_ELEMENTS, MAX_FRESH_VARIABLES, MAX_RANK>)));
    CUDA_CHECK(cudaMallocManaged(&states, count * sizeof(curandState)));
}

bool SchemeAdditionsReducer::read(std::ifstream &f) {
    f >> n1 >> n2 >> n3 >> m;
    std::cout << "Read scheme " << n1 << "x" << n2 << "x" << n3 << " with " << m << " multiplications" << std::endl;

    int u[MAX_RANK][MAX_MATRIX_ELEMENTS];
    int v[MAX_RANK][MAX_MATRIX_ELEMENTS];
    int w[MAX_MATRIX_ELEMENTS][MAX_RANK];

    for (int index = 0; index < m; index++)
        for (int i = 0; i < n1 * n2; i++)
            f >> u[index][i];

    for (int index = 0; index < m; index++)
        for (int i = 0; i < n2 * n3; i++)
            f >> v[index][i];

    for (int index = 0; index < m; index++)
        for (int i = 0; i < n3 * n1; i++)
            f >> w[i][index];

    bool correct = true;

    for (int i = 0; i < m && correct; i++)
        correct &= reducersU[count].addExpression(u[i], n1 * n2);

    for (int i = 0; i < m && correct; i++)
        correct &= reducersV[count].addExpression(v[i], n2 * n3);

    for (int i = 0; i < n3 * n1 && correct; i++)
        correct &= reducersW[count].addExpression(w[i], m);

    if (!correct)
        return false;

    bestAdditions[0] = reducersU[count].getAdditions();
    bestAdditions[1] = reducersV[count].getAdditions();
    bestAdditions[2] = reducersW[count].getAdditions();

    reducersU[count + 1].copyFrom(reducersU[count]);
    reducersV[count + 1].copyFrom(reducersV[count]);
    reducersW[count + 1].copyFrom(reducersW[count]);

    naiveAdditions = bestAdditions[0] + bestAdditions[1] + bestAdditions[2];
    reducedAdditions = naiveAdditions;

    std::cout << "Readed scheme uses " << bestAdditions[0] << " + " << bestAdditions[1] << " + " << bestAdditions[2] << " = " << naiveAdditions << " additions [naive]" << std::endl;
    return true;
}

void SchemeAdditionsReducer::reduce(int maxNoImprovements) {
    initialize();

    auto startTime = std::chrono::high_resolution_clock::now();
    std::vector<double> elapsedTimes;

    int noImprovements = 0;

    for (int iteration = 1; noImprovements < maxNoImprovements; iteration++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        reduceIteration();
        bool improved = updateBest();
        auto t2 = std::chrono::high_resolution_clock::now();

        elapsedTimes.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 1000.0);
        report(startTime, iteration, elapsedTimes);

        if (improved) {
            noImprovements = 0;
        }
        else {
            noImprovements++;
            std::cout << "No improvements for " << noImprovements << " iterations" << std::endl;
        }
    }
}

SchemeAdditionsReducer::~SchemeAdditionsReducer() {
    if (reducersU) {
        cudaFree(reducersU);
        reducersU = nullptr;
    }

    if (reducersV) {
        cudaFree(reducersV);
        reducersV = nullptr;
    }

    if (reducersW) {
        cudaFree(reducersW);
        reducersW = nullptr;
    }

    if (states) {
        cudaFree(states);
        states = nullptr;
    }
}

void SchemeAdditionsReducer::initialize() {
    initializeRandomKernel<<<numBlocks, blockSize>>>(states, count, seed);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void SchemeAdditionsReducer::reduceIteration() {
    runReducersKernel<<<numBlocks, blockSize>>>(reducersU, reducersV, reducersW, states, count);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

bool SchemeAdditionsReducer::updateBest() {
    std::partial_sort(indices[0].begin(), indices[0].begin() + topCount, indices[0].end(), [this](int index1, int index2) { return reducersU[index1].getAdditions() < reducersU[index2].getAdditions(); });
    std::partial_sort(indices[1].begin(), indices[1].begin() + topCount, indices[1].end(), [this](int index1, int index2) { return reducersV[index1].getAdditions() < reducersV[index2].getAdditions(); });
    std::partial_sort(indices[2].begin(), indices[2].begin() + topCount, indices[2].end(), [this](int index1, int index2) { return reducersW[index1].getAdditions() < reducersW[index2].getAdditions(); });

    int ua = reducersU[indices[0][0]].getAdditions();
    int va = reducersV[indices[1][0]].getAdditions();
    int wa = reducersW[indices[2][0]].getAdditions();

    if (ua < bestAdditions[0]) {
        bestAdditions[0] = ua;
        reducersU[count + 1].copyFrom(reducersU[indices[0][0]]);
    }

    if (va < bestAdditions[1]) {
        bestAdditions[1] = va;
        reducersV[count + 1].copyFrom(reducersV[indices[1][0]]);
    }

    if (wa < bestAdditions[2]) {
        bestAdditions[2] = wa;
        reducersW[count + 1].copyFrom(reducersW[indices[2][0]]);
    }

    int best = bestAdditions[0] + bestAdditions[1] + bestAdditions[2];

    if (best >= reducedAdditions)
        return false;

    std::cout << "Best additions improved from " << reducedAdditions << " to " << best << std::endl;
    reducedAdditions = best;
    save();
    return true;
}

void SchemeAdditionsReducer::report(std::chrono::high_resolution_clock::time_point startTime, int iteration, const std::vector<double> &elapsedTimes) {
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

    double lastTime = elapsedTimes[elapsedTimes.size() - 1];
    double minTime = *std::min_element(elapsedTimes.begin(), elapsedTimes.end());
    double maxTime = *std::max_element(elapsedTimes.begin(), elapsedTimes.end());
    double meanTime = std::accumulate(elapsedTimes.begin(), elapsedTimes.end(), 0.0) / elapsedTimes.size();

    std::cout << "+-----------+-----------+--------+--------+--------+-------+" << std::endl;
    std::cout << "|  elapsed  | iteration | curr U | curr V | curr W | total |" << std::endl;
    std::cout << "+-----------+-----------+--------+--------+--------+-------+" << std::endl;

    for (int i = 0; i < topCount && i < count; i++) {
        int currU = reducersU[indices[0][i]].getAdditions();
        int currV = reducersV[indices[1][i]].getAdditions();
        int currW = reducersW[indices[2][i]].getAdditions();
        int curr = currU + currV + currW;

        std::cout << "| ";
        std::cout << std::setw(9) << prettyTime(elapsed) << " | ";
        std::cout << std::setw(9) << iteration << " | ";
        std::cout << std::setw(6) << currU << " | ";
        std::cout << std::setw(6) << currV << " | ";
        std::cout << std::setw(6) << currW << " | ";
        std::cout << std::setw(5) << curr << " |";
        std::cout << std::endl;
    }

    std::cout << "+-----------+-----------+--------+--------+--------+-------+" << std::endl;
    std::cout << "- iteration time (last / min / max / mean): " << prettyTime(lastTime) << " / " << prettyTime(minTime) << " / " << prettyTime(maxTime) << " / " << prettyTime(meanTime) << std::endl;
    std::cout << "- best additions (U / V / W / total): " << bestAdditions[0] << " / " << bestAdditions[1] << " / " << bestAdditions[2] << " / " << reducedAdditions << std::endl;
    std::cout << std::endl;
}

void SchemeAdditionsReducer::save() const {
    std::string path = getSavePath();
    std::ofstream f(path);

    f << "{" << std::endl;
    f << "    \"n\": [" << n1 << ", " << n2 << ", " << n3 << "]," << std::endl;
    f << "    \"m\": " << m << "," << std::endl;
    f << "    \"z2\": false," << std::endl;
    f << "    \"complexity\": {\"naive\": " << naiveAdditions << ", \"reduced\": " << reducedAdditions << "}," << std::endl;
    reducersU[count + 1].write(f, "u", "    ");
    f << "," << std::endl;
    reducersV[count + 1].write(f, "v", "    ");
    f << "," << std::endl;
    reducersW[count + 1].write(f, "w", "    ");
    f << std::endl;
    f << "}" << std::endl;
    f.close();
}

std::string SchemeAdditionsReducer::getSavePath() const {
    std::stringstream ss;
    ss << outputPath << "/";
    ss << n1 << "x" << n2 << "x" << n3;
    ss << "_m" << m;
    ss << "_cr" << reducedAdditions;
    ss << "_cn" << naiveAdditions;
    ss << "_" << ring;
    ss << "_reduced.json";

    return ss.str();
}

__global__ void initializeRandomKernel(curandState *states, int count, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void runReducersKernel(AdditionsReducer<MAX_RANK, MAX_FRESH_VARIABLES, MAX_MATRIX_ELEMENTS> *reducersU, AdditionsReducer<MAX_RANK, MAX_FRESH_VARIABLES, MAX_MATRIX_ELEMENTS> *reducersV, AdditionsReducer<MAX_MATRIX_ELEMENTS, MAX_FRESH_VARIABLES, MAX_RANK> *reducersW, curandState *states, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    curandState &state = states[idx];
    int modeU = curand(&state) % 5;
    int modeV = curand(&state) % 5;
    int modeW = curand(&state) % 5;

    reducersU[idx].copyFrom(reducersU[count]);
    reducersV[idx].copyFrom(reducersV[count]);
    reducersW[idx].copyFrom(reducersW[count]);

    reducersU[idx].reduce(modeU, state);
    reducersV[idx].reduce(modeV, state);
    reducersW[idx].reduce(modeW, state);
}
