#include "scheme_additions_reducer.cuh"

SchemeAdditionsReducer::SchemeAdditionsReducer(int count, int schemesCount, int maxFlips, int seed, int blockSize, const std::string &outputPath, int topCount) {
    this->count = count;
    this->schemesCount = schemesCount;
    this->maxFlips = maxFlips;
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

    CUDA_CHECK(cudaMallocManaged(&reducersU, (count + 2) * sizeof(AdditionsReducer<MAX_UV_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_REAL_UV_VARIABLES>)));
    CUDA_CHECK(cudaMallocManaged(&reducersV, (count + 2) * sizeof(AdditionsReducer<MAX_UV_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_REAL_UV_VARIABLES>)));
    CUDA_CHECK(cudaMallocManaged(&reducersW, (count + 2) * sizeof(AdditionsReducer<MAX_W_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_REAL_W_VARIABLES>)));
    CUDA_CHECK(cudaMallocManaged(&schemes, (count + 1) * sizeof(SchemeInteger)));
    CUDA_CHECK(cudaMallocManaged(&states, count * sizeof(curandState)));
}

bool SchemeAdditionsReducer::read(std::ifstream &f) {
    f >> n1 >> n2 >> n3 >> m;
    std::cout << "Read scheme " << n1 << "x" << n2 << "x" << n3 << " with " << m << " multiplications" << std::endl;

    if (m > MAX_UV_EXPRESSIONS || m > MAX_REAL_W_VARIABLES) {
        std::cout << "Error: multiplications number (" << m << ") too big for compiled configuration" << std::endl;
        return false;
    }

    if (n1 * n2 > MAX_REAL_UV_VARIABLES || n2 * n3 > MAX_REAL_UV_VARIABLES || n3 * n1 > MAX_W_EXPRESSIONS) {
        std::cout << "Error: dimensions (" << n1 << "x" << n2 << "x" << n3 << ") too big for compiled configuration" << std::endl;
        return false;
    }

    if (!schemes[0].read(f, n1, n2, n3, m)) {
        std::cout << "Error: readed scheme is invalid" << std::endl;
        return false;
    }

    return true;
}

void SchemeAdditionsReducer::reduce(int maxNoImprovements, int startAdditions) {
    initialize();

    auto startTime = std::chrono::high_resolution_clock::now();
    std::vector<double> elapsedTimes;

    int noImprovements = 0;

    for (int iteration = 1; noImprovements < maxNoImprovements; iteration++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        reduceIteration(iteration);
        bool improved = updateBest(startAdditions);
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

    if (schemes) {
        cudaFree(schemes);
        schemes = nullptr;
    }

    if (states) {
        cudaFree(states);
        states = nullptr;
    }
}

void SchemeAdditionsReducer::initialize() {
    initializeKernel<<<numBlocks, blockSize>>>(reducersU, reducersV, reducersW, schemes, states, count, schemesCount, seed);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bestAdditions[0] = reducersU[count].getAdditions();
    bestAdditions[1] = reducersV[count].getAdditions();
    bestAdditions[2] = reducersW[count].getAdditions();

    reducedAdditions = bestAdditions[0] + bestAdditions[1] + bestAdditions[2];

    std::cout << "Readed scheme uses " << bestAdditions[0] << " + " << bestAdditions[1] << " + " << bestAdditions[2] << " = " << reducedAdditions << " additions [naive]" << std::endl;
}

void SchemeAdditionsReducer::reduceIteration(int iteration) {
    if (maxFlips > 0 && iteration > 1) {
        flipSchemesKernel<<<((schemesCount + blockSize - 1) / blockSize), blockSize>>>(schemes, states, schemesCount, maxFlips);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    runReducersKernel<<<numBlocks, blockSize>>>(reducersU, reducersV, reducersW, schemes, states, count, schemesCount);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

bool SchemeAdditionsReducer::updateBest(int startAdditions) {
    if (maxFlips == 0) {
        std::partial_sort(indices[0].begin(), indices[0].begin() + topCount, indices[0].end(), [this](int index1, int index2) {
            int additions1 = reducersU[index1].getAdditions();
            int additions2 = reducersU[index2].getAdditions();

            if (additions1 != additions2)
                return additions1 < additions2;

            return reducersU[index1].getFreshVars() < reducersU[index2].getFreshVars();
        });

        std::partial_sort(indices[1].begin(), indices[1].begin() + topCount, indices[1].end(), [this](int index1, int index2) {
            int additions1 = reducersV[index1].getAdditions();
            int additions2 = reducersV[index2].getAdditions();

            if (additions1 != additions2)
                return additions1 < additions2;

            return reducersV[index1].getFreshVars() < reducersV[index2].getFreshVars();
        });

        std::partial_sort(indices[2].begin(), indices[2].begin() + topCount, indices[2].end(), [this](int index1, int index2) {
            int additions1 = reducersW[index1].getAdditions();
            int additions2 = reducersW[index2].getAdditions();

            if (additions1 != additions2)
                return additions1 < additions2;

            return reducersW[index1].getFreshVars() < reducersW[index2].getFreshVars();
        });

        int ua = reducersU[indices[0][0]].getAdditions();
        int va = reducersV[indices[1][0]].getAdditions();
        int wa = reducersW[indices[2][0]].getAdditions();

        if (ua < bestAdditions[0]) {
            bestAdditions[0] = ua;
            reducersU[count].copyFrom(reducersU[indices[0][0]]);
        }

        if (va < bestAdditions[1]) {
            bestAdditions[1] = va;
            reducersV[count].copyFrom(reducersV[indices[1][0]]);
        }

        if (wa < bestAdditions[2]) {
            bestAdditions[2] = wa;
            reducersW[count].copyFrom(reducersW[indices[2][0]]);
        }
    }
    else {
        std::partial_sort(indices[0].begin(), indices[0].begin() + topCount, indices[0].end(), [this](int index1, int index2) {
            int additions1 = reducersU[index1].getAdditions() + reducersV[index1].getAdditions() + reducersW[index1].getAdditions();
            int additions2 = reducersU[index2].getAdditions() + reducersV[index2].getAdditions() + reducersW[index2].getAdditions();
            if (additions1 != additions2)
                return additions1 < additions2;

            int fresh1 = reducersU[index1].getFreshVars() + reducersV[index1].getFreshVars() + reducersW[index1].getFreshVars();
            int fresh2 = reducersU[index2].getFreshVars() + reducersV[index2].getFreshVars() + reducersW[index2].getFreshVars();
            return fresh1 < fresh2;
        });

        int topIndex = indices[0][0];
        int ua = reducersU[topIndex].getAdditions();
        int va = reducersV[topIndex].getAdditions();
        int wa = reducersW[topIndex].getAdditions();

        if (ua + va + wa < bestAdditions[0] + bestAdditions[1] + bestAdditions[2]) {
            bestAdditions[0] = ua;
            bestAdditions[1] = va;
            bestAdditions[2] = wa;
            reducersU[count].copyFrom(reducersU[topIndex]);
            reducersV[count].copyFrom(reducersV[topIndex]);
            reducersW[count].copyFrom(reducersW[topIndex]);
        }
    }

    int best = bestAdditions[0] + bestAdditions[1] + bestAdditions[2];

    if (best >= reducedAdditions)
        return false;

    std::cout << "Best additions improved from " << reducedAdditions << " to " << best << std::endl;
    reducedAdditions = best;

    if (reducedAdditions < startAdditions || startAdditions == 0)
        save();

    return true;
}

void SchemeAdditionsReducer::report(std::chrono::high_resolution_clock::time_point startTime, int iteration, const std::vector<double> &elapsedTimes) {
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

    double lastTime = elapsedTimes[elapsedTimes.size() - 1];
    double minTime = *std::min_element(elapsedTimes.begin(), elapsedTimes.end());
    double maxTime = *std::max_element(elapsedTimes.begin(), elapsedTimes.end());
    double meanTime = std::accumulate(elapsedTimes.begin(), elapsedTimes.end(), 0.0) / elapsedTimes.size();

    std::string dimensions = getDimensions();

    std::cout << std::endl << std::left;
    std::cout << "+----------------------------------------------------------------------------------------------------------------------------+" << std::endl;
    std::cout << "| ";
    std::cout << "Size: " << std::setw(24) << dimensions << "   ";
    std::cout << "Reducers: " << std::setw(20) << count << "   ";
    std::cout << "                                 ";
    std::cout << "Iteration: " << std::setw(12) << iteration;
    std::cout << " |" << std::endl;

    std::cout << "| ";
    std::cout << "Rank: " << std::setw(24) << m << "   ";
    std::cout << "Schemes: " << std::setw(21) << schemesCount << "   ";
    std::cout << "                                 ";
    std::cout << "Elapsed: " << std::setw(14) << prettyTime(elapsed);
    std::cout << " |" << std::endl;

    std::cout << std::right;
    std::cout << "+================================+================================+================================+=========================+" << std::endl;
    std::cout << "|           Reducers U           |           Reducers V           |           Reducers W           |          Total          |" << std::endl;
    std::cout << "+------+-------+---------+-------+------+-------+---------+-------+------+-------+---------+-------+-------+---------+-------|" << std::endl;
    std::cout << "| mode | naive | reduced | fresh | mode | naive | reduced | fresh | mode | naive | reduced | fresh | naive | reduced | fresh |" << std::endl;
    std::cout << "+------+-------+---------+-------+------+-------+---------+-------+------+-------+---------+-------+-------+---------+-------+" << std::endl;

    for (int i = 0; i < topCount && i < count; i++) {
        int ui = indices[0][i];
        int vi = indices[maxFlips > 0 ? 0 : 1][i];
        int wi = indices[maxFlips > 0 ? 0 : 2][i];

        std::string modeU = reducersU[ui].getMode();
        std::string modeV = reducersV[vi].getMode();
        std::string modeW = reducersW[wi].getMode();

        int naiveU = reducersU[ui].getNaiveAdditions();
        int naiveV = reducersV[vi].getNaiveAdditions();
        int naiveW = reducersW[wi].getNaiveAdditions();
        int naive = naiveU + naiveV + naiveW;

        int reducedU = reducersU[ui].getAdditions();
        int reducedV = reducersV[vi].getAdditions();
        int reducedW = reducersW[wi].getAdditions();
        int reduced = reducedU + reducedV + reducedW;

        int freshU = reducersU[ui].getFreshVars();
        int freshV = reducersV[vi].getFreshVars();
        int freshW = reducersW[wi].getFreshVars();
        int fresh = freshU + freshV + freshW;

        std::cout << "| ";
        std::cout << std::setw(4) << modeU << "   " << std::setw(5) << naiveU << "   " << std::setw(7) << reducedU << "   " << std::setw(5) << freshU << " | ";
        std::cout << std::setw(4) << modeV << "   " << std::setw(5) << naiveV << "   " << std::setw(7) << reducedV << "   " << std::setw(5) << freshV << " | ";
        std::cout << std::setw(4) << modeW << "   " << std::setw(5) << naiveW << "   " << std::setw(7) << reducedW << "   " << std::setw(5) << freshW << " | ";
        std::cout << std::setw(5) << naive << "   " << std::setw(7) << reduced << "   " << std::setw(5) << fresh << " | ";
        std::cout << std::endl;
    }

    std::cout << "+--------------------------------+--------------------------------+--------------------------------+-------------------------+" << std::endl;
    std::cout << "- iteration time (last / min / max / mean): " << prettyTime(lastTime) << " / " << prettyTime(minTime) << " / " << prettyTime(maxTime) << " / " << prettyTime(meanTime) << std::endl;
    std::cout << "- best additions (U / V / W / total): " << bestAdditions[0] << " / " << bestAdditions[1] << " / " << bestAdditions[2] << " / " << reducedAdditions << std::endl;
    std::cout << std::endl;
}

void SchemeAdditionsReducer::save() const {
    std::string path = getSavePath();
    int naiveAdditions = reducersU[count].getNaiveAdditions() + reducersV[count].getNaiveAdditions() + reducersW[count].getNaiveAdditions();

    std::ofstream f(path);

    f << "{" << std::endl;
    f << "    \"n\": [" << n1 << ", " << n2 << ", " << n3 << "]," << std::endl;
    f << "    \"m\": " << m << "," << std::endl;
    f << "    \"z2\": false," << std::endl;
    f << "    \"complexity\": {\"naive\": " << naiveAdditions << ", \"reduced\": " << reducedAdditions << "}," << std::endl;
    reducersU[count].write(f, "u", "    ");
    f << "," << std::endl;
    reducersV[count].write(f, "v", "    ");
    f << "," << std::endl;
    reducersW[count].write(f, "w", "    ");
    f << std::endl;
    f << "}" << std::endl;
    f.close();

    std::cout << "Reduced scheme saved to \"" << path << "\"" << std::endl;
}

std::string SchemeAdditionsReducer::getSavePath() const {
    std::stringstream ss;
    ss << outputPath << "/";
    ss << n1 << "x" << n2 << "x" << n3;
    ss << "_m" << m;
    ss << "_cr" << reducedAdditions;
    ss << "_cn" << (reducersU[count].getNaiveAdditions() + reducersV[count].getNaiveAdditions() + reducersW[count].getNaiveAdditions());
    ss << "_" << ring;
    ss << "_reduced.json";

    return ss.str();
}

std::string SchemeAdditionsReducer::getDimensions() const {
    std::stringstream ss;
    ss << n1 << "x" << n2 << "x" << n3;
    return ss.str();
}

__global__ void initializeKernel(AdditionsReducer<MAX_UV_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_REAL_UV_VARIABLES> *reducersU, AdditionsReducer<MAX_UV_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_REAL_UV_VARIABLES> *reducersV, AdditionsReducer<MAX_W_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_REAL_W_VARIABLES> *reducersW, SchemeInteger *schemes, curandState *states, int count, int schemesCount, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    if (idx == 0) {
        copySchemeToReducers(reducersU, reducersV, reducersW, count, schemes[0]);
    }
    else if (idx < schemesCount) {
        schemes[0].copyTo(schemes[idx]);
    }

    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void flipSchemesKernel(SchemeInteger *schemes, curandState *states, int schemesCount, int maxFlips) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0 || idx >= schemesCount)
        return;

    curandState &state = states[idx];
    int flips = curand(&state) % (maxFlips + 1);

    schemes[0].copyTo(schemes[idx]);

    for (int i = 0; i < flips; i++)
        if (!schemes[idx].tryFlip(state))
            break;
}

__device__ void copySchemeToReducers(AdditionsReducer<MAX_UV_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_REAL_UV_VARIABLES> *reducersU, AdditionsReducer<MAX_UV_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_REAL_UV_VARIABLES> *reducersV, AdditionsReducer<MAX_W_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_REAL_W_VARIABLES> *reducersW, int idx, const SchemeInteger &scheme) {
    int u[MAX_REAL_UV_VARIABLES];
    int v[MAX_REAL_UV_VARIABLES];
    int w[MAX_REAL_W_VARIABLES];

    reducersU[idx].clear();
    reducersV[idx].clear();
    reducersW[idx].clear();

    for (int index = 0; index < scheme.m; index++) {
        for (int i = 0; i < scheme.nn[0]; i++)
            u[i] = scheme.uvw[0][index][i];

        reducersU[idx].addExpression(u, scheme.nn[0]);
    }

    for (int index = 0; index < scheme.m; index++) {
        for (int i = 0; i < scheme.nn[1]; i++)
            v[i] = scheme.uvw[1][index][i];

        reducersV[idx].addExpression(v, scheme.nn[1]);
    }

    for (int i = 0; i < scheme.nn[2]; i++) {
        for (int index = 0; index < scheme.m; index++)
            w[index] = scheme.uvw[2][index][i];

        reducersW[idx].addExpression(w, scheme.m);
    }
}

__global__ void runReducersKernel(AdditionsReducer<MAX_UV_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_REAL_UV_VARIABLES> *reducersU, AdditionsReducer<MAX_UV_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_REAL_UV_VARIABLES> *reducersV, AdditionsReducer<MAX_W_EXPRESSIONS, MAX_FRESH_VARIABLES, MAX_REAL_W_VARIABLES> *reducersW, SchemeInteger *schemes, curandState *states, int count, int schemesCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    copySchemeToReducers(reducersU, reducersV, reducersW, idx, schemes[idx % schemesCount]);

    curandState &state = states[idx];

    reducersU[idx].setMode(idx == 0 ? 0 : 1 + curand(&state) % MIX_MODE);
    reducersV[idx].setMode(idx == 0 ? 0 : 1 + curand(&state) % MIX_MODE);
    reducersW[idx].setMode(idx == 0 ? 0 : 1 + curand(&state) % MIX_MODE);

    reducersU[idx].reduce(state);
    reducersV[idx].reduce(state);
    reducersW[idx].reduce(state);
}
