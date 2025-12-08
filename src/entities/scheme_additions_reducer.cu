#include "scheme_additions_reducer.cuh"

SchemeAdditionsReducer::SchemeAdditionsReducer(int count, int schemesCount, int maxFlips, int seed, int blockSize, const std::string &outputPath, int topCount) {
    this->count = count;
    this->schemesCount = schemesCount;
    this->maxFlips = maxFlips;
    this->seed = seed;
    this->blockSize = blockSize;
    this->numBlocks = (count + blockSize - 1) / blockSize;
    this->outputPath = outputPath;
    this->topCount = std::min(topCount, count);

    for (int i = 0; i < 3; i++) {
        this->indices[i].reserve(count);
        this->bestAdditions[i] = 0;

        for (int j = 0; j < count; j++)
            this->indices[i].push_back(j);
    }

    std::cout << "Start memory allocating" << std::endl;
    CUDA_CHECK(cudaMallocManaged(&reducersU, (count + 1) * sizeof(AdditionsReducer<MAX_U_EXPRESSIONS, MAX_U_FRESH_VARIABLES, MAX_U_REAL_VARIABLES, MAX_U_SUBEXPRESSIONS>)));
    CUDA_CHECK(cudaMallocManaged(&reducersV, (count + 1) * sizeof(AdditionsReducer<MAX_V_EXPRESSIONS, MAX_V_FRESH_VARIABLES, MAX_V_REAL_VARIABLES, MAX_V_SUBEXPRESSIONS>)));
    CUDA_CHECK(cudaMallocManaged(&reducersW, (count + 1) * sizeof(AdditionsReducer<MAX_W_EXPRESSIONS, MAX_W_FRESH_VARIABLES, MAX_W_REAL_VARIABLES, MAX_W_SUBEXPRESSIONS>)));
    CUDA_CHECK(cudaMallocManaged(&schemes, (count + 1) * sizeof(SchemeInteger)));
    CUDA_CHECK(cudaMallocManaged(&states, count * sizeof(curandState)));
}

bool SchemeAdditionsReducer::read(std::ifstream &f) {
    f >> n1 >> n2 >> n3 >> m;
    std::cout << "Start reading scheme " << n1 << "x" << n2 << "x" << n3 << " with " << m << " multiplications" << std::endl;

    if (m > MAX_U_EXPRESSIONS) {
        std::cout << "Error: expressions count for U ("<< m << ") too big for compiled configuration (" << MAX_U_EXPRESSIONS << ")" << std::endl;
        return false;
    }

    if (m > MAX_V_EXPRESSIONS) {
        std::cout << "Error: expressions count for V ("<< m << ") too big for compiled configuration (" << MAX_V_EXPRESSIONS << ")" << std::endl;
        return false;
    }

    if (n3 * n1 > MAX_W_EXPRESSIONS) {
        std::cout << "Error: expressions count for W ("<< (n3 * n1) << ") too big for compiled configuration (" << MAX_W_EXPRESSIONS << ")" << std::endl;
        return false;
    }

    if (!schemes[0].read(f, n1, n2, n3, m)) {
        std::cout << "Error: readed scheme is invalid" << std::endl;
        return false;
    }

    int realU = schemes[0].getMaxRealVariables(0);
    if (realU > MAX_U_REAL_VARIABLES) {
        std::cout << "Error: real variables for U (" << realU << ") too big for compiled configuration (" << MAX_U_REAL_VARIABLES << ")" << std::endl;
        return false;
    }

    int realV = schemes[0].getMaxRealVariables(1);
    if (realV > MAX_V_REAL_VARIABLES) {
        std::cout << "Error: real variables for V (" << realV << ") too big for compiled configuration (" << MAX_V_REAL_VARIABLES << ")" << std::endl;
        return false;
    }

    int realW = schemes[0].getMaxRealVariables(2);
    if (realU > MAX_W_REAL_VARIABLES) {
        std::cout << "Error: real variables for W (" << realW << ") too big for compiled configuration (" << MAX_W_REAL_VARIABLES << ")" << std::endl;
        return false;
    }

    int subexpressionsU = schemes[0].getMaxSubexpressions(0);
    if (subexpressionsU > MAX_U_SUBEXPRESSIONS) {
        std::cout << "Error: max subexpressions for U (" << subexpressionsU << ") too big for compiled configuration (" << MAX_U_SUBEXPRESSIONS << ")" << std::endl;
        return false;
    }

    int subexpressionsV = schemes[0].getMaxSubexpressions(1);
    if (subexpressionsV > MAX_V_SUBEXPRESSIONS) {
        std::cout << "Error: max subexpressions for V (" << subexpressionsV << ") too big for compiled configuration (" << MAX_V_SUBEXPRESSIONS << ")" << std::endl;
        return false;
    }

    int subexpressionsW = schemes[0].getMaxSubexpressions(2);
    if (subexpressionsW > MAX_W_SUBEXPRESSIONS) {
        std::cout << "Error: max subexpressions for W (" << subexpressionsW << ") too big for compiled configuration (" << MAX_W_SUBEXPRESSIONS << ")" << std::endl;
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

    bestFreshVars[0] = n1 * n2 * (n1 * n2 - 1);
    bestFreshVars[1] = n2 * n3 * (n2 * n3 - 1);
    bestFreshVars[2] = m * (m - 1);

    reducedAdditions = bestAdditions[0] + bestAdditions[1] + bestAdditions[2];
    reducedFreshVars = bestFreshVars[0] + bestFreshVars[1] + bestFreshVars[2];

    std::cout << "Readed scheme params:" << std::endl;
    std::cout << "- dimensions: " << n1 << "x" << n2 << "x" << n3 << std::endl;
    std::cout << "- multiplications (rank): " << m << std::endl;
    std::cout << "- naive additions (U / V / W / total): " << bestAdditions[0] << " / " << bestAdditions[1] << " / " << bestAdditions[2] << " / " << reducedAdditions << std::endl;
    std::cout << "- max real variables (U / V / W): " << reducersU[count].getMaxRealVariables() << " / " << reducersV[count].getMaxRealVariables() << " / " << reducersW[count].getMaxRealVariables() << std::endl;
    std::cout << "- max unique subexpressions (U / V / W): " << schemes[0].getMaxSubexpressions(0) << " / " << schemes[0].getMaxSubexpressions(1) << " / " << schemes[0].getMaxSubexpressions(2) << std::endl;
    std::cout << std::endl;
}

void SchemeAdditionsReducer::reduceIteration(int iteration) {
    if (maxFlips > 0 && iteration > 1) {
        flipSchemesKernel<<<((schemesCount + blockSize - 1) / blockSize), blockSize>>>(schemes, states, schemesCount, maxFlips);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    runReducersKernel<<<numBlocks, blockSize>>>(reducersU, reducersV, reducersW, schemes, states, count, schemesCount, maxFlips == 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

bool SchemeAdditionsReducer::updateBestIndependent() {
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

    int ui = indices[0][0];
    int vi = indices[1][0];
    int wi = indices[2][0];

    int ua = reducersU[ui].getAdditions();
    int va = reducersV[vi].getAdditions();
    int wa = reducersW[wi].getAdditions();

    int uf = reducersU[ui].getFreshVars();
    int vf = reducersV[vi].getFreshVars();
    int wf = reducersW[wi].getFreshVars();

    bool updated = false;

    if (ua < bestAdditions[0] || (ua == bestAdditions[0] && uf < bestFreshVars[0])) {
        bestAdditions[0] = ua;
        bestFreshVars[0] = uf;
        reducersU[count].copyFrom(reducersU[ui]);
        updated = true;
    }

    if (va < bestAdditions[1] || (va == bestAdditions[1] && vf < bestFreshVars[1])) {
        bestAdditions[1] = va;
        bestFreshVars[1] = vf;
        reducersV[count].copyFrom(reducersV[vi]);
        updated = true;
    }

    if (wa < bestAdditions[2] || (wa == bestAdditions[2] && wf < bestFreshVars[2])) {
        bestAdditions[2] = wa;
        bestFreshVars[2] = wf;
        reducersW[count].copyFrom(reducersW[wi]);
        updated = true;
    }

    return updated;
}

bool SchemeAdditionsReducer::updateBestTogether() {
    std::partial_sort(indices[0].begin(), indices[0].begin() + topCount, indices[0].end(), [this](int index1, int index2) {
        int additions1 = reducersU[index1].getAdditions() + reducersV[index1].getAdditions() + reducersW[index1].getAdditions();
        int additions2 = reducersU[index2].getAdditions() + reducersV[index2].getAdditions() + reducersW[index2].getAdditions();

        if (additions1 != additions2)
            return additions1 < additions2;

        int freshVars1 = reducersU[index1].getFreshVars() + reducersV[index1].getFreshVars() + reducersW[index1].getFreshVars();
        int freshVars2 = reducersU[index2].getFreshVars() + reducersV[index2].getFreshVars() + reducersW[index2].getFreshVars();
        return freshVars1 < freshVars2;
    });

    int topIndex = indices[0][0];

    int ua = reducersU[topIndex].getAdditions();
    int va = reducersV[topIndex].getAdditions();
    int wa = reducersW[topIndex].getAdditions();
    int additions = ua + va + wa;

    int uf = reducersU[topIndex].getFreshVars();
    int vf = reducersV[topIndex].getFreshVars();
    int wf = reducersW[topIndex].getFreshVars();
    int freshVars = uf + vf + wf;

    if (additions < reducedAdditions || (additions == reducedAdditions && freshVars < reducedFreshVars)) {
        bestAdditions[0] = ua;
        bestAdditions[1] = va;
        bestAdditions[2] = wa;

        bestFreshVars[0] = uf;
        bestFreshVars[1] = vf;
        bestFreshVars[2] = wf;

        reducersU[count].copyFrom(reducersU[topIndex]);
        reducersV[count].copyFrom(reducersV[topIndex]);
        reducersW[count].copyFrom(reducersW[topIndex]);
        return true;
    }

    return false;
}

bool SchemeAdditionsReducer::updateBest(int startAdditions) {
    bool updated = maxFlips == 0 ? updateBestIndependent() : updateBestTogether();
    int additions = bestAdditions[0] + bestAdditions[1] + bestAdditions[2];
    int freshVars = bestFreshVars[0] + bestFreshVars[1] + bestFreshVars[2];

    if (!updated)
        return false;

    if (additions < reducedAdditions)
        std::cout << "Reduced scheme improved from " << reducedAdditions << " to " << additions << " additions (fresh vars: " << freshVars << ")" << std::endl;
    else
        std::cout << "Reduced scheme improved from " << reducedFreshVars << " fresh vars to " << freshVars << " fresh vars (additions: " << reducedAdditions << ")" << std::endl;

    reducedAdditions = additions;
    reducedFreshVars = freshVars;

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
    std::cout << "- best fresh vars (U / V / W / total): " << bestFreshVars[0] << " / " << bestFreshVars[1] << " / " << bestFreshVars[2] << " / " << reducedFreshVars << std::endl;
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
    ss << "_fv" << reducedFreshVars;
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

__global__ void initializeKernel(AdditionsReducer<MAX_U_EXPRESSIONS, MAX_U_FRESH_VARIABLES, MAX_U_REAL_VARIABLES, MAX_U_SUBEXPRESSIONS> *reducersU, AdditionsReducer<MAX_V_EXPRESSIONS, MAX_V_FRESH_VARIABLES, MAX_V_REAL_VARIABLES, MAX_V_SUBEXPRESSIONS> *reducersV, AdditionsReducer<MAX_W_EXPRESSIONS, MAX_W_FRESH_VARIABLES, MAX_W_REAL_VARIABLES, MAX_W_SUBEXPRESSIONS> *reducersW, SchemeInteger *schemes, curandState *states, int count, int schemesCount, int seed) {
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

__device__ void copySchemeToReducers(AdditionsReducer<MAX_U_EXPRESSIONS, MAX_U_FRESH_VARIABLES, MAX_U_REAL_VARIABLES, MAX_U_SUBEXPRESSIONS> *reducersU, AdditionsReducer<MAX_V_EXPRESSIONS, MAX_V_FRESH_VARIABLES, MAX_V_REAL_VARIABLES, MAX_V_SUBEXPRESSIONS> *reducersV, AdditionsReducer<MAX_W_EXPRESSIONS, MAX_W_FRESH_VARIABLES, MAX_W_REAL_VARIABLES, MAX_W_SUBEXPRESSIONS> *reducersW, int idx, const SchemeInteger &scheme) {
    int u[MAX_MATRIX_ELEMENTS];
    int v[MAX_MATRIX_ELEMENTS];
    int w[MAX_RANK];

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

__global__ void runReducersKernel(AdditionsReducer<MAX_U_EXPRESSIONS, MAX_U_FRESH_VARIABLES, MAX_U_REAL_VARIABLES, MAX_U_SUBEXPRESSIONS> *reducersU, AdditionsReducer<MAX_V_EXPRESSIONS, MAX_V_FRESH_VARIABLES, MAX_V_REAL_VARIABLES, MAX_V_SUBEXPRESSIONS> *reducersV, AdditionsReducer<MAX_W_EXPRESSIONS, MAX_W_FRESH_VARIABLES, MAX_W_REAL_VARIABLES, MAX_W_SUBEXPRESSIONS> *reducersW, SchemeInteger *schemes, curandState *states, int count, int schemesCount, bool independent) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    copySchemeToReducers(reducersU, reducersV, reducersW, idx, schemes[idx % schemesCount]);

    curandState &state = states[idx];

    int modes[] = {
        GREEDY_INTERSECTIONS_MODE, GREEDY_INTERSECTIONS_MODE, GREEDY_INTERSECTIONS_MODE, GREEDY_INTERSECTIONS_MODE, GREEDY_INTERSECTIONS_MODE, GREEDY_INTERSECTIONS_MODE, GREEDY_INTERSECTIONS_MODE, GREEDY_INTERSECTIONS_MODE,
        GREEDY_ALTERNATIVE_MODE, GREEDY_ALTERNATIVE_MODE, GREEDY_ALTERNATIVE_MODE, GREEDY_ALTERNATIVE_MODE,
        GREEDY_RANDOM_MODE, GREEDY_RANDOM_MODE,
        MIX_MODE
    };

    int modesCount = sizeof(modes) / sizeof(modes[0]);

    reducersU[idx].setMode(idx == 0 ? 0 : modes[curand(&state) % modesCount]);
    reducersV[idx].setMode(idx == 0 ? 0 : modes[curand(&state) % modesCount]);
    reducersW[idx].setMode(idx == 0 ? 0 : modes[curand(&state) % modesCount]);

    if (independent) {
        if (curand_uniform(&state) < 0.3 && reducersU[count].getFreshVars() > 0)
            reducersU[idx].partialInitialize(reducersU[count], 1 + curand(&state) % reducersU[count].getFreshVars());

        if (curand_uniform(&state) < 0.3 && reducersV[count].getFreshVars() > 0)
            reducersV[idx].partialInitialize(reducersV[count], 1 + curand(&state) % reducersV[count].getFreshVars());

        if (curand_uniform(&state) < 0.3 && reducersW[count].getFreshVars() > 0)
            reducersW[idx].partialInitialize(reducersW[count], 1 + curand(&state) % reducersW[count].getFreshVars());
    }
    else if (curand_uniform(&state) < 0.3) {
        if (reducersU[count].getFreshVars() > 0)
            reducersU[idx].partialInitialize(reducersU[count], 1 + curand(&state) % reducersU[count].getFreshVars());

        if (reducersV[count].getFreshVars() > 0)
            reducersV[idx].partialInitialize(reducersV[count], 1 + curand(&state) % reducersV[count].getFreshVars());

        if (reducersW[count].getFreshVars() > 0)
            reducersW[idx].partialInitialize(reducersW[count], 1 + curand(&state) % reducersW[count].getFreshVars());
    }

    reducersU[idx].reduce(state);
    reducersV[idx].reduce(state);
    reducersW[idx].reduce(state);
}
