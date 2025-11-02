#include "scheme.cuh"

__device__ __host__ bool validateEquation(const Scheme &scheme, int i, int j, int k) {
    int i1 = i / scheme.n;
    int i2 = i % scheme.n;
    int j1 = j / scheme.n;
    int j2 = j % scheme.n;
    int k1 = k / scheme.n;
    int k2 = k % scheme.n;

    bool target = (i2 == j1) && (i1 == k2) && (j2 == k1);
    bool equation = false;

    for (int index = 0; index < scheme.m; index++)
        equation ^= ((scheme.uvw[0][index] >> i) & 1) && ((scheme.uvw[1][index] >> j) & 1) && ((scheme.uvw[2][index] >> k) & 1);

    return equation == target;
}

__device__ __host__ bool validateScheme(const Scheme &scheme) {
    bool valid = true;

    for (int i = 0; i < scheme.nn && valid; i++)
        for (int j = 0; j < scheme.nn && valid; j++)
            for (int k = 0; k < scheme.nn && valid; k++)
                valid &= validateEquation(scheme, i, j, k);

    return valid;
}

/*************************************************** device functions ****************************************************/
__device__ void initializeNaive(Scheme &scheme, int n) {
    scheme.n = n;
    scheme.nn = n * n;
    scheme.m = n * n * n;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                int index = (i * n + j) * n + k;
                scheme.uvw[0][index] |= 1 << (i * n + k);
                scheme.uvw[1][index] |= 1 << (k * n + j);
                scheme.uvw[2][index] |= 1 << (j * n + i);
            }
        }
    }
}

__device__ void copyScheme(const Scheme &scheme, Scheme &target) {
    target.n = scheme.n;
    target.m = scheme.m;
    target.nn = scheme.nn;

    for (int i = 0; i < 3; i++)
        for (int index = 0; index < scheme.m; index++)
            target.uvw[i][index] = scheme.uvw[i][index];
}

__device__ void removeZeroes(Scheme &scheme) {
    int m = 0;

    for (int index = 0; index < scheme.m; index++) {
        if (scheme.uvw[0][index] && scheme.uvw[1][index] && scheme.uvw[2][index]) {
            scheme.uvw[0][m] = scheme.uvw[0][index];
            scheme.uvw[1][m] = scheme.uvw[1][index];
            scheme.uvw[2][m] = scheme.uvw[2][index];
            m++;
        }
    }

    scheme.m = m;
}

__device__ void removeAt(Scheme &scheme, int index) {
    scheme.m--;

    if (index == scheme.m)
        return;

    scheme.uvw[0][index] = scheme.uvw[0][scheme.m];
    scheme.uvw[1][index] = scheme.uvw[1][scheme.m];
    scheme.uvw[2][index] = scheme.uvw[2][scheme.m];
}

__device__ void addTriplet(Scheme &scheme, int i, int j, int k, const T u, const T v, const T w) {
    scheme.uvw[i][scheme.m] = u;
    scheme.uvw[j][scheme.m] = v;
    scheme.uvw[k][scheme.m] = w;
    scheme.m++;
}

/******************************************************** helpers ********************************************************/
__device__ FlipCandidate getFlipCandidate(const Scheme &scheme, curandState &state) {
    int permutation[3];
    int indices[MAX_RANK];

    randomPermutation(permutation, 3, state);
    randomPermutation(indices, scheme.m, state);

    int variant = curand(&state) % 3;

    if (variant == 0) {
        #pragma unroll
        for (int i = 0; i < 3; i++)
            #pragma unroll
            for (int index1 = 0; index1 < scheme.m; index1++)
                #pragma unroll
                for (int index2 = index1 + 1; index2 < scheme.m; index2++)
                    if (scheme.uvw[permutation[i]][indices[index1]] == scheme.uvw[permutation[i]][indices[index2]])
                        return {(permutation[i] + 1) % 3, (permutation[i] + 2) % 3, indices[index1], indices[index2]};
    }
    else if (variant == 1) {
        #pragma unroll
        for (int index1 = 0; index1 < scheme.m; index1++)
            #pragma unroll
            for (int i = 0; i < 3; i++)
                #pragma unroll
                for (int index2 = index1 + 1; index2 < scheme.m; index2++)
                    if (scheme.uvw[permutation[i]][indices[index1]] == scheme.uvw[permutation[i]][indices[index2]])
                        return {(permutation[i] + 1) % 3, (permutation[i] + 2) % 3, indices[index1], indices[index2]};
    }
    else {
        int i = permutation[0];
        int j = permutation[1];
        int k = permutation[2];

        #pragma unroll
        for (int index1 = 0; index1 < scheme.m; index1++) {
            const T u1 = scheme.uvw[i][indices[index1]];
            const T v1 = scheme.uvw[j][indices[index1]];
            const T w1 = scheme.uvw[k][indices[index1]];

            #pragma unroll
            for (int index2 = index1 + 1; index2 < scheme.m; index2++) {
                if (u1 == scheme.uvw[i][indices[index2]])
                    return {j, k, indices[index1], indices[index2]};

                if (v1 == scheme.uvw[j][indices[index2]])
                    return {i, k, indices[index1], indices[index2]};

                if (w1 == scheme.uvw[k][indices[index2]])
                    return {i, j, indices[index1], indices[index2]};
            }
        }
    }

    return {-1, 0, 0, 0};
}

__device__ ReduceCandidate getReduceCandidate(const Scheme &scheme, curandState &state) {
    int permutation[3];
    int indices[MAX_RANK];

    randomPermutation(permutation, 3, state);
    randomPermutation(indices, scheme.m, state);

    #pragma unroll
    for (int index1 = 0; index1 < scheme.m; index1++) {
        #pragma unroll
        for (int index2 = index1 + 1; index2 < scheme.m; index2++) {
            const T u1 = scheme.uvw[permutation[0]][indices[index1]];
            const T u2 = scheme.uvw[permutation[0]][indices[index2]];

            const T v1 = scheme.uvw[permutation[1]][indices[index1]];
            const T v2 = scheme.uvw[permutation[1]][indices[index2]];

            const T w1 = scheme.uvw[permutation[2]][indices[index1]];
            const T w2 = scheme.uvw[permutation[2]][indices[index2]];

            if (u1 == u2) {
                if (w1 == w2)
                    return {permutation[1], indices[index1], indices[index2]};

                if (v1 == v2)
                    return {permutation[2], indices[index1], indices[index2]};
            }
            else if (v1 == v2 && w1 == w2) {
                return {permutation[0], indices[index1], indices[index2]};
            }
        }
    }

    return {-1, 0, 0};
}

__device__ ReduceGaussCandidate getReduceGaussCandidate(const Scheme &scheme, curandState &state) {
    int permutation[3];
    randomPermutation(permutation, 3, state);

    int indices[MAX_RANK];
    ReduceGaussCandidate possibleReduce;

    #pragma unroll
    for (int index = 0; index < scheme.m; index++)
        indices[index] = index;

    #pragma unroll
    for (int i = 0; i < 3; i++) {
        shellSort(indices, scheme.uvw[permutation[i]], scheme.m);

        int start = 0;

        #pragma unroll
        for (int index = start + 1; index <= scheme.m; index++) {
            if (index != scheme.m && scheme.uvw[permutation[i]][indices[index]] == scheme.uvw[permutation[i]][indices[start]])
                continue;

            // end of duplicate from indices[start] ... indices[index - 1]
            possibleReduce.i = (permutation[i] + 2) % 3;
            possibleReduce.size = findXorCombination(scheme, (permutation[i] + 1) % 3, indices + start, index - start, possibleReduce.combination);
            if (possibleReduce.size > 0)
                return possibleReduce;

            possibleReduce.i = (permutation[i] + 1) % 3;
            possibleReduce.size = findXorCombination(scheme, (permutation[i] + 2) % 3, indices + start, index - start, possibleReduce.combination);
            if (possibleReduce.size > 0)
                return possibleReduce;

            start = index;
        }
    }

    possibleReduce.i = -1;
    return possibleReduce;
}

__device__ int findXorCombination(const Scheme &scheme, int uvwIndex, int *indices, int size, int *combination) {
    if (size < 3)
        return 0;

    #pragma unroll
    for (int index = 0; index < size; index++) {
        const T target = scheme.uvw[uvwIndex][indices[index]];

        #pragma unroll
        for (int index1 = 0; index1 < size; index1++) {
            if (index1 == index)
                continue;

            const T v1 = scheme.uvw[uvwIndex][indices[index1]];

            #pragma unroll
            for (int index2 = index1 + 1; index2 < size; index2++) {
                if (index2 == index)
                    continue;

                const T v2 = scheme.uvw[uvwIndex][indices[index2]];

                #pragma unroll
                for (int index3 = index2 + 1; index3 < size; index3++) {
                    if (index3 == index)
                        continue;

                    const T v3 = scheme.uvw[uvwIndex][indices[index3]];

                    #pragma unroll
                    for (int index4 = index3 + 1; index4 < size; index4++) {
                        if (index4 == index)
                            continue;

                        const T v4 = scheme.uvw[uvwIndex][indices[index4]];

                        if ((v1 ^ v2 ^ v3 ^ v4) == target) {
                            combination[0] = indices[index1];
                            combination[1] = indices[index2];
                            combination[2] = indices[index3];
                            combination[3] = indices[index4];
                            combination[4] = indices[index];
                            return 5;
                        } 
                    }

                    if ((v1 ^ v2 ^ v3) == target) {
                        combination[0] = indices[index1];
                        combination[1] = indices[index2];
                        combination[2] = indices[index3];
                        combination[3] = indices[index];
                        return 4;
                    }  
                }

                if ((v1 ^ v2) == target) {
                    combination[0] = indices[index1];
                    combination[1] = indices[index2];
                    combination[2] = indices[index];
                    return 3;
                }
            }
        }
    }

    return 0;
}

__device__ void shellSort(int *indices, const T *values, int n) {
    int gaps[] = {701, 301, 132, 57, 23, 10, 4, 1};

    for (int g = 0; g < 8; g++) {
        int gap = gaps[g];

        if (gap > n)
            continue;

        for (int i = gap; i < n; i++) {
            int tmp = indices[i];
            int j = i;

            while (j >= gap && values[indices[j - gap]] > values[tmp]) {
                indices[j] = indices[j - gap];
                j -= gap;
            }

            indices[j] = tmp;
        }
    }
}

__device__ bool inverseMatrixZ2(int n, int *matrix, int *inverse) {
    int augmented[2 * MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
    int n2 = n * 2;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            augmented[i * n2 + j] = matrix[i * n + j];
            augmented[i * n2 + n + j] = i == j;
        }
    }

    for (int col = 0; col < n; col++) {
        int pivot_row = -1;
        for (int row = col; row < n; row++) {
            if (augmented[row * n2 + col]) {
                pivot_row = row;
                break;
            }
        }

        if (pivot_row == -1)
            return false;

        if (pivot_row != col) {
            for (int i = 0; i < 2 * n; i++) {
                int tmp = augmented[col * n2 + i];
                augmented[col * n2 + i] = augmented[pivot_row * n2 + i];
                augmented[pivot_row * n2 + i] = tmp;
            }
        }

        for (int row = 0; row < n; row++)
            if (row != col && augmented[row * n2 + col])
                for (int j = col; j < n2; j++)
                    augmented[row * n2 + j] = augmented[row * n2 + j] ^ augmented[col * n2 + j];
    }

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            inverse[i * n + j] = augmented[i * n2 + n + j];

    return true;
}

__device__ void invertibleMatrixZ2(int n, int *matrix, int *inverse, curandState &state) {
    do {
        randomMatrixZ2(n, matrix, state);
    } while (!inverseMatrixZ2(n, matrix, inverse));
}

__device__ T matmul(const T matrix, int *left, int *right, int n) {
    int result1[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
    int result2[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];

    #pragma unroll
    for (int i = 0; i < n * n; i++) {
        result1[i] = 0;
        result2[i] = 0;
    }

    #pragma unroll
    for (int i = 0; i < n; i++)
        #pragma unroll
        for (int j = 0; j < n; j++)
            #pragma unroll
            for (int k = 0; k < n; k++)
                result1[i * n + j] ^= left[i * n + k] && int((matrix >> (k * n + j)) & 1);

    #pragma unroll
    for (int i = 0; i < n; i++)
        #pragma unroll
        for (int j = 0; j < n; j++)
            #pragma unroll
            for (int k = 0; k < n; k++)
                result2[i * n + j] ^= result1[i * n + k] && right[k * n + j];

    T result = 0;

    #pragma unroll
    for (int i = 0; i < n * n; i++)
        result |= result2[i] << i;

    return result;
}

/******************************************************* operators *******************************************************/
__device__ void flip(Scheme &scheme, int first, int second, int index1, int index2) {
    scheme.uvw[first][index1] ^= scheme.uvw[first][index2];
    scheme.uvw[second][index2] ^= scheme.uvw[second][index1];

    if (!scheme.uvw[first][index1] || !scheme.uvw[second][index2])
        removeZeroes(scheme);
}

__device__ void plus(Scheme &scheme, int i, int j, int k, int index1, int index2, int variant) {
    const T a1 = scheme.uvw[i][index1];
    const T b1 = scheme.uvw[j][index1];
    const T c1 = scheme.uvw[k][index1];

    const T a2 = scheme.uvw[i][index2];
    const T b2 = scheme.uvw[j][index2];
    const T c2 = scheme.uvw[k][index2];

    const T a = a1 ^ a2;
    const T b = b1 ^ b2;
    const T c = c1 ^ c2;

    if (variant == 0) {
        scheme.uvw[j][index1] = b;
        scheme.uvw[i][index2] = a;
        addTriplet(scheme, i, j, k, a1, b2, c);
    }
    else if (variant == 1) {
        scheme.uvw[k][index1] = c;
        scheme.uvw[j][index2] = b;
        addTriplet(scheme, i, j, k, a, b1, c2);
    }
    else {
        scheme.uvw[i][index1] = a;
        scheme.uvw[k][index2] = c;
        addTriplet(scheme, i, j, k, a2, b, c1);
    }

    if (!a || !b || !c)
        removeZeroes(scheme);
}

__device__ void split(Scheme &scheme, int i, int j, int k, int index, const T a1) {
    T a2 = scheme.uvw[i][index] ^ a1;
    scheme.uvw[i][index] = a1;
    addTriplet(scheme, i, j, k, a2, scheme.uvw[j][index], scheme.uvw[k][index]);
}

__device__ void reduceGauss(Scheme &scheme, int i, int *combination, int combinationSize) {
    int last = combination[combinationSize - 1];

    #pragma unroll
    for (int index = 0; index < combinationSize - 1; index++)
        scheme.uvw[i][combination[index]] ^= scheme.uvw[i][last];

    removeAt(scheme, last);
}

__device__ void reduce(Scheme &scheme, int i, int index1, int index2) {
    scheme.uvw[i][index1] ^= scheme.uvw[i][index2];
    bool isZero = !scheme.uvw[i][index1];

    removeAt(scheme, index2);

    if (isZero)
        removeZeroes(scheme);
}

/*************************************************** random operators ****************************************************/
__device__ bool tryPlus(Scheme &scheme, curandState &state) {
    if (scheme.m >= MAX_RANK)
        return false;

    int index1 = curand(&state) % scheme.m;
    int index2 = curand(&state) % scheme.m;

    while (index1 == index2)
        index2 = curand(&state) % scheme.m;

    int permutation[3];
    randomPermutation(permutation, 3, state);
    int i = permutation[0];
    int j = permutation[1];
    int k = permutation[2];

    int variant = curand(&state) % 3;
    plus(scheme, i, j, k, index1, index2, variant);
    return true;
}

__device__ bool trySplit(Scheme &scheme, curandState &state) {
    if (scheme.m >= MAX_RANK)
        return false;

    int index = curand(&state) % scheme.m;

    int permutation[3];
    randomPermutation(permutation, 3, state);
    int i = permutation[0];
    int j = permutation[1];
    int k = permutation[2];

    T a1 = curand(&state) % (1 << scheme.nn);
    const T a2 = scheme.uvw[i][index];

    while (a1 == a2)
        a1 = curand(&state) % (1 << scheme.nn);

    split(scheme, i, j, k, index, a1);
    return true;
}

__device__ bool trySplitExisted(Scheme &scheme, curandState &state) {
    if (scheme.m >= MAX_RANK)
        return false;

    int permutation[3];
    randomPermutation(permutation, 3, state);
    int i = permutation[0];
    int j = permutation[1];
    int k = permutation[2];

    int index1 = curand(&state) % scheme.m;
    int index2 = curand(&state) % scheme.m;

    while (index1 == index2 || scheme.uvw[i][index1] == scheme.uvw[i][index2])
        index2 = curand(&state) % scheme.m;

    split(scheme, i, j, k, index1, scheme.uvw[i][index2]);
    return true;
}

__device__ bool tryFlip(Scheme &scheme, curandState &state) {
    FlipCandidate possibleFlip = getFlipCandidate(scheme, state);
    if (possibleFlip.first < 0)
        return false;

    flip(scheme, possibleFlip.first, possibleFlip.second, possibleFlip.index1, possibleFlip.index2);
    return true;
}

__device__ bool tryReduceGauss(Scheme &scheme, curandState &state) {
    ReduceGaussCandidate possibleReduce = getReduceGaussCandidate(scheme, state);
    if (possibleReduce.i == -1)
        return false;

    reduceGauss(scheme, possibleReduce.i, possibleReduce.combination, possibleReduce.size);
    return true;
}

__device__ bool tryReduce(Scheme &scheme, curandState &state) {
    ReduceCandidate possibleReduce = getReduceCandidate(scheme, state);
    if (possibleReduce.i == -1)
        return false;

    reduce(scheme, possibleReduce.i, possibleReduce.index1, possibleReduce.index2);
    return true;
}

__device__ void expand(Scheme &scheme, int count, curandState &state) {
    for (int i = 0; i < count && scheme.m < MAX_RANK; i++) {
        int v = curand(&state) % 3;

        if (v == 0) {
            tryPlus(scheme, state);
        }
        else if (v == 1) {
            trySplit(scheme, state);
        }
        else {
            trySplitExisted(scheme, state);
        }
    }
}

__device__ void sandwiching(Scheme &scheme, curandState &state) {
    int u[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
    int v[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
    int w[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];

    int u1[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
    int v1[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
    int w1[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];

    invertibleMatrixZ2(scheme.n, u, u1, state);
    invertibleMatrixZ2(scheme.n, v, v1, state);
    invertibleMatrixZ2(scheme.n, w, w1, state);

    for (int index = 0; index < scheme.m; index++) {
        scheme.uvw[0][index] = matmul(scheme.uvw[0][index], u, v1, scheme.n);
        scheme.uvw[1][index] = matmul(scheme.uvw[1][index], v, w1, scheme.n);
        scheme.uvw[2][index] = matmul(scheme.uvw[2][index], w, u1, scheme.n);
    }
}

/**************************************************** save *****************************************************/
void saveMatrix(std::ofstream &f, std::string name, int n, int m, const T *matrix) {
    f << "    \"" << name << "\": [" << std::endl;

    for (int index = 0; index < m; index++) {
        f << "        [";

        for (int i = 0; i < n * n; i++) {
            if (i > 0)
                f << ", ";

            f << ((matrix[index] >> i) & 1);
        }

        f << "]" << (index < m - 1 ? "," : "") << std::endl;
    }

    f << "    ]";
}

void saveScheme(const Scheme &scheme, const std::string &path) {
    std::ofstream f(path);

    f << "{" << std::endl;
    f << "    \"n\": " << scheme.n << "," << std::endl;
    f << "    \"m\": " << scheme.m << "," << std::endl;
    f << "    \"z2\": true," << std::endl;

    saveMatrix(f, "u", scheme.n, scheme.m, scheme.uvw[0]);
    f << "," << std::endl;
    saveMatrix(f, "v", scheme.n, scheme.m, scheme.uvw[1]);
    f << "," << std::endl;
    saveMatrix(f, "w", scheme.n, scheme.m, scheme.uvw[2]);
    f << std::endl;
    f << "}" << std::endl;

    f.close();
}
