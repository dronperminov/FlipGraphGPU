#include "scheme_z2.cuh"

__device__ __host__ bool SchemeZ2::validateEquation(int i, int j, int k) const {
    int i1 = i / n[1];
    int i2 = i % n[1];
    int j1 = j / n[2];
    int j2 = j % n[2];
    int k1 = k / n[0];
    int k2 = k % n[0];

    bool target = (i2 == j1) && (i1 == k2) && (j2 == k1);
    bool equation = false;

    for (int index = 0; index < m; index++)
        equation ^= ((uvw[0][index] >> i) & 1) && ((uvw[1][index] >> j) & 1) && ((uvw[2][index] >> k) & 1);

    return equation == target;
}

__device__ __host__ bool SchemeZ2::validate() const {
    bool valid = true;

    for (int i = 0; i < nn[0] && valid; i++)
        for (int j = 0; j < nn[1] && valid; j++)
            for (int k = 0; k < nn[2] && valid; k++)
                valid &= validateEquation(i, j, k);

    return valid;
}

/*************************************************** device functions ****************************************************/
__device__ __host__ void SchemeZ2::initializeNaive(int n1, int n2, int n3) {
    n[0] = n1;
    n[1] = n2;
    n[2] = n3;

    nn[0] = n1 * n2;
    nn[1] = n2 * n3;
    nn[2] = n3 * n1;

    m = n1 * n2 * n3;

    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n3; j++) {
            for (int k = 0; k < n2; k++) {
                int index = (i * n3 + j) * n2 + k;
                uvw[0][index] = T(1) << (i * n2 + k);
                uvw[1][index] = T(1) << (k * n3 + j);
                uvw[2][index] = T(1) << (j * n1 + i);
            }
        }
    }

    initFlips();
}

__device__ __host__ void SchemeZ2::copyTo(SchemeZ2 &target, bool withFlips) const {
    target.m = m;

    for (int i = 0; i < 3; i++) {
        target.n[i] = n[i];
        target.nn[i] = nn[i];

        for (int index = 0; index < m; index++)
            target.uvw[i][index] = uvw[i][index];
    }

    if (withFlips)
        target.initFlips();
}

__host__ bool SchemeZ2::read(std::istream &is, bool checkValidity) {
    is >> n[0] >> n[1] >> n[2] >> m;

    if (n[0] * n[1] > MAX_MATRIX_ELEMENTS || n[1] * n[2] > MAX_MATRIX_ELEMENTS || n[2] * n[0] > MAX_MATRIX_ELEMENTS || m > MAX_RANK) {
        printf("SchemeZ2::read(is): invalid scheme sizes (%d, %d, %d, %d)\n", n[0], n[1], n[2], m);
        return false;
    }

    int value;
    for (int i = 0; i < 3; i++) {
        nn[i] = n[i] * n[(i + 1) % 3];

        for (int index = 0; index < m; index++) {
            uvw[i][index] = 0;

            for (int j = 0; j < nn[i]; j++) {
                is >> value;
                uvw[i][index] |= T(abs(value) % 2) << j;
            }
        }
    }

    initFlips();
    return !checkValidity || validate();
}

__host__ bool SchemeZ2::read(std::istream &is, int n1, int n2, int n3, int m, bool checkValidity) {
    if (n1 * n2 > MAX_MATRIX_ELEMENTS || n2 * n3 > MAX_MATRIX_ELEMENTS || n3 * n1 > MAX_MATRIX_ELEMENTS || m > MAX_RANK) {
        printf("SchemeZ2::read(is, n1, n2, n3, m): invalid scheme sizes (%d, %d, %d, %d)\n", n1, n2, n3, m);
        return false;
    }

    this->n[0] = n1;
    this->n[1] = n2;
    this->n[2] = n3;
    this->m = m;

    int value;
    for (int i = 0; i < 3; i++) {
        nn[i] = n[i] * n[(i + 1) % 3];

        for (int index = 0; index < m; index++) {
            uvw[i][index] = 0;

            for (int j = 0; j < nn[i]; j++) {
                is >> value;
                uvw[i][index] |= T(abs(value) % 2) << j;
            }
        }
    }

    initFlips();
    return !checkValidity || validate();
}

__device__ __host__ int SchemeZ2::getComplexity() const {
    int count = 0;

    for (int index = 0; index < m; index++)
        for (int i = 0; i < 3; i++)
            #if defined(__CUDA_ARCH__)
                count += __popcll(uvw[i][index]);
            #else
                count += __builtin_popcountll(uvw[i][index]);
            #endif

    return count - 2 * m - nn[2];
}

__device__ __host__ void SchemeZ2::initFlips() {
    for (int i = 0; i < 3; i++) {
        flips[i].clear();

        for (int index1 = 0; index1 < m; index1++)
            for (int index2 = index1 + 1; index2 < m; index2++)
                if (uvw[i][index1] == uvw[i][index2])
                    flips[i].add(index1, index2);
    }
}

__device__ __host__ void SchemeZ2::removeZeroes() {
    while (m > 0 && !(uvw[0][m - 1] && uvw[1][m - 1] && uvw[2][m - 1]))
        m--;

    for (int index = 0; index < m; index++) {
        if (!(uvw[0][index] && uvw[1][index] && uvw[2][index])) {
            m--;
            uvw[0][index] = uvw[0][m];
            uvw[1][index] = uvw[1][m];
            uvw[2][index] = uvw[2][m];
        }
    }
}

__device__ __host__ void SchemeZ2::removeAt(int index) {
    m--;

    if (index == m)
        return;

    uvw[0][index] = uvw[0][m];
    uvw[1][index] = uvw[1][m];
    uvw[2][index] = uvw[2][m];
}

__device__ __host__ void SchemeZ2::addTriplet(int i, int j, int k, const T u, const T v, const T w) {
    uvw[i][m] = u;
    uvw[j][m] = v;
    uvw[k][m] = w;
    m++;
}

__device__ __host__ void SchemeZ2::excludeColumn(int matrix, int column) {
    int n1 = n[matrix];
    int n2 = n[(matrix + 1) % 3];
    int oldColumns[MAX_MATRIX_ELEMENTS];
    int size = 0;

    for (int j = 0; j < n2; j++)
        if (j != column)
            oldColumns[size++] = j;

    for (int index = 0; index < m; index++) {
        T value = 0;

        for (int i = 0; i < n1; i++)
            for (int j = 0; j < n2 - 1; j++)
                value |= T((uvw[matrix][index] >> (i * n2 + oldColumns[j])) & 1) << (i * (n2 - 1) + j);

        uvw[matrix][index] = value;
    }
}

__device__ __host__ void SchemeZ2::excludeRow(int matrix, int row) {
    int n1 = n[matrix];
    int n2 = n[(matrix + 1) % 3];
    int oldRows[MAX_MATRIX_ELEMENTS];
    int size = 0;

    for (int i = 0; i < n1; i++)
        if (i != row)
            oldRows[size++] = i;

    for (int index = 0; index < m; index++) {
        T value = 0;

        for (int i = 0; i < n1 - 1; i++)
            for (int j = 0; j < n2; j++)
                value |= T((uvw[matrix][index] >> (oldRows[i] * n2 + j)) & 1) << (i * n2 + j);

        uvw[matrix][index] = value;
    }
}

__device__ __host__ void SchemeZ2::addColumn(int matrix) {
    int n1 = n[matrix];
    int n2 = n[(matrix + 1) % 3];

    for (int index = 0; index < m; index++) {
        T value = 0;

        for (int i = 0; i < n1; i++)
            for (int j = 0; j < n2; j++)
                value |= T((uvw[matrix][index] >> (i * n2 + j)) & 1) << (i * (n2 + 1) + j);

        uvw[matrix][index] = value;
    }
}

__device__ __host__ void SchemeZ2::addRow(int matrix) {
    int n1 = n[matrix];
    int n2 = n[(matrix + 1) % 3];

    for (int index = 0; index < m; index++) {
        T value = 0;

        for (int i = 0; i < n1; i++)
            for (int j = 0; j < n2; j++)
                value |= T((uvw[matrix][index] >> (i * n2 + j)) & 1) << (i * n2 + j);

        uvw[matrix][index] = value;
    }
}

__device__ __host__ bool SchemeZ2::isValidProject(int i, int minN) const {
    return n[i] - 1 >= minN && n[(i + 1) % 3] >= minN && n[(i + 2) % 3] >= minN;
}

__device__ __host__ bool SchemeZ2::isValidExtension(int i, int maxN) const {
    int newN[3] = {n[0], n[1], n[2]};
    newN[i] += 1;

    if (m + newN[(i + 1) % 3] * newN[(i + 2) % 3] > MAX_RANK)
        return false;

    for (int p = 0; p < 3; p++) {
        if (newN[p] * newN[(p + 1) % 3] > MAX_MATRIX_ELEMENTS)
            return false;

        if (newN[p] > maxN)
            return false;
    }

    return true;
}

__device__ __host__ bool SchemeZ2::isValidProduct(int i, int maxN) const {
    int newN[3] = {n[0], n[1], n[2]};
    newN[i] *= 2;

    if (m * 2 > MAX_RANK)
        return false;

    for (int p = 0; p < 3; p++) {
        if (newN[p] * newN[(p + 1) % 3] > MAX_MATRIX_ELEMENTS)
            return false;

        if (newN[p] > maxN)
            return false;
    }

    return true;
}

__device__ __host__ bool SchemeZ2::isValidMerge(int i, const SchemeZ2 &scheme) const {
    if (m + scheme.m > MAX_RANK)
        return false;

    int j = (i + 1) % 3;
    int k = (i + 2) % 3;

    int eq2 = n[j] == scheme.n[j];
    int eq3 = n[k] == scheme.n[k];

    int n1 = n[i] + scheme.n[i];

    return n1 <= MAX_EXTENSION_N && n1 * n[j] <= MAX_MATRIX_ELEMENTS && n1 * n[k] <= MAX_MATRIX_ELEMENTS && eq2 && eq3;
}

/******************************************************** helpers ********************************************************/
__device__ ReduceGaussCandidate SchemeZ2::getReduceGaussCandidate(curandState &state) const {
    int permutation[3];
    randomPermutation(permutation, 3, state);

    int indices[MAX_RANK];
    ReduceGaussCandidate possibleReduce;

    #pragma unroll
    for (int index = 0; index < m; index++)
        indices[index] = index;

    #pragma unroll
    for (int i = 0; i < 3; i++) {
        shellSort(indices, uvw[permutation[i]], m);

        int start = 0;

        #pragma unroll
        for (int index = start + 1; index <= m; index++) {
            if (index != m && uvw[permutation[i]][indices[index]] == uvw[permutation[i]][indices[start]])
                continue;

            // end of duplicate from indices[start] ... indices[index - 1]
            possibleReduce.i = (permutation[i] + 2) % 3;
            possibleReduce.size = findXorCombination((permutation[i] + 1) % 3, indices + start, index - start, possibleReduce.combination);
            if (possibleReduce.size > 0)
                return possibleReduce;

            possibleReduce.i = (permutation[i] + 1) % 3;
            possibleReduce.size = findXorCombination((permutation[i] + 2) % 3, indices + start, index - start, possibleReduce.combination);
            if (possibleReduce.size > 0)
                return possibleReduce;

            start = index;
        }
    }

    possibleReduce.i = -1;
    return possibleReduce;
}

__device__ int SchemeZ2::findXorCombination(int uvwIndex, int *indices, int size, int *combination) const {
    if (size < 3)
        return 0;

    #pragma unroll
    for (int index = 0; index < size; index++) {
        const T target = uvw[uvwIndex][indices[index]];

        #pragma unroll
        for (int index1 = 0; index1 < size; index1++) {
            if (index1 == index)
                continue;

            const T v1 = uvw[uvwIndex][indices[index1]];

            #pragma unroll
            for (int index2 = index1 + 1; index2 < size; index2++) {
                if (index2 == index)
                    continue;

                const T v2 = uvw[uvwIndex][indices[index2]];

                #pragma unroll
                for (int index3 = index2 + 1; index3 < size; index3++) {
                    if (index3 == index)
                        continue;

                    const T v3 = uvw[uvwIndex][indices[index3]];

                    #pragma unroll
                    for (int index4 = index3 + 1; index4 < size; index4++) {
                        if (index4 == index)
                            continue;

                        const T v4 = uvw[uvwIndex][indices[index4]];

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

__device__ void SchemeZ2::shellSort(int *indices, const T *values, int n) const {
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

__device__ bool SchemeZ2::inverseMatrixZ2(int n, int *matrix, int *inverse) const {
    int augmented[2 * MAX_SANDWICHING_ELEMENTS];
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

__device__ void SchemeZ2::invertibleMatrixZ2(int n, int *matrix, int *inverse, curandState &state) const {
    do {
        randomMatrixZ2(n, matrix, state);
    } while (!inverseMatrixZ2(n, matrix, inverse));
}

__device__ T SchemeZ2::matmul(const T matrix, int *left, int *right, int n1, int n2) const {
    int result1[MAX_MATRIX_ELEMENTS];
    int result2[MAX_MATRIX_ELEMENTS];

    #pragma unroll
    for (int i = 0; i < n1 * n2; i++) {
        result1[i] = 0;
        result2[i] = 0;
    }

    #pragma unroll
    for (int i = 0; i < n1; i++)
        #pragma unroll
        for (int j = 0; j < n2; j++)
            #pragma unroll
            for (int k = 0; k < n1; k++)
                result1[i * n2 + j] ^= left[i * n1 + k] && int((matrix >> (k * n2 + j)) & 1);

    #pragma unroll
    for (int i = 0; i < n1; i++)
        #pragma unroll
        for (int j = 0; j < n2; j++)
            #pragma unroll
            for (int k = 0; k < n2; k++)
                result2[i * n2 + j] ^= result1[i * n2 + k] && right[k * n2 + j];

    T result = 0;

    #pragma unroll
    for (int i = 0; i < n1 * n2; i++)
        if (result2[i])
            result |= T(1) << i;

    return result;
}

/******************************************************* operators *******************************************************/
__device__ __host__ void SchemeZ2::flip(int i, int j, int k, int index1, int index2, bool checkReduce) {
    uvw[j][index1] ^= uvw[j][index2];
    uvw[k][index2] ^= uvw[k][index1];

    flips[j].remove(index1);
    flips[k].remove(index2);

    if (!uvw[j][index1] || !uvw[k][index2]) {
        removeZeroes();
        initFlips();
        return;
    }

    for (int index = 0; index < m; index++) {
        if (index != index1 && uvw[j][index] == uvw[j][index1]) {
            if (checkReduce) {
                if (uvw[i][index] == uvw[i][index1]) {
                    reduce(k, index, index1);
                    return;
                }

                if (uvw[k][index] == uvw[k][index1]) {
                    reduce(i, index, index1);
                    return;
                }
            }

            flips[j].add(index1, index);
        }

        if (index != index2 && uvw[k][index] == uvw[k][index2]) {
            if (checkReduce) {
                if (uvw[i][index] == uvw[i][index2]) {
                    reduce(j, index, index2);
                    return;
                }

                if (uvw[j][index] == uvw[j][index2]) {
                    reduce(i, index, index2);
                    return;
                }
            }

            flips[k].add(index2, index);
        }
    }
}

__device__ __host__ void SchemeZ2::plus(int i, int j, int k, int index1, int index2, int variant) {
    const T a1 = uvw[i][index1];
    const T b1 = uvw[j][index1];
    const T c1 = uvw[k][index1];

    const T a2 = uvw[i][index2];
    const T b2 = uvw[j][index2];
    const T c2 = uvw[k][index2];

    const T a = a1 ^ a2;
    const T b = b1 ^ b2;
    const T c = c1 ^ c2;

    if (variant == 0) {
        uvw[j][index1] = b;
        uvw[i][index2] = a;
        addTriplet(i, j, k, a1, b2, c);
    }
    else if (variant == 1) {
        uvw[k][index1] = c;
        uvw[j][index2] = b;
        addTriplet(i, j, k, a, b1, c2);
    }
    else {
        uvw[i][index1] = a;
        uvw[k][index2] = c;
        addTriplet(i, j, k, a2, b, c1);
    }

    if (!a || !b || !c)
        removeZeroes();

    initFlips();
}

__device__ __host__ void SchemeZ2::split(int i, int j, int k, int index, const T a1) {
    T a2 = uvw[i][index] ^ a1;
    uvw[i][index] = a1;
    addTriplet(i, j, k, a2, uvw[j][index], uvw[k][index]);
    initFlips();
}

__device__ __host__ void SchemeZ2::reduceGauss(int i, int *combination, int combinationSize) {
    int last = combination[combinationSize - 1];

    #pragma unroll
    for (int index = 0; index < combinationSize - 1; index++)
        uvw[i][combination[index]] ^= uvw[i][last];

    removeAt(last);
}

__device__ __host__ void SchemeZ2::reduce(int i, int index1, int index2) {
    uvw[i][index1] ^= uvw[i][index2];
    bool isZero = !uvw[i][index1];

    removeAt(index2);

    if (isZero)
        removeZeroes();

    initFlips();
}

__device__ __host__ void SchemeZ2::project(int p, int q) {
    excludeRow(p, q);
    excludeColumn((p + 2) % 3, q);
    n[p]--;

    for (int i = 0; i < 3; i++)
        nn[i] = n[i] * n[(i + 1) % 3];

    removeZeroes();
    initFlips();
}

__device__ __host__ void SchemeZ2::extend(int p) {
    if (p == 0) {
        addRow(0);
        addColumn(2);

        for (int i = 0; i < n[2]; i++)
            for (int j = 0; j < n[1]; j++)
                addTriplet(0, 1, 2, T(1) << (n[0] * n[1] + j), T(1) << (j * n[2] + i), T(1) << (i * (n[0] + 1) + n[0]));
    }
    else if (p == 1) {
        addRow(1);
        addColumn(0);

        for (int i = 0; i < n[0]; i++)
            for (int j = 0; j < n[2]; j++)
                addTriplet(0, 1, 2, T(1) << (i * (n[1] + 1) + n[1]), T(1) << (n[1] * n[2] + j), T(1) << (j * n[0] + i));
    }
    else {
        addRow(2);
        addColumn(1);

        for (int i = 0; i < n[0]; i++)
            for (int j = 0; j < n[1]; j++)
                addTriplet(0, 1, 2, T(1) << (i * n[1] + j), T(1) << (j * (n[2] + 1) + n[2]), T(1) << (n[2] * n[0] + i));
    }

    n[p]++;

    for (int i = 0; i < 3; i++)
        nn[i] = n[i] * n[(i + 1) % 3];

    initFlips();
}

__device__ __host__ void SchemeZ2::product(int p) {
    int nNew[3] = {n[0], n[1], n[2]};
    int nnNew[3];
    int d[3];
    int mNew = m * 2;

    nNew[p] *= 2;

    for (int i = 0; i < 3; i++) {
        nnNew[i] = nNew[i] * nNew[(i + 1) % 3];
        d[i] = p == i ? n[i] : 0;
    }

    for (int index = 0; index < m; index++) {
        T u1(0), u2(0);
        T v1(0), v2(0);
        T w1(0), w2(0);

        for (int i = 0; i < n[0]; i++) {
            for (int j = 0; j < n[1]; j++) {
                T uij = (uvw[0][index] >> (i * n[1] + j)) & 1;
                u1 |= uij << (i * nNew[1] + j);
                u2 |= uij << ((i + d[0]) * nNew[1] + j + d[1]);
            }
        }

        for (int i = 0; i < n[1]; i++) {
            for (int j = 0; j < n[2]; j++) {
                T vij = (uvw[1][index] >> (i * n[2] + j)) & 1;
                v1 |= vij << (i * nNew[2] + j);
                v2 |= vij << ((i + d[1]) * nNew[2] + j + d[2]);
            }
        }

        for (int i = 0; i < n[2]; i++) {
            for (int j = 0; j < n[0]; j++) {
                T wij = (uvw[2][index] >> (i * n[0] + j)) & 1;
                w1 |= wij << (i * nNew[0] + j);
                w2 |= wij << ((i + d[2]) * nNew[0] + j + d[0]);
            }
        }

        uvw[0][index] = u1;
        uvw[1][index] = v1;
        uvw[2][index] = w1;

        uvw[0][index + m] = u2;
        uvw[1][index + m] = v2;
        uvw[2][index + m] = w2;
    }

    for (int i = 0; i < 3; i++) {
        n[i] = nNew[i];
        nn[i] = nnNew[i];
    }

    m = mNew;
    initFlips();
}

__device__ __host__ void SchemeZ2::merge(const SchemeZ2 &scheme, int p) {
    int nNew[3];
    int nnNew[3];
    int d[3];

    for (int i = 0; i < 3; i++) {
        nNew[i] = i == p ? n[i] + scheme.n[i] : n[i];
        d[i] = i == p ? n[i] : 0;
    }

    for (int i = 0; i < 3; i++)
        nnNew[i] = nNew[i] * nNew[(i + 1) % 3];

    for (int index = 0; index < m; index++) {
        T u = 0;
        T v = 0;
        T w = 0;

        for (int i = 0; i < n[0]; i++)
            for (int j = 0; j < n[1]; j++)
                u |= T((uvw[0][index] >> (i * n[1] + j)) & 1) << (i * nNew[1] + j);

        for (int i = 0; i < n[1]; i++)
            for (int j = 0; j < n[2]; j++)
                v |= T((uvw[1][index] >> (i * n[2] + j)) & 1) << (i * nNew[2] + j);

        for (int i = 0; i < n[2]; i++)
            for (int j = 0; j < n[0]; j++)
                w |= T((uvw[2][index] >> (i * n[0] + j)) & 1) << (i * nNew[0] + j);

        uvw[0][index] = u;
        uvw[1][index] = v;
        uvw[2][index] = w;
    }

    for (int index = 0; index < scheme.m; index++) {
        T u = 0;
        T v = 0;
        T w = 0;

        for (int i = 0; i < scheme.n[0]; i++)
            for (int j = 0; j < scheme.n[1]; j++)
                u |= T((scheme.uvw[0][index] >> (i * scheme.n[1] + j)) & 1) << ((i + d[0]) * nNew[1] + j + d[1]);

        for (int i = 0; i < scheme.n[1]; i++)
            for (int j = 0; j < scheme.n[2]; j++)
                v |= T((scheme.uvw[1][index] >> (i * scheme.n[2] + j)) & 1) << ((i + d[1]) * nNew[2] + j + d[2]);

        for (int i = 0; i < scheme.n[2]; i++)
            for (int j = 0; j < scheme.n[0]; j++)
                w |= T((scheme.uvw[2][index] >> (i * scheme.n[0] + j)) & 1) << ((i + d[2]) * nNew[0] + j + d[0]);

        uvw[0][m + index] = u;
        uvw[1][m + index] = v;
        uvw[2][m + index] = w;
    }

    for (int i = 0; i < 3; i++) {
        n[i] = nNew[i];
        nn[i] = nnNew[i];
    }

    m += scheme.m;

    initFlips();

    // if (!validate())
    //     printf("invalid merge over %d (%d, %d, %d, %d) <- (%d, %d, %d, %d)\n", p, n[0], n[1], n[2], m, scheme.n[0], scheme.n[1], scheme.n[2], scheme.m);
}

__device__ __host__ void SchemeZ2::product(const SchemeZ2 &scheme2) {
    SchemeZ2 scheme1;
    copyTo(scheme1, false);

    for (int i = 0; i < 3; i++)
        n[i] = scheme1.n[i] * scheme2.n[i];

    for (int i = 0; i < 3; i++)
        nn[i] = n[i] * n[(i + 1) % 3];

    m = scheme1.m * scheme2.m;

    for (int index1 = 0; index1 < scheme1.m; index1++) {
        for (int index2 = 0; index2 < scheme2.m; index2++) {
            int index = index1 * scheme2.m + index2;

            for (int p = 0; p < 3; p++) {
                int p1 = (p + 1) % 3;

                uvw[p][index] = 0;

                for (int i = 0; i < scheme1.nn[p]; i++) {
                    for (int j = 0; j < scheme2.nn[p]; j++) {
                        int row1 = i / scheme1.n[p1];
                        int col1 = i % scheme1.n[p1];
                        T value1 = (scheme1.uvw[p][index1] >> i) & 1;

                        int row2 = j / scheme2.n[p1];
                        int col2 = j % scheme2.n[p1];
                        T value2 = (scheme2.uvw[p][index2] >> j) & 1;

                        int row = row1 * scheme2.n[p] + row2;
                        int col = col1 * scheme2.n[p1] + col2;
                        T value = value1 & value2;

                        uvw[p][index] |= value << (row * n[p1] + col);
                    }
                }
            }
        }
    }

    initFlips();
}

__device__ __host__ void SchemeZ2::swapBasisRows(int i1, int i2) {
    int rows[MAX_MATRIX_ELEMENTS];

    for (int row = 0; row < n[0]; row++)
        rows[row] = row;

    rows[i1] = i2;
    rows[i2] = i1;

    for (int index = 0; index < m; index++) {
        T u = 0;
        T w = 0;

        for (int i = 0; i < n[0]; i++)
            for (int j = 0; j < n[1]; j++)
                u |= ((uvw[0][index] >> (rows[i] * n[1] + j)) & 1) << (i * n[1] + j);

        for (int i = 0; i < n[2]; i++)
            for (int j = 0; j < n[0]; j++)
                w |= ((uvw[2][index] >> (i * n[0] + rows[j])) & 1) << (i * n[0] + j);

        uvw[0][index] = u;
        uvw[2][index] = w;
    }
}

__device__ __host__ void SchemeZ2::swapBasisColumns(int j1, int j2) {
    int columns[MAX_MATRIX_ELEMENTS];

    for (int column = 0; column < n[2]; column++)
        columns[column] = column;

    columns[j1] = j2;
    columns[j2] = j1;

    for (int index = 0; index < m; index++) {
        T v = 0;
        T w = 0;

        for (int i = 0; i < n[1]; i++)
            for (int j = 0; j < n[2]; j++)
                v |= ((uvw[1][index] >> (i * n[2] + columns[j])) & 1) << (i * n[2] + j);

        for (int i = 0; i < n[2]; i++)
            for (int j = 0; j < n[0]; j++)
                w |= ((uvw[2][index] >> (columns[i] * n[0] + j)) & 1) << (i * n[0] + j);

        uvw[1][index] = v;
        uvw[2][index] = w;
    }
}

__device__ __host__ void SchemeZ2::swapSize(int p1, int p2) {
    if (p1 == p2)
        return;

    if (p1 > p2) {
        int tmp = p1;
        p1 = p2;
        p2 = tmp;
    }

    int indices[3];
    int nNew[3];

    if (p1 == 0 && p2 == 1) {
        indices[0] = 0;
        indices[1] = 2;
        indices[2] = 1;
    }
    else if (p1 == 0 && p2 == 2) {
        indices[0] = 1;
        indices[1] = 0;
        indices[2] = 2;
    }
    else {
        indices[0] = 2;
        indices[1] = 1;
        indices[2] = 0;
    }

    for (int i = 0; i < 3; i++)
        nNew[i] = n[(indices[i] + 1) % 3];

    for (int index = 0; index < m; index++) {
        T u = 0;
        T v = 0;
        T w = 0;

        for (int i = 0; i < nNew[0]; i++)
            for (int j = 0; j < nNew[1]; j++)
                u |= T((uvw[indices[0]][index] >> (j * nNew[0] + i)) & 1) << (i * nNew[1] + j);

        for (int i = 0; i < nNew[1]; i++)
            for (int j = 0; j < nNew[2]; j++)
                v |= T((uvw[indices[1]][index] >> (j * nNew[1] + i)) & 1) << (i * nNew[2] + j);

        for (int i = 0; i < nNew[2]; i++)
            for (int j = 0; j < nNew[0]; j++)
                w |= T((uvw[indices[2]][index] >> (j * nNew[2] + i)) & 1) << (i * nNew[0] + j);

        uvw[0][index] = u;
        uvw[1][index] = v;
        uvw[2][index] = w;
    }

    for (int i = 0; i < 3; i++) {
        n[i] = nNew[i];
        nn[i] = nNew[i] * nNew[(i + 1) % 3];
    }

    initFlips();

    // if (!validate())
    //     printf("invalid swap %d %d\n", p1, p2);
}

/*************************************************** random operators ****************************************************/
__device__ bool SchemeZ2::tryFlip(curandState &state, bool checkReduce) {
    int size = flips[0].size + flips[1].size + flips[2].size;

    if (!size)
        return false;

    int index = curand(&state) % size;
    int i, j, k, index1, index2;

    if (index < flips[0].size) {
        i = 0;
        j = 1;
        k = 2;
        index1 = flips[0].index1(index);
        index2 = flips[0].index2(index);
    }
    else if (index < flips[0].size + flips[1].size) {
        i = 1;
        j = 0;
        k = 2;
        index1 = flips[1].index1(index - flips[0].size);
        index2 = flips[1].index2(index - flips[0].size);
    }
    else {
        i = 2;
        j = 0;
        k = 1;
        index1 = flips[2].index1(index - flips[0].size - flips[1].size);
        index2 = flips[2].index2(index - flips[0].size - flips[1].size);
    }

    if (curand(&state) % 2) {
        int tmp = j;
        j = k;
        k = tmp;
    }

    if (curand(&state) % 2) {
        int tmp = index1;
        index1 = index2;
        index2 = tmp;
    }

    flip(i, j, k, index1, index2, checkReduce);
    return true;
}

__device__ bool SchemeZ2::tryPlus(curandState &state) {
    if (m >= MAX_RANK || m >= n[0] * n[1] * n[2])
        return false;

    int index1 = curand(&state) % m;
    int index2 = curand(&state) % m;

    while (index1 == index2)
        index2 = curand(&state) % m;

    int permutation[3];
    randomPermutation(permutation, 3, state);
    int i = permutation[0];
    int j = permutation[1];
    int k = permutation[2];

    int variant = curand(&state) % 3;
    plus(i, j, k, index1, index2, variant);
    return true;
}

__device__ bool SchemeZ2::trySplit(curandState &state) {
    if (m >= MAX_RANK || m >= n[0] * n[1] * n[2])
        return false;

    int index = curand(&state) % m;

    int permutation[3];
    randomPermutation(permutation, 3, state);
    int i = permutation[0];
    int j = permutation[1];
    int k = permutation[2];

    T a1 = curand(&state) % (T(1) << nn[i]);
    const T a2 = uvw[i][index];
    int loop = 0;

    while (loop < 5 && a1 == a2) {
        a1 = curand(&state) % (T(1) << nn[i]);
        loop++;
    }

    if (loop == 5)
        return false;

    split(i, j, k, index, a1);
    return true;
}

__device__ bool SchemeZ2::trySplitExisted(curandState &state) {
    if (m >= MAX_RANK || m >= n[0] * n[1] * n[2])
        return false;

    int permutation[3];
    randomPermutation(permutation, 3, state);
    int i = permutation[0];
    int j = permutation[1];
    int k = permutation[2];

    int index1 = curand(&state) % m;
    int index2 = curand(&state) % m;
    int loop = 0;

    while (loop < 5 && (index1 == index2 || uvw[i][index1] == uvw[i][index2])) {
        index2 = curand(&state) % m;
        loop++;
    }

    if (loop == 5)
        return false;

    split(i, j, k, index1, uvw[i][index2]);
    return true;
}

__device__ bool SchemeZ2::tryExpand(int count, curandState &state) {
    int maxRank = n[0] * n[1] * n[2];
    bool result = false;

    for (int i = 0; i < count && m < maxRank; i++) {
        int v = curand(&state) % 3;

        if (v == 0) {
            result |= tryPlus(state);
        }
        else if (v == 1) {
            result |= trySplit(state);
        }
        else {
            result |= trySplitExisted(state);
        }
    }

    return result;
}

__device__ __host__ bool SchemeZ2::tryReduce() {
    for (size_t i = 0; i < flips[0].size; i++) {
        int index1 = flips[0].index1(i);
        int index2 = flips[0].index2(i);

        if (uvw[1][index1] == uvw[1][index2]) {
            reduce(2, index1, index2);
            return true;
        }

        if (uvw[2][index1] == uvw[2][index2]) {
            reduce(1, index1, index2);
            return true;
        }
    }

    for (size_t i = 0; i < flips[1].size; i++) {
        int index1 = flips[1].index1(i);
        int index2 = flips[1].index2(i);

        if (uvw[2][index1] == uvw[2][index2]) {
            reduce(0, index1, index2);
            return true;
        }
    }

    return false;
}

__device__ bool SchemeZ2::tryReduceGauss(curandState &state) {
    ReduceGaussCandidate possibleReduce = getReduceGaussCandidate(state);
    if (possibleReduce.i == -1)
        return false;

    reduceGauss(possibleReduce.i, possibleReduce.combination, possibleReduce.size);
    return true;
}

__device__ bool SchemeZ2::tryProject(curandState &state, int minN) {
    int indices[3];
    int size = 0;

    #pragma unroll
    for (int i = 0; i < 3; i++)
        if (isValidProject(i, minN))
            indices[size++] = i;

    if (!size)
        return false;

    int p = indices[curand(&state) % size];
    int q = curand(&state) % n[p];
    project(p, q);

    while (tryReduce())
        ;

    return true;
}

__device__ bool SchemeZ2::tryExtend(curandState &state, int maxN) {
    int indices[3];
    int size = 0;

    #pragma unroll
    for (int i = 0; i < 3; i++)
        if (isValidExtension(i, maxN))
            indices[size++] = i;

    if (!size)
        return false;

    int p = indices[curand(&state) % size];
    extend(p);

    while (tryReduce())
        ;

    return true;
}

__device__ bool SchemeZ2::tryProduct(curandState &state, int maxN) {
    int indices[3];
    int size = 0;

    #pragma unroll
    for (int i = 0; i < 3; i++)
        if (isValidProduct(i, maxN))
            indices[size++] = i;

    if (!size)
        return false;

    int p = indices[curand(&state) % size];
    product(p);
    return true;
}

__device__ bool SchemeZ2::tryMerge(const SchemeZ2 &scheme, curandState &state) {
    if (m + scheme.m > MAX_RANK)
        return false;

    bool eq1 = n[0] == scheme.n[0];
    bool eq2 = n[1] == scheme.n[1];
    bool eq3 = n[2] == scheme.n[2];

    int n1 = n[0] + scheme.n[0];
    int n2 = n[1] + scheme.n[1];
    int n3 = n[2] + scheme.n[2];

    int p[3];
    int size = 0;

    if (n1 <= MAX_EXTENSION_N && n1 * n[1] <= MAX_MATRIX_ELEMENTS && n1 * n[2] <= MAX_MATRIX_ELEMENTS && eq2 && eq3)
        p[size++] = 0;

    if (eq1 && n2 <= MAX_EXTENSION_N && n[0] * n2 <= MAX_MATRIX_ELEMENTS && n[2] * n2 <= MAX_MATRIX_ELEMENTS && eq3)
        p[size++] = 1;

    if (eq1 && eq2 && n3 <= MAX_EXTENSION_N && n[0] * n3 <= MAX_MATRIX_ELEMENTS && n[1] * n3 <= MAX_MATRIX_ELEMENTS)
        p[size++] = 2;

    if (size == 0)
        return false;

    merge(scheme, p[curand(&state) % size]);
    return true;
}

__device__ bool SchemeZ2::tryProduct(const SchemeZ2 &scheme) {
    if (m * scheme.m > MAX_RANK)
        return false;

    int sizes[3];

    for (int i = 0; i < 3; i++)
        sizes[i] = n[i] * scheme.n[i];

    for (int i = 0; i < 3; i++) {
        if (sizes[i] > MAX_EXTENSION_N)
            return false;

        if (sizes[i] * sizes[(i + 1) % 3] > MAX_MATRIX_ELEMENTS)
            return false;
    }

    product(scheme);
    return true;
}

__device__ void SchemeZ2::sandwiching(curandState &state) {
    int u[MAX_SANDWICHING_ELEMENTS];
    int v[MAX_SANDWICHING_ELEMENTS];
    int w[MAX_SANDWICHING_ELEMENTS];

    int u1[MAX_SANDWICHING_ELEMENTS];
    int v1[MAX_SANDWICHING_ELEMENTS];
    int w1[MAX_SANDWICHING_ELEMENTS];

    invertibleMatrixZ2(n[0], u, u1, state);
    invertibleMatrixZ2(n[1], v, v1, state);
    invertibleMatrixZ2(n[2], w, w1, state);

    for (int index = 0; index < m; index++) {
        uvw[0][index] = matmul(uvw[0][index], u, v1, n[0], n[1]);
        uvw[1][index] = matmul(uvw[1][index], v, w1, n[1], n[2]);
        uvw[2][index] = matmul(uvw[2][index], w, u1, n[2], n[0]);
    }

    initFlips();

    // if (!validate())
    //     printf("invalid sandwiching\n");
}

__device__ void SchemeZ2::swapBasis(curandState &state) {
    if (curand(&state) % 2) {
        int i1 = curand(&state) % n[0];
        int i2 = curand(&state) % n[0];
        swapBasisRows(i1, i2);
    }
    else {
        int j1 = curand(&state) % n[2];
        int j2 = curand(&state) % n[2];
        swapBasisColumns(j1, j2);
    }
}

__device__ void SchemeZ2::swapSize(curandState &state) {
    int p1, p2;

    do {
        p1 = curand(&state) % 3;
        p2 = curand(&state) % 3;
    } while (p1 == p2);

    swapSize(p1, p2);
}

/**************************************************** save *****************************************************/
void SchemeZ2::saveMatrix(std::ofstream &f, std::string name, int n1, int n2, int m, const T *matrix) const {
    f << "    \"" << name << "\": [" << std::endl;

    for (int index = 0; index < m; index++) {
        f << "        [";

        for (int i = 0; i < n1 * n2; i++) {
            if (i > 0)
                f << ", ";

            f << ((matrix[index] >> i) & 1);
        }

        f << "]" << (index < m - 1 ? "," : "") << std::endl;
    }

    f << "    ]";
}

void SchemeZ2::save(const std::string &path) {
    std::ofstream f(path);

    f << "{" << std::endl;
    f << "    \"n\": [" << n[0] << ", " << n[1] << ", " << n[2] << "]," << std::endl;
    f << "    \"m\": " << m << "," << std::endl;
    f << "    \"z2\": true," << std::endl;
    f << "    \"complexity\": " << getComplexity() << "," << std::endl;

    saveMatrix(f, "u", n[0], n[1], m, uvw[0]);
    f << "," << std::endl;
    saveMatrix(f, "v", n[1], n[2], m, uvw[1]);
    f << "," << std::endl;
    saveMatrix(f, "w", n[2], n[0], m, uvw[2]);
    f << std::endl;
    f << "}" << std::endl;

    f.close();
}
