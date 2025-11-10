#include "scheme_integer.cuh"

__device__ __host__ bool Scheme::validateEquation(int i, int j, int k) const {
    int i1 = i / n[1];
    int i2 = i % n[1];
    int j1 = j / n[2];
    int j2 = j % n[2];
    int k1 = k / n[0];
    int k2 = k % n[0];

    int target = (i2 == j1) && (i1 == k2) && (j2 == k1);
    int equation = 0;

    for (int index = 0; index < m; index++)
        equation += uvw[0][index][i] * uvw[1][index][j] * uvw[2][index][k];

    return equation == target;
}

__device__ __host__ bool Scheme::validate() const {
    bool valid = true;

    for (int i = 0; i < nn[0] && valid; i++)
        for (int j = 0; j < nn[1] && valid; j++)
            for (int k = 0; k < nn[2] && valid; k++)
                valid &= validateEquation(i, j, k);

    return valid;
}

/*************************************************** device functions ****************************************************/
__device__ __host__ void Scheme::initializeNaive(int n1, int n2, int n3) {
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
                uvw[0][index] = Addition(n1 * n2, i * n2 + k);
                uvw[1][index] = Addition(n2 * n3, k * n3 + j);
                uvw[2][index] = Addition(n3 * n1, j * n1 + i);
            }
        }
    }

    if (!validate())
        printf("not valid naive scheme\n");
}

__device__ __host__ void Scheme::initializeFrom(int n1, int n2, int n3, int m, int scheme[3][MAX_RANK][MAX_MATRIX_ELEMENTS]) {
    n[0] = n1;
    n[1] = n2;
    n[2] = n3;

    nn[0] = n1 * n2;
    nn[1] = n2 * n3;
    nn[2] = n3 * n1;

    m = n1 * n2 * n3;

    for (int i = 0; i < 3; i++)
        for (int index = 0; index < m; index++)
            uvw[i][index] = Addition(nn[i], scheme[i][index]);

    if (!validate())
        printf("not valid from scheme %d %d %d %d\n", n1, n2, n3, m);
}

__device__ __host__ void Scheme::copyTo(Scheme &target) {
    target.m = m;

    for (int i = 0; i < 3; i++) {
        target.n[i] = n[i];
        target.nn[i] = nn[i];

        for (int index = 0; index < m; index++)
            target.uvw[i][index] = uvw[i][index];
    }
}

__device__ __host__ void Scheme::removeZeroes() {
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
    // int i = 0;

    // for (int index = 0; index < m; index++) {
    //     if (uvw[0][index] && uvw[1][index] && uvw[2][index]) {
    //         if (i != index) {
    //             uvw[0][i] = uvw[0][index];
    //             uvw[1][i] = uvw[1][index];
    //             uvw[2][i] = uvw[2][index];
    //         }

    //         i++;
    //     }
    // }

    // m = i;
}

__device__ __host__ void Scheme::removeAt(int index) {
    m--;

    if (index == m)
        return;

    uvw[0][index] = uvw[0][m];
    uvw[1][index] = uvw[1][m];
    uvw[2][index] = uvw[2][m];
}

__device__ __host__ void Scheme::addTriplet(int i, int j, int k, const Addition &u, const Addition &v, const Addition &w) {
    u.copyTo(uvw[i][m]);
    v.copyTo(uvw[j][m]);
    w.copyTo(uvw[k][m]);
    m++;
}

__device__ __host__ void Scheme::excludeColumn(int matrix, int column) {
    int n1 = n[matrix];
    int n2 = n[(matrix + 1) % 3];
    int oldColumns[MAX_MATRIX_SIZE];
    int size = 0;

    for (int j = 0; j < n2; j++)
        if (j != column)
            oldColumns[size++] = j;

    for (int index = 0; index < m; index++) {
        Addition value(n1 * (n2 - 1));

        for (int i = 0; i < n1; i++)
            for (int j = 0; j < n2 - 1; j++)
                value.values[i * (n2 - 1) + j] = uvw[matrix][index][i * n2 + oldColumns[j]];

        value.copyTo(uvw[matrix][index]);
    }
}

__device__ __host__ void Scheme::excludeRow(int matrix, int row) {
    int n1 = n[matrix];
    int n2 = n[(matrix + 1) % 3];
    int oldRows[MAX_MATRIX_SIZE];
    int size = 0;

    for (int i = 0; i < n1; i++)
        if (i != row)
            oldRows[size++] = i;

    for (int index = 0; index < m; index++) {
        Addition value((n1 - 1) * n2);

        for (int i = 0; i < n1 - 1; i++)
            for (int j = 0; j < n2; j++)
                value.values[i * n2 + j] = uvw[matrix][index][oldRows[i] * n2 + j];

        value.copyTo(uvw[matrix][index]);
    }
}

__device__ __host__ void Scheme::addColumn(int matrix) {
    int n1 = n[matrix];
    int n2 = n[(matrix + 1) % 3];

    for (int index = 0; index < m; index++) {
        Addition value(n1 * (n2 + 1));

        for (int i = 0; i < n1; i++)
            for (int j = 0; j < n2; j++)
                value.values[i * (n2 + 1) + j] = uvw[matrix][index][i * n2 + j];

        value.copyTo(uvw[matrix][index]);
    }
}

__device__ __host__ void Scheme::addRow(int matrix) {
    int n1 = n[matrix];
    int n2 = n[(matrix + 1) % 3];

    for (int index = 0; index < m; index++) {
        Addition value((n1 + 1) * n2);

        for (int i = 0; i < n1; i++)
            for (int j = 0; j < n2; j++)
                value.values[i * n2 + j] = uvw[matrix][index][i * n2 + j];

        value.copyTo(uvw[matrix][index]);
    }
}

/******************************************************** helpers ********************************************************/
__device__ FlipCandidate Scheme::getFlipCandidate(curandState &state) const {
    int permutation[3];
    int indices[MAX_RANK];

    randomPermutation(permutation, 3, state);
    randomPermutation(indices, m, state);

    int variant = curand(&state) % 3;

    if (variant == 0) {
        #pragma unroll
        for (int p = 0; p < 3; p++) {
            int i = permutation[p];
            int j = permutation[(p + 1) % 3];
            int k = permutation[(p + 2) % 3];

            #pragma unroll
            for (int index1 = 0; index1 < m; index1++) {
                #pragma unroll
                for (int index2 = index1 + 1; index2 < m; index2++) {
                    if (uvw[i][indices[index1]] != uvw[i][indices[index2]])
                        continue;

                    if (uvw[j][indices[index1]].limitSum(uvw[j][indices[index2]], j != 2) && uvw[k][indices[index2]].limitSub(uvw[k][indices[index1]])) {
                        if (k == 2 || uvw[k][indices[index2]].positiveFirstNonZeroSub(uvw[k][indices[index1]]))
                            return {j, k, indices[index1], indices[index2]};
                        else
                            return {j, k, indices[index2], indices[index1]};
                    }

                    if (uvw[k][indices[index1]].limitSum(uvw[k][indices[index2]], k != 2) && uvw[j][indices[index2]].limitSub(uvw[j][indices[index1]])) {
                        if (j == 2 || uvw[j][indices[index2]].positiveFirstNonZeroSub(uvw[j][indices[index1]]))
                            return {k, j, indices[index1], indices[index2]};
                        else
                            return {k, j, indices[index2], indices[index1]};
                    }
                }
            }
        }
    }
    else if (variant == 1) {
        #pragma unroll
        for (int index1 = 0; index1 < m; index1++) {
            #pragma unroll
            for (int p = 0; p < 3; p++) {
                int i = permutation[p];
                int j = permutation[(p + 1) % 3];
                int k = permutation[(p + 2) % 3];

                #pragma unroll
                for (int index2 = index1 + 1; index2 < m; index2++) {
                    if (uvw[i][indices[index1]] != uvw[i][indices[index2]])
                        continue;

                    if (uvw[j][indices[index1]].limitSum(uvw[j][indices[index2]], j != 2) && uvw[k][indices[index2]].limitSub(uvw[k][indices[index1]])) {
                        if (k == 2 || uvw[k][indices[index2]].positiveFirstNonZeroSub(uvw[k][indices[index1]]))
                            return {j, k, indices[index1], indices[index2]};
                        else
                            return {j, k, indices[index2], indices[index1]};
                    }

                    if (uvw[k][indices[index1]].limitSum(uvw[k][indices[index2]], k != 2) && uvw[j][indices[index2]].limitSub(uvw[j][indices[index1]])) {
                        if (j == 2 || uvw[j][indices[index2]].positiveFirstNonZeroSub(uvw[j][indices[index1]]))
                            return {k, j, indices[index1], indices[index2]};
                        else
                            return {k, j, indices[index2], indices[index1]};
                    }
                }
            }
        }
    }
    else {
        #pragma unroll
        for (int index1 = 0; index1 < m; index1++) {
            #pragma unroll
            for (int index2 = index1 + 1; index2 < m; index2++) {
                #pragma unroll
                for (int p = 0; p < 3; p++) {
                    int i = permutation[p];
                    int j = permutation[(p + 1) % 3];
                    int k = permutation[(p + 2) % 3];

                    if (uvw[i][indices[index1]] != uvw[i][indices[index2]])
                        continue;

                    if (uvw[j][indices[index1]].limitSum(uvw[j][indices[index2]], j != 2) && uvw[k][indices[index2]].limitSub(uvw[k][indices[index1]])) {
                        if (k == 2 || uvw[k][indices[index2]].positiveFirstNonZeroSub(uvw[k][indices[index1]]))
                            return {j, k, indices[index1], indices[index2]};
                        else
                            return {j, k, indices[index2], indices[index1]};
                    }

                    if (uvw[k][indices[index1]].limitSum(uvw[k][indices[index2]], k != 2) && uvw[j][indices[index2]].limitSub(uvw[j][indices[index1]])) {
                        if (j == 2 || uvw[j][indices[index2]].positiveFirstNonZeroSub(uvw[j][indices[index1]]))
                            return {k, j, indices[index1], indices[index2]};
                        else
                            return {k, j, indices[index2], indices[index1]};
                    }
                }
            }
        }
    }

    return {-1, 0, 0, 0};
}

__device__ ReduceCandidate Scheme::getReduceCandidate(curandState &state) const {
    int permutation[3];
    int indices[MAX_RANK];

    randomPermutation(permutation, 3, state);
    randomPermutation(indices, m, state);

    int i = permutation[0];
    int j = permutation[1];
    int k = permutation[2];

    #pragma unroll
    for (int index1 = 0; index1 < m; index1++) {
        #pragma unroll
        for (int index2 = index1 + 1; index2 < m; index2++) {
            const Addition u1 = uvw[i][indices[index1]];
            const Addition u2 = uvw[i][indices[index2]];

            const Addition v1 = uvw[j][indices[index1]];
            const Addition v2 = uvw[j][indices[index2]];

            const Addition w1 = uvw[k][indices[index1]];
            const Addition w2 = uvw[k][indices[index2]];

            if (u1 == u2) {
                if (w1 == w2 && uvw[j][indices[index1]].limitSum(uvw[j][indices[index2]], j != 2))
                    return {j, indices[index1], indices[index2]};

                if (v1 == v2 && uvw[k][indices[index1]].limitSum(uvw[k][indices[index2]], k != 2))
                    return {k, indices[index1], indices[index2]};
            }
            else if (v1 == v2 && w1 == w2 && uvw[i][indices[index1]].limitSum(uvw[i][indices[index2]], i != 2)) {
                return {i, indices[index1], indices[index2]};
            }
        }
    }

    return {-1, 0, 0};
}

/******************************************************* operators *******************************************************/
__device__ __host__ void Scheme::flip(int first, int second, int index1, int index2) {
    uvw[first][index1] += uvw[first][index2];
    uvw[second][index2] -= uvw[second][index1];

    if (!uvw[first][index1] || !uvw[second][index2])
        removeZeroes();
}

__device__ __host__ void Scheme::plus(int i, int j, int k, int index1, int index2, int variant) {
    const Addition a1 = uvw[i][index1];
    const Addition b1 = uvw[j][index1];
    const Addition c1 = uvw[k][index1];

    const Addition a2 = uvw[i][index2];
    const Addition b2 = uvw[j][index2];
    const Addition c2 = uvw[k][index2];

    const Addition aAdd = a1 + a2;
    const Addition bAdd = b1 + b2;
    const Addition cAdd = c1 + c2;

    const Addition aSub = a2 - a1;
    const Addition bSub = b2 - b1;
    const Addition cSub = c2 - c1;

    if (variant == 0 && aSub.limit(i != 2) && bAdd.limit(j != 2) && cSub.limit(k != 2)) {
        uvw[j][index1] = bAdd;
        uvw[i][index2] = aSub;
        addTriplet(i, j, k, a1, b2, cSub);
    }
    else if (variant == 1 && aSub.limit(i != 2) && bSub.limit(j != 2) && cAdd.limit(k != 2)) {
        uvw[k][index1] = cAdd;
        uvw[j][index2] = bSub;
        addTriplet(i, j, k, aSub, b1, c2);
    }
    else if (aAdd.limit(i != 2) && bSub.limit(j != 2) && cSub.limit(k != 2)) {
        uvw[i][index1] = aAdd;
        uvw[k][index2] = cSub;
        addTriplet(i, j, k, a2, bSub, c1);
    }

    removeZeroes();
}

__device__ __host__ void Scheme::reduce(int i, int index1, int index2) {
    uvw[i][index1] += uvw[i][index2];
    bool isZero = !uvw[i][index1];

    removeAt(index2);

    if (isZero)
        removeZeroes();
}

__device__ __host__ void Scheme::project(int p, int q) {
    excludeRow(p, q);
    excludeColumn((p + 2) % 3, q);
    n[p]--;

    for (int i = 0; i < 3; i++)
        nn[i] = n[i] * n[(i + 1) % 3];

    removeZeroes();

    if (!validate())
        printf("project: invalid scheme %d %d (%d, %d, %d)\n", p, q, n[0], n[1], n[2]);
}

__device__ __host__ void Scheme::extend(int p) {
    if (p == 0) {
        addRow(0);
        addColumn(2);

        for (int i = 0; i < n[2]; i++)
            for (int j = 0; j < n[1]; j++)
                addTriplet(0, 1, 2, Addition((n[0] + 1) * n[1], n[0] * n[1] + j), Addition(n[1] * n[2], j * n[2] + i), Addition(n[2] * (n[0] + 1), i * (n[0] + 1) + n[0]));
    }
    else if (p == 1) {
        addRow(1);
        addColumn(0);

        for (int i = 0; i < n[0]; i++)
            for (int j = 0; j < n[2]; j++)
                addTriplet(0, 1, 2, Addition(n[0] * (n[1] + 1), i * (n[1] + 1) + n[1]), Addition((n[1] + 1) * n[2], n[1] * n[2] + j), Addition(n[2] * n[0], j * n[0] + i));
    }
    else {
        addRow(2);
        addColumn(1);

        for (int i = 0; i < n[0]; i++)
            for (int j = 0; j < n[1]; j++)
                addTriplet(0, 1, 2, Addition(n[0] * n[1], i * n[1] + j), Addition(n[1] * (n[2] + 1), j * (n[2] + 1) + n[2]), Addition((n[2] + 1) * n[0], n[2] * n[0] + i));
    }

    n[p]++;

    for (int i = 0; i < 3; i++)
        nn[i] = n[i] * n[(i + 1) % 3];

    if (!validate())
        printf("extend: invalid scheme %d (%d, %d, %d)\n", p, n[0], n[1], n[2]);
}

/*************************************************** random operators ****************************************************/
__device__ bool Scheme::tryFlip(curandState &state) {
    FlipCandidate possibleFlip = getFlipCandidate(state);
    if (possibleFlip.first < 0)
        return false;

    flip(possibleFlip.first, possibleFlip.second, possibleFlip.index1, possibleFlip.index2);
    return true;
}

__device__ bool Scheme::tryPlus(curandState &state) {
    if (m >= n[0] * n[1] * n[2])
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

__device__ bool Scheme::tryExpand(int count, curandState &state) {
    bool result = false;

    for (int i = 0; i < count; i++) {
        result |= tryPlus(state);
    }

    return result;
}

__device__ bool Scheme::tryReduce(curandState &state) {
    ReduceCandidate possibleReduce = getReduceCandidate(state);
    if (possibleReduce.i == -1)
        return false;

    reduce(possibleReduce.i, possibleReduce.index1, possibleReduce.index2);
    return true;
}

__device__ bool Scheme::tryProject(curandState &state, int n1, int n2, int n3) {
    int indices[3];
    int size = 0;

    if (n[0] - 1 >= n1 && n[1] >= n2 && n[2] >= n3)
        indices[size++] = 0;

    if (n[0] >= n1 && n[1] - 1 >= n2 && n[2] >= n3)
        indices[size++] = 1;

    if (n[0] >= n1 && n[1] >= n2 && n[2] - 1 >= n3)
        indices[size++] = 2;

    if (size == 0)
        return false;

    int p = indices[curand(&state) % size];
    int q = curand(&state) % n[p];
    project(p, q);

    while (tryReduce(state))
        ;

    return true;
}

__device__ bool Scheme::tryExtend(curandState &state, int n1, int n2, int n3) {
    int indices[3];
    int size = 0;

    if (n[0] + 1 <= n1 && n[1] <= n2 && n[2] <= n3 && (n[0] + 1) * n[1] <= MAX_SIZE && (n[0] + 1) * n[2] <= MAX_SIZE && m + n[1] * n[2] <= MAX_RANK)
        indices[size++] = 0;

    if (n[0] <= n1 && n[1] + 1 <= n2 && n[2] <= n3 && (n[1] + 1) * n[0] <= MAX_SIZE && (n[1] + 1) * n[2] <= MAX_SIZE && m + n[0] * n[2] <= MAX_RANK)
        indices[size++] = 1;

    if (n[0] <= n1 && n[1] <= n2 && n[2] + 1 <= n3 && (n[2] + 1) * n[0] <= MAX_SIZE && (n[2] + 1) * n[1] <= MAX_SIZE && m + n[0] * n[1] <= MAX_RANK)
        indices[size++] = 2;

    if (size == 0)
        return false;

    extend(indices[curand(&state) % size]);

    while (tryReduce(state))
        ;

    return true;
}

__device__ void Scheme::sandwiching(curandState &state) {
    // TODO
}

/**************************************************** save *****************************************************/
void Scheme::saveMatrix(std::ofstream &f, std::string name, int m, const Addition *additions) const {
    f << "    \"" << name << "\": [" << std::endl;

    for (int index = 0; index < m; index++) {
        f << "        [";

        for (int i = 0; i < additions[index].n; i++) {
            if (i > 0)
                f << ", ";

            f << int(additions[index][i]);
        }

        f << "]" << (index < m - 1 ? "," : "") << std::endl;
    }

    f << "    ]";
}

void Scheme::save(const std::string &path) {
    std::ofstream f(path);

    f << "{" << std::endl;
    f << "    \"n\": [" << n[0] << ", " << n[1] << ", " << n[2] << "]," << std::endl;
    f << "    \"m\": " << m << "," << std::endl;
    f << "    \"z2\": false," << std::endl;

    saveMatrix(f, "u", m, uvw[0]);
    f << "," << std::endl;
    saveMatrix(f, "v", m, uvw[1]);
    f << "," << std::endl;
    saveMatrix(f, "w", m, uvw[2]);
    f << std::endl;
    f << "}" << std::endl;

    f.close();
}

void Scheme::showTensor(const Addition &addition, int n1, int n2, std::string name, bool transpose) const {
    bool printed = false;

    std::cout << "(";

    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            auto value = addition[i * n2 + j];

            if (!value)
                continue;

            if (printed)
                std::cout << " ";

            if (value < 0)
                std::cout << "- ";
            else if (printed)
                std::cout << "+ ";

            if (value > 1)
                std::cout << value;
            else if (value < -1)
                std::cout << (-value);

            std::cout << name;

            if (transpose)
                std::cout << (j + 1) << (i + 1);
            else
                std::cout << (i + 1) << (j + 1);

            printed = true;
        }
    }

    std::cout << ")";
}

void Scheme::show() const {
    for (int index = 0; index < m; index++) {
        showTensor(uvw[0][index], n[0], n[1], "a", false);
        std::cout << " x ";
        showTensor(uvw[1][index], n[1], n[2], "b", false);
        std::cout << " x ";
        showTensor(uvw[2][index], n[2], n[0], "c", true);
        std::cout << std::endl;
    }
}
