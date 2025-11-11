#include "scheme_integer.cuh"

__device__ __host__ bool SchemeInteger::validateEquation(int i, int j, int k) const {
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

__device__ __host__ bool SchemeInteger::validate() const {
    bool valid = true;

    for (int i = 0; i < nn[0] && valid; i++)
        for (int j = 0; j < nn[1] && valid; j++)
            for (int k = 0; k < nn[2] && valid; k++)
                valid &= validateEquation(i, j, k);

    for (int index = 0;index < m; index++) {
        for (int i = 0; i < 3; i++) {
            if (uvw[i][index].carry) {
                printf("carry number\n");
                return false;
            }
        }
    }

    return valid;
}

/*************************************************** device functions ****************************************************/
__device__ __host__ void SchemeInteger::initializeNaive(int n1, int n2, int n3) {
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

    initFlips();
}

__device__ __host__ void SchemeInteger::initializeFrom(int n1, int n2, int n3, int m, int scheme[3][MAX_RANK][MAX_MATRIX_ELEMENTS]) {
    n[0] = n1;
    n[1] = n2;
    n[2] = n3;

    nn[0] = n1 * n2;
    nn[1] = n2 * n3;
    nn[2] = n3 * n1;

    this->m = m;

    for (int i = 0; i < 3; i++)
        for (int index = 0; index < m; index++)
            uvw[i][index] = Addition(nn[i], scheme[i][index]);

    initFlips();

    if (!validate())
        printf("not valid from scheme %d %d %d %d\n", n1, n2, n3, m);
}

__device__ __host__ void SchemeInteger::copyTo(SchemeInteger &target) {
    target.m = m;

    for (int i = 0; i < 3; i++) {
        target.n[i] = n[i];
        target.nn[i] = nn[i];

        for (int index = 0; index < m; index++)
            target.uvw[i][index] = uvw[i][index];
    }

    target.initFlips();
}

__device__ __host__ void SchemeInteger::initFlips() {
    for (int i = 0; i < 3; i++) {
        flips[i].clear();

        for (int index1 = 0; index1 < m; index1++)
            for (int index2 = index1 + 1; index2 < m; index2++)
                if (uvw[i][index1] == uvw[i][index2])
                    flips[i].add(index1, index2);
    }
}

__device__ __host__ void SchemeInteger::removeZeroes() {
    for (int index = 0; index < m; index++) {
        if (uvw[0][index] && uvw[1][index] && uvw[2][index])
            continue;

        m--;
        uvw[0][index] = uvw[0][m];
        uvw[1][index] = uvw[1][m];
        uvw[2][index] = uvw[2][m];
        index--;
    }
}

__device__ __host__ void SchemeInteger::removeAt(int index) {
    m--;

    if (index == m)
        return;

    uvw[0][index] = uvw[0][m];
    uvw[1][index] = uvw[1][m];
    uvw[2][index] = uvw[2][m];
}

__device__ __host__ void SchemeInteger::addTriplet(int i, int j, int k, const Addition &u, const Addition &v, const Addition &w) {
    u.copyTo(uvw[i][m]);
    v.copyTo(uvw[j][m]);
    w.copyTo(uvw[k][m]);
    m++;
}

__device__ __host__ void SchemeInteger::excludeColumn(int matrix, int column) {
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
                value.set(i * (n2 - 1) + j, uvw[matrix][index][i * n2 + oldColumns[j]]);

        uvw[matrix][index] = value;
    }
}

__device__ __host__ void SchemeInteger::excludeRow(int matrix, int row) {
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
                value.set(i * n2 + j, uvw[matrix][index][oldRows[i] * n2 + j]);

        uvw[matrix][index] = value;
    }
}

__device__ __host__ void SchemeInteger::addColumn(int matrix) {
    int n1 = n[matrix];
    int n2 = n[(matrix + 1) % 3];

    for (int index = 0; index < m; index++) {
        Addition value(n1 * (n2 + 1));

        for (int i = 0; i < n1; i++)
            for (int j = 0; j < n2; j++)
                value.set(i * (n2 + 1) + j, uvw[matrix][index][i * n2 + j]);

        uvw[matrix][index] = value;
    }
}

__device__ __host__ void SchemeInteger::addRow(int matrix) {
    int n1 = n[matrix];
    int n2 = n[(matrix + 1) % 3];

    for (int index = 0; index < m; index++) {
        Addition value((n1 + 1) * n2);

        for (int i = 0; i < n1; i++)
            for (int j = 0; j < n2; j++)
                value.set(i * n2 + j, uvw[matrix][index][i * n2 + j]);

        uvw[matrix][index] = value;
    }
}

/******************************************************* operators *******************************************************/
__device__ __host__ void SchemeInteger::flip(int i, int j, int k, int index1, int index2, bool checkReduce) {
    uvw[j][index1] += uvw[j][index2];
    uvw[k][index2] -= uvw[k][index1];

    flips[j].remove(index1);
    flips[k].remove(index2);

    if (!uvw[j][index1] || !uvw[k][index2]) {
        removeZeroes();
        initFlips();

        while (tryReduce())
            ;
        return;
    }

    for (int index = 0; index < m; index++) {
        if (index != index1 && uvw[j][index] == uvw[j][index1]) {
            if (checkReduce) {
                if (uvw[i][index] == uvw[i][index1] && uvw[k][index].limitSum(uvw[k][index1], k != 2)) {
                    reduceAdd(k, index, index1);
                    return;
                }

                if (i == 2 && uvw[i][index] == -uvw[i][index1] && uvw[k][index].limitSub(uvw[k][index1], k != 2)) {
                    reduceSub(k, index, index1);
                    return;
                }

                if (uvw[k][index] == uvw[k][index1] && uvw[i][index].limitSum(uvw[i][index1], i != 2)) {
                    reduceAdd(i, index, index1);
                    return;
                }

                if (k == 2 && uvw[k][index] == -uvw[k][index1] && uvw[i][index].limitSub(uvw[i][index1], i != 2)) {
                    reduceSub(i, index, index1);
                    return;
                }
            }

            flips[j].add(index1, index);
        }

        if (index != index2 && uvw[k][index] == uvw[k][index2]) {
            if (checkReduce) {
                if (uvw[i][index] == uvw[i][index2] && uvw[j][index].limitSum(uvw[j][index2], j != 2)) {
                    reduceAdd(j, index, index2);
                    return;
                }

                if (i == 2 && uvw[i][index] == -uvw[i][index2] && uvw[j][index].limitSub(uvw[j][index2], j != 2)) {
                    reduceSub(j, index, index2);
                    return;
                }

                if (uvw[j][index] == uvw[j][index2] && uvw[i][index].limitSum(uvw[i][index2], i != 2)) {
                    reduceAdd(i, index, index2);
                    return;
                }

                if (j == 2 && uvw[j][index] == -uvw[j][index2] && uvw[i][index].limitSub(uvw[i][index2], i != 2)) {
                    reduceSub(i, index, index2);
                    return;
                }
            }

            flips[k].add(index2, index);
        }
    }
}

__device__ __host__ void SchemeInteger::plus(int i, int j, int k, int index1, int index2, int variant) {
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
    initFlips();
}

__device__ __host__ void SchemeInteger::split(int i, int j, int k, int index, const Addition& addition) {
    addTriplet(i, j, k, uvw[i][index] - addition, uvw[j][index], uvw[k][index]);
    uvw[i][index] = addition;

    initFlips();
}

__device__ __host__ void SchemeInteger::reduceAdd(int i, int index1, int index2) {
    uvw[i][index1] += uvw[i][index2];
    bool isZero = !uvw[i][index1];

    removeAt(index2);

    if (isZero)
        removeZeroes();

    initFlips();
}

__device__ __host__ void SchemeInteger::reduceSub(int i, int index1, int index2) {
    uvw[i][index1] -= uvw[i][index2];
    bool isZero = !uvw[i][index1];

    removeAt(index2);

    if (isZero)
        removeZeroes();

    initFlips();
}

__device__ __host__ void SchemeInteger::project(int p, int q) {
    excludeRow(p, q);
    excludeColumn((p + 2) % 3, q);
    n[p]--;

    for (int i = 0; i < 3; i++)
        nn[i] = n[i] * n[(i + 1) % 3];

    removeZeroes();
    initFlips();

    if (!validate())
        printf("project: invalid scheme %d %d (%d, %d, %d)\n", p, q, n[0], n[1], n[2]);
}

__device__ __host__ void SchemeInteger::extend(int p) {
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

    initFlips();

    if (!validate())
        printf("extend: invalid scheme %d (%d, %d, %d)\n", p, n[0], n[1], n[2]);
}

/*************************************************** random operators ****************************************************/
__device__ bool SchemeInteger::tryFlip(curandState &state) {
    int size = flips[0].size + flips[1].size + flips[2].size;
    int indices[MAX_PAIRS];

    randomPermutation(indices, size, state);

    for (int p = 0; p < size; p++) {
        int index = indices[p];
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

        if (uvw[j][index1].limitSum(uvw[j][index2], j != 2) && uvw[k][index2].limitSub(uvw[k][index1])) {
            if (k == 2 || uvw[k][index2].positiveFirstNonZeroSub(uvw[k][index1])) {
                flip(i, j, k, index1, index2);
                return true;
            }
            else {
                flip(i, j, k, index2, index1);
                return true;
            }
        }

        if (uvw[k][index1].limitSum(uvw[k][index2], k != 2) && uvw[j][index2].limitSub(uvw[j][index1])) {
            if (j == 2 || uvw[j][index2].positiveFirstNonZeroSub(uvw[j][index1])) {
                flip(i, k, j, index1, index2);
                return true;
            }
            else {
                flip(i, k, j, index2, index1);
                return true;
            }
        }
    }

    return false;
}

__device__ bool SchemeInteger::tryPlus(curandState &state) {
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

__device__ bool SchemeInteger::trySplitExisted(curandState &state) {
    if (m >= MAX_RANK)
        return false;

    int index1, index2;
    int i;

    do {
        index1 = curand(&state) % m;
        index2 = curand(&state) % m;
        i = curand(&state) % 3;
    } while (index1 == index2 || uvw[i][index1] == uvw[i][index2] || !uvw[i][index1].limitSub(uvw[i][index2], i != 2));

    int j = (i + 1) % 3;
    int k = (i + 2) % 3;

    split(i, j, k, index1, uvw[i][index2]);
    return true;
}

__device__ bool SchemeInteger::tryExpand(int count, curandState &state) {
    int maxRank = n[0] * n[1] * n[2];
    bool result = false;

    for (int i = 0; i < count && m < maxRank; i++) {
        int v = curand(&state) % 2;

        if (v == 0)
            result |= tryPlus(state);
        else
            result |= trySplitExisted(state);
    }

    return result;
}

__device__ __host__ bool SchemeInteger::tryReduce() {
    for (size_t i = 0; i < flips[0].size; i++) {
        int index1 = flips[0].index1(i);
        int index2 = flips[0].index2(i);

        if (uvw[1][index1] == uvw[1][index2] && uvw[2][index1].limitSum(uvw[2][index2], false)) {
            reduceAdd(2, index1, index2);
            return true;
        }

        if (uvw[2][index1] == uvw[2][index2] && uvw[1][index1].limitSum(uvw[1][index2], true)) {
            reduceAdd(1, index1, index2);
            return true;
        }

        if (uvw[2][index1] == -uvw[2][index2] && uvw[1][index1].limitSub(uvw[1][index2], true)) {
            reduceSub(1, index1, index2);
            return true;
        }
    }

    for (size_t i = 0; i < flips[1].size; i++) {
        int index1 = flips[1].index1(i);
        int index2 = flips[1].index2(i);

        if (uvw[2][index1] == uvw[2][index2] && uvw[0][index1].limitSum(uvw[0][index2], true)) {
            reduceAdd(0, index1, index2);
            return true;
        }

        if (uvw[2][index1] == -uvw[2][index2] && uvw[0][index1].limitSub(uvw[0][index2], true)) {
            reduceSub(0, index1, index2);
            return true;
        }
    }

    return false;
}

__device__ bool SchemeInteger::tryProject(curandState &state, int n1, int n2, int n3) {
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

    while (tryReduce())
        ;

    return true;
}

__device__ bool SchemeInteger::tryExtend(curandState &state, int n1, int n2, int n3) {
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

    while (tryReduce())
        ;

    return true;
}

__device__ void SchemeInteger::sandwiching(curandState &state) {
    // TODO
}

/**************************************************** save *****************************************************/
void SchemeInteger::saveMatrix(std::ofstream &f, std::string name, int m, const Addition *additions) const {
    f << "    \"" << name << "\": [" << std::endl;

    for (int index = 0; index < m; index++)
        f << "        [" << additions[index] << "]" << (index < m - 1 ? "," : "") << std::endl;

    f << "    ]";
}

void SchemeInteger::save(const std::string &path) {
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

void SchemeInteger::showTensor(const Addition &addition, int n1, int n2, std::string name, bool transpose) const {
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

void SchemeInteger::show() const {
    for (int index = 0; index < m; index++) {
        showTensor(uvw[0][index], n[0], n[1], "a", false);
        std::cout << " x ";
        showTensor(uvw[1][index], n[1], n[2], "b", false);
        std::cout << " x ";
        showTensor(uvw[2][index], n[2], n[0], "c", true);
        std::cout << std::endl;
    }
}
