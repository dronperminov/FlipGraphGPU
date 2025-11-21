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
            if (!uvw[i][index].valid) {
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

__device__ __host__ void SchemeInteger::copyTo(SchemeInteger &target, bool withFlips) const {
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

__host__ bool SchemeInteger::read(std::istream &is, bool checkValidity) {
    is >> n[0] >> n[1] >> n[2] >> m;

    if (n[0] * n[1] > MAX_MATRIX_ELEMENTS || n[1] * n[2] > MAX_MATRIX_ELEMENTS || n[2] * n[0] > MAX_MATRIX_ELEMENTS || m > MAX_RANK) {
        printf("SchemeInteger::read(is): invalid scheme sizes (%d, %d, %d, %d)\n", n[0], n[1], n[2], m);
        return false;
    }

    int value;

    for (int i = 0; i < 3; i++) {
        nn[i] = n[i] * n[(i + 1) % 3];

        for (int index = 0; index < m; index++) {
            uvw[i][index] = Addition(nn[i]);

            for (int j = 0; j < nn[i]; j++) {
                is >> value;
                uvw[i][index].set(j, value);
            }
        }
    }

    fixSigns();
    initFlips();
    return !checkValidity || validate();
}

__host__ bool SchemeInteger::read(std::istream &is, int n1, int n2, int n3, int m, bool checkValidity) {
    if (n1 * n2 > MAX_MATRIX_ELEMENTS || n2 * n3 > MAX_MATRIX_ELEMENTS || n3 * n1 > MAX_MATRIX_ELEMENTS || m > MAX_RANK) {
        printf("SchemeInteger::read(is, n1, n2, n3, m): invalid scheme sizes (%d, %d, %d, %d)\n", n1, n2, n3, m);
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
            uvw[i][index] = Addition(nn[i]);

            for (int j = 0; j < nn[i]; j++) {
                is >> value;
                uvw[i][index].set(j, value);
            }
        }
    }

    fixSigns();
    initFlips();
    return !checkValidity || validate();
}

__device__ __host__ int SchemeInteger::getComplexity() const {
    int count = 0;

    for (int index = 0; index < m; index++)
        for (int i = 0; i < 3; i++)
            count += uvw[i][index].nonZeroCount();

    return count - 2 * m - nn[2];
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
    int oldColumns[MAX_MATRIX_ELEMENTS];
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
    int oldRows[MAX_MATRIX_ELEMENTS];
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

__device__ __host__ bool SchemeInteger::isValidProject(int i, int minN) const {
    return n[i] - 1 >= minN && n[(i + 1) % 3] >= minN && n[(i + 2) % 3] >= minN;
}

__device__ __host__ bool SchemeInteger::isValidExtension(int i, int maxN) const {
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

__device__ __host__ bool SchemeInteger::isValidProduct(int i, int maxN) const {
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

__device__ __host__ bool SchemeInteger::fixSigns() {
    bool changed = false;

    for (int index = 0; index < m; index++) {
        bool i = uvw[0][index].positiveFirstNonZero();
        bool j = uvw[1][index].positiveFirstNonZero();

        if (i && j)
            continue;

        if (!i && !j) {
            uvw[0][index].inverse();
            uvw[1][index].inverse();
        }
        else if (!i) {
            uvw[0][index].inverse();
            uvw[2][index].inverse();
        }
        else {
            uvw[1][index].inverse();
            uvw[2][index].inverse();
        }

        changed = true;
    }

    return changed;
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
                int cmpI = uvw[i][index].compare(uvw[i][index1]);
                if (cmpI == 1 && uvw[k][index].limitSum(uvw[k][index1], k != 2)) {
                    reduceAdd(k, index, index1);
                    return;
                }

                if (i == 2 && cmpI == -1 && uvw[k][index].limitSub(uvw[k][index1], true)) {
                    reduceSub(k, index, index1);
                    return;
                }

                int cmpK = uvw[k][index].compare(uvw[k][index1]);
                if (cmpK == 1 && uvw[i][index].limitSum(uvw[i][index1], i != 2)) {
                    reduceAdd(i, index, index1);
                    return;
                }

                if (k == 2 && cmpK == -1 && uvw[i][index].limitSub(uvw[i][index1], true)) {
                    reduceSub(i, index, index1);
                    return;
                }
            }

            flips[j].add(index1, index);
        }

        if (index != index2 && uvw[k][index] == uvw[k][index2]) {
            if (checkReduce) {
                int cmpI = uvw[i][index].compare(uvw[i][index2]);
                if (cmpI == 1 && uvw[j][index].limitSum(uvw[j][index2], j != 2)) {
                    reduceAdd(j, index, index2);
                    return;
                }

                if (i == 2 && cmpI == -1 && uvw[j][index].limitSub(uvw[j][index2], true)) {
                    reduceSub(j, index, index2);
                    return;
                }

                int cmpJ = uvw[j][index].compare(uvw[j][index2]);
                if (cmpJ == 1 && uvw[i][index].limitSum(uvw[i][index2], i != 2)) {
                    reduceAdd(i, index, index2);
                    return;
                }

                if (j == 2 && cmpJ == -1 && uvw[i][index].limitSub(uvw[i][index2], true)) {
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
    fixSigns();
    initFlips();
}

__device__ __host__ void SchemeInteger::split(int i, int j, int k, int index, const Addition& addition) {
    addTriplet(i, j, k, uvw[i][index] - addition, uvw[j][index], uvw[k][index]);
    uvw[i][index] = addition;

    fixSigns();
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
    fixSigns();
    initFlips();
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
}

__device__ __host__ void SchemeInteger::product(int p) {
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
        Addition u1(nnNew[0]), u2(nnNew[0]);
        Addition v1(nnNew[1]), v2(nnNew[1]);
        Addition w1(nnNew[2]), w2(nnNew[2]);

        for (int i = 0; i < n[0]; i++) {
            for (int j = 0; j < n[1]; j++) {
                int uij = uvw[0][index][i * n[1] + j];
                u1.set(i * nNew[1] + j, uij);
                u2.set((i + d[0]) * nNew[1] + j + d[1], uij);
            }
        }

        for (int i = 0; i < n[1]; i++) {
            for (int j = 0; j < n[2]; j++) {
                int vij = uvw[1][index][i * n[2] + j];
                v1.set(i * nNew[2] + j, vij);
                v2.set((i + d[1]) * nNew[2] + j + d[2], vij);
            }
        }

        for (int i = 0; i < n[2]; i++) {
            for (int j = 0; j < n[0]; j++) {
                int wij = uvw[2][index][i * n[0] + j];
                w1.set(i * nNew[0] + j, wij);
                w2.set((i + d[2]) * nNew[0] + j + d[0], wij);
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

__device__ __host__ void SchemeInteger::merge(const SchemeInteger &scheme, int p) {
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
        Addition u(nnNew[0]);
        Addition v(nnNew[1]);
        Addition w(nnNew[2]);

        for (int i = 0; i < n[0]; i++)
            for (int j = 0; j < n[1]; j++)
                u.set(i * nNew[1] + j, uvw[0][index][i * n[1] + j]);

        for (int i = 0; i < n[1]; i++)
            for (int j = 0; j < n[2]; j++)
                v.set(i * nNew[2] + j, uvw[1][index][i * n[2] + j]);

        for (int i = 0; i < n[2]; i++)
            for (int j = 0; j < n[0]; j++)
                w.set(i * nNew[0] + j, uvw[2][index][i * n[0] + j]);

        uvw[0][index] = u;
        uvw[1][index] = v;
        uvw[2][index] = w;
    }

    for (int index = 0; index < scheme.m; index++) {
        Addition u(nnNew[0]);
        Addition v(nnNew[1]);
        Addition w(nnNew[2]);

        for (int i = 0; i < scheme.n[0]; i++)
            for (int j = 0; j < scheme.n[1]; j++)
                u.set((i + d[0]) * nNew[1] + j + d[1], scheme.uvw[0][index][i * scheme.n[1] + j]);

        for (int i = 0; i < scheme.n[1]; i++)
            for (int j = 0; j < scheme.n[2]; j++)
                v.set((i + d[1]) * nNew[2] + j + d[2], scheme.uvw[1][index][i * scheme.n[2] + j]);

        for (int i = 0; i < scheme.n[2]; i++)
            for (int j = 0; j < scheme.n[0]; j++)
                w.set((i + d[2]) * nNew[0] + j + d[0], scheme.uvw[2][index][i * scheme.n[0] + j]);

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
}

__device__ __host__ void SchemeInteger::product(const SchemeInteger &scheme2) {
    SchemeInteger scheme1;
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

                uvw[p][index] = Addition(nn[p]);

                for (int i = 0; i < scheme1.nn[p]; i++) {
                    for (int j = 0; j < scheme2.nn[p]; j++) {
                        int row1 = i / scheme1.n[p1];
                        int col1 = i % scheme1.n[p1];
                        int value1 = scheme1.uvw[p][index1][i];

                        int row2 = j / scheme2.n[p1];
                        int col2 = j % scheme2.n[p1];
                        int value2 = scheme2.uvw[p][index2][j];

                        int row = row1 * scheme2.n[p] + row2;
                        int col = col1 * scheme2.n[p1] + col2;

                        uvw[p][index].set(row * n[p1] + col, value1 * value2);
                    }
                }
            }
        }
    }

    initFlips();
}

__device__ __host__ void SchemeInteger::swapBasisRows(int i1, int i2) {
    int rows[MAX_MATRIX_ELEMENTS];

    for (int row = 0; row < n[0]; row++)
        rows[row] = row;

    rows[i1] = i2;
    rows[i2] = i1;

    for (int index = 0; index < m; index++) {
        Addition u(n[0] * n[1]);
        Addition w(n[2] * n[0]);

        for (int i = 0; i < n[0]; i++)
            for (int j = 0; j < n[1]; j++)
                u.set(i * n[1] + j, uvw[0][index][rows[i] * n[1] + j]);

        for (int i = 0; i < n[2]; i++)
            for (int j = 0; j < n[0]; j++)
                w.set(i * n[0] + j, uvw[2][index][i * n[0] + rows[j]]);

        uvw[0][index] = u;
        uvw[2][index] = w;
    }
}

__device__ __host__ void SchemeInteger::swapBasisColumns(int j1, int j2) {
    int columns[MAX_MATRIX_ELEMENTS];

    for (int column = 0; column < n[2]; column++)
        columns[column] = column;

    columns[j1] = j2;
    columns[j2] = j1;

    for (int index = 0; index < m; index++) {
        Addition v(n[1] * n[2]);
        Addition w(n[2] * n[0]);

        for (int i = 0; i < n[1]; i++)
            for (int j = 0; j < n[2]; j++)
                v.set(i * n[2] + j, uvw[1][index][i * n[2] + columns[j]]);

        for (int i = 0; i < n[2]; i++)
            for (int j = 0; j < n[0]; j++)
                w.set(i * n[0] + j, uvw[2][index][columns[i] * n[0] + j]);

        uvw[1][index] = v;
        uvw[2][index] = w;
    }
}

__device__ __host__ void SchemeInteger::swapSize(int p1, int p2) {
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
        Addition u(nNew[0] * nNew[1]);
        Addition v(nNew[1] * nNew[2]);
        Addition w(nNew[2] * nNew[0]);

        for (int i = 0; i < nNew[0]; i++)
            for (int j = 0; j < nNew[1]; j++)
                u.set(i * nNew[1] + j, uvw[indices[0]][index][j * nNew[0] + i]);

        for (int i = 0; i < nNew[1]; i++)
            for (int j = 0; j < nNew[2]; j++)
                v.set(i * nNew[2] + j, uvw[indices[1]][index][j * nNew[1] + i]);

        for (int i = 0; i < nNew[2]; i++)
            for (int j = 0; j < nNew[0]; j++)
                w.set(i * nNew[0] + j, uvw[indices[2]][index][j * nNew[2] + i]);

        uvw[0][index] = u;
        uvw[1][index] = v;
        uvw[2][index] = w;
    }

    for (int i = 0; i < 3; i++) {
        n[i] = nNew[i];
        nn[i] = nNew[i] * nNew[(i + 1) % 3];
    }

    fixSigns();
    initFlips();
}

/*************************************************** random operators ****************************************************/
__device__ bool SchemeInteger::tryFlip(curandState &state, bool checkReduce) {
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
                flip(i, j, k, index1, index2, checkReduce);
                return true;
            }
            else {
                flip(i, j, k, index2, index1, checkReduce);
                return true;
            }
        }

        if (uvw[k][index1].limitSum(uvw[k][index2], k != 2) && uvw[j][index2].limitSub(uvw[j][index1])) {
            if (j == 2 || uvw[j][index2].positiveFirstNonZeroSub(uvw[j][index1])) {
                flip(i, k, j, index1, index2, checkReduce);
                return true;
            }
            else {
                flip(i, k, j, index2, index1, checkReduce);
                return true;
            }
        }
    }

    return false;
}

__device__ bool SchemeInteger::tryPlus(curandState &state) {
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

__device__ bool SchemeInteger::trySplit(curandState &state) {
    if (m >= MAX_RANK || m >= n[0] * n[1] * n[2])
        return false;

    int index = curand(&state) % m;

    int permutation[3];
    randomPermutation(permutation, 3, state);
    int i = permutation[0];
    int j = permutation[1];
    int k = permutation[2];

    Addition a(nn[i]);
    a.random(state);

    if (!uvw[i][index].limitSub(a, i != 2))
        return false;

    split(i, j, k, index, a);
    return true;
}

__device__ bool SchemeInteger::trySplitExisted(curandState &state) {
    if (m >= MAX_RANK || m >= n[0] * n[1] * n[2])
        return false;

    int index1, index2;
    int i;

    do {
        index1 = curand(&state) % m;
        index2 = curand(&state) % m;
        i = curand(&state) % 3;
    } while (index1 == index2 || uvw[i][index1] == uvw[i][index2]);

    if (!uvw[i][index1].limitSub(uvw[i][index2], i != 2))
        return false;

    int j = (i + 1) % 3;
    int k = (i + 2) % 3;

    split(i, j, k, index1, uvw[i][index2]);
    return true;
}

__device__ bool SchemeInteger::tryExpand(int count, curandState &state) {
    bool result = false;

    for (int i = 0; i < count && m < MAX_RANK; i++) {
        int v = curand(&state) % 3;

        if (v == 0)
            result |= tryPlus(state);
        else if (v == 1)
            result |= trySplit(state);
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

        int cmp2 = uvw[2][index1].compare(uvw[2][index2]);
        if (cmp2 == 1 && uvw[1][index1].limitSum(uvw[1][index2], true)) {
            reduceAdd(1, index1, index2);
            return true;
        }

        if (cmp2 == -1 && uvw[1][index1].limitSub(uvw[1][index2], true)) {
            reduceSub(1, index1, index2);
            return true;
        }
    }

    for (size_t i = 0; i < flips[1].size; i++) {
        int index1 = flips[1].index1(i);
        int index2 = flips[1].index2(i);
        int cmp2 = uvw[2][index1].compare(uvw[2][index2]);

        if (cmp2 == 1 && uvw[0][index1].limitSum(uvw[0][index2], true)) {
            reduceAdd(0, index1, index2);
            return true;
        }

        if (cmp2 == -1 && uvw[0][index1].limitSub(uvw[0][index2], true)) {
            reduceSub(0, index1, index2);
            return true;
        }
    }

    return false;
}

__device__ bool SchemeInteger::tryProject(curandState &state, int minN) {
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

__device__ bool SchemeInteger::tryExtend(curandState &state, int maxN) {
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

__device__ bool SchemeInteger::tryProduct(curandState &state, int maxN) {
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

__device__ bool SchemeInteger::tryMerge(const SchemeInteger &scheme, curandState &state) {
    if (m + scheme.m > MAX_RANK)
        return false;

    int eq1 = n[0] == scheme.n[0];
    int eq2 = n[1] == scheme.n[1];
    int eq3 = n[2] == scheme.n[2];

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

__device__ bool SchemeInteger::tryProduct(const SchemeInteger &scheme) {
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

__device__ void SchemeInteger::sandwiching(curandState &state) {
    // TODO
}

__device__ void SchemeInteger::swapBasis(curandState &state) {
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

    if (fixSigns())
        initFlips();
}

__device__ void SchemeInteger::swapSize(curandState &state) {
    int p1, p2;

    do {
        p1 = curand(&state) % 3;
        p2 = curand(&state) % 3;
    } while (p1 == p2);

    swapSize(p1, p2);
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
    f << "    \"complexity\": " << getComplexity() << "," << std::endl;

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
