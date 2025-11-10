#include "addition.cuh"

__device__ __host__ Addition::Addition() {
    n = 0;
}

__device__ __host__ Addition::Addition(int n) {
    this->n = n;

    for (int i = 0; i < n; i++)
        values[i] = 0;
}

__device__ __host__ Addition::Addition(int n, int index) {
    this->n = n;

    for (int i = 0; i < n; i++)
        values[i] = 0;

    values[index] = 1;
}

__device__ __host__ Addition::Addition(int n, int *values) {
    this->n = n;

    for (int i = 0; i < n; i++)
        this->values[i] = values[i];
}

__device__ __host__ void Addition::copyTo(Addition &target) const {
    target.n = n;

    for (int i = 0; i < n; i++)
        target.values[i] = values[i];
}

__device__ __host__ bool Addition::operator==(const Addition &addition) const {
    for (int i = 0; i < n; i++)
        if (values[i] != addition.values[i])
            return false;

    return true;
}

__device__ __host__ bool Addition::operator!=(const Addition &addition) const {
    for (int i = 0; i < n; i++)
        if (values[i] != addition.values[i])
            return true;

    return false;
}

__device__ __host__ int8_t Addition::operator[](int index) const {
    return values[index];
}

__device__ __host__ Addition Addition::operator+(const Addition &addition) const {
    Addition result(n);

    for (int i = 0; i < n; i++)
        result.values[i] = values[i] + addition.values[i];

    return result;
}

__device__ __host__ Addition Addition::operator-(const Addition &addition) const {
    Addition result(n);

    for (int i = 0; i < n; i++)
        result.values[i] = values[i] - addition.values[i];

    return result;
}

__device__ __host__ Addition& Addition::operator+=(const Addition &addition) {
    for (int i = 0; i < n; i++)
        values[i] += addition.values[i];

    return *this;
}

__device__ __host__ Addition& Addition::operator-=(const Addition &addition) {
    for (int i = 0; i < n; i++)
        values[i] -= addition.values[i];

    return *this;
}

__device__ __host__ Addition::operator bool() const {
    for (int i = 0; i < n; i++)
        if (values[i] != 0)
            return true;

    return false;
}

__device__ __host__ bool Addition::limit(bool firstPositiveNonZero) const {
    bool haveNonZero = false;

    for (int i = 0; i < n; i++) {
        if (firstPositiveNonZero && !haveNonZero && values[i] != 0) {
            if (values[i] < 0)
                return false;

            haveNonZero = true;
        }

        if (values[i] < LOWER_BOUND || values[i] > UPPER_BOUND)
            return false;
    }

    return true;
}

__device__ __host__ bool Addition::limitSum(const Addition &addition, bool firstPositiveNonZero) const {
    bool haveNonZero = false;

    for (int i = 0; i < n; i++) {
        int8_t value = values[i] + addition.values[i];

        if (firstPositiveNonZero && !haveNonZero && value != 0) {
            if (value < 0)
                return false;

            haveNonZero = true;
        }

        if (value < LOWER_BOUND || value > UPPER_BOUND)
            return false;
    }

    return true;
}

__device__ __host__ bool Addition::limitSub(const Addition &addition) const {
    for (int i = 0; i < n; i++) {
        int8_t value = values[i] - addition.values[i];

        if (value < LOWER_BOUND || value > UPPER_BOUND)
            return false;
    }

    return true;
}

__device__ __host__ bool Addition::positiveFirstNonZero() const {
    int i = 0;

    while (i < n && values[i] == 0)
        i++;

    return i == n || values[i] > 0;
}

__device__ __host__ bool Addition::positiveFirstNonZeroSub(const Addition &addition) const {
    for (int i = 0; i < n; i++) {
        int8_t value = values[i] - addition.values[i];

        if (value != 0)
            return value > 0;
    }

    return true;
}
