#include "addition.cuh"

__device__ __host__ Addition::Addition() {
    n = 0;
    values = 0;
    signs = 0;
    valid = true;
}

__device__ __host__ Addition::Addition(int n) {
    this->n = n;
    this->values = 0;
    this->signs = 0;
    this->valid = true;
}

__device__ __host__ Addition::Addition(int n, int index) {
    this->n = n;
    this->values = T(1) << index;
    this->signs = 0;
    this->valid = true;
}

__device__ __host__ Addition::Addition(int n, int *values) {
    this->n = n;
    this->values = 0;
    this->valid = true;
    this->signs = 0;

    for (int i = 0; i < n; i++)
        set(i, values[i]);
}

__device__ __host__ void Addition::copyTo(Addition &target) const {
    target.n = n;
    target.values = values;
    target.signs = signs;
    target.valid = valid;
}

__device__ __host__ void Addition::set(int index, int value) {
    T mask = T(1) << index;

    if (value == 0) {
        values &= ~mask;
        signs &= ~mask;
    }
    else if (value == 1) {
        values |= mask;
        signs &= ~mask;
    }
    else if (value == -1) {
        values |= mask;
        signs |= mask;
    }
    else {
        valid = false;
        printf("invalid set (%d, %d)\n", index, value);
    }
}

__device__ __host__ void Addition::inverse() {
    signs = (~signs) & values;
}

__device__ void Addition::random(curandState &state) {
    values = 0;
    signs = 0;

    for (int i = 0; i < n; i += 16) {
        uint16_t signBits = curand(&state) & 0xFFFF;
        uint16_t valuesBits = curand(&state) & 0xFFFF;

        values |= valuesBits << i;
        signs |= signBits << i;
    }

    signs &= values;
    valid = true;
}

__device__ __host__ bool Addition::operator==(const Addition &addition) const {
    return values == addition.values && signs == addition.signs;
}

__device__ __host__ bool Addition::operator!=(const Addition &addition) const {
    return values != addition.values || signs != addition.signs;
}

__device__ __host__ int Addition::operator[](int index) const {
    int value = int((values >> index) & 1);

    if ((signs >> index) & 1)
        value = -value;

    return value;
}

__device__ __host__ Addition Addition::operator+(const Addition &addition) const {
    Addition result(n);

    result.values = values ^ addition.values;
    result.signs = ((signs & values) | (addition.signs & addition.values)) & result.values;
    result.valid = !(values & addition.values & ~(signs ^ addition.signs));
    return result;
}

__device__ __host__ Addition Addition::operator-(const Addition &addition) const {
    Addition result(n);

    result.values = values ^ addition.values;
    result.signs = ((signs & values) | (~addition.signs & addition.values)) & result.values;
    result.valid = !(values & addition.values & (signs ^ addition.signs));
    return result;
}

__device__ __host__ Addition Addition::operator-() const {
    Addition result(n);
    result.values = values;
    result.signs = (~signs) & values;
    result.valid = valid;
    return result;
}

__device__ __host__ Addition& Addition::operator+=(const Addition &addition) {
    T sv1 = signs & values;
    T sv2 = addition.signs & addition.values;

    valid = !((values & addition.values & ~(signs ^ addition.signs)));
    values ^= addition.values;
    signs = (sv1 | sv2) & values;
    return *this;
}

__device__ __host__ Addition& Addition::operator-=(const Addition &addition) {
    T sv1 = signs & values;
    T sv2 = ~addition.signs & addition.values;

    valid = !(values & addition.values & (signs ^ addition.signs));
    values ^= addition.values;
    signs = (sv1 | sv2) & values;
    return *this;
}

__device__ __host__ Addition::operator bool() const {
    return values != 0;
}

__device__ __host__ bool Addition::limit(bool firstPositiveNonZero) const {
    if (!valid)
        return false;

    if (firstPositiveNonZero)
        return values == 0 || (values & ~(values & (values - 1)) & ~signs);

    return true;
}

__device__ __host__ bool Addition::limitSum(const Addition &addition, bool firstPositiveNonZero) const {
    bool invalid = values & addition.values & ~(signs ^ addition.signs);

    if (invalid)
        return false;

    if (firstPositiveNonZero) {
        T sumValues = values ^ addition.values;
        T sumSigns = ((signs & values) | (addition.signs & addition.values)) & sumValues;

        return sumValues == 0 || (sumValues & ~(sumValues & (sumValues - 1)) & ~sumSigns);
    }

    return true;
}

__device__ __host__ bool Addition::limitSub(const Addition &addition, bool firstPositiveNonZero) const {
    bool invalid = (values & addition.values & (signs ^ addition.signs));

    if (invalid)
        return false;

    if (firstPositiveNonZero) {
        T subValues = values ^ addition.values;
        T subSigns = ((signs & values) | (~addition.signs & addition.values)) & subValues;

        return subValues == 0 || (subValues & ~(subValues & (subValues - 1)) & ~subSigns);
    }

    return true;
}

__device__ __host__ bool Addition::positiveFirstNonZeroSub(const Addition &addition) const {
    T subValues = values ^ addition.values;
    T subSigns = ((signs & values) | (~addition.signs & addition.values)) & subValues;

    return subValues == 0 || (subValues & ~(subValues & (subValues - 1)) & ~subSigns);
}

__device__ __host__ bool Addition::positiveFirstNonZero() const {
    return values == 0 || (values & ~(values & (values - 1)) & ~signs);
}

__device__ __host__ int Addition::nonZeroCount() const {
#if defined(__CUDA_ARCH__)
    return __popcll(values);
#else
    return __builtin_popcountll(values);
#endif
}

std::ostream& operator<<(std::ostream &os, const Addition &addition) {
    for (int i = 0; i < addition.n; i++) {
        if (i > 0)
            os << ", ";

        os << addition[i];
    }

    return os;
}
