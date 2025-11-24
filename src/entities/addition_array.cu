#include "addition_array.cuh"

__device__ __host__ AdditionArray::AdditionArray() {
    n = 0;
    valid = true;
}

__device__ __host__ AdditionArray::AdditionArray(int n) {
    this->n = n;

    for (int i = 0; i < n; i++)
        this->values[i] = 0;

    this->valid = true;
}

__device__ __host__ AdditionArray::AdditionArray(int n, int index) {
    this->n = n;

    for (int i = 0; i < n; i++)
        this->values[i] = i == index ? 1 : 0;

    this->valid = true;
}

__device__ __host__ AdditionArray::AdditionArray(int n, int *values) {
    this->n = n;
    this->valid = true;

    for (int i = 0; i < n; i++)
        this->values[i] = values[i];
}

__device__ __host__ void AdditionArray::copyTo(AdditionArray &target) const {
    target.n = n;

    for (int i = 0; i < n; i++)
        target.values[i] = values[i];

    target.valid = valid;
}

__device__ __host__ void AdditionArray::set(int index, int value) {
    values[index] = value;
    valid = ADDITION_LOWER_BOUND <= value && value <= ADDITION_UPPER_BOUND;
}

__device__ __host__ void AdditionArray::inverse() {
    for (int i = 0; i < n; i++)
        values[i] = -values[i];
}

__device__ void AdditionArray::random(curandState &state) {
    for (int i = 0; i < n; i++)
        values[i] = randint(ADDITION_LOWER_BOUND, ADDITION_UPPER_BOUND, state);

    valid = true;
}

__device__ __host__ int AdditionArray::nonZeroCount() const {
    int count = 0;

    for (int i = 0; i < n; i++)
        if (values[i] != 0)
            count++;

    return count;
}

__device__ __host__ int AdditionArray::compare(const AdditionArray &addition) const {
    bool pos = true;
    bool neg = true;

    for (int i = 0; i < n && (pos || neg); i++) {
        pos &= values[i] == addition.values[i];
        neg &= values[i] == -addition.values[i];
    }

    if (pos)
        return 1;

    if (neg)
        return -1;

    return 0;
}

__device__ __host__ bool AdditionArray::operator==(const AdditionArray &addition) const {
    for (int i = 0; i < n; i++)
        if (values[i] != addition.values[i])
            return false;

    return true;
}

__device__ __host__ bool AdditionArray::operator!=(const AdditionArray &addition) const {
    for (int i = 0; i < n; i++)
        if (values[i] != addition.values[i])
            return true;

    return false;
}

__device__ __host__ int AdditionArray::operator[](int index) const {
    return values[index];
}

__device__ __host__ AdditionArray AdditionArray::operator+(const AdditionArray &addition) const {
    AdditionArray result(n);

    for (int i = 0; i < n; i++) {
        result.values[i] = values[i] + addition.values[i];

        if (result.values[i] < ADDITION_LOWER_BOUND || result.values[i] > ADDITION_UPPER_BOUND)
            result.valid = false;
    }

    return result;
}

__device__ __host__ AdditionArray AdditionArray::operator-(const AdditionArray &addition) const {
    AdditionArray result(n);

    for (int i = 0; i < n; i++) {
        result.values[i] = values[i] - addition.values[i];

        if (result.values[i] < ADDITION_LOWER_BOUND || result.values[i] > ADDITION_UPPER_BOUND)
            result.valid = false;
    }

    return result;
}

__device__ __host__ AdditionArray AdditionArray::operator-() const {
    AdditionArray result(n);

    for (int i = 0; i < n; i++)
        result.values[i] = -values[i];

    result.valid = valid;
    return result;
}

__device__ __host__ AdditionArray& AdditionArray::operator+=(const AdditionArray &addition) {
    for (int i = 0; i < n; i++) {
        values[i] += addition.values[i];

        if (values[i] < ADDITION_LOWER_BOUND || values[i] > ADDITION_UPPER_BOUND)
            valid = false;
    }

    return *this;
}

__device__ __host__ AdditionArray& AdditionArray::operator-=(const AdditionArray &addition) {
    for (int i = 0; i < n; i++) {
        values[i] -= addition.values[i];

        if (values[i] < ADDITION_LOWER_BOUND || values[i] > ADDITION_UPPER_BOUND)
            valid = false;
    }

    return *this;
}

__device__ __host__ AdditionArray::operator bool() const {
    for (int i = 0; i < n; i++)
        if (values[i])
            return true;

    return false;
}

__device__ __host__ bool AdditionArray::limit(bool firstPositiveNonZero) const {
    if (!valid)
        return false;

    if (firstPositiveNonZero)
        for (int i = 0; i < n; i++)
            if (values[i] != 0)
                return values[i] > 0;

    return true;
}

__device__ __host__ bool AdditionArray::limitSum(const AdditionArray &addition, bool firstPositiveNonZero) const {
    bool haveNonZero = false;

    for (int i = 0; i < n; i++) {
        int8_t value = values[i] + addition.values[i];

        if (value < ADDITION_LOWER_BOUND || value > ADDITION_UPPER_BOUND)
            return false;

        if (firstPositiveNonZero && !haveNonZero && value != 0) {
            haveNonZero = true;

            if (value < 0)
                return false;
        }
    }

    return true;
}

__device__ __host__ bool AdditionArray::limitSub(const AdditionArray &addition, bool firstPositiveNonZero) const {
    bool haveNonZero = false;

    for (int i = 0; i < n; i++) {
        int8_t value = values[i] - addition.values[i];

        if (value < ADDITION_LOWER_BOUND || value > ADDITION_UPPER_BOUND)
            return false;

        if (firstPositiveNonZero && !haveNonZero && value != 0) {
            haveNonZero = true;

            if (value < 0)
                return false;
        }
    }

    return true;
}

__device__ __host__ bool AdditionArray::positiveFirstNonZeroSub(const AdditionArray &addition) const {
    for (int i = 0; i < n; i++) {
        int8_t value = values[i] - addition.values[i];

        if (value != 0)
            return value > 0;
    }

    return true;
}

__device__ __host__ bool AdditionArray::positiveFirstNonZero() const {
    for (int i = 0; i < n; i++)
        if (values[i] != 0)
            return values[i] > 0;

    return true;
}

std::ostream& operator<<(std::ostream &os, const AdditionArray &addition) {
    for (int i = 0; i < addition.n; i++) {
        if (i > 0)
            os << ", ";

        int value = addition.values[i];
        os << value;
    }

    return os;
}
