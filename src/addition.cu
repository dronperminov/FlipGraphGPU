#include "addition.cuh"

__device__ __host__ Addition::Addition() {
    n = 0;
    values = 0;
    signs = 0;
    carry = 0;
}

__device__ __host__ Addition::Addition(int n) {
    this->n = n;
    this->values = 0;
    this->signs = 0;
    this->carry = 0;
}

__device__ __host__ Addition::Addition(int n, int index) {
    this->n = n;
    this->values = AdditionT(1) << index;
    this->signs = 0;
    this->carry = 0;
}

__device__ __host__ Addition::Addition(int n, int *values) {
    this->n = n;
    this->values = 0;
    this->carry = 0;
    this->signs = 0;

    for (int i = 0; i < n; i++)
        set(i, values[i]);
}

__device__ __host__ void Addition::copyTo(Addition &target) const {
    target.n = n;
    target.values = values;
    target.signs = signs;
    target.carry = carry;
}

__device__ __host__ void Addition::set(int index, int value) {
    AdditionT mask = AdditionT(1) << index;
    carry &= ~mask;

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
        printf("invalid set (%d, %d)\n", index, value);
    }
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
    result.carry = values & addition.values & ~(signs ^ addition.signs);
    return result;
}

__device__ __host__ Addition Addition::operator-(const Addition &addition) const {
    Addition result(n);

    result.values = values ^ addition.values;
    result.signs = ((signs & values) | (~addition.signs & addition.values)) & result.values;
    result.carry = values & addition.values & (signs ^ addition.signs);
    return result;
}

__device__ __host__ Addition& Addition::operator+=(const Addition &addition) {
    AdditionT sv1 = signs & values;
    AdditionT sv2 = addition.signs & addition.values;

    carry = values & addition.values & ~(signs ^ addition.signs);
    values ^= addition.values;
    signs = (sv1 | sv2) & values;
    return *this;
}

__device__ __host__ Addition& Addition::operator-=(const Addition &addition) {
    AdditionT sv1 = signs & values;
    AdditionT sv2 = ~addition.signs & addition.values;

    carry = values & addition.values & (signs ^ addition.signs);
    values ^= addition.values;
    signs = (sv1 | sv2) & values;
    return *this;
}

__device__ __host__ Addition::operator bool() const {
    return values != 0;
}

__device__ __host__ bool Addition::limit(bool firstPositiveNonZero) const {
    if (carry)
        return false;

    for (int i = 0; i < n; i++)
        if ((values >> i) & 1)
            return ((signs >> i) & 1) == 0;

    return true;
}

__device__ __host__ bool Addition::limitSum(const Addition &addition, bool firstPositiveNonZero) const {
    AdditionT sumCarry = values & addition.values & ~(signs ^ addition.signs);

    if (sumCarry)
        return false;

    if (firstPositiveNonZero) {
        AdditionT sumValues = values ^ addition.values;
        AdditionT sumSigns = ((signs & values) | (addition.signs & addition.values)) & sumValues;

        for (int i = 0; i < n; i++)
            if ((sumValues >> i) & 1)
                return ((sumSigns >> i) & 1) == 0;
    }

    return true;
}

__device__ __host__ bool Addition::limitSub(const Addition &addition, bool firstPositiveNonZero) const {
    AdditionT subCarry = (values & addition.values & (signs ^ addition.signs));

    if (subCarry)
        return false;

    if (firstPositiveNonZero) {
        AdditionT subValues = values ^ addition.values;
        AdditionT subSigns = ((signs & values) | (~addition.signs & addition.values)) & subValues;

        for (int i = 0; i < n; i++)
            if ((subValues >> i) & 1)
                return ((subSigns >> i) & 1) == 0;
    }

    return true;
}

__device__ __host__ bool Addition::positiveFirstNonZeroSub(const Addition &addition) const {
    AdditionT subValues = values ^ addition.values;
    AdditionT subSigns = ((signs & values) | (~addition.signs & addition.values)) & subValues;

    for (int i = 0; i < n; i++)
        if ((subValues >> i) & 1)
            return ((subSigns >> i) & 1) == 0;

    return true;
}

std::ostream& operator<<(std::ostream &os, const Addition &addition) {
    for (int i = 0; i < addition.n; i++) {
        if (i > 0)
            os << ", ";

        os << addition[i];
    }

    return os;
}
