#pragma once

#include <iostream>
#include <string>

#include "../config.cuh"
#include "pairs_counter.cuh"

enum SelectSubexpressionMode {
    GREEDY_MODE = 0,
    GREEDY_ALTERNATIVE_MODE = 1,
    GREEDY_RANDOM_MODE = 2,
    GREEDY_INTERSECTIONS_MODE = 3,
    WEIGHTED_RANDOM_MODE = 4,
    MIX_MODE = 5,
    RANDOM_MODE = 6,
};

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
class AdditionsReducer {
    int expressions[maxExpressionsCount][maxExpressionLength];
    int expressionSizes[maxExpressionsCount];
    int variables[maxVariablesCount][2];
    PairsCounter<maxSubexpressionsCount> subexpressions;

    int expressionsCount;
    int realVariables;
    int freshVariables;
    int naiveAdditions;
    int mode;
    float scale;

    __device__ __host__ void updateSubexpressions();
    __device__ __host__ void replaceSubexpression(const Pair &subexpression);
    __device__ __host__ void replaceExpression(int *expression, int &size, int index1, int index2, int varIndex);

    __device__ __host__ int binarySearch(const int *expression, int size, int value, int start) const;
    __device__ Pair selectSubexpression(int mode, curandState &state) const;
public:
    __device__ __host__ AdditionsReducer();

    __device__ __host__ bool addExpression(int *values, int count);
    __device__ void reduce(curandState &state);
    __device__ __host__ void copyFrom(const AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount> &reducer);
    __device__ __host__ void partialInitialize(const AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount> &reducer, int copy);
    __device__ __host__ void clear();
    __device__ __host__ void setMode(int mode);

    __device__ __host__ int getAdditions() const;
    __device__ __host__ int getMaxRealVariables() const;
    __device__ __host__ int getNaiveAdditions() const;
    __device__ __host__ int getFreshVars() const;
    std::string getMode() const;

    void write(std::ostream &os, const std::string &name, const std::string &indent) const;
};

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
__device__ __host__ AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::AdditionsReducer() {
    expressionsCount = 0;
    realVariables = 0;
    freshVariables = 0;
    naiveAdditions = 0;
    mode = GREEDY_MODE;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
__device__ __host__ bool AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::addExpression(int *values, int count) {
    int size = 0;

    for (int i = 0; i < count; i++) {
        if (values[i] == 1)
            expressions[expressionsCount][size++] = i + 1;
        else if (values[i] == -1)
            expressions[expressionsCount][size++] = -(i + 1);
        else if (values[i] != 0)
            return false;
    }

    if (count > realVariables)
        realVariables = count;

    expressionSizes[expressionsCount++] = size;
    naiveAdditions += size - 1;
    return true;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
__device__ void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::reduce(curandState &state) {
    if (scale < 0)
        scale = curand_uniform(&state) / 2;

    while (freshVariables < maxVariablesCount) {
        updateSubexpressions();

        if (!subexpressions)
            break;

        int stepMode = mode == MIX_MODE ? curand(&state) % MIX_MODE : mode;
        Pair subexpression = selectSubexpression(stepMode, state);
        replaceSubexpression(subexpression);
    }
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
__device__ __host__ void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::copyFrom(const AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount> &reducer) {
    expressionsCount = reducer.expressionsCount;
    realVariables = reducer.realVariables;
    freshVariables = reducer.freshVariables;
    mode = reducer.mode;
    scale = reducer.scale;

    for (int index = 0; index < expressionsCount; index++) {
        expressionSizes[index] = reducer.expressionSizes[index];

        for (int i = 0; i < expressionSizes[index]; i++)
            expressions[index][i] = reducer.expressions[index][i];
    }

    for (int i = 0; i < freshVariables; i++) {
        variables[i][0] = reducer.variables[i][0];
        variables[i][1] = reducer.variables[i][1];
    }

    subexpressions.copyFrom(reducer.subexpressions);
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
__device__ __host__ void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::partialInitialize(const AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount> &reducer, int count) {
    for (int index = 0; index < count && index < reducer.freshVariables; index++) {
        int i = reducer.variables[index][0];
        int j = reducer.variables[index][1];

        replaceSubexpression({i, j, 0});
    }
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
__device__ __host__ void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::clear() {
    expressionsCount = 0;
    realVariables = 0;
    freshVariables = 0;
    naiveAdditions = 0;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
__device__ __host__ void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::setMode(int mode) {
    this->mode = mode;
    scale = -1;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
__device__ __host__ int AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::getAdditions() const {
    int additions = freshVariables;

    for (int i = 0; i < expressionsCount; i++)
        additions += expressionSizes[i] - 1;

    return additions;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
__device__ __host__ int AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::getMaxRealVariables() const {
    int variables = 0;

    for (int i = 0; i < expressionsCount; i++)
        if (expressionSizes[i] > variables)
            variables = expressionSizes[i];

    return variables;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
__device__ __host__ int AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::getNaiveAdditions() const {
    return naiveAdditions;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
__device__ __host__ int AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::getFreshVars() const {
    return freshVariables;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
std::string AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::getMode() const {
    if (mode == GREEDY_MODE)
        return "g";

    if (mode == GREEDY_ALTERNATIVE_MODE)
        return "ga";

    if (mode == GREEDY_RANDOM_MODE)
        return "gr" + std::to_string(int(scale * 100));

    if (mode == GREEDY_INTERSECTIONS_MODE)
        return "gi" + std::to_string(int(scale * 100));

    if (mode == WEIGHTED_RANDOM_MODE)
        return "wr";

    if (mode == RANDOM_MODE)
        return "rnd";

    return "mix";
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
__device__ __host__ void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::updateSubexpressions() {
    subexpressions.clear();

    for (int index = 0; index < expressionsCount; index++)
        for (int i = 0; i < expressionSizes[index]; i++)
            for (int j = i + 1; j < expressionSizes[index]; j++)
                subexpressions.insert(expressions[index][i], expressions[index][j]);

    subexpressions.sort();
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
__device__ __host__ void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::replaceSubexpression(const Pair &subexpression) {
    int varIndex = realVariables + freshVariables + 1;

    for (int index = 0; index < expressionsCount; index++) {
        if (expressionSizes[index] < 2)
            continue;

        int i = binarySearch(expressions[index], expressionSizes[index], subexpression.i, 0);
        if (i == -1)
            continue;

        int j = binarySearch(expressions[index], expressionSizes[index], subexpression.j, i + 1);
        if (j == -1)
            continue;

        if (expressions[index][i] == subexpression.i && expressions[index][j] == subexpression.j) {
            replaceExpression(expressions[index], expressionSizes[index], i, j, varIndex);
        }
        else if (expressions[index][i] == -subexpression.i && expressions[index][j] == -subexpression.j) {
            replaceExpression(expressions[index], expressionSizes[index], i, j, -varIndex);
        }
    }

    variables[freshVariables][0] = subexpression.i;
    variables[freshVariables][1] = subexpression.j;
    freshVariables++;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
__device__ __host__ int AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::binarySearch(const int *expression, int size, int value, int start) const {
    int low = start;
    int high = size - 1;
    int valueAbs = abs(value);

    while (low <= high) {
        int mid = low + (high - low) / 2;
        int midAbs = abs(expression[mid]);

        if (midAbs < valueAbs) {
            low = mid + 1;
        }
        else if (midAbs > valueAbs) {
            high = mid - 1;
        }
        else {
            return mid;
        }
    }

    return -1;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
__device__ Pair AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::selectSubexpression(int mode, curandState &state) const {
    if (mode == GREEDY_MODE)
        return subexpressions.getGreedy();

    if (mode == GREEDY_ALTERNATIVE_MODE)
        return subexpressions.getGreedyAlternative(state);

    if (mode == GREEDY_RANDOM_MODE)
        return subexpressions.getGreedyRandom(state, scale);

    if (mode == GREEDY_INTERSECTIONS_MODE)
        return subexpressions.getGreedyIntersections(state, scale);

    if (mode == WEIGHTED_RANDOM_MODE)
        return subexpressions.getWeightedRandom(state);

    return subexpressions.getRandom(state);
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
__device__ __host__ void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::replaceExpression(int *expression, int &size, int index1, int index2, int varIndex) {
    int j = index1;

    for (int i = index1 + 1; i < size; i++)
        if (i != index2)
            expression[j++] = expression[i];

    size--;
    expression[j] = varIndex;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::write(std::ostream &os, const std::string &name, const std::string &indent) const {
    os << indent << "\"" << name << "_fresh\": [" << std::endl;

    for (int i = 0; i < freshVariables; i++) {
        int index1 = abs(variables[i][0]) - 1;
        int value1 = variables[i][0] > 0 ? 1 : -1;

        int index2 = abs(variables[i][1]) - 1;
        int value2 = variables[i][1] > 0 ? 1 : -1;

        os << indent << indent << "[{\"index\": " << index1 << ", \"value\": " << value1 << "}, {\"index\": " << index2 << ", \"value\": " << value2 << "}]";

        if (i < freshVariables - 1)
            os << ",";

        os << std::endl;
    }

    os << indent << "]," << std::endl;
    os << indent << "\"" << name << "\": [" << std::endl;

    for (int i = 0; i < expressionsCount; i++) {
        os << indent << indent << "[";

        for (int j = 0; j < expressionSizes[i]; j++) {
            int index = abs(expressions[i][j]) - 1;
            int value = expressions[i][j] > 0 ? 1 : -1;

            os << "{\"index\": " << index << ", \"value\": " << value << "}";

            if (j < expressionSizes[i] - 1)
                os << ", ";
        }

        os << "]";

        if (i < expressionsCount - 1)
            os << ",";

        os << std::endl;
    }

    os << indent << "]";
}
