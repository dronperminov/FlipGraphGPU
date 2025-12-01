#pragma once

#include <iostream>
#include <string>

#include "../config.cuh"
#include "pairs_counter.cuh"

enum SelectSubexpressionMode {
    GREEDY_MODE = 0,
    GREEDY_RANDOM_MODE = 1,
    TOP_RANDOM_MODE = 2,
    WEIGHTED_RANDOM_MODE = 3,
    RANDOM_MODE = 4,
    SHUFFLE_MODE = 5
};

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength>
class AdditionsReducer {
    int expressions[maxExpressionsCount][maxExpressionLength];
    int expressionSizes[maxExpressionsCount];
    int variables[maxVariablesCount][2];
    PairsCounter<maxExpressionLength * (maxExpressionLength - 1) / 2> subexpressions;

    int expressionsCount;
    int realVariables;
    int freshVariables;
    int naiveAdditions;

    __device__ __host__ void updateSubexpressions();
    __device__ __host__ void replaceSubexpression(const Pair &subexpression);
    __device__ __host__ void replaceExpression(int index, int i, int j, int varIndex);

    __device__ __host__ bool containsVariable(int exprIndex, int variable, int &index) const;
    __device__ Pair selectSubexpression(int mode, curandState &state) const;

    void showVariable(int variable, bool first) const;
public:
    __device__ __host__ AdditionsReducer();

    __device__ __host__ bool addExpression(int *values, int count);
    __device__ void reduce(int mode, curandState &state);
    __device__ __host__ void copyFrom(const AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength> &reducer);
    __device__ __host__ void clear();

    __device__ __host__ int getAdditions() const;
    __device__ __host__ int getNaiveAdditions() const;
    __device__ __host__ int getFreshVars() const;

    void show() const;
    void write(std::ostream &os, const std::string &name, const std::string &indent) const;
};

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength>
__device__ __host__ AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength>::AdditionsReducer() {
    expressionsCount = 0;
    realVariables = 0;
    freshVariables = 0;
    naiveAdditions = 0;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength>
__device__ __host__ bool AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength>::addExpression(int *values, int count) {
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

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength>
__device__ void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength>::reduce(int mode, curandState &state) {
    while (freshVariables < maxVariablesCount) {
        updateSubexpressions();

        if (!subexpressions)
            break;

        int stepMode = mode == SHUFFLE_MODE ? curand(&state) % 5 : mode;
        Pair subexpression = selectSubexpression(stepMode, state);
        replaceSubexpression(subexpression);
    }
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength>
__device__ __host__ void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength>::copyFrom(const AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength> &reducer) {
    expressionsCount = reducer.expressionsCount;
    realVariables = reducer.realVariables;
    freshVariables = reducer.freshVariables;

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

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength>
__device__ __host__ void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength>::clear() {
    expressionsCount = 0;
    realVariables = 0;
    freshVariables = 0;
    naiveAdditions = 0;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength>
__device__ __host__ int AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength>::getAdditions() const {
    int additions = freshVariables;

    for (int i = 0; i < expressionsCount; i++)
        additions += expressionSizes[i] - 1;

    return additions;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength>
__device__ __host__ int AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength>::getNaiveAdditions() const {
    return naiveAdditions;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength>
__device__ __host__ int AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength>::getFreshVars() const {
    return freshVariables;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength>
void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength>::show() const {
    std::cout << "fresh vars: " << freshVariables << std::endl;
    for (int i = 0; i < freshVariables; i++) {
        std::cout << "t" << (i + 1) << " = ";
        showVariable(variables[i][0], true);
        showVariable(variables[i][1], false);
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "expressions: " << expressionsCount << std::endl;
    for (int index = 0; index < expressionsCount; index++) {
        for (int i = 0; i < expressionSizes[index]; i++)
            showVariable(expressions[index][i], i == 0);

        std::cout << std::endl;
    }
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength>
__device__ __host__ void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength>::updateSubexpressions() {
    subexpressions.clear();

    for (int index = 0; index < expressionsCount; index++)
        for (int i = 0; i < expressionSizes[index]; i++)
            for (int j = i + 1; j < expressionSizes[index]; j++)
                subexpressions.insert(expressions[index][i], expressions[index][j]);

    subexpressions.sort();
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength>
__device__ __host__ void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength>::replaceSubexpression(const Pair &subexpression) {
    int varIndex = realVariables + freshVariables + 1;
    int i, j;

    for (int index = 0; index < expressionsCount; index++) {
        if (containsVariable(index, subexpression.i, i) && containsVariable(index, subexpression.j, j)) {
            replaceExpression(index, i, j, varIndex);
        }
        else if (containsVariable(index, -subexpression.i, i) && containsVariable(index, -subexpression.j, j)) {
            replaceExpression(index, i, j, -varIndex);
        }
    }

    variables[freshVariables][0] = subexpression.i;
    variables[freshVariables][1] = subexpression.j;
    freshVariables++;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength>
__device__ __host__ bool AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength>::containsVariable(int exprIndex, int variable, int &index) const {
    for (int i = 0; i < expressionSizes[exprIndex]; i++) {
        if (expressions[exprIndex][i] == variable) {
            index = i;
            return true;
        }
    }

    return false;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength>
__device__ Pair AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength>::selectSubexpression(int mode, curandState &state) const {
    if (mode == GREEDY_MODE)
        return subexpressions.getTop();

    if (mode == GREEDY_RANDOM_MODE)
        return subexpressions.getGreedyRandom(state);

    if (mode == TOP_RANDOM_MODE)
        return subexpressions.getTopRandom(state);

    if (mode == WEIGHTED_RANDOM_MODE)
        return subexpressions.getWeightedRandom(state);

    return subexpressions.getRandom(state);
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength>
__device__ __host__ void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength>::replaceExpression(int index, int i, int j, int varIndex) {
    int last = --expressionSizes[index];
    expressions[index][i] = varIndex;
    expressions[index][j] = expressions[index][last];
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength>
void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength>::showVariable(int variable, bool first) const {
    if (variable < 0)
        std::cout << "- ";
    else if (!first)
        std::cout << "+ ";

    int index = abs(variable);

    if (index <= realVariables)
        std::cout << "x" << index << " ";
    else
        std::cout << "t" << (index - realVariables) << " ";
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength>
void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength>::write(std::ostream &os, const std::string &name, const std::string &indent) const {
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
