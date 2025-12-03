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
    RANDOM_MODE = 5,
    MIX_MODE = 6
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
    float intersectionsScale;

    __device__ __host__ void updateSubexpressions();
    __device__ __host__ void replaceSubexpression(const Pair &subexpression);
    __device__ __host__ void replaceExpression(int index, int i, int j, int varIndex);

    __device__ __host__ bool containsVariables(int exprIndex, int variable1, int variable2, int &index1, int &index2) const;
    __device__ Pair selectSubexpression(int mode, curandState &state) const;

    void showVariable(int variable, bool first) const;
public:
    __device__ __host__ AdditionsReducer();

    __device__ __host__ bool addExpression(int *values, int count);
    __device__ void reduce(curandState &state);
    __device__ __host__ void copyFrom(const AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount> &reducer);
    __device__ __host__ void clear();
    __device__ __host__ void setMode(int mode);

    __device__ __host__ int getAdditions() const;
    __device__ __host__ int getMaxRealVariables() const;
    __device__ __host__ int getNaiveAdditions() const;
    __device__ __host__ int getFreshVars() const;
    std::string getMode() const;

    void show() const;
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
    if (intersectionsScale == 0)
        intersectionsScale = curand_uniform(&state);

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
__device__ __host__ void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::clear() {
    expressionsCount = 0;
    realVariables = 0;
    freshVariables = 0;
    naiveAdditions = 0;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
__device__ __host__ void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::setMode(int mode) {
    this->mode = mode;
    intersectionsScale = 0;
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
        return "gr";

    if (mode == GREEDY_INTERSECTIONS_MODE)
        return "gi" + std::to_string(int(intersectionsScale * 100));

    if (mode == WEIGHTED_RANDOM_MODE)
        return "wr";

    if (mode == RANDOM_MODE)
        return "rnd";

    return "mix";
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::show() const {
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
    int i, j;

    for (int index = 0; index < expressionsCount; index++) {
        if (containsVariables(index, subexpression.i, subexpression.j, i, j)) {
            replaceExpression(index, i, j, varIndex);
        }
        else if (containsVariables(index, -subexpression.i, -subexpression.j, i, j)) {
            replaceExpression(index, i, j, -varIndex);
        }
    }

    variables[freshVariables][0] = subexpression.i;
    variables[freshVariables][1] = subexpression.j;
    freshVariables++;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
__device__ __host__ bool AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::containsVariables(int exprIndex, int variable1, int variable2, int &index1, int &index2) const {
    index1 = -1;
    index2 = -1;

    for (int i = 0; i < expressionSizes[exprIndex]; i++) {
        if (expressions[exprIndex][i] == variable1) {
            index1 = i;

            if (index2 != -1)
                return true;
        }
        else if (expressions[exprIndex][i] == variable2) {
            index2 = i;

            if (index1 != -1)
                return true;
        }
    }

    return false;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
__device__ Pair AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::selectSubexpression(int mode, curandState &state) const {
    if (mode == GREEDY_MODE)
        return subexpressions.getGreedy();

    if (mode == GREEDY_ALTERNATIVE_MODE)
        return subexpressions.getGreedyAlternative(state);

    if (mode == GREEDY_RANDOM_MODE)
        return subexpressions.getGreedyRandom(state);

    if (mode == GREEDY_INTERSECTIONS_MODE)
        return subexpressions.getGreedyIntersections(state, intersectionsScale);

    if (mode == WEIGHTED_RANDOM_MODE)
        return subexpressions.getWeightedRandom(state);

    return subexpressions.getRandom(state);
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
__device__ __host__ void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::replaceExpression(int index, int i, int j, int varIndex) {
    int last = --expressionSizes[index];
    expressions[index][i] = varIndex;
    expressions[index][j] = expressions[index][last];
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::showVariable(int variable, bool first) const {
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
