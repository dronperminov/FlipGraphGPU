#pragma once

#include <iostream>
#include <algorithm>

#define SCHEME_INTEGER

typedef uint64_t T;

const int MAX_RANK = 350;
const int MAX_MATRIX_ELEMENTS = std::min(int(sizeof(T) * 8), 64);

// FlipSet capacity (for 16 elements - 112, 32 elements - 448, 64 elements - 1792)
const int MAX_PAIRS = std::min(MAX_RANK * (MAX_RANK - 1) / 2, 500);

// additions reducer capacity
const int MAX_U_EXPRESSIONS = MAX_RANK;
const int MAX_V_EXPRESSIONS = MAX_RANK;
const int MAX_W_EXPRESSIONS = MAX_MATRIX_ELEMENTS;

const int MAX_U_REAL_VARIABLES = MAX_MATRIX_ELEMENTS / 2;
const int MAX_V_REAL_VARIABLES = MAX_MATRIX_ELEMENTS / 2;
const int MAX_W_REAL_VARIABLES = MAX_RANK / 2;

const int MAX_U_SUBEXPRESSIONS = MAX_MATRIX_ELEMENTS * (MAX_MATRIX_ELEMENTS - 1) / 2;
const int MAX_V_SUBEXPRESSIONS = MAX_MATRIX_ELEMENTS * (MAX_MATRIX_ELEMENTS - 1) / 2;
const int MAX_W_SUBEXPRESSIONS = MAX_RANK * (MAX_RANK - 1) / 2;

const int MAX_U_FRESH_VARIABLES = 250;
const int MAX_V_FRESH_VARIABLES = 250;
const int MAX_W_FRESH_VARIABLES = 500;

// project limit
const int MIN_PROJECT_N = 2;

// extension limit
const int MAX_EXTENSION_N = 16;

// sandwiching limit
const int MAX_SANDWICHING_N = 16;
const int MAX_SANDWICHING_ELEMENTS = MAX_SANDWICHING_N * MAX_SANDWICHING_N;

#ifdef SCHEME_INTEGER
#define Scheme SchemeInteger
const std::string ring = "ZT";
#else
#define Scheme SchemeZ2
const std::string ring = "Z2";
#endif
