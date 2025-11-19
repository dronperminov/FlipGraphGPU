#pragma once

#include <iostream>
#include <algorithm>

#define SCHEME_INTEGER

typedef uint64_t T;

const int MAX_RANK = 270;
const int MAX_MATRIX_ELEMENTS = std::min(int(sizeof(T) * 8), 64);

// FlipSet capacity (for 16 elements - 112, 32 elements - 448, 64 elements - 1792)
const int MAX_PAIRS = std::min(MAX_RANK * (MAX_RANK - 1) / 2, 1800);

// project limit
const int MIN_PROJECT_N = 2;

// extension limit
const int MAX_EXTENSION_N = 10;

// sandwiching limit
const int MAX_SANDWICHING_N = 10;
const int MAX_SANDWICHING_ELEMENTS = MAX_SANDWICHING_N * MAX_SANDWICHING_N;

#ifdef SCHEME_INTEGER
#define Scheme SchemeInteger
const std::string ring = "ZT";
#else
#define Scheme SchemeZ2
const std::string ring = "Z2";
#endif
