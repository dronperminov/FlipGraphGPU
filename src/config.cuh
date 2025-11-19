#pragma once

#include <iostream>
#include <algorithm>

#define SCHEME_INTEGER

typedef uint64_t T;

const int MAX_RANK = 292;
const int MAX_MATRIX_ELEMENTS = std::min(int(sizeof(T) * 8), 64);

// project limit
const int MIN_PROJECT_N1 = 2;
const int MIN_PROJECT_N2 = 2;
const int MIN_PROJECT_N3 = 3;

// extension limit
const int MAX_EXTENSION_N1 = 10;
const int MAX_EXTENSION_N2 = 10;
const int MAX_EXTENSION_N3 = 10;

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
