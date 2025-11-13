#pragma once

#include <iostream>
#include <algorithm>

#define SCHEME_INTEGER

typedef uint16_t T;

const int MAX_RANK = 64;
const int MAX_MATRIX_ELEMENTS = std::min(int(sizeof(T) * 8), 16);

// project limit
const int MIN_PROJECT_N1 = 2;
const int MIN_PROJECT_N2 = 2;
const int MIN_PROJECT_N3 = 2;

// extension limit
const int MAX_EXTENSION_N1 = 8;
const int MAX_EXTENSION_N2 = 8;
const int MAX_EXTENSION_N3 = 8;

// sandwiching limit
const int MAX_SANDWICHING_N = std::max(std::max(MAX_EXTENSION_N1, MAX_EXTENSION_N2), MAX_EXTENSION_N3);
const int MAX_SANDWICHING_ELEMENTS = MAX_SANDWICHING_N * MAX_SANDWICHING_N;
