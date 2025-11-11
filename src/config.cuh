#pragma once

#define SCHEME_INTEGER

typedef uint16_t T;

const int MAX_RANK = 64;
const int MAX_MATRIX_SIZE = 4;
const int MAX_MATRIX_ELEMENTS = MAX_MATRIX_SIZE * MAX_MATRIX_SIZE;
const int MAX_SIZE = 8 * sizeof(T);

#ifdef SCHEME_INTEGER
#define Scheme SchemeInteger
#else
#define Scheme SchemeZ2
#endif
