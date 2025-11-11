#pragma once

#define SCHEME_INTEGER

const int MAX_RANK = 64;
const int MAX_MATRIX_SIZE = 4;
const int MAX_MATRIX_ELEMENTS = MAX_MATRIX_SIZE * MAX_MATRIX_SIZE;

typedef uint16_t T;

#ifdef SCHEME_INTEGER
#define Scheme SchemeInteger
const int MAX_SIZE = MAX_MATRIX_ELEMENTS;
#else
#define Scheme SchemeZ2
const int MAX_SIZE = 8 * sizeof(T);
#endif
