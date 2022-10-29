#pragma once
#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/**
 * CUDA handle error, if error occurs print message and exit program
*
* @param error: CUDA error status
*/
#define HANDLE_ERROR(error) { \
    if (error != cudaSuccess) { \
        fprintf(stderr, "%s in %s at line %d\n", \
                cudaGetErrorString(error), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} \

template <class T> static __global__
void printDevice(T* _data, size_t cnt) {
    for (int idx = 0; idx < cnt; idx++)
        printf("%f ", float(*(_data + idx)));
    printf("\n");
}

static __global__
void selfDivide(float* val1, int* val2) {
    val1[0] = val1[0] / (float)(*val2);
}

static __global__
void mod(int* op1, int* op2, int* quotient, int* remainder) {
    *quotient = *op1 / *op2;
    *remainder = *op1 % *op2;
}

void ArrayReduceSum(const float* input, float* output, size_t n);
void ArrayReduceSum(const int* input, int* output, size_t n);
void ArrayArgmax(const float* input, float* max_val, int* max_idx, size_t n, float trunc = -1.);
void ArrayArgmin(const float* input, float* min_val, int* min_idx, size_t n);