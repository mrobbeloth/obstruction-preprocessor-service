/* Kernel functions */

#include <stdio.h>
#include <cuda_runtime.h>

__global__ 
void multiplyByTen(const int *in, int *out) { *out = *in * 10; }