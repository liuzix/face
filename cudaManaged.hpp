#ifndef CUDAMANAGED_H
#define CUDAMANAGED_H

#include <stddef.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <exception>

#define CHECK(r) {_check((r), __LINE__);}

inline void _check(cudaError_t r, int line) {
  if (r != cudaSuccess) {
    printf("CUDA error on line %d: %s, line %d\n", line, cudaGetErrorString(r), line);
    throw std::exception();
  }
}

/* This class provides for unified memory management */
class Managed {
public:
  void *operator new(size_t len) {
    void *ptr;
    CHECK(cudaMallocManaged(&ptr, len));
    cudaDeviceSynchronize();
    return ptr;
  }

  void operator delete(void *ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
  }

  void *operator new[](size_t len) {
    void *ptr;
    CHECK(cudaMallocManaged(&ptr, len));
    cudaDeviceSynchronize();
    return ptr;
  }

  void operator delete[](void *ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
  }
};



#endif