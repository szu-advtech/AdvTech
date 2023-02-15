//
// Created by will on 20-1-9.
//

#ifndef TSDF_UTILS_H
#define TSDF_UTILS_H

#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <cmath>

/**
 * @brief Check CUDA kenerl function
 *
 * @input:cudaGetLastError()
 */
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char* const func, const char* const file, const int linenumber)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error at: " << file << " : " << linenumber << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

template<typename T>
void checkResultsExact(const T* const ref, const T* const gpu, size_t numElem) {
  //check that the GPU result matches the CPU result
  for (size_t i = 0; i < numElem; ++i) {
    if (ref[i] != gpu[i]) {
      std::cerr << "Difference at pos " << i << std::endl;
      //the + is magic to convert char to int without messing
      //with other types
      std::cerr << "Reference: " << std::setprecision(17) << +ref[i] <<
                 "\nGPU      : " << +gpu[i] << std::endl;
      exit(1);
    }
  }
}

#endif //TSDF_UTILS_H
