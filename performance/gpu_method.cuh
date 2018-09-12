#ifndef GPU_METHOD_H
#define GPU_METHOD_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <cufft.h>

#include <iostream>

#include "typedefs.h"
#include "log.h"

static __device__ __host__ inline GPU_COMPLEX _gpu_complex_scale(GPU_COMPLEX a, double s);

static __global__ void _gpu_vector_scale(GPU_COMPLEX *a, int size, double scale);

Result _gpu_init(cublasStatus_t* cublas_status, cublasHandle_t* cublas_handle);

Result _gpu_allocate_data(GPU_COMPLEX** device_data, size_t size);

Result _gpu_copy_to_device_data(F2C_COMPLEX* data, GPU_COMPLEX* device_data, size_t size);

Result _gpu_copy_to_host_data(GPU_COMPLEX* device_data, F2C_COMPLEX* data, size_t size);

Result _gpu_free(GPU_COMPLEX* device_data);

Result _gpu_preprocess_method_part1(GPU_COMPLEX* device_out_data, GPU_COMPLEX* device_matrices_data, GPU_COMPLEX* device_c_vector_data, int image_number, int chunk_size,cublasHandle_t* cublas_handle);

Result _gpu_preprocess_method_part2(GPU_COMPLEX* device_out_data, GPU_COMPLEX* device_matrices_data, GPU_COMPLEX* device_c_vector_data, int image_number, int chunk_size, int iteration_number, cublasHandle_t* cublas_handle);



#endif // GPU_METHOD_H