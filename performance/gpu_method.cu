#include "gpu_method.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <cufft.h>

#include <iostream>

#include "typedefs.h"
#include "log.h"


static __device__ __host__ inline GPU_COMPLEX _gpu_complex_scale(GPU_COMPLEX a, double s)
{
	GPU_COMPLEX c;
	c.x = s * a.x;
	c.y = s * a.y;
	return c;
}

static __global__ void _gpu_vector_scale(GPU_COMPLEX *a, int size, double scale)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = threadID; i < size; i += numThreads)
	{
		a[i] = _gpu_complex_scale(a[i], scale);
	}
}

Result _gpu_init(cublasStatus_t* cublas_status, cublasHandle_t* cublas_handle)
{
	*cublas_status = cublasCreate(cublas_handle);
	if (*cublas_status != CUBLAS_STATUS_SUCCESS){
		log_message("Cublas initialization failed.");
		return Result::CUDABLASInitializationError;
	} 

	return Result::Success;
}

Result _gpu_allocate_data(GPU_COMPLEX** device_data, size_t size)
{
	cudaError_t result = cudaMalloc((void**)device_data, sizeof(GPU_COMPLEX) * size);
	if (result != cudaSuccess){
		log_message("GPU memory allocation for out images failed.");
		return Result::CUDAMemoryAllocationError;
	}

	return Result::Success;
}

Result _gpu_free(GPU_COMPLEX* device_data)
{
	cudaError_t cuda_result = cudaFree((void*)device_data);
	if (cuda_result != cudaSuccess){
		log_message("Free CUDA memory for matrices failed.");
		return Result::CUDAMemoryAllocationError;
	}

	return Result::Success;
}

Result _gpu_copy_to_device_data(F2C_COMPLEX* data, GPU_COMPLEX* device_data, size_t size)
{
	cudaError_t	result = cudaMemcpy(device_data, data, (size_t)(size * sizeof(F2C_COMPLEX)), cudaMemcpyHostToDevice);
	if (result != cudaSuccess){
		log_message("HostToDevice memcpy for out images failed.");
		return Result::CUDAMemoryCopyError;
	}

	return Result::Success;
}

Result _gpu_copy_to_host_data(GPU_COMPLEX* device_data, F2C_COMPLEX* data, size_t size)
{
	cudaError_t	result = cudaMemcpy(data, device_data, (size_t)(size * sizeof(GPU_COMPLEX)), cudaMemcpyDeviceToHost);
	if (result != cudaSuccess){
		log_message("HostToDevice memcpy for out images failed.");
		return Result::CUDAMemoryCopyError;
	}

	return Result::Success;
}



Result _gpu_preprocess_method_part1(GPU_COMPLEX* device_out_data, GPU_COMPLEX* device_matrices_data, GPU_COMPLEX* device_c_vector_data, int image_number, int chunk_size,cublasHandle_t* cublas_handle)
{
	unsigned int s = chunk_size;
	int m = image_number;
	
	// constants for cublas*gemmStridedBatched method
	GPU_COMPLEX alpha, beta;
	alpha.x = 1.0;
	alpha.y = 0.0;
	beta.x = 0.0;
	beta.y = 0.0;

	// Calculating c = mu (E + mu A* A)^{-1} A* b
#ifdef DOUBLEPRECISION
		cublasStatus_t cublas_status = cublasZgemmStridedBatched(*cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, 1, m, &alpha, device_matrices_data, m, m*m, device_out_data, m, m, &beta, device_c_vector_data, m, m, s);
#else
		cublasStatus_t cublas_status = cublasCgemmStridedBatched(*cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, 1, m, &alpha, device_matrices_data, m, m*m, device_out_data, m, m, &beta, device_c_vector_data, m, m, s);
#endif
		if (cublas_status != CUBLAS_STATUS_SUCCESS){
			log_message("Strided Batched Gemm failed.");
			return Result::CUDABLASInitializationError;
		}

		// now in device_c_vectors data 'c = mu (E + mu A* A)^{-1} A* b'

		return Result::Success;
}

Result _gpu_preprocess_method_part2(GPU_COMPLEX* device_out_data, GPU_COMPLEX* device_matrices_data, GPU_COMPLEX* device_c_vector_data, int image_number, int chunk_size, int iteration_number, cublasHandle_t* cublas_handle)
{
	unsigned int s = chunk_size;
	int m = image_number;

	size_t x_np1_size = s * m * sizeof(GPU_COMPLEX);

	// constants for cublas*gemmStridedBatched method
	GPU_COMPLEX alpha, beta;
	alpha.x = 1.0;
	alpha.y = 0.0;
	beta.x = 0.0;
	beta.y = 0.0;

	cublasStatus_t cublas_status;
	// allocating GPU memory for the vector 'x_(n+1)'
	GPU_COMPLEX* device_xnp1 = NULL;
	GPU_COMPLEX* device_xn = device_out_data;
	GPU_COMPLEX* temp;
	Result gpu_result = _gpu_allocate_data(&device_xnp1, s * m);
	_react_on(gpu_result);

	// creating zero vector as 'x_0'(device_xn)
	_gpu_vector_scale << <32, 256 >> >(device_out_data, s * m, 0.0);
	beta.x = 1.0;

	for (int n = 0; n < iteration_number; n++){

		// *1: x_np1 becoming 'c'
		// 'c' -> 'x_(n+1)'
		cudaMemcpy(device_xnp1, device_c_vector_data, x_np1_size, cudaMemcpyDeviceToDevice);	
	
#ifdef DOUBLEPRECISION
		cublas_status = cublasZgemmStridedBatched(*cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, 1, m, &alpha, device_matrices_data, m, m*m, device_xn, m, m, &beta, device_xnp1, m, m, s);
#else
		cublas_status = cublasCgemmStridedBatched(*cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, 1, m, &alpha, device_matrices_data, m, m*m, device_xn, m, m, &beta, device_xnp1, m, m, s);
#endif
		if (cublas_status != CUBLAS_STATUS_SUCCESS){
			log_message("Strided Batched Gemm in iterations failed.");
			return Result::CUDABLASInitializationError;
		}
		// now 'x_(n+1) = B x_n + c'

		// changing pointers for the next iteration
		// 'x_(n+1)' -> temp
		temp = device_xnp1;
		// 'x_n' -> 'x_(n+1)
		device_xnp1 = device_xn;
		// 'x_(n+1)' -> 'x_n'
		device_xn = temp;

		// now:
		// x_n = 'x_(n+1)'
		// x_np1 = 'x_(n)' and ready to become 'c' on the next loop step (see *1)

	}



	gpu_result = _gpu_free(device_xnp1);
	_react_on(gpu_result);



	return Result::Success;

}

