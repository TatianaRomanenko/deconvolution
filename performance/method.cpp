#include "method.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <omp.h>

#include "log.h"
#include "cpu_method.h"
#include "gpu_method.cuh"
#include "auxilary.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"


Result _preprocess_out_images(int image_number, int chunk_number, DataSettings out_settings, DataSettings fft_ext_out_settings);

Result _preprocess_psf_images(int image_number, int chunk_number, DataSettings psf_settings, DataSettings fft_ext_psf_settings);


Result _preprocess_method(double* seconds, int image_number, int iteration_number, int chunk_number, 
						  DataSettings fft_ext_out_settings, DataSettings b_matrices_settings, DataSettings c_matrices_settings)
{
	// getting settings for method
	Result result = Result::Success;
	unsigned int chunk_size =			fft_ext_out_settings.resolution.dimX * 
										fft_ext_out_settings.resolution.dimY / chunk_number;
	unsigned int all_out_chunk_size =	chunk_size * image_number;

	unsigned int matrices_chunk_size =	b_matrices_settings.resolution.dimX *
										b_matrices_settings.resolution.dimY *
										image_number * image_number / chunk_number;

	omp_set_num_threads(omp_get_max_threads());

	// defining arrays for outs and matrices
	F2C_COMPLEX* out_data = NULL;
	F2C_COMPLEX* c_vector_data = NULL;
	F2C_COMPLEX* matrices_data = NULL;
	
	// allocating CPU memory for outs, matrices and c vectors(used only for CPU method)
	result = _cpu_allocate_data(&out_data, all_out_chunk_size);
	_react_on(result);

	result = _cpu_allocate_data(&matrices_data, matrices_chunk_size);
	_react_on(result);

#ifdef CPU
	// 'c' vectors is needed for calculation and is allocated and used independently for CPU and GPU
	result = _cpu_allocate_data(&c_vector_data, all_out_chunk_size);
	_react_on(result);

#endif

#ifdef GPU
	// defining CUDA handlers
	cublasStatus_t cublas_status;
	cublasHandle_t cublas_handle;
	Result gpu_result = Result::Success;

	// defining GPU arrays for outs, matrices and c vectors
	GPU_COMPLEX* device_out_data = NULL;
	GPU_COMPLEX* device_matrices_data = NULL;
	GPU_COMPLEX* device_c_vector_data = NULL;


	// initializing cublas
	gpu_result = _gpu_init(&cublas_status, &cublas_handle);
	_react_on(gpu_result);

	// allocating GPU memory
	gpu_result = _gpu_allocate_data(&device_out_data, all_out_chunk_size);
	_react_on(gpu_result);
	gpu_result = _gpu_allocate_data(&device_c_vector_data, all_out_chunk_size);
	_react_on(gpu_result);
	gpu_result = _gpu_allocate_data(&device_matrices_data, matrices_chunk_size);
	_react_on(gpu_result);
	
#endif

	// time measure stuff
	typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<float> fsec;

	auto start = Time::now();
	// performing method for every chunk
	for (int p = 0; p < chunk_number; p++){
		
		std::string out_chunk_filename = get_full_filename(fft_ext_out_settings, chunk_number, p);
		// reading out file data for current chunk
		result = _read_data(out_data, all_out_chunk_size, out_chunk_filename);
		_react_on(result);

		std::string matrices_chunk_filename = get_full_filename(c_matrices_settings, chunk_number, p);
		// reading preprocessed 'C = mu (E + mu A* A)^{-1}' matrices for current chunk
		result = _read_data(matrices_data, matrices_chunk_size, matrices_chunk_filename);
		_react_on(result);

		// C matrices data successfully read
		// starting part 1 of the method...
#ifdef CPU
		result = _cpu_preprocess_method_part1(out_data, matrices_data, c_vector_data, image_number, chunk_size);
		_react_on(result);
#elif defined GPU
		// copying out images and C matrices to GPU memory
		gpu_result = _gpu_copy_to_device_data(out_data, device_out_data, all_out_chunk_size);
		_react_on(gpu_result);
		gpu_result = _gpu_copy_to_device_data(matrices_data, device_matrices_data, matrices_chunk_size);
		_react_on(gpu_result);

		gpu_result = _gpu_preprocess_method_part1(device_out_data, device_matrices_data, device_c_vector_data, image_number, chunk_size, &cublas_handle);
		_react_on(gpu_result);
#endif
		
		// part 1 of the method finished
		// starting part 2 of the method


		matrices_chunk_filename = get_full_filename(b_matrices_settings, chunk_number, p);
		// reading preprocessed 'B = (E + mu A* A)^{-1}' matrices for current chunk
		result = _read_data(matrices_data, matrices_chunk_size, matrices_chunk_filename);
		_react_on(result);

		// B matrices data successfully read
		// starting part 2 of the method...
#ifdef CPU
		result = _cpu_preprocess_method_part2(out_data, matrices_data, c_vector_data, image_number, chunk_size, iteration_number);
		_react_on(result);
#elif defined GPU
		// copying B matrices to GPU memory
		gpu_result = _gpu_copy_to_device_data(matrices_data, device_matrices_data, matrices_chunk_size);
		_react_on(gpu_result);
		
		gpu_result = _gpu_preprocess_method_part2(device_out_data, device_matrices_data, device_c_vector_data, image_number, chunk_size, iteration_number, &cublas_handle);
		_react_on(gpu_result);

		// copying result to CPU memory
		gpu_result = _gpu_copy_to_host_data(device_out_data, out_data, all_out_chunk_size);
		_react_on(gpu_result);


#endif
		
	}

	auto finish = Time::now();
	fsec fs = finish - start;
	*seconds = fs.count();

#ifdef GPU
	gpu_result = _gpu_free(device_out_data);
	_react_on(gpu_result);
	gpu_result = _gpu_free(device_c_vector_data);
	_react_on(gpu_result);

	gpu_result = _gpu_free(device_matrices_data);
	_react_on(gpu_result);
#endif
	free(c_vector_data);
	free(matrices_data);
	free(out_data);
	return Result::Success;

}

