#include "preprocess.h"

#include <omp.h>

#include <f2c.h>
#include <clapack.h>

#include "auxilary.h"
#include "log.h"


Result _preprocess_out_images(int image_number, int chunk_number, DataSettings original_settings, DataSettings settings)
{
	Result result = Result::Success;
	TYPE* original_data = NULL;
	FFTW_COMPLEX* fftw_data = NULL;
	F2C_COMPLEX* data = NULL;
	size_t original_size = original_settings.resolution.dimX * original_settings.resolution.dimY;
	size_t size = settings.resolution.dimX * settings.resolution.dimY;
	size_t chunk_size = (size_t)(size / chunk_number);
	
	if (chunk_size * chunk_number != size){
		log_operation_result("Chunk size must be power of 2.", Result::MethodParametersError);
		return Result::MethodParametersError;
	}

	int fftw_rank = 2;
	int n0 = settings.resolution.dimX;
	int n1 = settings.resolution.dimY;

	// allocating data for single out file
	result = _cpu_allocate_data(&original_data, size);
	_react_on(result);

	// allocating data for Fourier transform of the extended image
	result = _cpu_fftw_allocate_data(&fftw_data, size);
	_react_on(result);

	// allocating data for the result single extended Fourier transform of out file
	result = _cpu_allocate_data(&data, size);
	_react_on(result);

	#ifdef DOUBLEPRECISION
	fftw_plan plan = fftw_plan_dft_r2c_2d(n0, n1, original_data, fftw_data, FFTW_ESTIMATE  | FFTW_DESTROY_INPUT);
#else
	fftwf_plan plan = fftwf_plan_dft_r2c_2d(n0, n1, original_data, fftw_data, FFTW_ESTIMATE  | FFTW_DESTROY_INPUT);
#endif

	if (plan == NULL){
		return log_operation_result("Failed to create plan.", Result::FFTWPlanCreatingError);
	}

	for (int k = 0; k < image_number; k++){

		// reading data from txt files
		result = _read_txt(original_data, size, get_full_txt_filename(original_settings, k));
		_react_on(result);

		//// extend images
		//result = _extend_image(original_data, fftw_data, original_settings.resolution, settings.resolution, ExtensionMode::Backward);
		//_react_on(result);

		// fourier transform

#ifdef DOUBLEPRECISION
		fftw_execute(plan);
#else
		fftwf_execute(plan);
#endif
		for (int i = 0; i < size; i++){
			data[i] = _to_F2C_COMPLEX(fftw_data[i]);
		}
		std::string chunk_filename = "";
		// save chunked data
		for (int p = 0; p < chunk_number; p++){
			chunk_filename = get_full_filename(settings, chunk_number, p);
			result = _write_data(data + p * chunk_size, chunk_size, chunk_filename);
			_react_on(result);
		}

	}

	free(data);
	free(original_data);
#ifdef DOUBLEPRECISION
	fftw_free(fftw_data);
#else
	fftwf_free(fftw_data);
#endif
}

Result _preprocess_psf_images(int image_number, int chunk_number, int sub_chunk_number,  DataSettings original_settings, DataSettings settings, DataSettings b_matrices_settings, DataSettings c_matrices_settings, double mu_parameter)
{
	Result result = Result::Success;
	TYPE* original_data = NULL;
	FFTW_COMPLEX* extended_original_data = NULL;
	FFTW_COMPLEX* fftw_data = NULL;
	F2C_COMPLEX* data = NULL;
	size_t original_size = original_settings.resolution.dimX * original_settings.resolution.dimY;
	size_t size = settings.resolution.dimX * settings.resolution.dimY;
	size_t chunk_size = (size_t)(size / chunk_number);
	integer m = image_number;

	if (chunk_size * chunk_number != size){
		log_operation_result("Chunk size must be power of 2.", Result::MethodParametersError);
		return Result::MethodParametersError;
	}

	int fftw_rank = 2;
	int n0 = settings.resolution.dimX;
	int n1 = settings.resolution.dimY;

	omp_set_num_threads(omp_get_max_threads());

	// allocating data for single psf file
	result = _cpu_allocate_data(&original_data, original_size);
	_react_on(result);

	// allocating data for single extended psf file
	result = _cpu_fftw_allocate_data(&extended_original_data, size);
	_react_on(result);

	// allocating data for Fourier transform of the extended image
	result = _cpu_fftw_allocate_data(&fftw_data, size);
	_react_on(result);

	// allocating data for the result single extended Fourier transform of out file
	result = _cpu_allocate_data(&data, size);
	_react_on(result);

#ifdef DOUBLEPRECISION
	fftw_plan plan = fftw_plan_dft_2d(n0, n1, extended_original_data, fftw_data,FFTW_FORWARD, FFTW_ESTIMATE  );
#else
	fftwf_plan plan = fftwf_plan_dft_2d(n0, n1, extended_original_data, fftw_data,FFTW_FORWARD, FFTW_ESTIMATE );
#endif

	if (plan == NULL){
		return log_operation_result("Failed to create plan.", Result::FFTWPlanCreatingError);
	}

	std::string chunk_filename = "";
	for (int k = 0; k < 2 * m - 1; k++){

		// reading data from txt files
		result = _read_txt(original_data, original_size, get_full_txt_filename(original_settings, k));
		_react_on(result);

		// extend images
		result = _extend_image(original_data, extended_original_data, original_settings.resolution, settings.resolution, ExtensionMode::Forward);
		_react_on(result);

		// fourier transform

#ifdef DOUBLEPRECISION
		fftw_execute(plan);
#else
		fftwf_execute(plan);
#endif
		for (int i = 0; i < size; i++){
			data[i] = _to_F2C_COMPLEX(fftw_data[i]);
		}
		
		// save chunked data
		for (int p = 0; p < chunk_number; p++){
			chunk_filename = get_full_filename(settings, chunk_number, p);
			_write_data(data + p * chunk_size, chunk_size, chunk_filename);
		}

	}

	free(data);
	free(original_data);
#ifdef DOUBLEPRECISION
	fftw_free(fftw_data);
	fftw_free(extended_original_data);
#else
	fftwf_free(fftw_data);
	fftwf_free(extended_original_data);
#endif

	// fft_ext_chunked_psf data written
	// starting preprocess matrices C and B
	// doing this for the whole chunk is too heavy 
	// splitting chunk into sub-chunks

	// reallocating 'data' for all psf data for single chunk
	size_t all_psf_chunk_size = chunk_size * ( 2 * m - 1 );
	result = _cpu_allocate_data(&data, all_psf_chunk_size);
	_react_on(result);

	// allocating data for sub-chunk
	F2C_COMPLEX* b_matrices_data = NULL;
	F2C_COMPLEX* c_matrices_data = NULL;
	size_t matrices_sub_chunk_size = m * m * chunk_size / sub_chunk_number;
	int sub_chunk_size = (int)(chunk_size / sub_chunk_number);
	int single_matrix_size = m * m;

	if (sub_chunk_size * sub_chunk_number != chunk_size){
		log_operation_result("Sub-chunk size must be power of 2.", Result::MethodParametersError);
		return Result::MethodParametersError;
	}

	// allocating data for matrices sub-chunk
	result = _cpu_allocate_data(&b_matrices_data, matrices_sub_chunk_size);
	_react_on(result);

	result = _cpu_allocate_data(&c_matrices_data, matrices_sub_chunk_size);
	_react_on(result);

	// blas zgemm_ function arguments for calculation (mu*A*A + E)^{-1}
	// operations with matrices 
	char no_operation = 'N';
	char conjugate_operation = 'C';
	// first dimension for matrices
	integer* matrix_size = &m;
	F2C_COMPLEX zero, identity;
	identity.r = 1.0;
	identity.i = 0.0;

	zero.r = 0.0;
	zero.i = 0.0;
	
	std::string b_matrices_filename = "";
	std::string c_matrices_filename = "";

	for (int p = 0; p < chunk_number; p++){
		chunk_filename = get_full_filename(settings, chunk_number, p);
		// reading fft_ext_chunked_psf 
		result = _read_data(data, all_psf_chunk_size, chunk_filename);
		_react_on(result);

		for (int sp = 0; sp < sub_chunk_number; sp++){

#pragma omp parallel for
			for (int i = 0 ; i < sub_chunk_size; i++){
				F2C_COMPLEX mu;
				mu.r = mu_parameter;
				mu.i = 0.0;

				unsigned int shift = i + sub_chunk_size * sp;
				unsigned int matrix_shift = i * single_matrix_size;

				F2C_COMPLEX* A = (F2C_COMPLEX*)malloc(single_matrix_size * sizeof(F2C_COMPLEX));
				F2C_COMPLEX* B = (F2C_COMPLEX*)malloc(single_matrix_size * sizeof(F2C_COMPLEX));
				F2C_COMPLEX* C = (F2C_COMPLEX*)malloc(single_matrix_size * sizeof(F2C_COMPLEX));

				// for matrix inverse methods
				integer* ipiv = (integer*)malloc(m * sizeof(integer));
				integer info;
				integer lwork = m*m;
				F2C_COMPLEX* work = (F2C_COMPLEX*)malloc(lwork * sizeof(F2C_COMPLEX));

				integer inc = 1;

			
				for (int l = 0; l < m; l++){
					for (int k = 0; k < m; k++){
						A[l + m * k] = data[shift + chunk_size * (l - k + m - 1)];
					
						B[l + m * k].i = 0.0;
						if (l == k){
							B[l + m * k].r = 1.0;
						}
						else{
							B[l + m * k].r = 0.0;
						}
					}

				
				}



				// now A is A and B = E
	#ifdef DOUBLEPRECISION
				zgemm_(&conjugate_operation, &no_operation, matrix_size, matrix_size, matrix_size, &mu, A, matrix_size, A, matrix_size, &identity, B, matrix_size);
	#else
				cgemm_(&conjugate_operation, &no_operation, matrix_size, matrix_size, matrix_size, &mu, A, matrix_size, A, matrix_size, &identity, B, matrix_size);
	#endif


				// now B = (mu A*A + E);
	#ifdef DOUBLEPRECISION
				zgetrf_(matrix_size, matrix_size, B, matrix_size, ipiv, &info);
	#else
				cgetrf_(matrix_size, matrix_size, B, matrix_size, ipiv, &info);
	#endif

				if (info != 0){
					if (info < 0){
						log_operation_result(to_string(Result::BLASMethodWrongArgument) + " in " + std::to_string(-info) + "-th argument.", Result::BLASMethodWrongArgument);
						//return Result::BLASMethodWrongArgument;
					}
					else{
						log_operation_result(to_string(Result::BLASMethodZeroDiagonalElement) + " for i = " + std::to_string(info) + "", Result::BLASMethodZeroDiagonalElement);
						//return Result::BLASMethodZeroDiagonalElement;
					}

				}
				// now B contains LU factorization
	#ifdef DOUBLEPRECISION
				zgetri_(matrix_size, B, matrix_size, ipiv, work, &lwork, &info);
	#else
				cgetri_(matrix_size, B, matrix_size, ipiv, work, &lwork, &info);
	#endif

				if (info != 0){
					if (info < 0){
						log_operation_result(to_string(Result::BLASMethodWrongArgument) + " in " + std::to_string(-info) + "-th argument for inverse method.", Result::BLASMethodWrongArgument);
						//return Result::BLASMethodWrongArgument;
					}
					else{
						log_operation_result(to_string(Result::BLASMethodZeroDiagonalElement) + "for inverse method for i = " + std::to_string(info) + ".", Result::BLASMethodZeroDiagonalElement);
						//return Result::BLASMethodZeroDiagonalElement;
					}

				}
				// now B = (mu A* A + E)^{-1}


	#ifdef DOUBLEPRECISION
				zgemm_(&no_operation, &conjugate_pperation, matrix_size, matrix_size, matrix_size, &mu, B, matrix_size, A, matrix_size, &zero, C, matrix_size);
	#else
				cgemm_(&no_operation, &conjugate_operation, matrix_size, matrix_size, matrix_size, &mu, B, matrix_size, A, matrix_size, &zero, C, matrix_size);
	#endif

				// now C = mu (mu A* A + E)^{-1} A*
			
				// writing preprocessed matrices to result arrays
				for (int l = 0; l < single_matrix_size; l++){
					b_matrices_data[matrix_shift + l] = B[l];
					c_matrices_data[matrix_shift + l] = C[l];
				}


				// deallocating memory
				free(A);
				free(B);
				free(C);
				free(ipiv);
				free(work);
			}

			// save sub-chunk matrices data to binary files
			b_matrices_filename = get_full_filename(b_matrices_settings, chunk_number, p);
			c_matrices_filename = get_full_filename(c_matrices_settings, chunk_number, p);

			result = _write_data(b_matrices_data, matrices_sub_chunk_size, b_matrices_filename);
			_react_on(result);

			result = _write_data(c_matrices_data, matrices_sub_chunk_size, c_matrices_filename);
			_react_on(result);

		}

	}

	free(b_matrices_data);
	free(c_matrices_data);
	free(data);

	// deleting temporary data with fft_ext_psf chunks
	for (int p = 0; p < chunk_number; p++){
		chunk_filename = get_full_filename(settings, chunk_number, p);
		if (remove(chunk_filename.c_str()) == -1){
			log_message("Preprocessed file delete failed.");
			return Result::FileWriteError;
		}

	}

	return Result::Success;
}
