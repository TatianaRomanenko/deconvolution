#include "generate_data.h"

#include <omp.h>
#include <sstream>
#include <fstream>
#include <iostream>

#include "log.h"
#include "auxilary.h"

double m_func(TYPE x, TYPE y, TYPE R2)
{
	if (x*x + y*y <= R2)
		return 1;
	else
		return 0;
}

double inline ps_func(TYPE x, TYPE y, TYPE ps_func_coef)
{
	return ps_func_coef *(x*x + y*y);
}

void accumulate_sum_product(FFTW_COMPLEX& accumulate_sum, FFTW_COMPLEX arg1, FFTW_COMPLEX arg2, TYPE koef)
{
	accumulate_sum[0] += koef*(arg1[0] * arg2[0] - arg1[1] * arg2[1]);
	accumulate_sum[1] += koef*(arg1[0] * arg2[1] + arg1[1] * arg2[0]);
}

Result _generate_psf_images(int image_number, DataSettings psf_settings, OpticalSystemSettings os_settings, FocalDomainSettings fd_settings)
{
	unsigned int dim_x = psf_settings.resolution.dimX;
	unsigned int dim_y = psf_settings  .resolution.dimY;
	unsigned int psf_number = 2 * image_number - 1;

	TYPE* data = NULL;
	Result result = _cpu_allocate_data(&data, dim_x * dim_y * psf_number);
	_react_on(result);

	FFTW_COMPLEX* psf_data = 0;
	result = _cpu_fftw_allocate_data(&psf_data, dim_x * dim_y * psf_number);
	_react_on(result);

	// creating parameters for fftw plan
	int fftw_rank = 2;
	int size_of_single_image[] = { dim_x, dim_y };
	int distance_between_images = dim_x * dim_y;
	int stride_between_neighbours = 1;
	int *inembeded = size_of_single_image;
	int *onembeded = size_of_single_image;

#ifdef DOUBLEPRECISION
	fftw_plan psf_forward_plan = fftw_plan_many_dft(fftw_rank, size_of_single_image, psf_number, psf_data, inembeded, stride_between_neighbours, distance_between_images, psf_data, onembeded, stride_between_neighbours, distance_between_images, FFTW_FORWARD, FFTW_PATIENT | FFTW_DESTROY_INPUT);
#else
	fftwf_plan psf_forward_plan = fftwf_plan_many_dft(fftw_rank, size_of_single_image, psf_number, psf_data, inembeded, stride_between_neighbours, distance_between_images, psf_data, onembeded, stride_between_neighbours, distance_between_images, FFTW_FORWARD, FFTW_PATIENT | FFTW_DESTROY_INPUT);
#endif

	TYPE range_x = fd_settings.maxX - fd_settings.minX;
	TYPE range_y = fd_settings.maxY - fd_settings.minY;
	// iris radius for scaled coordinates
	TYPE r2 = os_settings.irisRadius * os_settings.irisRadius / (os_settings.d1 * os_settings.d1 * os_settings.lambda * os_settings.lambda);
	
	omp_set_num_threads(omp_get_max_threads());


	#pragma omp parallel for
	for (int k = 0; k < psf_number; k++)
	{
		for (unsigned int i = 0; i < dim_x; i++)
		{
			for (unsigned int j = 0; j < dim_y; j++)
			{
				TYPE arg_x = TYPE(i) / dim_x * range_x + fd_settings.minX;
				TYPE arg_y = TYPE(j) / dim_y * range_y + fd_settings.minY;
				TYPE ps_func_coef = os_settings.expFactor * (k * os_settings.depth / (image_number - 1) - os_settings.depth );
				TYPE grade = ps_func(arg_x, arg_y, ps_func_coef);
				
				psf_data[i*dim_y + j + k*distance_between_images][0] = m_func(arg_x, arg_y, r2) * cos(grade);
				psf_data[i*dim_y + j + k*distance_between_images][1] = m_func(arg_x, arg_y, r2) * sin(grade);
			}
		}
	}

#ifdef DOUBLEPRECISION
	fftw_execute(psf_forward_plan);
#else
	fftwf_execute(psf_forward_plan);
#endif

	TYPE* max_ar = (TYPE*)malloc(sizeof(TYPE) * psf_number);
	for (int k = 0; k < psf_number; k++){
		max_ar[k] = -std::numeric_limits<TYPE>::infinity();
	}

	#pragma omp parallel for
	for (int k = 0; k < psf_number; k++){

		for (unsigned int i = 0; i < dim_x; i++){
			for (unsigned int j = 0; j < dim_y; j++){

				
				unsigned int n;
				
				if (i < (TYPE)dim_x / 2)
				{
					if (j < (TYPE)dim_y / 2)
					{
						n = (i + dim_x / 2) * dim_y + (j + dim_y / 2) + distance_between_images * k;
						data[n] = psf_data[i*dim_y + j + k*distance_between_images][0] * psf_data[i*dim_y + j + k*distance_between_images][0] + psf_data[i*dim_y + j + k*distance_between_images][1] * psf_data[i*dim_y + j + k*distance_between_images][1];
						
					}
					else
					{
						n = (i + dim_x / 2) * dim_y + (j - (int)((TYPE)dim_y / 2 + 0.5)) + distance_between_images * k;
						data[n] = psf_data[i*dim_y + j + k*distance_between_images][0] * psf_data[i*dim_y + j + k*distance_between_images][0] + psf_data[i*dim_y + j + k*distance_between_images][1] * psf_data[i*dim_y + j + k*distance_between_images][1];
					}
				}
				else
				{
					if (j < (TYPE)dim_y / 2)
					{
						n = (i - (int)((TYPE)dim_x / 2 + 0.5)) * dim_y + (j + dim_y / 2) + distance_between_images * k;
						data[n] = psf_data[i*dim_y + j + k*distance_between_images][0] * psf_data[i*dim_y + j + k*distance_between_images][0] + psf_data[i*dim_y + j + k*distance_between_images][1] * psf_data[i*dim_y + j + k*distance_between_images][1];
						
					}
					else
					{
						n = (i - (int)((TYPE)dim_x / 2 + 0.5)) * dim_y + (j - (int)((TYPE)dim_y / 2 + 0.5)) + distance_between_images * k;
						data[n] = psf_data[i*dim_y + j + k*distance_between_images][0] * psf_data[i*dim_x + j + k*distance_between_images][0] + psf_data[i*dim_y + j + k*distance_between_images][1] * psf_data[i*dim_y + j + k*distance_between_images][1];
					}
				}
				if (data[n] > max_ar[k]) max_ar[k] = data[n];
			}
		}

	}

	TYPE max = max_ar[0];
	for (int k = 0; k < psf_number; k++){
		if (max_ar[k] > max){
			max = max_ar[k];
		}
	}

#pragma omp parallel for
	for (int k = 0; k < psf_number; k++){

		if (max != 0){
			for (unsigned int i = 0; i < dim_x; i++){
				for (unsigned int j = 0; j < dim_y; j++){
					data[i*dim_y + j + k*distance_between_images] = data[i*dim_y + j + k*distance_between_images] / max;
				}
			}
		}
	}

#ifdef DOUBLEPRECISION
	fftw_free(psf_data);
#else
	fftwf_free(psf_data);
#endif


	_write_data(data, psf_number, psf_settings);

	free(data);

	return Result::Success;
}

Result _generate_big_images(int image_number, DataSettings original_settings, DataSettings settings)
{
	unsigned int original_dim_x = original_settings.resolution.dimX;
	unsigned int original_dim_y = original_settings.resolution.dimY;
	size_t original_size = original_dim_x * original_dim_y;
	unsigned int dim_x = settings.resolution.dimX;
	unsigned int dim_y = settings.resolution.dimY;
	size_t size = dim_x * dim_y;

	unsigned int coef = (unsigned int)(dim_x / original_dim_x);
	if ((coef != (unsigned int)(dim_y / original_dim_y)) || ((coef * original_dim_x) != dim_x)){
		return log_operation_result("Unsupported psf resolution", Result::MethodParametersError);
	}

	TYPE* original_data;
	TYPE* data;
	std::string original_filename = "";
	std::string filename = "";
	Result result = Result::Success;

	result = _cpu_allocate_data(&original_data, original_size);
	_react_on(result);

	result = _cpu_allocate_data(&data, size);
	_react_on(result);

	for (int i = 0; i < image_number; i++){
		original_filename = get_full_filename(original_settings, i);
		filename = get_full_filename(settings, i);
		result = _read_data(original_data, original_size, original_filename);
		_react_on(result);

		for (int c = 0; c < coef * coef; c++){
			memcpy(data + c * original_size, original_data, original_size * sizeof(TYPE));
		}
		
		result = _write_data(data, size, filename);
	}

	free(original_data);
	free(data);

	return Result::Success;
}


Result _generate_out_images(int image_number, DataSettings src_settings, DataSettings psf_settings)
{
	int psf_number = 2 * image_number - 1;

	TYPE* src_data = NULL;
	TYPE* psf_data = NULL;

	FFTW_COMPLEX* ext_src_data = NULL;
	FFTW_COMPLEX* ext_psf_data = NULL;

	// НЕ НУЖНО ДЕЛАТЬ, НУЖНО ТОЛЬКО ЧТЕНИЕ ТХТ И РАСШИРЕНИЕ ИМЕЮЩИХСЯ!!!!


	return Result::Success;
}

