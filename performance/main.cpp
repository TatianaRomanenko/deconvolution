#include "preprocess.h"

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "log.h"
#include "auxilary.h"
#include "method.h"


#ifdef PERFORMANCE
/*
* argv[0] - performance.exe
* argv[1] - image number
* argv[2] - image resolution
* argv[3] - chunk number
* argv[4] - iterations
*/
int main(int argc, char* argv[])
{
	std::string results_filename = "C:\\EYE\\results.txt";
	if (argc != 5){
		log_message("Wrong command line parameters count.");
		return -1;
	}

	int image_number = atoi(argv[1]);
	int image_resolution = atoi(argv[2]);
	int chunk_number = atoi(argv[3]);
	int iteration_number = atoi(argv[4]);

	int psf_resolution = image_resolution - 1;
	int result_resolution = 2 * image_resolution;


	DataSettings bin_out_settings;
	bin_out_settings.foldername = "C:\\EYE\\bin_data\\" + std::to_string(image_resolution) + "\\" + std::to_string(image_number);
	bin_out_settings.filename = "out";
	bin_out_settings.filename_mode = FileNameMode::ZERO0;
	bin_out_settings.resolution.dimX = result_resolution;
	bin_out_settings.resolution.dimY = result_resolution;

	DataSettings b_matrices_settings;
	b_matrices_settings.foldername = "C:\\EYE\\bin_data\\" + std::to_string(image_resolution) + "\\" + std::to_string(image_number);
	b_matrices_settings.filename = "b_matrices";
	b_matrices_settings.filename_mode = FileNameMode::ZERO0;
	b_matrices_settings.resolution.dimX = result_resolution;
	b_matrices_settings.resolution.dimY = result_resolution;

	DataSettings c_matrices_settings;
	c_matrices_settings.foldername = "C:\\EYE\\bin_data\\" + std::to_string(image_resolution) + "\\" + std::to_string(image_number);
	c_matrices_settings.filename = "c_matrices";
	c_matrices_settings.filename_mode = FileNameMode::ZERO0;
	c_matrices_settings.resolution.dimX = result_resolution;
	c_matrices_settings.resolution.dimY = result_resolution;

	double seconds = 0.0;

	//write_performance_header(results_filename);
	_preprocess_method(&seconds, image_number, iteration_number, chunk_number, bin_out_settings, b_matrices_settings, c_matrices_settings);
	write_performance_result(results_filename, seconds, image_number, image_resolution, iteration_number, chunk_number);

	return 0;
}
#elif defined _PREPROCESS
/*
* argv[0] - preprocess.exe
* argv[1] - image number
* argv[2] - image resolution
* argv[3] - chunk number
* argv[4] - sub-chunk number
* argv[5] - mu parameter
*/
int main(int argc, char* argv[])
{
	if (argc != 6){
		log_message("Wrong command line parameters count.");
		return -1;
	}

	int image_number = atoi(argv[1]);
	int image_resolution = atoi(argv[2]);
	int chunk_number = atoi(argv[3]);
	int sub_chunk_number = atoi(argv[4]);
	float mu_parameter = atof(argv[5]);

	int psf_resolution = image_resolution - 1;
	int result_resolution = 2 * image_resolution;

	DataSettings txt_out_settings;
	txt_out_settings.foldername = "C:\\EYE\\data\\" + std::to_string(image_resolution) + "\\" + std::to_string(image_number);
	txt_out_settings.filename = "out";
	txt_out_settings.filename_mode = FileNameMode::ZERO0;
	txt_out_settings.resolution.dimX = image_resolution;
	txt_out_settings.resolution.dimY = image_resolution;

	DataSettings bin_out_settings;
	bin_out_settings.foldername = "C:\\EYE\\bin_data\\" + std::to_string(image_resolution) + "\\" + std::to_string(image_number);
	bin_out_settings.filename = "out";
	bin_out_settings.filename_mode = FileNameMode::ZERO0;
	bin_out_settings.resolution.dimX = result_resolution;
	bin_out_settings.resolution.dimY = result_resolution;

	DataSettings txt_psf_settings;
	txt_psf_settings.foldername = "C:\\EYE\\data\\" + std::to_string(image_resolution) + "\\" + std::to_string(image_number);
	txt_psf_settings.filename = "psf";
	txt_psf_settings.filename_mode = FileNameMode::ZERO0;
	txt_psf_settings.resolution.dimX = psf_resolution;
	txt_psf_settings.resolution.dimY = psf_resolution;

	DataSettings bin_psf_settings;
	bin_psf_settings.foldername = "C:\\EYE\\bin_data\\" + std::to_string(image_resolution) + "\\" + std::to_string(image_number);
	bin_psf_settings.filename = "psf";
	bin_psf_settings.filename_mode = FileNameMode::ZERO0;
	bin_psf_settings.resolution.dimX = result_resolution;
	bin_psf_settings.resolution.dimY = result_resolution;

	DataSettings b_matrices_settings;
	b_matrices_settings.foldername = "C:\\EYE\\bin_data\\" + std::to_string(image_resolution) + "\\" + std::to_string(image_number);
	b_matrices_settings.filename = "b_matrices";
	b_matrices_settings.filename_mode = FileNameMode::ZERO0;
	b_matrices_settings.resolution.dimX = result_resolution;
	b_matrices_settings.resolution.dimY = result_resolution;

	DataSettings c_matrices_settings;
	c_matrices_settings.foldername = "C:\\EYE\\bin_data\\" + std::to_string(image_resolution) + "\\" + std::to_string(image_number);
	c_matrices_settings.filename = "c_matrices";
	c_matrices_settings.filename_mode = FileNameMode::ZERO0;
	c_matrices_settings.resolution.dimX = result_resolution;
	c_matrices_settings.resolution.dimY = result_resolution;
	
	_preprocess_out_images(image_number, chunk_number, txt_out_settings, bin_out_settings);
	_preprocess_psf_images(image_number, chunk_number, sub_chunk_number,
		txt_psf_settings, bin_psf_settings, b_matrices_settings, c_matrices_settings,
		mu_parameter);

	return 0;
}
#elif defined CLEAN
/*
* argv[0] - clean.exe
* argv[1] - image number
* argv[2] - image resolution
* argv[3] - chunk number
*/
int main(int argc, char* argv[])
{
	if (argc != 4){
		log_message("Wrong command line parameters count.");
		return -1;
	}

	int image_number = atoi(argv[1]);
	int image_resolution = atoi(argv[2]);
	int chunk_number = atoi(argv[3]);

	int psf_resolution = image_resolution - 1;
	int result_resolution = 2 * image_resolution;

	DataSettings bin_out_settings;
	bin_out_settings.foldername = "C:\\EYE\\bin_data\\" + std::to_string(image_resolution) + "\\" + std::to_string(image_number);
	bin_out_settings.filename = "out";
	bin_out_settings.filename_mode = FileNameMode::ZERO0;
	bin_out_settings.resolution.dimX = result_resolution;
	bin_out_settings.resolution.dimY = result_resolution;

	DataSettings b_matrices_settings;
	b_matrices_settings.foldername = "C:\\EYE\\bin_data\\" + std::to_string(image_resolution) + "\\" + std::to_string(image_number);
	b_matrices_settings.filename = "b_matrices";
	b_matrices_settings.filename_mode = FileNameMode::ZERO0;
	b_matrices_settings.resolution.dimX = result_resolution;
	b_matrices_settings.resolution.dimY = result_resolution;

	DataSettings c_matrices_settings;
	c_matrices_settings.foldername = "C:\\EYE\\bin_data\\" + std::to_string(image_resolution) + "\\" + std::to_string(image_number);
	c_matrices_settings.filename = "c_matrices";
	c_matrices_settings.filename_mode = FileNameMode::ZERO0;
	c_matrices_settings.resolution.dimX = result_resolution;
	c_matrices_settings.resolution.dimY = result_resolution;

	_clean(chunk_number, bin_out_settings, b_matrices_settings, c_matrices_settings);

	return 0;
}

#endif