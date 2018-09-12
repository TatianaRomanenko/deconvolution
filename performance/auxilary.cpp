#include "auxilary.h"

#include <sstream>
#include <fstream>
#include <iostream>

#include "log.h"

F2C_COMPLEX _to_F2C_COMPLEX(FFTW_COMPLEX arg)
{
	F2C_COMPLEX result;
	result.r = arg[0];
	result.i = arg[1];

	return result;
}

Result _cpu_allocate_data(F2C_COMPLEX** data, size_t size)
{
	*data = (F2C_COMPLEX*)malloc(size * sizeof(F2C_COMPLEX));
	if (*data == NULL){
		log_message("Allocating memory for out images chunk failed.");
		return Result::MemoryAllocationError;
	}

	return Result::Success;
}

Result _cpu_allocate_data(TYPE** data, size_t size)
{
	*data = (TYPE*)(malloc(sizeof(TYPE) * size));
	if (*data == 0){
		log_message("Memory allocation for data failed.");
		return Result::MemoryAllocationError;
	}

	return Result::Success;
}

Result _cpu_fftw_allocate_data(FFTW_COMPLEX** data, size_t size)
{
#ifdef DOUBLEPRECISION
	*data = fftw_alloc_complex(size);
#else
	*data = fftwf_alloc_complex(size);
#endif
	
	if (*data == 0){
		log_message("FFTW memory allocation for data failed.");
		return Result::MemoryAllocationError;
	}

	return Result::Success;
}

Result _write_data(TYPE* data, int image_number, DataSettings settings)
{
	std::string filename = "";
	unsigned int image_size = settings.resolution.dimX * settings.resolution.dimY;

	for (int i = 0; i < image_number; i++){
		filename = get_full_filename(settings, i);
		FILE* fp = fopen(filename.c_str(), "wb");
		if (fp == NULL){
			log_message("File opening for write failed for " + filename);
			return Result::FileCreateError;
		}
		if (fwrite(data + i*image_size, sizeof(TYPE), image_size, fp)!=image_size){
			log_message("File writing failed for " + filename);
			return Result::FileWriteError;
		}

		if (fclose(fp)){
			log_message("File closing failed for " + filename);
			return Result::FileCloseError;
		}
	}

	return Result::Success;

}

Result _write_data(TYPE* data, size_t size, std::string& filename)
{
	FILE* fp = fopen(filename.c_str(), "wb");
	if (fp == NULL){
		log_message("File opening for write failed for " + filename);
		return Result::FileCreateError;
	}
	if (fwrite(data, sizeof(TYPE), size, fp)!=size){
		log_message("File writing failed for " + filename);
		return Result::FileWriteError;
	}

	if (fclose(fp)){
		log_message("File closing failed for " + filename);
		return Result::FileCloseError;
	}

	return Result::Success;
}

Result _write_data(F2C_COMPLEX* data, size_t size, std::string& filename)
{
	FILE* fp = fopen(filename.c_str(), "ab");
	if (fp == NULL){
		log_message("File opening for write failed for " + filename);
		return Result::FileCreateError;
	}
	if (fwrite(data, sizeof(F2C_COMPLEX), size, fp)!=size){
		log_message("File writing failed for " + filename);
		return Result::FileWriteError;
	}

	if (fclose(fp)){
		log_message("File closing failed for " + filename);
		return Result::FileCloseError;
	}

	return Result::Success;
}


Result _read_data(TYPE* data, size_t size, std::string& filename)
{
	FILE* fp;
	fp = fopen(filename.c_str(), "rb");
	if (fp == NULL){
		log_message("File opening for read failed for " + filename);
		return Result::FileOpenError;
	}

	unsigned int result = fread(data, sizeof(TYPE), size, fp);
	if (result != size){
		if (feof(fp)){
			log_message("File reading failed for "	+ filename + ". Premature end of file.");
			return Result::FileDimensionError;
		}else{
			log_message("File reading failed for " + filename);
			return Result::FileReadError;
		}
	}

	if (fclose(fp)){
		log_message("File closing failed for " + filename);
		return Result::FileCloseError;
	} 

	return Result::Success;
}

Result _read_data(F2C_COMPLEX* data, size_t size, std::string& filename)
{
	FILE* fp;
	fp = fopen(filename.c_str(), "rb");
	if (fp == NULL){
		log_message("File opening for read failed for " + filename);
		return Result::FileOpenError;
	}

	unsigned int result = fread(data, sizeof(F2C_COMPLEX), size, fp);
	if (result != size){
		if (feof(fp)){
			log_message("File reading failed for "	+ filename + ". Premature end of file.");
			return Result::FileDimensionError;
		}else{
			log_message("File reading failed for " + filename);
			return Result::FileReadError;
		}
	}

	if (fclose(fp)){
		log_message("File closing failed for " + filename);
		return Result::FileCloseError;
	} 

	return Result::Success;
}

Result _read_txt_images(TYPE* data, int image_number, DataSettings settings)
{
	std::string filename = "";
	Result result = Result::Success;
	size_t size = settings.resolution.dimX * settings.resolution.dimY;

	for (int i = 0; i < image_number; i++){
		filename = get_full_txt_filename(settings, i);
		result = _read_txt(data + i * size, size, filename);
		_react_on(result);
	}
}

Result _read_txt(TYPE* arr, size_t size, std::string& filename)
{
	// try to open file
	std::ifstream read_stream;
	read_stream.open(filename, std::ios::in);
	if (!(read_stream.is_open()))
	{
		log_operation_result("Reading image failed, stream for reading cannot be created for file " + filename + ".", Result::FileReadError);
		return Result::FileOpenError;
	}
	

	//parse txt
	std::string str = "";
	int i = 0;

	while (read_stream.good())
	{
		std::getline(read_stream, str, '\n');

		std::stringstream ss(str);
		TYPE value = 0;

		while (ss >> value)
		{
			if (i >= size){
				log_operation_result("Reading image failed, wrong dimension for file " + filename + ".", Result::FileDimensionError);
				return Result::FileDimensionError;
			}
			arr[i] = value;
			i++;
			ss.get();
		}
	}

	if (i != size){
		log_operation_result("Reading image failed, wrong dimension for file " + filename + ".", Result::FileDimensionError);
		return Result::FileDimensionError;
	}

	read_stream.close();
	return Result::Success;
}

Result _extend_image(TYPE* original_data, FFTW_COMPLEX* data, ImageResolution original_resolution, ImageResolution resolution, ExtensionMode extension_mode)
{
	// getting desired image resolution
	unsigned int eDimX = resolution.dimX;
	unsigned int eDimY = resolution.dimY;


	unsigned int x_start, y_start, x_end, y_end;

	if (extension_mode == ExtensionMode::Forward)
	{
		x_start = 0;
		y_start = 0;

		x_end = original_resolution.dimX;
		y_end = original_resolution.dimY;
	}
	else
	{
		x_start = ((int)((float)eDimX / 2 + 0.5) - 1) - ((int)((float)original_resolution.dimX / 2 + 0.5) - 1);
		y_start = ((int)((float)eDimY / 2 + 0.5) - 1) - ((int)((float)original_resolution.dimY / 2 + 0.5) - 1);

		if ((eDimX & 1) && !(original_resolution.dimX & 1))
		{
			x_start--;
		}
		if ((eDimY & 1) && !(original_resolution.dimY & 1))
		{
			y_start--;
		}

		x_end = x_start + original_resolution.dimX;
		y_end = y_start + original_resolution.dimY;
	}

	
	// filling data with zeros
	for (unsigned int i = 0; i < eDimX; i++){
		for (unsigned int j = 0; j < eDimY; j++){
			if ((i >= x_start) && (i < x_end) && (j >= y_start) && (j < y_end))
			{
				//std::cout << originalImages[k].data[(i - x_start) * originalImages[k].settings.resolution.dimX + (j - y_start)] << "    ";
				data[i*eDimY + j][0] = original_data[(i - x_start) * original_resolution.dimY + (j - y_start)];
			}
			else{
				data[i*eDimY + j][0] = 0.0;
			}

			data[i*eDimY + j][1] = 0.0;
		}
	}


	return Result::Success;
}

Result _remove_file(std::string filename)
{
	if (remove(filename.c_str()) == -1){
		perror("File deleting error");
		log_message("Preprocessed file delete failed.");
		return Result::FileWriteError;
	}
	return Result::Success;
}

Result _clean(int chunk_number, DataSettings bin_out_settings, DataSettings b_matrices_settings, DataSettings c_matrices_settings)
{
	std::string filename = "";
	Result result = Result::Success;
	for (int k = 0; k < chunk_number; k++){
		filename = get_full_filename(bin_out_settings, chunk_number, k);
		result = _remove_file(filename);
		_react_on(result);

		filename = get_full_filename(b_matrices_settings, chunk_number, k);
		result = _remove_file(filename);
		_react_on(result);

		filename = get_full_filename(c_matrices_settings, chunk_number, k);
		result = _remove_file(filename);
		_react_on(result);

	}

	return Result::Success;
}


