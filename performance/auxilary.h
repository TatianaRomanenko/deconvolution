#ifndef AUXILARY_H
#define AUXILARY_H

#include "typedefs.h"

F2C_COMPLEX _to_F2C_COMPLEX(FFTW_COMPLEX arg);

Result _remove_file(std::string filename);

Result _clean(int chunk_number, DataSettings bin_out_settings, 
			 DataSettings b_matrices_settings, DataSettings c_matrices_settings);

Result _cpu_allocate_data(F2C_COMPLEX** data, size_t size);

Result _cpu_allocate_data(TYPE** data, size_t size);

Result _cpu_fftw_allocate_data(FFTW_COMPLEX** data, size_t size);

// writes (image_number) binary files
Result _write_data(TYPE* data, int image_number, DataSettings settings);

// reads single binary file
Result _read_data(TYPE* data, size_t size, std::string& filename);

// reads single binary file
Result _read_data(F2C_COMPLEX* data, size_t size, std::string& filename);

// writes single binary file
Result _write_data(TYPE* data, size_t size, std::string& filename);

// writes single binary file
Result _write_data(F2C_COMPLEX* data, size_t size, std::string& filename);

Result _read_txt_images(TYPE* data, int image_number, DataSettings settings);

Result _read_txt(TYPE* arr, size_t size, std::string& filename);

Result _extend_image(TYPE* original_data, FFTW_COMPLEX* data, ImageResolution original_resolution, ImageResolution resolution, ExtensionMode extension_mode);


#endif // AUXILARY_H