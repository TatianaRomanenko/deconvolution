#ifndef PREPROCESS_H
#define PREPROCESS_H

#include "typedefs.h"

Result _preprocess_out_images(int image_number, int chunk_number, DataSettings original_settings, DataSettings settings);

Result _preprocess_psf_images(int image_number, int chunk_number, int sub_chunk_number,  
							  DataSettings original_settings, DataSettings settings, 
							  DataSettings b_matrices_settings, DataSettings c_matrices_settings, double mu_parameter);

#endif // PREPROCESS_H