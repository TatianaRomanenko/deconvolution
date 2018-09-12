#ifndef METHOD_H
#define METHOD_H

#include "typedefs.h"


Result _preprocess_out_images(int image_number, int chunk_number, DataSettings out_settings, DataSettings fft_ext_out_settings);

Result _preprocess_psf_images(int image_number, int chunk_number, DataSettings psf_settings, DataSettings fft_ext_psf_settings);

/*
* method works assuming:
* - output images were properly extended for using the Convolution theorem and 
* the Fourier transform of the extended output images is chuncked abd saved to binary files;
* - PSF (point spread function) images were properly extended for using the Convolution theorem and
* the Fourier transform of the extended PSF images is saved to binary files
* i.e. OTF (optical transfer function) images are taken;
* method solves 'A x = b' system for every point of Fourier space
* using iterative regularization method
*/
Result _preprocess_method(double* seconds, int image_number, int iteration_number, int chunk_number, DataSettings fft_ext_out_settings, DataSettings b_matrices_settings, DataSettings c_matrices_settings);



#endif // METHOD_H