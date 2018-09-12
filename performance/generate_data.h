#ifndef GENERATE_DATA_H
#define GENERATE_DATA_H

#include "typedefs.h"


Result _generate_psf_images(int image_number, DataSettings psf_settings, OpticalSystemSettings os_settings, FocalDomainSettings fd_settings);

Result _generate_big_images(int image_number, DataSettings original_settings, DataSettings settings);

Result _generate_out_images(int image_number, DataSettings src_settings, DataSettings psf_settings);




#endif // GENERATE_DATA_H
