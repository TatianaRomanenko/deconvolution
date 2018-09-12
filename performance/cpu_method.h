#ifndef CPU_METHOD_H
#define CPU_METHOD_H

#include "log.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <omp.h>




/*
* matrices_data contains preprocessed 'C = mu (E + mu A* A)^{-1}' matrices,
* method is getting 'c = mu (E + mu A* A)^{-1} A* b' for every point in c_vector_data
* where right part 'b' of original system 'Ax = b' is stored in out_data
* while 'b' is stored sequentially image after image (there are (m=image_number) images size (s=chunk_size)):
* out_0[0] out_0[1] ... out_0[s-1] 
* out_1[0] out_1[1] ... out_1[s-1] 
* ... 
* out_(m-1)[0] out_(m-1)[1] ... out_(m-1)[s-1]
* 'c' is stored sequentially vector after vector (there are (s) vectors size (m)):
* c_0[0] c_0[1] ... c_0[m-1]
* c_1[0] c_1[1] ... c_1[m-1]
* ...
* c_(s-1)[0] c_(s-1)[1] ... c_(s-1][m-1]
*/
Result _cpu_preprocess_method_part1(F2C_COMPLEX* out_data, F2C_COMPLEX* matrices_data, F2C_COMPLEX* c_vector_data, int image_number, int chunk_size);

/*
* matrices_data contains preprocessed 'B = (E + mu A* A)^{-1}' matrices,
* c_vector_data contains 'c = mu (E + mu A* A)^{-1} A* b' vectors,
* result is saved to out_data
*/
Result _cpu_preprocess_method_part2(F2C_COMPLEX* out_data, F2C_COMPLEX* matrices_data, F2C_COMPLEX* c_vector_data, int image_number, int chunk_size, int iteration_number);


#endif //CPU_METHOD_H