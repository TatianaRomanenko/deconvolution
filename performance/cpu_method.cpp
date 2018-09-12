#include "cpu_method.h"

#include "clapack.h"




Result _cpu_preprocess_method_part1(F2C_COMPLEX* out_data, F2C_COMPLEX* matrices_data, F2C_COMPLEX* c_vector_data, int image_number, int chunk_size)
{
	unsigned int s = chunk_size;
	
	// parameters for *gemv_ method
	integer m = image_number;
	integer* matrixSize = &m;
	char noOperation = 'N';

	// constants for *gemv_ method
	integer inc = 1;
	F2C_COMPLEX zero, identity;
	identity.r = 1.0;
	identity.i = 0.0;
	zero.r = 0.0;
	zero.i = 0.0;


#pragma omp parallel for
	for (int i = 0; i < s; i++){
		unsigned int matrix_shift = i * m * m;
		unsigned int vector_shift = i * m;
		
		// allocating memory for the right part vector of the system 
		// 'A x = b' being solved for every point of the Fourier space.
		// WARNING: size of b is too small ( m * 64/132 bytes) 
		// so we do not check whether malloc could allocate memory
		F2C_COMPLEX* b = (F2C_COMPLEX*)malloc(m * sizeof(F2C_COMPLEX));

		// getting pointers to the appropriate matrix and result 'c' vector
		F2C_COMPLEX* C = matrices_data + matrix_shift;
		F2C_COMPLEX* c = c_vector_data + vector_shift;

		// initializing right part vector 'b' for BLAS calculating.
		// Vector of the right part 'b' is formed by the elements 
		// of each out image taken for the current point of the Fourier space,
		// i.e. for (i)-th point of the Fourier space 
		// (assuming 2d array in the Fourier space is stretched to 1d array)
		// b[0] = out_0[i], b[1] = out_1[i], ... , b[m-1] = out_(m-1)[i]. 
		for (int l = 0; l < m; l++){
			b[l] = out_data[s * l + i];
		}

		
		// calculating 'c = mu (E + mu A* A)^{-1} A* b'
#ifdef DOUBLEPRECISION
		zgemv_(&noOperation, matrixSize, matrixSize, &identity, C, matrixSize, b, &inc, &zero, c, &inc);
#else
		cgemv_(&noOperation, matrixSize, matrixSize, &identity, C, matrixSize, b, &inc, &zero, c, &inc);
#endif


		free(b);
	}

	return Result::Success;
}


Result _cpu_preprocess_method_part2(F2C_COMPLEX* out_data, F2C_COMPLEX* matrices_data, F2C_COMPLEX* c_vector_data, int image_number, int chunk_size, int iteration_number)
{		
	unsigned int s = chunk_size;
	
	// parameters for *gemv_ method
	integer m = image_number;
	integer* matrixSize = &m;
	char noOperation = 'N';

	// constants for *gemv_ method
	integer inc = 1;
	F2C_COMPLEX zero, identity;
	identity.r = 1.0;
	identity.i = 0.0;
	zero.r = 0.0;
	zero.i = 0.0;

#pragma omp parallel for
	for (int i = 0; i < s; i++){
		unsigned int matrix_shift = i * m * m;
		unsigned int vector_shift = i * m;

		// allocating memory for the solution of the system 'A x = b' for every point of the Fourier space
		// on the step (n) 'x_n' and on the step (n+1) 'x_{n+1}'
		// WARNING: sizes of xn and anp1 are too small ( m * 64/132 bytes) 
		// so we do not check whether malloc could allocate memory
		F2C_COMPLEX* xn = (F2C_COMPLEX*)malloc(m * sizeof(F2C_COMPLEX));
		F2C_COMPLEX* xnp1 = (F2C_COMPLEX*)malloc(m * sizeof(F2C_COMPLEX));

		// getting pointers to the appropriate matrix and 'c' vector
		F2C_COMPLEX* B = matrices_data + matrix_shift;
		F2C_COMPLEX* c = c_vector_data + vector_shift;

		// initializing 'x_0' with zero-vector
		for (int l = 0; l < m; l++){
			xn[l].r = 0.0;
			xn[l].i = 0.0;
		}

		for (int n = 0; n < iteration_number; n++){
			
			// copying 'c' vector to 'x_(n+1)' vector
			for (int l = 0; l < m; l++){
				xnp1[l] = c[l];
			}

			// calculating 'x_(n+1) = B x_n + c' for (iteration_number) iterations
#ifdef DOUBLEPRECISION
			zgemv_(&noOperation, matrixSize, matrixSize, &identity, B, matrixSize, xn, &inc, &identity, xnp1, &inc);
#else
			cgemv_(&noOperation, matrixSize, matrixSize, &identity, B, matrixSize, xn, &inc, &identity, xnp1, &inc);
#endif
			// copying 'x_(n+1)' vector to 'x_n' vector
			for (int l = 0; l < m; l++){
				xn[l] = xnp1[l];
			}

		}

		// copying solution 'x = x_(n)' to out images for (i)-th point of the Fourier space:
		// x[0] -> out_0[i], x[1] -> out_1[i], ... , x[m-1] -> out_(m-1)[i].
		for (int l = 0; l < m; l++){
			out_data[s * l + i] = xn[l];
		}

		free(xn);
		free(xnp1);

	}

	return Result::Success;

}

