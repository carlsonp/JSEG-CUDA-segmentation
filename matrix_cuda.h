
#ifndef _MATRIX_CUDA_H_
#define _MATRIX_CUDA_H_

// Internal Includes
#include "matrix.h"

// Library Includes
#include "cutil_inline.h"

// Standard Includes
#include <iostream>


// Allocate a device matrix of same size as M.
inline Matrix AllocateDeviceMatrix(const Matrix M)
{
	Matrix Mdevice = M;
	int size = M.width * M.height * sizeof(float);
	cudaError_t cu_err;
	cu_err = cudaMalloc((void**)&Mdevice.elements, size);
	if (cudaSuccess != cu_err)
	{
		std::cout << "Unable to cudaMalloc!" << std::endl;
		exit(0);
	}
	cudaMemset(&Mdevice.elements, 0, size); //initializes values to zero
	return Mdevice;
}


// Copy a host matrix to a device matrix.
inline void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
	int size = Mhost.width * Mhost.height * sizeof(float);
	Mdevice.height = Mhost.height;
	Mdevice.width = Mhost.width;
	Mdevice.pitch = Mhost.pitch;
	Mdevice.quantization = Mhost.quantization;
	cutilSafeCall( cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice));
}

// Copy a device matrix to a host matrix.
inline void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
	int size = Mdevice.width * Mdevice.height * sizeof(float);
	cutilSafeCall(cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost));
}


#endif // _MATRIX_CUDA_H_
