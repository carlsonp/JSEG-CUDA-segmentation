
#ifndef _JSEG_KERNEL_H_
#define _JSEG_KERNEL_H_


#include "matrix_cuda.h"

// CUDA includes
#include "cutil_inline.h"

// standard includes
#include <stdio.h>
#include <iostream>

const int BLOCKSIZE = 16;

// JSEG kernel thread
__global__ void JSEGKernel(Matrix M, Matrix P, int window_size)
{
	//Takes an input of quantization values in matrix
	//and returns an output matrix of j-values for each pixel
	
	unsigned int col = (blockIdx.x * blockDim.x) + threadIdx.x; //column
	unsigned int row = (blockIdx.y * blockDim.y) + threadIdx.y; //row

	// Make sure we're not outside the boundary of the image for calculations
	if (row >= M.height || col >= M.width)
		return;

	//We cannot dynamically allocate memory arrays of different sizes
	//in the kernel as it is not supported, the options are:
	//use shared memory (which can be dynamically allocated as part of the kernel call), 
	//use global memory with an indexing scheme that gives each thread a chunk of it, or declare a fixed length 
	//array which will be big enough to hold every case, and live with the wasted memory in cases where it is too large.
	//We cannot use shared memory in this case because the storage requirements are too high, so I went with
	//setting a fixed length array size

	float class_x_average[64];
	float class_y_average[64];
	float class_counter[64];
	float class_within_group_distance[64];
	for (unsigned int p = 0; p < 64; ++p)
	{
		//zero out matrices
		class_x_average[p] = 0;
		class_y_average[p] = 0;
		class_counter[p] = 0;
		class_within_group_distance[p] = 0;
	}

	int x_top_left = (col - ((window_size-1)/2));
	int y_top_left = (row - ((window_size-1)/2));

	//Iterate through the pixels to figure out the average position of each class
	for (int i = x_top_left; i <= (x_top_left + (window_size-1)); ++i)
	{
		for (int j = y_top_left; j <= (y_top_left + (window_size-1)); ++j)
		{
			if (i >= 0 && i < M.width && j >= 0 && j < M.height) //make sure we're in the boundary
			{
				//update average (just totaling at this point)
				class_x_average[(int)M.elements[(j * M.width) + i]] += i;
				class_y_average[(int)M.elements[(j * M.width) + i]] += j;
				class_counter[(int)M.elements[(j * M.width) + i]] += 1;
			}
		}
	}
	for (unsigned int i = 0; i < M.quantization; ++i)
	{
		if (class_counter[i] != 0)
		{
			class_x_average[i] = class_x_average[i] / class_counter[i];
			class_y_average[i] = class_y_average[i] / class_counter[i];
		}
	}

	//Iterate through the pixels to calculate each classes within group distance
	for (int i = x_top_left; i <= (x_top_left + (window_size-1)); ++i)
	{
		for (int j = y_top_left; j <= (y_top_left + (window_size-1)); ++j)
		{
			if (i >= 0 && i < M.width && j >= 0 && j < M.height) //make sure we're in the boundary
			{
				//update within group distance
				class_within_group_distance[(int)M.elements[(j * M.width) + i]] += pow((i - class_x_average[(int)M.elements[(j * M.width) + i]]), 2) + pow((j - class_y_average[(int)M.elements[(j * M.width) + i]]), 2); //calculate distance between two points
			}
		}
	}
	float final_within_group_distance = 0;
	for (unsigned int i = 0; i < M.quantization; ++i)
	{
		final_within_group_distance += class_within_group_distance[i];
	}
	final_within_group_distance = pow(final_within_group_distance, 2); //square value

	float class_between_group_distance = 0;

	//Iterate through each of the classes to calculate the between group distances
	//This is the distance between the average point of each class to the center (current pixel)
	for (unsigned int i = 0; i < M.quantization; ++i)
	{
		if (class_x_average[i] != 0 && class_y_average[i] != 0)
		{
			class_between_group_distance += pow((col - class_x_average[i]), 2) + pow((row - class_y_average[i]), 2); //calculate distance between two points
		}
	}
	class_between_group_distance = pow(class_between_group_distance, 2); //square value

	//store result in P
	if (final_within_group_distance != 0) //thou shalt not divide by zero
		P.elements[(row * M.width) + col] = (class_between_group_distance / final_within_group_distance);
	else
		P.elements[(row * M.width) + col] = 0;
}


void JSEGOnDevice(const Matrix M, Matrix P, int window_size)
{
	printf("Allocating device matrices\n");
	//Interface host call to the device kernel code and invoke the kernel
	Matrix Md = AllocateDeviceMatrix(M);
	Matrix Pd = AllocateDeviceMatrix(P);
	printf("Copying to device matrices\n");
	CopyToDeviceMatrix(Md, M);
	CopyToDeviceMatrix(Pd, P);

	printf("Done allocating matrices\n");
	cudaEvent_t start, stop;
	cutilSafeCall(cudaEventCreate(&start));
	cutilSafeCall(cudaEventCreate(&stop));

	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);

	//This ceiling takes into account the following three circumstances:
	//1) width < BLOCKSIZE so the grid size would be like .5,.5, so we round up
	//2) width/BLOCKSIZE is not evenly divisible, so we round up
	//3) width/BLOCKSIZE is evenly divisible, so rounding up does nothing
	std::cout << "Block dimensions: " << dimBlock.x << "," << dimBlock.y << std::endl;
	int gridsize_r = (int)ceil(M.height/(double)dimBlock.y);
	int gridsize_l = (int)ceil(M.width/(double)dimBlock.x);

	dim3 dimGrid(gridsize_l, gridsize_r);
	std::cout << "Grid dimensions: " << gridsize_l << "," << gridsize_r << std::endl;

	printf("About to call CUDA kernel\n");
	cudaEventRecord(start, 0);
	JSEGKernel<<<dimGrid, dimBlock>>>(Md, Pd, window_size);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	//cudaThreadSynchronize(); //<- probably not needed because cudaEventSynchronize does this?
	printf("Finished with CUDA kernel\n");
	CopyFromDeviceMatrix(P, Pd);

	float timeTaken = 0;
	cutilSafeCall( cudaEventElapsedTime(&timeTaken, start, stop));

	printf("GPU Time Taken = %f ms\n", timeTaken);

	// Free device matrices
	cudaFree(Md.elements);
	Md.elements = NULL;
	cudaFree(Pd.elements);
	Pd.elements = NULL;
}

#endif // #ifndef _JSEG_KERNEL_H_
