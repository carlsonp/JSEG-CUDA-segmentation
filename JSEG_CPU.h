
// Internal Includes
#include "matrix.h"

// standard includes
#include <stdio.h>
#include <iostream>
#include <math.h>


// JSEG on CPU test
inline Matrix JSEG_CPU(Matrix M, int window_size)
{
	//Takes an input of quantization values in matrix
	//and returns an output matrix of j-values for each pixel

	Matrix P;
	P.width = M.width;
	P.height = M.height;
	P.quantization = M.quantization;
	P.elements = (float*) malloc((M.height*M.width)*sizeof(float));
	if (P.elements == NULL)
	{
		std::cout << "Unable to malloc enough memory!" << std::endl;
		exit(0);
	}
	
	//std::cout << "Matrix width: " << M.width << " Matrix height: " << M.height << std::endl;
	//std::cout << "Window size: " << window_size << std::endl;
	for (unsigned int col = 0; col < M.width; ++col)
	{
		//std::cout << "col: " << col << std::endl;
		for (unsigned int row = 0; row < M.height; ++row)
		{
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

			//std::cout << "row: " << row << std::endl;
			int x_top_left = (col - ((window_size-1)/2));
			//std::cout << "X top left: " << x_top_left << std::endl;
			int y_top_left = (row - ((window_size-1)/2));
			//std::cout << "Y top left: " << y_top_left << std::endl;

			//Iterate through the pixels to figure out the average position of each class
			for (int i = x_top_left; i <= (x_top_left + (window_size-1)); ++i)
			{
				for (int j = y_top_left; j <= (y_top_left + (window_size-1)); ++j)
				{
					if (i >= 0 && i < M.width && j >= 0 && j < M.height) //make sure we're in the boundary
					{
						//std::cout << "Updating average for position: " << i << "," << j << std::endl;
						//std::cout << "At element number: " << (int)M.elements[(j * M.width) + i] << std::endl;
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
				//std::cout << "Class average: " << class_x_average[i] << "," << class_y_average[i] << std::endl;
			}

			//Iterate through the pixels to calculate each classes within group distance
			for (int i = x_top_left; i <= (x_top_left + (window_size-1)); ++i)
			{
				for (int j = y_top_left; j <= (y_top_left + (window_size-1)); ++j)
				{
					if (i >= 0 && i < M.width && j >= 0 && j < M.height) //make sure we're in the boundary
					{
						//update within group distance
						//std::cout << "We're at pixel position: " << i << "," << j << std::endl;
						//std::cout << "Class x average: " << class_x_average[(int)M.elements[(j * M.width) + i]] << " Class y average: " << class_y_average[(int)M.elements[(j * M.width) + i]] << std::endl;
						//std::cout << "Index point: " << (int)M.elements[(j * M.width) + i] << std::endl;
						class_within_group_distance[(int)M.elements[(j * M.width) + i]] += pow((i - class_x_average[(int)M.elements[(j * M.width) + i]]), 2) + pow((j - class_y_average[(int)M.elements[(j * M.width) + i]]), 2); //calculate distance between two points
						//std::cout << "Distance calculated: " << pow((i - class_x_average[(int)M.elements[(j * M.width) + i]]), 2) + pow((j - class_y_average[(int)M.elements[(j * M.width) + i]]), 2) << std::endl;
					}
				}
			}
			float final_within_group_distance = 0;
			for (unsigned int i = 0; i < M.quantization; ++i)
			{
				//std::cout << "Within group distance for class: " << i << " is: " << class_within_group_distance[i] << std::endl;
				final_within_group_distance += class_within_group_distance[i];
			}
			final_within_group_distance = pow(final_within_group_distance, 2); //square value
			//std::cout << "Within group ^2 distance: " << final_within_group_distance << std::endl;

			float class_between_group_distance = 0;

			//Iterate through each of the classes to calculate the between group distances
			//This is the distance between the average point of each class to the center (current pixel)
			for (unsigned int i = 0; i < M.quantization; ++i)
			{
				if (class_x_average[i] != 0 && class_y_average[i] != 0)
				{
					//std::cout << "Between group distance for class: " << i << " is: " << pow((col - class_x_average[i]), 2) + pow((row - class_y_average[i]), 2) << std::endl;
					class_between_group_distance += pow((col - class_x_average[i]), 2) + pow((row - class_y_average[i]), 2); //calculate distance between two points
				}
			}
			class_between_group_distance = pow(class_between_group_distance, 2); //square value
			//std::cout << "Between group ^2 distance: " << class_between_group_distance << std::endl;

			//store result in P
			if (final_within_group_distance != 0) //thou shalt not divide by zero
				P.elements[(row * M.width) + col] = (class_between_group_distance / final_within_group_distance);
			else
				P.elements[(row * M.width) + col] = 0;
			//std::cout << "J value for: " << col << "," << row << " is: " << P.elements[(row * M.width) + col] << std::endl;

			//stall before we run the next loop
			//int temp = 0;
			//std::cin >> temp;
		}
	}

	return P;
}
