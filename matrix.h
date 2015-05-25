
#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <cstdlib>
#include <iostream>

// Matrix Structure
class Matrix {
	public:
		Matrix() {
			//constructor
			min = 9999999999;
			max = 0;
		}
			
		void setDimension(int w, int h) {
			width = w;
			height = h;
		}
		
		void copyFrom(Matrix const & orig, float padVal = 0.0f) {
			for (unsigned int i = 0; i < height; ++i) {
				for (unsigned int j = 0; j < width; ++j) {
					float val = padVal;
					if (i < orig.height && j < orig.width) {
						val = orig.get(j, i);
					}
					get(j,i) = val;
				}			
			}
		
		}
		
		void set(unsigned int i, unsigned int j, float store) {
			elements[((i * width) + j)] = store; //i is row, j is column
		}
		
		float get(unsigned int i, unsigned int j) const {
			return elements[(i * width) + j]; //i is row, j is column
		}
		
		float & get(unsigned int i, unsigned int j) {
			return elements[(i * width) + j]; //i is row, j is column
		}

		void updateMinMax()
		{
			//goes through the elements and determines the smallest and largest value
			for (unsigned int i = 0; i < width; ++i)
			{
				for (unsigned int j = 0; j < height; ++j)
				{
					if (get(j,i) < min)
						min = get(j,i);
					if (get(j,i) > max)
						max = get(j,i);
				}
			}
		}
		
		// Prints the matrix to the screen
		void printMatrix()
		{
			for (unsigned int i = 0; i < width; ++i)
			{
				for (unsigned int j = 0; j < height; ++j)
				{
					std::cout << get(j,i) << ",";
				}
				std::cout << std::endl;
			}
		}

		//quantization class values (if applicable)
		unsigned int quantization;
		unsigned int width;
		unsigned int height;
		//min/max values, these get updated via the updateMinMax() method
		float min;
		float max;
		//number of elements between the beginnings of adjacent
		// rows in the memory layout (useful for representing sub-matrices)
		unsigned int pitch;
		//Pointer to the first element of the matrix represented
		float* elements;
};


// Allocate a random matrix of dimensions height*width
inline Matrix AllocateMatrix(int height, int width, int q)
{
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
	M.quantization = q;
    M.elements = NULL;
		
    M.elements = (float*) std::malloc(size*sizeof(float));

    for (unsigned int i = 0; i < M.height * M.width; ++i)
    {
		M.elements[i] = 0;
    }

    return M;
}

#endif // _MATRIX_H_
