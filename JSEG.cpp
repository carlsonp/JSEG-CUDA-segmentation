/*

CS 610 - GPU Programming
Patrick Carlson
JSEG Final Project
11/27/10

*/


// Internal includes
#include "matrix.h"
#include "JSEG_kernel.h"
#include "deviceQuery.h"
#include "JSEG_CPU.h"

// CUDA includes
#include "cutil_inline.h"


// OpenCV Includes
#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <cxcore.h>
#include <cvaux.h>

// System includes
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <string>
#include <boost/filesystem.hpp> //used to check to see if file exists


//Forward declarations
Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width);
Matrix JSEG_CPU(Matrix M, int window_size);
void printMatrix(Matrix m);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
void JSEGOnDevice(const Matrix M, Matrix P);


int main(int argc, char* argv[])
{
	int deviceCount = 0;
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess)
	{
		std::cout << "cudaGetDeviceCount FAILED CUDA Driver and Runtime version may be mismatched." << std::endl;
		exit(0);
	}

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
	{
		std::cout << "There is no device supporting CUDA" << std::endl;
		exit(0);
	}

	printDeviceInfo(0);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	unsigned int globalMem = deviceProp.totalGlobalMem; //in bytes
	
	IplImage* img; //original image will be stored here

	std::string input;
	bool movie = false;
	unsigned int quantization_classes = 0;
	unsigned int windowsize = 0;
	bool runGPU = false;

	std::cout << "[-i <input picture file> -m <input movie file> -q <number of quantization classes (<=64)> -w <window size in pixels (must be odd)> -g <run on GPU (default is false)>]" << std::endl;
	std::cout << "Example:" << std::endl;
	std::cout << argv[0] << " -i picture.jpg -q 10 -w 9 -g true" << std::endl;
	std::cout << argv[0] << " -m input.avi -q 10 -w 9 -g true" << std::endl;
	// Grab the parameters
	for (unsigned int i = 1; i < argc; i++) //start at 1 since we don't need where the program is running from
	{
		if (i + 1 != argc)
		{
			if (std::string(argv[i]) == "-i")
			{
				input = std::string(argv[i + 1]);
				movie = false;
			}
			else if (std::string(argv[i]) == "-m")
			{
				input = std::string(argv[i + 1]);
				movie = true;
			}
			else if (std::string(argv[i]) == "-q")
			{
				quantization_classes = atoi(argv[i + 1]);
			}
			else if (std::string(argv[i]) == "-w")
			{
				windowsize = atoi(argv[i + 1]);
			}
			else if (std::string(argv[i]) == "-g")
			{
				if (std::string(argv[i+1]) == "true")
					runGPU = true;
			}
		}
	}
	
	if (input.empty())
	{
		std::cout << "Error: You didn't specify a file for analysis." << std::endl;
		exit(0);
	}
	
	if (quantization_classes == 0)
	{
		std::cout << "Error: You didn't specify the number of quantization classes." << std::endl;
		exit(0);
	}
	else if (quantization_classes > 64)
	{
		std::cout << "Error: Sorry, we can only run with 64 or less quantization classes." << std::endl;
		exit(0);
	}
	
	if (windowsize == 0)
	{
		std::cout << "Error: You didn't specify the window size in pixels." << std::endl;
		exit(0);
	}
	
	if (windowsize % 2 == 0)
	{
		std::cout << "Error: The window size has to be odd." << std::endl;
		exit(0);
	}
	
  	if (!boost::filesystem::exists(input.c_str()))
  	{ 
		std::cout << "Error: The file doesn't exist." << std::endl;
		exit(0);
	}
	
	if (movie)
	{
		// OpenCV movie input
		// only some formats are supported:
		// http://opencv.willowgarage.com/wiki/VideoCodecs
		// additional info
		// http://www.cs.iit.edu/~agam/cs512/lect-notes/opencv-intro/opencv-intro.html#SECTION00070000000000000000
		// I'm not sure why this isn't working, I even tried the CPP way of opening video and it didn't work:
		// http://opencv.willowgarage.com/documentation/cpp/reading_and_writing_images_and_video.html

		std::cout << "Trying to open video file: " << input.c_str() << std::endl;
		CvCapture* capture = cvCreateFileCapture(input.c_str());
		if (!capture)
		{
			std::cout << "Something went wrong initializing capture from the video." << std::endl;
			exit(0);
		}
		
		int width = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
		int height = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
		unsigned int fps = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);
		std::cout << "Video FPS: " << fps << std::endl;
		std::cout << "Video Width: " << width << " Video Height: " << height << std::endl;

		//@TODO: video needs to be fixed
		std::cout << "Sorry, video is broken at the moment..." << std::endl;
		exit(0);
	}
	else
	{
		img = cvLoadImage(input.c_str());
		if (!img)
		{
			std::cout << "Something went wrong reading the image." << std::endl;
			exit(0);
		}
		
		std::cout << "Image Width: " << img->width << " Image Height: " << img->height << std::endl;
		
		cvShowImage("Input Image:", img);
		std::cout << "Press any key to continue..." << std::endl;
		cvWaitKey(0);		
	}
	
	// Matrices for the program
	Matrix M;
	Matrix P;

	cudaSetDevice( cutGetMaxGflopsDeviceId() );

	std::cout << "Available Global Memory: " << globalMem/1048576 << " MB" << std::endl;
	unsigned int memNeeded = (img->width * img->height)*sizeof(float) +
				(img->width * img->height)*sizeof(float); //in bytes
	std::cout << "Memory needed for matrices: " << memNeeded/1048576 << " MB" << std::endl;

	//Make sure there's enough global memory on the GPU device
	if (memNeeded > globalMem)
	{
		std::cout << "Exiting, we need more GPU global memory." << std::endl;
		exit(0);
	}

	std::cout << "Allocating matrices" << std::endl;
	M = AllocateMatrix(img->height, img->width, quantization_classes); //zeroed matrix
	P = AllocateMatrix(img->height, img->width, quantization_classes); //zeroed matrix
	
	// Figure out grayscale
	IplImage* grayscale = cvCreateImage(cvSize(img->width, img->height),IPL_DEPTH_8U, 1);
	cvCvtColor(img, grayscale, CV_BGR2GRAY); //convert to grayscale
	cvSmooth(grayscale, grayscale, CV_MEDIAN, 3, 3); //run median smoothing to fill in tiny spots
	cvShowImage("GrayScale Image:", grayscale);
	std::cout << "Press any key to continue..." << std::endl;
	cvWaitKey(0);
	
	// Figure out quantization and create image
	IplImage* quantization = cvCreateImage(cvSize(img->width, img->height),IPL_DEPTH_8U, 1);
	CvScalar s;
	for (unsigned int i = 0; i < img->width; i++)
	{
		for (unsigned int j = 0; j < img->height; j++)
		{
			s = cvGet2D(grayscale, j, i);
			int pixel = (int)s.val[0];
			//std::cout << "grayscale pixel value at: " << i << "," << j << " is: " << pixel << std::endl;
			//store quantization value into matrix data structure
			int quant = (int)round(pixel / (255 / quantization_classes));
			if (quant == quantization_classes) //make sure we don't go over bounds, since we're starting from 0
				quant--;
			//std::cout << "Quantization value for: " << i << "," << j << " is: " << quant << std::endl;
			M.set(j, i, quant);
			//Create the quantization image based on the quantization value for this pixel
			s.val[0] = quant * (255 / quantization_classes);
			//std::cout << "Quantization image pixel value at: " << i << "," << j << " is: " << s.val[0] << std::endl;
			cvSet2D(quantization, j, i, s);
		}
	}
	cvShowImage("Quantization Image:", quantization);
	std::cout << "Press any key to continue..." << std::endl;
	cvWaitKey(0);
	
	//print quantization matrix
	//M.printMatrix();

	if (runGPU == true)
	{
		std::cout << "---Running on GPU---" << std::endl;
		JSEGOnDevice(M, P, windowsize);
	}
	else
	{
		//Getting millisecond precision timing is a pain, see here:
		//http://stackoverflow.com/questions/588307/c-obtaining-milliseconds-time-on-linux-clock-doesnt-seem-to-work-properly
		//Is there a better way to do this?
		std::cout << "---Running on CPU---" << std::endl;
		struct timeval start, end;
		double mtime, seconds, useconds;
		gettimeofday(&start, NULL);

		P = JSEG_CPU(M, windowsize);

		gettimeofday(&end, NULL);

		seconds  = end.tv_sec  - start.tv_sec;
		useconds = end.tv_usec - start.tv_usec;

		mtime = ((seconds) * 1000 + useconds/1000.0f) + 0.5f;
		std::cout << "CPU Time Taken: " << mtime << " ms" << std::endl;
	}

	//P.printMatrix();

	//Convert J values back to grayscale so that we have something to look at
	//update min/max first
	P.updateMinMax();
	std::cout << "Min: " << P.min << " Max: " << P.max << std::endl;

	IplImage* JImage = cvCreateImage(cvSize(img->width, img->height),IPL_DEPTH_8U, 1);
	for (unsigned int i = 0; i < img->width; i++)
	{
		for (unsigned int j = 0; j < img->height; j++)
		{
			//std::cout << i << "," << j << ": J value: " << P.get(j,i) << std::endl;
			int new_pixel = (int)round(P.get(j,i) * (255 / (P.max  - P.min)));
			//std::cout << "New pixel value: " << new_pixel << std::endl;
			s = cvGet2D(JImage, j, i);
			s.val[0] = new_pixel;
			cvSet2D(JImage, j, i, s);
		}
	}
	cvShowImage("J Value Image:", JImage);
	std::cout << "Press any key to continue..." << std::endl;
	cvWaitKey(0);

	// Perform histogram equalization
	IplImage* JImage_equalized = cvCreateImage(cvSize(img->width, img->height),IPL_DEPTH_8U, 1);
	cvEqualizeHist(JImage, JImage_equalized);
	cvShowImage("J Value Image with Histogram Equalization:", JImage_equalized);
	std::cout << "Press any key to continue..." << std::endl;
	cvWaitKey(0);


	std::cout << "Cleaning up memory..." << std::endl;

	// Deallocate OpenCV items
	cvReleaseImage(&img);
	cvDestroyWindow("Input Image:");
	cvReleaseImage(&grayscale);
	cvDestroyWindow("GrayScale Image:");
	cvReleaseImage(&quantization);
	cvDestroyWindow("Quantization Image:");
	cvReleaseImage(&JImage);
	cvDestroyWindow("J Value Image:");
	cvReleaseImage(&JImage_equalized);
	cvDestroyWindow("J Value Image with Histogram Equalization:");
	

	// Free host matrices
	free(M.elements);
	M.elements = NULL;
	free(P.elements);
	P.elements = NULL;
}

