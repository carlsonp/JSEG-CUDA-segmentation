# JSEG-CUDA-segmentation
Partial implementation of JSEG image segmentation in CUDA


﻿CS 610 - GPU Programming
Patrick Carlson
JSEG Image Segmentation - Final Project
11/27/10

Requirements:

CUDA (on gpu1.cs.iastate.edu)
OpenCV (on gpu1.cs.iastate.edu)
CMake >= 2.8.1 (on gpu1.cs.iastate.edu)

Compiling:

I used CMake to generate the makefile.  Thanks to Ryan
for helping me with this.  I also used his cmake-modules
https://github.com/rpavlik/cmake-modules
which help look for OpenCV.  All his files are in the "cmake"
folder.

To build:

cd JSEG
mkdir "build"
cd build
ccmake ..

Specify the location of the NVIDIA CUDA SDK and whatever
else it asks for.  Then configure and assuming there are no
errors generate (which creates a makefile).

If you make a mistake, sometimes you have to delete the
CMakeCache.txt file and rerun cmake.

Then just run make.


Running the program:

Copy your image to the build directory so it's easier to run

./JSEG -i <input picture file> -m <input movie file> -q <number of quantization classes (<=64)> -w <window size in pixels (must be odd)> -g <run on GPU (default is false)>

Use GPU: (use -g true)
./JSEG -i picture.jpg -q 10 -w 9 -g true

Use CPU: (leave off -g parameter)
./JSEG -i picture.jpg -q 10 -w 9

Make sure -q <= 64 and -w is odd


Notes:

Sometimes I would get significantly different J value
results between the GPU and CPU.  However, after refactoring
some code and fixing some bugs I have not seen this issue resurface.
Perhaps I fixed it inadvertently?


Movie analysis is not finished.  For some reason, when I try to
initialize cvCapture and open up a file, it fails.  I'm not
sure if perhaps the version of OpenCV on gpu1 is old or what.
I tried both the C and CPP versions and both compiled but neither
worked.  So I gave up on that and just focused on single images.


Paper References

Deng, Y., & Manjunath, B.S. (2001).
Unsupervised segmentation of color-texture regions
in images and video. IEEE Transactions on Pattern
Analysis and Machine Intelligence, 23(8), 800-810.

Wang, Q., & Wang, Z. (2010). A Subjective
Method for Image Segmentation Evaluation.
Computer Vision - ACCV 2009, Lecture Notes in
Computer Science, 53-64.

