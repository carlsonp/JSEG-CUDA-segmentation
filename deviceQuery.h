//Standard includes
#include <iostream>

//Cuda includes
#include <cuda_runtime_api.h>

//Code modified from the deviceQuery example supplied with the Nvidia CUDA SDK
inline void printDeviceInfo(int dev)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	std::cout << "Using GPU device: " << dev << " - on a: " << deviceProp.name << std::endl;

#if CUDART_VERSION >= 2020
	int driverVersion = 0, runtimeVersion = 0;
	cudaDriverGetVersion(&driverVersion);
	std::cout << "CUDA Driver Version: " << (driverVersion/1000) << "." << (driverVersion%100) << std::endl;;
	cudaRuntimeGetVersion(&runtimeVersion);
	std::cout << "CUDA Runtime Version: " << (runtimeVersion/1000) << "." << (runtimeVersion%100) << std::endl;;
#endif
    std::cout << "CUDA Capability Major revision number: " << deviceProp.major << std::endl;
    std::cout << "CUDA Capability Minor revision number: " << deviceProp.minor << std::endl;

	std::cout << "Total amount of global memory: " << (deviceProp.totalGlobalMem/1048576) << " MB" << std::endl;
#if CUDART_VERSION >= 2000
    std::cout << "Number of multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
    //std::cout << "Number of cores: " << nGpuArchCoresPerSM[deviceProp.major] * deviceProp.multiProcessorCount << std::endl;
#endif
    std::cout << "Total amount of constant memory: " << deviceProp.totalConstMem << " bytes" << std::endl; 
    std::cout << "Total amount of shared memory per block: " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Total number of registers available per block: " << deviceProp.regsPerBlock << std::endl;
    std::cout << "Warp size: " << deviceProp.warpSize << std::endl;
    std::cout << "Maximum number of threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "Maximum sizes of each dimension of a block: " << deviceProp.maxThreadsDim[0] << " x " << deviceProp.maxThreadsDim[1] << " x " << deviceProp.maxThreadsDim[2] << std::endl;
    std::cout << "Maximum sizes of each dimension of a grid: " << deviceProp.maxGridSize[0] << " x " << deviceProp.maxGridSize[1] << " x " << deviceProp.maxGridSize[2] << std::endl;
    std::cout << "Maximum memory pitch: " << (deviceProp.memPitch/1048576) << " MB" << std::endl;
    std::cout << "Texture alignment: " << deviceProp.textureAlignment << " bytes" << std::endl;
    std::cout << "Clock rate: " << (deviceProp.clockRate * 1e-6f) << " GHz" << std::endl;
#if CUDART_VERSION >= 2000
    std::cout << "Concurrent copy and execution: " << (deviceProp.deviceOverlap ? "Yes" : "No") << std::endl;
#endif
#if CUDART_VERSION >= 2020
    std::cout << "Run time limit on kernels: " << (deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No") << std::endl;
    std::cout << "Integrated: " << (deviceProp.integrated ? "Yes" : "No") << std::endl;
    std::cout << "Support host page-locked memory mapping: " << (deviceProp.canMapHostMemory ? "Yes" : "No") << std::endl;
    std::cout << "Compute mode: " << (deviceProp.computeMode == cudaComputeModeDefault ?
		                                                            "Default (multiple host threads can use this device simultaneously)" :
	                                                                deviceProp.computeMode == cudaComputeModeExclusive ?
																	"Exclusive (only one host thread at a time can use this device)" :
	                                                                deviceProp.computeMode == cudaComputeModeProhibited ?
																	"Prohibited (no host thread can use this device)" :
																	"Unknown") << std::endl;
#endif
#if CUDART_VERSION >= 3000
    std::cout << "Concurrent kernel execution: " << (deviceProp.concurrentKernels ? "Yes" : "No") << std::endl;
#endif
#if CUDART_VERSION >= 3010
    std::cout << "Device has ECC support enabled: " << (deviceProp.ECCEnabled ? "Yes" : "No") << std::endl;
#endif

}
