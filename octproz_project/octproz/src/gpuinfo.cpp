#include "gpuinfo.h"
#include <cuda_runtime.h>

GpuInfo::GpuInfo(QObject* parent) : QObject(parent)
{
}

bool GpuInfo::isCudaAvailable() {
	int deviceCount = 0;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);

	if(err != cudaSuccess){
		emit error(QString("CUDA error: %1").arg(cudaGetErrorString(err)));
		return false;
	}

	return deviceCount > 0;
}

QVector<GpuDeviceInfo> GpuInfo::getAllDevices() {
	QVector<GpuDeviceInfo> devices;
	int deviceCount = 0;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);

	if(err != cudaSuccess){
		emit error(QString("CUDA error: %1").arg(cudaGetErrorString(err)));
		return devices;
	}

	for(int i = 0; i < deviceCount; ++i){
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);

		size_t freeMem = 0;
		size_t totalMem = 0;
		cudaSetDevice(i);
		cudaMemGetInfo(&freeMem, &totalMem);

		GpuDeviceInfo info;
		info.deviceId = i;
		info.name = QString(prop.name);
		info.major = prop.major;
		info.minor = prop.minor;
		info.totalGlobalMem = prop.totalGlobalMem;
		info.freeGlobalMem = freeMem;
		info.totalConstMem = prop.totalConstMem;
		info.sharedMemPerBlock = prop.sharedMemPerBlock;
		info.sharedMemPerMultiprocessor = prop.sharedMemPerMultiprocessor;
		info.regsPerBlock = prop.regsPerBlock;
		info.regsPerMultiprocessor = prop.regsPerMultiprocessor;
		info.l2CacheSize = prop.l2CacheSize;
		info.multiProcessorCount = prop.multiProcessorCount;
		info.clockRate = prop.clockRate;
		info.memoryBusWidth = prop.memoryBusWidth;
		info.memoryClockRate = prop.memoryClockRate;
		info.warpSize = prop.warpSize;
		info.maxThreadsPerBlock = prop.maxThreadsPerBlock;
		info.maxThreadsPerMultiprocessor = prop.maxThreadsPerMultiProcessor;
		for(int d = 0; d < 3; ++d){
			info.maxThreadsDim[d] = prop.maxThreadsDim[d];
			info.maxGridSize[d] = prop.maxGridSize[d];
		}
		info.integrated = prop.integrated;
		info.managedMemory = prop.managedMemory;
		info.concurrentKernels = prop.concurrentKernels;
		info.canMapHostMemory = prop.canMapHostMemory;
		info.asyncEngineCount = prop.asyncEngineCount;

		devices.append(info);
	}

	return devices;
}
