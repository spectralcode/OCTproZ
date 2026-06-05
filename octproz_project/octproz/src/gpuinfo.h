#ifndef GPUINFO_H
#define GPUINFO_H

#include <QObject>
#include <QString>
#include <QVector>
#include "cuda_to_hip.h"

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
// HIP device attribute type
#define cudaDeviceAttr hipDeviceAttribute_t
#else
#include <cuda_runtime.h>
#endif

struct GpuDeviceInfo {
	int deviceId;
	QString name;
	int major;
	int minor;

	size_t totalGlobalMem;
	size_t freeGlobalMem;
	size_t totalConstMem;
	size_t sharedMemPerBlock;
	size_t sharedMemPerMultiprocessor;
	int regsPerBlock;
	int regsPerMultiprocessor;
	int l2CacheSize;

	int multiProcessorCount;
	int clockRate;
	int memoryBusWidth;
	int memoryClockRate;

	int warpSize;
	int maxThreadsPerBlock;
	int maxThreadsPerMultiprocessor;
	int maxThreadsDim[3];
	int maxGridSize[3];

	bool integrated;
	bool managedMemory;
	bool concurrentKernels;
	bool canMapHostMemory;
	int asyncEngineCount;
};

class GpuInfo : public QObject
{
	Q_OBJECT

public:
	explicit GpuInfo(QObject* parent = nullptr);

	QVector<GpuDeviceInfo> getAllDevices();
	bool isCudaAvailable();

private:
	bool getAttr(int device, cudaDeviceAttr attr, int &out);

signals:
	void info(QString infoMessage);
	void error(QString errorMessage);
};

#endif // GPUINFO_H
