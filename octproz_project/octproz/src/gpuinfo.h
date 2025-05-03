#ifndef GPUINFO_H
#define GPUINFO_H

#include <QObject>
#include <QString>
#include <QVector>

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

signals:
	void info(QString infoMessage);
	void error(QString errorMessage);
};

#endif // GPUINFO_H
