#include "gpuinfowidget.h"

GpuInfoWidget::GpuInfoWidget(QWidget* parent)
	: QWidget(parent),
	  gpuInfo(new GpuInfo(this))
{
	this->initGui();
	this->setupConnections();

	this->refreshGpuInfo();

	this->setWindowTitle(tr("GPU Info"));
	this->setWindowFlags(this->windowFlags() | Qt::Tool);
}

GpuInfoWidget::~GpuInfoWidget()
{
}

bool GpuInfoWidget::checkCudaAvailability() {
	if (!this->gpuInfo->isCudaAvailable()) {
		emit error(tr("No CUDA-compatible GPU detected. OCT processing will not be available."));
		return false;
	}
	return true;
}

void GpuInfoWidget::initGui() {
	this->mainLayout = new QVBoxLayout(this);
	this->gpuInfoText = new QTextEdit();
	this->gpuInfoText->setReadOnly(true);

	this->mainLayout->addWidget(this->gpuInfoText);
}

void GpuInfoWidget::setupConnections() {
	connect(this->gpuInfo, &GpuInfo::info, this, &GpuInfoWidget::info);
	connect(this->gpuInfo, &GpuInfo::error, this, &GpuInfoWidget::error);
}

void GpuInfoWidget::showEvent(QShowEvent* event) {
	QWidget::showEvent(event);
	this->refreshGpuInfo();
}

void GpuInfoWidget::refreshGpuInfo() {
	this->gpuInfoText->clear();

	if (!this->gpuInfo->isCudaAvailable()) {
		this->gpuInfoText->append(tr("No CUDA-compatible GPU found."));
		return;
	}

	QVector<GpuDeviceInfo> devices = this->gpuInfo->getAllDevices();
	for (int i = 0; i < devices.size(); ++i) {
		const GpuDeviceInfo& device = devices[i];
		this->gpuInfoText->append(tr("<b>Device %1: %2</b>").arg(device.deviceId).arg(device.name));
		this->gpuInfoText->append(tr("  Compute Capability: %1.%2").arg(device.major).arg(device.minor));
		this->gpuInfoText->append(tr("  Total Global Memory: %1 MB").arg(device.totalGlobalMem / (1024 * 1024)));
		this->gpuInfoText->append(tr("  Free Global Memory: %1 MB").arg(device.freeGlobalMem / (1024 * 1024)));
		this->gpuInfoText->append(tr("  Constant Memory: %1 KB").arg(device.totalConstMem / 1024));
		this->gpuInfoText->append(tr("  Shared Memory per Block: %1 KB").arg(device.sharedMemPerBlock / 1024));
		this->gpuInfoText->append(tr("  Shared Memory per Multiprocessor: %1 KB").arg(device.sharedMemPerMultiprocessor / 1024));
		this->gpuInfoText->append(tr("  Registers per Block: %1").arg(device.regsPerBlock));
		this->gpuInfoText->append(tr("  Registers per Multiprocessor: %1").arg(device.regsPerMultiprocessor));
		this->gpuInfoText->append(tr("  L2 Cache Size: %1 KB").arg(device.l2CacheSize / 1024));
		this->gpuInfoText->append(tr("  Multiprocessors: %1").arg(device.multiProcessorCount));
		this->gpuInfoText->append(tr("  Core Clock Rate: %1 MHz").arg(device.clockRate / 1000));
		this->gpuInfoText->append(tr("  Memory Bus Width: %1 bits").arg(device.memoryBusWidth));
		this->gpuInfoText->append(tr("  Memory Clock Rate: %1 MHz").arg(device.memoryClockRate / 1000));
		this->gpuInfoText->append(tr("  Warp Size: %1").arg(device.warpSize));
		this->gpuInfoText->append(tr("  Max Threads per Block: %1").arg(device.maxThreadsPerBlock));
		this->gpuInfoText->append(tr("  Max Threads per Multiprocessor: %1").arg(device.maxThreadsPerMultiprocessor));
		this->gpuInfoText->append(tr("  Max Block Dimensions: [%1, %2, %3]").arg(device.maxThreadsDim[0]).arg(device.maxThreadsDim[1]).arg(device.maxThreadsDim[2]));
		this->gpuInfoText->append(tr("  Max Grid Dimensions: [%1, %2, %3]").arg(device.maxGridSize[0]).arg(device.maxGridSize[1]).arg(device.maxGridSize[2]));
		this->gpuInfoText->append(tr("  Integrated GPU: %1").arg(device.integrated ? "Yes" : "No"));
		this->gpuInfoText->append(tr("  Managed Memory Support: %1").arg(device.managedMemory ? "Yes" : "No"));
		this->gpuInfoText->append(tr("  Concurrent Kernels: %1").arg(device.concurrentKernels ? "Yes" : "No"));
		this->gpuInfoText->append(tr("  Can Map Host Memory: %1").arg(device.canMapHostMemory ? "Yes" : "No"));
		this->gpuInfoText->append(tr("  Async Engine Count: %1").arg(device.asyncEngineCount));
		this->gpuInfoText->append("");
	}
}
