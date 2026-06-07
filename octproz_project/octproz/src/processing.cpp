/**
**  This file is part of OCTproZ.
**  OCTproZ is an open source software for processig of optical
**  coherence tomography (OCT) raw data.
**  Copyright (C) 2019-2022 Miroslav Zabic
**
**  OCTproZ is free software: you can redistribute it and/or modify
**  it under the terms of the GNU General Public License as published by
**  the Free Software Foundation, either version 3 of the License, or
**  (at your option) any later version.
**
**  This program is distributed in the hope that it will be useful,
**  but WITHOUT ANY WARRANTY; without even the implied warranty of
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
**  GNU General Public License for more details.
**
**  You should have received a copy of the GNU General Public License
**  along with this program. If not, see http://www.gnu.org/licenses/.
**
****
** Author:	Miroslav Zabic
** Contact:	zabic
**			at
**			spectralcode.de
****
**/

#include "processing.h"
#ifdef Q_OS_WIN
#include <windows.h>
#endif

Processing::Processing(){
	///qRegisterMetaType is needed to enabel Qt::QueuedConnection for signal slot communication with "AcquisitionParams"
	qRegisterMetaType<AcquisitionParams >("AcquisitionParams");
	qRegisterMetaType<OctAlgorithmParameters::RecordingParams >("OctAlgorithmParameters::RecordingParams");
	this->guiWinId = 0;
	this->glInteropPossible = true;
	this->bscanGlBufferRegisteredWithCuda = false;
	this->enfaceGlBufferRegisteredWithCuda = false;
	this->volumeGlBufferRegisteredWithCuda = false;
	this->buffersPerSecond = 0.0;
	this->isProcessing = false;
	this->recordingRawEnabled = false;
	this->surface = new QOffscreenSurface();
	this->context = new QOpenGLContext();
	this->octParams = OctAlgorithmParameters::getInstance();
	this->streamingBuffer = new AcquisitionBuffer();
	this->floatStreamingBuffer = new AcquisitionBuffer();
	this->rawRecorder = nullptr;
	this->processedRecorder = nullptr;
	this->currBufferNr = 0;
	this->gpu2HostStreamingEnabled = false;
	this->streamingBufferSizeInBytes = 0;
	this->floatStreamingEnabled = false;
	this->floatStreamingBufferSizeInBytes = 0;
	this->rawOnlyMode.store(0);

	this->rawRecorder = new Recorder("raw");
	this->rawRecorder->moveToThread(&recordingRawThread);
	connect(this, &Processing::initRawRecorder, this->rawRecorder, &Recorder::slot_init);
	connect(this, &Processing::rawData, this->rawRecorder, &Recorder::slot_record);
	connect(this, &Processing::processingDone, this->rawRecorder, &Recorder::slot_abortRecording);
	connect(this->rawRecorder, &Recorder::error, this, &Processing::error);
	connect(this->rawRecorder, &Recorder::info, this, &Processing::info);
	connect(this->rawRecorder, &Recorder::recordingDone, this, &Processing::rawRecordDone);
	connect(this, &Processing::preallocateRawRecorder, this->rawRecorder, &Recorder::slot_preallocate);
	connect(this, &Processing::freePreallocatedRawRecorder, this->rawRecorder, &Recorder::slot_freePreallocated);
	connect(&recordingRawThread, &QThread::finished, this->rawRecorder, &Recorder::deleteLater);
	recordingRawThread.start();

	this->processedRecorder = new Recorder("processed");
	this->processedRecorder->moveToThread(&recordingProcessedThread);
	Gpu2HostNotifier* notifier = Gpu2HostNotifier::getInstance();
	connect(this, &Processing::initProcessedRecorder, this->processedRecorder, &Recorder::slot_init);
	connect(notifier, &Gpu2HostNotifier::newGpuDataAvailable, this->processedRecorder, &Recorder::slot_record);
	connect(this, &Processing::processingDone, this->processedRecorder, &Recorder::slot_abortRecording);
	connect(this->processedRecorder, &Recorder::error, this, &Processing::error);
	connect(this->processedRecorder, &Recorder::info, this, &Processing::info);
	connect(this->processedRecorder, &Recorder::recordingDone, this, &Processing::processedRecordDone);
	connect(this, &Processing::preallocateProcessedRecorder, this->processedRecorder, &Recorder::slot_preallocate);
	connect(this, &Processing::freePreallocatedProcessedRecorder, this->processedRecorder, &Recorder::slot_freePreallocated);
	connect(&recordingProcessedThread, &QThread::finished, this->processedRecorder, &Recorder::deleteLater);
	recordingProcessedThread.start();
}

Processing::~Processing(){
	recordingProcessedThread.quit();
	recordingProcessedThread.wait();
	recordingRawThread.quit();
	recordingRawThread.wait();
	// unregister and release streaming buffers before cleanup.
	// AcquisitionBuffer::~AcquisitionBuffer() calls releaseMemory() which frees host memory,
	// but does not cudaHostUnregister first. 
	this->releaseGpu2HostStreamingResources();
	this->releaseFloatGpu2HostStreamingResources();
	delete this->context;
	delete this->streamingBuffer;
	delete this->floatStreamingBuffer;
	this->surface->deleteLater();
	cleanupCuda();
	qDebug() << "Processing destructor. Thread ID: " << QThread::currentThreadId();
}

void Processing::initCudaOpenGlInterop(){
	this->bscanGlBufferRegisteredWithCuda = false;
	this->enfaceGlBufferRegisteredWithCuda = false;
	this->volumeGlBufferRegisteredWithCuda = false;
	int peekInterval = 220; //this value was determined on a Jetson Nano through trial and error. Lower values can cause issues where CUDA registration for one, two or all OpenGL buffers does not occur at all.

#if defined(Q_OS_LINUX)
	this->surface->destroy(); //without destroying the surface here random segmentation faults in makeCurrent(this->surface) on Jetson Nano happen. Todo: investigate what is going on and test behavior across different operating systems.
#endif
	emit initOpenGLenFaceView();
	emit initOpenGL((this->context), this->surface, this->thread());
	this->waitForCudaOpenGlInteropReady(peekInterval, 4000); //this is necessary because initOpenGL(...) triggers a signal emission in glwindow2d containing the OpenGL buffer. This buffer is subsequently registered with CUDA in a slot within this processing class. Thus, this wait time and processEvents() serve as a workaround to ensure the slot executes - meaning the OpenGL buffer gets registered with CUDA - before continuation. //todo: re-examine how the steps for interoperability are called, this many signal slot connections for this straight forward task are too convoluted, probably there is an easyier way for the sequence: create QOffscreenSurface in GUI thread --> allocate OpenGL buffer --> register with cuda --> map buffer to get cuda pointer --> pass pointer to cuda kernel --> unmap pointer
}

void Processing::setGuiWindowId(quintptr id){
	this->guiWinId = id;
}

void Processing::setGlInteropPossible(bool possible){
	this->glInteropPossible = possible;
}

void Processing::wakeGuiThread(){
	//Post a Win32 message to the GUI thread's message queue to reliably wake it.
	//Qt's wakeUp() has an atomic guard that can make repeated calls no-ops,
	//and postEvent() relies on the same mechanism. PostMessage bypasses Qt
	//entirely and directly wakes MsgWaitForMultipleObjectsEx.
#ifdef Q_OS_WIN
	if (this->guiWinId) {
		PostMessage(reinterpret_cast<HWND>(this->guiWinId), WM_NULL, 0, 0);
		return;
	}
#endif
	QObject* guiTarget = QCoreApplication::instance();
	if (guiTarget) {
		QCoreApplication::postEvent(guiTarget, new QEvent(QEvent::Type(QEvent::User + 1)));
	}
}

bool Processing::waitForCudaOpenGlInteropReady(int interval, int timeout){
	QCoreApplication::processEvents();
	QElapsedTimer timer;
	timer.start();

	while (!this->isCudaOpenGlInteropReady()) {
		QCoreApplication::processEvents();
		this->wakeGuiThread(); //this fixes a bug that only occurs on some machines: when start button is pressed the app seems to freeze right after initialization and only continues when the mouse is moved. this is just a temp workaround until waitForCudaOpenGlInteropReady gets replaced by a better solution
		if (timer.elapsed() > timeout){
			emit error(tr("Cuda-OpenGL Interoperability initialization timeout. If no OCT output is displayed, try restarting the processing."));
			return false;
		}
		QThread::msleep(interval);
	}
	return true;
}

bool Processing::isCudaOpenGlInteropReady(){
	return (this->bscanGlBufferRegisteredWithCuda || !this->octParams->bscanViewEnabled) &&
		(this->enfaceGlBufferRegisteredWithCuda || !this->octParams->enFaceViewEnabled) &&
		(this->volumeGlBufferRegisteredWithCuda || !this->octParams->volumeViewEnabled) &&
		this->context->isValid() &&
			this->surface->isValid();
}

void Processing::blockBuffersForAcquisitionSystem(AcquisitionSystem* system) {
	for(int i = 0; i < system->buffer->bufferReadyArray.size(); ++i) {
		system->buffer->bufferReadyArray[i] = true;
	}
}

void Processing::unblockBuffersForAcquisitionSystem(AcquisitionSystem* system) {
	for(int i = 0; i < system->buffer->bufferReadyArray.size(); ++i) {
		system->buffer->bufferReadyArray[i] = false;
	}
}

bool Processing::initializeGpuProcessing(AcquisitionBuffer* buffer) {
	emit info(tr("GPU processing initialization..."));

	if (buffer == nullptr || buffer->bufferArray.size() < 2) {
		emit error(tr("GPU processing needs two acquisition buffers."));
		emit initializationFailed();
		return false;
	}

	if (this->glInteropPossible) {
		this->initCudaOpenGlInterop();
	} else {
		this->octParams->bscanViewEnabled = false;
		this->octParams->enFaceViewEnabled = false;
		this->octParams->volumeViewEnabled = false;
	}

	void* h_buffer1 = buffer->bufferArray[0];
	void* h_buffer2 = buffer->bufferArray[1];
	if(!initializeCuda(h_buffer1, h_buffer2, this->octParams)){
		emit error(tr("GPU buffer initialization failed."));
		emit initializationFailed();
		return false;
	}

	//init streaming if streamToHost option was already checked on startup
	if (this->octParams->streamToHost && !this->octParams->streamingParamsChanged) {
		this->enableGpu2HostStreaming(this->octParams->streamToHost);
	}

	emit info(tr("GPU processing initialized."));
	return true;
}

void Processing::slot_start(AcquisitionSystem* system){
	if (system != nullptr) {
		this->blockBuffersForAcquisitionSystem(system);

		AcquisitionBuffer* buffer = system->buffer;
		unsigned int width = 0;
		unsigned int height = 0;
		unsigned int depth = 0;
		unsigned int bitDepth = 0;
		unsigned int buffersPerVolume = 0;
		auto applyParams = [&](const AcquisitionParams& params) {
			width = params.samplesPerLine;
			height = params.ascansPerBscan;
			depth = params.bscansPerBuffer;
			bitDepth = params.bitDepth;
			buffersPerVolume = params.buffersPerVolume;
		};
		applyParams(system->params->params);
		bool gpuInitialized = false;
		bool rawOnlyModeActive = this->rawOnlyMode.load() != 0;

		if (!rawOnlyModeActive) {
			gpuInitialized = this->initializeGpuProcessing(buffer);
			if(!gpuInitialized){
				this->unblockBuffersForAcquisitionSystem(system);
				return;
			}
		} else {
			applyParams(system->getRawOnlyModeParams());
			emit info(tr("Raw only mode active. GPU processing initialization skipped."));
		}

		this->currBufferNr = buffersPerVolume-1;
		size_t bufferSizeInBytes = buffer != nullptr ? buffer->bytesPerBuffer : 0;
		emit updateInfoBox("0", "0", "0", "0", QString::number((qreal)bufferSizeInBytes / 1048576.0), "0");

		//timer for volumes/second calculation
		QElapsedTimer timer;
		timer.start();
		unsigned int processedBuffers = 0;

		emit initializationDone();
		this->unblockBuffersForAcquisitionSystem(system);
		bool previousRawOnlyModeActive = rawOnlyModeActive;

		//acquisition and processing loop
		while (system->acqusitionRunning) {
			rawOnlyModeActive = this->rawOnlyMode.load() != 0;
			buffer = system->buffer;

			if (rawOnlyModeActive) {
				applyParams(system->getRawOnlyModeParams());
			} else if (previousRawOnlyModeActive && gpuInitialized) {
				applyParams(system->params->params);
			}

			if (!rawOnlyModeActive && !gpuInitialized) {
				this->blockBuffersForAcquisitionSystem(system);
				buffer = system->buffer;
				gpuInitialized = this->initializeGpuProcessing(buffer);
				if(!gpuInitialized){
					this->unblockBuffersForAcquisitionSystem(system);
					return;
				}

				applyParams(system->params->params);
				this->currBufferNr = buffersPerVolume-1;
				bufferSizeInBytes = buffer->bytesPerBuffer;
				emit updateInfoBox("0", "0", "0", "0", QString::number((qreal)bufferSizeInBytes / 1048576.0), "0");
				this->unblockBuffersForAcquisitionSystem(system);
			}
			previousRawOnlyModeActive = rawOnlyModeActive;

			if (buffer == nullptr || buffer->bufferArray.size() < 1) {
				QCoreApplication::processEvents();
				continue;
			}

			int bufferPos = buffer->currIndex;
			if (bufferPos >= 0) {
				if (bufferPos < buffer->bufferArray.size() && bufferPos < buffer->bufferReadyArray.size() && buffer->bufferReadyArray[bufferPos]) {
					bufferSizeInBytes = buffer->bytesPerBuffer;

					//emit rawData signal to record raw data if recorder is enabled
					this->currBufferNr = (this->currBufferNr+1)%buffersPerVolume;
					emit rawData(buffer->bufferArray[bufferPos], bitDepth, width, height, depth, buffersPerVolume, this->currBufferNr);
					QCoreApplication::processEvents();

					if (!rawOnlyModeActive && gpuInitialized) {
						//apply stream-to-host changes before the CUDA pipeline can use them
						if (this->octParams->streamingParamsChanged) {
							this->enableGpu2HostStreaming(this->octParams->streamToHost);
							this->octParams->streamingParamsChanged = false;
						}

						//make OpenGL context current and process raw data on GPU
						this->context->makeCurrent(this->surface);
						octCudaPipeline(buffer->bufferArray[bufferPos]); //todo: wrap cuda functions in extra class such that oct processing implementations with other gpu/multi threading frameworks (OpenCL, OpenMP, C++ AMP) can be used interchangeably
						this->context->doneCurrent();
					}

					//set bufferReadyArray flag to false to indicate that acquisition system is allowed to reuse this buffer
					buffer->bufferReadyArray[bufferPos] = false;

					//volumes/second calculation every 5 seconds
					processedBuffers++;
					qreal elapsedTime = timer.elapsed();
					qreal captureInfoTime = 5000;
					if (elapsedTime >= captureInfoTime) {
						this->buffersPerSecond  = (qreal)processedBuffers / (elapsedTime / 1000.0);
						qreal volumesPerSecond = buffersPerSecond / static_cast<qreal>(buffersPerVolume);
						qreal bscansPerSecond = this->buffersPerSecond * (qreal)depth;
						qreal ascansPerSecond = bscansPerSecond * (qreal)height;
						qreal bufferSizeMB = (qreal)bufferSizeInBytes / 1048576.0; //1 Kilobyte is 1024 Bytes. 1 Megabyte is equal to 1024 Kilobytes or 1048576 Bytes
						qreal dataThroughput = this->buffersPerSecond * bufferSizeMB;
						emit updateInfoBox(QString::number(volumesPerSecond), QString::number(this->buffersPerSecond), QString::number(bscansPerSecond), QString::number(ascansPerSecond), QString::number(bufferSizeMB), QString::number(dataThroughput));
						processedBuffers = 0;
						timer.restart();
					}
				}
			}
			QCoreApplication::processEvents();
			this->isProcessing = true;
		}
		this->buffersPerSecond = 0;
		this->isProcessing = false;
		emit processingDone();
		emit updateInfoBox("0", "0", "0", "0", "0", "0");

		this->releaseGpu2HostStreamingResources();
		this->releaseFloatGpu2HostStreamingResources();
		if (gpuInitialized) {
			cleanupCuda();
		}
	}
}

void Processing::setRawOnlyMode(bool enabled) {
	this->rawOnlyMode.store(enabled ? 1 : 0);
}

void Processing::slot_enableRecording(OctAlgorithmParameters::RecordingParams recParams) {
	if (recParams.recordRaw) {
		if(this->rawRecorder->recordingEnabled) {
			emit error(tr("Recording of raw data is already running."));
		}else{
			emit initRawRecorder(recParams);
		}
	}
	if (recParams.recordProcessed) {
		if(this->processedRecorder->recordingEnabled) {
			emit error(tr("Recording of processed data is already running."));
		}else{
			OctAlgorithmParameters::RecordingParams recProcessedParams = recParams;
			int truncDiv = this->octParams->getOutputTruncationDivisor();
			if(recParams.saveAs32bitFloat){
				recProcessedParams.bufferSizeInBytes = (this->octParams->samplesPerLine / truncDiv) * this->octParams->ascansPerBscan * this->octParams->bscansPerBuffer * sizeof (float);
				//this->enableFloatGpu2HostStreaming(true);
			} else {
				recProcessedParams.bufferSizeInBytes = recProcessedParams.bufferSizeInBytes/truncDiv; //todo: add option to change bitdepth of processed recording
			}
			
			// Disconnect previous signals
			Gpu2HostNotifier* notifier = Gpu2HostNotifier::getInstance();
			disconnect(notifier, nullptr, this->processedRecorder, nullptr);
			disconnect(this->processedRecorder, &Recorder::readyToRecord, this, &Processing::enableFloatGpu2HostStreaming);

			// Connect the appropriate signal
			if (recParams.saveAs32bitFloat) {
				connect(notifier, &Gpu2HostNotifier::newGpuFloatDataAvailable, this->processedRecorder, &Recorder::slot_record);
				connect(this->processedRecorder, &Recorder::readyToRecord, this, &Processing::enableFloatGpu2HostStreaming);
			} else {
				connect(notifier, &Gpu2HostNotifier::newGpuDataAvailable, this->processedRecorder, &Recorder::slot_record);
			}

			emit initProcessedRecorder(recProcessedParams);
		}
	}
}

void Processing::slot_updateDisplayedBscanFrame(unsigned int frameNr, unsigned int displayFunctionFrames, int displayFunction){
	this->octParams->frameNr = frameNr;
	this->octParams->functionFramesBscan = displayFunctionFrames;
	this->octParams->displayFunctionBscan = displayFunction;

	if(this->isProcessing && this->buffersPerSecond > 0.0 && this->buffersPerSecond < LOW_FRAMERATE){
		this->context->makeCurrent(this->surface);
		changeDisplayedBscanFrame(frameNr, displayFunctionFrames, displayFunction);
		this->context->swapBuffers(this->surface);
		this->context->doneCurrent();
	}
}

void Processing::slot_updateDisplayedEnFaceFrame(unsigned int frameNr, unsigned int displayFunctionFrames, int displayFunction){
	this->octParams->frameNrEnFaceView = frameNr;
	this->octParams->functionFramesEnFaceView = displayFunctionFrames;
	this->octParams->displayFunctionEnFaceView = displayFunction;

	if(this->isProcessing && this->buffersPerSecond > 0.0 && this->buffersPerSecond < LOW_FRAMERATE){
		this->context->makeCurrent(this->surface);
		changeDisplayedEnFaceFrame(frameNr, displayFunctionFrames, displayFunction);
		this->context->swapBuffers(this->surface);
		this->context->doneCurrent();
	}
}

void Processing::slot_registerBscanOpenGLbufferWithCuda(unsigned int bufferId){
	if(this->context->makeCurrent(this->surface)){
		this->bscanGlBufferRegisteredWithCuda = cuda_registerGlBufferBscan(bufferId);
		this->context->doneCurrent();
	}
}

void Processing::slot_registerEnFaceViewOpenGLbufferWithCuda(unsigned int bufferId){
	if(this->context->makeCurrent(this->surface)){
		this->enfaceGlBufferRegisteredWithCuda = cuda_registerGlBufferEnFaceView(bufferId);
		this->context->doneCurrent();
	}
}

void Processing::slot_registerVolumeViewOpenGLbufferWithCuda(unsigned int bufferId){
	if(this->context->makeCurrent(this->surface)){
		this->volumeGlBufferRegisteredWithCuda = cuda_registerGlBufferVolumeView(bufferId);
		this->context->doneCurrent();
	}
}

void Processing::enableGpu2HostStreaming(bool enableStreaming) {
	if (enableStreaming) {
		unsigned int width = this->octParams->samplesPerLine;
		unsigned int height = this->octParams->ascansPerBscan;
		unsigned int depth = this->octParams->bscansPerBuffer;
		unsigned int bytesPerSample = ceil((double)(this->octParams->bitDepth) / 8.0); //todo: avoid this calculation here. put bytesPerSample in octsalgorithmparameters.
		int truncDiv = this->octParams->getOutputTruncationDivisor();
		size_t bufferSizeInBytes = (width / truncDiv) * height * depth * bytesPerSample;

		if (this->gpu2HostStreamingEnabled && this->streamingBufferSizeInBytes == bufferSizeInBytes) {
			emit streamingBufferEnabled(true);
			return; // already enabled with correct size
		}
		if (this->gpu2HostStreamingEnabled && this->streamingBufferSizeInBytes != bufferSizeInBytes && this->isProcessing) {
			emit error(tr("Changing stream-to-host buffer size during acquisition is not supported. Stop processing first."));
			return;
		}
		if (this->gpu2HostStreamingEnabled) {
			this->unregisterStreamingHostBuffers();
			this->streamingBuffer->releaseMemory();
			this->gpu2HostStreamingEnabled = false;
			this->streamingBufferSizeInBytes = 0;
		}

		this->streamingBuffer->allocateMemory(2, bufferSizeInBytes);
		this->registerStreamingHostBuffers(streamingBuffer->bufferArray.at(0), streamingBuffer->bufferArray.at(1), bufferSizeInBytes);
		this->gpu2HostStreamingEnabled = true;
		this->streamingBufferSizeInBytes = bufferSizeInBytes;
		emit streamingBufferEnabled(true);
		emit info(tr("GPU to Host-Ram Streaming enabled."));
	} else {
		// Runtime disable: stop consumers, but keep buffers alive until acquisition stops.
		emit streamingBufferEnabled(false);
		emit info(tr("GPU to Host-Ram Streaming disabled."));
	}
}

void Processing::enableFloatGpu2HostStreaming(bool enableStreaming) {
	if (enableStreaming) {
		unsigned int width = this->octParams->samplesPerLine;
		unsigned int height = this->octParams->ascansPerBscan;
		unsigned int depth = this->octParams->bscansPerBuffer;
		int truncDiv = this->octParams->getOutputTruncationDivisor();
		size_t bufferSizeInBytes = static_cast<size_t>(width / truncDiv) * height * depth * sizeof(float);

		if (this->floatStreamingEnabled && this->floatStreamingBufferSizeInBytes == bufferSizeInBytes) {
			return; // already enabled with correct size
		}
		if (this->floatStreamingEnabled && this->floatStreamingBufferSizeInBytes != bufferSizeInBytes && this->isProcessing) {
			emit error(tr("Changing float stream-to-host buffer size during acquisition is not supported. Stop processing first."));
			return;
		}
		if (this->floatStreamingEnabled) {
			this->unregisterFloatStreamingHostBuffers();
			this->floatStreamingBuffer->releaseMemory();
			this->floatStreamingEnabled = false;
			this->floatStreamingBufferSizeInBytes = 0;
		}

		this->floatStreamingBuffer->allocateMemory(2, bufferSizeInBytes);
		this->registerFloatStreamingHostBuffers(
			this->floatStreamingBuffer->bufferArray.at(0),
			this->floatStreamingBuffer->bufferArray.at(1),
			bufferSizeInBytes
		);
		this->floatStreamingEnabled = true;
		this->floatStreamingBufferSizeInBytes = bufferSizeInBytes;
	} else {
		// Runtime disable: keep buffers alive until acquisition stops.
	}
}

void Processing::releaseGpu2HostStreamingResources() {
	if (!this->gpu2HostStreamingEnabled) {
		return;
	}
	emit streamingBufferEnabled(false);
	QCoreApplication::processEvents();
	QThread::msleep(500);
	QCoreApplication::processEvents();
	this->unregisterStreamingHostBuffers();
	this->streamingBuffer->releaseMemory();
	this->gpu2HostStreamingEnabled = false;
	this->streamingBufferSizeInBytes = 0;
}

void Processing::releaseFloatGpu2HostStreamingResources() {
	if (!this->floatStreamingEnabled) {
		return;
	}
	this->unregisterFloatStreamingHostBuffers();
	this->floatStreamingBuffer->releaseMemory();
	this->floatStreamingEnabled = false;
	this->floatStreamingBufferSizeInBytes = 0;
}

void Processing::registerStreamingHostBuffers(void* h_streamingBuffer1, void* h_streamingBuffer2, size_t bytesPerBuffer) {
	cuda_registerStreamingBuffers(h_streamingBuffer1, h_streamingBuffer2, bytesPerBuffer);
}

void Processing::unregisterStreamingHostBuffers() {
	cuda_unregisterStreamingBuffers();
}

void Processing::registerFloatStreamingHostBuffers(void* h_streamingBuffer1, void* h_streamingBuffer2, size_t bytesPerBuffer) {
	cuda_registerFloatStreamingBuffers(h_streamingBuffer1, h_streamingBuffer2, bytesPerBuffer);
}

void Processing::unregisterFloatStreamingHostBuffers() {
	cuda_unregisterFloatStreamingBuffers();
}

void Processing::slot_preallocateRecordingBuffers(bool enabled) {
	if (enabled) {
		size_t rawSize = this->octParams->recParams.buffersToRecord * this->octParams->recParams.bufferSizeInBytes;
		if (rawSize > 0) {
			emit preallocateRawRecorder(rawSize);
		}
		int truncDiv = this->octParams->getOutputTruncationDivisor();
		size_t processedBufferSize;
		if (this->octParams->recParams.saveAs32bitFloat) {
			processedBufferSize = (this->octParams->samplesPerLine / truncDiv) * this->octParams->ascansPerBscan * this->octParams->bscansPerBuffer * sizeof(float);
		} else {
			processedBufferSize = this->octParams->recParams.bufferSizeInBytes / truncDiv;
		}
		size_t processedSize = processedBufferSize * this->octParams->recParams.buffersToRecord;
		if (processedSize > 0) {
			emit preallocateProcessedRecorder(processedSize);
		}
	} else {
		emit freePreallocatedRawRecorder();
		emit freePreallocatedProcessedRecorder();
	}
}
