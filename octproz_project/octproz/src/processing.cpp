/**
**  This file is part of OCTproZ.
**  OCTproZ is an open source software for processig of optical
**  coherence tomography (OCT) raw data.
**  Copyright (C) 2019-2021 Miroslav Zabic
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
**			iqo.uni-hannover.de
****
**/

#include "processing.h"

Processing::Processing(){
	///qRegisterMetaType is needed to enabel Qt::QueuedConnection for signal slot communication with "AcquisitionParams"
	qRegisterMetaType<AcquisitionParams >("AcquisitionParams");
	qRegisterMetaType<RecordingParams >("RecordingParams");
	this->buffersPerSecond = 0.0;
	this->isProcessing = false;
	this->recordingRawEnabled = false;
	this->surface = new QOffscreenSurface();
	this->context = new QOpenGLContext();
	this->octParams = OctAlgorithmParameters::getInstance();
	this->streamingBuffer = new AcquisitionBuffer();
	this->rawRecorder = nullptr;
	this->processedRecorder = nullptr;
	this->currBufferNr = 0;


	this->rawRecorder = new Recorder("raw");
	this->rawRecorder->moveToThread(&recordingRawThread);
	connect(this, &Processing::initRawRecorder, this->rawRecorder, &Recorder::slot_init);
	connect(this, &Processing::rawData, this->rawRecorder, &Recorder::slot_record);
	connect(this, &Processing::processingDone, this->rawRecorder, &Recorder::slot_abortRecording);
	connect(this->rawRecorder, &Recorder::error, this, &Processing::error);
	connect(this->rawRecorder, &Recorder::info, this, &Processing::info);
	connect(this->rawRecorder, &Recorder::recordingDone, this, &Processing::rawRecordDone);
	connect(&recordingRawThread, &QThread::finished, this->rawRecorder, &Recorder::deleteLater);
	recordingRawThread.start();


	this->processedRecorder = new Recorder("processed");
	this->processedRecorder->moveToThread(&recordingProcessedThread);
	Gpu2HostNotifier* notifier = Gpu2HostNotifier::getInstance();
	connect(this, &Processing::initProcessedRecorder, this->processedRecorder, &Recorder::slot_init);
	connect(notifier, &Gpu2HostNotifier::newGpuDataAvailible, this->processedRecorder, &Recorder::slot_record);
	connect(this, &Processing::processingDone, this->processedRecorder, &Recorder::slot_abortRecording);
	connect(this->processedRecorder, &Recorder::error, this, &Processing::error);
	connect(this->processedRecorder, &Recorder::info, this, &Processing::info);
	connect(this->processedRecorder, &Recorder::recordingDone, this, &Processing::processedRecordDone);
	connect(&recordingProcessedThread, &QThread::finished, this->processedRecorder, &Recorder::deleteLater);
	recordingProcessedThread.start();
}

Processing::~Processing(){
	recordingProcessedThread.quit();
	recordingProcessedThread.wait();
	recordingRawThread.quit();
	recordingRawThread.wait();
	delete this->context;
	delete this->streamingBuffer;
	this->surface->deleteLater();
	cleanupCuda();
	qDebug() << "Processing destructor. Thread ID: " << QThread::currentThreadId();
}


void Processing::slot_start(AcquisitionSystem* system){
	if (system != nullptr) {
		//emit initOpenGL(&(this->context), &(this->surface), this->thread());
		emit info(tr("GPU processing initialization..."));
		emit initOpenGLenFaceView();
		emit initOpenGL((this->context), (this->surface), this->thread());
		QCoreApplication::processEvents();

		AcquisitionBuffer* buffer = system->buffer;
		void* h_buffer1 = buffer->bufferArray[0];
		void* h_buffer2 = buffer->bufferArray[1];
		unsigned int width = this->octParams->samplesPerLine;
		unsigned int height = this->octParams->ascansPerBscan;
		unsigned int depth = this->octParams->bscansPerBuffer;
		unsigned int bitDepth = this->octParams->bitDepth;
		unsigned int buffersPerVolume = this->octParams->buffersPerVolume;
		this->currBufferNr = buffersPerVolume-1;
		initializeCuda(h_buffer1, h_buffer2, this->octParams);

		//init streaming if streamToHost option was already checked on startup
		if (this->octParams->streamToHost && !this->octParams->streamingParamsChanged) {
			this->enableGpu2HostStreaming(this->octParams->streamToHost);
		}

		size_t bufferSizeInBytes = buffer->bytesPerBuffer;
		emit updateInfoBox("0", "0", "0", "0", QString::number((qreal)bufferSizeInBytes / 1048576.0), "0");

		//timer for volumes/second calculation
		QElapsedTimer timer;
		timer.start();
		unsigned int processedBuffers = 0;

		emit info(tr("GPU processing initialized."));
		emit initializationDone();

		//acquisition and processing loop
		while (system->acqusitionRunning) {
			int bufferPos = buffer->currIndex;
			if (bufferPos >= 0) {
				if (buffer->bufferReadyArray[bufferPos]) {
					//emit rawData signal to record raw data if recorder is enabled
					this->currBufferNr = (this->currBufferNr+1)%buffersPerVolume;
					emit rawData(buffer->bufferArray[bufferPos], bitDepth, width, height, depth, buffersPerVolume, this->currBufferNr);
					//QCoreApplication::processEvents(); //todo: check if processEvents is needed here (two cases: processing in gui thread, processing in extra thread)

					//make OpenGL context current and process raw data on GPU
					this->context->makeCurrent(this->surface);
					octCudaPipeline(buffer->bufferArray[bufferPos]); //todo: wrap cuda functions in extra class such that oct processing implementations with other gpu/multi threading frameworks (OpenCL, OpenMP, C++ AMP) can be used interchangeably
					this->context->swapBuffers(this->surface);
					this->context->doneCurrent();

					//set bufferReadyArray flag to false to indicate that acquisition system is allowed to reuse this buffer
					buffer->bufferReadyArray[bufferPos] = false;

					//volumes/second calculation every 5 seconds
					processedBuffers++;
					qreal elapsedTime = timer.elapsed();
					qreal captureInfoTime = 5000;
					if (elapsedTime >= captureInfoTime) {
						this->buffersPerSecond  = (qreal)processedBuffers / (elapsedTime / 1000.0);
						qreal volumesPerSecond = buffersPerSecond / static_cast<qreal>(this->octParams->buffersPerVolume);
						qreal bscansPerSecond = this->buffersPerSecond * (qreal)depth;
						qreal ascansPerSecond = bscansPerSecond * (qreal)height;
						qreal bufferSizeMB = (qreal)bufferSizeInBytes / 1048576.0; //1 Kilobyte is 1024 Bytes. 1 Megabyte is equal to 1024 Kilobytes or 1048576 Bytes
						qreal dataThroughput = this->buffersPerSecond * bufferSizeMB;
						emit updateInfoBox(QString::number(volumesPerSecond), QString::number(this->buffersPerSecond), QString::number(bscansPerSecond), QString::number(ascansPerSecond), QString::number(bufferSizeMB), QString::number(dataThroughput));
						processedBuffers = 0;
						timer.restart();
					}

					//gpu 2 host-ram streaming
					if (this->octParams->streamingParamsChanged) {
						this->enableGpu2HostStreaming(this->octParams->streamToHost);
						this->octParams->streamingParamsChanged = false;
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

		if (this->octParams->streamToHost) {
			this->enableGpu2HostStreaming(false);
		}
		cleanupCuda();
	}
}

void Processing::slot_enableRecording(bool enableRawRecording, bool enableProcessedRecording) {
	RecordingParams recParams;
	unsigned int width = this->octParams->samplesPerLine;
	unsigned int height = this->octParams->ascansPerBscan;
	unsigned int depth = this->octParams->bscansPerBuffer;
	unsigned int bytesPerSample = ceil((double)(this->octParams->bitDepth) / 8.0); //todo: avoid this calculation here. put bytesPerSample in octsalgorithmparameters.
	recParams.bufferSizeInBytes = width * height*depth*bytesPerSample;
	recParams.buffersToRecord = this->octParams->numberOfBuffersToRecord;
	recParams.fileName = Settings::getInstance()->recordSettings.value(REC_NAME).toString();
	if (recParams.fileName != "") {
		recParams.fileName = "_" + recParams.fileName;
	}
	recParams.savePath = Settings::getInstance()->recordSettings.value(REC_PATH).toString();
	recParams.startWithFirstBuffer = Settings::getInstance()->recordSettings.value(REC_START_WITH_FIRST_BUFFER).toBool(); //todo: do not use settings object to get recParams. emit recParams from octproz
	recParams.timeStamp = Settings::getInstance()->getTimestamp();
	
	if (enableRawRecording) {
		if(this->rawRecorder->recordingEnabled) {
			emit error(tr("Recording of raw data is already running."));
		}else{
			emit initRawRecorder(recParams);
		}
	}
	if (enableProcessedRecording) {
		if(this->processedRecorder->recordingEnabled) {
			emit error(tr("Recording of processed data is already running."));
		}else{
			RecordingParams recProcessedParams = recParams;
			recProcessedParams.bufferSizeInBytes = recProcessedParams.bufferSizeInBytes /2; //todo: add option to change bitdepth of processed recording
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
		cuda_registerGlBufferBscan(bufferId);
		this->context->doneCurrent();
	}
}

void Processing::slot_registerEnFaceViewOpenGLbufferWithCuda(unsigned int bufferId){
	if(this->context->makeCurrent(this->surface)){
		cuda_registerGlBufferEnFaceView(bufferId);
		this->context->doneCurrent();
	}
}

void Processing::slot_registerVolumeViewOpenGLbufferWithCuda(unsigned int bufferId){
	if(this->context->makeCurrent(this->surface)){
		cuda_registerGlBufferVolumeView(bufferId);
		this->context->doneCurrent();
	}
}

void Processing::enableGpu2HostStreaming(bool enableStreaming) {
	if (enableStreaming) {
		unsigned int width = this->octParams->samplesPerLine;
		unsigned int height = this->octParams->ascansPerBscan;
		unsigned int depth = this->octParams->bscansPerBuffer;
		unsigned int bytesPerSample = ceil((double)(this->octParams->bitDepth) / 8.0); //todo: avoid this calculation here. put bytesPerSample in octsalgorithmparameters.
		size_t bufferSizeInBytes = width * height*depth*bytesPerSample;
		this->streamingBuffer->releaseMemory();
		this->streamingBuffer->allocateMemory(2, bufferSizeInBytes);
		this->registerStreamingHostBuffers(streamingBuffer->bufferArray.at(0), streamingBuffer->bufferArray.at(1), bufferSizeInBytes);
		emit streamingBufferEnabled(true); //inform extensions (plug-ins) and PlotWindow1D that streaming of processed data is enabled
		emit info(tr("GPU to Host-Ram Streaming enabled."));
	}
	else {
		emit streamingBufferEnabled(false); //inform extensions (plug-ins) and PlotWindow1D that streaming of processed data is disabled
		//dirty workaround to ensure that all extensions and the 1d plot window are not accessing the streaming buffer anymore after it gets freed. todo: improve thread safety for buffer access, so that this workaround becomes unnecessary
		QCoreApplication::processEvents();
		QThread::msleep(500);
		QCoreApplication::processEvents();
		this->unregisterStreamingdHostBuffers();
		this->streamingBuffer->releaseMemory();
		emit info(tr("GPU to Host-Ram Streaming disabled."));
	}
}

void Processing::registerRecordHostBuffer(void* buffer, size_t size){
	cuda_registerProcessedRecordBuffer(buffer, size);
	octParams->recordingProcessedEnabled = true;
	octParams->copiedBuffers = 0;
}

void Processing::unregisterRecordHostBuffer(void* buffer) {
	cuda_unregisterProcessedRecordBuffer(buffer);
	octParams->recordingProcessedEnabled = false;
	octParams->copiedBuffers = 0;
}

void Processing::registerStreamingHostBuffers(void* h_streamingBuffer1, void* h_streamingBuffer2, size_t bytesPerBuffer) {
	cuda_registerStreamingBuffers(h_streamingBuffer1, h_streamingBuffer2, bytesPerBuffer);
}

void Processing::unregisterStreamingdHostBuffers() {
	cuda_unregisterStreamingBuffers();
}
