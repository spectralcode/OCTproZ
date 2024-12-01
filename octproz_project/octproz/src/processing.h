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

#ifndef PROCESSING_H
#define PROCESSING_H

#include <QObject>
#include "polynomial.h"
#include "octproz_devkit.h"
#include "kernels.h"
#include "recorder.h"
#include "settings.h"
#include "octalgorithmparameters.h"
#include "gpu2hostnotifier.h"
#include <QCoreApplication>
#include <QElapsedTimer>
#include <QOffscreenSurface>
#include <QOpenGLContext>

#define LOW_FRAMERATE 12.5


class Processing : public QObject
{
	Q_OBJECT
	QThread recordingRawThread;
	QThread recordingProcessedThread;

public:
	Processing();
	~Processing();
	QOpenGLContext* context;
	QOffscreenSurface* surface;


private:
	void initCudaOpenGlInterop();
	bool waitForCudaOpenGlInteropReady(int interval, int timeout);
	bool isCudaOpenGlInteropReady();
	void blockBuffersForAcquisitionSystem(AcquisitionSystem* system);
	void unblockBuffersForAcquisitionSystem(AcquisitionSystem* system);

	bool bscanGlBufferRegisteredWithCuda;
	bool enfaceGlBufferRegisteredWithCuda;
	bool volumeGlBufferRegisteredWithCuda;
	qreal buffersPerSecond;
	bool isProcessing;
	OctAlgorithmParameters* octParams;
	bool recordingRawEnabled;
	Recorder* rawRecorder;
	Recorder* processedRecorder;
	AcquisitionBuffer* streamingBuffer;
	AcquisitionBuffer* floatStreamingBuffer; // for optional 32-bit float recording
	unsigned int currBufferNr;


public slots :
	//todo: decide if prefix "slot_" should be used or not and change naming of slots accordingly
	void slot_start(AcquisitionSystem* system);
	void slot_enableRecording(OctAlgorithmParameters::RecordingParams recParams);
	void slot_updateDisplayedBscanFrame(unsigned int frameNr, unsigned int displayFunctionFrames, int displayFunction);
	void slot_updateDisplayedEnFaceFrame(unsigned int frameNr, unsigned int displayFunctionFrames, int displayFunction);
	void slot_registerBscanOpenGLbufferWithCuda(unsigned int openGLbufferId);
	void slot_registerEnFaceViewOpenGLbufferWithCuda(unsigned int openGLbufferId);
	void slot_registerVolumeViewOpenGLbufferWithCuda(unsigned int openGLbufferId);
	void enableGpu2HostStreaming(bool enableStreaming);
	void enableFloatGpu2HostStreaming(bool enableStreaming);
	void registerStreamingHostBuffers(void* h_streamingBuffer1, void* h_streamingBuffer2, size_t bytesPerBuffer);
	void unregisterStreamingHostBuffers();
	void registerFloatStreamingHostBuffers(void* h_streamingBuffer1, void* h_streamingBuffer2, size_t bytesPerBuffer);
	void unregisterFloatStreamingHostBuffers();

signals :
	//void initOpenGL(QOpenGLContext** processingContext, QOffscreenSurface** processingSurface, QThread* processingThread);
	void initializationDone();
	void initializationFailed();
	void initOpenGL(QOpenGLContext* processingContext, QOffscreenSurface* processingSurface, QThread* processingThread);
	void initOpenGLenFaceView();
	void initRawRecorder(OctAlgorithmParameters::RecordingParams params);
	void initProcessedRecorder(OctAlgorithmParameters::RecordingParams params);
	void processingDone();
	void streamingBufferEnabled(bool enabled);

	void processedRecordDone();
	void rawRecordDone();
	void rawData(void* rawBuffer, unsigned bitDepth, unsigned int samplesPerLine, unsigned int linesPerFrame, unsigned int framesPerBuffer, unsigned int buffersPerVolume, unsigned int currentBufferNr);
	void info(QString info);
	void error(QString error);
	void updateInfoBox(QString volumesPerSecond, QString buffersPerSecond, QString bscansPerSecond, QString ascansPerSecond, QString bufferSizeMB, QString dataThroughput);
};

#endif // PROCESSING_H
