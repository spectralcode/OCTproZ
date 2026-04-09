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

#include "recorder.h"
#include <QDir>

Recorder::Recorder(QString name){
	this->name = name;
	this->savePath = "";
	this->recordingEnabled = false;
	this->recordingFinished = false;
	this->isRecording = false;
	this->recordedBuffers = 0;
	this->recBuffer = nullptr;
	this->initialized = false;
	this->preallocatedSize = 0;
	this->bufferPreallocated = false;
	this->currRecParams.savePath = "";
	this->currRecParams.buffersToRecord = 0;
	this->currRecParams.bufferSizeInBytes = 0;	
}

Recorder::~Recorder(){
	if(this->recBuffer != nullptr){
		free(this->recBuffer);
	}
	qDebug() << "Recorder destructor. Thread ID: " << QThread::currentThreadId();
}

void Recorder::slot_abortRecording(){
	if(this->recordingEnabled){
		if (!this->recordingFinished) {
			emit error(tr("Recording aborted!"));
			this->recordingEnabled = false;
			this->saveToDisk();
			this->uninit();
		}
		return;
	}
}

void Recorder::slot_init(OctAlgorithmParameters::RecordingParams recParams){
	this->currRecParams = recParams;

	QDir dir(this->currRecParams.savePath);
	if (this->currRecParams.savePath.isEmpty() || !dir.exists()) {
		emit error(tr("Recording not initialized: save path is empty or invalid."));
		this->uninit();
		return;
	}

	size_t neededSize = this->currRecParams.buffersToRecord * this->currRecParams.bufferSizeInBytes;
	if (this->bufferPreallocated && this->preallocatedSize == neededSize && this->recBuffer != nullptr) {
		// reuse preallocated buffer
	} else {
		if (this->recBuffer != nullptr) {
			free(this->recBuffer);
		}
		this->recBuffer = (char*)malloc(neededSize);
		this->bufferPreallocated = false;
		this->preallocatedSize = 0;
	}
	QString userSetFileName = this->currRecParams.fileName;
		if (userSetFileName != "") {
		userSetFileName = "_" + userSetFileName;
	}

	this->savePath = this->currRecParams.savePath + "/" + this->currRecParams.timestamp + userSetFileName + "_" + this->name + ".raw";
	this->initialized = true;
	this->recordingFinished = false;
	this->recordingEnabled = true;
	this->isRecording = false;
	emit readyToRecord(true);
	emit info(tr("Recording initialized..."));
}

void Recorder::uninit(){
	if (!this->bufferPreallocated) {
		free(this->recBuffer);
		this->recBuffer = nullptr;
	}
	this->initialized = false;
	this->recordingFinished = true;
	this->recordedBuffers = 0;
	emit readyToRecord(false);
	emit recordingDone();
}

void Recorder::slot_record(void* buffer, unsigned int bitDepth, unsigned int samplesPerLine, unsigned int linesPerFrame, unsigned int framesPerBuffer, unsigned int buffersPerVolume, unsigned int currentBufferNr){
	Q_UNUSED(bitDepth);
	Q_UNUSED(samplesPerLine);
	Q_UNUSED(linesPerFrame);
	Q_UNUSED(framesPerBuffer);
	Q_UNUSED(buffersPerVolume);

	if (!this->recordingEnabled) {
		return;
	}
	//check if initialization was done
	if (!this->initialized) {
		emit error(tr("Recording not possible. Record buffer not initialized."));
		return;
	}

	//check if recording should start with first buffer of volume
	if(this->currRecParams.startWithFirstBuffer && !this->isRecording && currentBufferNr != 0){
		return;
	}
	this->isRecording = true;

	//record/copy buffer to current position in recBuffer
	void* recBufferPointer = &(this->recBuffer[(this->recordedBuffers)*this->currRecParams.bufferSizeInBytes]);
	memcpy(recBufferPointer, buffer, this->currRecParams.bufferSizeInBytes);
	this->recordedBuffers++;

	//stop recording if enough buffers have been recorded, save recBuffer to disk and release reBuffer memory
	if (this->recordedBuffers >= this->currRecParams.buffersToRecord) {
		this->recordingEnabled = false;
		this->isRecording = false;
		this->saveToDisk();
		this->uninit();
	}
}

void Recorder::saveToDisk() {
	if (!this->initialized) {
		emit error(tr("Save recording to disk not possible. Record buffer not initialized."));
		return;
	}
	QString fileName = this->savePath;
	QFile outputFile(fileName);
	if (!outputFile.open(QIODevice::WriteOnly)) {
		emit error(tr("Recording failed! Could not write file to disk."));
		return;
	}
	emit info(tr("Captured buffers: ") + QString::number(this->recordedBuffers) + "/" + QString::number(this->currRecParams.buffersToRecord));
	emit info(tr("Writing data to disk..."));
	QCoreApplication::processEvents();
	outputFile.write(recBuffer, this->recordedBuffers * this->currRecParams.bufferSizeInBytes);
	outputFile.close();
	emit info(tr("Data written to disk! ") + fileName);
}

void Recorder::slot_preallocate(size_t totalBytes) {
	if (totalBytes == 0) {
		return;
	}
	if (this->recBuffer != nullptr && this->preallocatedSize == totalBytes) {
		return;
	}
	if (this->recBuffer != nullptr) {
		free(this->recBuffer);
	}
	this->recBuffer = (char*)malloc(totalBytes);
	if (this->recBuffer == nullptr) {
		this->preallocatedSize = 0;
		this->bufferPreallocated = false;
		emit error(tr("Failed to preallocate recording buffer (%1 MB)").arg(totalBytes / 1048576.0, 0, 'f', 1));
		return;
	}
	memset(this->recBuffer, 0, totalBytes); // touch every page to force physical memory allocation
	this->preallocatedSize = totalBytes;
	this->bufferPreallocated = true;
	emit info(tr("Recording buffer preallocated: %1 MB").arg(totalBytes / 1048576.0, 0, 'f', 1));
}

void Recorder::slot_freePreallocated() {
	if (this->bufferPreallocated && this->recBuffer != nullptr) {
		free(this->recBuffer);
		this->recBuffer = nullptr;
		emit info(tr("Preallocated recording buffer freed."));
	}
	this->preallocatedSize = 0;
	this->bufferPreallocated = false;
}
