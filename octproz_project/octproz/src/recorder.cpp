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

Recorder::Recorder(QString name){
	this->name = name;
	this->savePath = "";
	this->recordingEnabled = false;
	this->recordingFinished = false;
	this->isRecording = false;
	this->recordedBuffers = 0;
	this->recBuffer = nullptr;
	this->initialized = false;
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

void Recorder::slot_init(RecordingParams recParams){
	this->currRecParams = recParams;
	this->recBuffer = (char*)malloc(this->currRecParams.buffersToRecord * this->currRecParams.bufferSizeInBytes);
	this->savePath = this->currRecParams.savePath + "/" + this->currRecParams.timeStamp + this->currRecParams.fileName + "_" + this->name + ".raw";
	this->initialized = true;
	this->recordingFinished = false;
	this->recordingEnabled = true;
	this->isRecording = false;
	emit info(tr("Recording initialized..."));
}

void Recorder::uninit(){
	free(this->recBuffer);
	this->recBuffer = nullptr;
	this->initialized = false;
	this->recordingFinished = true;
	this->recordedBuffers = 0;
	emit recordingDone();
}

void Recorder::slot_record(void* buffer, unsigned bitDepth, unsigned int samplesPerLine, unsigned int linesPerFrame, unsigned int framesPerBuffer, unsigned int buffersPerVolume, unsigned int currentBufferNr){
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
