/**
**  This file is part of OCTproZ.
**  OCTproZ is an open source software for processig of optical
**  coherence tomography (OCT) raw data.
**  Copyright (C) 2019-2020 Miroslav Zabic
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

#include "recorder.h"

Recorder::Recorder(QString name){
	this->name = name;
	this->savePath = "";
	this->recordingEnabled = false;
	this->recordingFinished = false;
	this->recordedBuffers = 0;
	this->receivedBuffers = 0;
	this->recordBuffer = new AcquisitionBuffer();
	this->initialized = false;
	this->currRecParams.savePath = "";
	this->currRecParams.buffersToSkip = 0;
	this->currRecParams.buffersToRecord = 0;
	this->currRecParams.bufferSizeInBytes = 0;
}

Recorder::~Recorder(){
	delete this->recordBuffer;
	qDebug() << "Recorder destructor. Thread ID: " << QThread::currentThreadId();
}

void* Recorder::getRecordBuffer(){
	return this->recordBuffer->bufferArray[0];
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
	this->recordBuffer->allocateMemory(1, this->currRecParams.buffersToRecord * this->currRecParams.bufferSizeInBytes); //todo: check if allocation of one big page aligned memory block is better or worse than allocateMemory(currRecParams.buffersToRecord, currRecParams.bufferSizeInBytes)
	this->savePath = this->currRecParams.savePath + "/" + this->currRecParams.timeStamp + this->currRecParams.fileName + "_" + this->name + ".raw";
	this->initialized = true;
	this->recordingFinished = false;
	this->recordingEnabled = true;
	emit info(tr("Recording initialized..."));
}

void Recorder::uninit(){
	this->recordBuffer->releaseMemory();
	this->initialized = false;
	this->recordingFinished = true;
	this->recordedBuffers = 0;
	this->receivedBuffers = 0;
	emit recordingDone();
}

void Recorder::slot_record(void* buffer, unsigned bitDepth, unsigned int samplesPerLine, unsigned int linesPerFrame, unsigned int framesPerBuffer, unsigned int buffersPerVolume, unsigned int currentBufferNr){
	if (!this->recordingEnabled) {
		return;
	}
	//check if initialization was done
	if (!this->initialized) {
		emit error(tr("Recording not possible. Record buffer not initialized."));
		return;
	}

	//check if received buffer should be skipped (time lapse recording)
	if (this->receivedBuffers % (this->currRecParams.buffersToSkip + 1) == 0){
		this->receivedBuffers = 0; //set to zero to avoid overflow
		//record/copy buffer to current position in recordBuffer
		char* recBufferPointer = (char*)(this->recordBuffer->bufferArray[0]);
		void* currPosiotionInRecBuffer = &(recBufferPointer[this->recordedBuffers * this->currRecParams.bufferSizeInBytes]);
		memcpy(currPosiotionInRecBuffer, buffer, this->currRecParams.bufferSizeInBytes);
		this->recordedBuffers++;

		//stop recording if enough buffers have been recorded, save recordBuffer to disk and release bufferArray memory
		if (this->recordedBuffers >= this->currRecParams.buffersToRecord) {
			this->recordingEnabled = false;
			this->saveToDisk();
			this->uninit();
		}
	}
	this->receivedBuffers++;
}

void Recorder::saveToDisk() {
	if (!this->initialized) {
		emit error(tr("Save recording to disk not possible. Record buffer not initialized."));
		return;
	}
	QString fileName = this->savePath;
	char* recBuffer = (char*)(this->recordBuffer->bufferArray[0]);
	QFile outputFile(fileName);
	if (!outputFile.open(QIODevice::WriteOnly)) {
		emit error(tr("Recording failed! Could not write file to disk."));
		return;
	}
	emit info(tr("Captured buffers: ") + QString::number(this->recordedBuffers) + "/" + QString::number(this->currRecParams.buffersToRecord));
	emit info(tr("Writing data to disk..."));
	QCoreApplication::processEvents();
	//outputFile.write(recBuffer, this->currRecParams.buffersToRecord * this->currRecParams.bufferSizeInBytes);
	outputFile.write(recBuffer, this->recordedBuffers * this->currRecParams.bufferSizeInBytes);
	outputFile.close();
	emit info(tr("Data written to disk! ") + fileName);
}
