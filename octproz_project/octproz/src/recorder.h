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

#ifndef RECORDER_H
#define RECORDER_H

#include <QObject>
#include <QFile>
#include "octproz_devkit.h"
#include <QCoreApplication>
#include <QDateTime>
#include "octalgorithmparameters.h"


class Recorder : public QObject
{
	Q_OBJECT

public:
	Recorder(QString name);
	~Recorder();
	
	bool recordingEnabled;
	bool recordingFinished;
	bool isRecording;



private:
	QString name;
	QString savePath;
	char* recBuffer;
	unsigned int recordedBuffers;
	bool initialized;
	OctAlgorithmParameters::RecordingParams currRecParams;

	void uninit();
	void saveToDisk();


public slots :	
	void slot_abortRecording();
	void slot_init(OctAlgorithmParameters::RecordingParams recParams);
	void slot_record(void* buffer, unsigned int bitDepth, unsigned int samplesPerLine, unsigned int linesPerFrame, unsigned int framesPerBuffer, unsigned int buffersPerVolume, unsigned int currentBufferNr);


signals :
	void info(QString info);
	void error(QString error);
	void recordingDone();
	void readyToRecord(bool);
};

#endif // RECORDER_H
