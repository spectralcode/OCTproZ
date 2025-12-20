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

#include "octalgorithmparameters.h"
#include <stdio.h>
#include <QFile>

//////////////////////////////////////////////////////////////////////////
//			constructor (singleton pattern!), destructor				//
//////////////////////////////////////////////////////////////////////////
OctAlgorithmParameters* OctAlgorithmParameters::octAlgorithmParameters = nullptr;

OctAlgorithmParameters::OctAlgorithmParameters()
	: samplesPerLine(1024),
	ascansPerBscan(128),
	bscansPerBuffer(1),
	buffersPerVolume(1),
	bitDepth(8),
	acquisitionParamsChanged(false),
	bitshift(false),
	bscanFlip(false),
	signalLogScaling(false),
	sinusoidalScanCorrection(false),
	signalGrayscaleMin(0.0f),
	signalGrayscaleMax(60.0f),
	signalMultiplicator(1.0f),
	signalAddend(0.0f),
	backgroundRemoval(false),
	rollingAverageWindowSize(1),
	resampleCurve(nullptr),
	customResampleCurve(nullptr),
	resampleReferenceCurve(nullptr),
	c0(0.0f),
	c1(0.0f),
	c2(0.0f),
	c3(0.0f),
	resampleCurveLength(0),
	customResampleCurveLength(0),
	resampling(false),
	resamplingUpdated(false),
	useCustomResampleCurve(false),
	resamplingInterpolation(INTERPOLATION::LINEAR),
	dispersionCurve(nullptr),
	dispersionReferenceCurve(nullptr),
	d0(0.0f),
	d1(0.0f),
	d2(0.0f),
	d3(0.0f),
	dispersionCompensation(false),
	dispersionUpdated(false),
	windowCurve(nullptr),
	windowReferenceCurve(nullptr),
	window(WindowFunction::Rectangular),
	windowCenter(0.5f),
	windowFillFactor(1.0f),
	windowing(false),
	windowUpdated(false),
	fixedPatternNoiseRemoval(false),
	continuousFixedPatternNoiseDetermination(false),
	redetermineFixedPatternNoise(false),
	bscansForNoiseDetermination(1),
	postProcessBackgroundRemoval(false),
	postProcessBackgroundRecordingRequested(false),
	postProcessBackgroundWeight(1.0f),
	postProcessBackgroundOffset(0.0f),
	postProcessBackground(nullptr),
	postProcessBackgroundLength(0),
	postProcessBackgroundUpdated(false),
	fullRangeMode(false),
	fullRangeModeChanged(false),
	ccArtifactRemoval(false),
	ccRectCenterFreq(0.25f),
	ccRectWidth(0.5f),
	ccKeepPositiveSideband(true),
	backgroundFrameSubtraction(false),
	backgroundFrameRecordingRequested(false),
	backgroundFrameRecordingInProgress(false),
	backgroundFrameBscansToAverage(10),
	backgroundFrameBscansRecorded(0),
	backgroundFrame(nullptr),
	backgroundFrameSamplesPerLine(0),
	backgroundFrameAscansPerBscan(0),
	backgroundFrameValid(false),
	backgroundFrameUpdated(false),
	backgroundFrameFilePath(QString()),
	continuousBackgroundUpdate(false),
	continuousBackgroundUseEMA(true),
	frameNr(0),
	frameNrEnFaceView(0),
	functionFramesEnFaceView(0),
	functionFramesBscan(0),
	displayFunctionBscan(0),
	displayFunctionEnFaceView(0),
	bscanViewEnabled(true),
	enFaceViewEnabled(true),
	volumeViewEnabled(false),
	recParams{QString(), QString(), QString(), AUTO, 0, 1, false, false, false, false, false, false, false},
	streamingParamsChanged(true),
	streamToHost(false),
	streamingBuffersToSkip(0),
	currentBufferNr(0),
	resamplingCurveCalculator(new Polynomial()),
	resamplingReferenceCurveCalculator(new Polynomial()),
	dispersionCurveCalculator(new Polynomial()),
	dispersionReferenceCurveCalculator(new Polynomial()),
	windowCurveCalculator(new WindowFunction()),
	windowReferenceCurveCalculator(new WindowFunction())
{
	
}


OctAlgorithmParameters* OctAlgorithmParameters::getInstance() {
	octAlgorithmParameters = octAlgorithmParameters != nullptr ? octAlgorithmParameters : new OctAlgorithmParameters();
	return octAlgorithmParameters;
}

OctAlgorithmParameters::~OctAlgorithmParameters()
{
	delete this->resamplingCurveCalculator;
	delete this->resamplingReferenceCurveCalculator;
	delete this->dispersionCurveCalculator;
	delete this->dispersionReferenceCurveCalculator;
	delete this->windowCurveCalculator;
	delete this->windowReferenceCurveCalculator;

	if(this->customResampleCurve != nullptr){
		free(this->customResampleCurve);
	}
	if(this->backgroundFrame != nullptr){
		free(this->backgroundFrame);
	}
}

void OctAlgorithmParameters::updateBufferSizeInBytes() {
	unsigned int bytesPerSample = ceil((double)(this->bitDepth) / 8.0);
	this->recParams.bufferSizeInBytes = bytesPerSample * this->samplesPerLine * this->ascansPerBscan * this->bscansPerBuffer; //todo: replace bufferSizeInBytes by bufferSizeInBytesRaw and bufferSizeInBytesProcessed. then no bufferSize calculation is needed in processing class in slot_enableRecording
}

void OctAlgorithmParameters::updateResampleCurve() {
	unsigned int size = 0;
	if (this->resampling || this->acquisitionParamsChanged) {
		size = this->samplesPerLine;
		if (size <= 0) { return; }

		//update polynomial fit for resample curve if custom curve is not used
		if(!this->useCustomResampleCurve || this->customResampleCurve == nullptr){
			float c0 = this->c0;
			float c1 = this->c1 / static_cast<float>(size - 1);
			float c2 = this->c2 / powf(static_cast<float>((size - 1)), 2);
			float c3 = this->c3 / powf(static_cast<float>((size - 1)), 3);
			this->resamplingCurveCalculator->setSize(size);
			this->resamplingCurveCalculator->setCoeff(c0, 0);
			this->resamplingCurveCalculator->setCoeff(c1, 1);
			this->resamplingCurveCalculator->setCoeff(c2, 2);
			this->resamplingCurveCalculator->setCoeff(c3, 3);
			this->resampleCurve = this->resamplingCurveCalculator->getData();
			this->resampleCurveLength = size;
		}else{
			if(this->customResampleCurveLength != (int)this->samplesPerLine) {
				this->customResampleCurve = this->resizeCurve(this->customResampleCurve, this->customResampleCurveLength, (int)this->samplesPerLine);
			}
			this->resampleCurve = this->customResampleCurve;
			this->resampleCurveLength = this->customResampleCurveLength;
		}
		// Clamp resample curve values based on interpolation method to avoid memory access violations
		// Each interpolation method accesses different sample ranges around the index:
		// - Linear: accesses n and n+1, needs [0, samplesPerLine-2]
		// - Cubic: accesses n-1 to n+2, needs [1, samplesPerLine-3]
		// - Lanczos: accesses n-7 to n+8 (16-tap filter), needs [7, samplesPerLine-9]
		int clampMin = 0;
		int clampMax = static_cast<int>(this->samplesPerLine) - 2;
		if (this->resamplingInterpolation == INTERPOLATION::CUBIC) {
			clampMin = 1;
			clampMax = static_cast<int>(this->samplesPerLine) - 3;
		} else if (this->resamplingInterpolation == INTERPOLATION::LANCZOS) {
			clampMin = 7;
			clampMax = static_cast<int>(this->samplesPerLine) - 9;
		}
		Polynomial::clamp(this->resampleCurve, this->samplesPerLine, clampMin, clampMax);
		this->resamplingUpdated = true;

		//update resample reference curve for plot in sidebar
		if(this->acquisitionParamsChanged){
			this->resamplingReferenceCurveCalculator->setSize(size);
			this->resamplingReferenceCurveCalculator->setCoeff(0, 0);
			this->resamplingReferenceCurveCalculator->setCoeff(1, 1);
			this->resampleReferenceCurve = this->resamplingReferenceCurveCalculator->getData();
			Polynomial::clamp(this->resampleReferenceCurve, this->samplesPerLine, 0, static_cast<int>(this->samplesPerLine) - 2);
		}
	}
}

void OctAlgorithmParameters::loadCustomResampleCurve(float* externalCurve, int size) {
	if(this->customResampleCurve != nullptr){
		free(this->customResampleCurve);
	}
	this->customResampleCurve = (float*)malloc(size*sizeof(float));
	this->customResampleCurveLength = size;
	this->samplesPerLine = size; //todo: Reconsider if samplesPerLine should really be modified here. This might lead to a crash if a wrong file is loaded with more or fewer samples than expected.
	for(int i = 0; i < size; i++){
		this->customResampleCurve[i] = externalCurve[i];
	}
	this->resamplingUpdated = true;
}

void OctAlgorithmParameters::loadPostProcessingBackground(float* background, int size) {
	if(this->postProcessBackground != nullptr){
		free(this->postProcessBackground);
	}
	this->postProcessBackground = (float*)malloc(size*sizeof(float));
	this->postProcessBackgroundLength = size;
	for(int i = 0; i < size; i++){
		this->postProcessBackground[i] = background[i];
	}
	this->postProcessBackgroundUpdated = true;
}

void OctAlgorithmParameters::updateDispersionCurve(){
	unsigned int size = 0;
	if (this->dispersionCompensation || this->acquisitionParamsChanged) {
		size = this->samplesPerLine;
		if (size <= 0) { return; }
		float d0 = this->d0;
		float d1 = this->d1 / static_cast<float>(size - 1);
		float d2 = this->d2 / powf(static_cast<float>((size - 1)), 2);
		float d3 = this->d3 / powf(static_cast<float>((size - 1)), 3);

		this->dispersionCurveCalculator->setSize(size);
		this->dispersionCurveCalculator->setCoeff(d0, 0);
		this->dispersionCurveCalculator->setCoeff(d1, 1);
		this->dispersionCurveCalculator->setCoeff(d2, 2);
		this->dispersionCurveCalculator->setCoeff(d3, 3);
		this->dispersionCurve = this->dispersionCurveCalculator->getData();
		this->dispersionUpdated = true;

		//update dispersion reference curve for plot in sidebar
		if(this->acquisitionParamsChanged){
			this->dispersionReferenceCurveCalculator->setSize(size);
			this->dispersionReferenceCurveCalculator->setCoeff(0, 0);
			this->dispersionReferenceCurveCalculator->setCoeff(0, 1);
			this->dispersionReferenceCurve = this->dispersionReferenceCurveCalculator->getData();
		}
	}
}

void OctAlgorithmParameters::updateWindowCurve(){
	unsigned int size = 0;
	if (this->windowing || this->acquisitionParamsChanged) {
		size = this->samplesPerLine;
		if (size <= 0) { return; }
		this->windowCurveCalculator->setFunctionParams(this->window, this->windowCenter, this->windowFillFactor, size);
		this->windowCurve = this->windowCurveCalculator->getData();
		this->windowUpdated = true;

		//update window reference curve for plot in sidebar
		if(this->acquisitionParamsChanged){
			this->windowReferenceCurveCalculator->setFunctionParams(WindowFunction::Rectangular, 0.5, 1.0, size);
			this->windowReferenceCurve = this->windowReferenceCurveCalculator->getData();
		}
	}
}

void OctAlgorithmParameters::updatePostProcessingBackgroundCurve() {
	if (this->postProcessBackgroundRemoval || this->acquisitionParamsChanged) {
		int newSize = this->samplesPerLine / this->getOutputTruncationDivisor();
		if (newSize <= 0) { return; }

		if(this->postProcessBackgroundLength != newSize) {
			this->postProcessBackground = this->resizeCurve(this->postProcessBackground, this->postProcessBackgroundLength, newSize);
			this->postProcessBackgroundLength = newSize;
		}
	}
}

float* OctAlgorithmParameters::resizeCurve(float* curve, int currentSize, int newSize) {
	float* newCurve = (float*)realloc(curve, sizeof(float)*newSize); //todo: check if realloc failed
	if(newSize > currentSize){
		for(int i = currentSize; i < newSize; i++){
			newCurve[i] = 0;
		}
	}
	return newCurve;
}

bool OctAlgorithmParameters::saveBackgroundFrameToFile(const QString& filePath) {
	if (this->backgroundFrame == nullptr) {
		return false;
	}

	QFile file(filePath);
	if (!file.open(QIODevice::WriteOnly)) {
		return false;
	}

	// Write header (32 bytes)
	char magic[4] = {'B', 'G', 'F', 'R'};
	file.write(magic, 4);
	uint32_t version = 1;
	file.write(reinterpret_cast<char*>(&version), 4);
	file.write(reinterpret_cast<char*>(&this->backgroundFrameSamplesPerLine), 4);
	file.write(reinterpret_cast<char*>(&this->backgroundFrameAscansPerBscan), 4);
	uint32_t reserved = 0;
	file.write(reinterpret_cast<char*>(&reserved), 4);
	char reservedBytes[12] = {0};
	file.write(reservedBytes, 12);

	// Write data
	int dataSize = this->backgroundFrameSamplesPerLine * this->backgroundFrameAscansPerBscan * sizeof(float);
	file.write(reinterpret_cast<char*>(this->backgroundFrame), dataSize);
	file.close();

	this->backgroundFrameFilePath = filePath;
	return true;
}

bool OctAlgorithmParameters::loadBackgroundFrameFromFile(const QString& filePath) {
	QFile file(filePath);
	if (!file.open(QIODevice::ReadOnly)) {
		return false;
	}

	// Read and validate header
	char magic[4];
	file.read(magic, 4);
	if (magic[0] != 'B' || magic[1] != 'G' || magic[2] != 'F' || magic[3] != 'R') {
		file.close();
		return false;
	}

	uint32_t version;
	file.read(reinterpret_cast<char*>(&version), 4);
	if (version != 1) {
		file.close();
		return false;
	}

	uint32_t fileSamplesPerLine, fileAscansPerBscan;
	file.read(reinterpret_cast<char*>(&fileSamplesPerLine), 4);
	file.read(reinterpret_cast<char*>(&fileAscansPerBscan), 4);
	file.skip(16); // skip reserved bytes

	// Allocate and read data
	int frameSize = fileSamplesPerLine * fileAscansPerBscan;
	if (this->backgroundFrame != nullptr) {
		free(this->backgroundFrame);
	}
	this->backgroundFrame = (float*)malloc(frameSize * sizeof(float));
	if (this->backgroundFrame == nullptr) {
		file.close();
		return false;
	}
	file.read(reinterpret_cast<char*>(this->backgroundFrame), frameSize * sizeof(float));
	file.close();

	// Store dimensions
	this->backgroundFrameSamplesPerLine = fileSamplesPerLine;
	this->backgroundFrameAscansPerBscan = fileAscansPerBscan;
	this->backgroundFrameFilePath = filePath;

	this->backgroundFrameValid = false; // Must re-validate dimensions before use
	this->backgroundFrameUpdated = true;

	return true;
}

void OctAlgorithmParameters::updateBackgroundFrameValidity() {
	if (this->backgroundFrame == nullptr) {
		this->backgroundFrameValid = false;
		return;
	}

	// Check if stored dimensions match current acquisition dimensions
	this->backgroundFrameValid = (this->backgroundFrameSamplesPerLine == this->samplesPerLine &&
	                              this->backgroundFrameAscansPerBscan == this->ascansPerBscan);
}

void OctAlgorithmParameters::clearBackgroundFrame() {
	if (this->backgroundFrame != nullptr) {
		free(this->backgroundFrame);
		this->backgroundFrame = nullptr;
	}
	this->backgroundFrameSamplesPerLine = 0;
	this->backgroundFrameAscansPerBscan = 0;
	this->backgroundFrameValid = false;
	this->backgroundFrameUpdated = false;
	this->backgroundFrameFilePath = QString();
}
