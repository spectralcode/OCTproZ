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
	frameNr(0),
	frameNrEnFaceView(0),
	functionFramesEnFaceView(0),
	functionFramesBscan(0),
	displayFunctionBscan(0),
	displayFunctionEnFaceView(0),
	bscanViewEnabled(true),
	enFaceViewEnabled(true),
	volumeViewEnabled(false),
	recParams{QString(), QString(), QString(), 0, 1, false, false, false, false, false, false},
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
}

void OctAlgorithmParameters::updateBufferSizeInBytes() {
	unsigned int bytesPerSample = ceil((double)(this->bitDepth) / 8.0);
	recParams.bufferSizeInBytes = bytesPerSample * this->samplesPerLine * this->ascansPerBscan * this->bscansPerBuffer;
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
		}
		Polynomial::clamp(this->resampleCurve, this->samplesPerLine, 0, this->samplesPerLine-3); //resampling curve values shall remain between 0 and number of samples per line - 3 (a line is a raw A-scan). If a value is outside these boundaries the resampling (k-linearization) during processing will fail with a memory access violation //todo: rethink this approach, maybe there is a better way to avoid memeory access violation during interpolation in klinerization kernels
		this->resamplingUpdated = true;

		//update resample reference curve for plot in sidebar
		if(this->acquisitionParamsChanged){
			this->resamplingReferenceCurveCalculator->setSize(size);
			this->resamplingReferenceCurveCalculator->setCoeff(0, 0);
			this->resamplingReferenceCurveCalculator->setCoeff(1, 1);
			this->resampleReferenceCurve = this->resamplingReferenceCurveCalculator->getData();
			Polynomial::clamp(this->resampleReferenceCurve, this->samplesPerLine, 0, this->samplesPerLine-3);
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
		int newSize = this->samplesPerLine/2;
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
