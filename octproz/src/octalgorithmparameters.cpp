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

#include "octalgorithmparameters.h"
#include <stdio.h>

//////////////////////////////////////////////////////////////////////////
//			constructor (singleton pattern!), destructor				//
//////////////////////////////////////////////////////////////////////////
OctAlgorithmParameters* OctAlgorithmParameters::octAlgorithmParameters = nullptr;

OctAlgorithmParameters::OctAlgorithmParameters()
{
	this->acquisitionParamsChanged = false;
	this->resampling = false;
	this->dispersionCompensation = false;
	this->windowing = false;
	this->resamplingUpdated = false;
	this->useCustomResampleCurve = false;
	this->dispersionUpdated = false;
	this->windowUpdated = false;
	this->stopAfterRecord = false;
	this->recordingProcessedEnabled = false;
	this->streamingParamsChanged = true;
	this->streamToHost = false;
	this->buffersToSkip = 0;
	this->streamingBuffersToSkip = 0;
	this->numberOfBuffersToRecord = 0;
	this->copiedBuffers = 0;

	this->resampleCurve = nullptr;
	this->customResampleCurve = nullptr;
	this->resampleReferenceCurve = nullptr;
	this->dispersionCurve = nullptr;
	this->dispersionReferenceCurve = nullptr;
	this->windowCurve = nullptr;
	this->windowReferenceCurve = nullptr;

	this->resamplingCurveCalculator = new Polynomial();
	this->resamplingReferenceCurveCalculator = new Polynomial();
	this->dispersionCurveCalculator = new Polynomial();
	this->dispersionReferenceCurveCalculator = new Polynomial();
	this->windowCurveCalculator = new WindowFunction();
	this->windowReferenceCurveCalculator = new WindowFunction();

	this->fixedPatternNoiseRemoval = false;
	this->continuousFixedPatternNoiseDetermination = false;
	this->redetermineFixedPatternNoise = false;
	this->bscansForNoiseDetermination = 1;
	this->sinusoidalScanCorrection = false;

	this->resamplingInterpolation = INTERPOLATION::LINEAR;
	this->frameNr = 0;
	this->frameNrEnFaceView = 0;
	this->functionFramesEnFaceView = 0;
	this->functionFramesBscan = 0;
	this->displayFunctionBscan = 0;
	this->displayFunctionEnFaceView = 0;
	this->bscanViewEnabled = true;
	this->enFaceViewEnabled = true;
	this->volumeViewEnabled = false;

	this->samplesPerLine = 100;
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
		}else{
			this->resampleCurve =  this->customResampleCurve;
		}
		Polynomial::clamp(this->resampleCurve, this->samplesPerLine, 0, this->samplesPerLine-4); //resampling curve values shall remain between 0 and number of samples per line - 3 (a line is a raw A-scan). If a value is outside these boundaries the resampling (k-linearization) during processing will fail with a memory access violation //todo: rethink this approach, maybe there is a better way to avoid memeory access violation during interpolation in klinerization kernels
		this->resamplingUpdated = true;

		//update resample reference curve for plot in sidebar
		if(this->acquisitionParamsChanged){
			this->resamplingReferenceCurveCalculator->setSize(size);
			this->resamplingReferenceCurveCalculator->setCoeff(0, 0);
			this->resamplingReferenceCurveCalculator->setCoeff(1, 1);
			this->resampleReferenceCurve = this->resamplingReferenceCurveCalculator->getData();
			Polynomial::clamp(this->resampleReferenceCurve, this->samplesPerLine, 0, this->samplesPerLine-4);
		}
	}
}

void OctAlgorithmParameters::loadCustomResampleCurve(float* externalCurve, int size) {
	if(this->customResampleCurve != nullptr){
		free(this->customResampleCurve);
	}
	this->customResampleCurve = (float*)malloc(size*sizeof(float));
	for(int i = 0; i < size; i++){
		this->customResampleCurve[i] = externalCurve[i];
	}
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
