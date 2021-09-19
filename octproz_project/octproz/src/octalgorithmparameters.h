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

#ifndef OCTALGORITHMPARAMETERS_H
#define OCTALGORITHMPARAMETERS_H

#include "polynomial.h"
#include "windowfunction.h"

#define FIXED_PATTERN_NOISE_REMOVAL_SEGMENTS 9


enum INTERPOLATION {
	LINEAR,
	CUBIC,
	LANCZOS
};


class OctAlgorithmParameters
{
public:
	/**
	* Constructor
	* @note Singleton pattern
	*
	**/
	static OctAlgorithmParameters* getInstance();


	/**
	* Destructor
	*
	**/
	~OctAlgorithmParameters();

	void updateResampleCurve();
	void updateDispersionCurve();
	void updateWindowCurve();
	void loadCustomResampleCurve(float *externalCurve, int size);

	//acquisition
	unsigned int samplesPerLine;
	unsigned int ascansPerBscan;
	unsigned int bscansPerBuffer;
	unsigned int buffersPerVolume;
	unsigned int bitDepth;
	bool acquisitionParamsChanged;
	
	//processing
	bool bitshift;	/// Activating/Deactivating bit shift. This is needed if 12 bit values are transported as 2 bytes (= 16 bit) from the Alazar digitizer board ATS9373 for example
	bool bscanFlip; ///	Activating/Deactivating flipping of every second B-scan. This is needed if B-scans are acquired in forward and backward scan direction
	bool signalLogScaling; /// This variable is for activating/deactivating log scaling in OCT signal processing
	bool sinusoidalScanCorrection; /// Activating/Deactivating sinusoidal scan correction (corrects image distortion due to sinus scan of fast axis scanner)
	float signalGrayscaleMin; /// User defined minimal signal value. Values greater than this will be cut off.
	float signalGrayscaleMax; /// User defined maximal signal value. Values smaller than this will be cut off.
	float signalMultiplicator; /// User defined multiplicator. Each signal value will be multiplied with this value. 
	float signalAddend;	/// User defined addend. This value will be added to each signal value. 

	float* resampleCurve;
	float* customResampleCurve;
	float* resampleReferenceCurve;
	float c0, c1, c2, c3;
	int resampleCurveLength;
	int curstomResampleCurveLength;
	bool resampling;
	bool resamplingUpdated;
	bool useCustomResampleCurve;
	INTERPOLATION resamplingInterpolation;

	float* dispersionCurve;
	float* dispersionReferenceCurve;
	float d0, d1, d2, d3;
	bool dispersionCompensation;
	bool dispersionUpdated;

	float* windowCurve;
	float* windowReferenceCurve;
	WindowFunction::WindowType window;
	float windowCenter, windowFillFactor;
	bool windowing;
	bool windowUpdated;

	bool fixedPatternNoiseRemoval;
	bool continuousFixedPatternNoiseDetermination;
	bool redetermineFixedPatternNoise;
	unsigned int bscansForNoiseDetermination;

	//visualization
	//todo: put all visualization params in a single struct and use an enum for displayfunction
	unsigned int frameNr; /// Current number of the displayed frame.
	unsigned int frameNrEnFaceView;
	unsigned int functionFramesEnFaceView;
	unsigned int functionFramesBscan;
	int displayFunctionBscan;
	int displayFunctionEnFaceView;
	enum DISPLAY_FUNCTION {
		AVERAGING,
		MIP
	};
	bool bscanViewEnabled;
	bool enFaceViewEnabled;
	bool volumeViewEnabled;

	//recording
	bool recordingProcessedEnabled;
	bool recordingBufferRegistered;
	bool stopAfterRecord;
	unsigned int numberOfBuffersToRecord;
	unsigned int copiedBuffers;

	//streaming
	bool streamingParamsChanged;
	bool streamToHost;
	unsigned int streamingBuffersToSkip;
	unsigned int currentBufferNr;


private:
	OctAlgorithmParameters(void);
	static OctAlgorithmParameters* octAlgorithmParameters;

	float* resizeCurve(float* curve, int currentSize, int newSize);


	Polynomial* resamplingCurveCalculator;
	Polynomial* resamplingReferenceCurveCalculator;
	Polynomial* dispersionCurveCalculator;
	Polynomial* dispersionReferenceCurveCalculator;
	WindowFunction* windowCurveCalculator;
	WindowFunction* windowReferenceCurveCalculator;
};


#endif // OCTALGORITHMPARAMETERS_H
