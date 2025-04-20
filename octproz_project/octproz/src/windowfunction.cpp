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

#include "windowfunction.h"

WindowFunction::WindowFunction(WindowType type, float centerPosition, float fillFactor, unsigned int size)
{
	this->functionChanged = false;
	this->data = nullptr;
	this->type = Hanning;
	this->centerPosition = 0;
	this->fillFactor = 0;
	this->size = 0;
	this->setFunctionParams(type, centerPosition, fillFactor, size);
}

WindowFunction::WindowFunction() {
	this->functionChanged = false;
	this->data = nullptr;
	this->type = Hanning;
	this->centerPosition = 0;
	this->fillFactor = 0;
	this->size = 0;
}


WindowFunction::~WindowFunction() {
	if (this->data != nullptr) {
		free(this->data);
		this->data = nullptr;
	}
}

void WindowFunction::setFunctionParams(WindowType type, float centerPosition, float fillFactor, unsigned int size) {
	if (this->size != size) {
		this->setSize(size);
	}
	if (this->type != type || this->centerPosition != centerPosition || this->fillFactor != fillFactor) {
		this->type = type;

		if (centerPosition > 1) {
			this->centerPosition = 1.0f;
		}
		else if (centerPosition < 0) {
			this->centerPosition = 0;
		}
		else {
			this->centerPosition = centerPosition;
		}

		this->fillFactor = fillFactor;
		this->functionChanged = true;
	}
}

void WindowFunction::setSize(unsigned int size) {
	if (this->size != size) {
		this->data = static_cast<float*>(realloc(this->data, sizeof(float)*size)); //todo: check if data pointer is nullptr after realloc to check if realloc failed
		this->size = size;
		this->functionChanged = true;
	}
}

float * WindowFunction::getData() {
	if (this->functionChanged) {
		this->updateData();
		this->functionChanged = false;
	}
	return this->data;
}

void WindowFunction::updateData() {
	if (this->data != nullptr && this->size > 0) {
		switch (this->type) {
		case WindowType::Hanning:
			this->calculateHanning();
			break;
		case WindowType::Gauss:
			this->calculateGauss();
			break;
		case WindowType::Sine:
			this->calculateSineWindow();
			break;
		case WindowType::Lanczos:
			this->calculateLanczosWindow();
			break;
		case WindowType::Rectangular:
			this->calculateRectangular();
			break;
		case WindowType::FlatTop:
			this->calculateFlatTopWindow();
			break;
		}
	}
}

void WindowFunction::calculateRectangular() {
	unsigned int width = static_cast<unsigned int>(this->fillFactor * this->size);
	unsigned int center = static_cast<unsigned int>(this->centerPosition * this->size);
	int minPos = static_cast<int>(center - width / 2);
	int maxPos = minPos + static_cast<int>(width);
	if (maxPos < minPos) {
		int tmp = minPos;
		minPos = maxPos;
		maxPos = tmp;
	}
	for (unsigned int i = 0; i < this->size; i++) {
		int xi = static_cast<int>(i) - minPos;
		float xiNorm = (static_cast<float>(xi) / (static_cast<float>(width) - 1.0f));
		if (xiNorm > 0.999f || xiNorm < 0.0001f) {
			data[i] = 0.0f;
		}
		else {
			data[i] = 1.0f;
		}
	}
}

void WindowFunction::calculateHanning() {
	unsigned int width = static_cast<unsigned int>(this->fillFactor * this->size);
	unsigned int center = static_cast<unsigned int>(this->centerPosition * this->size);
	int minPos = static_cast<int>(center - width / 2);
	int maxPos = minPos + static_cast<int>(width);
	if (maxPos < minPos) {
		int tmp = minPos;
		minPos = maxPos;
		maxPos = tmp;
	}
	for (unsigned int i = 0; i < this->size; i++) {
		int xi = static_cast<int>(i) - minPos;
		float xiNorm = (static_cast<float>(xi) / (static_cast<float>(width) - 1.0f));
		if (xiNorm > 0.999f || xiNorm < 0.0001f) {
			data[i] = 0.0f;
		}
		else {
			data[i] = static_cast<float>(0.5 * (1.0 - cos(2.0 * M_PI * static_cast<double>(xiNorm))));
		}
	}
}

void WindowFunction::calculateGauss(){
	unsigned int center = static_cast<unsigned int>(this->centerPosition * this->size);
	for (unsigned int i = 0; i < this->size; i++) {
		int xi = static_cast<int>(i) - static_cast<int>(center);
		float xiNorm = (static_cast<float>(xi) / (static_cast<float>(this->size) - 1.0f))/ (this->fillFactor);
		data[i] = expf(-10.0f*(powf(xiNorm,2.0f)));
	}
}

void WindowFunction::calculateSineWindow(){
	unsigned int width = static_cast<unsigned int>(this->fillFactor * this->size);
	unsigned int center = static_cast<unsigned int>(this->centerPosition * this->size);
	int minPos = static_cast<int>(center - width / 2);
	int maxPos = minPos + static_cast<int>(width);
	if (maxPos < minPos) {
		int tmp = minPos;
		minPos = maxPos;
		maxPos = tmp;
	}
	for (unsigned int i = 0; i < this->size; i++) {
		int xi = static_cast<int>(i) - minPos;
		float xiNorm = (static_cast<float>(xi) / (static_cast<float>(width) - 1.0f));
		if (xiNorm > 0.999f || xiNorm < 0.0001f) {
			data[i] = 0.0f;
		}
		else {
			data[i] = static_cast<float>(sin(M_PI * static_cast<double>(xiNorm)));
		}
	}
}

void WindowFunction::calculateLanczosWindow() {
	unsigned int width = static_cast<unsigned int>(this->fillFactor * this->size);
	unsigned int center = static_cast<unsigned int>(this->centerPosition * this->size);
	int minPos = static_cast<int>(center - width / 2);
	int maxPos = minPos + static_cast<int>(width);
	if (maxPos < minPos) {
		int tmp = minPos;
		minPos = maxPos;
		maxPos = tmp;
	}
	for (unsigned int i = 0; i < this->size; i++) {
		int xi = static_cast<int>(i) - minPos;
		float xiNorm = (static_cast<float>(xi) / (static_cast<float>(width) - 1.0f));
		if (xiNorm > 0.999f || xiNorm < 0.0001f) {
			data[i] = 0.0f;
		}
		else {
			float argument = 2.0f * xiNorm - 1.0f;
			if (argument == 0.0f) {
				data[i] = 1.0f;
			}
			else {
				data[i] = static_cast<float>(sin(M_PI * static_cast<double>(argument)) / (M_PI * static_cast<double>(argument)));
			}
		}
	}
}


void WindowFunction::calculateFlatTopWindow(){
	unsigned int width = static_cast<unsigned int>(this->fillFactor * this->size);
	unsigned int center = static_cast<unsigned int>(this->centerPosition * this->size);
	int minPos = static_cast<int>(center - width / 2);
	int maxPos = minPos + static_cast<int>(width);
	if (maxPos < minPos) {
		int tmp = minPos;
		minPos = maxPos;
		maxPos = tmp;
	}
	float a0 = 0.215578948f;
	float a1 = 0.416631580f;
	float a2 = 0.277263158f;
	float a3 = 0.083578947f;
	float a4 = 0.006947368f;
	for (unsigned int i = 0; i < this->size; i++) {
		int xi = static_cast<int>(i) - minPos;
		float xiNorm = (static_cast<float>(xi) / (static_cast<float>(width) - 1.0f));
		if (xiNorm > 0.999f || xiNorm < 0.0001f) {
			data[i] = 0.0f;
		}
		else {
			data[i] = a0 - a1*static_cast<float>(cos(2.0*M_PI*static_cast<double>(xiNorm))) + 
                     a2*static_cast<float>(cos(4.0*M_PI*static_cast<double>(xiNorm))) - 
                     a3*static_cast<float>(cos(6.0*M_PI*static_cast<double>(xiNorm))) +
                     a4*static_cast<float>(cos(8.0*M_PI*static_cast<double>(xiNorm)));
		}
	}
}

void WindowFunction::calculateTaylorWindow(){
//see: Doerry, Armin W. "Catalog of window taper functions for sidelobe control." Sandia National Laboratories (2017).
	unsigned int width = static_cast<unsigned int>(this->fillFactor * this->size);
	unsigned int center = static_cast<unsigned int>(this->centerPosition * this->size);
	int minPos = static_cast<int>(center - width / 2);
	int maxPos = minPos + static_cast<int>(width);
	if (maxPos < minPos) {
		int tmp = minPos;
		minPos = maxPos;
		maxPos = tmp;
	}
	for (unsigned int i = 0; i < this->size; i++) {
		data[i] = 0.0f;
	}

	float nbar = 7.0f;
	float sidelobeLevel = -50.0f;

	float nbarf = nbar;
	float nbarf2 = nbarf*nbarf;

	float eta = static_cast<float>(pow(10.0, - static_cast<double>(sidelobeLevel) / 20.0));
	float a = static_cast<float>(acosh(static_cast<double>(eta)) / M_PI);
	float a2 = a*a;
	float sigma2 = (nbarf2) / (a2 + (((nbarf - 0.5f)*((nbarf - 0.5f)))));

	for (int m = 1; m < static_cast<int>(nbar); m++) {
		float numerator = 1.0f;
		float denominator = 1.0f;

		float mf = static_cast<float>(m);
		float mf2 = mf*mf;

		for (int n = 1; n < static_cast<int>(nbar); n++) {
			float nf = static_cast<float>(n);

			numerator *= (1.0f - ((mf*mf) / (sigma2)) / (a2 + ((nf - 0.5f)*(nf - 0.5f))));
			if (n != m) {
				denominator *= (1.0f - (mf2 / (nf*nf)));
			}
		}
		float sign = static_cast<float>(pow(-1.0, static_cast<double>(m)));
		numerator = numerator * sign;

		float Fm = (numerator / denominator);

		for (unsigned int i = 0; i < this->size; i++) {
			int xi = static_cast<int>(i) - minPos;
			float xiNorm = (static_cast<float>(xi) / (static_cast<float>(width) - 1.0f));
			if (xiNorm > 0.999f || xiNorm < 0.0001f) {
				data[i] = -999.99f;
			}
			else {
				data[i] += Fm * static_cast<float>(cos(static_cast<double>(mf) * 2.0 * M_PI * static_cast<double>(xiNorm)));
			}
		}
	}

	//normalize window to 1
	double maxVal = 0.0;
	double minVal = 100000.0;
	for (unsigned int i = 0; i < this->size; i++) {
		if(data[i]>static_cast<float>(maxVal)){
			maxVal = static_cast<double>(data[i]);
		}
		if(data[i] > -999.0f && data[i]<static_cast<float>(minVal)){
			minVal = static_cast<double>(data[i]);
		}
	}
	for (unsigned int i = 0; i < this->size; i++) {
		if(data[i] < -999.0f){
			data[i] = static_cast<float>(minVal);
		}
		data[i] -= static_cast<float>(minVal);
		data[i] /= static_cast<float>(maxVal-minVal);
	}
}