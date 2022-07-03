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
			this->centerPosition = 1.0;
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
	int width = static_cast<int>(this->fillFactor * this->size);
	int center = static_cast<int>(this->centerPosition * this->size);
	int minPos = center - width / 2;
	int maxPos = minPos + width;
	if (maxPos < minPos) {
		int tmp = minPos;
		minPos = maxPos;
		maxPos = tmp;
	}
	for (int i = 0; i < this->size; i++) {
		int xi = i - minPos;
		float xiNorm = (static_cast<float>(xi) / (static_cast<float>(width) - 1.0f));
		if (xiNorm > 0.999f || xiNorm < 0.0001f) {
			data[i] = 0.0;
		}
		else {
			data[i] = 1.0;
		}
	}
}

void WindowFunction::calculateHanning() {
	int width = static_cast<int>(this->fillFactor * this->size);
	int center = static_cast<int>(this->centerPosition * this->size);
	int minPos = center - width / 2;
	int maxPos = minPos + width;
	if (maxPos < minPos) {
		int tmp = minPos;
		minPos = maxPos;
		maxPos = tmp;
	}
	for (int i = 0; i < this->size; i++) {
		int xi = i - minPos;
		float xiNorm = (static_cast<float>(xi) / (static_cast<float>(width) - 1.0f));
		if (xiNorm > 0.999f || xiNorm < 0.0001f) {
			data[i] = 0.0;
		}
		else {
			data[i] = (0.5) * (1 - cos(2.0 * M_PI * (xiNorm)));
		}
	}
}

void WindowFunction::calculateGauss(){
	int center = static_cast<int>(this->centerPosition * this->size);
	for (int i = 0; i < this->size; i++) {
		int xi = i - center;
		float xiNorm = (static_cast<float>(xi) / (static_cast<float>(this->size) - 1.0f))/ (this->fillFactor);
		data[i] = expf(-10.0f*(powf(xiNorm,2.0f)));
	}
}

void WindowFunction::calculateSineWindow(){
	int width = static_cast<int>(this->fillFactor * this->size);
	int center = static_cast<int>(this->centerPosition * this->size);
	int minPos = center - width / 2;
	int maxPos = minPos + width;
	if (maxPos < minPos) {
		int tmp = minPos;
		minPos = maxPos;
		maxPos = tmp;
	}
	for (int i = 0; i < this->size; i++) {
		int xi = i - minPos;
		float xiNorm = (static_cast<float>(xi) / (static_cast<float>(width) - 1.0f));
		if (xiNorm > 0.999f || xiNorm < 0.0001f) {
			data[i] = 0.0;
		}
		else {
			data[i] = sin(M_PI * xiNorm);
		}
	}
}

void WindowFunction::calculateLanczosWindow() {
	int width = static_cast<int>(this->fillFactor * this->size);
	int center = static_cast<int>(this->centerPosition * this->size);
	int minPos = center - width / 2;
	int maxPos = minPos + width;
	if (maxPos < minPos) {
		int tmp = minPos;
		minPos = maxPos;
		maxPos = tmp;
	}
	for (int i = 0; i < this->size; i++) {
		int xi = i - minPos;
		float xiNorm = (static_cast<float>(xi) / (static_cast<float>(width) - 1.0f));
		if (xiNorm > 0.999f || xiNorm < 0.0001f) {
			data[i] = 0.0;
		}
		else {
			float argument = 2 * xiNorm - 1;
			if (argument == 0) {
				data[i] = 1;
			}
			else {
				data[i] = sin(M_PI * argument) / (M_PI * argument);
			}
		}
	}
}


void WindowFunction::calculateFlatTopWindow(){
	int width = static_cast<int>(this->fillFactor * this->size);
	int center = static_cast<int>(this->centerPosition * this->size);
	int minPos = center - width / 2;
	int maxPos = minPos + width;
	if (maxPos < minPos) {
		int tmp = minPos;
		minPos = maxPos;
		maxPos = tmp;
	}
	float a0 = 0.215578948;
	float a1 = 0.416631580;
	float a2 = 0.277263158;
	float a3 = 0.083578947;
	float a4 = 0.006947368;
	for (int i = 0; i < this->size; i++) {
		int xi = i - minPos;
		float xiNorm = (static_cast<float>(xi) / (static_cast<float>(width) - 1.0f));
		if (xiNorm > 0.999f || xiNorm < 0.0001f) {
			data[i] = 0.0;
		}
		else {
			data[i] = a0 - a1*cos(2*M_PI*xiNorm) + a2*cos(4*M_PI*xiNorm) - a3*cos(6*M_PI*xiNorm) +a4*cos(8*M_PI*xiNorm);
		}
	}
}

void WindowFunction::calculateTaylorWindow(){
//see: Doerry, Armin W. "Catalog of window taper functions for sidelobe control." Sandia National Laboratories (2017).
	int width = static_cast<int>(this->fillFactor * this->size);
	int center = static_cast<int>(this->centerPosition * this->size);
	int minPos = center - width / 2;
	int maxPos = minPos + width;
	if (maxPos < minPos) {
		int tmp = minPos;
		minPos = maxPos;
		maxPos = tmp;
	}
	for (unsigned i = 0; i < this->size; i++) {
		data[i] = 0.0;
	}

	float nbar = 7;
	float sidelobeLevel = -50;

	float nbarf = static_cast<float>(nbar);
	float nbarf2 = nbarf*nbarf;

	float eta = pow(10.0, - sidelobeLevel / 20.0);
	float a = acosh(eta) / M_PI;
	float a2 = a*a;
	float sigma2 = (nbarf2) / (a2 + (((nbarf - 0.5)*((nbarf - 0.5)))));

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
		float sign = pow(-1, m);
		numerator = numerator * sign;

		float Fm = (numerator / denominator);

		for (int i = 0; i < this->size; i++) {
			int xi = i - minPos;
			float xiNorm = (static_cast<float>(xi) / (static_cast<float>(width) - 1.0f));
			if (xiNorm > 0.999f || xiNorm < 0.0001f) {
				data[i] = -999.99;
			}
			else {
				data[i] += Fm * cos(mf * 2.0f * M_PI * xiNorm);
			}
		}
	}

	//normalize window to 1
	double maxVal = 0;
	double minVal = 100000;
	for (int i = 0; i < this->size; i++) {
		if(data[i]>maxVal){
		maxVal = data[i];
		}
		if(data[i] > -999 && data[i]<minVal){
		minVal = data[i];
		}
	}
	for (int i = 0; i < this->size; i++) {
		if(data[i] < -999){
			data[i] = minVal;
		}
		data[i] -= minVal;
		data[i] /= (maxVal-minVal);
	}

}
