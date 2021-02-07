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

#include "polynomial.h"

Polynomial::Polynomial(float* coeffs, unsigned int order, unsigned int size){
	this->polynomialChanged = false;
	this->size = 0;
	this->order = 0;
	this->size = 0;
	this->data = nullptr;
	this->coeffs = nullptr;
	this->setSize(size);
	this->setCoeffs(coeffs, order);
}

Polynomial::Polynomial() {
	this->polynomialChanged = false;
	this->size = 0;
	this->order = 0;
	this->size = 0;
	this->data = nullptr;
	this->coeffs = nullptr;
	this->setOrder(1);
}

Polynomial::~Polynomial() {
	if (this->coeffs != nullptr) {
		free(this->coeffs);
		this->coeffs = nullptr;
	}
	if (this->data != nullptr) {
		free(this->data);
		this->data = nullptr;
	}
}

void Polynomial::setOrder(unsigned int order) {
	if (order != this->order) {
		this->coeffs = (float*)realloc(this->coeffs, sizeof(float)*(order + 1));
		this->order = order;
		this->polynomialChanged = true;
	}
}

void Polynomial::setCoeffs(float * coeffs, unsigned int order) {
	this->setOrder(order);
	for (unsigned int i = 0; i <= order; i++) {
		if (this->coeffs != nullptr) { this->coeffs[i] = coeffs[i]; }
	}
	this->polynomialChanged = true;
	//todo: set lower order coeffs to zero if they are not set to any value
}

void Polynomial::setCoeff(float coeff, unsigned int coeffNr) {
	//if coeff already exists change value
	if (this->order >= coeffNr) {
		this->coeffs[coeffNr] = coeff;
	}
	else {
		//coeff does not exist. extend polynomial. copy previous coeffs and add new one. fill all others with 0.
		this->coeffs = (float*)realloc(this->coeffs, sizeof(float)*(coeffNr + 1));
		for (unsigned int i = this->order + 1; i < (coeffNr); i++) {
			this->coeffs[i] = 0;
		}
		this->coeffs[coeffNr] = coeff;
		this->order = coeffNr;
	}
	this->polynomialChanged = true;
}

float Polynomial::getCoeff(unsigned int coeffNr) {
	return coeffNr <= this->order ? this->coeffs[coeffNr] : 0.0;
}

void Polynomial::setSize(unsigned int size) {
	if (this->size != size) {
		this->data = (float*)realloc(this->data, sizeof(float)*size); //todo: check if data pointer is nullptr after realloc to check if realloc failed
		this->size = size;
		this->polynomialChanged = true;
	}
}

float Polynomial::getValueAt(float x) {
	//Horner's method
	float result = 0.0;
	for (unsigned int i = 0; i <= this->order; i++) {
		int j = this->order - i;
		result = fma(result, x, coeffs[j]);
	}
	return result;
}

float* Polynomial::getData() {
	if (this->polynomialChanged) {
		this->updateData();
		this->polynomialChanged = false;
	}
	return this->data;
}

void Polynomial::clamp(float* inputData, unsigned int inputLength, float min, float max) {
	if (inputData != nullptr) {
		for (unsigned int i = 0; i < inputLength; i++) {
			if (inputData[i] < min){
				inputData[i] = min;
			}
			if (inputData[i] > max){
				inputData[i] = max;
			}
		}
	}
}

void Polynomial::updateData() {
	if (this->data != nullptr && this->coeffs != nullptr) {
		for (unsigned int i = 0; i < this->size; i++) {
			this->data[i] = this->getValueAt(i);
		}
	}
}
