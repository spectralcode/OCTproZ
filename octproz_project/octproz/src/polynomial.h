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
**			iqo.uni-hannover.de
****
**/

#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H

#include <stdlib.h>
#include <math.h>

class Polynomial
{
public:
	Polynomial(float* coeffs, unsigned int order, unsigned int size);
	Polynomial();
	~Polynomial();

	void setOrder(unsigned int order);
	unsigned int getOrder() { return this->order; }
	void setCoeffs(float* coeffs, unsigned int order);
	void setCoeff(float coeff, unsigned int coeffNr);
	float getCoeff(unsigned int coeffNr);
	void setSize(unsigned int size);
	unsigned int getSize() { return this->size; }
	float getValueAt(float x);
	float* getData();
	static void clamp(float* inputData, unsigned int inputLength, float min, float max);
	

private:
	void updateData();

	float* data;
	float* coeffs;
	unsigned int size;
	unsigned int order;
	bool polynomialChanged;
};
#endif // POLYNOMIAL_H
