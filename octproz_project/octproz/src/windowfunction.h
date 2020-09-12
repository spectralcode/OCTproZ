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

#ifndef WINDOWFUNCTION_H
#define WINDOWFUNCTION_H

#ifndef _USE_MATH_DEFINES
	#define _USE_MATH_DEFINES
#endif
#include "math.h"
#include <stdlib.h>

class WindowFunction
{
public:
	enum WindowType {
		Hanning,
		Gauss,
		Sine,
		Lanczos,
		Rectangular
	};

	WindowFunction(WindowType type, float centerPosition, float fillFactor, unsigned int size);
	WindowFunction();
	~WindowFunction();

	void setFunctionParams(WindowType type, float centerPosition, float fillFactor, unsigned int size);
	void setSize(unsigned int size);
	unsigned int getSize() { return this->size; }
	float* getData();

private:
	void updateData();
	void calculateRectangular();
	void calculateHanning();
	void calculateGauss();
	void calculateSineWindow();
	void calculateLanczosWindow();

	float* data;
	WindowType type;
	float centerPosition;
	float fillFactor;
	unsigned int size;
	bool functionChanged;
};
#endif // WINDOWFUNCTION_H
