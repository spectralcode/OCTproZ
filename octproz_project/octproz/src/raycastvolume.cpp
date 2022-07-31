//This file is a modified version of code originally created by Martino Pilia, please see: https://github.com/m-pilia/volume-raycasting

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

/*
 * Copyright © 2018 Martino Pilia <martino.pilia@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */


#include "raycastvolume.h"

#include <algorithm>
#include <cmath>
#include <QOpenGLTexture>
#include <QRandomGenerator>


RayCastVolume::RayCastVolume()
	: volumeTexture {0}
	, noiseTexture {0}
	, lutTexture {0}
	, cubeVao {
		  {
			  -1.0f, -1.0f,  1.0f,
			   1.0f, -1.0f,  1.0f,
			   1.0f,  1.0f,  1.0f,
			  -1.0f,  1.0f,  1.0f,
			  -1.0f, -1.0f, -1.0f,
			   1.0f, -1.0f, -1.0f,
			   1.0f,  1.0f, -1.0f,
			  -1.0f,  1.0f, -1.0f,
		  },
		  {
			  // front
			  0, 1, 2,
			  0, 2, 3,
			  // right
			  1, 5, 6,
			  1, 6, 2,
			  // back
			  5, 4, 7,
			  5, 7, 6,
			  // left
			  4, 0, 3,
			  4, 3, 7,
			  // top
			  2, 6, 7,
			  2, 7, 3,
			  // bottom
			  4, 5, 1,
			  4, 1, 0,
		  }
	  }
{
	this->spacing.setX(1.0);
	this->spacing.setY(1.0);
	this->spacing.setZ(1.0);

	initializeOpenGLFunctions();
}



RayCastVolume::~RayCastVolume()
{
}

void RayCastVolume::setLUT(QImage image) {
	const uchar* data = image.bits();
	int width = image.width();

	glDeleteBuffers(1, &lutTexture);
	glGenBuffers(1, &lutTexture);
	glBindBuffer(GL_TEXTURE_1D, lutTexture);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, width, 0, GL_BGRA, GL_UNSIGNED_BYTE, data);
	glBindTexture(GL_TEXTURE_1D, 0);
}

void RayCastVolume::generateTestVolume() {
	float x = 512/8;
	float y = 512/8;
	float z = 512/8;
	this->size = QVector3D(x, y, z);
	this->origin = QVector3D(0, 0, 0);
	this->spacing = QVector3D(1.0, 1.0, 1.0);

	int elementsInVolume = x*y*z;
	float* testData = (float*)malloc(elementsInVolume*sizeof(float));
	for(int i = 0; i < elementsInVolume; i++){
		float argument = static_cast<float>(i)/static_cast<float>(elementsInVolume/20.0);
		testData[i] = abs(pow(sinf(argument)/argument, 0.5));
	}

	glDeleteTextures(1, &this->volumeTexture);
	glGenTextures(1, &this->volumeTexture);
	glBindTexture(GL_TEXTURE_3D, this->volumeTexture);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);  // The array on the host has 1 byte alignment
	glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, this->size.x(), this->size.y(), this->size.z(), 0, GL_RED, GL_FLOAT, testData);
	glBindTexture(GL_TEXTURE_3D, 0);

	free(testData);
}

void RayCastVolume::createNoise() {
	GLint viewport[4];
	glGetIntegerv(GL_VIEWPORT, viewport);
	int width = qMin(qMax(viewport[2], 128), GL_MAX_TEXTURE_SIZE);
	int height = qMin(qMax(viewport[3], 128), GL_MAX_TEXTURE_SIZE);

	QVector<unsigned char>* noiseData = new QVector<unsigned char>(width*height);
	for(int i = 0; i<noiseData->size(); i++){
		(*noiseData)[i] = static_cast<unsigned char>(QRandomGenerator::global()->generate() % 256);
	}

	glDeleteTextures(1, &this->noiseTexture);
	glGenTextures(1, &this->noiseTexture);
	glBindTexture(GL_TEXTURE_2D, this->noiseTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, noiseData->data());
	glBindTexture(GL_TEXTURE_2D, 0);

	delete noiseData;
}

void RayCastVolume::paint() {
	glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_3D, this->volumeTexture);
	glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, this->noiseTexture);

	this->cubeVao.paint();
}

QVector3D RayCastVolume::extent() {
	auto e = this->size * this->spacing;
	return e / std::max({e.x(), e.y(), e.z()});
}

QMatrix4x4 RayCastVolume::modelMatrix(bool shift) {
	QMatrix4x4 modelMatrix;
	if (shift) {
		modelMatrix.translate(-this->origin / this->getScaleFactor());
	}
	modelMatrix.scale(0.5f * extent());
	return modelMatrix;
}

QVector3D RayCastVolume::top(bool shift) {
	auto t = extent() / 2.0;
	if (shift) {
		t -= this->origin / this->getScaleFactor();
	}
	return t;
}

QVector3D RayCastVolume::bottom(bool shift) {
	auto b = -extent() / 2.0;
	if (shift) {
		b -= this->origin / this->getScaleFactor();
	}
	return b;
}

float RayCastVolume::getScaleFactor() {
	auto e = this->size * this->spacing;
	return std::max({e.x(), e.y(), e.z()});
}

void RayCastVolume::changeTextureSize(unsigned int width, unsigned int height, unsigned int depth){
	this->size = {static_cast<float>(height), static_cast<float>(depth), static_cast<float>(width)};
	this->origin = {0, 0, 0};

	glDeleteTextures(1, &this->volumeTexture);
	glGenTextures(1, &this->volumeTexture);
	glBindTexture(GL_TEXTURE_3D, this->volumeTexture);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, this->size.x(), this->size.y(), this->size.z(), 0, GL_RED, GL_FLOAT, NULL);
	glBindTexture(GL_TEXTURE_3D, 0);
}
