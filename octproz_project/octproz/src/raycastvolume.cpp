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


/*!
 * \brief Create a two-unit cube mesh as the bounding box for the volume.
 */
RayCastVolume::RayCastVolume(void)
	: m_volume_texture {0}
	, m_noise_texture {0}
	, m_cube_vao {
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
	m_spacing.setX(1.0);
	m_spacing.setY(1.0);
	m_spacing.setZ(1.0);

	initializeOpenGLFunctions();
	//this->generateTestVolume();
	std::srand(std::time(nullptr)); //random numbers are used in createNoise()
}


/*!
 * \brief Destructor.
 */
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


/*!
 * \brief Load a volume from file.
 * \param File to be loaded.
 */
void RayCastVolume::generateTestVolume() {
	int elementsInVolume = 0;
	float* testData = nullptr;

	float x = 512/8;
	float y = 512/8;
	float z = 832/8;
	m_size = QVector3D(x, y, z);
	float x0 = x/2.0f;
	float y0 = y/2.0f;
	float z0 = z/2.0f;
	m_origin = QVector3D(x0, y0, z0);

	elementsInVolume = x*y*z;
	testData = (float*)malloc(elementsInVolume*sizeof(float));
	for(int i = 0; i < elementsInVolume; i++){
		//testData[i] = (static_cast<float>(i)/static_cast<float>(elementsInVolume))*1.0;
		float argument = static_cast<float>(i)/static_cast<float>(elementsInVolume/20.0);
		testData[i] = abs(pow(sinf(argument)/argument, 1))/1.1;
	}

	glDeleteBuffers(1, &m_volume_texture);
	glGenBuffers(1, &m_volume_texture);
	glBindBuffer(GL_TEXTURE_3D, m_volume_texture);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, (x * y * z*  sizeof(float)), 0, GL_DYNAMIC_COPY);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);  // The array on the host has 1 byte alignment
	glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, m_size.x(), m_size.y(), m_size.z(), 0, GL_RED, GL_FLOAT, testData);
	glBindTexture(GL_TEXTURE_3D, 0);

	free(testData);
}


/*!
 * \brief Create a noise texture with the size of the viewport.
 */
void RayCastVolume::createNoise(){
	GLint viewport[4];
	glGetIntegerv(GL_VIEWPORT, viewport);
	int width = viewport[2];
	int height = viewport[3];

	QVector<unsigned char>* noiseData = new QVector<unsigned char>(width*height);
	for(int i = 0; i<noiseData->size(); i++){
	   (*noiseData)[i] = static_cast<unsigned char>(std::rand() % 256);
	}

	glDeleteTextures(1, &m_noise_texture);
	glGenTextures(1, &m_noise_texture);
	glBindTexture(GL_TEXTURE_2D, m_noise_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, noiseData->data());
	glBindTexture(GL_TEXTURE_2D, 0);

	delete noiseData;
}


/*!
 * \brief Render the bounding box.
 */
void RayCastVolume::paint(void) {
	glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_3D, m_volume_texture);
	glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, m_noise_texture);

	m_cube_vao.paint();
}


/*!
 * \brief Scale factor to model space.
 *
 * Scale the bounding box such that the longest side equals 1.
 */
float RayCastVolume::scale_factor(void) {
	auto e = m_size * m_spacing;
	return std::max({e.x(), e.y(), e.z()});
}


void RayCastVolume::changeBufferAndTextureSize(unsigned int width, unsigned int height, unsigned int depth){
	m_size = {static_cast<float>(height), static_cast<float>(depth), static_cast<float>(width)};

	float x0 = 0;
	float y0 = 0;
	float z0 = 0;
	m_origin = {x0, y0, z0};

	glDeleteBuffers(1, &m_volume_texture);
	glGenTextures(1, &m_volume_texture);
	glBindTexture(GL_TEXTURE_3D, m_volume_texture);
	//glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, (width * height * depth*  sizeof(float)), 0, GL_DYNAMIC_COPY);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
	//glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, m_size.x(), m_size.y(), m_size.z(), 0, GL_RED, GL_FLOAT, NULL);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, m_size.x(), m_size.y(), m_size.z(), 0, GL_RED, GL_FLOAT, NULL);
	glBindTexture(GL_TEXTURE_3D, 0);

}
