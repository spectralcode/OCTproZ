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

#pragma once

#include <QMatrix4x4>
#include <QOpenGLExtraFunctions>
#include <QVector3D>

#include "mesh.h"
#include <ctime>

/*!
 * \brief Class for a raycasting volume.
 */
class RayCastVolume : protected QOpenGLExtraFunctions
{
public:
	RayCastVolume(void);
	virtual ~RayCastVolume();

	void setLUT(QImage image);
	void generateTestVolume();
	void createNoise();
	void paint();


	/*!
	 * \brief Get the extent of the volume.
	 * \return A vector holding the extent of the bounding box.
	 *
	 * The extent is normalised such that the longest side of the bounding
	 * box is equal to 1.
	 */
	QVector3D extent() {
		auto e = m_size * m_spacing;
		return e / std::max({e.x(), e.y(), e.z()});
	}

	/*!
	 * \brief Return the model matrix for the volume.
	 * \param shift Shift the volume by its origin.
	 * \return A matrix in homogeneous coordinates.
	 *
	 * The model matrix scales a two-unit side cube to the
	 * extent of the volume.
	 */
	QMatrix4x4 modelMatrix(bool shift = false) {
		QMatrix4x4 modelMatrix;
		if (shift) {
			modelMatrix.translate(-m_origin / scale_factor());
		}
		modelMatrix.scale(0.5f * extent());
		return modelMatrix;
	}

	/*!
	 * \brief Top planes forming the AABB.
	 * \param shift Shift the volume by its origin.
	 * \return A vector holding the intercept of the top plane for each axis.
	 */
	QVector3D top(bool shift = false) {
		auto t = extent() / 2.0;
		if (shift) {
			t -= m_origin / scale_factor();
		}
		return t;
	}

	/*!
	 * \brief Bottom planes forming the AABB.
	 * \param shift Shift the volume by its origin.
	 * \return A vector holding the intercept of the bottom plane for each axis.
	 */
	QVector3D bottom(bool shift = false) {
		auto b = -extent() / 2.0;
		if (shift) {
			b -= m_origin / scale_factor();
		}
		return b;
	}


	void changeBufferAndTextureSize(unsigned int width, unsigned int height, unsigned int depth);


	void setStretch(float x, float y, float z){m_spacing.setX(x); m_spacing.setY(y); m_spacing.setZ(z);}
	float getStretchX(){return m_spacing.x();}
	float getStretchY(){return m_spacing.y();}
	float getStretchZ(){return m_spacing.z();}

	GLuint getVolumeTexture(){return this->m_volume_texture;}

	void setOrigin(float x, float y, float z){this->m_origin={x,y,z};}

private:
	GLuint m_volume_texture;
	GLuint m_noise_texture;
	GLuint lutTexture;
	Mesh m_cube_vao;
	QVector3D m_origin;
	QVector3D m_spacing;
	QVector3D m_size;

	float scale_factor();
};
