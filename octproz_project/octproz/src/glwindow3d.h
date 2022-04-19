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
**			iqo.uni-hannover.de
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

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QMouseEvent>
#include <QCoreApplication>
#include <QOffscreenSurface>

#include <QOpenGLExtraFunctions>
#include <QOpenGLShaderProgram>

#include <functional>
#include <vector>

#include <QtMath>

#include "outputwindow.h"
#include "mesh.h"
#include "raycastvolume.h"
#include "trackball.h"
#include "controlpanel.h"
#include "settings.h"


/*!
 * \brief Class for a raycasting canvas widget.
 */
class GLWindow3D : public QOpenGLWidget, protected QOpenGLExtraFunctions, public OutputWindow
{
	Q_OBJECT
public:
	explicit GLWindow3D(QWidget *parent = nullptr);
	~GLWindow3D();

	void setSettings(QVariantMap settings) override;
	QVariantMap getSettings() override;

	void setStepLength(const GLfloat step_length) {
		m_stepLength = step_length;
		update();
	}

	void setThreshold(const GLfloat threshold) {
		m_threshold = threshold;
		update();
	}

	void setDepthWeight(const GLfloat depth_weight) {
		m_depth_weight = depth_weight;
		update();
	}

	void setMode(const QString& mode) {
		m_active_mode = mode;
		update();
	}

	void setBackground(const QColor& colour) {
		m_background = colour;
		update();
	}

	void setStretch(const qreal x, const qreal y, const qreal z);
	void setGammaCorrection(float gamma){m_gamma = gamma;}
	void generateTestVolume();

	std::vector<QString> getModes(void) {
		std::vector<QString> modes;

		foreach(auto element, m_modes){
		   QString key = element.first;
		   modes.push_back(key);
		}

		return modes;
	}

	QColor getBackground(void) {
		return m_background;
	}

	std::pair<double, double> getRange(void) {
		return raycastingVolume->range();
	}

signals:
	void registerBufferCudaGL(unsigned int bufferId);
	void initCudaGl();

	void dialogAboutToOpen();
	void dialogClosed();
	void error(QString);
	void info(QString);

public slots:
	virtual void mouseDoubleClickEvent(QMouseEvent* event) override;
	virtual void mouseMoveEvent(QMouseEvent* event) override;
	virtual void mousePressEvent(QMouseEvent* event) override;
	virtual void mouseReleaseEvent(QMouseEvent* event) override;
	virtual void wheelEvent(QWheelEvent* event) override;

	void slot_saveScreenshot(QString savePath, QString fileName);
	void slot_screenshot();
	void slot_changeBufferAndTextureSize(unsigned int width, unsigned int height, unsigned int depth);
	void slot_initProcessingThreadOpenGL(QOpenGLContext* processingContext, QOffscreenSurface* processingSurface, QThread* processingThread);
	void slot_registerGLbufferWithCuda();
	void slot_updateDisplayParams(GLWindow3DParams params);
	void saveSettings();

protected:

	/*!
	 * \brief Initialise OpenGL-related state.
	 */
	void initializeGL();

	/*!
	 * \brief Paint a frame on the canvas.
	 */
	void paintGL();

	/*!
	 * \brief Callback to handle canvas resizing.
	 * \param w New width.
	 * \param h New height.
	 */
	void resizeGL(int width, int height);
	void contextMenuEvent(QContextMenuEvent* event) override;
	void enterEvent(QEvent* event) override;
	void leaveEvent(QEvent* event) override;

private:
	bool initialized;
	bool changeTextureSizeFlag;
	bool updateContinuously;
	GLWindow3DParams displayParams;

	unsigned int volumeWidth;
	unsigned int volumeHeight;
	unsigned int volumeDepth;

	QMenu* contextMenu;
	QAction* screenshotAction;

	ControlPanel3D* panel;
	QVBoxLayout* layout;



	QMatrix4x4 m_viewMatrix;
	QMatrix4x4 m_modelViewProjectionMatrix;
	QMatrix3x3 m_normalMatrix;

	const GLfloat m_fov = 50.0f;										  /*!< Vertical field of view. */
	const GLfloat m_focalLength = 1.0 / qTan(M_PI / 180.0 * m_fov / 2.0); /*!< Focal length. */
	GLfloat m_aspectRatio;												/*!< width / height */

	QVector2D m_viewportSize;
	QVector3D m_rayOrigin; /*!< Camera position in model space coordinates. */

	QVector3D m_lightPosition {3.0, 0.0, 3.0};	/*!< In camera coordinates. */
	QVector3D m_diffuseMaterial {1.0, 1.0, 1.0};  /*!< Material colour. */
	GLfloat m_stepLength;						 /*!< Step length for ray march. */
	GLfloat m_threshold;						  /*!< Isosurface intensity threshold. */
	QColor m_background;						  /*!< Viewport background colour. */

	GLfloat m_gamma = 2.2f; /*!< Gamma correction parameter. */
	GLfloat m_depth_weight;

	RayCastVolume *raycastingVolume;

	std::map<QString, QOpenGLShaderProgram*> m_shaders;
	std::map<QString, std::function<void(void)>> m_modes;
	QString m_active_mode;

	TrackBall m_trackBall {};	   /*!< Trackball holding the model rotation. */
	TrackBall m_scene_trackBall {}; /*!< Trackball holding the scene rotation. */

	GLint m_distExp = -500;

	/*!
	 * \brief Width scaled by the pixel ratio (for HiDPI devices).
	 */
	GLuint scaled_width();

	/*!
	 * \brief Height scaled by the pixel ratio (for HiDPI devices).
	 */
	GLuint scaled_height();

	/*!
	 * \brief Perform raycasting.
	 */
	void raycasting(const QString& shader);

	/*!
	 * \brief Convert a mouse position into normalised canvas coordinates.
	 * \param p Mouse position.
	 * \return Normalised coordinates for the mouse position.
	 */
	QPointF pixelPosToViewPos(const QPointF& p);

	/*!
	 * \brief Add a shader.
	 * \param name Name for the shader.
	 * \param vertex Vertex shader source file.
	 * \param fragment Fragment shader source file.
	 */
	void addShader(const QString& name, const QString& vector, const QString& fragment);

	void initContextMenu();
};





