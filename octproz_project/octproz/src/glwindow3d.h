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

#define DELAY_TIME_IN_ms 80


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

	void setStepLength(const GLfloat stepLength);
	void setThreshold(const GLfloat threshold);
	void setDepthWeight(const GLfloat depthWeight);
	void setSmoothFactor(const GLint smoothFactor);
	void setAlphaExponent(const GLfloat alphaExponent);
	void enableShading(const GLboolean shadingEnabled);
	void setMode(const QString& mode);
	void setBackground(const QColor& colour);
	void setStretch(const qreal x, const qreal y, const qreal z);
	void setGammaCorrection(float gamma);
	void generateTestVolume();

	QVector<QString> getModes(void);

	QColor getBackground(void){return this->background;}

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

	void saveScreenshot(QString savePath, QString fileName);
	void openScreenshotDialog();
	void changeTextureSize(unsigned int width, unsigned int height, unsigned int depth);
	void createOpenGLContextForProcessing(QOpenGLContext* processingContext, QOffscreenSurface* processingSurface, QThread* processingThread);
	void registerOpenGLBufferWithCuda();
	void updateDisplayParams(GLWindow3DParams params);
	void openLUTFromImage(QImage lut);
	void saveSettings();

protected:

	/*!
	 * \brief Initialise OpenGL-related state.
	 */
	void initializeGL() override;

	/*!
	 * \brief Paint a frame on the canvas.
	 */
	void paintGL() override;

	/*!
	 * \brief Callback to handle canvas resizing.
	 * \param w New width.
	 * \param h New height.
	 */
	void resizeGL(int width, int height) override;
	void contextMenuEvent(QContextMenuEvent* event) override;
	void enterEvent(QEvent* event) override;
	void leaveEvent(QEvent* event) override;

private:
	bool delayedUpdatingRunning;
	bool initialized;
	bool changeTextureSizeFlag;
	bool updateContinuously;
	GLWindow3DParams displayParams;

	unsigned int volumeWidth;
	unsigned int volumeHeight;
	unsigned int volumeDepth;
	QPoint mousePos;
	QPointF viewPos;
	qreal fps;
	QElapsedTimer timer;
	int counter;

	QMenu* contextMenu;
	QAction* screenshotAction;

	ControlPanel3D* panel;
	QVBoxLayout* layout;

	QMatrix4x4 viewMatrix;
	QMatrix4x4 modelViewProjectionMatrix;
	QMatrix3x3 normalMatrix;

	const GLfloat fov = 50.0f;										  /*!< Vertical field of view. */
	const GLfloat focalLength = 1.0 / qTan(M_PI / 180.0 * this->fov / 2.0); /*!< Focal length. */
	GLfloat aspectRatio;												/*!< width / height */

	QVector2D viewportSize;
	QVector3D rayOrigin; /*!< Camera position in model space coordinates. */

	QVector3D lightPosition {1.0, 3.0, 3.0};	/*!< In camera coordinates. */
	QVector3D diffuseMaterial {1.0, 1.0, 1.0};  /*!< Material colour. */
	GLfloat stepLength;						 /*!< Step length for ray march. */
	GLfloat threshold;						  /*!< Isosurface intensity threshold. */
	QColor background;						  /*!< Viewport background colour. */

	GLfloat gamma = 2.2f; /*!< Gamma correction parameter. */
	GLfloat depthWeight;
	GLfloat alphaExponent;
	GLint smoothFactor;
	GLboolean shadingEnabled;
	GLboolean lutEnabled;

	RayCastVolume *raycastingVolume;

	std::map<QString, QOpenGLShaderProgram*> shaders;
	std::map<QString, std::function<void(void)>> modes;
	QString activeMode;

	TrackBall trackBall {};	   /*!< Trackball holding the model rotation. */
	TrackBall sceneTrackBall {}; /*!< Trackball holding the scene rotation. */

	GLint distExp = -500;

	/*!
	 * \brief Width scaled by the pixel ratio (for HiDPI devices).
	 */
	GLuint scaledWidth();

	/*!
	 * \brief Height scaled by the pixel ratio (for HiDPI devices).
	 */
	GLuint scaledHeight();

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
	void restoreLUTSettingsFromPreviousSession();

	void initContextMenu();
	void delayedUpdate();
	void countFPS();
};





