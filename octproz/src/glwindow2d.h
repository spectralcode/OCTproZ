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

#ifndef GLWINDOW2D_H
#define GLWINDOW2D_H

#ifdef _WIN32
	#define WINDOWS_LEAN_AND_MEAN
	//#define NOMINMAX
	#include <windows.h>
	#include "GL/glew.h"
#endif

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QMouseEvent>
#include <QCoreApplication>
#include <QOffscreenSurface>
//#include <kernels.h>

// CUDA Runtime, Interop, and includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_profiler_api.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// CUDA utilities
#include <helper_cuda.h>
//#include <helper_cuda_gl.h>


// Helper functions
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>


#include <QSpinBox>
#include <QLabel>
#include <QLayout>
#include <QSlider>
#include <QMenu>
#include <QComboBox>
#include <QFileDialog>
#include <QLineEdit>

#include "stringspinbox.h"

//#include "octalgorithmparameters.h" //needed for the definition of DISPLAY_FUNCTION enum


#define DEFAULT_WIDTH  2048
#define DEFAULT_HEIGHT 512

struct DisplayFrameParams {
	unsigned int frameNr;
	unsigned int displayFunctionFrames;
	int displayFunction;
};

struct LineCoordinates{
	float x1;
	float y1;
	float x2;
	float y2;
};

enum FRAME_EDGE {
	TOP,
	BOTTOM,
	LEFT,
	RIGHT
};



class ControlPanel2D;
class GLWindow2D : public QOpenGLWidget, protected QOpenGLFunctions
{
	Q_OBJECT

public:
	GLWindow2D(QWidget* parent);
	~GLWindow2D();

	ControlPanel2D* getControlPanel()const {return this->panel;}
	FRAME_EDGE getMarkerOrigin() const {return this->markerOrigin;}
	void setMarkerOrigin(FRAME_EDGE markerOrigin);
	QAction* getMarkerAction(){return this->markerAction;}

private:
	//void* cudaBufHandle;
	unsigned int width;
	unsigned int height;
	unsigned int depth;
	float xTranslation;
	float yTranslation;
	float scaleFactor;
	float screenWidthScaled;
	float screenHeightScaled;
	bool initialized;
	bool changeBufferSizeFlag;
	bool keepAspectRatio;
	double rotationAngle;
	//OctAlgorithmParameters::DISPLAY_FUNCTION displayFuntion;
	unsigned int markerPosition;
	bool markerVisible;
    //rotation
    float screenWidthScaled01;
    float screenWidthScaled11;
    float screenWidthScaled10;
    float screenWidthScaled00;
    float screenHeightScaled01;
    float screenHeightScaled11;
    float screenHeightScaled10;
    float screenHeightScaled00;

	QMenu* contextMenu;
	QAction* keepAspectRatioAction;
	QAction* markerAction;
	QAction* screenshotAction;

	FRAME_EDGE markerOrigin;
	LineCoordinates markerCoordinates;

	GLuint buf;
	GLuint texture;


	QPoint mousePos;

	size_t frameNr;

	ControlPanel2D* panel;
	QVBoxLayout* layout;

	void initContextMenu();




protected:
	void initializeGL() override;
	void paintGL() override;
	void resizeGL(int width, int height) override;
	void mousePressEvent(QMouseEvent* event) override;
	void mouseMoveEvent(QMouseEvent* event) override;
	void wheelEvent(QWheelEvent* event) override;
	void mouseDoubleClickEvent(QMouseEvent *event) override;
	void enterEvent(QEvent* event) override;
	void leaveEvent(QEvent* event) override;
	void contextMenuEvent(QContextMenuEvent* event) override;
	//void paintEvent(QPaintEvent* event) override;


public slots:
	void slot_saveScreenshot(QString savePath, QString fileName);
	void slot_changeBufferAndTextureSize(unsigned int width, unsigned int height, unsigned int depth);
	//void slot_initProcessingThreadOpenGL(QOpenGLContext** processingContext, QOffscreenSurface** processingSurface, QThread* processingThread);
	void slot_initProcessingThreadOpenGL(QOpenGLContext* processingContext, QOffscreenSurface* processingSurface, QThread* processingThread);
	void slot_registerGLbufferWithCuda();
	void setKeepAspectRatio(bool keepAspectRatio);
	void setRotationAngle(double rotationAngle);
	void slot_screenshot();
	void slot_displayMarker(bool display);
	void slot_setMarkerPosition(unsigned int position);


signals:
	void registerBufferCudaGL(unsigned int bufferId);
	void initCudaGl();
	void currentFrameNr(unsigned int frameNr);

	void dialogAboutToOpen();
	void dialogClosed();
	void error(QString);
	void info(QString);

};













class ControlPanel2D : public QWidget
{
	Q_OBJECT

public:
	ControlPanel2D(QWidget* parent);
	~ControlPanel2D();

	void setMaxFrame(unsigned int maxFrame);
	void setMaxAverage(unsigned int maxAverage);

private:
	QWidget* panel;
	QSpinBox* spinBoxAverage;
	QSpinBox* spinBoxFrame;
	QDoubleSpinBox* doubleSpinBoxRotationAngle;
	QLabel* labelFrame;
	QLabel* labelRotationAngle;
	QLabel* labelDisplayFunctionFrames;
	QLabel* labelDisplayFunction;
	QSlider* slider;
	//QComboBox* comboBox; //due to a Qt bug QComboBox is not displayed correctly on top of a QOpenGLWidget. StringSpinBox is used as work around
	StringSpinBox* stringBoxFunctions;
	QHBoxLayout* widgetLayout;
	QGridLayout* layout;


protected:



public slots:
	void updateDisplayFrameSettings();



signals:
   void displayFrameSettingsChanged(unsigned int frameNr, unsigned int displayFunctionFrames, int displayFunction);

friend class GLWindow2D;
};





#endif  // GLWINDOW2D_H
