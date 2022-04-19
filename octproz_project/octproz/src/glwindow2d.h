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

#ifndef GLWINDOW2D_H
#define GLWINDOW2D_H

#define EXTENDED_PANEL "extended_panel"
#define DISPLAYED_FRAMES "displayed_frames"
#define CURRENT_FRAME "current_frame"
#define ROTATION_ANGLE "rotation_angle"
#define DISPLAY_MODE "display_mode"
#define STRETCH_X "strecth_x"
#define STRETCH_Y "strecth_y"
#define HORIZONTAL_SCALE_BAR_ENABLED "horizontal_scale_bar_enabeld"
#define VERTICAL_SCALE_BAR_ENABLED "vertical_scale_bar_enabeld"
#define HORIZONTAL_SCALE_BAR_TEXT "horizontal_scale_bar_text"
#define VERTICAL_SCALE_BAR_TEXT "vertical_scale_bar_text"
#define HORIZONTAL_SCALE_BAR_LENGTH "horizontal_scale_bar_length"
#define VERTICAL_SCALE_BAR_LENGTH "vertical_scale_bar_length"

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QMouseEvent>
#include <QCoreApplication>
#include <QOffscreenSurface>

// CUDA Runtime, Interop  includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_profiler_api.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// CUDA helper functions
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>

// Qt gui includes
#include <QSpinBox>
#include <QLabel>
#include <QLayout>
#include <QSlider>
#include <QMenu>
#include <QComboBox>
#include <QFileDialog>
#include <QLineEdit>
#include <QToolButton>
#include <QPainter>
#include <QCheckBox>
#include <QLineEdit>

#include "stringspinbox.h"
#include "outputwindow.h"

//#include "octalgorithmparameters.h" //needed for the definition of DISPLAY_FUNCTION enum


#define DEFAULT_WIDTH  2048
#define DEFAULT_HEIGHT 512


struct LineCoordinates{
	float x1;
	float y1;
	float x2;
	float y2;
};

struct GLWindow2DParams {
	bool extendedViewEnabled;
	int displayedFrames;
	int currentFrame;
	double rotationAngle;
	int displayFunction;
	double stretchX;
	double stretchY;
	bool horizontalScaleBarEnabled;
	bool verticalScaleBarEnabled;
	int horizontalScaleBarLength;
	int verticalScaleBarLength;
	QString horizontalScaleBarText;
	QString verticalScaleBarText;
};

enum FRAME_EDGE {
	TOP,
	BOTTOM,
	LEFT,
	RIGHT
};


class ScaleBar;
class ControlPanel2D;
class GLWindow2D : public QOpenGLWidget, protected QOpenGLFunctions, public OutputWindow
{
	Q_OBJECT

public:
	GLWindow2D(QWidget* parent);
	~GLWindow2D();

	ControlPanel2D* getControlPanel()const {return this->panel;}
	FRAME_EDGE getMarkerOrigin() const {return this->markerOrigin;}
	void setMarkerOrigin(FRAME_EDGE origin);
	QAction* getMarkerAction(){return this->markerAction;}

	void setSettings(QVariantMap settings) override;
	QVariantMap getSettings() override;


private:
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
	float stretchX;
	float stretchY;
	float rotationAngle;
	//OctAlgorithmParameters::DISPLAY_FUNCTION displayFuntion;
	unsigned int markerPosition;
	bool markerEnabled;

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

	ScaleBar* horizontalScaleBar;
	ScaleBar* verticalScaleBar;

	void initContextMenu();
	void displayScalebars();
	void displayMarker();
	void displayOrientationLine(int x, int y, int length);


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


public slots:
	void slot_saveScreenshot(QString savePath, QString fileName);
	void slot_changeBufferAndTextureSize(unsigned int width, unsigned int height, unsigned int depth);
	void slot_initProcessingThreadOpenGL(QOpenGLContext* processingContext, QOffscreenSurface* processingSurface, QThread* processingThread);
	void slot_registerGLbufferWithCuda();
	void setKeepAspectRatio(bool enable);
	void setRotationAngle(float angle);
	void setStretchX(float stretchFactor);
	void setStretchY(float stretchFactor);
	void slot_screenshot();
	void enableMarker(bool enable);
	void setMarkerPosition(unsigned int position);
	void saveSettings();


signals:
	void registerBufferCudaGL(unsigned int bufferId);
	void initCudaGl();
	void currentFrameNr(unsigned int frameNr);

	void dialogAboutToOpen();
	void dialogClosed();
	void error(QString);
	void info(QString);

};






class ScaleBar : public QObject, public QPainter
{
	Q_OBJECT
public:
	enum ScaleBarOrientation {
		Horizontal,
		Vertical
	};

	ScaleBar();
	~ScaleBar();

	void enable(bool enable){this->enabled = enable;}
	bool isEnabled(){return this->enabled;}
	void setOrientation(ScaleBarOrientation orientation){this->orientation = orientation;}
	void setPos(QPoint pos){this->pos = pos;}
	void setLength(int lengthInPx){this->lenghInPx = lengthInPx;}
	void setText(QString text);
	void draw(QPaintDevice* paintDevice, float scaleFactor = 1.0);


private:
	void updateTextSizeInfo();

	bool enabled;
	ScaleBarOrientation orientation;
	QPoint pos;
	int lenghInPx;
	QString text;
	int textDistanceToScaleBarInPx;
	int textWidth;
	int textHeight;
	bool textChanged;
};





class ControlPanel2D : public QWidget
{
	Q_OBJECT

public:
	ControlPanel2D(QWidget* parent);
	~ControlPanel2D();

	void setMaxFrame(unsigned int maxFrame);
	void setMaxAverage(unsigned int maxAverage);
	GLWindow2DParams getParams();


private:
	void findGuiElements();
	void enableExtendedView(bool enable);
	void updateParams();
	void connectGuiToSettingsChangedSignal();
	void disconnectGuiFromSettingsChangedSignal();

	GLWindow2DParams params;
	bool extendedView;
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
	QToolButton* toolButtonMore;
	QDoubleSpinBox* doubleSpinBoxStretchX;
	QLabel* labelStretchX;
	QDoubleSpinBox* doubleSpinBoxStretchY;
	QLabel* labelStretchY;
	QCheckBox* checkBoxHorizontalScaleBar;
	QLabel* labelHorizontalScaleBar;
	QSpinBox* spinBoxHorizontalScaleBar;
	QLabel* labelHorizontalScaleBarText;
	QLineEdit* lineEditHorizontalScaleBarText;
	QCheckBox* checkBoxVerticalScaleBar;
	QLabel* labelVerticalScaleBar;
	QLabel* labelVerticalScaleBarText;
	QSpinBox* spinBoxVerticalScaleBar;
	QLineEdit* lineEditVerticalScaleBarText;

	QHBoxLayout* widgetLayout;
	QGridLayout* layout;

	QList<QLineEdit*> lineEdits;
	QList<QCheckBox*> checkBoxes;
	QList<QDoubleSpinBox*> doubleSpinBoxes;
	QList<QSpinBox*> spinBoxes;
	QList<StringSpinBox*> stringSpinBoxes;
	QList<QComboBox*> comboBoxes;


protected:


public slots:
	void setParams(GLWindow2DParams params);
	void updateDisplayFrameSettings();
	void toggleExtendedView();


signals:
	void displayFrameSettingsChanged(unsigned int frameNr, unsigned int displayFunctionFrames, int displayFunction);
	void settingsChanged();


friend class GLWindow2D;
};

#endif // GLWINDOW2D_H
