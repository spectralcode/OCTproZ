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

#ifndef CONTROLPANEL_H
#define CONTROLPANEL_H

#define EXTENDED_PANEL "extended_panel"
#define DISPLAY_MODE "display_mode"
#define DISPLAY_MODE_INDEX "display_mode_index"
#define ISO_SURFACE_THRESHOLD "iso_surface_threshold"
#define RAY_STEP_LENGTH "ray_step_length"
#define STRETCH_X "strecth_x"
#define STRETCH_Y "strecth_y"
#define STRETCH_Z "strecth_z"
#define CONTINUOUS_UPDATE_ENABLED "continuous_update_enabled"
#define GAMMA "gamma"

#include <QSpinBox>
#include <QLabel>
#include <QLayout>
#include <QSlider>
#include <QMenu>
#include <QComboBox>
#include <QFileDialog>
#include <QLineEdit>
#include <QCheckBox>
#include <QToolButton>
#include <QObject>

#include "stringspinbox.h"


//todo: common base class for ControlPanel2D and ControlPanel3D



struct GLWindow3DParams {
	bool extendedViewEnabled;
	qreal rayMarchStepLength;
	qreal isosurfaceThreshold;
	bool updateContinuously;
	QString displayMode;
	int displayModeIndex;
	qreal stretchX;
	qreal stretchY;
	qreal stretchZ;
	qreal gamma;
};

class ControlPanel3D : public QWidget
{
	Q_OBJECT

public:
	ControlPanel3D(QWidget* parent);
	~ControlPanel3D();
	void setModes(QStringList modes);
	void enableContinuousUpdate(bool enable);
	GLWindow3DParams getParams();


private:
	void enableExtendedView(bool enable);
	void findGuiElements();
	void connectGuiToSettingsChangedSignal();
	void disconnectGuiFromSettingsChangedSignal();

	GLWindow3DParams params;
	bool extendedView;
	QWidget* panel;
	QDoubleSpinBox* doubleSpinBoxStepLength;
	QDoubleSpinBox* doubleSpinBoxIsosurfaceThreshold;
	QLabel* labelStepLength;
	QLabel* labelIsosurfaceThreshold;
	QLabel* labelMode;
	StringSpinBox* stringBoxModes;
	QCheckBox* checkBoxUpdateContinuously;
	QHBoxLayout* widgetLayout;
	QGridLayout* layout;

	QDoubleSpinBox* doubleSpinBoxStretchX;
	QLabel* labelStretchX;
	QDoubleSpinBox* doubleSpinBoxStretchY;
	QLabel* labelStretchY;
	QDoubleSpinBox* doubleSpinBoxStretchZ;
	QLabel* labelStretchZ;
	QToolButton* toolButtonMore;

	QDoubleSpinBox* doubleSpinBoxGamma;
	QLabel* labelGamma;

	QList<QLineEdit*> lineEdits;
	QList<QCheckBox*> checkBoxes;
	QList<QDoubleSpinBox*> doubleSpinBoxes;
	QList<QSpinBox*> spinBoxes;
	QList<StringSpinBox*> stringSpinBoxes;
	QList<QComboBox*> comboBoxes;

protected:


public slots:
	void updateDisplayParameters();
	void toggleExtendedView();
	void setParams(GLWindow3DParams params);


signals:
	void displayParametersChanged(GLWindow3DParams params);
	void settingsChanged();

};





#endif// CONTROLPANEL_H
