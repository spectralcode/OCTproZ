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

#ifndef CONTROLPANEL_H
#define CONTROLPANEL_H

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

#include "stringspinbox.h"


//todo: common base class for ControlPanel2D and ControlPanel3D


struct DisplayParams {
	qreal rayMarchStepLength;
	qreal isosurfaceThreashold;
	bool updateContinuously;
	QString mode;
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


private:
	void enableExtendedView(bool enable);

	bool extendedView;
	DisplayParams currParams;
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

protected:



public slots:
	void updateDisplayParameters();
	void toggleExtendedView();




signals:
   void displayParametersChanged(DisplayParams params);

};





#endif  // CONTROLPANEL_H
