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

#include "controlpanel.h"


ControlPanel3D::ControlPanel3D(QWidget *parent) : QWidget(parent){
	this->panel = new QWidget(parent);
	QPalette pal;
	pal.setColor(QPalette::Background, QColor(32,32,32,128));
	this->panel->setAutoFillBackground(true);
	this->panel->setPalette(pal);
	this->widgetLayout = new QHBoxLayout(this);
	this->widgetLayout->addWidget(this->panel);

	this->doubleSpinBoxIsosurfaceThreshold = new QDoubleSpinBox(this->panel);
	this->doubleSpinBoxStepLength = new QDoubleSpinBox(this->panel);
	this->stringBoxModes = new StringSpinBox(this->panel);
	this->checkBoxUpdateContinuously = new QCheckBox(this->panel);
	this->labelIsosurfaceThreshold = new QLabel(tr("Threshold:"), this->panel);
	this->labelStepLength = new QLabel(tr("Ray Step:"), this->panel);
	this->labelMode = new QLabel(tr("Mode:"), this->panel);
	this->checkBoxUpdateContinuously->setText(tr("Update Continuously"));

	this->doubleSpinBoxStretchX = new QDoubleSpinBox(this->panel);
	this->labelStretchX = new QLabel(tr("Stretch X:"), this->panel);
	this->doubleSpinBoxStretchY = new QDoubleSpinBox(this->panel);
	this->labelStretchY = new QLabel(tr("Stretch Y:"), this->panel);
	this->doubleSpinBoxStretchZ = new QDoubleSpinBox(this->panel);
	this->labelStretchZ = new QLabel(tr("Stretch Z:"), this->panel);

	this->doubleSpinBoxGamma = new QDoubleSpinBox(this->panel);
	this->labelGamma = new QLabel(tr("Gamma:"), this->panel);

	this->toolButtonMore = new QToolButton(this->panel);
	this->toolButtonMore->setArrowType(Qt::DownArrow);
	this->toolButtonMore->setAutoRaise(true);


	this->layout = new QGridLayout(this->panel);
	this->layout->setContentsMargins(0,0,0,0);
	this->layout->setVerticalSpacing(1);
	this->layout->addWidget(this->labelMode, 0, 0, 1, 1, Qt::AlignRight);
	this->layout->addWidget(this->stringBoxModes, 0, 1, 1, 1);
	this->layout->addWidget(this->labelStepLength, 1, 0, 1, 1, Qt::AlignRight);
	this->layout->addWidget(this->doubleSpinBoxStepLength, 1, 1, 1, 1);
	this->layout->addWidget(this->labelIsosurfaceThreshold, 0, 2, 1, 1, Qt::AlignRight);
	this->layout->addWidget(this->doubleSpinBoxIsosurfaceThreshold, 0, 3, 1, 1);
	this->layout->setColumnStretch(2, 10);
	this->layout->addWidget(this->toolButtonMore, 1, 2, 1, 1, Qt::AlignCenter);
	this->layout->addWidget(this->checkBoxUpdateContinuously, 1, 3, 1, 1, Qt::AlignLeft);

	this->layout->addWidget(this->labelStretchX, 3, 0, 1, 1, Qt::AlignRight);
	this->layout->addWidget(this->doubleSpinBoxStretchX,3, 1, 1, 1);
	this->layout->addWidget(this->labelStretchY, 4, 0, 1, 1, Qt::AlignRight);
	this->layout->addWidget(this->doubleSpinBoxStretchY,4, 1, 1, 1);
	this->layout->addWidget(this->labelStretchZ, 5, 0, 1, 1, Qt::AlignRight);
	this->layout->addWidget(this->doubleSpinBoxStretchZ, 5, 1, 1, 1);

	this->layout->addWidget(this->labelGamma, 5, 2, 1, 1, Qt::AlignRight);
	this->layout->addWidget(this->doubleSpinBoxGamma,5, 3, 1, 1);

	this->doubleSpinBoxStepLength->setMinimum(0.001);
	this->doubleSpinBoxStepLength->setMaximum(10.0);
	this->doubleSpinBoxStepLength->setSingleStep(0.001);
	this->doubleSpinBoxStepLength->setValue(0.01);
	this->doubleSpinBoxStepLength->setDecimals(3);

	this->doubleSpinBoxIsosurfaceThreshold->setMinimum(0.0);
	this->doubleSpinBoxIsosurfaceThreshold->setMaximum(1);
	this->doubleSpinBoxIsosurfaceThreshold->setSingleStep(0.01);
	this->doubleSpinBoxIsosurfaceThreshold->setValue(0.5);
	this->doubleSpinBoxIsosurfaceThreshold->setDecimals(2);

	this->checkBoxUpdateContinuously->setChecked(false);

	//todo: refactor this class. avoid repeating code
	this->doubleSpinBoxStretchX->setMaximum(10.00);
	this->doubleSpinBoxStretchX->setMinimum(0.1);
	this->doubleSpinBoxStretchX->setSingleStep(0.1);
	this->doubleSpinBoxStretchX->setValue(1.0);

	this->doubleSpinBoxStretchY->setMaximum(10.00);
	this->doubleSpinBoxStretchY->setMinimum(0.1);
	this->doubleSpinBoxStretchY->setSingleStep(0.1);
	this->doubleSpinBoxStretchY->setValue(1.0);

	this->doubleSpinBoxStretchZ->setMaximum(10.00);
	this->doubleSpinBoxStretchZ->setMinimum(0.1);
	this->doubleSpinBoxStretchZ->setSingleStep(0.1);
	this->doubleSpinBoxStretchZ->setValue(1.0);

	this->doubleSpinBoxGamma->setMaximum(10.00);
	this->doubleSpinBoxGamma->setMinimum(0.1);
	this->doubleSpinBoxGamma->setSingleStep(0.1);
	this->doubleSpinBoxGamma->setValue(2.2);

	connect(this->checkBoxUpdateContinuously, &QCheckBox::toggled, this, &ControlPanel3D::updateDisplayParameters);
	connect(this->doubleSpinBoxStepLength, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &ControlPanel3D::updateDisplayParameters);
	connect(this->doubleSpinBoxIsosurfaceThreshold, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &ControlPanel3D::updateDisplayParameters);
	connect(this->stringBoxModes, &StringSpinBox::indexChanged, this, &ControlPanel3D::updateDisplayParameters);
	connect(this->doubleSpinBoxStretchX, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &ControlPanel3D::updateDisplayParameters);
	connect(this->doubleSpinBoxStretchY, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &ControlPanel3D::updateDisplayParameters);
	connect(this->doubleSpinBoxStretchZ, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &ControlPanel3D::updateDisplayParameters);
	connect(this->doubleSpinBoxGamma, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &ControlPanel3D::updateDisplayParameters);
	connect(this->toolButtonMore, &QToolButton::clicked, this, &ControlPanel3D::toggleExtendedView);

	this->enableExtendedView(false);
	this->updateDisplayParameters();
}


ControlPanel3D::~ControlPanel3D()
{

}

void ControlPanel3D::setModes(QStringList modes) {
	this->stringBoxModes->setStrings(modes);
}

void ControlPanel3D::enableContinuousUpdate(bool enable) {
	this->checkBoxUpdateContinuously->setChecked(enable);
}

void ControlPanel3D::updateDisplayParameters() {
	this->currParams.mode = this->stringBoxModes->getText();
	this->currParams.isosurfaceThreashold = this->doubleSpinBoxIsosurfaceThreshold->value();
	this->currParams.rayMarchStepLength = this->doubleSpinBoxStepLength->value();
	this->currParams.updateContinuously = this->checkBoxUpdateContinuously->isChecked();
	this->currParams.stretchX = this->doubleSpinBoxStretchX->value();
	this->currParams.stretchY = this->doubleSpinBoxStretchY->value();
	this->currParams.stretchZ = this->doubleSpinBoxStretchZ->value();
	this->currParams.gamma = this->doubleSpinBoxGamma->value();

	if(this->currParams.mode == "Isosurface"){
		this->labelIsosurfaceThreshold->setVisible(true);
		this->doubleSpinBoxIsosurfaceThreshold->setVisible(true);
	}else{
		this->labelIsosurfaceThreshold->setVisible(false);
		this->doubleSpinBoxIsosurfaceThreshold->setVisible(false);
	}

	emit displayParametersChanged(this->currParams);
}

void ControlPanel3D::toggleExtendedView(){
	this->enableExtendedView(!this->extendedView);
}

void ControlPanel3D::enableExtendedView(bool enable) {
	this->extendedView = enable;
	enable ? this->toolButtonMore->setArrowType(Qt::UpArrow) : this->toolButtonMore->setArrowType(Qt::DownArrow);
	this->doubleSpinBoxStretchX->setVisible(enable);
	this->doubleSpinBoxStretchY->setVisible(enable);
	this->doubleSpinBoxStretchZ->setVisible(enable);
	this->labelStretchX->setVisible(enable);
	this->labelStretchY->setVisible(enable);
	this->labelStretchZ->setVisible(enable);
	this->labelGamma->setVisible(enable);
	this->doubleSpinBoxGamma->setVisible(enable);
}
