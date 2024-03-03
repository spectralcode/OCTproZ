/**
**  This file is part of OCTproZ.
**  OCTproZ is an open source software for processig of optical
**  coherence tomography (OCT) raw data.
**  Copyright (C) 2019-2024 Miroslav Zabic
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

#include "stringspinbox.h"
#include <QDoubleSpinBox>


StringSpinBox::StringSpinBox(QWidget *parent) : QAbstractSpinBox(parent) {
	this->lineEdit()->setReadOnly(true);
	this->currentIndex = -1;
}


StringSpinBox::~StringSpinBox() {

}

void StringSpinBox::setStrings(QStringList strings) {
	this->strings = strings;
	this->currentIndex  = 0;
	this->lineEdit()->setText(this->strings.at(this->currentIndex));
	this->adjustWidth();
}

void StringSpinBox::stepBy(int steps) {
	this->currentIndex += steps;
	this->currentIndex = qBound(0, this->currentIndex, this->strings.size() - 1);
	lineEdit()->setText(this->strings.at(this->currentIndex));
	emit indexChanged();
}

QSize StringSpinBox::sizeHint() const {
	//get default height of a QDoubleSpinBox
	QDoubleSpinBox tmpDoubleSpinBox;
	int defaultHeight = tmpDoubleSpinBox.sizeHint().height();

	//get width based on the content of the StringSpinBox
	int preferredWidth = this->getPreferredWidth();

	return QSize(preferredWidth, defaultHeight);
}

int StringSpinBox::getIndexOf(QString text) {
	return this->strings.indexOf(text);
}

void StringSpinBox::setIndex(int index) {
	int elements = this->strings.size();
	if(elements > 0 && index < elements){
		this->currentIndex = index;
		lineEdit()->setText(this->strings.at(this->currentIndex));
		emit indexChanged();
	}
}

int StringSpinBox::getPreferredWidth() const{
	const int spinButtonsWidthEstimate = 25; //todo: get actual width of buttons
	QString longestString = "";
	foreach(QString string, this->strings) {
		if (longestString.size() < string.size()) {
			longestString = string;
		}
	}
	QFontMetrics fontMetric = this->lineEdit()->fontMetrics();
	int w = fontMetric.boundingRect(longestString).width()+spinButtonsWidthEstimate;
	return w;
}

void StringSpinBox::adjustWidth() {
	int w = this->getPreferredWidth();
	this->setMaximumWidth(w);
}

QAbstractSpinBox::StepEnabled StringSpinBox::stepEnabled() const {
	StepEnabled enabled = StepUpEnabled | StepDownEnabled;
	int maxIndex = this->strings.size() - 1;
	if(qBound(0, this->currentIndex, maxIndex) == 0) {
		enabled ^= StepDownEnabled;
	}
	if (qBound(0, this->currentIndex, maxIndex) == maxIndex) {
		enabled ^= StepUpEnabled;
	}
	return enabled;
}
