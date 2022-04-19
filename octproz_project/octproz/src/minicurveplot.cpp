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

#include "minicurveplot.h"
#include <QPainterPathStroker>

MiniCurvePlot::MiniCurvePlot(QWidget *parent) : QCustomPlot(parent){
	//default colors
	this->referenceCurveAlpha = 100;
	this->setBackground( QColor(50, 50, 50));
	this->axisRect()->setBackground(QColor(25, 25, 25));
	this->curveColor.setRgb(55, 100, 250);
	this->referenceCurveColor.setRgb(250, 250, 250, referenceCurveAlpha);

	//configure curve graph
	this->addGraph();
	this->setCurveColor(curveColor);

	//configure reference curve graph
	this->addGraph();
	this->setReferenceCurveColor(referenceCurveColor);

	//configure axis
	this->yAxis->setVisible(false);
	this->xAxis->setVisible(false);
	this->setAxisColor(Qt::white);

	//user interactions
	this->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);

	//maximize size of plot area
	this->axisRect()->setAutoMargins(QCP::msNone);
	this->axisRect()->setMargins(QMargins(0,0,0,0));

	//todo: proper implementation of >>smooth<< round corners
	//round corners
	this->roundCorners(false);
}

MiniCurvePlot::~MiniCurvePlot() {
}

void MiniCurvePlot::setCurveColor(QColor color) {
	this->curveColor = color;
	QPen curvePen = QPen(color);
	curvePen.setWidth(1);
	this->graph(0)->setPen(curvePen);
}

void MiniCurvePlot::setReferenceCurveColor(QColor color) {
	this->referenceCurveColor = color;
	QPen referenceCurvePen = QPen(color);
	referenceCurvePen.setWidth(1);
	this->graph(1)->setPen(referenceCurvePen);
}

void MiniCurvePlot::plotCurves(float *curve, float *referenceCurve, unsigned int samples) {
	if(samples == 0){return;}
	int size = static_cast<int>(samples);

	 //set values for x-axis
	 if(this->sampleNumbers.size() != size){
		 this->sampleNumbers.resize(size);
		 for(int i = 0; i<size; i++){
			 this->sampleNumbers[i] = i;
		 }
	 }

	 //fill curve data
	 if(curve != nullptr){
		 if(this->curve.size() != size){
			 this->curve.resize(size);
		 }
		 for(int i = 0; i<size; i++){
			 this->curve[i] = static_cast<double>(curve[i]);
		 }
		 this->graph(0)->setData(this->sampleNumbers, this->curve, true);
	 }

	 //fill reference curve data
	 if(referenceCurve != nullptr){
		 if(this->referenceCurve.size() != size){
			 this->referenceCurve.resize(size);
		 }
		 for(int i = 0; i<size; i++){
			 this->referenceCurve[i] = static_cast<double>(referenceCurve[i]);
		 }
		 this->graph(1)->setData(this->sampleNumbers, this->referenceCurve, true);
	 }

	 //update plot
	 this->rescaleAxes();
	 this->zoomOutSlightly();
	 this->replot();
}


void MiniCurvePlot::setAxisColor(QColor color) {
	this->xAxis->setBasePen(QPen(color, 1));
	this->yAxis->setBasePen(QPen(color, 1));
	this->xAxis->setTickPen(QPen(color, 1));
	this->yAxis->setTickPen(QPen(color, 1));
	this->xAxis->setSubTickPen(QPen(color, 1));
	this->yAxis->setSubTickPen(QPen(color, 1));
	this->xAxis->setTickLabelColor(color);
	this->yAxis->setTickLabelColor(color);
	this->xAxis->setLabelColor(color);
	this->yAxis->setLabelColor(color);
}

void MiniCurvePlot::zoomOutSlightly() {
	this->yAxis->scaleRange(1.1, this->yAxis->range().center());
	this->xAxis->scaleRange(1.1, this->xAxis->range().center());
}


void MiniCurvePlot::contextMenuEvent(QContextMenuEvent *event){
	QMenu menu(this);
	QAction savePlotAction(tr("Save Plot as..."), this);
	connect(&savePlotAction, &QAction::triggered, this, &MiniCurvePlot::slot_saveToDisk);
	menu.addAction(&savePlotAction);
	menu.exec(event->globalPos());
}

void MiniCurvePlot::mouseMoveEvent(QMouseEvent *event){
	if(!(event->buttons() & Qt::LeftButton)){
		double x = this->xAxis->pixelToCoord(event->pos().x());
		double y = this->yAxis->pixelToCoord(event->pos().y());
		this->setToolTip(QString("%1 , %2").arg(x).arg(y));
	}else{
		QCustomPlot::mouseMoveEvent(event);
	}
}

void MiniCurvePlot::resizeEvent(QResizeEvent *event) {
	if(this->drawRoundCorners){
		QRect plotRect = this->rect();
		const int radius = 6;
		QPainterPath path;
		path.addRoundedRect(plotRect, radius, radius);
		QRegion mask = QRegion(path.toFillPolygon().toPolygon());
		this->setMask(mask);
	}
	QCustomPlot::resizeEvent(event);
}

void MiniCurvePlot::changeEvent(QEvent *event) {
	if(event->ActivationChange){
		if(!this->isEnabled()){
			this->curveColor.setAlpha(55);
			this->referenceCurveColor.setAlpha(25);
			this->setCurveColor(this->curveColor);
			this->setReferenceCurveColor(this->referenceCurveColor);
			this->replot();
		} else {
			this->curveColor.setAlpha(255);
			this->referenceCurveColor.setAlpha(this->referenceCurveAlpha);
			this->setCurveColor(this->curveColor);
			this->setReferenceCurveColor(this->referenceCurveColor);
			this->replot();
		}
	}
	QCustomPlot::changeEvent(event);

}

void MiniCurvePlot::mouseDoubleClickEvent(QMouseEvent *event) {
	this->rescaleAxes();
	this->zoomOutSlightly();
	this->replot();
}

void MiniCurvePlot::slot_saveToDisk() {
	emit dialogAboutToOpen();
	QString filters("Image (*.png);;Vector graphic (*.pdf);;CSV (*.csv)");
	QString defaultFilter("CSV (*.csv)");
	QString fileName = QFileDialog::getSaveFileName(this, tr("Save Plot"), QDir::currentPath(), filters, &defaultFilter);
	emit dialogClosed();
	if(fileName == ""){
		emit error(tr("Save plot to disk canceled."));
		return;
	}
	bool saved = false;
	if(defaultFilter == "Image (*.png)"){
		saved = this->savePng(fileName);
	}else if(defaultFilter == "Vector graphic (*.pdf)"){
		saved = this->savePdf(fileName);
	}else if(defaultFilter == "CSV (*.csv)"){
		saved = this->saveCurveDataToFile(fileName);
	}
	if(saved){
		emit info(tr("Plot saved to ") + fileName);
	}else{
		emit error(tr("Could not save plot to disk."));
	}
}


bool MiniCurvePlot::saveCurveDataToFile(QString fileName) {
	bool saved = false;
	QFile file(fileName);
	if (file.open(QFile::WriteOnly|QFile::Truncate)) {
		QTextStream stream(&file);
		stream << "Sample Number" << ";" << "Sample Value" << "\n";
		for(int i = 0; i < this->sampleNumbers.size(); i++){
			stream << QString::number(this->sampleNumbers.at(i)) << ";" << QString::number(this->curve.at(i)) << "\n";
		}
	file.close();
	saved = true;
	}
	return saved;
}
