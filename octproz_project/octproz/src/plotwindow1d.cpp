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

#include "plotwindow1d.h"


PlotWindow1D::PlotWindow1D(QWidget *parent) : QCustomPlot(parent){
	this->setBackground( QColor(50, 50, 50));
	this->axisRect()->setBackground(QColor(55, 55, 55));

	this->setMinimumHeight(320);
	this->setMinimumWidth(640);
	this->addGraph();
	this->addGraph(this->xAxis2, this->yAxis2);
	this->processedColor = QColor(72, 99, 160);
	this->rawColor = QColor(Qt::white);
	this->plotLayout()->setMargins(QMargins(0,0,0,0));

	QPen rawLinePen = QPen(QColor(250, 250, 250));
	rawLinePen.setWidth(1);
	this->graph(0)->setPen(rawLinePen);
	this->graph(0)->setName(this->rawLineName + "0");
	this->xAxis->setLabel("Sample number\n\n\n\n");
	this->yAxis->setLabel("Raw value");

	this->yAxis2->setVisible(true);
	this->xAxis2->setVisible(true);
	QPen processedLinePen = QPen(processedColor);
	processedLinePen.setWidth(1); //info: line width greater 1 slows down plotting
	this->graph(1)->setPen(processedLinePen);
	this->graph(1)->setName(this->processedLineName + "0");
	this->xAxis2->setLabel("Sample number");
	this->yAxis2->setLabel("Processed value");

	this->setRawAxisColor(rawColor);
	this->setProcessedAxisColor(processedColor);

	this->line = 0;
	this->linesPerBuffer = 0;
	this->currentRawBitdepth = 0;

	this->legend->setVisible(true);
	this->legend->setBrush(QColor(80, 80, 80, 200));
	this->legend->setTextColor(Qt::white);
	this->legend->setBorderPen(QColor(180, 180, 180, 200));

	this->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);
	this->axisRect()->setRangeZoomAxes(0, this->yAxis);
	this->axisRect()->setRangeDragAxes(0, this->yAxis);

	this->isPlottingRaw = false;
	this->isPlottingProcessed = false;
	this->autoscaling = true;
	this->displayRaw = true;
	this->displayProcessed = false;
	this->bitshift = false;
	this->rawGrabbingAllowed = true;
	this->processedGrabbingAllowed = true;
	this->rawLineName = tr("Raw Line: ");
	this->processedLineName = tr("A-scan Nr.: ");

	this->setRawPlotVisible(this->displayRaw);
	this->setProcessedPlotVisible(this->displayProcessed);

	this->panel = new ControlPanel1D(this);
	this->layout = new QVBoxLayout(this);
	this->layout->addStretch();
	this->layout->addWidget(this->panel);
	this->layout->setContentsMargins(3,0,3,0);
	this->panel->checkBoxAutoscale->setChecked(this->autoscaling);
	this->panel->checkBoxProcessed->setChecked(this->displayProcessed);
	this->panel->checkBoxRaw->setChecked(this->displayRaw);

	connect(this->panel->checkBoxProcessed, &QCheckBox::stateChanged, this, &PlotWindow1D::slot_displayProcessed);
	connect(this->panel->checkBoxRaw, &QCheckBox::stateChanged, this, &PlotWindow1D::slot_displayRaw);
	connect(this->panel->checkBoxAutoscale, &QCheckBox::stateChanged, this, &PlotWindow1D::slot_activateAutoscaling);
	connect(this->panel->spinBoxLine, QOverload<int>::of(&QSpinBox::valueChanged), this, &PlotWindow1D::slot_setLine);
}

PlotWindow1D::~PlotWindow1D(){
}

void PlotWindow1D::setRawAxisColor(QColor color) {
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

void PlotWindow1D::setProcessedAxisColor(QColor color) {
	this->xAxis2->setBasePen(QPen(color, 1));
	this->yAxis2->setBasePen(QPen(color, 1));
	this->xAxis2->setTickPen(QPen(color, 1));
	this->yAxis2->setTickPen(QPen(color, 1));
	this->xAxis2->setSubTickPen(QPen(color, 1));
	this->yAxis2->setSubTickPen(QPen(color, 1));
	this->xAxis2->setTickLabelColor(color);
	this->yAxis2->setTickLabelColor(color);
	this->xAxis2->setLabelColor(color);
	this->yAxis2->setLabelColor(color);
}

void PlotWindow1D::setRawPlotVisible(bool visible) {
	int transparency = visible ? 255 : 0;
	this->rawColor.setAlpha(transparency);
	this->setRawAxisColor(this->rawColor);
	this->graph(0)->setVisible(visible);
	visible ? this->graph(0)->addToLegend() : this->graph(0)->removeFromLegend();
}

void PlotWindow1D::setProcessedPlotVisible(bool visible) {
	int transparency = visible ? 255 : 0;
	this->processedColor.setAlpha(transparency);
	this->setProcessedAxisColor(this->processedColor);
	this->graph(1)->setVisible(visible);
	visible ? this->graph(1)->addToLegend() : this->graph(1)->removeFromLegend();
}

void PlotWindow1D::contextMenuEvent(QContextMenuEvent *event) {
	QMenu menu(this);
	QAction bitshiftRawValuesAction(tr("Bit shift raw values by 4"), this);
	bitshiftRawValuesAction.setCheckable(true);
	bitshiftRawValuesAction.setChecked(this->bitshift);
	connect(&bitshiftRawValuesAction, &QAction::toggled, this, &PlotWindow1D::slot_enableBitshift);
	if(this->currentRawBitdepth > 8 && this->currentRawBitdepth <= 16){
		menu.addAction(&bitshiftRawValuesAction);
	}
	QAction savePlotAction(tr("Save Plot as..."), this);
	connect(&savePlotAction, &QAction::triggered, this, &PlotWindow1D::slot_saveToDisk);
	menu.addAction(&savePlotAction);
	menu.exec(event->globalPos());
}

void PlotWindow1D::mouseDoubleClickEvent(QMouseEvent *event) {
	this->rescaleAxes();
	this->replot();
}
void PlotWindow1D::slot_plotRawData(void* buffer, unsigned bitDepth, unsigned int samplesPerLine, unsigned int linesPerFrame, unsigned int framesPerBuffer, unsigned int buffersPerVolume, unsigned int currentBufferNr){
	if(!this->isPlottingRaw && this->displayRaw && this->rawGrabbingAllowed){
		this->isPlottingRaw = true;
		this->currentRawBitdepth = bitDepth;
		if(buffer != nullptr && this->isVisible()){
			//get length of one raw line (unprocessed a-scan) and resize plot vectors if necessary
			if(this->sampleValues.size() != samplesPerLine){
				this->sampleValues.resize(samplesPerLine);
				this->sampleNumbers.resize(samplesPerLine);
				for(int i = 0; i<samplesPerLine; i++){
					this->sampleNumbers[i] = i;
				}
			}
			//change line number if buffer size got smaller and there are now fewer lines in buffer than the current line number
			if(this->line > this->linesPerBuffer-1){
				this->line = this->linesPerBuffer-1;
				this->graph(0)->setName(this->rawLineName + QString::number(this->line));
			}
			//copy values from buffer to plot vector
			qreal max = -1000000000000;
			qreal min = 1000000000000;
			for(int i = 0; i<samplesPerLine && this->rawGrabbingAllowed; i++){
				//char
				if(bitDepth <= 8){
					char* bufferPointer = static_cast<char*>(buffer);
					this->sampleValues[i] = bufferPointer[this->line*samplesPerLine+i];
				}
				//ushort
				if(bitDepth > 8 && bitDepth <= 16){
					unsigned short* bufferPointer = static_cast<unsigned short*>(buffer);
					if(this->bitshift){
						this->sampleValues[i] = bufferPointer[this->line*samplesPerLine+i] >> 4;
					}else{
						this->sampleValues[i] = bufferPointer[this->line*samplesPerLine+i];
					}
				}
				//unsigned long int
				if(bitDepth > 16 && bitDepth <= 32){
					unsigned long int* bufferPointer = static_cast<unsigned long int*>(buffer);
					this->sampleValues[i] = bufferPointer[this->line*samplesPerLine+i];
				}

				if (this->sampleValues[i] < min) { min = this->sampleValues[i]; }
				if (this->sampleValues[i] > max) { max = this->sampleValues[i]; }
			}
			//update plot
			this->graph(0)->setName(this->rawLineName + QString::number(this->line)+" - Min: " + QString::number(min) + "   Max: " + QString::number(max) + "\nBuffer Id: " + QString::number(currentBufferNr));
			this->graph(0)->setData(this->sampleNumbers, this->sampleValues, true);
			if(this->autoscaling){
				this->graph(0)->rescaleAxes();
			}
			this->graph(0)->rescaleKeyAxis(true);
			this->replot();
			QCoreApplication::processEvents();
		}
		this->isPlottingRaw = false;
	}
}

void PlotWindow1D::slot_plotProcessedData(void* buffer, unsigned bitDepth, unsigned int samplesPerLine, unsigned int linesPerFrame, unsigned int framesPerBuffer, unsigned int buffersPerVolume, unsigned int currentBufferNr) {
	if(!this->isPlottingProcessed && this->displayProcessed && this->processedGrabbingAllowed){
		this->isPlottingProcessed = true;
		if(buffer != nullptr && this->isVisible()){
			//get length of one A-scan and resize plot vectors if necessary
			if(this->sampleValuesProcessed.size() != samplesPerLine){
				this->sampleValuesProcessed.resize(samplesPerLine);
				this->sampleNumbersProcessed.resize(samplesPerLine);
				for(int i = 0; i<samplesPerLine; i++){
					this->sampleNumbersProcessed[i] = i;
				}
			}
			//change line number if buffer size got smaller and there are now fewer lines in buffer than the current line number
			this->linesPerBuffer = linesPerFrame * framesPerBuffer;
			if(this->line > this->linesPerBuffer-1){
				this->line = this->linesPerBuffer-1;
				this->graph(1)->setName(this->processedLineName + QString::number(this->line));
			}
			//copy values from buffer to plot vector
			qreal max = -1000000000000;
			qreal min = 1000000000000;
			for(int i = 0; i<samplesPerLine && this->processedGrabbingAllowed; i++){
				//uchar
				if(bitDepth <= 8){
					unsigned char* bufferPointer = static_cast<unsigned char*>(buffer);
					this->sampleValuesProcessed[i] = bufferPointer[this->line*samplesPerLine+i];
				}
				//ushort
				if(bitDepth > 8 && bitDepth <= 16){
					unsigned short* bufferPointer = static_cast<unsigned short*>(buffer);
					this->sampleValuesProcessed[i] = bufferPointer[this->line*samplesPerLine+i];
				}
				//unsigned long int
				if(bitDepth > 16 && bitDepth <= 32){
					unsigned long int* bufferPointer = static_cast<unsigned long int*>(buffer);
					this->sampleValuesProcessed[i] = bufferPointer[this->line*samplesPerLine+i];
				}

				if (this->sampleValuesProcessed[i] < min) { min = this->sampleValuesProcessed[i]; }
				if (this->sampleValuesProcessed[i] > max) { max = this->sampleValuesProcessed[i]; }
			}
			//update plot
			this->graph(1)->setName(this->processedLineName + QString::number(this->line)+" - Min: " + QString::number(min) + "   Max: " + QString::number(max) + "\nBuffer Id: " + QString::number(currentBufferNr));
			this->graph(1)->setData(this->sampleNumbersProcessed, this->sampleValuesProcessed, true);
			if(this->autoscaling){
				this->graph(1)->rescaleAxes();
			}
			this->graph(1)->rescaleKeyAxis(true);
			this->replot();
			QCoreApplication::processEvents();
		}
		this->isPlottingProcessed = false;
	}
}

void PlotWindow1D::slot_changeLinesPerBuffer(int linesPerBuffer){
	this->linesPerBuffer = linesPerBuffer;
	this->panel->setMaxLineNr(linesPerBuffer);
}

void PlotWindow1D::slot_setLine(int lineNr){
	this->line = lineNr;

}

void PlotWindow1D::slot_displayRaw(bool display){
	this->displayRaw = display;
	this->setRawPlotVisible(display);
	this->replot();
}

void PlotWindow1D::slot_displayProcessed(bool display){
	this->displayProcessed = display;
	this->setProcessedPlotVisible(display);
	this->replot();
}

void PlotWindow1D::slot_activateAutoscaling(bool activate) {
	this->autoscaling = activate;
	this->replot();
}

void PlotWindow1D::slot_saveToDisk() {
	QString filters("Image (*.png);;Vector graphic (*.pdf)");
	QString defaultFilter("Image (*.png)");
	QString fileName = QFileDialog::getSaveFileName(this, tr("Save 1D Plot"), QDir::currentPath(), filters, &defaultFilter);
	if(fileName == ""){
		emit error(tr("Save plot to disk canceled."));
		return;
	}
	bool saved = false;
	if(defaultFilter == "Image (*.png)"){
		saved = this->savePng(fileName);
	}else if(defaultFilter == "Vector graphic (*.pdf)"){
		saved = this->savePdf(fileName);
	}
	if(saved){
		emit info(tr("Plot saved to ") + fileName);
	}else{
		emit error(tr("Could not save plot to disk."));
	}
}

void PlotWindow1D::slot_enableRawGrabbing(bool enable) {
	this->rawGrabbingAllowed = enable;
}

void PlotWindow1D::slot_enableProcessedGrabbing(bool enable) {
	this->processedGrabbingAllowed = enable;
}

void PlotWindow1D::slot_enableBitshift(bool enable) {
	this->bitshift = enable;
}












ControlPanel1D::ControlPanel1D(QWidget *parent) : QWidget(parent){
	this->panel = new QWidget(parent);
	QPalette pal;
	pal.setColor(QPalette::Background, QColor(32,32,32,128));
	this->panel->setAutoFillBackground(true);
	this->panel->setPalette(pal);
	this->labelLines = new QLabel(tr("Line Nr:"), this->panel);
	this->slider = new QSlider(Qt::Horizontal, this->panel);
	this->slider->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);
	this->slider->setMinimum(0);
	this->spinBoxLine = new QSpinBox(this->panel);
	this->spinBoxLine->setMinimum(0);
	this->checkBoxRaw = new QCheckBox(tr("Raw"), this->panel);
	this->checkBoxProcessed = new QCheckBox(tr("Processed"), this->panel);
	this->checkBoxAutoscale = new QCheckBox(tr("Autoscale Axis"), this->panel);

	this->widgetLayout = new QHBoxLayout(this);
	this->widgetLayout->addWidget(this->panel);

	this->layout = new QGridLayout(this->panel);
	this->layout->setContentsMargins(3,0,3,0);
	this->layout->setVerticalSpacing(1);
	this->layout->addWidget(this->labelLines, 0, 0, 1, 1, Qt::AlignRight);
	this->layout->addWidget(this->slider, 0, 1, 1, 5);
	this->layout->setColumnStretch(4, 100); //set high stretch factor such that slider gets max available space
	this->layout->addWidget(this->spinBoxLine, 0, 6, 1, 1, Qt::AlignLeft);
	this->layout->addWidget(this->checkBoxRaw, 1, 0, 1, 1, Qt::AlignRight);
	this->layout->addWidget(this->checkBoxProcessed, 1, 1, 1, 1, Qt::AlignRight);
	this->layout->addWidget(this->checkBoxAutoscale, 1, 3, 1, 1, Qt::AlignRight);

	connect(this->slider, &QSlider::valueChanged, this->spinBoxLine, &QSpinBox::setValue);
	connect(this->spinBoxLine, QOverload<int>::of(&QSpinBox::valueChanged), this->slider, &QSlider::setValue);
}

ControlPanel1D::~ControlPanel1D()
{
}

void ControlPanel1D::setMaxLineNr(unsigned int maxLineNr){
	this->slider->setMaximum(maxLineNr);
	this->spinBoxLine->setMaximum(maxLineNr);
}
