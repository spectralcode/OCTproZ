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

#ifndef PLOTWINDOW1D_H
#define PLOTWINDOW1D_H

#define PLOT1D_DISPLAY_RAW "plot1d_display_raw"
#define PLOT1D_DISPLAY_PROCESSED "plot1d_display_processed"
#define PLOT1D_AUTOSCALING "plot1d_autoscaling_enabled"
#define PLOT1D_BITSHIFT "plot1d_bitshift_enabled"
#define PLOT1D_LINE_NR "plot1d_line_nr"
#define PLOT1D_DATA_CURSOR "plot1d_data_cursor_enabled"
#define PLOT1D_SHOW_LEGEND "plot1d_show_legend"

#include "qcustomplot.h"
#include "octproz_devkit.h"

class ControlPanel1D;
class PlotWindow1D : public QCustomPlot
{
	Q_OBJECT
public:
	explicit PlotWindow1D(QWidget *parent = nullptr);
	~PlotWindow1D();

	QString getName() const { return this->name; }
	void setName(const QString &name) { this->name = name; }
	void setSettings(QVariantMap settings);
	QVariantMap getSettings();

	QSize sizeHint() const override;

	//QCustomPlot* customPlot;
	QVector<qreal> sampleNumbers;
	QVector<qreal> sampleValues;
	QVector<qreal> sampleNumbersProcessed;
	QVector<qreal> sampleValuesProcessed;



private:
	void setRawAxisColor(QColor color);
	void setProcessedAxisColor(QColor color);
	void setRawPlotVisible(bool visible);
	void setProcessedPlotVisible(bool visible);
	bool saveAllCurvesToFile(QString fileName);

	int line;
	int linesPerBuffer;
	int currentRawBitdepth;
	bool isPlottingRaw;
	bool isPlottingProcessed;
	bool autoscaling;
	bool displayRaw;
	bool displayProcessed;
	bool bitshift;
	bool rawGrabbingAllowed;
	bool processedGrabbingAllowed;
	QString rawLineName;
	QString processedLineName;
	QString name;

	QColor processedColor;
	QColor rawColor;

	ControlPanel1D* panel;
	QVBoxLayout* layout;

	bool dataCursorEnabled;
	QLabel* coordinateDisplay;


protected:
	void contextMenuEvent(QContextMenuEvent* event) override;
	void mouseDoubleClickEvent(QMouseEvent *event) override;
	void mouseMoveEvent(QMouseEvent* event) override;
	void leaveEvent(QEvent* event) override;

signals:
	void info(QString info);
	void error(QString error);


public slots:
	void slot_plotRawData(void* buffer, unsigned bitDepth, unsigned int samplesPerLine, unsigned int linesPerFrame, unsigned int framesPerBuffer, unsigned int buffersPerVolume, unsigned int currentBufferNr);
	void slot_plotProcessedData(void* buffer, unsigned bitDepth, unsigned int samplesPerLine, unsigned int linesPerFrame, unsigned int framesPerBuffer, unsigned int buffersPerVolume, unsigned int currentBufferNr);
	void slot_changeLinesPerBuffer(int linesPerBuffer);
	void slot_setLine(int lineNr);
	void slot_displayRaw(bool display);
	void slot_displayProcessed(bool display);
	void slot_activateAutoscaling(bool activate);
	void slot_saveToDisk();
	void slot_enableRawGrabbing(bool enable);
	void slot_enableProcessedGrabbing(bool enable);
	void slot_enableBitshift(bool enable);
	void slot_toggleDualCoordinates(bool enabled);
	void slot_toggleLegend(bool enabled);
	void zoomSelectedAxisWithMouseWheel();
	void dragSelectedAxes();
	void combineSelections();
};




class ControlPanel1D : public QWidget
{
	Q_OBJECT

public:
	ControlPanel1D(QWidget* parent);
	~ControlPanel1D();

	void setMaxLineNr(unsigned int maxLineNr);

private:
	QWidget* panel;
	QLabel* labelLines;
	QSlider* slider;
	QSpinBox* spinBoxLine;

	QCheckBox* checkBoxRaw;
	QCheckBox* checkBoxProcessed;
	QCheckBox* checkBoxAutoscale;

	QHBoxLayout* widgetLayout;
	QGridLayout* layout;

protected:
	void enterEvent(QEvent *event) override;

public slots:


signals:
	void mouseEntered();

friend class PlotWindow1D;
};





#endif // PLOTWINDOW1D_H
