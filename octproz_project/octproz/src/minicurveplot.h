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
**			iqo.uni-hannover.de
****
**/

#ifndef MINICURVEPLOT_H
#define MINICURVEPLOT_H

#include "qcustomplot.h"

class MiniCurvePlot : public QCustomPlot
{
	Q_OBJECT
public:
	explicit MiniCurvePlot(QWidget *parent = nullptr);
	~MiniCurvePlot();

	void setCurveColor(QColor color);
	void setReferenceCurveColor(QColor color);
	void plotCurves(float* curve, float* referenceCurve, unsigned int samples);
	void roundCorners(bool enable){this->drawRoundCorners = enable;}


private:
	void setAxisColor(QColor color);
	void zoomOutSlightly();

	QVector<qreal> sampleNumbers;
	QVector<qreal> curve;
	QVector<qreal> referenceCurve;
	bool drawRoundCorners;
	QColor curveColor;
	QColor referenceCurveColor;
	int referenceCurveAlpha;

protected:
	void contextMenuEvent(QContextMenuEvent* event) override;
	void mouseMoveEvent(QMouseEvent* event) override;
	void resizeEvent(QResizeEvent* event) override;
	void changeEvent(QEvent* event) override;

signals:
	void info(QString info);
	void error(QString error);
	void dialogAboutToOpen(); //this is necessary as workaround for a bug that occurs on Linux systems: if an OpenGL window is open QFileDialog is not usable (the error message "GtkDialog mapped without a transient parent" occurs and software freezes)
	void dialogClosed();


public slots:
	virtual void mouseDoubleClickEvent(QMouseEvent* event) override;
	bool saveCurveDataToFile(QString fileName);
	void slot_saveToDisk();
};


#endif // MINICURVEPLOT_H
