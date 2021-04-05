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

#ifndef SIDEBAR_H
#define SIDEBAR_H

#include <QWidget>
#include <QButtonGroup>
#include <QDockWidget>
#include <QFileDialog>
#include "minicurveplot.h"
#include "settings.h"
#include "octalgorithmparameters.h"
#include "eventguard.h"

#include "ui_sidebar.h"


class Sidebar : public QWidget
{
	Q_OBJECT
public:
	explicit Sidebar(QWidget *parent = nullptr);
	~Sidebar();

	QDockWidget* getDock() { return this->dock; }
	Ui::Sidebar getUi() { return this->ui; }

	void init(QAction* start, QAction* stop, QAction* rec, QAction* system, QAction* settings);
	void loadSettings();
	void saveSettings();
	void connectSaveSettings();
	void disconnectSaveSettings();
	void connectUpdateProcessingParams();
	void updateProcessingParams();
	void updateStreamingParams(); //todo: find a nice way to enable/disable streaming (allocate/release memory for streaming buffers)
	void enableRecordTab(bool enable);
	void addActionsForKlinGroupBoxMenu(QList<QAction*> actions);

private:
	Ui::Sidebar 			ui;
	QDockWidget*			dock;
	QButtonGroup			recModeGroup;
	QWidget*				spacer;
	QList<QLineEdit*>		lineEdits;
	QList<QCheckBox*>		checkBoxes;
	QList<QDoubleSpinBox*>	doubleSpinBoxes;
	QList<QSpinBox*>		spinBoxes;
	QList<QGroupBox*>		groupBoxes;
	QList<QComboBox*>		comboBoxes;
	QList<QRadioButton*>	radioButtons;
	QList<MiniCurvePlot*>	curvePlots;
	MiniCurvePlot*			resampleCurvePlot;
	MiniCurvePlot*			dispersionCurvePlot;
	MiniCurvePlot*			windowCurvePlot;
	unsigned int			defaultWidth;
	QAction*				copyInfoAction;

	void initGui();
	void findGuiElements();
	void makeSpinBoxesScrollSave(QList<QSpinBox*> widgetList);
	void makeDoubleSpinBoxesScrollSave(QList<QDoubleSpinBox*> widgetList);
	void makeComboBoxesScrollSave(QList<QComboBox*> widgetList);
	void makeCurvePlotsScrollSave(QList<MiniCurvePlot*> widgetList);
	void updateResamplingParams();
	void updateDispersionParams();
	void updateWindowingParams();

public slots:
	void slot_selectSaveDir();
	void slot_updateInfoBox(QString volumesPerSecond, QString buffersPerSecond, QString bscansPerSecond, QString ascansPerSecond, QString volumeSizeMB, QString dataThroughput);
	void slot_updateProcessingParams();
	void slot_redetermineFixedPatternNoise();
	void slot_disableRedetermineButtion(bool disable);
	void slot_setMaximumBscansForNoiseDetermination(unsigned int max);
	void slot_setKLinCoeffs(double* k0, double* k1, double* k2, double* k3);
	void slot_setDispCompCoeffs(double* d0, double* d1, double* d2, double* d3);
	void disableKlinCoeffInput(bool disable);
	void copyInfoToClipboard();
	void show();

signals:
	void streamingParamsChanged();
	void dialogAboutToOpen();
	void dialogClosed();
	void klinCoeffs(double k0, double k1, double k2, double k3);
	void dispCompCoeffs(double d0, double d1, double d2, double d3);
	void error(QString);
	void info(QString);
};
#endif // SIDEBAR_H












