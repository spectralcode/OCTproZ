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




#define REC "record"
#define PROC "processing"
#define STREAM "streaming"
#define REC_PATH "path"
#define REC_RAW "record_raw"
#define REC_PROCESSED "record_processed"
#define REC_SCREENSHOTS "record_screenshots"
#define REC_STOP "stop_after_record"
#define REC_META "save_meta_info"
#define REC_VOLUMES "volumes"
#define REC_NAME "name"
#define REC_START_WITH_FIRST_BUFFER "start_with_first_buffer"
#define REC_DESCRIPTION "description"
#define PROC_FLIP_BSCANS "flip_bscans"
#define PROC_BITSHIFT "bitshift"
#define PROC_MIN "min"
#define PROC_MAX "max"
#define PROC_LOG "log"
#define PROC_COEFF "coeff"
#define PROC_ADDEND "addend"
#define PROC_RESAMPLING "resampling"
#define PROC_RESAMPLING_INTERPOLATION "resampling_interpolation"
#define PROC_RESAMPLING_C0 "resampling_c0"
#define PROC_RESAMPLING_C1 "resampling_c1"
#define PROC_RESAMPLING_C2 "resampling_c2"
#define PROC_RESAMPLING_C3 "resampling_c3"
#define PROC_DISPERSION_COMPENSATION "dispersion_compensation"
#define PROC_DISPERSION_COMPENSATION_D0 "dispersion_compensation_d0"
#define PROC_DISPERSION_COMPENSATION_D1 "dispersion_compensation_d1"
#define PROC_DISPERSION_COMPENSATION_D2 "dispersion_compensation_d2"
#define PROC_DISPERSION_COMPENSATION_D3 "dispersion_compensation_d3"
#define PROC_WINDOWING "windowing"
#define PROC_WINDOWING_TYPE "window_type"
#define PROC_WINDOWING_FILL_FACTOR "window_fill_factor"
#define PROC_WINDOWING_CENTER_POSITION "window_center_position"
#define PROC_FIXED_PATTERN_REMOVAL "fixed_pattern_removal"
#define PROC_FIXED_PATTERN_REMOVAL_CONTINUOUSLY "fixed_pattern_removal_continuously"
#define PROC_FIXED_PATTERN_REMOVAL_BSCANS "fixed_pattern_removal_bscans"
#define PROC_SINUSOIDAL_SCAN_CORRECTION "sinusoidal_scan_correction"
#define STREAM_STREAMING "streaming_enabled"
#define STREAM_STREAMING_SKIP "streaming_skip"





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
	void connectGuiElementsToAutosave();
	void disconnectGuiElementsFromAutosave();
	void connectUpdateProcessingParams();
	void updateProcessingParams();
	void updateStreamingParams(); //todo: find a nice way to enable/disable streaming (allocate/release memory for streaming buffers)
	void updateRecordingParams();
	void enableRecordTab(bool enable);
	void addActionsForKlinGroupBoxMenu(QList<QAction*> actions);

private:
	Ui::Sidebar 			ui;
	QDockWidget*			dock;
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
	QVariantMap recordSettings;
	QVariantMap processingSettings;
	QVariantMap streamingSettings;

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
	void updateSettingsMaps();

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












