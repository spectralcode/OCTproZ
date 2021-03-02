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

#include "sidebar.h"


Sidebar::Sidebar(QWidget *parent) : QWidget(parent) {
	this->dock = new QDockWidget();
	ui.setupUi(dock); 

	//Prevent accidental change of widget values by user inside QScrollArea
	this->findGuiElements();
	this->makeSpinBoxesScrollSave(this->spinBoxes);
	this->makeDoubleSpinBoxesScrollSave(this->doubleSpinBoxes);
	this->makeComboBoxesScrollSave(this->comboBoxes);
	this->makeCurvePlotsScrollSave(this->curvePlots); //todo: figure out why this does not work

	//Connect curve plots to dialogAboutToOpen signals (this is necessary as workaround for a bug that occurs on Linux systems: if an OpenGL window is open QFileDialog is not usable (the error message "GtkDialog mapped without a transient parent" occurs and software freezes)
	foreach(MiniCurvePlot* widget, this->curvePlots){
		connect(widget, &MiniCurvePlot::dialogAboutToOpen, this, &Sidebar::dialogAboutToOpen);
		connect(widget, &MiniCurvePlot::dialogClosed, this, &Sidebar::dialogClosed);
		connect(widget, &MiniCurvePlot::info, this, &Sidebar::info);
		connect(widget, &MiniCurvePlot::error, this, &Sidebar::error);
	}

	//Add record mode radiobuttons to QButtonGroup
	this->recModeGroup.addButton(this->ui.radioButton_snapshot);
	this->recModeGroup.addButton(this->ui.radioButton_raw);
	this->recModeGroup.addButton(this->ui.radioButton_processed);
	this->recModeGroup.addButton(this->ui.radioButton_rawAndProcessed);

	this->defaultWidth = static_cast<unsigned int>(this->dock->width());
	this->spacer = nullptr;
	this->initGui();
	
	//Connect gui signals to save changed settings
	this->connectSaveSettings();
	this->connectUpdateProcessingParams();

	this->resampleCurvePlot = this->ui.widget_resampleCurvePlot;
	this->dispersionCurvePlot = this->ui.widget_dispersionCurvePlot;
	this->dispersionCurvePlot->setCurveColor(QColor(55, 250, 100));
	this->windowCurvePlot = this->ui.widget_windowCurvePlot;
	this->windowCurvePlot->setCurveColor(QColor(250, 100, 55));
}


Sidebar::~Sidebar(){
	this->saveSettings();
	if (this->spacer != nullptr) {
		delete this->spacer;
	}
	delete this->dock;
}

void Sidebar::initGui() {
	//Init spacer which is used to hide/show sidebar properly
	this->spacer = new QWidget();
	this->spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	this->ui.dockWidgetContents->layout()->addWidget(this->spacer);
	this->spacer->hide();

	//Remove title bar
	this->dock->setTitleBarWidget(new QWidget());

	//Interpolation ComboBox
	QStringList interpolationOptions = { "Linear", "Cubic", "Lanczos"}; //todo: think of better way to add available options
	this->ui.comboBox_interpolation->addItems(interpolationOptions);

	//Windownig ComboBox
	QStringList windowingOptions = { "Hanning", "Gauss", "Sine", "Lanczos", "Rectangular" }; //todo: think of better way to add available windows to gui
	this->ui.comboBox_windowType->addItems(windowingOptions);

	//Gui connects
	connect(this->ui.pushButton_showSidebar, &QPushButton::clicked, this, &Sidebar::show);
	connect(this->ui.pushButton_redetermine, &QPushButton::clicked, this, &Sidebar::slot_redetermineFixedPatternNoise);
	connect(this->ui.radioButton_continuously, &QRadioButton::toggled, this, &Sidebar::slot_disableRedetermineButtion);
}

void Sidebar::findGuiElements() {
	this->checkBoxes = this->dock->findChildren<QCheckBox*>();
	this->doubleSpinBoxes = this->dock->findChildren<QDoubleSpinBox*>();
	this->spinBoxes = this->dock->findChildren<QSpinBox*>();
	this->groupBoxes = this->dock->findChildren<QGroupBox*>();
	this->comboBoxes = this->dock->findChildren<QComboBox*>();
	this->radioButtons = this->dock->findChildren<QRadioButton*>();
	this->curvePlots = this->dock->findChildren<MiniCurvePlot*>();
}

void Sidebar::makeSpinBoxesScrollSave(QList<QSpinBox*> widgetList) {
	foreach(auto widget, widgetList){
		widget->setFocusPolicy(Qt::StrongFocus);
		widget->installEventFilter(new EventGuard(widget));
	}
}

void Sidebar::makeDoubleSpinBoxesScrollSave(QList<QDoubleSpinBox*> widgetList) {
	foreach(auto widget, widgetList){
		widget->setFocusPolicy(Qt::StrongFocus);
		widget->installEventFilter(new EventGuard(widget));
	}
}

void Sidebar::makeComboBoxesScrollSave(QList<QComboBox*> widgetList) {
	foreach(auto widget, widgetList){
		widget->setFocusPolicy(Qt::StrongFocus);
		widget->installEventFilter(new EventGuard(widget));
	}
}

void Sidebar::makeCurvePlotsScrollSave(QList<MiniCurvePlot*> widgetList) {
	foreach(auto widget, widgetList){
		widget->setFocusPolicy(Qt::StrongFocus);
		widget->installEventFilter(new EventGuard(widget));
	}
}

void Sidebar::init(QAction* start, QAction* stop, QAction* rec, QAction* system, QAction* settings) {
	this->ui.toolButton_start->setDefaultAction(start);
	this->ui.toolButton_stop->setDefaultAction(stop);
	this->ui.toolButton_rec->setDefaultAction(rec);
	this->ui.toolButton_system->setDefaultAction(system);
	this->ui.toolButton_settings->setDefaultAction(settings);
	this->loadSettings();
}

void Sidebar::loadSettings() {
	this->disconnectSaveSettings();
	Settings* settings = Settings::getInstance();
	settings->loadSettings(SETTINGS_PATH);

	//Recording
	this->ui.lineEdit_saveFolder->setText(settings->recordSettings.value(REC_PATH).toString());
	RECORD_MODE currRecMode = (RECORD_MODE)settings->recordSettings.value(REC_MODE).toInt();
	switch (currRecMode) {
		case SNAPSHOT: this->ui.radioButton_snapshot->setChecked(true); break;
		case RAW: this->ui.radioButton_raw->setChecked(true); break;
		case PROCESSED: this->ui.radioButton_processed->setChecked(true); break;
		default: this->ui.radioButton_rawAndProcessed->setChecked(true);
	}
	this->ui.checkBox_stopAfterRec->setChecked(settings->recordSettings.value(REC_STOP).toBool());
	this->ui.checkBox_meta->setChecked(settings->recordSettings.value(REC_META).toBool());
	this->ui.spinBox_volumes->setValue(settings->recordSettings.value(REC_VOLUMES).toUInt());
	this->ui.spinBox_buffersToSkip->setValue(settings->recordSettings.value(REC_SKIP).toUInt());
	this->ui.lineEdit_recName->setText(settings->recordSettings.value(REC_NAME).toString());
	this->ui.plainTextEdit_description->setPlainText(settings->recordSettings.value(REC_DESCRIPTION).toString());

	//Processing
	this->ui.checkBox_bitshift->setChecked(settings->processingSettings.value(PROC_BITSHIFT).toBool());
	this->ui.checkBox_bscanFlip->setChecked(settings->processingSettings.value(PROC_FLIP_BSCANS).toBool());
	this->ui.checkBox_logScaling->setChecked(settings->processingSettings.value(PROC_LOG).toBool());
	this->ui.doubleSpinBox_signalMax->setValue(settings->processingSettings.value(PROC_MAX).toDouble());
	this->ui.doubleSpinBox_signalMin->setValue(settings->processingSettings.value(PROC_MIN).toDouble());
	this->ui.doubleSpinBox_signalMultiplicator->setValue(settings->processingSettings.value(PROC_COEFF).toDouble());
	this->ui.doubleSpinBox_signalAddend->setValue(settings->processingSettings.value(PROC_ADDEND).toDouble());
	this->ui.groupBox_resampling->setChecked(settings->processingSettings.value(PROC_RESAMPLING).toBool());
	this->ui.comboBox_interpolation->setCurrentIndex(settings->processingSettings.value(PROC_RESAMPLING_INTERPOLATION).toUInt());
	this->ui.doubleSpinBox_c0->setValue(settings->processingSettings.value(PROC_RESAMPLING_C0).toDouble());
	this->ui.doubleSpinBox_c1->setValue(settings->processingSettings.value(PROC_RESAMPLING_C1).toDouble());
	this->ui.doubleSpinBox_c2->setValue(settings->processingSettings.value(PROC_RESAMPLING_C2).toDouble());
	this->ui.doubleSpinBox_c3->setValue(settings->processingSettings.value(PROC_RESAMPLING_C3).toDouble());
	this->ui.groupBox_dispersionCompensation->setChecked(settings->processingSettings.value(PROC_DISPERSION_COMPENSATION).toBool());
	this->ui.doubleSpinBox_d0->setValue(settings->processingSettings.value(PROC_DISPERSION_COMPENSATION_D0).toDouble());
	this->ui.doubleSpinBox_d1->setValue(settings->processingSettings.value(PROC_DISPERSION_COMPENSATION_D1).toDouble());
	this->ui.doubleSpinBox_d2->setValue(settings->processingSettings.value(PROC_DISPERSION_COMPENSATION_D2).toDouble());
	this->ui.doubleSpinBox_d3->setValue(settings->processingSettings.value(PROC_DISPERSION_COMPENSATION_D3).toDouble());
	this->ui.groupBox_windowing->setChecked(settings->processingSettings.value(PROC_WINDOWING).toBool());
	this->ui.comboBox_windowType->setCurrentIndex(settings->processingSettings.value(PROC_WINDOWING_TYPE).toUInt());
	this->ui.doubleSpinBox_windowFillFactor->setValue(settings->processingSettings.value(PROC_WINDOWING_FILL_FACTOR).toDouble());
	this->ui.doubleSpinBox_windowCenterPosition->setValue(settings->processingSettings.value(PROC_WINDOWING_CENTER_POSITION).toDouble());
	this->ui.groupBox_fixedPatternNoiseRemoval->setChecked(settings->processingSettings.value(PROC_FIXED_PATTERN_REMOVAL).toBool());
	this->ui.radioButton_continuously->setChecked(settings->processingSettings.value(PROC_FIXED_PATTERN_REMOVAL_Continuously).toBool());
	this->ui.spinBox_bscansFixedNoise->setValue(settings->processingSettings.value(PROC_FIXED_PATTERN_REMOVAL_BSCANS).toUInt());
	this->ui.checkBox_sinusoidalScanCorrection->setChecked(settings->processingSettings.value(PROC_SINUSOIDAL_SCAN_CORRECTION).toBool());

	//GPU to RAM Streaming
	this->ui.groupBox_streaming->setChecked(settings->streamingSettings.value(STREAM_STREAMING).toBool());
	this->ui.spinBox_streamingBuffersToSkip->setValue(settings->streamingSettings.value(STREAM_STREAMING_SKIP).toUInt());

	this->connectSaveSettings();
}

void Sidebar::saveSettings() {
	Settings* settings = Settings::getInstance();

	//Recording
	settings->recordSettings.insert(REC_PATH, this->ui.lineEdit_saveFolder->text());
	RECORD_MODE currRecMode = this->ui.radioButton_raw->isChecked() ? RAW : this->ui.radioButton_snapshot->isChecked() ? SNAPSHOT : this->ui.radioButton_processed->isChecked() ? PROCESSED : ALL;
	settings->recordSettings.insert(REC_MODE, currRecMode);
	settings->recordSettings.insert(REC_STOP, this->ui.checkBox_stopAfterRec->isChecked());
	settings->recordSettings.insert(REC_META, this->ui.checkBox_meta->isChecked());
	settings->recordSettings.insert(REC_VOLUMES, this->ui.spinBox_volumes->value());
	settings->recordSettings.insert(REC_SKIP, this->ui.spinBox_buffersToSkip->value());
	settings->recordSettings.insert(REC_NAME, this->ui.lineEdit_recName->text());
	settings->recordSettings.insert(REC_DESCRIPTION, this->ui.plainTextEdit_description->toPlainText());

	//Processing
	settings->processingSettings.insert(PROC_BITSHIFT, this->ui.checkBox_bitshift->isChecked());
	settings->processingSettings.insert(PROC_FLIP_BSCANS, this->ui.checkBox_bscanFlip->isChecked());
	settings->processingSettings.insert(PROC_LOG, this->ui.checkBox_logScaling->isChecked());
	settings->processingSettings.insert(PROC_MAX, this->ui.doubleSpinBox_signalMax->value());
	settings->processingSettings.insert(PROC_MIN, this->ui.doubleSpinBox_signalMin->value());
	settings->processingSettings.insert(PROC_COEFF, this->ui.doubleSpinBox_signalMultiplicator->value());
	settings->processingSettings.insert(PROC_ADDEND, this->ui.doubleSpinBox_signalAddend->value());
	settings->processingSettings.insert(PROC_RESAMPLING, this->ui.groupBox_resampling->isChecked());
	settings->processingSettings.insert(PROC_RESAMPLING_INTERPOLATION, this->ui.comboBox_interpolation->currentIndex());
	settings->processingSettings.insert(PROC_RESAMPLING_C0, this->ui.doubleSpinBox_c0->value());
	settings->processingSettings.insert(PROC_RESAMPLING_C1, this->ui.doubleSpinBox_c1->value());
	settings->processingSettings.insert(PROC_RESAMPLING_C2, this->ui.doubleSpinBox_c2->value());
	settings->processingSettings.insert(PROC_RESAMPLING_C3, this->ui.doubleSpinBox_c3->value());
	settings->processingSettings.insert(PROC_DISPERSION_COMPENSATION, this->ui.groupBox_dispersionCompensation->isChecked());
	settings->processingSettings.insert(PROC_DISPERSION_COMPENSATION_D0, this->ui.doubleSpinBox_d0->value());
	settings->processingSettings.insert(PROC_DISPERSION_COMPENSATION_D1, this->ui.doubleSpinBox_d1->value());
	settings->processingSettings.insert(PROC_DISPERSION_COMPENSATION_D2, this->ui.doubleSpinBox_d2->value());
	settings->processingSettings.insert(PROC_DISPERSION_COMPENSATION_D3, this->ui.doubleSpinBox_d3->value());
	settings->processingSettings.insert(PROC_WINDOWING, this->ui.groupBox_windowing->isChecked());
	settings->processingSettings.insert(PROC_WINDOWING_TYPE, this->ui.comboBox_windowType->currentIndex());
	settings->processingSettings.insert(PROC_WINDOWING_FILL_FACTOR, this->ui.doubleSpinBox_windowFillFactor->value());
	settings->processingSettings.insert(PROC_WINDOWING_CENTER_POSITION, this->ui.doubleSpinBox_windowCenterPosition->value());
	settings->processingSettings.insert(PROC_FIXED_PATTERN_REMOVAL, this->ui.groupBox_fixedPatternNoiseRemoval->isChecked());
	settings->processingSettings.insert(PROC_FIXED_PATTERN_REMOVAL_Continuously, this->ui.radioButton_continuously->isChecked());
	settings->processingSettings.insert(PROC_FIXED_PATTERN_REMOVAL_BSCANS, this->ui.spinBox_bscansFixedNoise->value());
	settings->processingSettings.insert(PROC_SINUSOIDAL_SCAN_CORRECTION, this->ui.checkBox_sinusoidalScanCorrection->isChecked());

	//GPU to RAM Streaming
	settings->streamingSettings.insert(STREAM_STREAMING, this->ui.groupBox_streaming->isChecked());
	settings->streamingSettings.insert(STREAM_STREAMING_SKIP, this->ui.spinBox_streamingBuffersToSkip->value());


	settings->storeSettings(SETTINGS_PATH);
	//emit info("Settings saved");
}

void Sidebar::connectSaveSettings() { //todo: check if this method is really necessary
	//Connects to store recording settings
	connect(this->ui.toolButton, &QToolButton::clicked, this, &Sidebar::slot_selectSaveDir);
	connect(this->ui.lineEdit_saveFolder, &QLineEdit::textChanged, this, &Sidebar::saveSettings);
	connect(this->ui.lineEdit_saveFolder, &QLineEdit::editingFinished, this, &Sidebar::saveSettings);
	connect(&(this->recModeGroup), static_cast<void (QButtonGroup::*)(int)>(&QButtonGroup::buttonClicked), this, &Sidebar::saveSettings);
	connect(this->ui.checkBox_stopAfterRec, &QCheckBox::clicked, this, &Sidebar::saveSettings);
	connect(this->ui.checkBox_meta, &QCheckBox::clicked, this, &Sidebar::saveSettings);
	connect(this->ui.spinBox_volumes, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, &Sidebar::saveSettings);
	connect(this->ui.spinBox_buffersToSkip, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, &Sidebar::saveSettings);
	connect(this->ui.lineEdit_recName, &QLineEdit::textChanged, this, &Sidebar::saveSettings);
	connect(this->ui.plainTextEdit_description, &QPlainTextEdit::textChanged, this, &Sidebar::saveSettings); //this may result in too many hard drive write operations
}

void Sidebar::disconnectSaveSettings() {
	//Disconnects to store recording settings
	disconnect(this->ui.toolButton, &QToolButton::clicked, this, &Sidebar::slot_selectSaveDir);
	disconnect(this->ui.lineEdit_saveFolder, &QLineEdit::textChanged, this, &Sidebar::saveSettings);
	disconnect(this->ui.lineEdit_saveFolder, &QLineEdit::editingFinished, this, &Sidebar::saveSettings);
	disconnect(&(this->recModeGroup), static_cast<void (QButtonGroup::*)(int)>(&QButtonGroup::buttonClicked), this, &Sidebar::saveSettings);
	disconnect(this->ui.checkBox_stopAfterRec, &QCheckBox::clicked, this, &Sidebar::saveSettings);
	disconnect(this->ui.checkBox_meta, &QCheckBox::clicked, this, &Sidebar::saveSettings);
	disconnect(this->ui.spinBox_volumes, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, &Sidebar::saveSettings);
	disconnect(this->ui.spinBox_buffersToSkip, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, &Sidebar::saveSettings);
	disconnect(this->ui.lineEdit_recName, &QLineEdit::textChanged, this, &Sidebar::saveSettings);
	disconnect(this->ui.plainTextEdit_description, &QPlainTextEdit::textChanged, this, &Sidebar::saveSettings);
}

void Sidebar::connectUpdateProcessingParams() {
	//Connect gui elements to updateProcessingParams slot to allow live update of oct algorithm parameters
	foreach(auto checkBox, this->checkBoxes){
		connect(checkBox, &QCheckBox::clicked, this, &Sidebar::slot_updateProcessingParams);
	}
	foreach(auto doubleSpinBox, this->doubleSpinBoxes){
		connect(doubleSpinBox, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, &Sidebar::slot_updateProcessingParams);
	}
	foreach(auto spinBox, this->spinBoxes){
		connect(spinBox, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, &Sidebar::slot_updateProcessingParams);
	}
	foreach(auto groupBox, this->groupBoxes){
		connect(groupBox, &QGroupBox::toggled, this, &Sidebar::slot_updateProcessingParams);
	}
	foreach(auto comboBox, this->comboBoxes){
		connect(comboBox, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &Sidebar::slot_updateProcessingParams);
	}
	foreach(auto radioButton, this->radioButtons){
		connect(radioButton, &QRadioButton::toggled, this, &Sidebar::slot_updateProcessingParams);
	}
}

void Sidebar::updateProcessingParams() {
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	params->bitshift = this->ui.checkBox_bitshift->isChecked();
	params->bscanFlip = this->ui.checkBox_bscanFlip->isChecked();
	params->signalLogScaling = this->ui.checkBox_logScaling->isChecked();
	params->signalGrayscaleMax = this->ui.doubleSpinBox_signalMax->value();
	params->signalGrayscaleMin = this->ui.doubleSpinBox_signalMin->value();
	params->signalMultiplicator = this->ui.doubleSpinBox_signalMultiplicator->value();
	params->signalAddend = this->ui.doubleSpinBox_signalAddend->value();
	params->stopAfterRecord = this->ui.checkBox_stopAfterRec->isChecked();
	params->numberOfBuffersToRecord = this->ui.spinBox_volumes->value();
	params->buffersToSkip = this->ui.spinBox_buffersToSkip->value();
	params->fixedPatternNoiseRemoval = this->ui.groupBox_fixedPatternNoiseRemoval->isChecked();
	params->continuousFixedPatternNoiseDetermination = this->ui.radioButton_continuously->isChecked();
	params->bscansForNoiseDetermination = this->ui.spinBox_bscansFixedNoise->value();
	params->sinusoidalScanCorrection = this->ui.checkBox_sinusoidalScanCorrection->isChecked();
}

void Sidebar::updateStreamingParams() {
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	params->streamingParamsChanged = params->streamToHost == this->ui.groupBox_streaming->isChecked() ? false : true;
	params->streamToHost = this->ui.groupBox_streaming->isChecked();
	params->streamingBuffersToSkip = this->ui.spinBox_streamingBuffersToSkip->value();
}

void Sidebar::enableRecordTab(bool enable){
	this->ui.tab_3->setEnabled(enable);
}

void Sidebar::updateResamplingParams() {
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	int interpolation = this->ui.comboBox_interpolation->currentIndex();
	INTERPOLATION interpolationOption = (INTERPOLATION) interpolation;
	params->resamplingInterpolation = interpolationOption;
	double c0 = this->ui.doubleSpinBox_c0->value();
	double c1 = this->ui.doubleSpinBox_c1->value();
	double c2 = this->ui.doubleSpinBox_c2->value();
	double c3 = this->ui.doubleSpinBox_c3->value();
	bool resampling = this->ui.groupBox_resampling->isChecked();
	if (c0 != params->c0 || c1 != params->c1 || c2 != params->c2 || c3 != params->c3 || params->acquisitionParamsChanged) {
		params->c0 = c0;
		params->c1 = c1;
		params->c2 = c2;
		params->c3 = c3;
		params->updateResampleCurve();
		this->resampleCurvePlot->plotCurves(params->resampleCurve, params->resampleReferenceCurve, params->samplesPerLine);
	}
	params->resampling = resampling;
}

void Sidebar::updateDispersionParams() {
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	double d0 = this->ui.doubleSpinBox_d0->value();
	double d1 = this->ui.doubleSpinBox_d1->value();
	double d2 = this->ui.doubleSpinBox_d2->value();
	double d3 = this->ui.doubleSpinBox_d3->value();
	bool dispersionCompensation = this->ui.groupBox_dispersionCompensation->isChecked();

	if (d0 != params->d0 || d1 != params->d1 || d2 != params->d2 || d3 != params->d3 || params->acquisitionParamsChanged) {
		params->d0 = d0;
		params->d1 = d1;
		params->d2 = d2;
		params->d3 = d3;
		params->updateDispersionCurve();
		this->dispersionCurvePlot->plotCurves(params->dispersionCurve, params->dispersionReferenceCurve, params->samplesPerLine);
	}
	params->dispersionCompensation = dispersionCompensation;
}

void Sidebar::updateWindowingParams() {
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	double center = this->ui.doubleSpinBox_windowCenterPosition->value();
	double fillFactor = this->ui.doubleSpinBox_windowFillFactor->value();
	int window = this->ui.comboBox_windowType->currentIndex();
	WindowFunction::WindowType windowType = (WindowFunction::WindowType) window;
	bool windowing = this->ui.groupBox_windowing->isChecked();

	if (windowType != params->window || center != params->windowCenter || fillFactor != params->windowFillFactor || params->acquisitionParamsChanged) {
		params->windowCenter = center;
		params->windowFillFactor = fillFactor;
		params->window =  windowType;
		params->updateWindowCurve();
		this->windowCurvePlot->plotCurves(params->windowCurve, params->windowReferenceCurve, params->samplesPerLine);
	}
	params->windowing = windowing;
}

void Sidebar::slot_selectSaveDir() {
	emit dialogAboutToOpen();
	QCoreApplication::processEvents();
	Settings* settings = Settings::getInstance();
	QString savedPath = settings->recordSettings.value(REC_PATH).toString();
	QString standardLocation = savedPath.size() == 0 ? QStandardPaths::writableLocation(QStandardPaths::DesktopLocation) : savedPath;
	this->ui.lineEdit_saveFolder->setText(QFileDialog::getExistingDirectory(this, tr("Select OCTrpoZ Save Folder"), standardLocation, QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks ));
	emit dialogClosed();
}

void Sidebar::slot_updateInfoBox(QString volumesPerSecond, QString buffersPerSecond, QString bscansPerSecond, QString ascansPerSecond, QString bufferSizeMB, QString dataThroughput) {
	this->ui.label_volumesPerSecond->setText(volumesPerSecond);
	this->ui.label_buffersPerSecond->setText(buffersPerSecond);
	this->ui.label_bscansPerSecond->setText(bscansPerSecond);
	this->ui.label_ascansPerSecond->setText(ascansPerSecond);
	this->ui.label_bufferSize->setText(bufferSizeMB);
	this->ui.label_dataThroughput->setText(dataThroughput);
}

void Sidebar::slot_updateProcessingParams() {
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	this->updateResamplingParams();
	this->updateDispersionParams();
	this->updateWindowingParams();
	this->updateStreamingParams();
	this->updateProcessingParams();
	params->acquisitionParamsChanged = false;
}

void Sidebar::slot_redetermineFixedPatternNoise() {
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	params->redetermineFixedPatternNoise = true;
}

void Sidebar::slot_disableRedetermineButtion(bool disable){
	this->ui.pushButton_redetermine->setDisabled(disable);
}

void Sidebar::slot_setMaximumBscansForNoiseDetermination(unsigned int max) {
	this->ui.spinBox_bscansFixedNoise->setMaximum(max);
}

void Sidebar::slot_setKLinCoeffs(double* k0, double* k1, double* k2, double* k3) {
	if(k0 != nullptr){
		this->ui.doubleSpinBox_c0->setValue(*k0);
	}
	if(k1 != nullptr){
		this->ui.doubleSpinBox_c1->setValue(*k1);
	}
	if(k2 != nullptr){
		this->ui.doubleSpinBox_c2->setValue(*k2);
	}
	if(k3 != nullptr){
		this->ui.doubleSpinBox_c3->setValue(*k3);
	}
	QApplication::processEvents();
	emit klinCoeffs(this->ui.doubleSpinBox_c0->value(), this->ui.doubleSpinBox_c1->value(), this->ui.doubleSpinBox_c2->value(), this->ui.doubleSpinBox_c3->value());
}

void Sidebar::slot_setDispCompCoeffs(double *d0, double *d1, double *d2, double *d3) {
	if(d0 != nullptr){
		this->ui.doubleSpinBox_d0->setValue(*d0);
	}
	if(d1 != nullptr){
		this->ui.doubleSpinBox_d1->setValue(*d1);
	}
	if(d2 != nullptr){
		this->ui.doubleSpinBox_c2->setValue(*d2);
	}
	if(d3 != nullptr){
		this->ui.doubleSpinBox_c3->setValue(*d3);
	}
	QApplication::processEvents();
	emit dispCompCoeffs(this->ui.doubleSpinBox_d0->value(), this->ui.doubleSpinBox_d1->value(), this->ui.doubleSpinBox_d2->value(), this->ui.doubleSpinBox_d3->value());
}

void Sidebar::show() {
	bool visible = this->ui.widget_sidebarContent->isVisible();
	if (visible) {
		this->ui.widget_sidebarContent->setVisible(false);
		this->dock->setFixedWidth(this->ui.pushButton_showSidebar->width());
		this->spacer->show();
		this->ui.pushButton_showSidebar->setText(">");
		this->ui.pushButton_showSidebar->setToolTip(tr("Show sidebar"));
	}
	else {
		this->ui.widget_sidebarContent->setVisible(true);
		this->dock->setFixedWidth(this->defaultWidth);
		this->spacer->hide();
		this->ui.pushButton_showSidebar->setText("<");
		this->ui.pushButton_showSidebar->setToolTip(tr("Hide sidebar"));
	}
}
