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

#include "sidebar.h"
#include <QtGlobal>


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

	this->defaultWidth = static_cast<unsigned int>(this->dock->width());
	this->spacer = nullptr;
	this->initGui();
	
	//Connect gui signals to save changed settings
	this->connectGuiElementsToAutosave(); //todo: rethink if this is really useful. maybe make this optional and add it to the general application settings
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
	QStringList windowingOptions = { "Hanning", "Gauss", "Sine", "Lanczos", "Rectangular", "Flat Top" }; //todo: think of better way to add available windows to gui
	this->ui.comboBox_windowType->addItems(windowingOptions);

	//Gui connects
	connect(this->ui.pushButton_showSidebar, &QPushButton::clicked, this, &Sidebar::show);
	connect(this->ui.pushButton_redetermine, &QPushButton::clicked, this, &Sidebar::slot_redetermineFixedPatternNoise);
	connect(this->ui.radioButton_continuously, &QRadioButton::toggled, this, &Sidebar::slot_disableRedetermineButtion);
	connect(this->ui.toolButton_recPath, &QToolButton::clicked, this, &Sidebar::slot_selectSaveDir);
	connect(this->ui.pushButton_postProcRec, &QPushButton::clicked, this, &Sidebar::slot_recordPostProcessingBackground);
	connect(this->ui.pushButton_postProcSave, &QPushButton::clicked, this, &Sidebar::slot_savePostProcessingBackground);
	connect(this->ui.pushButton_postProcLoad, &QPushButton::clicked, this, &Sidebar::slot_loadPostProcessingBackground);

	this->copyInfoAction = new QAction(tr("Copy info to clipboard"), this);
	connect(copyInfoAction, &QAction::triggered, this, &Sidebar::copyInfoToClipboard);
	this->ui.groupBox_info->addAction(copyInfoAction);

	//Tool tips
	this->ui.groupBox_streaming->setToolTip("<html><head/><body><p>"+tr("This setting enables continuous transfer of processed OCT data to memory. This allows all plugins to access the processed OCT data. It must be activated if you want to display processed A-scans in the 1D plot.")+"</p></body></html>"); //html tags are used to enable word wrapping inside tool tip
}

void Sidebar::findGuiElements() {
	this->lineEdits = this->dock->findChildren<QLineEdit*>();
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
	this->disconnectGuiElementsFromAutosave();
	Settings* settingsManager = Settings::getInstance();

	//load setting maps
	this->recordSettings = settingsManager->getStoredSettings(REC);
	this->processingSettings = settingsManager->getStoredSettings(PROC);
	this->streamingSettings = settingsManager->getStoredSettings(STREAM);

	//Recording
	this->ui.lineEdit_saveFolder->setText(this->recordSettings.value(REC_PATH).toString());
	this->ui.checkBox_recordScreenshots->setChecked(this->recordSettings.value(REC_SCREENSHOTS).toBool());
	this->ui.checkBox_recordRawBuffers->setChecked(this->recordSettings.value(REC_RAW).toBool());
	this->ui.checkBox_recordProcessedBuffers->setChecked(this->recordSettings.value(REC_PROCESSED).toBool());
	this->ui.checkBox_startWithFirstBuffer->setChecked(this->recordSettings.value(REC_START_WITH_FIRST_BUFFER).toBool());
	this->ui.checkBox_stopAfterRec->setChecked(this->recordSettings.value(REC_STOP).toBool());
	this->ui.checkBox_meta->setChecked(this->recordSettings.value(REC_META).toBool());
	this->ui.spinBox_volumes->setValue(this->recordSettings.value(REC_VOLUMES).toUInt());
	this->ui.lineEdit_recName->setText(this->recordSettings.value(REC_NAME).toString());
	this->ui.plainTextEdit_description->setPlainText(this->recordSettings.value(REC_DESCRIPTION).toString());

	//Processing
	this->ui.checkBox_bitshift->setChecked(this->processingSettings.value(PROC_BITSHIFT).toBool());
	this->ui.checkBox_bscanFlip->setChecked(this->processingSettings.value(PROC_FLIP_BSCANS).toBool());
	this->ui.groupBox_backgroundremoval->setChecked(this->processingSettings.value(PROC_REMOVEBACKGROUND).toBool());
	this->ui.spinBox_rollingAverageWindowSize->setValue(this->processingSettings.value(PROC_REMOVEBACKGROUND_WINDOW_SIZE).toUInt());
	this->ui.checkBox_logScaling->setChecked(this->processingSettings.value(PROC_LOG).toBool());
	this->ui.doubleSpinBox_signalMax->setValue(this->processingSettings.value(PROC_MAX).toDouble());
	this->ui.doubleSpinBox_signalMin->setValue(this->processingSettings.value(PROC_MIN).toDouble());
	this->ui.doubleSpinBox_signalMultiplicator->setValue(this->processingSettings.value(PROC_COEFF).toDouble());
	this->ui.doubleSpinBox_signalAddend->setValue(this->processingSettings.value(PROC_ADDEND).toDouble());
	this->ui.groupBox_resampling->setChecked(this->processingSettings.value(PROC_RESAMPLING).toBool());
	this->ui.comboBox_interpolation->setCurrentIndex(this->processingSettings.value(PROC_RESAMPLING_INTERPOLATION).toUInt());
	this->ui.doubleSpinBox_c0->setValue(this->processingSettings.value(PROC_RESAMPLING_C0).toDouble());
	this->ui.doubleSpinBox_c1->setValue(this->processingSettings.value(PROC_RESAMPLING_C1).toDouble());
	this->ui.doubleSpinBox_c2->setValue(this->processingSettings.value(PROC_RESAMPLING_C2).toDouble());
	this->ui.doubleSpinBox_c3->setValue(this->processingSettings.value(PROC_RESAMPLING_C3).toDouble());
	//this->actionUseCustomKLinCurve->setChecked(this->processingSettings.value(PROC_CUSTOM_RESAMPLING).toBool());
	OctAlgorithmParameters::getInstance()->useCustomResampleCurve = this->processingSettings.value(PROC_CUSTOM_RESAMPLING).toBool(); //todo: move all actions for klin from octproz to sidebar
	QString customResamplingFilePath = this->processingSettings.value(PROC_CUSTOM_RESAMPLING_FILEPATH).toString();
	if (customResamplingFilePath.isEmpty() || !QFile::exists(customResamplingFilePath)) {
		emit loadResamplingCurveRequested(SETTINGS_PATH_RESAMPLING_FILE);
	} else {
		emit loadResamplingCurveRequested(customResamplingFilePath);
	}
	this->ui.groupBox_dispersionCompensation->setChecked(this->processingSettings.value(PROC_DISPERSION_COMPENSATION).toBool());
	this->ui.doubleSpinBox_d0->setValue(this->processingSettings.value(PROC_DISPERSION_COMPENSATION_D0).toDouble());
	this->ui.doubleSpinBox_d1->setValue(this->processingSettings.value(PROC_DISPERSION_COMPENSATION_D1).toDouble());
	this->ui.doubleSpinBox_d2->setValue(this->processingSettings.value(PROC_DISPERSION_COMPENSATION_D2).toDouble());
	this->ui.doubleSpinBox_d3->setValue(this->processingSettings.value(PROC_DISPERSION_COMPENSATION_D3).toDouble());
	this->ui.groupBox_windowing->setChecked(this->processingSettings.value(PROC_WINDOWING).toBool());
	this->ui.comboBox_windowType->setCurrentIndex(this->processingSettings.value(PROC_WINDOWING_TYPE).toUInt());
	this->ui.doubleSpinBox_windowFillFactor->setValue(this->processingSettings.value(PROC_WINDOWING_FILL_FACTOR).toDouble());
	this->ui.doubleSpinBox_windowCenterPosition->setValue(this->processingSettings.value(PROC_WINDOWING_CENTER_POSITION).toDouble());
	this->ui.groupBox_fixedPatternNoiseRemoval->setChecked(this->processingSettings.value(PROC_FIXED_PATTERN_REMOVAL).toBool());
	this->ui.radioButton_continuously->setChecked(this->processingSettings.value(PROC_FIXED_PATTERN_REMOVAL_CONTINUOUSLY).toBool());
	this->ui.spinBox_bscansFixedNoise->setValue(this->processingSettings.value(PROC_FIXED_PATTERN_REMOVAL_BSCANS).toUInt());
	this->ui.checkBox_sinusoidalScanCorrection->setChecked(this->processingSettings.value(PROC_SINUSOIDAL_SCAN_CORRECTION).toBool());
	this->ui.groupBox_postProcessBackgroundRemoval->setChecked(this->processingSettings.value(PROC_POST_BACKGROUND_REMOVAL).toBool());
	this->ui.doubleSpinBox_postProcessBackgroundWeight->setValue(this->processingSettings.value(PROC_POST_BACKGROUND_WEIGHT).toDouble());
	this->ui.doubleSpinBox_postProcessBackgroundOffset->setValue(this->processingSettings.value(PROC_POST_BACKGROUND_OFFSET).toDouble());
	emit loadPostProcessBackgroundRequested(SETTINGS_PATH_BACKGROUND_FILE);
	
	//GPU to RAM Streaming
	this->ui.groupBox_streaming->setChecked(this->streamingSettings.value(STREAM_STREAMING).toBool());
	this->ui.spinBox_streamingBuffersToSkip->setValue(this->streamingSettings.value(STREAM_STREAMING_SKIP).toUInt());

	this->connectGuiElementsToAutosave();
}

void Sidebar::saveSettings() {
	this->updateSettingsMaps();
	Settings* settingsManager = Settings::getInstance();
	settingsManager->storeSettings(REC, this->recordSettings);
	settingsManager->storeSettings(PROC, this->processingSettings);
	settingsManager->storeSettings(STREAM, this->streamingSettings);
}

void Sidebar::connectGuiElementsToAutosave() {
	//Connects to store recording settings
	foreach(auto element,this->spinBoxes) {
		connect(element, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, &Sidebar::saveSettings);
	}
	foreach(auto element,this->doubleSpinBoxes) {
		connect(element, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, &Sidebar::saveSettings);
	}
	foreach(auto element,this->lineEdits) {
		connect(element, &QLineEdit::textChanged, this, &Sidebar::saveSettings);
	}
	foreach(auto element,this->checkBoxes) {
		connect(element, &QCheckBox::clicked, this, &Sidebar::saveSettings);
	}
	foreach(auto element,this->groupBoxes) {
		connect(element, &QGroupBox::clicked, this, &Sidebar::saveSettings);
	}
	connect(this->ui.plainTextEdit_description, &QPlainTextEdit::textChanged, this, &Sidebar::saveSettings);
}

void Sidebar::disconnectGuiElementsFromAutosave() {
	//Disconnects to store recording settings
	foreach(auto element,this->spinBoxes) {
		disconnect(element, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, &Sidebar::saveSettings);
	}
	foreach(auto element,this->doubleSpinBoxes) {
		disconnect(element, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, &Sidebar::saveSettings);
	}
	foreach(auto element,this->lineEdits) {
		disconnect(element, &QLineEdit::textChanged, this, &Sidebar::saveSettings);
	}
	foreach(auto element,this->checkBoxes) {
		disconnect(element, &QCheckBox::clicked, this, &Sidebar::saveSettings);
	}
	foreach(auto element,this->groupBoxes) {
		disconnect(element, &QGroupBox::clicked, this, &Sidebar::saveSettings);
	}
	disconnect(this->ui.plainTextEdit_description, &QPlainTextEdit::textChanged, this, &Sidebar::saveSettings);
}

void Sidebar::connectUpdateProcessingParams() {
	//Connect gui elements to updateProcessingParams slot to allow live update of oct algorithm parameters
	foreach(auto lineEdit, this->lineEdits){
		connect(lineEdit, &QLineEdit::textChanged, this, &Sidebar::slot_updateProcessingParams);
	}
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
	params->fixedPatternNoiseRemoval = this->ui.groupBox_fixedPatternNoiseRemoval->isChecked();
	params->continuousFixedPatternNoiseDetermination = this->ui.radioButton_continuously->isChecked();
	params->bscansForNoiseDetermination = this->ui.spinBox_bscansFixedNoise->value();
	params->sinusoidalScanCorrection = this->ui.checkBox_sinusoidalScanCorrection->isChecked();
	params->rollingAverageWindowSize = this->ui.spinBox_rollingAverageWindowSize->value();
	params->backgroundRemoval = this->ui.groupBox_backgroundremoval->isChecked();
	params->postProcessBackgroundRemoval = this->ui.groupBox_postProcessBackgroundRemoval->isChecked();
	params->postProcessBackgroundWeight = this->ui.doubleSpinBox_postProcessBackgroundWeight->value();
	params->postProcessBackgroundOffset = this->ui.doubleSpinBox_postProcessBackgroundOffset->value();
}

void Sidebar::updateStreamingParams() {
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	params->streamingParamsChanged = params->streamToHost == this->ui.groupBox_streaming->isChecked() ? false : true;
	params->streamToHost = this->ui.groupBox_streaming->isChecked();
	params->streamingBuffersToSkip = this->ui.spinBox_streamingBuffersToSkip->value();
}

void Sidebar::updateRecordingParams() {
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	params->updateBufferSizeInBytes();
	params->recParams.fileName = this->ui.lineEdit_recName->text();
	params->recParams.savePath = this->ui.lineEdit_saveFolder->text();
	params->recParams.stopAfterRecord = this->ui.checkBox_stopAfterRec->isChecked();
	params->recParams.buffersToRecord = this->ui.spinBox_volumes->value();
	params->recParams.startWithFirstBuffer = this->ui.checkBox_startWithFirstBuffer->isChecked();
	params->recParams.recordProcessed = this->ui.checkBox_recordProcessedBuffers->isChecked();
	params->recParams.recordRaw = this->ui.checkBox_recordRawBuffers->isChecked();
	params->recParams.recordScreenshot = this->ui.checkBox_recordScreenshots->isChecked();
	params->recParams.saveMetaData = this->ui.checkBox_meta->isChecked();
}

void Sidebar::enableRecordTab(bool enable) {
	this->ui.tab_3->setEnabled(enable);
}

void Sidebar::addActionsForKlinGroupBoxMenu(QList<QAction *> actions) {
	this->ui.groupBox_resampling->addActions(actions);
	this->actionUseSidebarKLinCurve = actions.at(0);
	this->actionUseCustomKLinCurve = actions.at(1);
	this->actionSetCustomKLinCurve = actions.at(3);
}

void Sidebar::updateResamplingParams() {
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	int interpolation = this->ui.comboBox_interpolation->currentIndex();
	OctAlgorithmParameters::INTERPOLATION interpolationOption = (OctAlgorithmParameters::INTERPOLATION) interpolation;
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
	OctAlgorithmParametersManager paramsManager;
	if(params->useCustomResampleCurve){
		paramsManager.saveCustomResamplingCurveToFile(SETTINGS_PATH_RESAMPLING_FILE);
	}
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

void Sidebar::updateBackgroundPlot() {
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	this->ui.widget_postProcessBackgroundPlot->plotCurves(params->postProcessBackground, nullptr, params->postProcessBackgroundLength);
	this->ui.widget_postProcessBackgroundPlot->saveCurveDataToFile(SETTINGS_PATH_BACKGROUND_FILE);
}

void Sidebar::slot_selectSaveDir() {
	emit dialogAboutToOpen();
	QCoreApplication::processEvents();
	QString savedPath = this->recordSettings.value(REC_PATH).toString();
	QString standardLocation = savedPath.size() == 0 ? QStandardPaths::writableLocation(QStandardPaths::DesktopLocation) : savedPath;
	QString selectedSaveDir = QFileDialog::getExistingDirectory(this, tr("Select OCTrpoZ Save Folder"), standardLocation, QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks );
	if (selectedSaveDir == "") {
		selectedSaveDir = standardLocation;
	}
	this->ui.lineEdit_saveFolder->setText(selectedSaveDir);
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
	this->updateRecordingParams();
	params->acquisitionParamsChanged = false;
}

void Sidebar::slot_recordPostProcessingBackground() {
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	params->postProcessBackgroundRecordingRequested = true;
}

void Sidebar::slot_savePostProcessingBackground() {
	emit dialogAboutToOpen();
	QCoreApplication::processEvents();
	QString fileName = "";
	QString filters("CSV (*.csv)");
	QString defaultFilter("CSV (*.csv)");
	QString savedPath = this->recordSettings.value(REC_PATH).toString();
	QString standardLocation = savedPath.size() == 0 ? QStandardPaths::writableLocation(QStandardPaths::DesktopLocation) : savedPath;
	fileName = QFileDialog::getSaveFileName(this, tr("Save background data"), QDir::currentPath(), filters, &defaultFilter);
	emit savePostProcessBackgroundRequested(fileName);
	emit dialogClosed();
}

void Sidebar::slot_loadPostProcessingBackground() {
	emit dialogAboutToOpen();
	QCoreApplication::processEvents();
	QString fileName = "";
	QString filters("CSV (*.csv)");
	QString defaultFilter("CSV (*.csv)");
	QString savedPath = this->recordSettings.value(REC_PATH).toString();
	QString standardLocation = savedPath.size() == 0 ? QStandardPaths::writableLocation(QStandardPaths::DesktopLocation) : savedPath;
	fileName = QFileDialog::getOpenFileName(this, tr("Load background data"), QDir::currentPath(), filters, &defaultFilter);
	emit loadPostProcessBackgroundRequested(fileName);
	emit dialogClosed();
}

void Sidebar::slot_redetermineFixedPatternNoise() {
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	params->redetermineFixedPatternNoise = true;
}

void Sidebar::slot_disableRedetermineButtion(bool disable){
	this->ui.pushButton_redetermine->setDisabled(disable);
}

void Sidebar::slot_setMaximumRollingAverageWindowSize(unsigned int max) {
	this->ui.spinBox_rollingAverageWindowSize->setMaximum(max);
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
		this->ui.doubleSpinBox_d2->setValue(*d2);
	}
	if(d3 != nullptr){
		this->ui.doubleSpinBox_d3->setValue(*d3);
	}
	QApplication::processEvents();
	emit dispCompCoeffs(this->ui.doubleSpinBox_d0->value(), this->ui.doubleSpinBox_d1->value(), this->ui.doubleSpinBox_d2->value(), this->ui.doubleSpinBox_d3->value());
}

void Sidebar::slot_setGrayscaleConversion(bool enableLogScaling, double max, double min, double multiplicator, double offset) {
	this->ui.checkBox_logScaling->setChecked(enableLogScaling);

	if(!qIsNaN(max)) {
		this->ui.doubleSpinBox_signalMax->setValue(max);
	}
	if(!qIsNaN(min)) {
		this->ui.doubleSpinBox_signalMin->setValue(min);
	}
	if(!qIsNaN(multiplicator)) {
		this->ui.doubleSpinBox_signalMultiplicator->setValue(multiplicator);
	}
	if(!qIsNaN(offset)) {
		this->ui.doubleSpinBox_signalAddend->setValue(offset);
	}
}

void Sidebar::disableKlinCoeffInput(bool disable) {
	bool groupBoxState = ui.groupBox_resampling->isChecked();
	this->ui.groupBox_resampling->setChecked(!groupBoxState);
	this->ui.doubleSpinBox_c0->setDisabled(disable);
	this->ui.doubleSpinBox_c1->setDisabled(disable);
	this->ui.doubleSpinBox_c2->setDisabled(disable);
	this->ui.doubleSpinBox_c3->setDisabled(disable);
	this->ui.groupBox_resampling->setChecked(groupBoxState);
}

void Sidebar::copyInfoToClipboard() {
	QClipboard *clipboard = QApplication::clipboard();
	QString infoText = this->ui.label_name_volumesPerSecond->text() + "\t" + this->ui.label_volumesPerSecond->text() + "\n"
		+ this->ui.label_name_buffersPerSecond->text() + "\t" + this->ui.label_buffersPerSecond->text() + "\n"
		+ this->ui.label_name_bscansPerSecond->text() + "\t" + this->ui.label_bscansPerSecond->text() + "\n"
		+ this->ui.label_name_ascansPerSecond->text() + "\t" + this->ui.label_ascansPerSecond->text() + "\n"
		+ this->ui.label_name_bufferSize->text() + "\t" + this->ui.label_bufferSize->text() + "\n"
		+ this->ui.label_name_dataThroughput->text() + "\t" + this->ui.label_dataThroughput->text() + "\n";
	clipboard->setText(infoText);
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

void Sidebar::updateSettingsMaps() {
	//Recording
	this->recordSettings.insert(REC_PATH, this->ui.lineEdit_saveFolder->text());
	this->recordSettings.insert(REC_SCREENSHOTS, this->ui.checkBox_recordScreenshots->isChecked());
	this->recordSettings.insert(REC_RAW, this->ui.checkBox_recordRawBuffers->isChecked());
	this->recordSettings.insert(REC_PROCESSED, this->ui.checkBox_recordProcessedBuffers->isChecked());
	this->recordSettings.insert(REC_START_WITH_FIRST_BUFFER, this->ui.checkBox_startWithFirstBuffer->isChecked());
	this->recordSettings.insert(REC_STOP, this->ui.checkBox_stopAfterRec->isChecked());
	this->recordSettings.insert(REC_META, this->ui.checkBox_meta->isChecked());
	this->recordSettings.insert(REC_VOLUMES, this->ui.spinBox_volumes->value());
	this->recordSettings.insert(REC_NAME, this->ui.lineEdit_recName->text());
	this->recordSettings.insert(REC_DESCRIPTION, this->ui.plainTextEdit_description->toPlainText());

	//Processing
	this->processingSettings.insert(PROC_BITSHIFT, this->ui.checkBox_bitshift->isChecked());
	this->processingSettings.insert(PROC_FLIP_BSCANS, this->ui.checkBox_bscanFlip->isChecked());
	this->processingSettings.insert(PROC_REMOVEBACKGROUND, this->ui.groupBox_backgroundremoval->isChecked());
	this->processingSettings.insert(PROC_REMOVEBACKGROUND_WINDOW_SIZE, this->ui.spinBox_rollingAverageWindowSize->value());
	this->processingSettings.insert(PROC_LOG, this->ui.checkBox_logScaling->isChecked());
	this->processingSettings.insert(PROC_MAX, this->ui.doubleSpinBox_signalMax->value());
	this->processingSettings.insert(PROC_MIN, this->ui.doubleSpinBox_signalMin->value());
	this->processingSettings.insert(PROC_COEFF, this->ui.doubleSpinBox_signalMultiplicator->value());
	this->processingSettings.insert(PROC_ADDEND, this->ui.doubleSpinBox_signalAddend->value());
	this->processingSettings.insert(PROC_RESAMPLING, this->ui.groupBox_resampling->isChecked());
	this->processingSettings.insert(PROC_RESAMPLING_INTERPOLATION, this->ui.comboBox_interpolation->currentIndex());
	this->processingSettings.insert(PROC_RESAMPLING_C0, this->ui.doubleSpinBox_c0->value());
	this->processingSettings.insert(PROC_RESAMPLING_C1, this->ui.doubleSpinBox_c1->value());
	this->processingSettings.insert(PROC_RESAMPLING_C2, this->ui.doubleSpinBox_c2->value());
	this->processingSettings.insert(PROC_RESAMPLING_C3, this->ui.doubleSpinBox_c3->value());
	this->processingSettings.insert(PROC_CUSTOM_RESAMPLING, this->actionUseCustomKLinCurve->isChecked());
	this->processingSettings.insert(PROC_CUSTOM_RESAMPLING_FILEPATH, OctAlgorithmParametersManager().getParams()->customResampleCurveFilePath);
	this->processingSettings.insert(PROC_DISPERSION_COMPENSATION, this->ui.groupBox_dispersionCompensation->isChecked());
	this->processingSettings.insert(PROC_DISPERSION_COMPENSATION_D0, this->ui.doubleSpinBox_d0->value());
	this->processingSettings.insert(PROC_DISPERSION_COMPENSATION_D1, this->ui.doubleSpinBox_d1->value());
	this->processingSettings.insert(PROC_DISPERSION_COMPENSATION_D2, this->ui.doubleSpinBox_d2->value());
	this->processingSettings.insert(PROC_DISPERSION_COMPENSATION_D3, this->ui.doubleSpinBox_d3->value());
	this->processingSettings.insert(PROC_WINDOWING, this->ui.groupBox_windowing->isChecked());
	this->processingSettings.insert(PROC_WINDOWING_TYPE, this->ui.comboBox_windowType->currentIndex());
	this->processingSettings.insert(PROC_WINDOWING_FILL_FACTOR, this->ui.doubleSpinBox_windowFillFactor->value());
	this->processingSettings.insert(PROC_WINDOWING_CENTER_POSITION, this->ui.doubleSpinBox_windowCenterPosition->value());
	this->processingSettings.insert(PROC_FIXED_PATTERN_REMOVAL, this->ui.groupBox_fixedPatternNoiseRemoval->isChecked());
	this->processingSettings.insert(PROC_FIXED_PATTERN_REMOVAL_CONTINUOUSLY, this->ui.radioButton_continuously->isChecked());
	this->processingSettings.insert(PROC_FIXED_PATTERN_REMOVAL_BSCANS, this->ui.spinBox_bscansFixedNoise->value());
	this->processingSettings.insert(PROC_SINUSOIDAL_SCAN_CORRECTION, this->ui.checkBox_sinusoidalScanCorrection->isChecked());
	this->processingSettings.insert(PROC_POST_BACKGROUND_REMOVAL, this->ui.groupBox_postProcessBackgroundRemoval->isChecked());
	this->processingSettings.insert(PROC_POST_BACKGROUND_WEIGHT, this->ui.doubleSpinBox_postProcessBackgroundWeight->value());
	this->processingSettings.insert(PROC_POST_BACKGROUND_OFFSET, this->ui.doubleSpinBox_postProcessBackgroundOffset->value());

	//GPU to RAM Streaming
	this->streamingSettings.insert(STREAM_STREAMING, this->ui.groupBox_streaming->isChecked());
	this->streamingSettings.insert(STREAM_STREAMING_SKIP, this->ui.spinBox_streamingBuffersToSkip->value());
}
