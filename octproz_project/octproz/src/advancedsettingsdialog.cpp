#include "advancedsettingsdialog.h"
#include "ui_advancedsettingsdialog.h"
#include "octalgorithmparameters.h"
#include "settingsfilemanager.h"
#include <QSettings>
#include <QMessageBox>
#include <QFileDialog>

AdvancedSettingsDialog::AdvancedSettingsDialog(QWidget *parent) :
	QDialog(parent),
	ui(new Ui::AdvancedSettingsDialog),
	recordingStatusTimer(new QTimer(this))
{
	ui->setupUi(this);
	loadSettings();

	// Connect timer for polling recording status
	connect(recordingStatusTimer, &QTimer::timeout, this, &AdvancedSettingsDialog::checkRecordingStatus);
}

AdvancedSettingsDialog::~AdvancedSettingsDialog()
{
	if (recordingStatusTimer->isActive()) {
		recordingStatusTimer->stop();
	}
	delete ui;
}

void AdvancedSettingsDialog::connectSignals(){
	connect(ui->checkBox_fullRangeMode, &QCheckBox::toggled,
			this, &AdvancedSettingsDialog::applyFullRangeModeSettings);

	// CC Artifact Removal signals
	connect(ui->groupBox_ccArtifactRemoval, &QGroupBox::toggled,
			this, &AdvancedSettingsDialog::applyCCSettings);
	connect(ui->doubleSpinBox_ccRectCenter, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
			this, &AdvancedSettingsDialog::applyCCSettings);
	connect(ui->doubleSpinBox_ccRectWidth, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
			this, &AdvancedSettingsDialog::applyCCSettings);
	connect(ui->checkBox_ccKeepPositive, &QCheckBox::toggled,
			this, &AdvancedSettingsDialog::applyCCSettings);

	// Background Frame Subtraction signals
	connect(ui->checkBox_bgFrameEnabled, &QCheckBox::toggled,
			this, &AdvancedSettingsDialog::applyBackgroundFrameSettings);
	connect(ui->spinBox_bscansToAverage, QOverload<int>::of(&QSpinBox::valueChanged),
			this, &AdvancedSettingsDialog::applyBackgroundFrameSettings);
	connect(ui->pushButton_recordBackground, &QPushButton::clicked,
			this, &AdvancedSettingsDialog::recordBackgroundFrame);
	connect(ui->pushButton_saveBackground, &QPushButton::clicked,
			this, &AdvancedSettingsDialog::saveBackgroundFrame);
	connect(ui->pushButton_loadBackground, &QPushButton::clicked,
			this, &AdvancedSettingsDialog::loadBackgroundFrame);
}

void AdvancedSettingsDialog::disconnectSignals(){
	disconnect(ui->checkBox_fullRangeMode, &QCheckBox::toggled,
			   this, &AdvancedSettingsDialog::applyFullRangeModeSettings);

	// CC Artifact Removal signals
	disconnect(ui->groupBox_ccArtifactRemoval, &QGroupBox::toggled,
			   this, &AdvancedSettingsDialog::applyCCSettings);
	disconnect(ui->doubleSpinBox_ccRectCenter, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
			   this, &AdvancedSettingsDialog::applyCCSettings);
	disconnect(ui->doubleSpinBox_ccRectWidth, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
			   this, &AdvancedSettingsDialog::applyCCSettings);
	disconnect(ui->checkBox_ccKeepPositive, &QCheckBox::toggled,
			   this, &AdvancedSettingsDialog::applyCCSettings);

	// Background Frame Subtraction signals
	disconnect(ui->checkBox_bgFrameEnabled, &QCheckBox::toggled,
			   this, &AdvancedSettingsDialog::applyBackgroundFrameSettings);
	disconnect(ui->spinBox_bscansToAverage, QOverload<int>::of(&QSpinBox::valueChanged),
			   this, &AdvancedSettingsDialog::applyBackgroundFrameSettings);
	disconnect(ui->pushButton_recordBackground, &QPushButton::clicked,
			   this, &AdvancedSettingsDialog::recordBackgroundFrame);
	disconnect(ui->pushButton_saveBackground, &QPushButton::clicked,
			   this, &AdvancedSettingsDialog::saveBackgroundFrame);
	disconnect(ui->pushButton_loadBackground, &QPushButton::clicked,
			   this, &AdvancedSettingsDialog::loadBackgroundFrame);
}

void AdvancedSettingsDialog::applyFullRangeModeSettings(bool enable){
	saveSettings();

	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	bool wasFullRange = params->fullRangeMode;
	params->fullRangeMode = enable;

	if (wasFullRange != enable) {
		params->fullRangeModeChanged = true;
	}

	// Enable/disable CC artifact removal group based on full range mode
	ui->groupBox_ccArtifactRemoval->setEnabled(enable);

	emit settingsChanged();
}

void AdvancedSettingsDialog::loadSettings(){
	disconnectSignals();

	QSettings settings(SETTINGS_PATH, QSettings::IniFormat);

	// Full Range Mode
	ui->checkBox_fullRangeMode->setChecked(
		settings.value(ADV_FULL_RANGE_MODE, false).toBool());

	// CC Artifact Removal
	ui->groupBox_ccArtifactRemoval->setChecked(
		settings.value(ADV_CC_ARTIFACT_REMOVAL, false).toBool());
	ui->doubleSpinBox_ccRectCenter->setValue(
		settings.value(ADV_CC_RECT_CENTER, 0.25).toDouble());
	ui->doubleSpinBox_ccRectWidth->setValue(
		settings.value(ADV_CC_RECT_WIDTH, 0.5).toDouble());
	ui->checkBox_ccKeepPositive->setChecked(
		settings.value(ADV_CC_KEEP_POSITIVE, true).toBool());

	// Background Frame Subtraction
	ui->checkBox_bgFrameEnabled->setChecked(
		settings.value(ADV_BG_FRAME_ENABLED, false).toBool());
	ui->spinBox_bscansToAverage->setValue(
		settings.value(ADV_BG_FRAME_BSCANS_TO_AVG, 10).toInt());

	// Sync with OctAlgorithmParameters on load
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	params->fullRangeMode = ui->checkBox_fullRangeMode->isChecked();
	params->ccArtifactRemoval = ui->groupBox_ccArtifactRemoval->isChecked();
	params->ccRectCenterFreq = ui->doubleSpinBox_ccRectCenter->value();
	params->ccRectWidth = ui->doubleSpinBox_ccRectWidth->value();
	params->ccKeepPositiveSideband = ui->checkBox_ccKeepPositive->isChecked();

	// Background Frame parameters
	params->backgroundFrameSubtraction = ui->checkBox_bgFrameEnabled->isChecked();
	params->backgroundFrameBscansToAverage = ui->spinBox_bscansToAverage->value();
	params->backgroundFrameFilePath = settings.value(ADV_BG_FRAME_FILE_PATH, "").toString();

	// Load background frame from file if path exists
	if (!params->backgroundFrameFilePath.isEmpty()) {
		params->loadBackgroundFrameFromFile(params->backgroundFrameFilePath);
	}

	// Update CC group enabled state
	ui->groupBox_ccArtifactRemoval->setEnabled(params->fullRangeMode);

	// Update background frame status indicator
	updateBackgroundFrameStatus();

	connectSignals();
}

void AdvancedSettingsDialog::saveSettings() {
	QSettings settings(SETTINGS_PATH, QSettings::IniFormat);
	settings.setValue(ADV_FULL_RANGE_MODE, ui->checkBox_fullRangeMode->isChecked());
	settings.setValue(ADV_CC_ARTIFACT_REMOVAL, ui->groupBox_ccArtifactRemoval->isChecked());
	settings.setValue(ADV_CC_RECT_CENTER, ui->doubleSpinBox_ccRectCenter->value());
	settings.setValue(ADV_CC_RECT_WIDTH, ui->doubleSpinBox_ccRectWidth->value());
	settings.setValue(ADV_CC_KEEP_POSITIVE, ui->checkBox_ccKeepPositive->isChecked());

	// Background Frame Subtraction
	settings.setValue(ADV_BG_FRAME_ENABLED, ui->checkBox_bgFrameEnabled->isChecked());
	settings.setValue(ADV_BG_FRAME_BSCANS_TO_AVG, ui->spinBox_bscansToAverage->value());
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	settings.setValue(ADV_BG_FRAME_FILE_PATH, params->backgroundFrameFilePath);
}

void AdvancedSettingsDialog::applyCCSettings(){
	saveSettings();

	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	params->ccArtifactRemoval = ui->groupBox_ccArtifactRemoval->isChecked();
	params->ccRectCenterFreq = ui->doubleSpinBox_ccRectCenter->value();
	params->ccRectWidth = ui->doubleSpinBox_ccRectWidth->value();
	params->ccKeepPositiveSideband = ui->checkBox_ccKeepPositive->isChecked();

	emit settingsChanged();
}

void AdvancedSettingsDialog::applyBackgroundFrameSettings(){
	saveSettings();

	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	params->backgroundFrameSubtraction = ui->checkBox_bgFrameEnabled->isChecked();
	params->backgroundFrameBscansToAverage = ui->spinBox_bscansToAverage->value();

	updateBackgroundFrameStatus();
	emit settingsChanged();
}

void AdvancedSettingsDialog::recordBackgroundFrame(){
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();

	// Set recording request flag - the CUDA pipeline will handle the actual recording
	params->backgroundFrameRecordingRequested = true;
	params->backgroundFrameBscansRecorded = 0;

	// Disable record button and show recording status
	ui->pushButton_recordBackground->setEnabled(false);
	ui->pushButton_recordBackground->setText("Recording...");
	ui->label_bgStatusIndicator->setText("Recording...");
	ui->label_bgStatusIndicator->setStyleSheet("color: blue;");

	// Start polling for recording completion
	recordingStatusTimer->start(100); // Poll every 100ms
}

void AdvancedSettingsDialog::saveBackgroundFrame(){
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();

	QString filePath = QFileDialog::getSaveFileName(this,
		tr("Save Background Frame"), params->backgroundFrameFilePath, tr("Raw Files (*.raw)"));

	if (!filePath.isEmpty()) {
		if (params->saveBackgroundFrameToFile(filePath)) {
			saveSettings(); // Save the file path
			QMessageBox::information(this, tr("Success"), tr("Background frame saved successfully."));
		} else {
			QMessageBox::warning(this, tr("Error"), tr("Failed to save background frame."));
		}
	}
}

void AdvancedSettingsDialog::loadBackgroundFrame(){
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();

	QString filePath = QFileDialog::getOpenFileName(this,
		tr("Load Background Frame"), params->backgroundFrameFilePath, tr("Raw Files (*.raw)"));

	if (!filePath.isEmpty()) {
		if (params->loadBackgroundFrameFromFile(filePath)) {
			saveSettings(); // Save the file path
			updateBackgroundFrameStatus();
			QMessageBox::information(this, tr("Success"), tr("Background frame loaded successfully."));
		} else {
			QMessageBox::warning(this, tr("Error"), tr("Failed to load background frame. Invalid file format."));
		}
	}
}

void AdvancedSettingsDialog::checkRecordingStatus(){
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();

	// Update progress display
	if (params->backgroundFrameRecordingInProgress) {
		int progress = (params->backgroundFrameBscansRecorded * 100) / params->backgroundFrameBscansToAverage;
		ui->label_bgStatusIndicator->setText(QString("Recording... %1%").arg(progress));
	}

	// Check if recording is complete
	if (!params->backgroundFrameRecordingInProgress && !params->backgroundFrameRecordingRequested) {
		recordingStatusTimer->stop();

		// Re-enable record button
		ui->pushButton_recordBackground->setEnabled(true);
		ui->pushButton_recordBackground->setText("Record Background");

		// Update status indicator
		updateBackgroundFrameStatus();
	}
}

void AdvancedSettingsDialog::updateBackgroundFrameStatus(){
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();

	if (params->backgroundFrameValid) {
		ui->label_bgStatusIndicator->setText(QString("Valid (%1x%2)")
			.arg(params->backgroundFrameSamplesPerLine)
			.arg(params->backgroundFrameAscansPerBscan));
		ui->label_bgStatusIndicator->setStyleSheet("color: green; font-weight: bold;");
		ui->pushButton_saveBackground->setEnabled(true);
	} else if (params->backgroundFrame != nullptr) {
		// Frame loaded but not yet validated (will validate when processing starts)
		ui->label_bgStatusIndicator->setText(QString("Loaded (%1x%2)")
			.arg(params->backgroundFrameSamplesPerLine)
			.arg(params->backgroundFrameAscansPerBscan));
		ui->label_bgStatusIndicator->setStyleSheet("color: blue;");
		ui->pushButton_saveBackground->setEnabled(true);
	} else {
		ui->label_bgStatusIndicator->setText("No background loaded");
		ui->label_bgStatusIndicator->setStyleSheet("color: gray;");
		ui->pushButton_saveBackground->setEnabled(false);
	}
}
