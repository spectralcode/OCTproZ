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
	connect(ui->radioButton_bgSubtractionOnly, &QRadioButton::toggled,
			this, &AdvancedSettingsDialog::applyBackgroundFrameSettings);
	connect(ui->radioButton_bgSubtractionAndNormalization, &QRadioButton::toggled,
			this, &AdvancedSettingsDialog::applyBackgroundFrameSettings);
	connect(ui->spinBox_bscansToAverage, QOverload<int>::of(&QSpinBox::valueChanged),
			this, &AdvancedSettingsDialog::applyBackgroundFrameSettings);
	connect(ui->checkBox_bgAverageSpectra, &QCheckBox::toggled,
			this, &AdvancedSettingsDialog::applyBackgroundFrameSettings);
	connect(ui->checkBox_bgSmoothSpectra, &QCheckBox::toggled,
			this, &AdvancedSettingsDialog::applyBackgroundFrameSettings);
	connect(ui->spinBox_bgSmoothingWindow, QOverload<int>::of(&QSpinBox::valueChanged),
			this, &AdvancedSettingsDialog::applyBackgroundFrameSettings);
	connect(ui->pushButton_recordBackground, &QPushButton::clicked,
			this, &AdvancedSettingsDialog::recordBackgroundFrame);
	connect(ui->pushButton_saveBackground, &QPushButton::clicked,
			this, &AdvancedSettingsDialog::saveBackgroundFrame);
	connect(ui->pushButton_loadBackground, &QPushButton::clicked,
			this, &AdvancedSettingsDialog::loadBackgroundFrame);
	connect(ui->checkBox_continuousBackground, &QCheckBox::toggled,
			this, &AdvancedSettingsDialog::applyContinuousBackgroundSettings);
	connect(ui->comboBox_avgMethod, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &AdvancedSettingsDialog::applyContinuousBackgroundSettings);

	// Frame Correction signals
	connect(ui->checkBox_fcNormalizeByAvgSpectra, &QCheckBox::toggled,
			this, &AdvancedSettingsDialog::applyFrameCorrectionSettings);
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
	disconnect(ui->radioButton_bgSubtractionOnly, &QRadioButton::toggled,
			   this, &AdvancedSettingsDialog::applyBackgroundFrameSettings);
	disconnect(ui->radioButton_bgSubtractionAndNormalization, &QRadioButton::toggled,
			   this, &AdvancedSettingsDialog::applyBackgroundFrameSettings);
	disconnect(ui->spinBox_bscansToAverage, QOverload<int>::of(&QSpinBox::valueChanged),
			   this, &AdvancedSettingsDialog::applyBackgroundFrameSettings);
	disconnect(ui->checkBox_bgAverageSpectra, &QCheckBox::toggled,
			   this, &AdvancedSettingsDialog::applyBackgroundFrameSettings);
	disconnect(ui->checkBox_bgSmoothSpectra, &QCheckBox::toggled,
			   this, &AdvancedSettingsDialog::applyBackgroundFrameSettings);
	disconnect(ui->spinBox_bgSmoothingWindow, QOverload<int>::of(&QSpinBox::valueChanged),
			   this, &AdvancedSettingsDialog::applyBackgroundFrameSettings);
	disconnect(ui->pushButton_recordBackground, &QPushButton::clicked,
			   this, &AdvancedSettingsDialog::recordBackgroundFrame);
	disconnect(ui->pushButton_saveBackground, &QPushButton::clicked,
			   this, &AdvancedSettingsDialog::saveBackgroundFrame);
	disconnect(ui->pushButton_loadBackground, &QPushButton::clicked,
			   this, &AdvancedSettingsDialog::loadBackgroundFrame);
	disconnect(ui->checkBox_continuousBackground, &QCheckBox::toggled,
			   this, &AdvancedSettingsDialog::applyContinuousBackgroundSettings);
	disconnect(ui->comboBox_avgMethod, QOverload<int>::of(&QComboBox::currentIndexChanged),
			   this, &AdvancedSettingsDialog::applyContinuousBackgroundSettings);

	// Frame Correction signals
	disconnect(ui->checkBox_fcNormalizeByAvgSpectra, &QCheckBox::toggled,
			   this, &AdvancedSettingsDialog::applyFrameCorrectionSettings);
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
	int correctionMode = settings.value(ADV_BG_FRAME_CORRECTION_MODE,
		OctAlgorithmParameters::BACKGROUND_FRAME_SUBTRACTION_ONLY).toInt();
	if (correctionMode == OctAlgorithmParameters::BACKGROUND_FRAME_SUBTRACTION_AND_NORMALIZATION) {
		ui->radioButton_bgSubtractionAndNormalization->setChecked(true);
	} else {
		ui->radioButton_bgSubtractionOnly->setChecked(true);
	}
	ui->spinBox_bscansToAverage->setValue(
		settings.value(ADV_BG_FRAME_BSCANS_TO_AVG, 10).toInt());
	ui->checkBox_continuousBackground->setChecked(
		settings.value(ADV_BG_FRAME_CONTINUOUS, false).toBool());
	ui->checkBox_bgAverageSpectra->setChecked(
		settings.value(ADV_BG_FRAME_AVERAGE_SPECTRA, false).toBool());
	ui->checkBox_bgSmoothSpectra->setChecked(
		settings.value(ADV_BG_FRAME_SMOOTH_SPECTRA, false).toBool());
	ui->spinBox_bgSmoothingWindow->setValue(
		settings.value(ADV_BG_FRAME_SMOOTHING_WINDOW, 10).toInt());
	ui->comboBox_avgMethod->setCurrentIndex(
		settings.value(ADV_BG_FRAME_USE_EMA, true).toBool() ? 0 : 1);

	// Frame Correction
	ui->checkBox_fcNormalizeByAvgSpectra->setChecked(
		settings.value(ADV_FRAME_CORRECTION_NORMALIZE_AVG_SPECTRA, false).toBool());

	// Sync with OctAlgorithmParameters on load
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	params->fullRangeMode = ui->checkBox_fullRangeMode->isChecked();
	params->ccArtifactRemoval = ui->groupBox_ccArtifactRemoval->isChecked();
	params->ccRectCenterFreq = ui->doubleSpinBox_ccRectCenter->value();
	params->ccRectWidth = ui->doubleSpinBox_ccRectWidth->value();
	params->ccKeepPositiveSideband = ui->checkBox_ccKeepPositive->isChecked();

	// Background Frame parameters
	params->backgroundFrameSubtraction = ui->checkBox_bgFrameEnabled->isChecked();
	params->backgroundFrameCorrectionMode = ui->radioButton_bgSubtractionAndNormalization->isChecked()
		? OctAlgorithmParameters::BACKGROUND_FRAME_SUBTRACTION_AND_NORMALIZATION
		: OctAlgorithmParameters::BACKGROUND_FRAME_SUBTRACTION_ONLY;
	params->backgroundFrameBscansToAverage = ui->spinBox_bscansToAverage->value();
	params->backgroundFrameFilePath = settings.value(ADV_BG_FRAME_FILE_PATH, "").toString();
	params->continuousBackgroundUpdate = ui->checkBox_continuousBackground->isChecked();
	params->continuousBackgroundUseEMA = (ui->comboBox_avgMethod->currentIndex() == 0);
	params->backgroundFrameAverageSpectra = ui->checkBox_bgAverageSpectra->isChecked();
	params->backgroundFrameSmoothSpectra = ui->checkBox_bgSmoothSpectra->isChecked();
	params->backgroundFrameSmoothingWindowSize = ui->spinBox_bgSmoothingWindow->value();
	params->frameCorrectionNormalizeByAvgSpectra = ui->checkBox_fcNormalizeByAvgSpectra->isChecked();

	// Load background frame from file if path exists
	if (!params->backgroundFrameFilePath.isEmpty()) {
		params->loadBackgroundFrameFromFile(params->backgroundFrameFilePath);
	}

	// Update CC group enabled state
	ui->groupBox_ccArtifactRemoval->setEnabled(params->fullRangeMode);

	// Update background frame status indicator
	updateBackgroundFrameStatus();
	updateBackgroundCorrectionModeControls();

	// Enable/disable controls based on continuous mode
	bool continuous = params->continuousBackgroundUpdate;
	ui->comboBox_avgMethod->setEnabled(continuous);
	ui->label_avgMethod->setEnabled(continuous);
	ui->pushButton_recordBackground->setEnabled(!continuous);
	ui->pushButton_saveBackground->setEnabled(!continuous && params->backgroundFrame != nullptr);
	ui->pushButton_loadBackground->setEnabled(!continuous);
	ui->label_bgStatus->setEnabled(!continuous);
	ui->label_bgStatusIndicator->setEnabled(!continuous);
	ui->label_bgFileInUse->setEnabled(!continuous);
	ui->lineEdit_bgFilePath->setEnabled(!continuous);

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
	settings.setValue(ADV_BG_FRAME_CORRECTION_MODE,
		ui->radioButton_bgSubtractionAndNormalization->isChecked()
			? OctAlgorithmParameters::BACKGROUND_FRAME_SUBTRACTION_AND_NORMALIZATION
			: OctAlgorithmParameters::BACKGROUND_FRAME_SUBTRACTION_ONLY);
	settings.setValue(ADV_BG_FRAME_BSCANS_TO_AVG, ui->spinBox_bscansToAverage->value());
	settings.setValue(ADV_BG_FRAME_CONTINUOUS, ui->checkBox_continuousBackground->isChecked());
	settings.setValue(ADV_BG_FRAME_USE_EMA, ui->comboBox_avgMethod->currentIndex() == 0);
	settings.setValue(ADV_BG_FRAME_AVERAGE_SPECTRA, ui->checkBox_bgAverageSpectra->isChecked());
	settings.setValue(ADV_BG_FRAME_SMOOTH_SPECTRA, ui->checkBox_bgSmoothSpectra->isChecked());
	settings.setValue(ADV_BG_FRAME_SMOOTHING_WINDOW, ui->spinBox_bgSmoothingWindow->value());

	// Frame Correction
	settings.setValue(ADV_FRAME_CORRECTION_NORMALIZE_AVG_SPECTRA, ui->checkBox_fcNormalizeByAvgSpectra->isChecked());
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
	// "Average spectra" and "Smooth spectra" are mutually exclusive
	if (ui->checkBox_bgAverageSpectra->isChecked() && ui->checkBox_bgSmoothSpectra->isChecked()) {
		QCheckBox* toUncheck = (sender() == ui->checkBox_bgSmoothSpectra)
			? ui->checkBox_bgAverageSpectra
			: ui->checkBox_bgSmoothSpectra;
		QSignalBlocker blocker(toUncheck);
		toUncheck->setChecked(false);
	}

	saveSettings();

	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	params->backgroundFrameSubtraction = ui->checkBox_bgFrameEnabled->isChecked();
	params->backgroundFrameCorrectionMode = ui->radioButton_bgSubtractionAndNormalization->isChecked()
		? OctAlgorithmParameters::BACKGROUND_FRAME_SUBTRACTION_AND_NORMALIZATION
		: OctAlgorithmParameters::BACKGROUND_FRAME_SUBTRACTION_ONLY;
	params->backgroundFrameBscansToAverage = ui->spinBox_bscansToAverage->value();
	params->backgroundFrameAverageSpectra = ui->checkBox_bgAverageSpectra->isChecked();
	params->backgroundFrameSmoothSpectra = ui->checkBox_bgSmoothSpectra->isChecked();
	params->backgroundFrameSmoothingWindowSize = ui->spinBox_bgSmoothingWindow->value();

	updateBackgroundFrameStatus();
	updateBackgroundCorrectionModeControls();
	emit settingsChanged();
}

void AdvancedSettingsDialog::recordBackgroundFrame(){
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();

	// Set recording request flag - the CUDA pipeline will handle the actual recording
	params->backgroundFrameRecordingRequested = true;
	params->backgroundFrameBscansRecorded = 0;
	params->backgroundFrameFilePath.clear(); // Clear old path - this is a new recording

	// Disable record button and show recording status
	ui->pushButton_recordBackground->setEnabled(false);
	ui->pushButton_recordBackground->setText("Recording...");
	ui->label_bgStatusIndicator->setText("Recording...");

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
			updateBackgroundFrameStatus(); // Update lineEdit to show new file path
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
			params->updateBackgroundFrameValidity(); // Check if dimensions match current acquisition
			updateBackgroundFrameStatus();
			if (params->backgroundFrameValid) {
				QMessageBox::information(this, tr("Success"), tr("Background frame loaded successfully."));
			} else {
				QMessageBox::warning(this, tr("Dimension Mismatch"),
					tr("Background frame loaded but dimensions (%1x%2) don't match current acquisition (%3x%4). "
					   "The background will not be applied until dimensions match.")
					.arg(params->backgroundFrameSamplesPerLine)
					.arg(params->backgroundFrameAscansPerBscan)
					.arg(params->samplesPerLine)
					.arg(params->ascansPerBscan));
			}
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

		// Update status indicator and save settings (clears old file path from settings)
		updateBackgroundFrameStatus();
		saveSettings();
	}
}

void AdvancedSettingsDialog::updateBackgroundFrameStatus(){
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();

	if (params->backgroundFrameValid) {
		ui->label_bgStatusIndicator->setText(QString("Valid (%1x%2)")
			.arg(params->backgroundFrameSamplesPerLine)
			.arg(params->backgroundFrameAscansPerBscan));
		ui->pushButton_saveBackground->setEnabled(true);
	} else if (params->backgroundFrame != nullptr) {
		// Check if we have acquisition settings to compare against
		bool hasAcquisitionSettings = (params->samplesPerLine > 0 && params->ascansPerBscan > 0);
		bool dimensionsMismatch = hasAcquisitionSettings &&
			(params->backgroundFrameSamplesPerLine != params->samplesPerLine ||
			 params->backgroundFrameAscansPerBscan != params->ascansPerBscan);

		if (dimensionsMismatch) {
			// Dimensions don't match current acquisition
			ui->label_bgStatusIndicator->setText(QString("Mismatch (%1x%2)")
				.arg(params->backgroundFrameSamplesPerLine)
				.arg(params->backgroundFrameAscansPerBscan));
		} else {
			// Frame loaded but not yet validated (no acquisition running yet)
			ui->label_bgStatusIndicator->setText(QString("Loaded (%1x%2)")
				.arg(params->backgroundFrameSamplesPerLine)
				.arg(params->backgroundFrameAscansPerBscan));
		}
		ui->pushButton_saveBackground->setEnabled(true);
	} else {
		ui->label_bgStatusIndicator->setText("No background loaded");
		ui->pushButton_saveBackground->setEnabled(false);
	}

	// Update background file path display
	if (params->backgroundFrame != nullptr && params->backgroundFrameValid) {
		if (params->backgroundFrameFilePath.isEmpty()) {
			ui->lineEdit_bgFilePath->setText(tr("Recorded background (not saved to file)"));
		} else {
			ui->lineEdit_bgFilePath->setText(params->backgroundFrameFilePath);
		}
	} else {
		ui->lineEdit_bgFilePath->clear(); // Shows placeholder "No background loaded"
	}
}

void AdvancedSettingsDialog::updateBackgroundCorrectionModeControls(){
	bool enabled = ui->checkBox_bgFrameEnabled->isChecked();
	ui->radioButton_bgSubtractionOnly->setEnabled(enabled);
	ui->radioButton_bgSubtractionAndNormalization->setEnabled(enabled);
	ui->checkBox_bgAverageSpectra->setEnabled(enabled);
	ui->checkBox_bgSmoothSpectra->setEnabled(enabled);
	bool smoothingEnabled = enabled && ui->checkBox_bgSmoothSpectra->isChecked();
	ui->label_bgSmoothingWindow->setEnabled(smoothingEnabled);
	ui->spinBox_bgSmoothingWindow->setEnabled(smoothingEnabled);
}

void AdvancedSettingsDialog::applyContinuousBackgroundSettings(){
	saveSettings();

	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	params->continuousBackgroundUpdate = ui->checkBox_continuousBackground->isChecked();
	params->continuousBackgroundUseEMA = (ui->comboBox_avgMethod->currentIndex() == 0);

	// Enable/disable continuous mode controls
	bool continuous = params->continuousBackgroundUpdate;
	ui->comboBox_avgMethod->setEnabled(continuous);
	ui->label_avgMethod->setEnabled(continuous);

	// When continuous mode enabled, also enable subtraction
	if (continuous) {
		ui->checkBox_bgFrameEnabled->setChecked(true);
		params->backgroundFrameSubtraction = true;
	} else {
		// Restore original recorded background when switching back to static mode
		if (params->backgroundFrameValid) {
			params->backgroundFrameUpdated = true;
		}
	}

	// Enable/disable static mode controls based on continuous mode
	ui->pushButton_recordBackground->setEnabled(!continuous);
	ui->pushButton_saveBackground->setEnabled(!continuous && params->backgroundFrame != nullptr);
	ui->pushButton_loadBackground->setEnabled(!continuous);
	ui->label_bgStatus->setEnabled(!continuous);
	ui->label_bgStatusIndicator->setEnabled(!continuous);
	ui->label_bgFileInUse->setEnabled(!continuous);
	ui->lineEdit_bgFilePath->setEnabled(!continuous);
	updateBackgroundCorrectionModeControls();

	emit settingsChanged();
}

void AdvancedSettingsDialog::applyFrameCorrectionSettings(){
	saveSettings();

	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	params->frameCorrectionNormalizeByAvgSpectra = ui->checkBox_fcNormalizeByAvgSpectra->isChecked();

	emit settingsChanged();
}
