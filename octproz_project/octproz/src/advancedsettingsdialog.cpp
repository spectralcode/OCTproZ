#include "advancedsettingsdialog.h"
#include "ui_advancedsettingsdialog.h"
#include "octalgorithmparameters.h"
#include "settingsfilemanager.h"
#include <QSettings>
#include <QMessageBox>

AdvancedSettingsDialog::AdvancedSettingsDialog(QWidget *parent) :
	QDialog(parent),
	ui(new Ui::AdvancedSettingsDialog)
{
	ui->setupUi(this);
	loadSettings();
	connectSignals();
}

AdvancedSettingsDialog::~AdvancedSettingsDialog()
{
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

	// Sync with OctAlgorithmParameters on load
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	params->fullRangeMode = ui->checkBox_fullRangeMode->isChecked();
	params->ccArtifactRemoval = ui->groupBox_ccArtifactRemoval->isChecked();
	params->ccRectCenterFreq = ui->doubleSpinBox_ccRectCenter->value();
	params->ccRectWidth = ui->doubleSpinBox_ccRectWidth->value();
	params->ccKeepPositiveSideband = ui->checkBox_ccKeepPositive->isChecked();

	// Update CC group enabled state
	ui->groupBox_ccArtifactRemoval->setEnabled(params->fullRangeMode);

	connectSignals();
}

void AdvancedSettingsDialog::saveSettings() {
	QSettings settings(SETTINGS_PATH, QSettings::IniFormat);
	settings.setValue(ADV_FULL_RANGE_MODE, ui->checkBox_fullRangeMode->isChecked());
	settings.setValue(ADV_CC_ARTIFACT_REMOVAL, ui->groupBox_ccArtifactRemoval->isChecked());
	settings.setValue(ADV_CC_RECT_CENTER, ui->doubleSpinBox_ccRectCenter->value());
	settings.setValue(ADV_CC_RECT_WIDTH, ui->doubleSpinBox_ccRectWidth->value());
	settings.setValue(ADV_CC_KEEP_POSITIVE, ui->checkBox_ccKeepPositive->isChecked());
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
