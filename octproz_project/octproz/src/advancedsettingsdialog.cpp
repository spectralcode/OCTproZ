#include "advancedsettingsdialog.h"
#include "ui_advancedsettingsdialog.h"
#include "octalgorithmparameters.h"
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

void AdvancedSettingsDialog::connectSignals()
{
	connect(ui->checkBox_fullRangeMode, &QCheckBox::toggled,
			this, &AdvancedSettingsDialog::onFullRangeModeChanged);
}

void AdvancedSettingsDialog::onFullRangeModeChanged(bool checked)
{
	saveSettings();

	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	bool wasFullRange = params->fullRangeMode;
	params->fullRangeMode = checked;

	if (wasFullRange != checked) {
		params->fullRangeModeChanged = true;
		QMessageBox::information(this, tr("Restart Required"),
			tr("Full Range Mode change will take effect after restarting processing."));
	}

	// Enable/disable CC artifact removal group based on full range mode
	ui->groupBox_ccArtifactRemoval->setEnabled(checked);

	emit settingsChanged();
}

void AdvancedSettingsDialog::loadSettings()
{
	QSettings settings;
	ui->checkBox_fullRangeMode->setChecked(
		settings.value(ADV_FULL_RANGE_MODE, false).toBool());
	ui->doubleSpinBox_ccFilterCutoff->setValue(
		settings.value(ADV_CC_FILTER_CUTOFF, 0.5).toDouble());
	ui->doubleSpinBox_ccFilterShift->setValue(
		settings.value(ADV_CC_FILTER_SHIFT, 0.5).toDouble());

	// Sync with OctAlgorithmParameters on load
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	params->fullRangeMode = ui->checkBox_fullRangeMode->isChecked();

	// Update CC group enabled state
	ui->groupBox_ccArtifactRemoval->setEnabled(params->fullRangeMode);
}

void AdvancedSettingsDialog::saveSettings()
{
	QSettings settings;
	settings.setValue(ADV_FULL_RANGE_MODE, ui->checkBox_fullRangeMode->isChecked());
	settings.setValue(ADV_CC_FILTER_CUTOFF, ui->doubleSpinBox_ccFilterCutoff->value());
	settings.setValue(ADV_CC_FILTER_SHIFT, ui->doubleSpinBox_ccFilterShift->value());
}
