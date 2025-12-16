#ifndef ADVANCEDSETTINGSDIALOG_H
#define ADVANCEDSETTINGSDIALOG_H

#include <QDialog>

namespace Ui {
class AdvancedSettingsDialog;
}

// Settings keys
#define ADV_FULL_RANGE_MODE "advanced/full_range_mode"
#define ADV_CC_ARTIFACT_REMOVAL "advanced/cc_artifact_removal"
#define ADV_CC_FILTER_CUTOFF "advanced/cc_filter_cutoff"
#define ADV_CC_FILTER_SHIFT "advanced/cc_filter_shift"

class AdvancedSettingsDialog : public QDialog
{
	Q_OBJECT

public:
	explicit AdvancedSettingsDialog(QWidget *parent = nullptr);
	~AdvancedSettingsDialog();

	void loadSettings();
	void saveSettings();

signals:
	void settingsChanged();

private slots:
	void onFullRangeModeChanged(bool checked);

private:
	Ui::AdvancedSettingsDialog *ui;
	void connectSignals();
};

#endif // ADVANCEDSETTINGSDIALOG_H
