#ifndef ADVANCEDSETTINGSDIALOG_H
#define ADVANCEDSETTINGSDIALOG_H

#include <QDialog>

namespace Ui {
class AdvancedSettingsDialog;
}

// Settings keys
#define ADV_FULL_RANGE_MODE "advanced_full_range_mode"
#define ADV_CC_ARTIFACT_REMOVAL "advanced_cc_artifact_removal"
#define ADV_CC_RECT_CENTER "advanced_cc_rect_center"
#define ADV_CC_RECT_WIDTH "advanced_cc_rect_width"
#define ADV_CC_KEEP_POSITIVE "advanced_cc_keep_positive"

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
	void applyFullRangeModeSettings(bool enable);
	void applyCCSettings();

private:
	Ui::AdvancedSettingsDialog *ui;
	void connectSignals();
	void disconnectSignals();
};

#endif // ADVANCEDSETTINGSDIALOG_H
