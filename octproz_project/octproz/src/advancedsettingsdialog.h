#ifndef ADVANCEDSETTINGSDIALOG_H
#define ADVANCEDSETTINGSDIALOG_H

#include <QDialog>
#include <QTimer>

namespace Ui {
class AdvancedSettingsDialog;
}

// Settings keys
#define ADV_FULL_RANGE_MODE "advanced_full_range_mode"
#define ADV_CC_ARTIFACT_REMOVAL "advanced_cc_artifact_removal"
#define ADV_CC_RECT_CENTER "advanced_cc_rect_center"
#define ADV_CC_RECT_WIDTH "advanced_cc_rect_width"
#define ADV_CC_KEEP_POSITIVE "advanced_cc_keep_positive"

// Background Frame Subtraction settings keys
#define ADV_BG_FRAME_ENABLED "advanced_bg_frame_enabled"
#define ADV_BG_FRAME_BSCANS_TO_AVG "advanced_bg_frame_bscans_to_average"
#define ADV_BG_FRAME_FILE_PATH "advanced_bg_frame_file_path"

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
	void applyBackgroundFrameSettings();
	void recordBackgroundFrame();
	void saveBackgroundFrame();
	void loadBackgroundFrame();
	void checkRecordingStatus();

private:
	Ui::AdvancedSettingsDialog *ui;
	QTimer* recordingStatusTimer;
	void connectSignals();
	void disconnectSignals();
	void updateBackgroundFrameStatus();
};

#endif // ADVANCEDSETTINGSDIALOG_H
