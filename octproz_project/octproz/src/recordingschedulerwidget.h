#ifndef RECORDINGSCHEDULERWIDGET_H
#define RECORDINGSCHEDULERWIDGET_H

#define SCHEDULER_START_DELAY "scheduler_start_delay_seconds"
#define SCHEDULER_INTERVAL "scheduler_interval_seconds"
#define SCHEDULER_TOTAL_RECORDINGS "scheduler_total_recordings"
#define SCHEDULER_GEOMETRY "scheduler_window_geometry"
#define SCHEDULER_VISIBLE "scheduler_window_visible"

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QLabel>
#include <QSpinBox>
#include <QPushButton>
#include <QProgressBar>
#include <QTimer>
#include <QTimeEdit>
#include <QComboBox>
#include <QStyle>
#include <QStyleOption>
#include <QContextMenuEvent>

#include "recordingscheduler.h"

class RecordingSchedulerWidget : public QWidget
{
	Q_OBJECT

public:
	explicit RecordingSchedulerWidget(QObject* parent = nullptr);
	~RecordingSchedulerWidget();

	QString getName() const { return name; }
	void setName(const QString &name) { this->name = name; }

	RecordingScheduler* getScheduler() const { return scheduler; }

	void setSettings(QVariantMap settings);
	QVariantMap getSettings() const;

public slots:
	void scheduleStarted();
	void scheduleStopped();
	void progressUpdated(int completed, int total);
	void timeUntilNextUpdated(int seconds);
	void scheduleCompleted();

signals:
	void info(QString infoMessage);
	void error(QString errorMessage);

private slots:
	void startSchedule();
	void stopSchedule();
	void updateStartDelaySecondsValue();
	void updateIntervalSecondsValue();

private:
	RecordingScheduler* scheduler;
	QString name;
	int startDelaySeconds;
	int intervalSeconds;
	QString formatTimeRemaining(int seconds);
	void initGui();
	void setupConnections();
	static const QString DEFAULT_TIME_LABEL_STYLE;

	// UI elements
	QVBoxLayout* mainLayout;

	QGroupBox* settingsGroup;
	QVBoxLayout* settingsLayout;

	QHBoxLayout* startDelayLayout;
	QLabel* startDelayLabel;
	QTimeEdit* startDelayTimeEdit;

	QHBoxLayout* intervalLayout;
	QLabel* intervalLabel;
	QTimeEdit* intervalTimeEdit;

	QHBoxLayout* totalRecordingsLayout;
	QLabel* totalRecordingsLabel;
	QSpinBox* totalRecordingsSpinBox;

	QHBoxLayout* buttonsLayout;
	QPushButton* startButton;
	QPushButton* stopButton;

	QGroupBox* statusGroup;
	QVBoxLayout* statusLayout;

	QHBoxLayout* progressLayout;
	QLabel* progressLabel;
	QProgressBar* progressBar;

	QHBoxLayout* nextRecordingLayout;
	QLabel* nextRecordingLabel;
	QLabel* nextRecordingTimeLabel;
};

#endif // RECORDINGSCHEDULERWIDGET_H
