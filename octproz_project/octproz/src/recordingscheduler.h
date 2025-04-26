#ifndef RECORDINGSCHEDULER_H
#define RECORDINGSCHEDULER_H

#include <QObject>
#include <QTimer>
#include <QDateTime>

class RecordingScheduler : public QObject
{
	Q_OBJECT

public:
	const int DELAY_RETRY_SECONDS = 10;

	explicit RecordingScheduler(QObject* parent = nullptr);
	~RecordingScheduler();

	bool isActive() const;
	int getRecordingsCompleted() const;
	int getTotalRecordings() const;
	QDateTime getNextRecordingTime() const;
	int getTimeUntilNextRecording() const; // in seconds
	bool isPreviousRecordingComplete() const;

public slots:
	void startSchedule(int startDelaySeconds, int intervalSeconds, int totalRecordings);
	void stopSchedule();
	void recordingFinished();

signals:
	void scheduleStarted();
	void scheduleStopped();
	void recordingTriggered();
	void progressUpdated(int recordingsCompleted, int totalRecordings);
	void timeUntilNextRecordingUpdated(int seconds);
	void scheduleCompleted();
	void delayOccurred(int delaySeconds);
	void info(QString infoMessage);
	void error(QString errorMessage);

private slots:
	void startDelayFinished();
	void interval();
	void updateTimeUntilNext();

private:
	QTimer* startDelayTimer;
	QTimer* intervalTimer;
	QTimer* updateTimer;

	int startDelaySeconds;
	int intervalSeconds;
	int totalRecordings;
	int recordingsCompleted;
	bool isScheduleActive;
	bool previousRecordingComplete;
	QDateTime nextRecordingTime;
};

#endif // RECORDINGSCHEDULER_H
