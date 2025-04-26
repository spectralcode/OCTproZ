#include "recordingscheduler.h"

RecordingScheduler::RecordingScheduler(QObject* parent)
	: QObject(parent),
	  startDelayTimer(new QTimer(this)),
	  intervalTimer(new QTimer(this)),
	  updateTimer(new QTimer(this)),
	  startDelaySeconds(0),
	  intervalSeconds(0),
	  totalRecordings(0),
	  recordingsCompleted(0),
	  isScheduleActive(false),
	  previousRecordingComplete(true)
{
	this->startDelayTimer->setSingleShot(true);
	this->intervalTimer->setSingleShot(true);
	this->updateTimer->setInterval(500); // Update time display every 500 ms

	connect(startDelayTimer, &QTimer::timeout, this, &RecordingScheduler::startDelayFinished);
	connect(intervalTimer, &QTimer::timeout, this, &RecordingScheduler::interval);
	connect(updateTimer, &QTimer::timeout, this, &RecordingScheduler::updateTimeUntilNext);
}

RecordingScheduler::~RecordingScheduler()
{
	this->stopSchedule();
}

bool RecordingScheduler::isActive() const {
	return this->isScheduleActive;
}

int RecordingScheduler::getRecordingsCompleted() const {
	return this->recordingsCompleted;
}

int RecordingScheduler::getTotalRecordings() const {
	return this->totalRecordings;
}

QDateTime RecordingScheduler::getNextRecordingTime() const {
	return this->nextRecordingTime;
}

int RecordingScheduler::getTimeUntilNextRecording() const {
	if(!this->isScheduleActive || this->nextRecordingTime.isNull()) {
		return 0;
	}

	return QDateTime::currentDateTime().secsTo(this->nextRecordingTime);
}

bool RecordingScheduler::isPreviousRecordingComplete() const {
	return this->previousRecordingComplete;
}

void RecordingScheduler::startSchedule(int startDelaySeconds, int intervalSeconds, int totalRecordings) {
	if(this->isScheduleActive) {
		this->stopSchedule();
	}

	this->startDelaySeconds = startDelaySeconds;
	this->intervalSeconds = intervalSeconds;
	this->totalRecordings = totalRecordings;
	this->recordingsCompleted = 0;
	this->isScheduleActive = true;
	this->previousRecordingComplete = true;

	this->nextRecordingTime = QDateTime::currentDateTime().addSecs(startDelaySeconds);

	this->startDelayTimer->start(startDelaySeconds * 1000);
	this->updateTimer->start();

	emit info(tr("Recording schedule started. Total planned recordings: %1").arg(totalRecordings));
	emit scheduleStarted();
	emit progressUpdated(recordingsCompleted, totalRecordings);
	emit timeUntilNextRecordingUpdated(startDelaySeconds);
}

void RecordingScheduler::stopSchedule() {
	if(!this->isScheduleActive) {
		return;
	}

	this->startDelayTimer->stop();
	this->intervalTimer->stop();
	this->updateTimer->stop();

	this->isScheduleActive = false;
	this->nextRecordingTime = QDateTime();

	emit info(tr("Recording schedule stopped"));
	emit scheduleStopped();
}

void RecordingScheduler::recordingFinished() {
	if(!this->isScheduleActive) {
		return;
	}

	this->previousRecordingComplete = true;
	this->recordingsCompleted++;
	emit progressUpdated(this->recordingsCompleted, this->totalRecordings);
	emit info(tr("Scheduled recording %1 of %2 completed").arg(this->recordingsCompleted).arg(this->totalRecordings));

	if(this->recordingsCompleted >= this->totalRecordings) {
		this->stopSchedule();
		emit info(tr("Recording schedule completed. All %1 recordings finished.").arg(this->totalRecordings));
		emit scheduleCompleted();
		return;
	}
}

void RecordingScheduler::startDelayFinished() {
	if(!this->isScheduleActive) {
		return;
	}

	if (!this->previousRecordingComplete) {
		emit error(tr("Previous recording is still in progress. Delaying first recording by 10 seconds."));
		this->nextRecordingTime = QDateTime::currentDateTime().addSecs(DELAY_RETRY_SECONDS);
		this->startDelayTimer->start(DELAY_RETRY_SECONDS * 1000);
		return;
	}

	this->previousRecordingComplete = false;
	emit info(tr("Starting scheduled recording 1 of %1").arg(this->totalRecordings));
	emit recordingTriggered();

	// Schedule next start time (start-to-start)
	this->nextRecordingTime = QDateTime::currentDateTime().addSecs(this->intervalSeconds);
	this->intervalTimer->start(this->intervalSeconds * 1000);
}

void RecordingScheduler::interval() {
	if(!this->isScheduleActive) {
		return;
	}

	// Schedule next start time (start-to-start timing)
	this->nextRecordingTime = QDateTime::currentDateTime().addSecs(this->intervalSeconds);
	this->intervalTimer->start(this->intervalSeconds * 1000);

	// If the previous recording hasn't completed, delay this attempt
	if (!this->previousRecordingComplete) {
		emit error(tr("Previous recording is still in progress. Delaying this recording by 10 seconds."));
		emit delayOccurred(DELAY_RETRY_SECONDS);
		this->nextRecordingTime = QDateTime::currentDateTime().addSecs(DELAY_RETRY_SECONDS);
		this->intervalTimer->start(DELAY_RETRY_SECONDS * 1000);
		return;
	}

	// Safe to start the next recording
	this->previousRecordingComplete = false;
	emit info(tr("Starting scheduled recording %1 of %2").arg(this->recordingsCompleted + 1).arg(this->totalRecordings));
	emit recordingTriggered();
}

void RecordingScheduler::updateTimeUntilNext() {
	if (!this->isScheduleActive || this->nextRecordingTime.isNull()) {
		return;
	}

	int secondsRemaining = QDateTime::currentDateTime().secsTo(this->nextRecordingTime);

	if (secondsRemaining <= 0) {
		secondsRemaining = 0;
	}

	emit timeUntilNextRecordingUpdated(secondsRemaining);
}
