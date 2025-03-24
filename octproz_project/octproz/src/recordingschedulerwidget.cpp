#include "recordingschedulerwidget.h"
#include <QTime>
#include <QPainter>
#include <QMenu>
#include <QAction>

const QString RecordingSchedulerWidget::DEFAULT_TIME_LABEL_STYLE = "QLabel { font-weight: bold; }";

RecordingSchedulerWidget::RecordingSchedulerWidget(QObject* parent)
	: QWidget(qobject_cast<QWidget*>(parent)),
	  scheduler(new RecordingScheduler(this)),
	  name("recording-scheduler"),
	  startDelaySeconds(0),
	  intervalSeconds(0)
{
	this->setWindowTitle(tr("Recording Scheduler"));
	//this->setWindowIcon(QIcon(":/icons/octproz_time_icon.png"));
	this->setWindowFlags(windowFlags() | Qt::Tool);

	this->initGui();
	this->setupConnections();

	this->updateStartDelaySecondsValue();
	this->updateIntervalSecondsValue();
}

RecordingSchedulerWidget::~RecordingSchedulerWidget()
{
}

void RecordingSchedulerWidget::setSettings(QVariantMap settings) {
	// Get values with defaults
	int startDelay = settings.value(SCHEDULER_START_DELAY, 0).toInt();
	int interval = settings.value(SCHEDULER_INTERVAL, 300).toInt();
	int totalRec = settings.value(SCHEDULER_TOTAL_RECORDINGS, 10).toInt();

	// Update internal values
	this->startDelaySeconds = startDelay;
	this->intervalSeconds = interval;

	// Update UI controls
	this->totalRecordingsSpinBox->setValue(totalRec);

	// Convert seconds to HH:MM:SS for the time edits
	int startHours = startDelay / 3600;
	int startMinutes = (startDelay % 3600) / 60;
	int startSecs = startDelay % 60;
	this->startDelayTimeEdit->setTime(QTime(startHours, startMinutes, startSecs));

	int intervalHours = interval / 3600;
	int intervalMinutes = (interval % 3600) / 60;
	int intervalSecs = interval % 60;
	this->intervalTimeEdit->setTime(QTime(intervalHours, intervalMinutes, intervalSecs));

	// Restore window geometry if available
	if(settings.contains(SCHEDULER_GEOMETRY)) {
		this->restoreGeometry(settings.value(SCHEDULER_GEOMETRY).toByteArray());
	}

	// Restore visibility if needed
	if(settings.value(SCHEDULER_VISIBLE, false).toBool()) {
		// Use a short delay to ensure main window is loaded first
		QTimer::singleShot(100, this, &QWidget::show);
	}

	// Ensure internal values are set (in case time edit signals haven't fired)
	this->updateStartDelaySecondsValue();
	this->updateIntervalSecondsValue();
}

QVariantMap RecordingSchedulerWidget::getSettings() const {
	QVariantMap settings;
	settings.insert(SCHEDULER_START_DELAY, this->startDelaySeconds);
	settings.insert(SCHEDULER_INTERVAL, this->intervalSeconds);
	settings.insert(SCHEDULER_TOTAL_RECORDINGS, this->totalRecordingsSpinBox->value());

	settings.insert(SCHEDULER_GEOMETRY, this->saveGeometry());
	settings.insert(SCHEDULER_VISIBLE, this->isVisible());

	return settings;
}

void RecordingSchedulerWidget::scheduleStarted() {
	this->startButton->setEnabled(false);
	this->stopButton->setEnabled(true);
	this->startDelayTimeEdit->setEnabled(false);
	this->intervalTimeEdit->setEnabled(false);
	this->totalRecordingsSpinBox->setEnabled(false);
}

void RecordingSchedulerWidget::scheduleStopped() {
	this->startButton->setEnabled(true);
	this->stopButton->setEnabled(false);
	this->startDelayTimeEdit->setEnabled(true);
	this->intervalTimeEdit->setEnabled(true);
	this->totalRecordingsSpinBox->setEnabled(true);

	// Add visual feedback that schedule was stopped
	if(this->scheduler->getRecordingsCompleted() >= this->scheduler->getTotalRecordings()) {
		// Schedule completed successfully
		this->nextRecordingLabel->setVisible(false);
		this->nextRecordingTimeLabel->setText(tr("Schedule completed!"));
		this->nextRecordingTimeLabel->setStyleSheet("QLabel { font-weight: bold; color: green; }");
	} else {
		// Schedule was stopped by user
		this->nextRecordingLabel->setVisible(false);
		this->nextRecordingTimeLabel->setText(tr("Schedule stopped"));
		this->nextRecordingTimeLabel->setStyleSheet("QLabel { font-weight: bold; color: #D32F2F; }");
	}
}

void RecordingSchedulerWidget::progressUpdated(int completed, int total) {
	if(total > 0) {
		int percentage = (completed * 100) / total;
		this->progressBar->setValue(percentage);
		this->progressBar->setFormat(tr("%p% (%1/%2)").arg(completed).arg(total));
	} else {
		this->progressBar->setValue(0);
		this->progressBar->setFormat("%p%");
	}

	this->progressLabel->setText(tr("Progress:"));
}

void RecordingSchedulerWidget::timeUntilNextUpdated(int seconds) {
	this->nextRecordingLabel->setVisible(seconds > 0);
	this->nextRecordingTimeLabel->setText(this->formatTimeRemaining(seconds));
}

void RecordingSchedulerWidget::scheduleCompleted(){
	//for future use
}

void RecordingSchedulerWidget::startSchedule() {
	this->updateStartDelaySecondsValue();
	this->updateIntervalSecondsValue();

	this->nextRecordingTimeLabel->setStyleSheet(DEFAULT_TIME_LABEL_STYLE);

	this->scheduler->startSchedule(this->startDelaySeconds, this->intervalSeconds, this->totalRecordingsSpinBox->value());
}

void RecordingSchedulerWidget::stopSchedule() {
	this->scheduler->stopSchedule();
}

void RecordingSchedulerWidget::updateStartDelaySecondsValue() {
	QTime delayTime = this->startDelayTimeEdit->time();
	this->startDelaySeconds = QTime(0, 0, 0).secsTo(delayTime);
}

void RecordingSchedulerWidget::updateIntervalSecondsValue() {
	QTime interval = intervalTimeEdit->time();
	this->intervalSeconds = QTime(0, 0, 0).secsTo(interval);

	//ensure minimum interval is 1 second
	if (this->intervalSeconds < 1) {
		this->intervalSeconds = 1;
		this->intervalTimeEdit->setTime(QTime(0, 0, 1));
	}
}

QString RecordingSchedulerWidget::formatTimeRemaining(int seconds) {
	if (seconds <= 0) {
		return tr("Recording now...");
	}

	int hours = seconds / 3600;
	int minutes = (seconds % 3600) / 60;
	int secs = seconds % 60;

	if (hours > 0) {
		return tr("%1:%2:%3").arg(hours, 2, 10, QChar('0')).arg(minutes, 2, 10, QChar('0')).arg(secs, 2, 10, QChar('0'));
	} else {
		return tr("%1:%2").arg(minutes, 2, 10, QChar('0')).arg(secs, 2, 10, QChar('0'));
	}
}

void RecordingSchedulerWidget::initGui() {
	// Main layout
	this->mainLayout = new QVBoxLayout(this);
	this->mainLayout->setSpacing(6);

	// Settings group
	this->settingsGroup = new QGroupBox(tr("Recording Schedule Settings"));
	this->settingsLayout = new QVBoxLayout(settingsGroup);
	this->settingsLayout->setSpacing(6);

	// Start delay setting
	this->startDelayLayout = new QHBoxLayout();
	this->startDelayLabel = new QLabel(tr("Start first recording in:"));
	this->startDelayTimeEdit = new QTimeEdit();
	this->startDelayTimeEdit->setDisplayFormat("HH:mm:ss");
	this->startDelayTimeEdit->setTime(QTime(0, 1, 0));
	this->startDelayTimeEdit->setMinimumTime(QTime(0, 0, 0));
	this->startDelayTimeEdit->setToolTip(tr("Time until first recording starts (hours:minutes:seconds)\nSet to 00:00:00 for immediate start"));
	this->startDelayLayout->addWidget(startDelayLabel);
	this->startDelayLayout->addWidget(startDelayTimeEdit);
	this->settingsLayout->addLayout(startDelayLayout);

	// Interval setting
	this->intervalLayout = new QHBoxLayout();
	this->intervalLabel = new QLabel(tr("Wait between recordings:"));
	this->intervalTimeEdit = new QTimeEdit();
	this->intervalTimeEdit->setDisplayFormat("HH:mm:ss");
	this->intervalTimeEdit->setTime(QTime(0, 0, 30));
	this->intervalTimeEdit->setMinimumTime(QTime(0, 0, 0));
	this->intervalTimeEdit->setToolTip("<html><head/><body><p>" + tr("Time interval between consecutive recordings (hours:minutes:seconds)") + "</p></body></html>");

	this->intervalLayout->addWidget(intervalLabel);
	this->intervalLayout->addWidget(intervalTimeEdit);
	this->settingsLayout->addLayout(intervalLayout);

	// Total recordings setting
	this->totalRecordingsLayout = new QHBoxLayout();
	this->totalRecordingsLabel = new QLabel(tr("Total recordings:"));
	this->totalRecordingsSpinBox = new QSpinBox();
	this->totalRecordingsSpinBox->setRange(1, 1000);
	this->totalRecordingsSpinBox->setValue(10);
	this->totalRecordingsLayout->addWidget(totalRecordingsLabel);
	this->totalRecordingsLayout->addWidget(totalRecordingsSpinBox);
	this->totalRecordingsLayout->addStretch(1);
	this->settingsLayout->addLayout(totalRecordingsLayout);

	// Control buttons
	this->buttonsLayout = new QHBoxLayout();
	this->startButton = new QPushButton(tr("Start Schedule"));
	this->startButton->setIcon(QIcon(":/icons/octproz_play_icon.png"));
	this->startButton->setCursor(Qt::PointingHandCursor);
	this->startButton->setStyleSheet(
		"QPushButton { color: white; padding: 6px; border: none; border-radius: 3px; }"
		"QPushButton:enabled { background-color: #4CAF50; }"
		"QPushButton:pressed { background-color: #388E3C; }"
	);

	this->stopButton = new QPushButton(tr("Stop Schedule"));
	this->stopButton->setIcon(QIcon(":/icons/octproz_stop_icon.png"));
	this->stopButton->setCursor(Qt::PointingHandCursor);
	this->stopButton->setStyleSheet(
		"QPushButton { color: white; padding: 6px; border: none; border-radius: 3px; }"
		"QPushButton:enabled { background-color: #F44336; }"
		"QPushButton:pressed { background-color: #D32F2F; }"
	);

	this->stopButton->setEnabled(false);
	this->buttonsLayout->addWidget(startButton);
	this->buttonsLayout->addWidget(stopButton);
	this->settingsLayout->addLayout(buttonsLayout);

	// Status group
	this->statusGroup = new QGroupBox(tr("Recording Schedule Status"));
	this->statusLayout = new QVBoxLayout(statusGroup);
	this->statusLayout->setSpacing(6);

	// Progress status
	this->progressLayout = new QHBoxLayout();
	this->progressLabel = new QLabel(tr("Progress: 0/0"));
	this->progressBar = new QProgressBar();
	this->progressBar->setRange(0, 100);
	this->progressBar->setValue(0);
	this->progressBar->setTextVisible(true);
	this->progressLayout->addWidget(progressLabel);
	this->progressLayout->addWidget(progressBar);
	this->statusLayout->addLayout(progressLayout);

	// Next recording status
	this->nextRecordingLayout = new QHBoxLayout();
	this->nextRecordingLabel = new QLabel(tr("Next recording in:"));
	this->nextRecordingTimeLabel = new QLabel(tr("Waiting to Start"));
	this->nextRecordingTimeLabel->setStyleSheet("QLabel { font-weight: bold; }");

	this->nextRecordingLabel->setVisible(false);
	this->nextRecordingLayout->addWidget(nextRecordingLabel);
	this->nextRecordingLayout->addWidget(nextRecordingTimeLabel);
	this->nextRecordingLayout->addStretch(1);
	this->statusLayout->addLayout(nextRecordingLayout);

	// Add groups to main layout
	this->mainLayout->addWidget(settingsGroup);
	this->mainLayout->addWidget(statusGroup);
}

void RecordingSchedulerWidget::setupConnections() {
	connect(this->startButton, &QPushButton::clicked, this, &RecordingSchedulerWidget::startSchedule);
	connect(this->stopButton, &QPushButton::clicked, this, &RecordingSchedulerWidget::stopSchedule);

	connect(this->startDelayTimeEdit, &QTimeEdit::timeChanged, this, &RecordingSchedulerWidget::updateStartDelaySecondsValue);
	connect(this->intervalTimeEdit, &QTimeEdit::timeChanged, this, &RecordingSchedulerWidget::updateIntervalSecondsValue);

	connect(this->scheduler, &RecordingScheduler::scheduleStarted, this, &RecordingSchedulerWidget::scheduleStarted);
	connect(this->scheduler, &RecordingScheduler::scheduleStopped, this, &RecordingSchedulerWidget::scheduleStopped);
	connect(this->scheduler, &RecordingScheduler::progressUpdated, this, &RecordingSchedulerWidget::progressUpdated);
	connect(this->scheduler, &RecordingScheduler::timeUntilNextRecordingUpdated, this, &RecordingSchedulerWidget::timeUntilNextUpdated);
	connect(this->scheduler, &RecordingScheduler::scheduleCompleted, this, &RecordingSchedulerWidget::scheduleCompleted);
	connect(this->scheduler, &RecordingScheduler::info, this, &RecordingSchedulerWidget::info);
	connect(this->scheduler, &RecordingScheduler::error, this, &RecordingSchedulerWidget::error);
}
