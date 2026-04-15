#include "octprozapp.h"
#include <QHash>

namespace {
using AppCommandHandler = void (OCTproZApp::*)(const QVariantMap &params);
}

bool OCTproZApp::isBackgroundFrameRecordingActive() const {
	return this->octParams->backgroundFrameRecordingRequested || this->octParams->backgroundFrameRecordingInProgress;
}

bool OCTproZApp::isProcessingActive() const {
	return this->currSystem != nullptr && this->currSystem->acqusitionRunning;
}

void OCTproZApp::slot_handleAppCommand(const QString &command, const QVariantMap &params) {
	static const QHash<QString, AppCommandHandler> handlers = {
		{"set_rec_path", &OCTproZApp::handleSetRecPathCommand},
		{"set_rec_name", &OCTproZApp::handleSetRecNameCommand},
		{"set_buffers_to_record", &OCTproZApp::handleSetBuffersToRecordCommand},
		{"record", &OCTproZApp::handleRecordCommand},
		{"set_rec_options", &OCTproZApp::handleSetRecOptionsCommand},
		{"set_preallocation", &OCTproZApp::handleSetPreallocationCommand},
		{"set_bg_frame", &OCTproZApp::handleSetBgFrameCommand},
		{"set_continuous_bg", &OCTproZApp::handleSetContinuousBgCommand},
		{"record_bg_frame", &OCTproZApp::handleRecordBgFrameCommand},
		{"load_bg_frame", &OCTproZApp::handleLoadBgFrameCommand},
		{"save_bg_frame", &OCTproZApp::handleSaveBgFrameCommand},
		{"clear_bg_frame", &OCTproZApp::handleClearBgFrameCommand},
		{"set_full_range", &OCTproZApp::handleSetFullRangeCommand},
		{"set_cc", &OCTproZApp::handleSetCcCommand},
	};

	const auto handler = handlers.constFind(command);
	if (handler == handlers.cend()) {
		emit error(tr("Unknown app command: ") + command);
		return;
	}

	(this->*handler.value())(params);
}

void OCTproZApp::handleSetRecPathCommand(const QVariantMap &params) {
	QString path = params.value("path").toString();
	if (!QDir(path).exists()) {
		emit error(tr("Recording path does not exist: ") + path);
		return;
	}

	this->octParams->recParams.savePath = path;
	emit info(tr("Recording path set to: ") + path);
}

void OCTproZApp::handleSetRecNameCommand(const QVariantMap &params) {
	QString name = params.value("name").toString();
	this->octParams->recParams.fileName = name;
	emit info(tr("Recording name set to: ") + name);
}

void OCTproZApp::handleSetBuffersToRecordCommand(const QVariantMap &params) {
	bool ok;
	unsigned int count = params.value("count").toUInt(&ok);
	if (!ok || count == 0) {
		emit error(tr("Invalid buffer count: ") + params.value("count").toString());
		return;
	}

	this->octParams->recParams.buffersToRecord = count;
	emit info(tr("Buffers to record set to: ") + QString::number(count));
}

void OCTproZApp::handleRecordCommand(const QVariantMap &params) {
	if (params.contains("path")) {
		QString path = params.value("path").toString();
		if (!QDir(path).exists()) {
			emit error(tr("Recording path does not exist: ") + path);
			return;
		}
		this->octParams->recParams.savePath = path;
	}
	if (params.contains("name")) {
		this->octParams->recParams.fileName = params.value("name").toString();
	}
	if (params.contains("buffers")) {
		bool ok;
		unsigned int count = params.value("buffers").toUInt(&ok);
		if (!ok || count == 0) {
			emit error(tr("Invalid buffer count: ") + params.value("buffers").toString());
			return;
		}
		this->octParams->recParams.buffersToRecord = count;
	}

	this->slot_record();
}

void OCTproZApp::handleSetRecOptionsCommand(const QVariantMap &params) {
	auto& rp = this->octParams->recParams;
	if (params.contains("raw")) rp.recordRaw = params.value("raw").toBool();
	if (params.contains("processed")) rp.recordProcessed = params.value("processed").toBool();
	if (params.contains("screenshot")) rp.recordScreenshot = params.value("screenshot").toBool();
	if (params.contains("meta")) rp.saveMetaData = params.value("meta").toBool();
	if (params.contains("stop_after")) rp.stopAfterRecord = params.value("stop_after").toBool();
	if (params.contains("start_first")) rp.startWithFirstBuffer = params.value("start_first").toBool();
	if (params.contains("float32")) rp.saveAs32bitFloat = params.value("float32").toBool();

	emit info(tr("Recording options updated"));
}

void OCTproZApp::handleSetPreallocationCommand(const QVariantMap &params) {
	bool enable = params.value("enable").toBool();
	QMetaObject::invokeMethod(this->signalProcessing, "slot_preallocateRecordingBuffers",
		Qt::QueuedConnection, Q_ARG(bool, enable));
	emit info(QString(tr("Buffer preallocation %1")).arg(enable ? tr("enabled") : tr("disabled")));
}

void OCTproZApp::handleSetBgFrameCommand(const QVariantMap &params) {
	if (this->isBackgroundFrameRecordingActive()) {
		emit error(tr("Cannot change background frame settings while background recording is active."));
		return;
	}

	if (params.contains("bscans")) {
		bool ok;
		unsigned int bscans = params.value("bscans").toUInt(&ok);
		if (!ok || bscans == 0) {
			emit error(tr("Invalid background frame B-scan count: ") + params.value("bscans").toString());
			return;
		}
		this->octParams->backgroundFrameBscansToAverage = bscans;
	}
	if (params.contains("enable")) {
		this->octParams->backgroundFrameSubtraction = params.value("enable").toBool();
	}

	emit info(tr("Background frame settings updated"));
}

void OCTproZApp::handleSetContinuousBgCommand(const QVariantMap &params) {
	if (this->isBackgroundFrameRecordingActive()) {
		emit error(tr("Cannot change continuous background settings while background recording is active."));
		return;
	}

	bool continuousDisabled = false;
	if (params.contains("enable")) {
		bool enable = params.value("enable").toBool();
		this->octParams->continuousBackgroundUpdate = enable;
		if (enable) {
			this->octParams->backgroundFrameSubtraction = true;
		} else {
			continuousDisabled = true;
		}
	}
	if (params.contains("ema")) {
		this->octParams->continuousBackgroundUseEMA = params.value("ema").toBool();
	}
	if (continuousDisabled && this->octParams->backgroundFrameValid) {
		this->octParams->backgroundFrameUpdated = true;
	}

	emit info(tr("Continuous background settings updated"));
}

void OCTproZApp::handleRecordBgFrameCommand(const QVariantMap &params) {
	Q_UNUSED(params)

	if (this->octParams->continuousBackgroundUpdate) {
		emit error(tr("Cannot record a background frame while continuous background mode is enabled."));
		return;
	}
	if (this->isBackgroundFrameRecordingActive()) {
		emit error(tr("Background frame recording is already active."));
		return;
	}

	this->octParams->backgroundFrameRecordingRequested = true;
	this->octParams->backgroundFrameBscansRecorded = 0;
	this->octParams->backgroundFrameFilePath.clear();

	emit info(tr("Background frame recording requested"));
}

void OCTproZApp::handleLoadBgFrameCommand(const QVariantMap &params) {
	QString path = params.value("path").toString().trimmed();
	if (path.isEmpty()) {
		emit error(tr("Invalid background frame path."));
		return;
	}
	if (this->octParams->continuousBackgroundUpdate) {
		emit error(tr("Cannot load a background frame while continuous background mode is enabled."));
		return;
	}
	if (this->isBackgroundFrameRecordingActive()) {
		emit error(tr("Cannot load a background frame while background recording is active."));
		return;
	}
	if (!this->octParams->loadBackgroundFrameFromFile(path)) {
		emit error(tr("Failed to load background frame from: ") + path);
		return;
	}

	this->octParams->updateBackgroundFrameValidity();
	if (this->octParams->samplesPerLine > 0 && this->octParams->ascansPerBscan > 0 && !this->octParams->backgroundFrameValid) {
		emit info(tr("Background frame loaded, but dimensions do not match the current acquisition. It will remain inactive until dimensions match."));
	} else if (!this->octParams->backgroundFrameValid) {
		emit info(tr("Background frame loaded. It will be validated once acquisition dimensions are known."));
	} else {
		emit info(tr("Background frame loaded from: ") + path);
	}
}

void OCTproZApp::handleSaveBgFrameCommand(const QVariantMap &params) {
	QString path = params.value("path").toString().trimmed();
	if (path.isEmpty()) {
		emit error(tr("Invalid background frame path."));
		return;
	}
	if (this->octParams->continuousBackgroundUpdate) {
		emit error(tr("Cannot save a background frame while continuous background mode is enabled."));
		return;
	}
	if (this->isBackgroundFrameRecordingActive()) {
		emit error(tr("Cannot save a background frame while background recording is active."));
		return;
	}
	if (this->octParams->backgroundFrame == nullptr) {
		emit error(tr("No background frame is available to save."));
		return;
	}
	if (!this->octParams->saveBackgroundFrameToFile(path)) {
		emit error(tr("Failed to save background frame to: ") + path);
		return;
	}

	emit info(tr("Background frame saved to: ") + path);
}

void OCTproZApp::handleClearBgFrameCommand(const QVariantMap &params) {
	Q_UNUSED(params)

	if (this->isBackgroundFrameRecordingActive()) {
		emit error(tr("Cannot clear the background frame while background recording is active."));
		return;
	}

	this->octParams->clearBackgroundFrame();
	this->octParams->backgroundFrameSubtraction = false;
	this->octParams->continuousBackgroundUpdate = false;

	emit info(tr("Background frame cleared and background subtraction disabled"));
}

void OCTproZApp::handleSetFullRangeCommand(const QVariantMap &params) {
	bool enable = params.value("enable").toBool();
	bool changed = this->octParams->fullRangeMode != enable;
	this->octParams->fullRangeMode = enable;
	if (changed) {
		this->octParams->fullRangeModeChanged = true;
	}

	emit info(QString(tr("Full-range mode %1")).arg(enable ? tr("enabled") : tr("disabled")));
	if (this->isProcessingActive()) {
		emit info(tr("Full-range mode changes will take effect after processing is stopped and started again."));
	}
}

void OCTproZApp::handleSetCcCommand(const QVariantMap &params) {
	if (params.contains("center")) {
		bool ok;
		double center = params.value("center").toDouble(&ok);
		if (!ok || center < 0.0 || center > 1.0) {
			emit error(tr("Invalid CC center frequency: ") + params.value("center").toString());
			return;
		}
		this->octParams->ccRectCenterFreq = static_cast<float>(center);
	}
	if (params.contains("width")) {
		bool ok;
		double width = params.value("width").toDouble(&ok);
		if (!ok || width < 0.0 || width > 1.0) {
			emit error(tr("Invalid CC width: ") + params.value("width").toString());
			return;
		}
		this->octParams->ccRectWidth = static_cast<float>(width);
	}
	if (params.contains("enable")) {
		this->octParams->ccArtifactRemoval = params.value("enable").toBool();
	}
	if (params.contains("keep_positive")) {
		this->octParams->ccKeepPositiveSideband = params.value("keep_positive").toBool();
	}

	emit info(tr("Complex conjugate artifact removal settings updated"));
	if (!this->octParams->fullRangeMode) {
		emit info(tr("Complex conjugate artifact removal only applies when full-range mode is active."));
	}
}
