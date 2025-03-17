#include "octprozapp.h"
#include <QApplication>
#include <QMessageBox>
#include <QThread>
#include <QDateTime>

OCTproZApp::OCTproZApp(QObject* parent) :
	QObject(parent)
{
	qRegisterMetaType<AcquisitionParams>("AcquisitionParams");
	qRegisterMetaType<OctAlgorithmParameters::RecordingParams>("OctAlgorithmParameters::RecordingParams");
	qRegisterMetaType<size_t>("size_t");

	qApp->setApplicationVersion(APP_VERSION);
	qApp->setApplicationName(APP_NAME);

	// Initialize core components
	this->appSettings = new Settings(this);
	connect(this->appSettings, &Settings::info, this, &OCTproZApp::info);
	connect(this->appSettings, &Settings::error, this, &OCTproZApp::error);
	this->sysManager = new SystemManager();
	this->currSystem = nullptr;
	this->currSystemName = "";
	this->octParams = OctAlgorithmParameters::getInstance();
	this->paramsManager = new OctAlgorithmParametersManager();
	connect(this->paramsManager, &OctAlgorithmParametersManager::info, this, &OCTproZApp::info);
	connect(this->paramsManager, &OctAlgorithmParametersManager::error, this, &OCTproZApp::error);

	this->extManager = new ExtensionManager();
	this->messageBus = new PluginMessageBus(this);

	// Initialize signal processing
	this->processingInThread = false;
	this->signalProcessing = new Processing();

	#if defined(Q_OS_WIN) || defined(__aarch64__)
		this->signalProcessing->moveToThread(&this->processingThread);
		this->processingInThread = true;
	#elif defined(Q_OS_LINUX)
		this->signalProcessing->moveToThread(&this->processingThread);
		this->processingInThread = true;
		//todo: fix linux bug: opengl window seems to be laggy on ubuntu test system if signalProcessing is moved to thread and Virtual OCT System is used with small wait time (wait time that is set in the gui under "Wait after file read")
	#endif

	connect(&this->processingThread, &QThread::finished, this->signalProcessing, &Processing::deleteLater);

	connect(this, &OCTproZApp::enableRecording, this->signalProcessing, &Processing::slot_enableRecording);
	connect(this->signalProcessing, &Processing::info, this, &OCTproZApp::info);
	connect(this->signalProcessing, &Processing::error, this, &OCTproZApp::error);
	connect(this->signalProcessing, &Processing::processedRecordDone, this, &OCTproZApp::slot_resetGpu2HostSettings);
	connect(this->signalProcessing, &Processing::processedRecordDone, this, &OCTproZApp::slot_recordingDone);
	connect(this->signalProcessing, &Processing::rawRecordDone, this, &OCTproZApp::slot_recordingDone);
	connect(this->signalProcessing, &Processing::initializationFailed, this, &OCTproZApp::slot_stop);

	// GPU to CPU notifier
	this->processedDataNotifier = Gpu2HostNotifier::getInstance();
	this->processedDataNotifier->moveToThread(&this->notifierThread);
	connect(&this->notifierThread, &QThread::finished, this->processedDataNotifier, &Gpu2HostNotifier::deleteLater);

	// Default values to memorize stream2host setings
	this->streamToHostMemorized = this->octParams->streamToHost;
	this->streamingBuffersToSkipMemorized = this->octParams->streamingBuffersToSkip;
}

OCTproZApp::~OCTproZApp() {
	// stop any running system
	if (this->currSystem && this->currSystem->acqusitionRunning) {
		this->slot_stop();
		QCoreApplication::processEvents();
		// some delay to ensure processing completes
		QThread::msleep(500);
	}

	// terminate threads
	this->notifierThread.quit();
	this->processingThread.quit();
	this->acquisitionThread.quit();
	if (!this->acquisitionThread.wait(1000))
		this->acquisitionThread.terminate();
	if (!this->processingThread.wait(1000))
		this->processingThread.terminate();
	if (!this->notifierThread.wait(1000))
		this->notifierThread.terminate();

	// delete resources
	delete this->paramsManager;
	delete this->sysManager;
	delete this->extManager;
}

void OCTproZApp::initialize() {
	this->processingThread.start();
	this->notifierThread.start();
	this->extManager->initialize(this, this->signalProcessing, this->processedDataNotifier);
	this->loadSystemsAndExtensions();
}

void OCTproZApp::loadSystemsAndExtensions() {
	QDir pluginsDir = QDir(qApp->applicationDirPath());

	// Check if plugins dir exists. If it does not exist change to the share_dev directory, this makes software development easier because plugins can be copied automatically to the share_dev during the build process
	bool pluginsDirExists = pluginsDir.cd("plugins");
	if (!pluginsDirExists) {
		#if defined(Q_OS_WIN)
		if (pluginsDir.dirName().toLower() == "debug" || pluginsDir.dirName().toLower() == "release") {
			pluginsDir.cdUp();
		}
		#endif

		pluginsDir.cdUp();
		pluginsDir.cdUp();
		pluginsDir.cd("octproz_project");
		pluginsDir.cd("octproz_share_dev");
		pluginsDir.cd("plugins");

		// Change directory if debug or release folder exist
		#ifdef QT_DEBUG
			pluginsDir.cd("debug");
		#else
			pluginsDir.cd("release");
		#endif
	}

	for (auto fileName : pluginsDir.entryList(QDir::Files)) {
		QPluginLoader loader(pluginsDir.absoluteFilePath(fileName));
		QObject *plugin = loader.instance(); //todo: figure out why qobject_cast<Plugin*>(loader.instance()) does not work. probably because Qt plugin API is not used correctly; see https://github.com/spectralcode/OCTproZ/issues/11
		if (plugin) {
			Plugin* actualPlugin = (Plugin*)(plugin);

			// Register with message bus
			this->messageBus->registerPlugin(actualPlugin->getName(), actualPlugin);
			connect(actualPlugin, &Plugin::sendCommand, this->messageBus, &PluginMessageBus::sendCommand);

			// Connect plugin signals to app functions
			connect(actualPlugin, &Plugin::startProcessingRequest, this, &OCTproZApp::slot_start);
			connect(actualPlugin, &Plugin::stopProcessingRequest, this, &OCTproZApp::slot_stop);
			connect(actualPlugin, &Plugin::startRecordingRequest, this, &OCTproZApp::slot_record);
			connect(actualPlugin, &Plugin::setCustomResamplingCurveRequest, this, &OCTproZApp::slot_setCustomResamplingCurve);
			connect(actualPlugin, &Plugin::loadSettingsFileRequest, this, &OCTproZApp::loadSettingsFromFile);
			connect(actualPlugin, &Plugin::saveSettingsFileRequest, this, &OCTproZApp::saveSettingsToFile);
			connect(actualPlugin, &Plugin::info, this, &OCTproZApp::info);
			connect(actualPlugin, &Plugin::error, this, &OCTproZApp::error);

			connect(actualPlugin, &Plugin::setKLinCoeffsRequest, this, &OCTproZApp::slot_setKLinCoeffs); //todo: maybe remove this signal and slot and come up with a more general way to transfer parameter between plugins and app
			connect(actualPlugin, &Plugin::setDispCompCoeffsRequest, this, &OCTproZApp::slot_setDispCompCoeffs);
			connect(actualPlugin, &Plugin::setGrayscaleConversionRequest, this, &OCTproZApp::grayscaleConversionRequested);

			enum PLUGIN_TYPE type = actualPlugin->getType();
			switch (type) {
				case SYSTEM: {
					this->sysManager->addSystem(qobject_cast<AcquisitionSystem*>(plugin));
					break;
				}
				case EXTENSION: {
					this->extManager->addExtension(qobject_cast<Extension*>(plugin));
					break;
				}
				default: {
					emit error(tr("Could not load Plugin"));
				}
			}
		} else {
			emit error(tr("Could not load ") + fileName);
		}
	}
}

void OCTproZApp::forceUpdateProcessingParams() {
	this->octParams->acquisitionParamsChanged = true;
	this->octParams->postProcessBackgroundUpdated = true;
	emit processingParamsUpdateRequested(); //todo: recheck why this->sidebar->slot_updateProcessingParams() needs to be called, otherwise processing output windows remain black
}

void OCTproZApp::slot_start() {
	if (this->currSystem != nullptr) {
		if(this->currSystem->acqusitionRunning) {
			return;
		}
	}

	// (Re-)init resampling curve, dispersion curve, window curve, streaming
	this->forceUpdateProcessingParams();

	// set current system time as timestamp
	this->appSettings->setCurrentTimeStamp();

	// Emit start signal to activate acquisition of current AcquisitionSystem
	emit processingStarted();
	emit start();

	// For debugging purposes: read out thread affinity of current thread
	qDebug() << "Main Thread ID start emit: " << QThread::currentThreadId();
}

void OCTproZApp::slot_stop() {
	// Emit stop signal to stop acquisition system if it is still running
	if (this->currSystem != nullptr) {
		if(this->currSystem->acqusitionRunning) {
			emit stop();
			QApplication::processEvents();
			this->currSystem->acqusitionRunning = false; //todo: think about whether OCTproZ should really set the AcquisitionRunning flag to false here, or whether only the acquisition system itself should be responsible for setting this flag to false
			emit processingStopped();
		}
	}
}

void OCTproZApp::slot_record() {
	// Check if system is open
	if (this->currSystem == nullptr) {
		emit error(tr("Nothing to record, no system opened."));
		return;
	}

	// Check if the user selected anything to record
	OctAlgorithmParameters::RecordingParams recParams = this->octParams->recParams;
	if (!recParams.recordScreenshot && !recParams.recordRaw && !recParams.recordProcessed) {
		emit error(tr("Nothing to record! Please select what to record in the recording settings!"));
		return;
	}

	// set current system time as timestamp
	this->appSettings->setCurrentTimeStamp();

	// Set time stamp so it can be used in all file names of the same recording session
	recParams.timestamp = this->appSettings->getTimestamp();

	// Get user defined rec name
	QString recName = recParams.fileName;
	if (recName != "") {
		recName = "_" + recName;
	}

	// Enable raw and processed recording
	if (recParams.recordProcessed) {
		this->slot_prepareGpu2HostForProcessedRecording();
	}

	emit recordingStarted();

	if (recParams.recordRaw || recParams.recordProcessed) {
		emit this->enableRecording(recParams);
		if (!this->currSystem->acqusitionRunning) {
			this->slot_start();
		}
	}

	if (recParams.recordScreenshot) {
		QString savePath = recParams.savePath;
		QString fileName = recParams.timestamp + recName + "_";
		emit screenshotsRequested(savePath, fileName);
	}

	// Check if meta information should be saved
	if (recParams.saveMetaData) {
		QString metaFileName = recParams.savePath + "/" + recParams.timestamp + recName + "_meta.txt";
		this->appSettings->copySettingsFile(metaFileName);
	}
}

void OCTproZApp::setSystem(QString systemName) {
	if(this->currSystemName == systemName) { // System already activated
		emit info(tr("System is already open."));
		return;
	}

	AcquisitionSystem* system = this->sysManager->getSystemByName(systemName);

	if(system == nullptr) {
		emit error(tr("Opening of OCT system failed. Could not find a system with the name: ") + systemName);
		return;
	}

	if(this->currSystem != nullptr) {
		this->deactivateSystem(this->currSystem);
	}

	if(!this->activatedSystems.contains(systemName)) { // System got selected for the first time
		this->activatedSystems.append(systemName);
		this->activateSystem(system);
	} else { // System was once active and needs to be reactivated now
		this->reactivateSystem(system);
	}

	this->currSystem = system;
	this->currSystemName = systemName;
	this->octParams->acquisitionParamsChanged = true; //todo: if this flag is not set here the minicurveplots dont show correct reference curves on startup when "use polynomial curve" is selected. however this flag should be set in slot_updateAcquistionParameter when a new acquisitionsystem is set or parameter change (and it is!) --> investigate

	emit systemChanged(systemName);
	emit loadPluginSettings(this->appSettings->getStoredSettings(systemName));
	emit info(tr("System opened: ") + this->currSystemName);
}

void OCTproZApp::activateSystem(AcquisitionSystem* system) {
	if(system != nullptr) {
		if(this->currSystem != system) {
			system->moveToThread(&this->acquisitionThread);
			connect(this, &OCTproZApp::start, system, &AcquisitionSystem::startAcquisition);
			connect(this, &OCTproZApp::stop, system, &AcquisitionSystem::stopAcquisition);
			connect(this, &OCTproZApp::loadPluginSettings, system, &AcquisitionSystem::settingsLoaded);
			connect(system, &AcquisitionSystem::storeSettings, this, &OCTproZApp::slot_storePluginSettings);
			connect(system, &AcquisitionSystem::acquisitionStarted, this->signalProcessing, &Processing::slot_start);
			connect(system, &AcquisitionSystem::acquisitionStopped, this, &OCTproZApp::slot_stop);
			connect(system->params, &AcquisitionParameter::updated, this, &OCTproZApp::slot_updateAcquistionParameter);
			connect(qApp, &QCoreApplication::aboutToQuit, system, &QObject::deleteLater);
			connect(system->buffer, &AcquisitionBuffer::info, this, &OCTproZApp::info);
			connect(system->buffer, &AcquisitionBuffer::error, this, &OCTproZApp::error);
			emit newSystem(system);
			this->acquisitionThread.start();
		}
	}
}

void OCTproZApp::deactivateSystem(AcquisitionSystem* system) {
	this->slot_stop();
	QCoreApplication::processEvents(); // Process events to ensure that acquisition is not running
	disconnect(this, &OCTproZApp::start, system, &AcquisitionSystem::startAcquisition);
	disconnect(this, &OCTproZApp::stop, system, &AcquisitionSystem::stopAcquisition);
	disconnect(this, &OCTproZApp::loadPluginSettings, system, &AcquisitionSystem::settingsLoaded);
}

void OCTproZApp::reactivateSystem(AcquisitionSystem* system) {
	connect(this, &OCTproZApp::start, system, &AcquisitionSystem::startAcquisition);
	connect(this, &OCTproZApp::stop, system, &AcquisitionSystem::stopAcquisition);
	connect(this, &OCTproZApp::loadPluginSettings, system, &AcquisitionSystem::settingsLoaded);
}

void OCTproZApp::slot_updateAcquistionParameter(AcquisitionParams newParams) {
	// Save old values
	unsigned int oldSamplesPerLine = this->octParams->samplesPerLine;
	unsigned int oldAscansPerBscan = this->octParams->ascansPerBscan;
	unsigned int oldBscansPerBuffer = this->octParams->bscansPerBuffer;
	unsigned int oldBuffersPerVolume = this->octParams->buffersPerVolume;

	// Update params
	this->octParams->samplesPerLine = newParams.samplesPerLine;
	this->octParams->ascansPerBscan = newParams.ascansPerBscan;
	this->octParams->bscansPerBuffer = newParams.bscansPerBuffer;
	this->octParams->buffersPerVolume = newParams.buffersPerVolume;
	this->octParams->bitDepth = newParams.bitDepth;
	this->octParams->updatePostProcessingBackgroundCurve();

	// Emit changes (this is used by 1d plot dock, en face view, bscan view to update buffer and texture sizes)
	if(oldSamplesPerLine != newParams.samplesPerLine ||
	   oldAscansPerBscan != newParams.ascansPerBscan ||
	   oldBscansPerBuffer != newParams.bscansPerBuffer ||
	   oldBuffersPerVolume != newParams.buffersPerVolume) {
		emit bScanDimensionsChanged(newParams.samplesPerLine/2, newParams.ascansPerBscan, newParams.bscansPerBuffer*newParams.buffersPerVolume);
		emit enFaceViewDimensionsChanged(newParams.ascansPerBscan, newParams.bscansPerBuffer*newParams.buffersPerVolume, newParams.samplesPerLine/2);
	}
	if(oldAscansPerBscan != newParams.ascansPerBscan || oldBscansPerBuffer != newParams.bscansPerBuffer){
		emit linesPerBufferChanged(newParams.ascansPerBscan * newParams.bscansPerBuffer);
	}
	if(oldSamplesPerLine != newParams.samplesPerLine) {
		emit samplesPerLineChanged(newParams.samplesPerLine);
	}
	if(oldBscansPerBuffer != newParams.bscansPerBuffer) {
		emit bscansPerBufferChanged(newParams.bscansPerBuffer);
	}

	this->forceUpdateProcessingParams();
}

void OCTproZApp::slot_storePluginSettings(QString pluginName, QVariantMap settings) {
	this->appSettings->storeSettings(pluginName, settings);
}

void OCTproZApp::slot_prepareGpu2HostForProcessedRecording() {
	this->streamToHostMemorized = this->octParams->streamToHost;
	this->streamingBuffersToSkipMemorized = this->octParams->streamingBuffersToSkip;

	this->octParams->streamingBuffersToSkip = 0;
	this->octParams->streamToHost = true;

	emit streamToHostSettingsChanged(true, 0);
}

void OCTproZApp::slot_resetGpu2HostSettings() {
	this->octParams->streamingBuffersToSkip = this->streamingBuffersToSkipMemorized;
	this->octParams->streamToHost = this->streamToHostMemorized;
	emit streamToHostSettingsReset();
}

void OCTproZApp::slot_recordingDone() {
	emit recordingFinished();

	if(this->octParams->recParams.stopAfterRecord && this->currSystem->acqusitionRunning) {
		this->slot_stop();
	}
}

void OCTproZApp::slot_useCustomResamplingCurve(bool use) {
	this->octParams->useCustomResampleCurve = use; //todo: this slot is called twice when user loads custom resampling curve, because customResamplingCurveChanged signal is emitted in two locations: when loading custom curve and when swichting between polynomial to custom. maybe start by checking if this->octParams->useCustomResampleCurve != use
	this->octParams->acquisitionParamsChanged = true;

	emit processingParamsUpdateRequested(); //todo: check if/why this is ecactly needed here
	emit customResamplingCurveChanged(use);
}

void OCTproZApp::slot_setKLinCoeffs(double* k0, double* k1, double* k2, double* k3) {
	if(k0 != nullptr) {
		this->octParams->c0 = *k0;
	}
	if(k1 != nullptr) {
		this->octParams->c1 = *k1;
	}
	if(k2 != nullptr) {
		this->octParams->c2 = *k2;
	}
	if(k3 != nullptr) {
		this->octParams->c3 = *k3;
	}

//todo: the invokeMethod call below can be used to inform the plugin about the currently used parameters
//maybe remove this and come up with a more general way to transfer parameters from and to plugins from the main app
//Plugin* requestingPlugin = qobject_cast<Plugin*>(sender());// doesnt work, probably because Qt plugin API is not used correctly; see https://github.com/spectralcode/OCTproZ/issues/11
Plugin* requestingPlugin = (Plugin*)(sender());
if (requestingPlugin) {
	QMetaObject::invokeMethod(requestingPlugin,
								"setKLinCoeffsRequestAccepted",
								Qt::QueuedConnection,
								Q_ARG(double, this->octParams->c0),
								Q_ARG(double, this->octParams->c1),
								Q_ARG(double, this->octParams->c2),
								Q_ARG(double, this->octParams->c3));
}

	// Notify sidebar to update UI
	emit klinCoeffsChanged(k0, k1, k2, k3);

	this->slot_useCustomResamplingCurve(false);
}

void OCTproZApp::slot_setDispCompCoeffs(double* d0, double* d1, double* d2, double* d3) {
	if(d0 != nullptr) {
		this->octParams->d0 = *d0;
	}
	if(d1 != nullptr) {
		this->octParams->d1 = *d1;
	}
	if(d2 != nullptr) {
		this->octParams->d2 = *d2;
	}
	if(d3 != nullptr) {
		this->octParams->d3 = *d3;
	}

//inform the requesting plugin about the currently used parameters
//todo: maybe remove, see comment above in slot_setKLinCoeffs()
Plugin* requestingPlugin = (Plugin*)(sender());
if (requestingPlugin) {
	QMetaObject::invokeMethod(requestingPlugin,
								"setDispCompCoeffsRequestAccepted",
								Qt::QueuedConnection,
								Q_ARG(double, this->octParams->d0),
								Q_ARG(double, this->octParams->d1),
								Q_ARG(double, this->octParams->d2),
								Q_ARG(double, this->octParams->d3));
}
	emit dispCompCoeffsChanged(d0, d1, d2, d3);
}

void OCTproZApp::slot_setCustomResamplingCurve(QVector<float> curve) {
	this->octParams->loadCustomResampleCurve(curve.data(), curve.size());
	this->octParams->acquisitionParamsChanged = true;
	emit customResamplingCurveChanged(true);
	this->slot_useCustomResamplingCurve(true); //todo: probably it is not neccassry to have  emit customResamplingCurveChanged(true); AND this slot here. rewrite this more cleanly, avoid redundant parameter updates
}

void OCTproZApp::loadSettingsFromFile(const QString& settingsFilePath) { //todo: extend OctAlgorithmParametersManager and use it for loading/saving setting from/to file
	if (settingsFilePath.isEmpty()) {
		emit error(tr("Invalid File: ") + tr("No settings file selected or file path is invalid."));
		return;
	}

	// Backup current settings file
	QString backupPath = SETTINGS_PATH + ".backup";
	QFile::remove(backupPath);  // Remove any existing backup
	QFile::copy(SETTINGS_PATH, backupPath);

	// Attempt to copy the selected settings file to the default settings path
	if (!QFile::remove(SETTINGS_PATH)) {
		emit error(tr("Failed to remove the existing settings file."));
		return;
	}

	if (!QFile::copy(settingsFilePath, SETTINGS_PATH)) {
		emit error(tr("Failed to load settings from: ") + settingsFilePath);
		QFile::copy(backupPath, SETTINGS_PATH);
		return;
	}

	// Reload the settings;
	emit loadSettingsRequested(); //todo: this connects to a slot that reloads the parameters from the QMaps (see loadSettings in sidebar) and does not load the parameters from a file. re-check if this is really intended here.
	emit windowStateReloadRequested(); //todo: implement this as a user-selectable option.

	if(!this->currSystemName.isEmpty()) {
		emit loadPluginSettings(this->appSettings->getStoredSettings(this->currSystemName));
	}

	// Inform the user that the settings have been loaded
	emit info(tr("Settings have been loaded successfully from: ") + settingsFilePath);
}

void OCTproZApp::saveSettingsToFile(const QString &fileName) { //todo: extend OctAlgorithmParametersManager and use it for loading/saving setting from/to file
	if (fileName.isEmpty()) {
		emit error(tr("Settings file not saved."));
		return;
	}

	// Ensure the file has an extension - use .ini as default
	QString finalFileName = fileName;
	QFileInfo fileInfo(fileName);
	if (fileInfo.suffix().isEmpty()) {
		finalFileName = fileName + ".ini";
	}

	// save current system time to settings file
	this->appSettings->setCurrentTimeStamp();

	// Save the settings by copying the existing settings file
	if (this->appSettings->copySettingsFile(finalFileName)) {
		emit info(tr("Settings have been saved successfully to: ") + finalFileName); //there are already success and error messages emitted in settingsManager but not displayed in message console. todo: more consistend error handling.
	} else {
		emit error(tr("Failed to save settings to: ") + finalFileName);
	}
}

void OCTproZApp::loadResamplingCurveFromFile(QString fileName) {
	if(fileName == "") {
		return;
	}

	QFile file(fileName);
	QVector<float> curve;
	file.open(QIODevice::ReadOnly);
	QTextStream txtStream(&file);
	QString line = txtStream.readLine();
	while (!txtStream.atEnd()) {
		line = txtStream.readLine();
		curve.append((line.section(";", 1, 1).toFloat()));
	}
	file.close();

	if(curve.size() > 0) {
		this->slot_setCustomResamplingCurve(curve);
		this->octParams->customResampleCurveFilePath = fileName;
		emit info(tr("Custom resampling curve loaded. File used: ") + fileName);
	} else {
		emit error(tr("Custom resampling curve has a size of 0. Check if .csv file with resampling curve is not empty has right format."));
	}
}
