/**
* OCTproZ - Optical coherence tomography processing
* License: GNU GPL v3
* Copyright 2019-2025 Miroslav Zabic
*
*
* OCTproZApp: Core application logic that manages plugin loading
* and coordinates data acquisition, processing, and recording
*/

#ifndef OCTPROZAPP_H
#define OCTPROZAPP_H

#include <QObject>
#include <QThread>
#include <QVariantMap>
#include <QDir>
#include <QPluginLoader>
#include <QSettings>
#include "octalgorithmparameters.h"
#include "octalgorithmparametersmanager.h"
#include "systemmanager.h"
#include "extensionmanager.h"
#include "extensioneventfilter.h"
#include "pluginmessagebus.h"
#include "processing.h"
#include "octproz_devkit.h"
#include "settingsconstants.h"

#define APP_VERSION "1.7.0"
#define APP_VERSION_DATE "2 December 2023"
#define APP_NAME "OCTproZ"


class OCTproZApp : public QObject
{
	Q_OBJECT
	QThread acquisitionThread;
	QThread processingThread;
	QThread notifierThread;

public:
	explicit OCTproZApp(QObject* parent = nullptr);
	~OCTproZApp();

	void initialize();
	void loadSystemsAndExtensions();

	// Accessor methods
	SystemManager* getSysManager() const { return sysManager; }
	AcquisitionSystem* getCurrentSystem() const { return currSystem; }
	ExtensionManager* getExtManager() const { return extManager; }
	Processing* getSignalProcessing() const { return signalProcessing; }
	OctAlgorithmParameters* getOctParams() const { return octParams; }
	OctAlgorithmParametersManager* getParamsManager() const { return paramsManager; }
	Gpu2HostNotifier* getProcessedDataNotifier() const { return processedDataNotifier; }
	QString getCurrentSystemName() const { return currSystemName; }
	bool isProcessingInThread() const { return processingInThread; }
	bool getStreamToHostMemorized() const { return streamToHostMemorized; }
	unsigned int getStreamingBuffersToSkipMemorized() const { return streamingBuffersToSkipMemorized; }
	void setActiveExtensions(const QStringList& extensions) { this->activeExtensions = extensions; }
	PluginMessageBus* getMessageBus() const { return messageBus; }

	// System management
	void setSystem(QString systemName);
	void activateSystem(AcquisitionSystem* system);
	void deactivateSystem(AcquisitionSystem* system);
	void reactivateSystem(AcquisitionSystem* system);

	// Parameters & settings management
	void forceUpdateProcessingParams();
	void loadSettingsFromFile(const QString& settingsFilePath);
	void saveSettingsToFile(const QString& fileName);
	void loadResamplingCurveFromFile(QString fileName);


public slots:
	// Action slots
	void slot_start();
	void slot_stop();
	void slot_record();
	void slot_updateAcquistionParameter(AcquisitionParams newParams);
	void slot_storePluginSettings(QString pluginName, QVariantMap settings);
	void slot_prepareGpu2HostForProcessedRecording();
	void slot_resetGpu2HostSettings();
	void slot_recordingDone();
	void slot_useCustomResamplingCurve(bool use);
	void slot_setKLinCoeffs(double* k0, double* k1, double* k2, double* k3);
	void slot_setDispCompCoeffs(double* d0, double* d1, double* d2, double* d3);
	void slot_setCustomResamplingCurve(QVector<float> resamplingCurve);

signals:
	void start();
	void stop();
	void record();
	void enableRecording(OctAlgorithmParameters::RecordingParams recParams);
	void newSystemSelected();
	void newSystem(AcquisitionSystem*);
	void loadPluginSettings(QVariantMap);
	void error(QString);
	void info(QString);

	// UI update signals
	void systemChanged(const QString& systemName);
	void processingStarted();
	void processingStopped();
	void recordingStarted();
	void recordingFinished();
	void grayscaleConversionRequested(bool enableLogScaling, double max, double min, double multiplicator, double offset);
	void windowStateReloadRequested();
	void loadSettingsRequested();

	void enFaceViewDimensionsChanged(unsigned int width, unsigned int height, unsigned int depth);
	void bScanDimensionsChanged(unsigned int width, unsigned int height, unsigned int depth);
	void linesPerBufferChanged(int linesPerBuffer);
	void bscansPerBufferChanged(unsigned int value);
	void samplesPerLineChanged(unsigned int value);
	void streamToHostSettingsChanged(bool enabled, int buffersToSkip);
	void customResamplingCurveChanged(bool enabled);
	void klinCoeffsChanged(double* k0, double* k1, double* k2, double* k3);
	void dispCompCoeffsChanged(double* d0, double* d1, double* d2, double* d3);

	void processingParamsUpdateRequested();
	void screenshotsRequested(const QString& savePath, const QString& baseName);
	void streamToHostSettingsReset();


private:
	// Core components
	SettingsFileManager* appSettings;
	SystemManager* sysManager;
	AcquisitionSystem* currSystem;
	ExtensionManager* extManager;
	PluginMessageBus* messageBus;
	Processing* signalProcessing;
	OctAlgorithmParameters* octParams;
	OctAlgorithmParametersManager* paramsManager;
	Gpu2HostNotifier* processedDataNotifier;

	// State variables
	bool processingInThread;
	bool streamToHostMemorized;
	unsigned int streamingBuffersToSkipMemorized;
	QString currSystemName;
	QList<QString> activatedSystems;
	int rerunCounter = 0;

	QStringList activeExtensions;
};

#endif // OCTPROZAPP_H
