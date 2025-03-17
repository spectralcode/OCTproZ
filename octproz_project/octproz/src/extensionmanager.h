#ifndef EXTENSIONMMANAGER_H
#define EXTENSIONMMANAGER_H

#include <QObject>
#include <QList>
#include <QStringList>
#include <QAction>
#include "octproz_devkit.h"
#include "processing.h"
#include "gpu2hostnotifier.h"
#include "settingsconstants.h"


class OCTproZApp;

class ExtensionManager : public QObject
{
	Q_OBJECT
public:
	explicit ExtensionManager(QObject* parent = nullptr);
	~ExtensionManager();

	void initialize(OCTproZApp* app, Processing* processing, Gpu2HostNotifier* notifier);

	// Basic extension management
	void addExtension(Extension* extension);
	Extension* getExtensionByName(QString name);
	QList<Extension*> getExtensions() { return this->extensions; }
	QList<QString> getExtensionNames() { return this->extensionNames; }

	// Extension activation/deactivation
	void activateExtension(Extension* extension);
	void deactivateExtension(Extension* extension);

	// Connection management
	void connectExtensionSignals(Extension* extension);
	void disconnectExtensionSignals(Extension* extension);

	// State management
	QStringList getActiveExtensionNames() { return this->activeExtensions; }
	void setActiveExtensions(const QStringList& extensions) { this->activeExtensions = extensions; }
	void saveExtensionStates();
	QStringList loadActiveExtensions();
	void autoLoadExtensions(QList<QAction*>& extensionActions);

public slots:
	void slot_extensionMenuItemTriggered(const QString& extensionName, bool checked);
	void slot_enableRawGrabbing(bool allowed);
	void slot_storePluginSettings(QString pluginName, QVariantMap settings);

signals:
	void extensionLoadRequested(Extension* extension, bool load);
	void allowRawGrabbing(bool allowed);
	void error(QString);
	void info(QString);

private:
	QList<Extension*> extensions;
	QList<QString> extensionNames;
	QStringList activeExtensions;

	// External components for signal connections
	OCTproZApp* app;
	Processing* signalProcessing;
	Gpu2HostNotifier* notifier;
	bool rawGrabbingAllowed;

	Settings* appSettings;
};

#endif // EXTENSIONMMANAGER_H
