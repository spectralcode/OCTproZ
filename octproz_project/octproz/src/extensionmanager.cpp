#include "extensionmanager.h"
#include "settings.h"
#include "octprozapp.h"

ExtensionManager::ExtensionManager(QObject *parent) : QObject(parent)
{
	this->app = nullptr;
	this->signalProcessing = nullptr;
	this->notifier = nullptr;
	this->rawGrabbingAllowed = false;
	this->appSettings = new Settings(this); //todo: consider using a separate file for extension settings.
}

ExtensionManager::~ExtensionManager()
{
	qDeleteAll(this->extensions);
	this->extensions.clear();
	this->extensionNames.clear();
	this->activeExtensions.clear();
}

void ExtensionManager::initialize(OCTproZApp* app, Processing* processing, Gpu2HostNotifier* notifier)
{
	this->app = app;
	this->signalProcessing = processing;
	this->notifier = notifier;
}

void ExtensionManager::addExtension(Extension* extension) {
	if (extension != nullptr) {
		if (!extensions.contains(extension)) {
			this->extensions.append(extension);
			this->extensionNames.append(extension->getName());
			this->extensionNames.last().detach(); //force deep copy of appended extension name to avoid possible problems if plugin lives at some point in a thread
		}
	}
}

Extension* ExtensionManager::getExtensionByName(QString name) {
	int index = this->extensionNames.indexOf(name);
	return index == -1 ? nullptr : this->extensions.at(index);
}

void ExtensionManager::activateExtension(Extension* extension) {
	if (!extension) return;

	extension->setParent(this->app); // todo: intially this was necessary to automatically close all open extension windows when main application is closed, but it seems that now extension windows also close without this -> investigate and maybe remove this line and remove this->app pointer from extensionmanager

	// Load settings and activate extension
	//info: the order is important here. loading of settings need to be done first, otherwise the extension window geometry will not be restored
	extension->settingsLoaded(this->appSettings->getStoredSettings(extension->getName()));
	extension->activateExtension();

	// Connect non-GUI signals
	this->connectExtensionSignals(extension);
}

void ExtensionManager::deactivateExtension(Extension* extension) {
	if (!extension) return;

	// Deactivate extension
	extension->deactivateExtension();

	// Disconnect non-GUI signals
	this->disconnectExtensionSignals(extension);
}

void ExtensionManager::connectExtensionSignals(Extension* extension) {
	if (!extension || !this->app || !this->signalProcessing || !this->notifier) {
		return;
	}

	connect(extension, &Extension::storeSettings, this, &ExtensionManager::slot_storePluginSettings);

	// Connect data flow signals
	extension->enableRawDataGrabbing(this->rawGrabbingAllowed); //todo: is initialization of rawGrabbingAllowed really needed here?
	connect(this, &ExtensionManager::allowRawGrabbing, extension, &Extension::enableRawDataGrabbing);
	connect(this->signalProcessing, &Processing::streamingBufferEnabled, extension, &Extension::enableProcessedDataGrabbing);
	connect(this->notifier, &Gpu2HostNotifier::newGpuDataAvailable, extension, &Extension::processedDataReceived);
	connect(this->signalProcessing, &Processing::rawData, extension, &Extension::rawDataReceived);
}

void ExtensionManager::disconnectExtensionSignals(Extension* extension) {
	if (!extension) {
		return;
	}

	// Disconnect non-GUI signals
	disconnect(extension, &Extension::storeSettings, this, &ExtensionManager::slot_storePluginSettings);

	// Disconnect data flow signals
	disconnect(this, &ExtensionManager::allowRawGrabbing, extension, &Extension::enableRawDataGrabbing);
	disconnect(this->signalProcessing, &Processing::streamingBufferEnabled, extension, &Extension::enableProcessedDataGrabbing);
	disconnect(this->notifier, &Gpu2HostNotifier::newGpuDataAvailable, extension, &Extension::processedDataReceived);
	disconnect(this->signalProcessing, &Processing::rawData, extension, &Extension::rawDataReceived);
}

void ExtensionManager::slot_extensionMenuItemTriggered(const QString& extensionName, bool checked) {
	Extension* extension = this->getExtensionByName(extensionName);
	if (extension == nullptr) {
		emit error(tr("No Extension with name ") + extensionName + tr(" exists."));
		return;
	}

	// Update active extensions list
	if (checked && !this->activeExtensions.contains(extensionName)) {
		this->activeExtensions.append(extensionName);
	} else if (!checked && this->activeExtensions.contains(extensionName)) {
		this->activeExtensions.removeOne(extensionName);
	}

	emit extensionLoadRequested(extension, checked);
}

void ExtensionManager::saveExtensionStates() {
	QVariantMap windowSettings;
	windowSettings[MAIN_ACTIVE_EXTENSIONS] = this->activeExtensions;
	this->appSettings->storeSettings(MAIN_WINDOW_SETTINGS_GROUP, windowSettings);
}

QStringList ExtensionManager::loadActiveExtensions() {
	// Get settings for auto-loading extensions
	QVariantMap mainWindowSettings = this->appSettings->getStoredSettings(MAIN_WINDOW_SETTINGS_GROUP);

	// Check if the settings contain autoload extension list
	if (!mainWindowSettings.contains(MAIN_ACTIVE_EXTENSIONS)) {
		return QStringList();
	}

	return mainWindowSettings.value(MAIN_ACTIVE_EXTENSIONS).toStringList();
}

void ExtensionManager::autoLoadExtensions(QList<QAction*>& extensionActions) {
	// Get auto-load extension list from settings
	QStringList autoloadExtensions = this->loadActiveExtensions();
	if (autoloadExtensions.isEmpty())
		return;

	// Set active extensions
	this->activeExtensions = autoloadExtensions;

	// Trigger the actions for extensions in the autoload list
	foreach (QAction* action, extensionActions) {
		if (autoloadExtensions.contains(action->text())) {
			if (!action->isChecked()) {
				action->trigger();
			}
		}
	}
}

void ExtensionManager::slot_enableRawGrabbing(bool allowed) {
	this->rawGrabbingAllowed = allowed;
	emit allowRawGrabbing(allowed);
}

void ExtensionManager::slot_storePluginSettings(QString pluginName, QVariantMap settings) {
	this->appSettings->storeSettings(pluginName, settings);
}
