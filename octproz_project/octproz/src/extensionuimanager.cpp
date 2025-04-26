#include "extensionuimanager.h"
#include "extensionmanager.h"
#include "settingsfilemanager.h"
#include "settingsconstants.h"

ExtensionUIManager::ExtensionUIManager(QMenu* extensionMenu, QTabWidget* tabWidget,
									  MessageConsole* console, Sidebar* sidebar,
									  OCTproZApp* app, QObject* parent) :
	QObject(parent),
	extensionMenu(extensionMenu),
	tabWidget(tabWidget),
	console(console),
	sidebar(sidebar),
	app(app),
	extManager(nullptr),
	appSettings(new SettingsFileManager(this))
{
}

ExtensionUIManager::~ExtensionUIManager()
{

}

void ExtensionUIManager::initialize(ExtensionManager* extManager) {
	this->extManager = extManager;

	// Create extension actions for each extension
	auto extensionNames = extManager->getExtensionNames();
	foreach(QString extensionName, extensionNames) {
		QAction* extAction = this->extensionMenu->addAction(extensionName, this, &ExtensionUIManager::slot_menuExtensions);
		this->extensionActions.append(extAction);
		extAction->setCheckable(true);
		extAction->setChecked(false);

		// Get extension tooltip
		Extension* extension = extManager->getExtensionByName(extensionName);
		QString extensionToolTip = extension == nullptr ? "" : extension->getToolTip();
		extAction->setStatusTip(extensionToolTip);
	}

	connect(this, &ExtensionUIManager::extensionActivationRequested, extManager, &ExtensionManager::slot_extensionMenuItemTriggered);
	connect(extManager, &ExtensionManager::extensionLoadRequested, this, &ExtensionUIManager::slot_handleExtensionLoading);
}

void ExtensionUIManager::setupExtensionGUI(Extension* extension) {
	if (!extension) return;

	// Setup UI components based on display style
	if (extension->getDisplayStyle() == SIDEBAR_TAB) {
		this->addExtensionToSidebar(this->tabWidget, extension);
	} else if (extension->getDisplayStyle() == SEPARATE_WINDOW) {
		this->createExtensionWindow(extension);
	}

	this->connectExtensionWithSidebar(extension);
}

void ExtensionUIManager::cleanupExtensionGUI(Extension* extension) {
	if (!extension) return;

	if (extension->getDisplayStyle() == SIDEBAR_TAB) {
		this->removeExtensionFromSidebar(this->tabWidget, extension);
	} else if (extension->getDisplayStyle() == SEPARATE_WINDOW) {
		this->closeExtensionWindow(extension);
	}

	this->disconnectExtensionFromSidebar(extension);
}

void ExtensionUIManager::shutdownAllExtensions() {
	QList<QString> extensionNames = this->extManager->getExtensionNames();

	foreach(QString name, extensionNames) {
		Extension* extension = this->extManager->getExtensionByName(name);
		if (extension) {
			this->cleanupExtensionGUI(extension);
			this->extManager->deactivateExtension(extension);
		}
	}
}

QStringList ExtensionUIManager::getActiveExtensionNames() const {
	QStringList activeExtensions;
	foreach (QAction* action, this->extensionActions) {
		if (action->isChecked()) {
			activeExtensions.append(action->text());
		}
	}
	return activeExtensions;
}

void ExtensionUIManager::saveExtensionStates() {
	QStringList activeExtensions = this->getActiveExtensionNames();
	this->extManager->setActiveExtensions(activeExtensions);
	this->extManager->saveExtensionStates();
}

void ExtensionUIManager::autoLoadExtensions() {
	QVariantMap mainWindowSettings = this->appSettings->getStoredSettings(MAIN_WINDOW_SETTINGS_GROUP);

	if (!mainWindowSettings.contains(MAIN_ACTIVE_EXTENSIONS)) {
		return;
	}

	QStringList autoloadExtensions = mainWindowSettings.value(MAIN_ACTIVE_EXTENSIONS).toStringList();
	if (autoloadExtensions.isEmpty()){
		return;
	}

	foreach (QAction* action, this->extensionActions) {
		if (autoloadExtensions.contains(action->text())) {
			if (!action->isChecked()) {
				//triggering the action will call slot_menuExtensions with the action as sender
				action->trigger();
			}
		}
	}
}

void ExtensionUIManager::slot_menuExtensions() {
	QAction* currAction = qobject_cast<QAction*>(sender());
	if(currAction) {
		emit extensionActivationRequested(currAction->text(), currAction->isChecked());
	}
}

void ExtensionUIManager::slot_uncheckExtensionInMenu(Extension* extension) {
	QString extensionName = extension->getName();
	QAction* currAction = nullptr;
	foreach(auto action, this->extensionActions) {
		if(action->text() == extensionName){
			currAction = action;
			break;
		}
	}

	if(currAction) {
		currAction->setChecked(false);
	}

	emit extensionActivationRequested(extensionName, false);
}

void ExtensionUIManager::slot_handleExtensionLoading(Extension* extension, bool load) {
	if (load) {
		// Note: The order is important here. In ExtensionManager::activateExtension,
		// settings are loaded first, then extension is activated
		this->extManager->activateExtension(extension);
		this->setupExtensionGUI(extension);
	} else {
		this->extManager->deactivateExtension(extension);
		this->cleanupExtensionGUI(extension);
	}
}

void ExtensionUIManager::slot_enableRawGrabbing(bool allowed) {
	this->extManager->slot_enableRawGrabbing(allowed);
}

void ExtensionUIManager::connectExtensionWithSidebar(Extension* extension) {
	if (!extension) return;

	connect(extension, &Plugin::setKLinCoeffsRequest, this->app, &OCTproZApp::slot_setKLinCoeffs);
	connect(extension, &Plugin::setDispCompCoeffsRequest, this->sidebar, &Sidebar::slot_setDispCompCoeffs);
	connect(this->sidebar, &Sidebar::klinCoeffs, extension, &Plugin::setKLinCoeffsRequestAccepted);
	connect(this->sidebar, &Sidebar::dispCompCoeffs, extension, &Plugin::setDispCompCoeffsRequestAccepted);
}

void ExtensionUIManager::disconnectExtensionFromSidebar(Extension* extension) {
	if (!extension) return;

	disconnect(extension, &Plugin::setKLinCoeffsRequest, this->app, &OCTproZApp::slot_setKLinCoeffs);
	disconnect(extension, &Plugin::setDispCompCoeffsRequest, this->sidebar, &Sidebar::slot_setDispCompCoeffs);
	disconnect(this->sidebar, &Sidebar::klinCoeffs, extension, &Plugin::setKLinCoeffsRequestAccepted);
	disconnect(this->sidebar, &Sidebar::dispCompCoeffs, extension, &Plugin::setDispCompCoeffsRequestAccepted);
}

void ExtensionUIManager::addExtensionToSidebar(QTabWidget* tabWidget, Extension* extension) {
	if (!extension || !tabWidget) return;

	QWidget* extensionWidget = extension->getWidget();
	tabWidget->addTab(extensionWidget, extension->getName());
}

void ExtensionUIManager::removeExtensionFromSidebar(QTabWidget* tabWidget, Extension* extension) {
	if (!extension || !tabWidget) return;

	QWidget* extensionWidget = extension->getWidget();
	int index = tabWidget->indexOf(extensionWidget);
	if (index != -1) {
		tabWidget->removeTab(index);
	}
}

void ExtensionUIManager::createExtensionWindow(Extension* extension) {
	if (!extension) return;

	QWidget* extensionWidget = extension->getWidget();
	extensionWidget->setParent(qobject_cast<QWidget*>(this->parent()), Qt::Tool);
	extensionWidget->setAttribute(Qt::WA_DeleteOnClose, false);
	extensionWidget->setWindowTitle(extension->getName());

	if (!extensionWidget->property("eventFilterInstalled").toBool()) {
		//first-time setup only: create and install the event filter
		ExtensionEventFilter* extensionCloseSignalForwarder = new ExtensionEventFilter(extensionWidget);
		extensionCloseSignalForwarder->setExtension(extension);
		extensionWidget->installEventFilter(extensionCloseSignalForwarder);

		connect(extensionCloseSignalForwarder, &ExtensionEventFilter::extensionWidgetClosed,
			this, &ExtensionUIManager::slot_uncheckExtensionInMenu);

		extensionWidget->setProperty("eventFilterInstalled", true);
	}

	//delay show() to the next event loop cycle (0ms in QTimer::singleShot means it runs after pending events)
	//to ensure proper window initialization before starting extension logic (required for the camera extension to work on startup, and maybe for other extensions too)
	QTimer::singleShot(0, this, [this, extension, extensionWidget]() {
		extensionWidget->show();
	});
}

void ExtensionUIManager::closeExtensionWindow(Extension* extension) {
	if (!extension) return;

	QWidget* extensionWidget = extension->getWidget();
	extensionWidget->setParent(nullptr); //prevents the app from crashing on close. Each extensionWidget is deleted by its respective extension; if it still has a parent, use-after-free might occur because the parent will attempt to access or delete it during its own destruction.
	extensionWidget->close();
}
