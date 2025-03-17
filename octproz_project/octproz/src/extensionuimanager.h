/**
 * ExtensionUIManager: Manages the UI aspects of OCTproZ extensions
 * including display windows and extension menu actions.
 */

#ifndef EXTENSIONUIMANAGER_H
#define EXTENSIONUIMANAGER_H

#include <QObject>
#include <QAction>
#include <QMenu>
#include <QTabWidget>
#include <QList>
#include <QStringList>
#include "messageconsole.h"
#include "sidebar.h"
#include "extensionuimanager.h"
#include "extensionmanager.h"
#include "octprozapp.h"
#include "extensioneventfilter.h"
#include "octproz_devkit.h"
#include "settings.h"
#include "settingsconstants.h"

// Forward declarations to avoid OpenGL dependency chain
class Extension;
class ExtensionManager;
class OCTproZApp;
class ExtensionEventFilter;

class ExtensionUIManager : public QObject
{
	Q_OBJECT
public:
	explicit ExtensionUIManager(QMenu* extensionMenu, QTabWidget* tabWidget,
								MessageConsole* console, Sidebar* sidebar,
								OCTproZApp* app, QObject* parent = nullptr);
	~ExtensionUIManager();

	// Complete extension management
	void initialize(ExtensionManager* extManager);
	void setupExtensionGUI(Extension* extension);
	void cleanupExtensionGUI(Extension* extension);
	void shutdownAllExtensions();

	// Loading/saving extensions
	void autoLoadExtensions();
	QStringList getActiveExtensionNames() const;
	void saveExtensionStates();

	// Get extension actions
	QList<QAction*> getExtensionActions() const { return this->extensionActions; }

public slots:
	void slot_handleExtensionLoading(Extension* extension, bool load);
	void slot_enableRawGrabbing(bool allowed);

private slots:
	void slot_menuExtensions();
	void slot_uncheckExtensionInMenu(Extension* extension);

signals:
	void extensionActivationRequested(const QString& extensionName, bool activate);

private:
	// UI components
	QMenu* extensionMenu;
	QTabWidget* tabWidget;
	MessageConsole* console;
	Sidebar* sidebar;
	OCTproZApp* app;
	ExtensionManager* extManager;
	Settings* appSettings;

	// Extension state
	QList<QAction*> extensionActions;

	// Extension connection handling
	void connectExtensionWithSidebar(Extension* extension);
	void disconnectExtensionFromSidebar(Extension* extension);

	// UI helper methods - add these to resolve the errors
	void addExtensionToSidebar(QTabWidget* tabWidget, Extension* extension);
	void removeExtensionFromSidebar(QTabWidget* tabWidget, Extension* extension);
	void createExtensionWindow(Extension* extension);
	void closeExtensionWindow(Extension* extension);
};
#endif // EXTENSIONUIMANAGER_H
