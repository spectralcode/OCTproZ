/**
* OCTproZ - Optical coherence tomography processing
* License: GNU GPL v3
* Copyright 2019-2025 Miroslav Zabic
*
*
* OCTproZMainWindow: Main application window that manages the UI components
* and user interactions. Handles connections between UI and the application
* logic in OCTproZApp.
*/

#ifndef OCTPROZMAINWINDOW_H
#define OCTPROZMAINWINDOW_H

#include <QMainWindow>
#include <QDockWidget>
#include <QToolBar>
#include <QAction>
#include <QMenu>
#include <QCloseEvent>
#include <QDesktopServices>
#include <QFileDialog>
#include <QSettings>
#include <QVariantMap>
#include "sidebar.h"
#include "messageconsole.h"
#include "plotwindow1d.h"
#include "glwindow2d.h"
#include "glwindow3d.h"
#include "aboutdialog.h"
#include "systemchooser.h"
#include "octprozapp.h"
#include "extensionuimanager.h"
#include "settingsconstants.h"


class OCTproZMainWindow : public QMainWindow
{
	Q_OBJECT

public:
	explicit OCTproZMainWindow(OCTproZApp* app, QWidget* parent = nullptr);
	~OCTproZMainWindow();

	void closeEvent(QCloseEvent* event) override;

	// UI initialization methods
	void initialize();
	void initGui();
	void initMenu();
	void setupExtensions();
	void initActionsAndDocks();
	void prepareDockWidget(QDockWidget*& dock, QWidget* widgetForDock, QAction*& action,
						  const QIcon& icon, QString iconText);

	void setupConnections();
	void loadWindowState();
	void saveWindowStates();
	void loadActiveSystem();
	void saveActiveSystem();

	// Set window title
	void setAppWindowTitle(const QString& title);

public slots:
	// Window management slots
	void slot_enableBscanViewProcessing(bool enable);
	void slot_enableEnFaceViewProcessing(bool enable);
	void slot_enableVolumeViewProcessing(bool enable);
	void slot_closeOpenGLwindows();
	void slot_reopenOpenGLwindows();
	void slot_enableStopAction();
	void slot_easterEgg();
	void slot_takeScreenshots(const QString& savePath, const QString& baseName);

	void slot_onAppSettingsLoaded();

private slots:
	void onProcessingStarted();

	void openLoadSettingsFileDialog();
	void openSaveSettingsFileDialog();
	void openLoadGuiSettingsFileDialog();
	void openSaveGuiSettingsFileDialog();
	void openLoadResamplingCurveDialog();
	void openSystemSettingsDialog();
	void openSelectSystemDialog();
	void openUserManualDialog();


signals:
	void error(QString);
	void info(QString);
	void glBufferTextureSizeBscan(unsigned int width, unsigned int height, unsigned int depth);
	void glBufferTextureSizeEnFaceView(unsigned int width, unsigned int height, unsigned int depth);
	void linesPerBufferChanged(int linesPerBuffer);
	void closeDock2D();
	void reopenDock2D();
	void allowRawGrabbing(bool allowed);

private:
	void loadGuiSettingsFromFile(const QString& settingsFilePath);
	void saveGuiSettingsToFile(const QString& fileName);

	OCTproZApp* app;

	// UI components
	AboutDialog* aboutWindow;
	Sidebar* sidebar;
	MessageConsole* console;
	PlotWindow1D* plot1D;
	GLWindow2D* bscanWindow;
	GLWindow2D* enFaceViewWindow;
	GLWindow3D* volumeWindow;
	SystemChooser* systemChooser;

	// UI helper classes
	ExtensionUIManager* extensionUIManager;

	// Docks
	QDockWidget* dockConsole;
	QDockWidget* dock1D;
	QDockWidget* dock2D;
	QDockWidget* dockEnFaceView;
	QDockWidget* dockVolumeView;

	// Toolbars
	QToolBar* viewToolBar;
	QToolBar* view2DExtrasToolBar;
	QToolBar* controlToolBar;

	// Actions
	QAction* actionStart;
	QAction* actionStop;
	QAction* actionRecord;
	QAction* actionSelectSystem;
	QAction* actionSystemSettings;
	QAction* action1D;
	QAction* action2D;
	QAction* actionEnFaceView;
	QAction* action3D;
	QAction* actionConsole;
	QAction* actionUseSidebarKLinCurve;
	QAction* actionUseCustomKLinCurve;
	QAction* actionSetCustomKLinCurve;

	// State variables
	bool isDock2DClosed;
	bool isDockEnFaceViewClosed;
	bool isDockVolumeViewClosed;
	bool initialWindowStateLoadingDone;

	// Menu
	QMenu* extrasMenu;

	// Settings handling
	SettingsFileManager* appSettings;
	SettingsFileManager* guiSettings;
	QVariantMap mainWindowSettings;
};

#endif // OCTPROZMAINWINDOW_H
