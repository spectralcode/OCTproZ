/**
**  This file is part of OCTproZ.
**  OCTproZ is an open source software for processig of optical
**  coherence tomography (OCT) raw data.
**  Copyright (C) 2019-2021 Miroslav Zabic
**
**  OCTproZ is free software: you can redistribute it and/or modify
**  it under the terms of the GNU General Public License as published by
**  the Free Software Foundation, either version 3 of the License, or
**  (at your option) any later version.
**
**  This program is distributed in the hope that it will be useful,
**  but WITHOUT ANY WARRANTY; without even the implied warranty of
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
**  GNU General Public License for more details.
**
**  You should have received a copy of the GNU General Public License
**  along with this program. If not, see http://www.gnu.org/licenses/.
**
****
** Author:	Miroslav Zabic
** Contact:	zabic
**			at
**			iqo.uni-hannover.de
****
**/

#ifndef OCTPROZ_H
#define OCTPROZ_H

#ifdef _WIN32
	#define WINDOWS_LEAN_AND_MEAN
	//#define NOMINMAX
	#include <windows.h>
	#include "GL/glew.h"
#endif

#include <QMainWindow>
#include <QThread>
#include <qdir.h>
#include <qpluginloader.h>
#include "sidebar.h"
#include "messageconsole.h"
#include "plotwindow1d.h"
#include "glwindow2d.h"
#include "glwindow3d.h"
#include "systemmanager.h"
#include "systemchooser.h"
#include "extensionmanager.h"
#include "extensioneventfilter.h"
#include "processing.h"
#include "octproz_devkit.h"
#include "octalgorithmparameters.h"
#include "aboutdialog.h"

#include "ui_octproz.h"

#define APP_VERSION "1.4.0"
#define APP_VERSION_DATE "6 April 2021"
#define APP_NAME "OCTproZ"


namespace Ui {
class OCTproZ;
}

class OCTproZ : public QMainWindow
{
	Q_OBJECT
	QThread acquisitionThread;
	QThread processingThread;
	QThread notifierThread;

public:
	explicit OCTproZ(QWidget* parent = 0);
	~OCTproZ();

	void closeEvent(QCloseEvent* event);

	void initMenu();
	void initGui();
	void prepareDockWidget(QDockWidget*& dock, QWidget* widgetForDock,  QAction*& action, const QIcon& icon, QString iconText);
	void initActionsAndDocks();
	void loadSystemsAndExtensions();
	void initExtensionsMenu();

public slots:
	void slot_start();
	void slot_stop();
	void slot_record();
	void slot_selectSystem();
	void slot_menuUserManual();
	void slot_menuAbout();
	void slot_menuApplicationSettings();
	void slot_menuSystemSettings();
	void slot_menuExtensions();
	void slot_uncheckExtensionInMenu(Extension* extension);
	void slot_enableStopAction();
	void slot_updateAcquistionParameter(AcquisitionParams newParams);
	void slot_closeOpenGLwindows();
	void slot_reopenOpenGLwindows();
	void slot_storePluginSettings(QString pluginName, QVariantMap settings);
	void slot_prepareGpu2HostForProcessedRecording();
	void slot_resetGpu2HostSettings();
	void slot_recordingDone();
	void slot_enableEnFaceViewProcessing(bool enable);
	void slot_enableBscanViewProcessing(bool enable);
	void slot_enableVolumeViewProcessing(bool enable);
	void slot_easterEgg();
	void slot_useCustomResamplingCurve(bool use);
	void slot_loadCustomResamplingCurve();


private:
	void setSystem(QString systemName);
	void activateSystem(AcquisitionSystem* system);
	void deactivateSystem(AcquisitionSystem* system);
	void reactivateSystem(AcquisitionSystem* system);
	void forceUpdateProcessingParams();
	void saveSettings();

	Ui::OCTproZ *ui;
	AboutDialog* aboutWindow;
	Sidebar* sidebar;
	SystemManager* sysManager;
	SystemChooser* sysChooser;
	AcquisitionSystem* currSystem;
	ExtensionManager* extManager;
	Processing* signalProcessing; 
	OctAlgorithmParameters* octParams;
	Gpu2HostNotifier* processedDataNotifier;

	bool processingInThread;
	bool streamToHostMemorized;
	unsigned int streamingBuffersToSkipMemorized;

	QString currSystemName;
	QList<QString> activatedSystems;
	QList<QAction*> extensionActions;

	QMenu* extrasMenu;
	QToolBar* viewToolBar;
	QToolBar* view2DExtrasToolBar;
	QToolBar* controlToolBar;

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

	MessageConsole* console;

	QDockWidget* dockConsole;
	QDockWidget* dock1D;
	QDockWidget* dock2D;
	QDockWidget* dockEnFaceView;
	QDockWidget* dockVolumeView;

	GLWindow3D* volumeWindow;
	GLWindow2D* bscanWindow;
	GLWindow2D* enFaceViewWindow;
	PlotWindow1D* plot1D;

	bool isDock2DClosed;
	bool isDockEnFaceViewClosed;
	bool isDockVolumeViewClosed;

	QOpenGLContext *context;

signals:
	void start();
	void stop();
	void allowRawGrabbing(bool allowed);
	void record();
	void enableRecording(bool enableRawRecording, bool enableProcessedRecording);
	void pluginSettingsRequest();
	void newSystemSelected();
	void newSystem(AcquisitionSystem*);
	void error(QString);
	void info(QString);
	void glBufferTextureSizeBscan(unsigned int width, unsigned int height, unsigned int depth);
	void glBufferTextureSizeEnFaceView(unsigned int width, unsigned int height, unsigned int depth);
	void linesPerBufferChanged(int linesPerBuffer);
	void closeDock2D();
	void reopenDock2D();
	void loadPluginSettings(QVariantMap);


};

#endif // OCTPROZ_H
