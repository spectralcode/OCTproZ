/**
**  This file is part of OCTproZ.
**  OCTproZ is an open source software for processig of optical
**  coherence tomography (OCT) raw data.
**  Copyright (C) 2019-2020 Miroslav Zabic
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

#include "octproz.h"


OCTproZ::OCTproZ(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::OCTproZ)
{
	///qRegisterMetaType is needed to enabel Qt::QueuedConnection for signal slot communication with "AcquisitionParams"
	qRegisterMetaType<AcquisitionParams >("AcquisitionParams");
	qRegisterMetaType<size_t >("size_t");

	qApp->setApplicationVersion(APP_VERSION);
	qApp->setApplicationName(APP_NAME);

	this->console = new MessageConsole(this);
	this->console->setObjectName("Message Console");
	this->dockConsole = new QDockWidget((tr("Message Console")), this);

	Settings* settings = Settings::getInstance();
	connect(settings, &Settings::info, this->console, &MessageConsole::slot_displayInfo);
	connect(settings, &Settings::error, this->console, &MessageConsole::slot_displayError);

	this->aboutWindow = new AboutDialog(this);
	connect(this->aboutWindow, &AboutDialog::easterEgg, this, &OCTproZ::slot_easterEgg);

	this->sysManager = new SystemManager();
	this->sysChooser = new SystemChooser();
	this->currSystem = nullptr;
	this->currSystemName = "";
	this->octParams = OctAlgorithmParameters::getInstance();

	this->extManager = new ExtensionManager();
	this->plot1D = new PlotWindow1D(this);
	connect(this, &OCTproZ::allowRawGrabbing, this->plot1D, &PlotWindow1D::slot_enableRawGrabbing);
	connect(this->plot1D, &PlotWindow1D::info, this->console, &MessageConsole::slot_displayInfo);
	connect(this->plot1D, &PlotWindow1D::error, this->console, &MessageConsole::slot_displayError);
	connect(this, &OCTproZ::linesPerBufferChanged, this->plot1D, &PlotWindow1D::slot_changeLinesPerBuffer);

	this->dock1D = new QDockWidget(tr("1D"), this);
	this->dock1D->setObjectName("1D");

	this->bscanWindow = new GLWindow2D(this);
	this->bscanWindow->setMarkerOrigin(TOP);
	this->enFaceViewWindow = new GLWindow2D(this);
	this->enFaceViewWindow->setMarkerOrigin(LEFT);

	connect(this->bscanWindow, &GLWindow2D::currentFrameNr, this->enFaceViewWindow, &GLWindow2D::slot_setMarkerPosition);
	connect(this->enFaceViewWindow, &GLWindow2D::currentFrameNr, this->bscanWindow, &GLWindow2D::slot_setMarkerPosition);
	connect(this->bscanWindow, &GLWindow2D::dialogAboutToOpen, this, &OCTproZ::slot_closeOpenGLwindows); //GL windows need to be closed to avoid linux bug where QFileDialog is not usable when a GL window is opend in background
	connect(this->enFaceViewWindow, &GLWindow2D::dialogAboutToOpen, this, &OCTproZ::slot_closeOpenGLwindows); //GL windows need to be closed to avoid linux bug where QFileDialog is not usable when a GL window is opend in background
	connect(this->bscanWindow, &GLWindow2D::dialogClosed, this, &OCTproZ::slot_reopenOpenGLwindows);
	connect(this->enFaceViewWindow, &GLWindow2D::dialogClosed, this, &OCTproZ::slot_reopenOpenGLwindows);

	connect(this, &OCTproZ::glBufferTextureSizeBscan, this->bscanWindow, &GLWindow2D::slot_changeBufferAndTextureSize);
	this->dock2D = new QDockWidget(tr("2D - B-scan"), this);
	this->dock2D->setObjectName("2D - B-scan");
	//this->dock2D->setFeatures(QDockWidget::DockWidgetMovable|QDockWidget::DockWidgetClosable); //make dock not floatable
	connect(this->dock2D, &QDockWidget::visibilityChanged, this, &OCTproZ::slot_enableBscanViewProcessing);

	connect(this, &OCTproZ::glBufferTextureSizeEnFaceView, this->enFaceViewWindow, &GLWindow2D::slot_changeBufferAndTextureSize);
	this->dockEnFaceView = new QDockWidget(tr("2D - En Face View"), this);
	this->dockEnFaceView->setObjectName("2D - En Face View");
	connect(this->dockEnFaceView, &QDockWidget::visibilityChanged, this, &OCTproZ::slot_enableEnFaceViewProcessing);

	this->volumeWindow = new GLWindow3D(this);
	this->dockVolumeView = new QDockWidget(tr("3D - Volume"), this);
	this->dockVolumeView->setObjectName("3D - Volume");
	this->dockVolumeView->setFeatures(QDockWidget::DockWidgetClosable); //make dock not floatable, and not movable
	connect(this, &OCTproZ::glBufferTextureSizeBscan, this->volumeWindow, &GLWindow3D::slot_changeBufferAndTextureSize);
	connect(this->dockVolumeView, &QDockWidget::visibilityChanged, this, &OCTproZ::slot_enableVolumeViewProcessing);

	this->sidebar = new Sidebar(this);
	this->sidebar->setObjectName("Sidebar");
	connect(this->sidebar, &Sidebar::info, this->console, &MessageConsole::slot_displayInfo);
	connect(this->sidebar, &Sidebar::error, this->console, &MessageConsole::slot_displayError);
	connect(this->sidebar, &Sidebar::dialogAboutToOpen, this, &OCTproZ::slot_closeOpenGLwindows); //GL windows need to be closed to avoid linux bug where QFileDialog is not usable when a GL window is opend in background
	connect(this->sidebar, &Sidebar::dialogClosed, this, &OCTproZ::slot_reopenOpenGLwindows);

	this->processingInThread = false;
	this->signalProcessing = new Processing();
	#if defined(Q_OS_WIN)
		this->signalProcessing->moveToThread(&processingThread);
		this->processingInThread = true;
		connect(&processingThread, &QThread::finished, this->signalProcessing, &Processing::deleteLater);
	#elif defined(Q_OS_LINUX)
		//this->signalProcessing->moveToThread(&processingThread);
		//todo: fix linux bug: opengl window seems to be laggy on ubuntu test system if signalProcessing is moved to thread -> check visual profiler
		this->processingInThread = false;
	#endif

	connect(this, &OCTproZ::enableRecording, this->signalProcessing, &Processing::slot_enableRecording);
	connect(this->signalProcessing, &Processing::info, this->console, &MessageConsole::slot_displayInfo);
	connect(this->signalProcessing, &Processing::error, this->console, &MessageConsole::slot_displayError);
	connect(this->signalProcessing, &Processing::initializationDone, this, &OCTproZ::slot_enableStopAction);
	connect(this->signalProcessing, &Processing::streamingBufferEnabled, this->plot1D, &PlotWindow1D::slot_enableProcessedGrabbing);
	connect(this->signalProcessing, &Processing::rawData, this->plot1D, &PlotWindow1D::slot_plotRawData);
	connect(this->signalProcessing, &Processing::processedRecordDone, this, &OCTproZ::slot_resetGpu2HostSettings);
	connect(this->signalProcessing, &Processing::processedRecordDone, this, &OCTproZ::slot_recordingDone);
	connect(this->signalProcessing, &Processing::rawRecordDone, this, &OCTproZ::slot_recordingDone);
	//B-scan window connections:
	connect(this->bscanWindow->getControlPanel(), &ControlPanel2D::displayFrameSettingsChanged, this->signalProcessing, &Processing::slot_updateDisplayedBscanFrame);
	connect(this->bscanWindow, &GLWindow2D::registerBufferCudaGL, this->signalProcessing, &Processing::slot_registerBscanOpenGLbufferWithCuda);
	//En face view window connections:
	connect(this->enFaceViewWindow->getControlPanel(), &ControlPanel2D::displayFrameSettingsChanged, this->signalProcessing, &Processing::slot_updateDisplayedEnFaceFrame);
	connect(this->enFaceViewWindow, &GLWindow2D::registerBufferCudaGL, this->signalProcessing, &Processing::slot_registerEnFaceViewOpenGLbufferWithCuda);
	//Volume window connections:
	connect(this->volumeWindow, &GLWindow3D::registerBufferCudaGL, this->signalProcessing, &Processing::slot_registerVolumeViewOpenGLbufferWithCuda);
	//Processing connections:
	connect(this->signalProcessing, &Processing::updateInfoBox, this->sidebar, &Sidebar::slot_updateInfoBox);
	connect(this->signalProcessing, &Processing::initOpenGL, this->bscanWindow, &GLWindow2D::slot_initProcessingThreadOpenGL);
	if(!this->processingInThread){
		connect(this->signalProcessing, &Processing::initOpenGL, this->enFaceViewWindow, &GLWindow2D::slot_initProcessingThreadOpenGL); //due to opengl context sharing this connect is not necessary
	}
	connect(this->signalProcessing, &Processing::initOpenGLenFaceView, this->enFaceViewWindow, &GLWindow2D::slot_registerGLbufferWithCuda);
	connect(this->signalProcessing, &Processing::initOpenGLenFaceView, this->volumeWindow, &GLWindow3D::slot_registerGLbufferWithCuda);
	processingThread.start();

	this->initGui();
	this->loadSystemsAndExtensions();

	this->processedDataNotifier = Gpu2HostNotifier::getInstance();
	this->processedDataNotifier->moveToThread(&notifierThread);
	connect(this->processedDataNotifier, &Gpu2HostNotifier::newGpuDataAvailible, this->plot1D, &PlotWindow1D::slot_plotProcessedData);
	connect(&notifierThread, &QThread::finished, this->processedDataNotifier, &Gpu2HostNotifier::deleteLater);
	notifierThread.start();

	//set position of main window near upper left corner of screen
	QRect screenGeometry = QApplication::desktop()->availableGeometry();
	int x = (screenGeometry.width()) / 10;
	int y = (screenGeometry.height()) / 10;
	this->move(x, y);

	//restore main window position and size (if application was already open once)
	this->restoreGeometry(settings->mainWindowSettings.value(MAIN_GEOMETRY).toByteArray());
	//this->restoreState(settings->mainWindowSettings.value(MAIN_STATE).toByteArray()); //this crashes the application under certain circumstances

	//memorize stream2host setings
	this->streamToHostMemorized = this->octParams->streamToHost;
	this->streamingBuffersToSkipMemorized = octParams->streamingBuffersToSkip;
}

OCTproZ::~OCTproZ(){
	qDebug() << "OCTproZ destructor";
	processingThread.quit();
	processingThread.wait();
	notifierThread.quit();
	notifierThread.wait();
	acquisitionThread.quit();
	acquisitionThread.wait();

	delete ui;
	delete this->sysManager;
	delete this->sysChooser;
	delete this->extManager;
	delete this->console;
	delete this->dockConsole;
	delete this->bscanWindow;
	delete this->enFaceViewWindow;
	delete this->dock2D;
	delete this->dockEnFaceView;
	delete this->dockVolumeView;
}

void OCTproZ::closeEvent(QCloseEvent* event) {
	//save main window position and size
	(Settings::getInstance())->mainWindowSettings.insert(MAIN_GEOMETRY, this->saveGeometry());
	//(Settings::getInstance())->mainWindowSettings.insert(MAIN_STATE, this->saveState());

	//Stop acquisition if it is running
	if (this->currSystem != nullptr) {
		if (this->currSystem->acqusitionRunning) {
			this->slot_stop();
			QCoreApplication::processEvents(); //process events to ensure that acquisition is not running
			QThread::msleep(1000); //provide some time to let gpu computation finish
		}
	}
	event->accept();
}

void OCTproZ::initActionsAndDocks() {
	//create tool bars
	this->viewToolBar = this->addToolBar(tr("View Toolbar"));
	this->viewToolBar->setObjectName("View Toolbar");
	this->view2DExtrasToolBar = this->addToolBar(tr("2D View Tools"));
	this->view2DExtrasToolBar->setObjectName("2D View Tools");
	//this->viewToolBar->setIconSize(QSize(64,64));
	this->controlToolBar = this->addToolBar(tr("Control Toolbar"));
	this->controlToolBar->setObjectName("Control Toolbar");
	this->controlToolBar->setVisible(false);

	//init 2d view tools toolbar
	QAction* bscanMarkerAction = this->bscanWindow->getMarkerAction();
	QAction* enfaceMarkerAction = this->enFaceViewWindow->getMarkerAction();
	bscanMarkerAction->setIcon(QIcon(":/icons/octproz_bscanmarker_icon.png"));
	bscanMarkerAction->setToolTip(tr("Display orthogonal marker in B-scan"));
	enfaceMarkerAction->setIcon(QIcon(":/icons/octproz_enfacemarker_icon.png"));
	enfaceMarkerAction->setToolTip(tr("Display orthogonal marker in en face view"));
	this->view2DExtrasToolBar->addAction(bscanMarkerAction);
	this->view2DExtrasToolBar->addAction(enfaceMarkerAction);

	this->setCentralWidget(this->sidebar->getDock());

	this->dockConsole->setFloating(false);
	this->dockConsole->setVisible(true);
	this->dockConsole->setTitleBarWidget(new QWidget()); //this removes title bar

	this->actionStart = new QAction("Start", this);
	this->actionStart->setIcon(QIcon(":/icons/octproz_play_icon.png"));
	this->controlToolBar->addAction(actionStart);

	this->actionStop = new QAction("Stop", this);
	this->actionStop->setIcon(QIcon(":/icons/octproz_stop_icon.png"));
	this->controlToolBar->addAction(actionStop);

	this->actionRecord = new QAction("Rec", this);
	this->actionRecord->setIcon(QIcon(":/icons/octproz_record_icon.png"));
	this->controlToolBar->addAction(actionRecord);

	this->actionSelectSystem = new QAction("Open System", this);
	this->actionSelectSystem->setIcon(QIcon(":/icons/octproz_connect_icon.png"));
	this->controlToolBar->addAction(actionSelectSystem);

	this->actionSystemSettings = new QAction("System Settings", this);
	this->actionSystemSettings->setIcon(QIcon(":/icons/octproz_settings_icon.png"));
	this->actionSystemSettings->setStatusTip(tr("Settings of the currently loaded OCT system"));
	this->controlToolBar->addAction(actionSystemSettings);

	this->prepareDockWidget(this->dock1D, this->plot1D, this->action1D, QIcon(":/icons/octproz_rawsignal_icon.png"), "1D");
	this->prepareDockWidget(this->dock2D, this->bscanWindow, this->action2D, QIcon(":/icons/octproz_bscan_icon.png"), "2D - B-scan");
	this->dock2D->setFloating(false);
	this->dock2D->setVisible(true);

	this->prepareDockWidget(this->dockEnFaceView, this->enFaceViewWindow, this->actionEnFaceView, QIcon(":/icons/octproz_enface_icon.png"), "2D - En Face View");
	this->dockEnFaceView->setFloating(false);
	this->dockEnFaceView->setVisible(true);
	//this->dockEnFaceView->setLayoutDirection(Qt::Horizontal);

	this->prepareDockWidget(this->dockVolumeView, this->volumeWindow, this->action3D, QIcon(":/icons/octproz_volume_icon.png"), "3D - Volume");
	this->dockVolumeView->setFloating(false);
	this->dockVolumeView->setVisible(false);
	this->dockVolumeView->setMinimumWidth(320);

	this->prepareDockWidget(this->dockConsole, this->console, this->actionConsole, QIcon(":/icons/octproz_log_icon.png"), "Console");
	this->addDockWidget(Qt::BottomDockWidgetArea, this->dockConsole);
	this->dockConsole->setFloating(false);
	this->dockConsole->setVisible(true);
	this->dockConsole->setTitleBarWidget(new QWidget()); //this removes title bar

	connect(this->actionStart, &QAction::triggered, this, &OCTproZ::slot_start);
	connect(this->actionStop, &QAction::triggered, this, &OCTproZ::slot_stop);
	connect(this->actionRecord, &QAction::triggered, this, &OCTproZ::slot_record);
	connect(this->actionSelectSystem, &QAction::triggered, this, &OCTproZ::slot_selectSystem);
	connect(this->actionSystemSettings, &QAction::triggered, this, &OCTproZ::slot_menuSystemSettings);

	this->actionStart->setEnabled(false);
	this->actionStop->setEnabled(false);
	this->actionRecord->setEnabled(false);
}

void OCTproZ::initGui() {
	ui->setupUi(this);
	this->initActionsAndDocks(); //initActionsAndDocks() must be called before initMenu()
	this->sidebar->init(this->actionStart, this->actionStop, this->actionRecord, this->actionSelectSystem, this->actionSystemSettings);
	this->initMenu();
	this->forceUpdateProcessingParams();

	//Message Console connects
	connect(this, &OCTproZ::info, this->console, &MessageConsole::slot_displayInfo);
	connect(this, &OCTproZ::error, this->console, &MessageConsole::slot_displayError);
	connect(this->bscanWindow, &GLWindow2D::info, this->console, &MessageConsole::slot_displayInfo);
	connect(this->bscanWindow, &GLWindow2D::error, this->console, &MessageConsole::slot_displayError);
	connect(this->enFaceViewWindow, &GLWindow2D::info, this->console, &MessageConsole::slot_displayInfo);
	connect(this->enFaceViewWindow, &GLWindow2D::error, this->console, &MessageConsole::slot_displayError);
	connect(this->volumeWindow, &GLWindow3D::info, this->console, &MessageConsole::slot_displayInfo);
	connect(this->volumeWindow, &GLWindow3D::error, this->console, &MessageConsole::slot_displayError);

	//Connects to bypass Linux/Qt bugs
	connect(this, &OCTproZ::closeDock2D, this, &OCTproZ::slot_closeOpenGLwindows);
	connect(this, &OCTproZ::reopenDock2D, this, &OCTproZ::slot_reopenOpenGLwindows);

	//for debugging:
	qDebug() << "Main Thread ID: " << QThread::currentThreadId();
}

void OCTproZ::prepareDockWidget(QDockWidget*& dock, QWidget* widgetForDock, QAction*& action, const QIcon& icon, QString iconText) {
	dock->setWidget(widgetForDock);
	this->addDockWidget(Qt::RightDockWidgetArea, dock, Qt::Horizontal);
	dock->setVisible(false);
	dock->setFloating(true);
	action = dock->toggleViewAction();
	if(!icon.isNull()){
		action->setIcon(icon);
	}
	action->setIconText(iconText);
	this->viewToolBar->addAction(action);
}

void OCTproZ::initMenu() {
	//file menu
	this->actionSelectSystem->setShortcut(QKeySequence::Open);
	this->actionSystemSettings->setShortcut(QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_O));
	QMenu *fileMenu = menuBar()->addMenu(tr("&File"));
	fileMenu->addAction(this->actionSelectSystem);
	fileMenu->addAction(this->actionSystemSettings);
	fileMenu->addSeparator();
	const QIcon exitIcon = QIcon(":/icons/octproz_close_icon.png");
	QAction *exitAct = fileMenu->addAction(exitIcon, tr("E&xit"), this, &QWidget::close);
	exitAct->setShortcuts(QKeySequence::Quit);
	exitAct->setStatusTip(tr("Close OCTproZ"));

	//view menu
	QMenu* viewMenu = this->menuBar()->addMenu(tr("&View"));
	//get view toolbar
	QAction* viewToolBarAction = this->viewToolBar->toggleViewAction();
	viewToolBarAction->setText(tr("Show View Toolbar"));
	//get 2d tools toolbar
	QAction* view2DExtrasToolBarAction = this->view2DExtrasToolBar->toggleViewAction();
	view2DExtrasToolBarAction->setText(tr("Show 2D View Tools"));
	//get control toolbar
	QAction* controlToolBarAction = this->controlToolBar->toggleViewAction();
	controlToolBarAction->setText(tr("Show Control Toolbar"));
	//add toolbar view actions to view menu
	viewMenu->addActions({ viewToolBarAction, view2DExtrasToolBarAction, controlToolBarAction, this->action1D, this->action2D, this->actionEnFaceView, this->action3D, this->actionConsole });

	//extras menu
	this->extrasMenu = this->menuBar()->addMenu(tr("&Extras"));
	QMenu* klinMenu = this->extrasMenu->addMenu(tr("&Resampling curve for k-linearization"));
	klinMenu->setToolTipsVisible(true);
	klinMenu->setStatusTip(tr("Settings for k-linearization resampling curve"));
	klinMenu->setIcon(QIcon(":/icons/octproz_klincurve_icon.png"));
	QActionGroup* customKlinCurveGroup = new QActionGroup(this);
	this->actionSetCustomKLinCurve = new QAction(tr("&Load custom curve from file..."), this);
	connect(this->actionSetCustomKLinCurve, &QAction::triggered, this, &OCTproZ::slot_loadCustomResamplingCurve);
	this->actionUseSidebarKLinCurve = new QAction(tr("Use &polynomial curve from sidebar"), this);
	this->actionUseCustomKLinCurve = new QAction(tr("Use &custom curve from file"), this);
	this->actionUseSidebarKLinCurve ->setCheckable(true);
	this->actionUseCustomKLinCurve->setCheckable(true);
	this->actionUseSidebarKLinCurve->setActionGroup(customKlinCurveGroup);
	this->actionUseCustomKLinCurve->setActionGroup(customKlinCurveGroup);
	this->actionUseSidebarKLinCurve->setChecked(true);
	connect(this->actionUseCustomKLinCurve, &QAction::toggled, this, &OCTproZ::slot_useCustomResamplingCurve);
	klinMenu->addAction(this->actionUseSidebarKLinCurve);
	klinMenu->addAction(this->actionUseCustomKLinCurve);
	klinMenu->addSeparator();
	klinMenu->addAction(this->actionSetCustomKLinCurve);

	//help menu
	QMenu *helpMenu = this->menuBar()->addMenu(tr("&Help"));
	//user manual
	QAction *manualAct = helpMenu->addAction(tr("&User Manual"), this, &OCTproZ::slot_menuUserManual);
	manualAct->setStatusTip(tr("OCTproZ user manual"));
	manualAct->setIcon(QIcon(":/icons/octproz_manual_icon.png"));
	manualAct->setShortcut(QKeySequence::HelpContents);
	//about dialog
	QAction *aboutAct = helpMenu->addAction(tr("&About"), this, &OCTproZ::slot_menuAbout);
	aboutAct->setStatusTip(tr("About OCTproZ"));
	aboutAct->setIcon(QIcon(":/icons/octproz_info_icon.png"));
}

void OCTproZ::loadSystemsAndExtensions() {
	QDir pluginsDir = QDir(qApp->applicationDirPath());

	//check if plugins dir exists. if it does not exist change to the share_dev directory. this makes software development easier as plugins can be copied to the share_dev during the build process
	bool pluginsDirExists = pluginsDir.cd("plugins");
	if(!pluginsDirExists){
		#if defined(Q_OS_WIN)
		if (pluginsDir.dirName().toLower() == "debug" || pluginsDir.dirName().toLower() == "release") {
			pluginsDir.cdUp();
		}
		#endif

		pluginsDir.cdUp();
		pluginsDir.cd("octproz_share_dev");
		pluginsDir.cd("plugins");

		//change directory if debug or release folder exist
		#ifdef QT_DEBUG
			pluginsDir.cd("debug");
		#else
			pluginsDir.cd("release");
		#endif
	}

	for(auto fileName : pluginsDir.entryList(QDir::Files)) {
		QPluginLoader loader(pluginsDir.absoluteFilePath(fileName));
		QObject *plugin = loader.instance(); //todo: figure out why qobject_cast<Plugin*>(loader.instance()) does not work and fix it
		if (plugin) {
			Plugin* actualPlugin = (Plugin*)(plugin);
			enum PLUGIN_TYPE type = actualPlugin->getType();
			connect(actualPlugin, &Plugin::setKLinCoeffsRequest, this->sidebar, &Sidebar::slot_setKLinCoeffs); //Experimental! May be removed in future versions.
			connect(actualPlugin, &Plugin::setDispCompCoeffsRequest, this->sidebar, &Sidebar::slot_setDispCompCoeffs); //Experimental! May be removed in future versions.
			connect(this->sidebar, &Sidebar::klinCoeffs, actualPlugin, &Plugin::setKLinCoeffsRequestAccepted); //Experimental! May be removed in future versions.
			connect(this->sidebar, &Sidebar::dispCompCoeffs, actualPlugin, &Plugin::setDispCompCoeffsRequestAccepted); //Experimental! May be removed in future versions.
			connect(actualPlugin, &Plugin::startProcessingRequest, this, &OCTproZ::slot_start); //Experimental! May be removed in future versions.
			connect(actualPlugin, &Plugin::stopProcessingRequest, this, &OCTproZ::slot_stop); //Experimental! May be removed in future versions.
			switch (type) {
				case SYSTEM:{
					this->sysManager->addSystem(qobject_cast<AcquisitionSystem*>(plugin));
					break;
				}
				case EXTENSION:{
					Extension* extension = qobject_cast<Extension*>(plugin);
					this->extManager->addExtension(extension);
					if(extension->getDisplayStyle() == SEPARATE_WINDOW){
						//init extension window
						QWidget* extensionWidget = extension->getWidget();
						extensionWidget->setParent(this); //this is necessary to automatically close all open extension windows when main application is closed
						extensionWidget->setWindowFlags(Qt::Window);
						extensionWidget->setAttribute(Qt::WA_DeleteOnClose, false);
						extensionWidget->setWindowTitle(extension->getName());
						ExtensionEventFilter* extensionCloseSignalForwarder = new ExtensionEventFilter(extensionWidget); //setting extensionWidget as parent causes ExtensionEventFilter object to be destroyed when extensionWidget gets destroyed
						extensionCloseSignalForwarder->setExtension(extension);
						extensionWidget->installEventFilter(extensionCloseSignalForwarder);
						connect(extensionCloseSignalForwarder, &ExtensionEventFilter::extensionWidgetClosed, this, &OCTproZ::slot_uncheckExtensionInMenu); //this connection is used to automatically uncheck extension in menu if user closes a separate window extension by clicking on x
					}
					break;
				}
				default: {
					emit error(tr("Could not load Plugin"));
				}
			}
		}
	}
	if (this->extManager->getExtensions().size() > 0) {
		this->initExtensionsMenu();
	}
}

void OCTproZ::initExtensionsMenu() {
	QMenu* extensionMenu = this->extrasMenu->addMenu(tr("&Extensions"));
	extensionMenu->setIcon(QIcon(":/icons/octproz_extensions_icon.png"));

	auto extensionNames = this->extManager->getExtensionNames();
	foreach(QString extensionName, extensionNames) {
		QAction* extAction = extensionMenu->addAction(extensionName, this, &OCTproZ::slot_menuExtensions);
		this->extensionActions.append(extAction);
		extAction->setCheckable(true);
		extAction->setChecked(false);
		Extension* extension = this->extManager->getExtensionByName(extensionName);
		QString extensionToolTip = extension == nullptr ? "" : extension->getToolTip(); //todo: error handling if extension is nullptr
		extAction->setStatusTip(extensionToolTip);
	}
}

void OCTproZ::slot_start() {
	//(re-)init resampling curve, dispersion curve, window curve, streaming //todo: carefully check if this is really necessary here
	this->forceUpdateProcessingParams();
	
	//save current parameters to hdd
	this->sidebar->saveSettings();
	
	//disable start/stop buttons
	this->actionStart->setEnabled(false);
	this->actionStop->setEnabled(false); //stop button will be enabled again as soon as processing initialization is done

	//enalbe 1d plotting and extensions
	emit allowRawGrabbing(true);

	//emit start signal to activate acquisition of current AcquisitionSystem 
	emit start();

	//for debugging purposes: read out thread affinity of current thread
	qDebug() << "Main Thread ID start emit: " << QThread::currentThreadId();
}

void OCTproZ::slot_stop() {
	//adjust start stop buttons
	this->actionStart->setEnabled(true);
	this->actionStop->setEnabled(false);

	//disable 1d plotting and acquisition data grabbing by extensions
	emit allowRawGrabbing(false);
	QApplication::processEvents();

	//emit stop signal to stop acquisition system if it is still running
	if (this->currSystem != nullptr) {
		if(this->currSystem->acqusitionRunning){
			emit stop();
			QApplication::processEvents();
			this->currSystem->acqusitionRunning = false; //todo: think about whether OCTproZ should really set the AcquisitionRunning flag to false here, or whether only the acquisition system itself should be responsible for setting this flag to false
		}
	}
}

void OCTproZ::slot_record() {
	//check if system is open
	if (this->currSystem == nullptr) {
		emit error(tr("Nothing to record, no system opened."));
		return;
	}

	//generate timestamp for file name of recording and if needed for the file name of the corresponding meta info
	QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hh-mm-ss-zzz");
	Settings::getInstance()->setTimestamp(timestamp);

	bool enableRawRecording = false;
	bool enableProcessedRecording = false;

	//get user defined rec name
	QString recName = Settings::getInstance()->recordSettings.value(REC_NAME).toString();
	if (recName != "") {
		recName = "_" + recName;
	}

	RECORD_MODE recMode = (RECORD_MODE)Settings::getInstance()->recordSettings.value(REC_MODE).toUInt();
	//check if record mode "raw" is activated and start processing if necessary
	if (recMode == RECORD_MODE::RAW || recMode == RECORD_MODE::ALL) {
		enableRawRecording = true;
	}

	//check if record mode "processed" is activated and start processing if necessary
	if (recMode == RECORD_MODE::PROCESSED || recMode == RECORD_MODE::ALL) {
		enableProcessedRecording = true;
		this->slot_prepareGpu2HostForProcessedRecording();
	}

	if (enableRawRecording || enableProcessedRecording) {
		this->sidebar->enableRecordTab(false);
		emit this->enableRecording(enableRawRecording, enableProcessedRecording);
		if (!this->currSystem->acqusitionRunning) {
			this->slot_start();
		}
	}

	//if record mode "snaptshot" is activated save snapshot ind record directory
	if (Settings::getInstance()->recordSettings.value(REC_MODE).toUInt() == RECORD_MODE::SNAPSHOT) {
		QString savePath = Settings::getInstance()->recordSettings.value(REC_PATH).toString();
		if(this->bscanWindow->isVisible()) {
			QString fileName = timestamp + recName + "_" + "bscan_snapshot" + ".png";
			this->bscanWindow->slot_saveScreenshot(savePath, fileName);
		}
		if(this->enFaceViewWindow->isVisible()) {
			QString fileName = timestamp + recName + "_" + "enfaceview_snapshot" + ".png";
			this->enFaceViewWindow->slot_saveScreenshot(savePath, fileName);
		}
		if(this->volumeWindow->isVisible()) {
			QString fileName = timestamp + recName + "_" + "volume_snapshot" + ".png";
			this->volumeWindow->slot_saveScreenshot(savePath, fileName);
		}
	}

	//check if meta information should be saved
	if (Settings::getInstance()->recordSettings.value(REC_META).toBool() == true) {
		QString metaFileName = Settings::getInstance()->recordSettings.value(REC_PATH).toString() + "/" + Settings::getInstance()->getTimestamp() + recName + "_meta.txt";
		Settings::getInstance()->copySettingsFile(metaFileName);
	}

}

void OCTproZ::slot_selectSystem() {
	QString selectedSystem = this->sysChooser->selectSystem(this->sysManager->getSystemNames());
	this->setSystem(selectedSystem);
}

void OCTproZ::slot_menuUserManual() {
	QDesktopServices::openUrl(QUrl("file:///" + QCoreApplication::applicationDirPath() + "/docs/index.html"));
}

void OCTproZ::slot_menuAbout() {
	this->aboutWindow->show();
}

void OCTproZ::slot_menuApplicationSettings() {
	//todo: application settings dialog
}

void OCTproZ::slot_menuSystemSettings() {
	if (this->currSystem != nullptr) {
		emit closeDock2D(); //GL window needs to be closed to avoid linux bug where QFileDialog is not usable when GL window is opend in background
		emit pluginSettingsRequest();
	}else{
		emit error(tr("No system opened!"));
	}
}

void OCTproZ::slot_menuExtensions() {
	//todo: refactor this method. Think of a better way to asociate corresponding extensions and qactions. Maybe store extensions and qactins in a qmap or in a qlist with qpairs.
	QAction* currAction = qobject_cast<QAction*>(sender());
	if(currAction == 0){return;}
	QString extensionName = currAction->text();
	Extension* extension = this->extManager->getExtensionByName(extensionName); //this just works if extension names are unique
	extension->settingsLoaded(Settings::getInstance()->getStoredSystemSettings(extensionName)); //todo: use only signal slot to interact with extension (do not call methods directly like in this line) and move extensions to threads. Similar to AcquisitionSystems, see: implementation of activateSystem(AcquisitionSystem* system)
	if(extension == nullptr){
		emit error(tr("No Extension with name ") + extensionName + tr(" exists."));
		return;
	}
	QWidget* extensionWidget = extension->getWidget();
	QTabWidget* tabWidget = this->sidebar->getUi().tabWidget;

	//if extension is deactivated (i. e. not visible as tab within sidebar and not visible as separate window) and user checked the extension in the menu then activate it.
	if ((extension->getDisplayStyle() == SIDEBAR_TAB && tabWidget->indexOf(extensionWidget) == -1) || (extension->getDisplayStyle() == SEPARATE_WINDOW && !extensionWidget->isVisible())) {
			if(currAction->isChecked()){
				if(extension->getDisplayStyle() == SIDEBAR_TAB){
					tabWidget->addTab(extensionWidget, extensionName);
				} else if( extension->getDisplayStyle() == SEPARATE_WINDOW){
					extensionWidget->setWindowFlag(Qt::WindowStaysOnTopHint);
					extensionWidget->show();
				}
				connect(extension, &Extension::info, this->console, &MessageConsole::slot_displayInfo);
				connect(extension, &Extension::error, this->console, &MessageConsole::slot_displayError);
				connect(extension, &Extension::storeSettings, this, &OCTproZ::slot_storePluginSettings);
				extension->activateExtension(); //todo: do not call extension methods directly, use signal slot (or invokeMethod -> see below) and run extension in separate thread
				//QMetaObject::invokeMethod(extension, "activateExtension", Qt::QueuedConnection); //todo: move activateExtension method to "slots" in extensions.h in devkit! move extension in separate thread
				connect(this, &OCTproZ::allowRawGrabbing, extension, &Extension::enableRawDataGrabbing);
				connect(this->signalProcessing, &Processing::streamingBufferEnabled, extension, &Extension::enableProcessedDataGrabbing);
				connect(this->processedDataNotifier, &Gpu2HostNotifier::newGpuDataAvailible, extension, &Extension::processedDataReceived);
				connect(this->signalProcessing, &Processing::rawData, extension, &Extension::rawDataReceived);
			}
	}
	//else (i.e. extension is visible within sidebar or as separate window) deactivate extension if user unchecked extension in menu
	else {
			if(!currAction->isChecked()) {
				if(extension->getDisplayStyle() == SIDEBAR_TAB){
					int index = tabWidget->indexOf(extensionWidget);
					tabWidget->removeTab(index);

					extension->deactivateExtension();
					disconnect(extension, &Extension::info, this->console, &MessageConsole::slot_displayInfo);
					disconnect(extension, &Extension::error, this->console, &MessageConsole::slot_displayError);
					disconnect(extension, &Extension::storeSettings, this, &OCTproZ::slot_storePluginSettings);
					disconnect(this, &OCTproZ::allowRawGrabbing, extension, &Extension::enableRawDataGrabbing);
					disconnect(this->signalProcessing, &Processing::streamingBufferEnabled, extension, &Extension::enableProcessedDataGrabbing);
					disconnect(this->processedDataNotifier, &Gpu2HostNotifier::newGpuDataAvailible, extension, &Extension::processedDataReceived);
					disconnect(this->signalProcessing, &Processing::rawData, extension, &Extension::rawDataReceived);
				} else if( extension->getDisplayStyle() == SEPARATE_WINDOW){
					extensionWidget->close();
				}
			}
	}
}

void OCTproZ::slot_uncheckExtensionInMenu(Extension* extension) {
	//get corresponding action to closed extension
	QString extensionName = extension->getName();
	QAction* currAction = nullptr;
	foreach(auto action, this->extensionActions) {
		if(action->text() == extensionName){
			currAction = action;
			break;
		}
	}

	//uncheck action in menu
	currAction->setChecked(false);

	//disconnect signal slots from closed extension
	extension->deactivateExtension();
	disconnect(extension, &Extension::info, this->console, &MessageConsole::slot_displayInfo);
	disconnect(extension, &Extension::error, this->console, &MessageConsole::slot_displayError);
	disconnect(extension, &Extension::storeSettings, this, &OCTproZ::slot_storePluginSettings);
	disconnect(this, &OCTproZ::allowRawGrabbing, extension, &Extension::enableRawDataGrabbing);
	disconnect(this->signalProcessing, &Processing::streamingBufferEnabled, extension, &Extension::enableProcessedDataGrabbing);
	disconnect(this->processedDataNotifier, &Gpu2HostNotifier::newGpuDataAvailible, extension, &Extension::processedDataReceived);
	disconnect(this->signalProcessing, &Processing::rawData, extension, &Extension::rawDataReceived);
}

void OCTproZ::slot_enableStopAction() {
	this->actionStop->setEnabled(true);
}

void OCTproZ::slot_updateAcquistionParameter(AcquisitionParams newParams){
	if(this->octParams->samplesPerLine != newParams.samplesPerLine || this->octParams->ascansPerBscan != newParams.ascansPerBscan || this->octParams->bscansPerBuffer != newParams.bscansPerBuffer || this->octParams->buffersPerVolume != newParams.buffersPerVolume){
		emit glBufferTextureSizeBscan(newParams.samplesPerLine/2, newParams.ascansPerBscan, newParams.bscansPerBuffer*newParams.buffersPerVolume);
		//emit glBufferTextureSizeEnFaceView(newParams.bscansPerBuffer, newParams.ascansPerBscan, newParams.samplesPerLine/2);
		emit glBufferTextureSizeEnFaceView(newParams.ascansPerBscan, newParams.bscansPerBuffer*newParams.buffersPerVolume, newParams.samplesPerLine/2);
	}
	if(this->octParams->ascansPerBscan != newParams.ascansPerBscan || this->octParams->bscansPerBuffer != newParams.bscansPerBuffer){
		emit linesPerBufferChanged(newParams.ascansPerBscan * newParams.bscansPerBuffer);
	}

	this->octParams->samplesPerLine = newParams.samplesPerLine;
	this->octParams->ascansPerBscan = newParams.ascansPerBscan;
	this->octParams->bscansPerBuffer = newParams.bscansPerBuffer;
	this->octParams->buffersPerVolume = newParams.buffersPerVolume;
	this->octParams->bitDepth = newParams.bitDepth;
	this->sidebar->slot_setMaximumBscansForNoiseDetermination(this->octParams->bscansPerBuffer);
	this->forceUpdateProcessingParams();
}

void OCTproZ::slot_closeOpenGLwindows() {
#if !defined(Q_OS_WIN)
	if (this->dock2D->isVisible()) {
		this->isDock2DClosed = true;
		this->dock2D->setVisible(false);
	}
	if (this->dockEnFaceView->isVisible()) {
		this->isDockEnFaceViewClosed = true;
		this->dockEnFaceView->setVisible(false);
	}
	if (this->dockVolumeView->isVisible()) {
		this->isDockVolumeViewClosed = true;
		this->dockVolumeView->setVisible(false);
	}
#endif
}

void OCTproZ::slot_reopenOpenGLwindows() {
#if !defined(Q_OS_WIN)
	if (!this->dock2D->isVisible() && this->isDock2DClosed) {
		this->isDock2DClosed = false;
		this->dock2D->setVisible(true);
		this->dock2D->setFocus();
	}
	if (!this->dockEnFaceView->isVisible() && this->isDockEnFaceViewClosed) {
		this->isDockEnFaceViewClosed = false;
		this->dockEnFaceView->setVisible(true);
		this->dockEnFaceView->setFocus();
	}
	if (!this->dockVolumeView->isVisible() && this->isDockVolumeViewClosed) {
		this->isDockVolumeViewClosed = false;
		this->dockVolumeView->setVisible(true);
		this->dockVolumeView->setFocus();
	}
#endif
}

void OCTproZ::slot_storePluginSettings(QString pluginName, QVariantMap settings) {
	Settings::getInstance()->storeSystemSettings(pluginName, settings);
}

void OCTproZ::slot_prepareGpu2HostForProcessedRecording() {
	this->streamToHostMemorized = this->octParams->streamToHost;
	this->streamingBuffersToSkipMemorized = this->octParams->streamingBuffersToSkip;

	this->sidebar->getUi().spinBox_streamingBuffersToSkip->setValue(0);
	this->sidebar->getUi().groupBox_streaming->setChecked(true);
	this->sidebar->getUi().groupBox_streaming->setEnabled(false);
}

void OCTproZ::slot_resetGpu2HostSettings() {
	this->sidebar->getUi().spinBox_streamingBuffersToSkip->setValue(this->streamingBuffersToSkipMemorized);
	this->sidebar->getUi().groupBox_streaming->setChecked(this->streamToHostMemorized);
	this->sidebar->getUi().groupBox_streaming->setEnabled(true);
}

void OCTproZ::slot_recordingDone() {
	this->sidebar->enableRecordTab(true);
	if(this->octParams->stopAfterRecord && this->currSystem->acqusitionRunning){
		this->slot_stop();
	}
}

void OCTproZ::slot_enableEnFaceViewProcessing(bool enable) {
	this->octParams->enFaceViewEnabled = enable;
}

void OCTproZ::slot_enableBscanViewProcessing(bool enable) {
	this->octParams->bscanViewEnabled = enable;
}

void OCTproZ::slot_enableVolumeViewProcessing(bool enable) {
	this->octParams->volumeViewEnabled = enable;
	QCoreApplication::processEvents();
	if(enable){
		this->volumeWindow->update();
	}
}

void OCTproZ::slot_easterEgg() {
	if(this->dockVolumeView->isVisible()){
		if(this->currSystem == nullptr){
			this->volumeWindow->generateTestVolume();
		}
	}
}

void OCTproZ::slot_useCustomResamplingCurve(bool use) {
	this->octParams->useCustomResampleCurve = use;
	this->octParams->acquisitionParamsChanged = true;
	this->sidebar->slot_updateProcessingParams();
}

void OCTproZ::slot_loadCustomResamplingCurve() {
	QString filters("CSV (*.csv)");
	QString defaultFilter("CSV (*.csv)");
	this->slot_closeOpenGLwindows();
	QString fileName = QFileDialog::getOpenFileName(this, tr("Load Curve"), QDir::currentPath(), filters, &defaultFilter);
	this->slot_reopenOpenGLwindows();
	if(fileName == ""){
		emit error(tr("Loading of custom resampling curve for k-linearization canceled."));
		return;
	}

	QFile file(fileName);
	QVector<float> curve;
	file.open(QIODevice::ReadOnly);
	QTextStream txtStream(&file);
	QString line = txtStream.readLine();
	int i = 0;
	while (!txtStream.atEnd()){
		line = txtStream.readLine();
		curve.append((line.section(";", 1, 1).toFloat()));
	}
	file.close();
	this->octParams->loadCustomResampleCurve(curve.data(), curve.size());
	this->octParams->acquisitionParamsChanged = true;
	this->sidebar->slot_updateProcessingParams();
}

void OCTproZ::setSystem(QString systemName) {
	if(this->currSystemName == systemName){ //system already activated
		return;
	}

	AcquisitionSystem* system = this->sysManager->getSystemByName(systemName);

	if(this->currSystem != nullptr){
		this->deactivateSystem(this->currSystem);
	}
	if(!this->activatedSystems.contains(systemName)){ //system got selected for the first time
		this->activatedSystems.append(systemName);
		this->activateSystem(system);
	}else{  //system was once active and needs to be reactivated now
		this->reactivateSystem(system);
	}
	this->currSystem = system;
	this->currSystemName = systemName;
	this->setWindowTitle("OCTproZ - " + systemName);
	emit loadPluginSettings(Settings::getInstance()->getStoredSystemSettings(systemName));
	this->actionStart->setEnabled(true);
	this->actionRecord->setEnabled(true);
}

void OCTproZ::activateSystem(AcquisitionSystem* system){
	if(system != nullptr){
		if(this->currSystem != system){
			system->moveToThread(&acquisitionThread);
			connect(this, &OCTproZ::start, system, &AcquisitionSystem::startAcquisition);
			connect(this, &OCTproZ::stop, system, &AcquisitionSystem::stopAcquisition);
			connect(this, &OCTproZ::loadPluginSettings, system, &AcquisitionSystem::settingsLoaded);
			connect(system, &AcquisitionSystem::storeSettings, this, &OCTproZ::slot_storePluginSettings);
			connect(system, &AcquisitionSystem::acquisitionStarted, this->signalProcessing, &Processing::slot_start);
			connect(system, &AcquisitionSystem::acquisitionStopped, this, &OCTproZ::slot_stop);
			connect(system->params, &AcquisitionParameter::updated, this, &OCTproZ::slot_updateAcquistionParameter);
			connect(this, &OCTproZ::pluginSettingsRequest, system->settingsDialog, &QDialog::show);
			connect(this, &OCTproZ::pluginSettingsRequest, system->settingsDialog, &QDialog::raise);
			connect(system->settingsDialog, &QDialog::finished, this, &OCTproZ::slot_reopenOpenGLwindows); //GL window needs to be closed to avoid linux bug where QFileDialog is not usable when GL window is opend in background
			connect(system, &AcquisitionSystem::info, this->console, &MessageConsole::slot_displayInfo);
			connect(system, &AcquisitionSystem::error, this->console, &MessageConsole::slot_displayError);
			connect(qApp, &QCoreApplication::aboutToQuit, system, &QObject::deleteLater);
			connect(system->buffer, &AcquisitionBuffer::info, this->console, &MessageConsole::slot_displayInfo);
			connect(system->buffer, &AcquisitionBuffer::error, this->console, &MessageConsole::slot_displayError);
			emit newSystem(system);
			acquisitionThread.start();
		}
	}
}

void OCTproZ::deactivateSystem(AcquisitionSystem* system){
	this->slot_stop();
	QCoreApplication::processEvents(); //process events to ensure that acquisition is not running
	disconnect(this, &OCTproZ::start, system, &AcquisitionSystem::startAcquisition);
	disconnect(this, &OCTproZ::stop, system, &AcquisitionSystem::stopAcquisition);
	disconnect(this, &OCTproZ::loadPluginSettings, system, &AcquisitionSystem::settingsLoaded);
	disconnect(this, &OCTproZ::pluginSettingsRequest, system->settingsDialog, &QDialog::show);
	disconnect(this, &OCTproZ::pluginSettingsRequest, system->settingsDialog, &QDialog::raise);
}

void OCTproZ::reactivateSystem(AcquisitionSystem* system){
	connect(this, &OCTproZ::start, system, &AcquisitionSystem::startAcquisition);
	connect(this, &OCTproZ::stop, system, &AcquisitionSystem::stopAcquisition);
	connect(this, &OCTproZ::loadPluginSettings, system, &AcquisitionSystem::settingsLoaded);
	connect(this, &OCTproZ::pluginSettingsRequest, system->settingsDialog, &QDialog::show);
	connect(this, &OCTproZ::pluginSettingsRequest, system->settingsDialog, &QDialog::raise);
}

void OCTproZ::forceUpdateProcessingParams() {
	this->octParams->acquisitionParamsChanged = true;
	this->sidebar->slot_updateProcessingParams();
}
