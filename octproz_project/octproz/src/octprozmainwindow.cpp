#include "octprozmainwindow.h"
#include <QGuiApplication>
#include <QDesktopWidget>
#include <QStyleFactory>

OCTproZMainWindow::OCTproZMainWindow(OCTproZApp* app, QWidget* parent) :
	QMainWindow(parent),
	app(app)
{
	// Initialize UI components
	this->console = new MessageConsole(this);
	this->console->setObjectName("Message Console");
	this->dockConsole = new QDockWidget((tr("Message Console")), this);
	connect(this->app, &OCTproZApp::info, this->console, &MessageConsole::displayInfo);
	connect(this->app, &OCTproZApp::error, this->console, &MessageConsole::displayError);
	connect(this, &OCTproZMainWindow::info, this->console, &MessageConsole::displayInfo);
	connect(this, &OCTproZMainWindow::error, this->console, &MessageConsole::displayError);

	this->aboutWindow = new AboutDialog(this);
	connect(this->aboutWindow, &AboutDialog::easterEgg, this, &OCTproZMainWindow::slot_easterEgg);

	this->plot1D = new PlotWindow1D(this);
	this->plot1D->setName("1d-plot");
	connect(this, &OCTproZMainWindow::allowRawGrabbing, this->plot1D, &PlotWindow1D::enableRawGrabbing);
	connect(this->plot1D, &PlotWindow1D::info, this->console, &MessageConsole::displayInfo);
	connect(this->plot1D, &PlotWindow1D::error, this->console, &MessageConsole::displayError);
	connect(this->app, &OCTproZApp::linesPerBufferChanged, this->plot1D, &PlotWindow1D::changeLinesPerBuffer);

	this->dock1D = new QDockWidget(tr("1D"), this);
	this->dock1D->setObjectName("1D");

	// Initialize B-scan window
	this->bscanWindow = new GLWindow2D(this);
	this->bscanWindow->setMarkerOrigin(TOP);
	this->bscanWindow->setName("bscan-window");

	// Initialize En Face View window
	this->enFaceViewWindow = new GLWindow2D(this);
	this->enFaceViewWindow->setMarkerOrigin(LEFT);
	this->enFaceViewWindow->setName("enfaceview-window");

	// Connect window signals
	connect(this->bscanWindow, &GLWindow2D::currentFrameNr, this->enFaceViewWindow, &GLWindow2D::setMarkerPosition);
	connect(this->enFaceViewWindow, &GLWindow2D::currentFrameNr, this->bscanWindow, &GLWindow2D::setMarkerPosition);
	connect(this, &OCTproZMainWindow::glBufferTextureSizeBscan, this->bscanWindow, &GLWindow2D::changeTextureSize);
	this->dock2D = new QDockWidget(tr("2D - B-scan"), this);
	this->dock2D->setObjectName("2D - B-scan");
	connect(this->dock2D, &QDockWidget::visibilityChanged, this, &OCTproZMainWindow::slot_enableBscanViewProcessing);

	connect(this, &OCTproZMainWindow::glBufferTextureSizeEnFaceView, this->enFaceViewWindow, &GLWindow2D::changeTextureSize);
	this->dockEnFaceView = new QDockWidget(tr("2D - En Face View"), this);
	this->dockEnFaceView->setObjectName("2D - En Face View");
	connect(this->dockEnFaceView, &QDockWidget::visibilityChanged, this, &OCTproZMainWindow::slot_enableEnFaceViewProcessing);

	// Initialize Volume window
	this->volumeWindow = new GLWindow3D(this);
	this->volumeWindow->setName("3d-volume-window");
	this->dockVolumeView = new QDockWidget(tr("3D - Volume"), this);
	this->dockVolumeView->setObjectName("3D - Volume");
	this->dockVolumeView->setFeatures(QDockWidget::DockWidgetClosable);
	connect(this, &OCTproZMainWindow::glBufferTextureSizeBscan, this->volumeWindow, &GLWindow3D::changeTextureSize);
	connect(this->dockVolumeView, &QDockWidget::visibilityChanged, this, &OCTproZMainWindow::slot_enableVolumeViewProcessing);

	// Connects for OpenGL texture size update
	connect(this->app, &OCTproZApp::enFaceViewDimensionsChanged, this->enFaceViewWindow, &GLWindow2D::changeTextureSize);
	connect(this->app, &OCTproZApp::bScanDimensionsChanged, this->bscanWindow, &GLWindow2D::changeTextureSize);
	connect(this->app, &OCTproZApp::bScanDimensionsChanged, this->volumeWindow, &GLWindow3D::changeTextureSize);

	//GL windows need to be closed to avoid linux bug where QFileDialog is not usable when a GL window is opend in background
	connect(this->bscanWindow, &GLWindow2D::dialogAboutToOpen, this, &OCTproZMainWindow::slot_closeOpenGLwindows);
	connect(this->enFaceViewWindow, &GLWindow2D::dialogAboutToOpen, this, &OCTproZMainWindow::slot_closeOpenGLwindows);
	connect(this->bscanWindow, &GLWindow2D::dialogClosed, this, &OCTproZMainWindow::slot_reopenOpenGLwindows);
	connect(this->enFaceViewWindow, &GLWindow2D::dialogClosed, this, &OCTproZMainWindow::slot_reopenOpenGLwindows);
	connect(this->volumeWindow, &GLWindow3D::dialogAboutToOpen, this, &OCTproZMainWindow::slot_closeOpenGLwindows);
	connect(this->volumeWindow, &GLWindow3D::dialogClosed, this, &OCTproZMainWindow::slot_reopenOpenGLwindows);

	// Initialize state variables
	this->isDock2DClosed = false;
	this->isDockEnFaceViewClosed = false;
	this->isDockVolumeViewClosed = false;
	this->initialWindowStateLoadingDone = false;

	// System chooser dialog
	this->systemChooser = new SystemChooser();

	// Initialize sidebar
	this->sidebar = new Sidebar(this);
	this->sidebar->setObjectName("Sidebar");
	connect(this->sidebar, &Sidebar::info, this->console, &MessageConsole::displayInfo);
	connect(this->sidebar, &Sidebar::error, this->console, &MessageConsole::displayError);
	connect(this->sidebar, &Sidebar::dialogAboutToOpen, this, &OCTproZMainWindow::slot_closeOpenGLwindows);
	connect(this->sidebar, &Sidebar::dialogClosed, this, &OCTproZMainWindow::slot_reopenOpenGLwindows);
	connect(this->app, &OCTproZApp::processingParamsUpdateRequested, this->sidebar, &Sidebar::slot_updateProcessingParams); //todo: figure out why opengl windows stay black without this

	// Scheduled Recording Widget
	this->recordingSchedulerWidget = new RecordingSchedulerWidget(this);
	connect(this->recordingSchedulerWidget, &RecordingSchedulerWidget::info, this->console, &MessageConsole::displayInfo);
	connect(this->recordingSchedulerWidget, &RecordingSchedulerWidget::error, this->console, &MessageConsole::displayError);
	connect(this->recordingSchedulerWidget->getScheduler(), &RecordingScheduler::recordingTriggered, this->app, &OCTproZApp::slot_record);
	connect(this->app, &OCTproZApp::recordingFinished, this->recordingSchedulerWidget->getScheduler(), &RecordingScheduler::recordingFinished);

	//GPU info widget
	this->gpuInfoWidget = new GpuInfoWidget(this);
	connect(this->gpuInfoWidget, &GpuInfoWidget::info, this->console, &MessageConsole::displayInfo);
	connect(this->gpuInfoWidget, &GpuInfoWidget::error, this->console, &MessageConsole::displayError);
	this->gpuInfoWidget->checkCudaAvailability();

	// Connect sidebar to paramsManager
	connect(this->sidebar, &Sidebar::savePostProcessBackgroundRequested,
			this->app->getParamsManager(), &OctAlgorithmParametersManager::savePostProcessBackgroundToFile);
	connect(this->app->getParamsManager(), &OctAlgorithmParametersManager::backgroundDataUpdated,
			this->sidebar, &Sidebar::updateBackgroundPlot);
	connect(this->sidebar, &Sidebar::loadPostProcessBackgroundRequested,
			this->app->getParamsManager(), &OctAlgorithmParametersManager::loadPostProcessBackgroundFromFile);
	connect(this->sidebar, &Sidebar::loadResamplingCurveRequested,
			this->app->getParamsManager(), &OctAlgorithmParametersManager::loadCustomResamplingCurveFromFile);

	// Set window position (applies only to the first OCTproZ start, before gui_settings.ini exists)
	QScreen *screen = QGuiApplication::primaryScreen();
	QRect availableGeometry = screen->availableGeometry();
	QPoint offset(availableGeometry.width() / 10, availableGeometry.height() / 10);
	this->move(availableGeometry.topLeft() + offset);

	// Initialize UI
	setWindowTitle("OCTproZ");
	resize(1024, 768);
	QStatusBar* statusBar = new QStatusBar(this);
	setStatusBar(statusBar);

	// Initialize Settings manager
	this->appSettings = new SettingsFileManager(this);
	this->guiSettings = new SettingsFileManager(GUI_SETTINGS_PATH, this);
}

OCTproZMainWindow::~OCTproZMainWindow() {
	//delete this->ui;
	delete this->console;
	delete this->dockConsole;
	delete this->bscanWindow;
	delete this->enFaceViewWindow;
	delete this->dock2D;
	delete this->dockEnFaceView;
	delete this->dockVolumeView;
	delete this->systemChooser;
	delete this->extensionUIManager;
	delete this->recordingSchedulerWidget;
}

void OCTproZMainWindow::closeEvent(QCloseEvent* event) {
	this->guiSettings->setCurrentTimeStamp();
	this->saveWindowStates();
	QApplication::processEvents();

	if (this->extensionUIManager) {
		this->extensionUIManager->shutdownAllExtensions();
	}

	// Stop any active recording schedule
	if (this->recordingSchedulerWidget->getScheduler()->isActive()) {
		this->recordingSchedulerWidget->getScheduler()->stopSchedule();
	}
	this->recordingSchedulerWidget->hide();

	// Disconnect all signals
	disconnect(this->app, nullptr, this, nullptr);
	disconnect(this, nullptr, this->app, nullptr);
	disconnect(this->app->getSignalProcessing(), nullptr, nullptr, nullptr);
	disconnect(this->app->getProcessedDataNotifier(), nullptr, nullptr, nullptr);

	// Ensure system is stopped
	if (this->app->getCurrentSystem() != nullptr) {
		if(this->app->getCurrentSystem()->acqusitionRunning) {
			this->app->slot_stop();
			QCoreApplication::processEvents();
			QThread::msleep(1000); //provide some time to let gpu computation finish
		}
	}

	event->accept();
}

void OCTproZMainWindow::initialize() {
	this->initGui();
	this->setupConnections();
	this->setupExtensions();

	this->loadActiveSystem();

	this->loadWindowState();

	this->extensionUIManager->autoLoadExtensions();

	if (!this->app->getCurrentSystemName().isEmpty()) {
		this->actionStart->setEnabled(true);
		this->actionRecord->setEnabled(true);
	}
}

void OCTproZMainWindow::initGui() {
	this->initActionsAndDocks();
	this->sidebar->init(this->actionStart, this->actionStop, this->actionRecord,
					   this->actionSelectSystem, this->actionSystemSettings);
	this->initMenu();

	// Connect error/info messages
	connect(this->bscanWindow, &GLWindow2D::info, this->console, &MessageConsole::displayInfo);
	connect(this->bscanWindow, &GLWindow2D::error, this->console, &MessageConsole::displayError);
	connect(this->enFaceViewWindow, &GLWindow2D::info, this->console, &MessageConsole::displayInfo);
	connect(this->enFaceViewWindow, &GLWindow2D::error, this->console, &MessageConsole::displayError);
	connect(this->volumeWindow, &GLWindow3D::info, this->console, &MessageConsole::displayInfo);
	connect(this->volumeWindow, &GLWindow3D::error, this->console, &MessageConsole::displayError);

	// Connect to bypass Linux/Qt bugs
	connect(this, &OCTproZMainWindow::closeDock2D, this, &OCTproZMainWindow::slot_closeOpenGLwindows);
	connect(this, &OCTproZMainWindow::reopenDock2D, this, &OCTproZMainWindow::slot_reopenOpenGLwindows);
}

void OCTproZMainWindow::setupConnections() {
	// Connect app signals to UI
	connect(this->app, &OCTproZApp::processingStarted, this, &OCTproZMainWindow::onProcessingStarted);
	connect(this->app, &OCTproZApp::screenshotsRequested, this, &OCTproZMainWindow::slot_takeScreenshots);
	connect(this->app, &OCTproZApp::loadSettingsRequested, this, &OCTproZMainWindow::slot_onAppSettingsLoaded);
	connect(this->app, &OCTproZApp::windowStateReloadRequested, this, &OCTproZMainWindow::loadWindowState);

	connect(this->app, &OCTproZApp::processingStopped, this, [this]() {
		this->actionStart->setEnabled(true);
		this->actionStop->setEnabled(false);
		emit this->allowRawGrabbing(false);
	});

	connect(this->app, &OCTproZApp::recordingStarted, this, [this]() {
		// Only disable the tab for raw or processed recording
		OctAlgorithmParameters::RecordingParams params = this->app->getOctParams()->recParams;
		if (params.recordRaw || params.recordProcessed) {
			this->sidebar->enableRecordTab(false);
		}
	});

	connect(this->app, &OCTproZApp::recordingFinished, this, [this]() {
		this->sidebar->enableRecordTab(true);
	});

	connect(this->app, &OCTproZApp::customResamplingCurveChanged, this, [this](bool enabled) {
			this->actionUseCustomKLinCurve->setEnabled(true);
			this->actionUseCustomKLinCurve->setChecked(enabled);

			this->sidebar->disableKlinCoeffInput(enabled);
			this->sidebar->slot_updateProcessingParams();
	});

	connect(this->app, &OCTproZApp::streamToHostSettingsChanged, this, [this](bool enabled, int buffersToSkip) {
			this->sidebar->getUi().spinBox_streamingBuffersToSkip->setValue(buffersToSkip);
			this->sidebar->getUi().groupBox_streaming->setChecked(enabled);
			this->sidebar->getUi().groupBox_streaming->setEnabled(!enabled);
		});

	connect(this->app, &OCTproZApp::streamToHostSettingsReset,this, [this]() {
			this->sidebar->getUi().spinBox_streamingBuffersToSkip->setValue(
				this->app->getStreamingBuffersToSkipMemorized());
			this->sidebar->getUi().groupBox_streaming->setChecked(
				this->app->getStreamToHostMemorized());
			this->sidebar->getUi().groupBox_streaming->setEnabled(true);
		});

	connect(this->app, &OCTproZApp::bscansPerBufferChanged, this->sidebar, &Sidebar::slot_setMaximumBscansForNoiseDetermination);
	connect(this->app, &OCTproZApp::samplesPerLineChanged, this->sidebar, &Sidebar::slot_setMaximumRollingAverageWindowSize);
	connect(this->app, &OCTproZApp::grayscaleConversionRequested, this->sidebar, &Sidebar::slot_setGrayscaleConversion);
	connect(this->app, &OCTproZApp::klinCoeffsChanged, this->sidebar, &Sidebar::slot_setKLinCoeffs);
	connect(this->app, &OCTproZApp::dispCompCoeffsChanged, this->sidebar, &Sidebar::slot_setDispCompCoeffs);

	connect(this->app, &OCTproZApp::systemChanged, this, &OCTproZMainWindow::setAppWindowTitle);
	connect(this->app, &OCTproZApp::systemChanged, this, [this](const QString& /*systemName*/) {
		this->actionStart->setEnabled(true);
		this->actionRecord->setEnabled(true);
	});

	// Connect UI actions to app
	connect(this->actionStart, &QAction::triggered, this->app, &OCTproZApp::slot_start);
	connect(this->actionStop, &QAction::triggered, this->app, &OCTproZApp::slot_stop);
	connect(this->actionRecord, &QAction::triggered, this->app, &OCTproZApp::slot_record);
	connect(this->actionSelectSystem, &QAction::triggered, this, &OCTproZMainWindow::openSelectSystemDialog);
	connect(this->actionSystemSettings, &QAction::triggered, this, &OCTproZMainWindow::openSystemSettingsDialog);

	// Connect app to signal processing
	connect(this->app->getSignalProcessing(), &Processing::initializationDone, this, &OCTproZMainWindow::slot_enableStopAction);
	connect(this->app->getSignalProcessing(), &Processing::streamingBufferEnabled, this->plot1D, &PlotWindow1D::enableProcessedGrabbing);
	connect(this->app->getSignalProcessing(), &Processing::rawData, this->plot1D, &PlotWindow1D::plotRawData);
	connect(this->app->getSignalProcessing(), &Processing::updateInfoBox, this->sidebar, &Sidebar::slot_updateInfoBox);

	// Connect B-scan window connections
	connect(this->bscanWindow->getControlPanel(), &ControlPanel2D::displayFrameSettingsChanged,
			this->app->getSignalProcessing(), &Processing::slot_updateDisplayedBscanFrame);
	connect(this->bscanWindow, &GLWindow2D::registerBufferCudaGL,
			this->app->getSignalProcessing(), &Processing::slot_registerBscanOpenGLbufferWithCuda);

	// Connect En face view window connections
	connect(this->enFaceViewWindow->getControlPanel(), &ControlPanel2D::displayFrameSettingsChanged,
			this->app->getSignalProcessing(), &Processing::slot_updateDisplayedEnFaceFrame);
	connect(this->enFaceViewWindow, &GLWindow2D::registerBufferCudaGL,
			this->app->getSignalProcessing(), &Processing::slot_registerEnFaceViewOpenGLbufferWithCuda);

	// Connect Volume window connections
	connect(this->volumeWindow, &GLWindow3D::registerBufferCudaGL,
			this->app->getSignalProcessing(), &Processing::slot_registerVolumeViewOpenGLbufferWithCuda);

	// Connect processing connections
	connect(this->app->getSignalProcessing(), &Processing::initOpenGL, this->bscanWindow, &GLWindow2D::createOpenGLContextForProcessing);

	if (!this->app->isProcessingInThread()) {
		connect(this->app->getSignalProcessing(), &Processing::initOpenGL, this->enFaceViewWindow, &GLWindow2D::createOpenGLContextForProcessing); //due to opengl context sharing this connect might not be necessary
	}

	connect(this->app->getSignalProcessing(), &Processing::initOpenGLenFaceView, this->enFaceViewWindow, &GLWindow2D::registerOpenGLBufferWithCuda);
	connect(this->app->getSignalProcessing(), &Processing::initOpenGLenFaceView, this->volumeWindow, &GLWindow3D::registerOpenGLBufferWithCuda);

	// Connect with notifier
	connect(this->app->getProcessedDataNotifier(), &Gpu2HostNotifier::newGpuDataAvailable,
			this->plot1D, &PlotWindow1D::plotProcessedData);
	connect(this->app->getProcessedDataNotifier(), &Gpu2HostNotifier::backgroundRecorded,
			this->sidebar, &Sidebar::updateBackgroundPlot);

	//connects to update opengl displays only when new data is written to display buffers //todo: further investigation. this seems to give hugely different results on linux vs windows. on windows this seems to lock the processing speed to max 60 Hz.
	//connect(this->app->getProcessedDataNotifier(), &Gpu2HostNotifier::bscanDisplayBufferReady, this->bscanWindow, QOverload<>::of(&GLWindow2D::update));
	//connect(this->app->getProcessedDataNotifier(), &Gpu2HostNotifier::enfaceDisplayBufferReady, this->enFaceViewWindow, QOverload<>::of(&GLWindow2D::update));
	//connect(this->app->getProcessedDataNotifier(), &Gpu2HostNotifier::volumeDisplayBufferReady, this->volumeWindow, QOverload<>::of(&GLWindow3D::update));
}

//todo: add load window state from file, similar to load settings from file. separate window state from settings file into its own file.
void OCTproZMainWindow::loadWindowState() {
	QVariantMap windowSettings = this->guiSettings->getStoredSettings(MAIN_WINDOW_SETTINGS_GROUP);
	if (!windowSettings.isEmpty()) {
		// Restore geometry and state
		if (windowSettings.contains(MAIN_GEOMETRY))
			this->restoreGeometry(windowSettings[MAIN_GEOMETRY].toByteArray());
		if (windowSettings.contains(MAIN_STATE))
			this->restoreState(windowSettings[MAIN_STATE].toByteArray());
	}

	MessageConsoleParams consoleParams;
	consoleParams.newestMessageAtBottom = windowSettings.value(MESSAGE_CONSOLE_BOTTOM, false).toBool();
	consoleParams.preferredHeight = windowSettings.value(MESSAGE_CONSOLE_HEIGHT, 100).toInt();
	this->console->setParams(consoleParams);
	//todo: fix bug: when "newest message on the bottom" is activated, on startup the newest message appears at the very top. Only after the next message is displayed are the messages correctly ordered.

	if(!this->initialWindowStateLoadingDone) { //this prevents reopening of docks when app is already running and window state is loades from file.
		this->dock1D->setVisible(false);
		this->dock2D->setVisible(true);
		this->dockEnFaceView->setVisible(true); //info: en face view needs to be visible at some point to ensure opengl intialization. otherwise the app will crash when starting processing
		this->dockVolumeView->setVisible(false);
		this->initialWindowStateLoadingDone = true;
	}

	QTimer::singleShot(500, this, [this, windowSettings]() {
		this->dock1D->setVisible(windowSettings.value(DOCK_1DPLOT_VISIBLE, false).toBool());
		this->dock2D->setVisible(windowSettings.value(DOCK_BSCAN_VISIBLE, true).toBool());
		this->dockEnFaceView->setVisible(windowSettings.value(DOCK_ENFACEVIEW_VISIBLE, true).toBool());
		this->dockVolumeView->setVisible(windowSettings.value(DOCK_VOLUME_VISIBLE, false).toBool());

		//todo: why is geometry of docks restorend even without calling restoreGeometry?
		//this->dock1D->restoreGeometry(windowSettings.value(DOCK_1DPLOT_GEOMETRY).toByteArray());
		//this->dock2D->restoreGeometry(windowSettings.value(DOCK_BSCAN_GEOMETRY).toByteArray());
		//this->dockEnFaceView->restoreGeometry(windowSettings.value(DOCK_ENFACEVIEW_GEOMETRY).toByteArray());
		//this->dockVolumeView->restoreGeometry(windowSettings.value(DOCK_VOLUME_GEOMETRY).toByteArray());
	});

	this->plot1D->setSettings(this->guiSettings->getStoredSettings(this->plot1D->getName()));
	this->bscanWindow->setSettings(this->guiSettings->getStoredSettings(this->bscanWindow->getName()));
	this->enFaceViewWindow->setSettings(this->guiSettings->getStoredSettings(this->enFaceViewWindow->getName()));
	this->volumeWindow->setSettings(this->guiSettings->getStoredSettings(this->volumeWindow->getName()));

	//todo orthogonal views marker state saving and loading

	// resampling curve
	if(this->app->getOctParams()->customResampleCurve != nullptr) {
		bool useCustomCurve = this->app->getOctParams()->useCustomResampleCurve;
		this->actionUseCustomKLinCurve->setEnabled(true);
		this->actionUseCustomKLinCurve->setChecked(useCustomCurve);
		this->actionUseSidebarKLinCurve->setChecked(!useCustomCurve);
		this->sidebar->disableKlinCoeffInput(useCustomCurve);
	}

	// sidebar max values //todo: move this into sidebar class
	this->sidebar->slot_setMaximumBscansForNoiseDetermination(this->app->getOctParams()->bscansPerBuffer);
	this->sidebar->slot_setMaximumRollingAverageWindowSize(this->app->getOctParams()->samplesPerLine);

	// Fix console display issues - todo: even without this singleShot I do not experience issues any more. maybe this is related to the used qt version? For the latest test i used Qt. 5.12.12 -> check if this singleShot code here is really needed
	QTimer::singleShot(0, this, [this]() {
		QSize dockSize = this->dockConsole->size();
		int originalHeight = dockSize.height();
		dockSize.setHeight(originalHeight + 1);
		this->dockConsole->resize(dockSize);
		dockSize.setHeight(originalHeight);
		this->dockConsole->resize(dockSize);
	});

	// auto-load extensions
	if (this->extensionUIManager) {
		this->extensionUIManager->autoLoadExtensions(); //when loading a settings.ini file, this will re-open the previously active extensions
	}

	// recording scheduler
	this->recordingSchedulerWidget->setSettings(this->guiSettings->getStoredSettings(this->recordingSchedulerWidget->getName()));
}

void OCTproZMainWindow::saveWindowStates() {
	this->saveActiveSystem();
	//save paramters
	this->sidebar->saveSettings();
	this->bscanWindow->saveSettings();
	this->enFaceViewWindow->saveSettings();
	this->volumeWindow->saveSettings();
	//todo: implement saveSettings method for console
	//todo: maybe remove settingsfilemanager from sidebar, bscanWindow,enFaceViewWindow and volumeWindow. instead use "getSettings" and "getName" to store settings. just like for plot1D below

	this->guiSettings->storeSettings(this->plot1D->getName(), this->plot1D->getSettings());
	this->guiSettings->storeSettings(this->recordingSchedulerWidget->getName(), this->recordingSchedulerWidget->getSettings());

	// Save active extensions //todo: move to extensionmanager
	this->extensionUIManager->saveExtensionStates();

	//save appearance
	QVariantMap windowSettings;
	windowSettings[MAIN_GEOMETRY] = this->saveGeometry();
	windowSettings[MAIN_STATE] = this->saveState();
	windowSettings[DOCK_1DPLOT_VISIBLE] = this->dock1D->isVisible();
	windowSettings[DOCK_BSCAN_VISIBLE] = this->dock2D->isVisible();
	windowSettings[DOCK_ENFACEVIEW_VISIBLE] = this->dockEnFaceView->isVisible();
	windowSettings[DOCK_VOLUME_VISIBLE] = this->dockVolumeView->isVisible();
	windowSettings[DOCK_1DPLOT_GEOMETRY] = this->dock1D->saveGeometry();
	windowSettings[DOCK_BSCAN_GEOMETRY] = this->dock2D->saveGeometry();
	windowSettings[DOCK_ENFACEVIEW_GEOMETRY] = this->dockEnFaceView->saveGeometry();
	windowSettings[DOCK_VOLUME_GEOMETRY] = this->dockVolumeView->saveGeometry();

	MessageConsoleParams consoleParams = this->console->getParams();
	windowSettings[MESSAGE_CONSOLE_BOTTOM] = consoleParams.newestMessageAtBottom;
	windowSettings[MESSAGE_CONSOLE_HEIGHT] = consoleParams.preferredHeight;

	this->guiSettings->storeSettings(MAIN_WINDOW_SETTINGS_GROUP, windowSettings);
}

void OCTproZMainWindow::loadActiveSystem() { //todo: move this to octprozapp
	// Load settings
	QVariantMap windowSettings = this->appSettings->getStoredSettings(MAIN_WINDOW_SETTINGS_GROUP);

	// Get active system name
	QString systemName = windowSettings.value(MAIN_ACTIVE_SYSTEM).toString();

	// Set system if available
	if (!systemName.isEmpty()) {
		this->app->setSystem(systemName);
	}
}

void OCTproZMainWindow::saveActiveSystem() { //todo: move this to octprozapp
	// Create a map just for the active system
	QVariantMap systemSettings;
	systemSettings[MAIN_ACTIVE_SYSTEM] = this->app->getCurrentSystemName();

	// Store in settings
	this->appSettings->storeSettings(MAIN_WINDOW_SETTINGS_GROUP, systemSettings);
}

void OCTproZMainWindow::initActionsAndDocks() {
	// Create tool bars
	this->viewToolBar = this->addToolBar(tr("View Toolbar"));
	this->viewToolBar->setObjectName("View Toolbar");
	this->view2DExtrasToolBar = this->addToolBar(tr("2D View Tools"));
	this->view2DExtrasToolBar->setObjectName("2D View Tools");
	this->controlToolBar = this->addToolBar(tr("Control Toolbar"));
	this->controlToolBar->setObjectName("Control Toolbar");
	this->controlToolBar->setVisible(false);

	// Init 2D view tools toolbar
	QAction* bscanMarkerAction = this->bscanWindow->getMarkerAction();
	QAction* enfaceMarkerAction = this->enFaceViewWindow->getMarkerAction();
	bscanMarkerAction->setIcon(QIcon(":/icons/octproz_bscanmarker_icon.png"));
	bscanMarkerAction->setToolTip(tr("Display en face view position marker in B-scan"));
	enfaceMarkerAction->setIcon(QIcon(":/icons/octproz_enfacemarker_icon.png"));
	enfaceMarkerAction->setToolTip(tr("Display B-scan position marker in en face view"));
	this->view2DExtrasToolBar->addAction(bscanMarkerAction);
	this->view2DExtrasToolBar->addAction(enfaceMarkerAction);

	// Set central widget
	this->setCentralWidget(this->sidebar->getDock());

	// Setup console dock
	this->dockConsole->setFloating(false);
	this->dockConsole->setVisible(true);
	this->dockConsole->setTitleBarWidget(new QWidget());

	// Create main actions
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

	// Setup dock widgets
	this->prepareDockWidget(this->dock1D, this->plot1D, this->action1D,
						   QIcon(":/icons/octproz_rawsignal_icon.png"), "1D");

	this->prepareDockWidget(this->dock2D, this->bscanWindow, this->action2D,
						   QIcon(":/icons/octproz_bscan_icon.png"), "2D - B-scan");
	this->dock2D->setFloating(false);
	this->dock2D->setVisible(true);

	this->prepareDockWidget(this->dockEnFaceView, this->enFaceViewWindow, this->actionEnFaceView,
						   QIcon(":/icons/octproz_enface_icon.png"), "2D - En Face View");
	this->dockEnFaceView->setFloating(false);
	this->dockEnFaceView->setVisible(true);

	this->prepareDockWidget(this->dockVolumeView, this->volumeWindow, this->action3D,
						   QIcon(":/icons/octproz_volume_icon.png"), "3D - Volume");
	this->dockVolumeView->setFloating(false);
	this->dockVolumeView->setVisible(false);
	this->dockVolumeView->setMinimumWidth(320);

	this->prepareDockWidget(this->dockConsole, this->console, this->actionConsole,
						   QIcon(":/icons/octproz_log_icon.png"), "Console");
	this->addDockWidget(Qt::BottomDockWidgetArea, this->dockConsole);
	this->dockConsole->setFloating(false);
	this->dockConsole->setVisible(true);
	this->dockConsole->setTitleBarWidget(new QWidget());

	// Initialize action states
	this->actionStart->setEnabled(false);
	this->actionStop->setEnabled(false);
	this->actionRecord->setEnabled(false);
}

void OCTproZMainWindow::prepareDockWidget(QDockWidget*& dock, QWidget* widgetForDock, QAction*& action, const QIcon& icon, QString iconText) {
	dock->setWidget(widgetForDock);
	dock->setObjectName(iconText);
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

void OCTproZMainWindow::initMenu() {
	// File menu
	this->actionSelectSystem->setShortcut(QKeySequence::Open);
	this->actionSystemSettings->setShortcut(QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_O));
	QMenu *fileMenu = menuBar()->addMenu(tr("&File"));
	fileMenu->addAction(this->actionSelectSystem);
	fileMenu->addAction(this->actionSystemSettings);

	// Load settings from file
	fileMenu->addSeparator();
	QAction *loadSettingsAction = new QAction(tr("&Load Settings from File"), this);
	loadSettingsAction->setIcon(QIcon(":/icons/octproz_load_icon.png"));
	connect(loadSettingsAction, &QAction::triggered, this, &OCTproZMainWindow::openLoadSettingsFileDialog);
	fileMenu->addAction(loadSettingsAction);
	// Save settings to file
	QAction *saveSettingsAction = new QAction(tr("&Save Settings to File"), this);
	saveSettingsAction->setIcon(QIcon(":/icons/octproz_save_icon.png"));
	connect(saveSettingsAction, &QAction::triggered, this, &OCTproZMainWindow::openSaveSettingsFileDialog);
	fileMenu->addAction(saveSettingsAction);
	fileMenu->addSeparator();
	const QIcon exitIcon = QIcon(":/icons/octproz_close_icon.png");
	QAction *exitAct = fileMenu->addAction(exitIcon, tr("E&xit"), this, &QWidget::close);
	exitAct->setShortcuts(QKeySequence::Quit);
	exitAct->setStatusTip(tr("Close OCTproZ"));

	// View menu
	QMenu* viewMenu = this->menuBar()->addMenu(tr("&View"));
	// Get view toolbar
	QAction* viewToolBarAction = this->viewToolBar->toggleViewAction();
	viewToolBarAction->setText(tr("Show View Toolbar"));
	// Get 2D tools toolbar
	QAction* view2DExtrasToolBarAction = this->view2DExtrasToolBar->toggleViewAction();
	view2DExtrasToolBarAction->setText(tr("Show 2D View Tools"));
	// Get control toolbar
	QAction* controlToolBarAction = this->controlToolBar->toggleViewAction();
	controlToolBarAction->setText(tr("Show Control Toolbar"));
	// Add toolbar view actions to view menu
	viewMenu->addActions({ viewToolBarAction, view2DExtrasToolBarAction, controlToolBarAction,
						 this->action1D, this->action2D, this->actionEnFaceView, this->action3D, this->actionConsole });
	// Load GUI layout from file
	viewMenu->addSeparator();
	QAction *loadLayoutAction = new QAction(tr("&Load Layout from File"), this);
	loadLayoutAction->setIcon(QIcon(":/icons/octproz_load_icon.png"));
	connect(loadLayoutAction, &QAction::triggered, this, &OCTproZMainWindow::openLoadGuiSettingsFileDialog);
	viewMenu->addAction(loadLayoutAction);
	// Save GUI layout to file
	QAction *saveLayoutAction = new QAction(tr("&Save current Layout to File"), this);
	saveLayoutAction->setIcon(QIcon(":/icons/octproz_save_icon.png"));
	connect(saveLayoutAction, &QAction::triggered, this, &OCTproZMainWindow::openSaveGuiSettingsFileDialog);
	viewMenu->addAction(saveLayoutAction);

	// Extras menu
	this->extrasMenu = this->menuBar()->addMenu(tr("&Extras"));

	QAction* scheduledRecordingAction = new QAction(tr("&Scheduled Recording"), this);
	scheduledRecordingAction->setIcon(QIcon(":/icons/octproz_time_icon.png"));
	scheduledRecordingAction->setStatusTip(tr("Schedule automatic recordings"));
	connect(scheduledRecordingAction, &QAction::triggered, this, &OCTproZMainWindow::openRecordingScheduler);
	this->extrasMenu->addAction(scheduledRecordingAction);

	QMenu* klinMenu = this->extrasMenu->addMenu(tr("&Resampling curve for k-linearization"));
	klinMenu->setToolTipsVisible(true);
	klinMenu->setStatusTip(tr("Settings for k-linearization resampling curve"));
	klinMenu->setIcon(QIcon(":/icons/octproz_klincurve_icon.png"));
	QActionGroup* customKlinCurveGroup = new QActionGroup(this);
	this->actionSetCustomKLinCurve = new QAction(tr("&Load custom curve from file..."), this);
	connect(this->actionSetCustomKLinCurve, &QAction::triggered, this, &OCTproZMainWindow::openLoadResamplingCurveDialog);
	this->actionUseSidebarKLinCurve = new QAction(tr("Use &polynomial curve from sidebar"), this);
	this->actionUseCustomKLinCurve = new QAction(tr("Use &custom curve from file"), this);
	this->actionUseSidebarKLinCurve ->setCheckable(true);
	this->actionUseCustomKLinCurve->setCheckable(true);
	this->actionUseSidebarKLinCurve->setActionGroup(customKlinCurveGroup);
	this->actionUseCustomKLinCurve->setActionGroup(customKlinCurveGroup);
	this->actionUseCustomKLinCurve->setEnabled(false);
	this->actionUseSidebarKLinCurve->setChecked(true);
	connect(this->actionUseCustomKLinCurve, &QAction::toggled, this->app, &OCTproZApp::slot_useCustomResamplingCurve);
	klinMenu->addAction(this->actionUseSidebarKLinCurve);
	klinMenu->addAction(this->actionUseCustomKLinCurve);
	QAction* klinSeparator = klinMenu->addSeparator();
	klinMenu->addAction(this->actionSetCustomKLinCurve);
	QList<QAction*> klinActions;
	klinActions << this->actionUseSidebarKLinCurve << this->actionUseCustomKLinCurve << klinSeparator << this->actionSetCustomKLinCurve;
	this->sidebar->addActionsForKlinGroupBoxMenu(klinActions);


	// Help menu
	QMenu *helpMenu = this->menuBar()->addMenu(tr("&Help"));
	// User manual
	QAction *manualAct = helpMenu->addAction(tr("&User Manual"), this, &OCTproZMainWindow::openUserManualDialog);
	manualAct->setStatusTip(tr("OCTproZ user manual"));
	manualAct->setIcon(QIcon(":/icons/octproz_manual_icon.png"));
	manualAct->setShortcut(QKeySequence::HelpContents);
	// GPU Info action
	QAction* gpuInfoAct = helpMenu->addAction(tr("GPU &Info"), this, &OCTproZMainWindow::openGpuInfoWindow);
	gpuInfoAct->setStatusTip(tr("Show GPU information and capabilities"));
	gpuInfoAct->setIcon(QIcon(":/icons/octproz_gpu_icon.png"));
	// About dialog
	QAction *aboutAct = helpMenu->addAction(tr("&About"), this, [this]() {
		this->aboutWindow->show();
	});
	aboutAct->setStatusTip(tr("About OCTproZ"));
	aboutAct->setIcon(QIcon(":/icons/octproz_info_icon.png"));
}

void OCTproZMainWindow::setupExtensions() {
	// Create extension menu
	QMenu* extensionMenu = this->extrasMenu->addMenu(tr("&Extensions"));
	extensionMenu->setIcon(QIcon(":/icons/octproz_extensions_icon.png"));

	// Create the ExtensionUIManager
	this->extensionUIManager = new ExtensionUIManager( //todo: check if the interactions between ExtensionUIManager, ExtensionManager, OCTproZMainWindow and OCTproZApp can be simplified.
		extensionMenu,
		this->sidebar->getUi().tabWidget,
		this->console,
		this->sidebar,
		this->app,
		this
	);

	// Initialize with extension manager
	this->extensionUIManager->initialize(this->app->getExtManager());

	// Connect signals for raw data grabbing
	connect(this, &OCTproZMainWindow::allowRawGrabbing,
			this->extensionUIManager, &ExtensionUIManager::slot_enableRawGrabbing);
}


void OCTproZMainWindow::slot_enableBscanViewProcessing(bool enable) {
	this->app->getOctParams()->bscanViewEnabled = enable;
}

void OCTproZMainWindow::slot_enableEnFaceViewProcessing(bool enable) {
	this->app->getOctParams()->enFaceViewEnabled = enable;
}

void OCTproZMainWindow::slot_enableVolumeViewProcessing(bool enable) {
	this->app->getOctParams()->volumeViewEnabled = enable;
	QCoreApplication::processEvents();
	if(enable) {
		this->volumeWindow->update();
	}
}

void OCTproZMainWindow::slot_closeOpenGLwindows() {
#if !defined(Q_OS_WIN) && !defined(__aarch64__)
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

void OCTproZMainWindow::slot_reopenOpenGLwindows() {
#if !defined(Q_OS_WIN) && !defined(__aarch64__)
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

void OCTproZMainWindow::slot_enableStopAction() {
	this->actionStop->setEnabled(true);
}

void OCTproZMainWindow::slot_easterEgg() {
	if(this->dockVolumeView->isVisible()){
		if(this->app->getCurrentSystem() == nullptr){
			this->volumeWindow->generateTestVolume();
		}
		if(this->app->getCurrentSystem() != nullptr){
			if(!this->app->getCurrentSystem()->acqusitionRunning){
				this->volumeWindow->generateTestVolume();
			}
		}
	}
}

void OCTproZMainWindow::slot_takeScreenshots(const QString& savePath, const QString& baseName) {
	if(this->bscanWindow->isVisible()) {
		this->bscanWindow->saveScreenshot(savePath, baseName + "bscan_snapshot.png");
	}
	if(this->enFaceViewWindow->isVisible()) {
		this->enFaceViewWindow->saveScreenshot(savePath, baseName + "enfaceview_snapshot.png");
	}
	if(this->volumeWindow->isVisible()) {
		this->volumeWindow->saveScreenshot(savePath, baseName + "volume_snapshot.png");
	}
}

void OCTproZMainWindow::slot_onAppSettingsLoaded() {
	// Load sidebar settings
	this->sidebar->loadSettings();

	// Update resampling curve UI based on app's octParams
	if(this->app->getOctParams()->customResampleCurve != nullptr) {
		bool useCustomCurve = this->app->getOctParams()->useCustomResampleCurve;
		this->actionUseCustomKLinCurve->setEnabled(true);
		this->actionUseCustomKLinCurve->setChecked(useCustomCurve);
		this->actionUseSidebarKLinCurve->setChecked(!useCustomCurve);
		this->sidebar->disableKlinCoeffInput(useCustomCurve);
	}

	// Force updating processing parameters
	this->app->getOctParams()->acquisitionParamsChanged = true;
	this->app->getOctParams()->postProcessBackgroundUpdated = true;
	this->sidebar->slot_updateProcessingParams();
}

void OCTproZMainWindow::setAppWindowTitle(const QString& title) {
	this->setWindowTitle("OCTproZ - " + title);
}

void OCTproZMainWindow::onProcessingStarted() {
		//under certain circumstances, the OpenGL windows remain black. this fixes this issue
		this->resize(static_cast<float>(this->size().width()-1), static_cast<float>(this->size().height()-1));
		this->resize(static_cast<float>(this->size().width()+1), static_cast<float>(this->size().height()+1));

		// Update OpenGL texture size
		// todo: remove this, it is only needed when test volume is used
		//.figure out how to best inform opengl windows about texture size if test volume from slot_easterEgg is used
		emit this->glBufferTextureSizeBscan(this->app->getOctParams()->samplesPerLine/2,
										   this->app->getOctParams()->ascansPerBscan,
										   this->app->getOctParams()->bscansPerBuffer *
										   this->app->getOctParams()->buffersPerVolume);

		// Disable buttons
		this->actionStart->setEnabled(false);
		this->actionStop->setEnabled(false);

		// Enable data grabbing
		emit this->allowRawGrabbing(true); //todo: move this to octprozapp
}

void OCTproZMainWindow::openLoadSettingsFileDialog() {
	emit closeDock2D(); // Close OpenGL windows to handle Linux/Qt bug where QFileDialog is empty if a OpenGL window is visible in the background

	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Select Settings File"),
		"",
		tr("Settings Files (*.ini *.txt)")
	);

	emit reopenDock2D();

	if (!fileName.isEmpty()) {
		this->app->loadSettingsFromFile(fileName);
	}
}

void OCTproZMainWindow::openSaveSettingsFileDialog() {
	emit closeDock2D();

	QString fileName = QFileDialog::getSaveFileName(this,
		tr("Save Settings File"),
		"",
		tr("Settings Files (*.ini *.txt);;All Files (*.*)")
	);

	emit reopenDock2D();

	if (!fileName.isEmpty()) {
		this->app->saveSettingsToFile(fileName);
	}
}

void OCTproZMainWindow::openLoadGuiSettingsFileDialog() {
	emit closeDock2D();

	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Select GUI Settings File"),
		"",
		tr("Settings Files (*.ini *.txt)")
	);

	emit reopenDock2D();

	if (!fileName.isEmpty()) {
		this->loadGuiSettingsFromFile(fileName);
	}
}

void OCTproZMainWindow::openSaveGuiSettingsFileDialog() {
	emit closeDock2D();

	QString fileName = QFileDialog::getSaveFileName(this,
		tr("Save GUI Settings File"),
		"",
		tr("Settings Files (*.ini *.txt);;All Files (*.*)")
	);

	emit reopenDock2D();

	if (!fileName.isEmpty()) {
		this->saveGuiSettingsToFile(fileName);
	}
}

void OCTproZMainWindow::openLoadResamplingCurveDialog() {
	emit closeDock2D();

	QString filters("CSV (*.csv)");
	QString defaultFilter("CSV (*.csv)");
	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Load Curve"),
		QDir::currentPath(),
		filters,
		&defaultFilter
	);

	emit reopenDock2D();

	if (!fileName.isEmpty()) {
		this->app->loadResamplingCurveFromFile(fileName);
	}
}

void OCTproZMainWindow::openSystemSettingsDialog() {
	AcquisitionSystem* system = this->app->getCurrentSystem();
	if (system && system->settingsDialog) {
		system->settingsDialog->show();
		system->settingsDialog->raise();
		system->settingsDialog->activateWindow();
	} else {
		this->console->displayError(tr("No system opened!"));
	}
}

void OCTproZMainWindow::openSelectSystemDialog() {
	QString selectedSystem = this->systemChooser->selectSystem(this->app->getSysManager()->getSystemNames());
	if (!selectedSystem.isEmpty()) {
		this->app->setSystem(selectedSystem);
	}
}

void OCTproZMainWindow::openUserManualDialog() {
	QDesktopServices::openUrl(QUrl("file:///" + QCoreApplication::applicationDirPath() + "/docs/index.html"));
}

void OCTproZMainWindow::openRecordingScheduler() {
	this->recordingSchedulerWidget->show();
	this->recordingSchedulerWidget->raise();
	this->recordingSchedulerWidget->activateWindow();
}

void OCTproZMainWindow::openGpuInfoWindow() {
	this->gpuInfoWidget->show();
	this->gpuInfoWidget->raise();
	this->gpuInfoWidget->activateWindow();
}


void OCTproZMainWindow::loadGuiSettingsFromFile(const QString &settingsFilePath) {
	if (settingsFilePath.isEmpty()) {
		emit error(tr("No GUI settings file selected or file path is invalid. Nothing loaded."));
		return;
	}

	// Backup current GUI settings file
	QString backupPath = GUI_SETTINGS_PATH + ".backup";
	QFile::remove(backupPath);  // Remove any existing backup
	QFile::copy(GUI_SETTINGS_PATH, backupPath);

	// Attempt to copy the selected settings file to the default settings path
	if (!QFile::remove(GUI_SETTINGS_PATH)) {
		emit error(tr("Failed to remove the existing GUI settings file."));
		return;
	}

	if (!QFile::copy(settingsFilePath, GUI_SETTINGS_PATH)) {
		emit error(tr("Failed to load GUI settings from: ") + settingsFilePath);
		QFile::copy(backupPath, GUI_SETTINGS_PATH);
		return;
	}

	// Reload the settings
	this->loadWindowState();

	// Inform the user that the settings have been loaded
	emit info(tr("GUI settings have been loaded successfully from: ") + settingsFilePath);
}

void OCTproZMainWindow::saveGuiSettingsToFile(const QString &fileName) {
	if (fileName.isEmpty()) {
		emit error(tr("File path is empty. GUI settings file not saved."));
		return;
	}

	// Ensure the file has an extension - use .ini as default
	QString finalFileName = fileName;
	QFileInfo fileInfo(fileName);
	if (fileInfo.suffix().isEmpty()) {
		finalFileName = fileName + ".ini";
	}

	// Make sure window states are saved first
	this->saveWindowStates();

	// Save the settings by copying the existing settings file
	if (this->guiSettings->copySettingsFile(finalFileName)) {
		emit info(tr("GUI settings have been saved successfully to: ") + finalFileName);
	} else {
		emit error(tr("Failed to save GUI settings to: ") + finalFileName);
	}
}


