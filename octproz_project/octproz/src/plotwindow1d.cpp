#include "plotwindow1d.h"

// ---------------------- PlotWindow1D Implementation ----------------------
PlotWindow1D::PlotWindow1D(QWidget *parent) : QWidget(parent),
	line(0), linesPerBuffer(0), currentRawBitdepth(0),
	isPlottingRaw(false), isPlottingProcessed(false),
	autoscaling(true), displayRaw(true), displayProcessed(false),
	bitshift(false), rawGrabbingAllowed(true), processedGrabbingAllowed(true),
	showStatistics(true), showInfoInLegend(false), dataCursorEnabled(false),
	canUpdateRawPlot(true), canUpdateProcessedPlot(true)
{
	this->setContentsMargins(0, 0, 0, 0);

	this->mainLayout = new QHBoxLayout();
	this->mainLayout->setContentsMargins(0, 0, 0, 0);
	this->mainLayout->setSpacing(0);

	this->plotArea = new PlotArea1D(this);
	this->plotArea->setMinimumHeight(256);
	this->plotArea->setMinimumWidth(320);

	this->statsLabel = new StatsLabel(this);
	this->statsLabel->setVisible(this->showStatistics);

	this->mainLayout->addWidget(this->plotArea, 1);
	this->mainLayout->addWidget(this->statsLabel, 0);

	this->panel = new ControlPanel1D(this);

	this->containerLayout = new QVBoxLayout(this);
	this->containerLayout->addLayout(this->mainLayout, 1);
	this->containerLayout->addStretch();
	this->containerLayout->addWidget(this->panel);
	this->containerLayout->setContentsMargins(9, 0, 9, 9);

	this->coordinateDisplay = new QLabel(this);
	this->coordinateDisplay->setStyleSheet("QLabel { background-color: rgba(0, 0, 0, 150); color: white; }");
	this->coordinateDisplay->setVisible(false);
	this->coordinateDisplay->setAttribute(Qt::WA_TransparentForMouseEvents);
	this->coordinateDisplay->raise();

	this->setRawPlotVisible(this->displayRaw);
	this->setProcessedPlotVisible(this->displayProcessed);

	this->changeLinesPerBuffer(999999);

	this->panel->checkBoxAutoscale->setChecked(this->autoscaling);
	this->panel->checkBoxProcessed->setChecked(this->displayProcessed);
	this->panel->checkBoxRaw->setChecked(this->displayRaw);

	connect(this->panel->checkBoxProcessed, &QCheckBox::stateChanged, this, &PlotWindow1D::enableDisplayProcessed);
	connect(this->panel->checkBoxRaw, &QCheckBox::stateChanged, this, &PlotWindow1D::enableDisplayRaw);
	connect(this->panel->checkBoxAutoscale, &QCheckBox::stateChanged, this, &PlotWindow1D::activateAutoscaling);
	connect(this->panel->spinBoxLine, QOverload<int>::of(&QSpinBox::valueChanged), this, &PlotWindow1D::setLineNr);

	//connect mouseEntered signals to disable data cursor lable when mouse is outside of plot area
	connect(this->panel, &ControlPanel1D::mouseEntered, this, [this]() {
		if (this->dataCursorEnabled) {
			this->coordinateDisplay->setVisible(false);
		}
	});
	connect(this->statsLabel, &StatsLabel::mouseEntered, this, [this]() {
		if (this->dataCursorEnabled) {
			this->coordinateDisplay->setVisible(false);
		}
	});

	connect(this->plotArea, &PlotArea1D::cursorCoordinates, this, &PlotWindow1D::handleCursorCoordinates);

	//set up timer to limit refresh rate
	//Limiting refresh rate fixes this bug: when right clicking on the plot
	//to open the context menu, sometimes the context menu is not displayed correctly and/or
	//the plot stops updating when the plot is updated very fast
	this->rawUpdateTimer = new QTimer(this);
	this->rawUpdateTimer->setSingleShot(true);
	this->rawUpdateTimer->setInterval(20); // 20ms = 50Hz
	connect(this->rawUpdateTimer, &QTimer::timeout, this, &PlotWindow1D::enableRawPlotUpdate);

	this->processedUpdateTimer = new QTimer(this);
	this->processedUpdateTimer->setSingleShot(true);
	this->processedUpdateTimer->setInterval(20); // 20ms = 50Hz
	connect(this->processedUpdateTimer, &QTimer::timeout, this, &PlotWindow1D::enableProcessedPlotUpdate);
}

PlotWindow1D::~PlotWindow1D() {
}

void PlotWindow1D::setSettings(QVariantMap settings) {
	this->displayRaw = settings.value(PLOT1D_DISPLAY_RAW, true).toBool();
	this->setRawPlotVisible(this->displayRaw);
	this->panel->checkBoxRaw->setChecked(this->displayRaw);
	this->statsLabel->setRawStatsVisible(this->displayRaw && this->showStatistics);

	this->displayProcessed = settings.value(PLOT1D_DISPLAY_PROCESSED, false).toBool();
	this->setProcessedPlotVisible(this->displayProcessed);
	this->panel->checkBoxProcessed->setChecked(this->displayProcessed);
	this->statsLabel->setProcessedStatsVisible(this->displayProcessed && this->showStatistics);

	this->autoscaling = settings.value(PLOT1D_AUTOSCALING, true).toBool();
	this->panel->checkBoxAutoscale->setChecked(this->autoscaling);

	// Restore the axis ranges
	if (settings.contains(PLOT1D_XAXIS_MIN) && settings.contains(PLOT1D_XAXIS_MAX)) {
		double xMin = settings.value(PLOT1D_XAXIS_MIN).toDouble();
		double xMax = settings.value(PLOT1D_XAXIS_MAX).toDouble();
		this->plotArea->xAxis->setRange(xMin, xMax);
	}
	if (settings.contains(PLOT1D_YAXIS_MIN) && settings.contains(PLOT1D_YAXIS_MAX)) {
		double yMin = settings.value(PLOT1D_YAXIS_MIN).toDouble();
		double yMax = settings.value(PLOT1D_YAXIS_MAX).toDouble();
		this->plotArea->yAxis->setRange(yMin, yMax);
	}
	if (settings.contains(PLOT1D_XAXIS2_MIN) && settings.contains(PLOT1D_XAXIS2_MAX)) {
		double x2Min = settings.value(PLOT1D_XAXIS2_MIN).toDouble();
		double x2Max = settings.value(PLOT1D_XAXIS2_MAX).toDouble();
		this->plotArea->xAxis2->setRange(x2Min, x2Max);
	}
	if (settings.contains(PLOT1D_YAXIS2_MIN) && settings.contains(PLOT1D_YAXIS2_MAX)) {
		double y2Min = settings.value(PLOT1D_YAXIS2_MIN).toDouble();
		double y2Max = settings.value(PLOT1D_YAXIS2_MAX).toDouble();
		this->plotArea->yAxis2->setRange(y2Min, y2Max);
	}

	this->bitshift = settings.value(PLOT1D_BITSHIFT, false).toBool();

	this->line = settings.value(PLOT1D_LINE_NR, 0).toInt();
	this->panel->spinBoxLine->setValue(this->line);

	this->dataCursorEnabled = settings.value(PLOT1D_DATA_CURSOR, false).toBool();
	if (this->dataCursorEnabled) {
		this->plotArea->setCursor(Qt::CrossCursor);
	} else {
		this->plotArea->unsetCursor();
		this->coordinateDisplay->setVisible(false);
	}

	bool legendVisible = settings.value(PLOT1D_SHOW_LEGEND, true).toBool();
	this->plotArea->legend->setVisible(legendVisible);

	this->setShowInfoInLegend(settings.value(PLOT1D_INFO_IN_LEGEND, false).toBool());

	this->plotArea->applyLegendSettings(settings);
	this->plotArea->replot();

	this->showStatistics = settings.value(PLOT1D_SHOW_STATS, false).toBool();
	this->statsLabel->setVisible(this->showStatistics);
	this->statsLabel->setRawStatsVisible(this->displayRaw && this->showStatistics);
	this->statsLabel->setProcessedStatsVisible(this->displayProcessed && this->showStatistics);
}

QVariantMap PlotWindow1D::getSettings() {
	QVariantMap settings;
	settings.insert(PLOT1D_DISPLAY_RAW, this->displayRaw);
	settings.insert(PLOT1D_DISPLAY_PROCESSED, this->displayProcessed);
	settings.insert(PLOT1D_AUTOSCALING, this->autoscaling);
	settings.insert(PLOT1D_BITSHIFT, this->bitshift);
	settings.insert(PLOT1D_LINE_NR, this->line);
	settings.insert(PLOT1D_DATA_CURSOR, this->dataCursorEnabled);
	settings.insert(PLOT1D_SHOW_LEGEND, this->plotArea->legend->visible());
	settings.insert(PLOT1D_SHOW_STATS, this->showStatistics);
	settings.insert(PLOT1D_INFO_IN_LEGEND, this->showInfoInLegend);

	settings.insert(PLOT1D_XAXIS_MIN, this->plotArea->xAxis->range().lower);
	settings.insert(PLOT1D_XAXIS_MAX, this->plotArea->xAxis->range().upper);
	settings.insert(PLOT1D_YAXIS_MIN, this->plotArea->yAxis->range().lower);
	settings.insert(PLOT1D_YAXIS_MAX, this->plotArea->yAxis->range().upper);
	settings.insert(PLOT1D_XAXIS2_MIN, this->plotArea->xAxis2->range().lower);
	settings.insert(PLOT1D_XAXIS2_MAX, this->plotArea->xAxis2->range().upper);
	settings.insert(PLOT1D_YAXIS2_MIN, this->plotArea->yAxis2->range().lower);
	settings.insert(PLOT1D_YAXIS2_MAX, this->plotArea->yAxis2->range().upper);

	// Get legend settings from plot area
	QVariantMap legendSettings = this->plotArea->getLegendSettings();
	for (auto it = legendSettings.begin(); it != legendSettings.end(); ++it) {
		settings.insert(it.key(), it.value());
	}

	return settings;
}

QSize PlotWindow1D::sizeHint() const {
	if (this->statsLabel && this->statsLabel->isVisible()) {
		return QSize(840, 320); // Wider to accommodate stats panel
	} else {
		return QSize(640, 320); // Standard size without stats panel
	}
}

// Implementation of the methods needed by the external API
void PlotWindow1D::setRawPlotVisible(bool visible) {
	this->plotArea->setRawPlotVisible(visible);
	this->statsLabel->setRawStatsVisible(visible && this->showStatistics);
	this->plotArea->replot();
}

void PlotWindow1D::setProcessedPlotVisible(bool visible) {
	this->plotArea->setProcessedPlotVisible(visible);
	this->statsLabel->setProcessedStatsVisible(visible && this->showStatistics);
	this->plotArea->replot();
}

void PlotWindow1D::contextMenuEvent(QContextMenuEvent *event) {
	QMenu menu(this);

	QAction savePlotAction(tr("Save Plot as..."), this);
	connect(&savePlotAction, &QAction::triggered, this, &PlotWindow1D::saveToDisk);
	menu.addAction(&savePlotAction);

	QAction bitshiftRawValuesAction(tr("Bit shift raw values by 4"), this);
	bitshiftRawValuesAction.setCheckable(true);
	bitshiftRawValuesAction.setChecked(this->bitshift);
	connect(&bitshiftRawValuesAction, &QAction::toggled, this, &PlotWindow1D::enableBitshift);
	if(this->currentRawBitdepth > 8 && this->currentRawBitdepth <= 16){
		menu.addAction(&bitshiftRawValuesAction);
	}

	QAction dualCoordAction(tr("Show Values at Cursor"), this);
	dualCoordAction.setCheckable(true);
	dualCoordAction.setChecked(this->dataCursorEnabled);
	connect(&dualCoordAction, &QAction::toggled, this, &PlotWindow1D::toggleDualCoordinates);
	menu.addAction(&dualCoordAction);

	QAction toggleLegendAction(tr("Show Legend"), this);
	toggleLegendAction.setCheckable(true);
	toggleLegendAction.setChecked(this->plotArea->legend->visible());
	connect(&toggleLegendAction, &QAction::toggled, this, &PlotWindow1D::toggleLegend);
	menu.addAction(&toggleLegendAction);

	QAction showInfoAction(tr("Show Line Coordinates in Legend"), this);
	showInfoAction.setCheckable(true);
	showInfoAction.setChecked(this->plotArea->isInfoInLegendEnabled());
	connect(&showInfoAction, &QAction::toggled, this, &PlotWindow1D::setShowInfoInLegend);
	menu.addAction(&showInfoAction);

	QAction showStatsAction(tr("Show Statistics"), this);
	showStatsAction.setCheckable(true);
	showStatsAction.setChecked(this->showStatistics);
	connect(&showStatsAction, &QAction::toggled, this, &PlotWindow1D::toggleStatsInLegend);
	menu.addAction(&showStatsAction);

	QAction resetLegendAction(tr("Reset Legend Position"), this);
	connect(&resetLegendAction, &QAction::triggered, this, [this]() {
		this->plotArea->axisRect()->insetLayout()->setInsetPlacement(0, QCPLayoutInset::ipBorderAligned);
		this->plotArea->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignTop | Qt::AlignRight);
		this->plotArea->replot();
	});
	menu.addAction(&resetLegendAction);

	menu.exec(event->globalPos());

	event->accept();
}

void PlotWindow1D::mouseDoubleClickEvent(QMouseEvent *event) {
	this->plotArea->rescaleAxes();
	this->plotArea->replot();
	event->accept();
}

void PlotWindow1D::leaveEvent(QEvent *event) {
	if (this->dataCursorEnabled) {
		this->coordinateDisplay->setVisible(false);
	}
	event->accept();
}

void PlotWindow1D::plotRawData(void* buffer, unsigned bitDepth, unsigned int samplesPerLine, unsigned int linesPerFrame, unsigned int framesPerBuffer, unsigned int buffersPerVolume, unsigned int currentBufferNr) {
	Q_UNUSED(linesPerFrame);
	Q_UNUSED(framesPerBuffer);
	Q_UNUSED(buffersPerVolume);
	if(!this->isPlottingRaw && this->displayRaw && this->rawGrabbingAllowed && this->canUpdateRawPlot){
		this->canUpdateRawPlot = false;
		this->isPlottingRaw = true;
		this->currentRawBitdepth = bitDepth;

		if(buffer != nullptr && this->isVisible()){
			//get length of one raw line (unprocessed a-scan) and resize plot vectors if necessary
			if(this->sampleValues.size() != samplesPerLine){
				this->sampleValues.resize(samplesPerLine);
				this->sampleNumbers.resize(samplesPerLine);
				for(unsigned int i = 0; i<samplesPerLine; i++){
					this->sampleNumbers[i] = i;
				}
			}
			//change line number if buffer size got smaller and there are now fewer lines in buffer than the current line number
			if(this->line > this->linesPerBuffer-1){
				this->line = this->linesPerBuffer-1;
			}

			//copy values from buffer to plot vector
			qreal max = -qInf();
			qreal min = qInf();
			qreal mean = 0.0;
			qreal sumOfSquaredDiffs = 0.0;

			for(unsigned int i = 0; i<samplesPerLine && this->rawGrabbingAllowed; i++){
				//char
				if(bitDepth <= 8){
					unsigned char* bufferPointer = static_cast<unsigned char*>(buffer);
					this->sampleValues[i] = bufferPointer[this->line*samplesPerLine+i];
				}
				//ushort
				if(bitDepth > 8 && bitDepth <= 16){
					unsigned short* bufferPointer = static_cast<unsigned short*>(buffer);
					if(this->bitshift){
						this->sampleValues[i] = bufferPointer[this->line*samplesPerLine+i] >> 4;
					}else{
						this->sampleValues[i] = bufferPointer[this->line*samplesPerLine+i];
					}
				}
				//unsigned long int
				if(bitDepth > 16 && bitDepth <= 32){
					unsigned long int* bufferPointer = static_cast<unsigned long int*>(buffer);
					this->sampleValues[i] = bufferPointer[this->line*samplesPerLine+i];
				}

				if (this->sampleValues[i] < min) { min = this->sampleValues[i]; }
				if (this->sampleValues[i] > max) { max = this->sampleValues[i]; }

				//Welford's method for mean and variance calculation
				qreal delta = this->sampleValues[i] - mean;
				mean += delta / (i + 1);
				sumOfSquaredDiffs += delta * (this->sampleValues[i] - mean);
			}

			qreal stdDeviation = qSqrt(sumOfSquaredDiffs / samplesPerLine);

			//update plot
			if(this->plotArea->isInfoInLegendEnabled()) {
				this->plotArea->updateInfoInRawLegend(this->line, currentBufferNr);
			}

			this->plotArea->setRawData(this->sampleNumbers, this->sampleValues);

			if(this->autoscaling){
				this->plotArea->graph(0)->rescaleAxes();
			}

			this->plotArea->replot();

			if (this->showStatistics && this->statsLabel) {
				this->statsLabel->updateRawStats(min, max, mean, stdDeviation, this->line, currentBufferNr);
			}

			//QCoreApplication::processEvents(); //this processEvents is required here if no timer is used to limit update rate
		}
		this->isPlottingRaw = false;
		this->rawUpdateTimer->start();
	}
}

void PlotWindow1D::plotProcessedData(void* buffer, unsigned bitDepth, unsigned int samplesPerLine, unsigned int linesPerFrame, unsigned int framesPerBuffer, unsigned int buffersPerVolume, unsigned int currentBufferNr) {
	Q_UNUSED(linesPerFrame);
	Q_UNUSED(framesPerBuffer);
	Q_UNUSED(buffersPerVolume);
	if(!this->isPlottingProcessed && this->displayProcessed && this->processedGrabbingAllowed && this->canUpdateProcessedPlot){
		this->canUpdateProcessedPlot = false;
		this->isPlottingProcessed = true;
		if(buffer != nullptr && this->isVisible()){
			//get length of one A-scan and resize plot vectors if necessary
			if(this->sampleValuesProcessed.size() != samplesPerLine){
				this->sampleValuesProcessed.resize(samplesPerLine);
				this->sampleNumbersProcessed.resize(samplesPerLine);
				for(unsigned int i = 0; i<samplesPerLine; i++){
					this->sampleNumbersProcessed[i] = i;
				}
			}
			//change line number if buffer size got smaller and there are now fewer lines in buffer than the current line number
			this->linesPerBuffer = linesPerFrame * framesPerBuffer;
			if(this->line > this->linesPerBuffer-1){
				this->line = this->linesPerBuffer-1;
			}

			//copy values from buffer to plot vector
			qreal max = -qInf();
			qreal min = qInf();
			qreal mean = 0.0;
			qreal sumOfSquaredDiffs = 0.0;

			for(unsigned int i = 0; i<samplesPerLine && this->processedGrabbingAllowed; i++){
				//uchar
				if(bitDepth <= 8){
					unsigned char* bufferPointer = static_cast<unsigned char*>(buffer);
					this->sampleValuesProcessed[i] = bufferPointer[this->line*samplesPerLine+i];
				}
				//ushort
				if(bitDepth > 8 && bitDepth <= 16){
					unsigned short* bufferPointer = static_cast<unsigned short*>(buffer);
					this->sampleValuesProcessed[i] = bufferPointer[this->line*samplesPerLine+i];
				}
				//unsigned long int
				if(bitDepth > 16 && bitDepth <= 32){
					unsigned long int* bufferPointer = static_cast<unsigned long int*>(buffer);
					this->sampleValuesProcessed[i] = bufferPointer[this->line*samplesPerLine+i];
				}

				if (this->sampleValuesProcessed[i] < min) { min = this->sampleValuesProcessed[i]; }
				if (this->sampleValuesProcessed[i] > max) { max = this->sampleValuesProcessed[i]; }

				// Welford's method for mean and variance calculation
				qreal delta = this->sampleValuesProcessed[i] - mean;
				mean += delta / (i + 1);
				sumOfSquaredDiffs += delta * (this->sampleValuesProcessed[i] - mean);
			}

			qreal stdDeviation = qSqrt(sumOfSquaredDiffs / samplesPerLine);

			//update plot
			if(this->plotArea->isInfoInLegendEnabled()) {
				this->plotArea->updateInfoInProcessedLegend(this->line, currentBufferNr);
			}

			this->plotArea->setProcessedData(this->sampleNumbersProcessed, this->sampleValuesProcessed);

			if(this->autoscaling){
				this->plotArea->graph(1)->rescaleAxes();
			}

			this->plotArea->replot();

			// Update the stats panel with processed data statistics
			if (this->showStatistics && this->statsLabel) {
				this->statsLabel->updateProcessedStats(min, max, mean, stdDeviation, this->line, currentBufferNr);
			}

			//QCoreApplication::processEvents(); //this processEvents is required here if no timer is used to limit update rate
		}
		this->isPlottingProcessed = false;
		this->processedUpdateTimer->start();
	}
}

void PlotWindow1D::changeLinesPerBuffer(int linesPerBuffer) {
	this->linesPerBuffer = linesPerBuffer;
	this->panel->setMaxLineNr(linesPerBuffer);
}

void PlotWindow1D::setLineNr(int lineNr) {
	this->line = lineNr;
}

void PlotWindow1D::enableDisplayRaw(bool display) {
	this->displayRaw = display;
	this->setRawPlotVisible(display);
}

void PlotWindow1D::enableDisplayProcessed(bool display) {
	this->displayProcessed = display;
	this->setProcessedPlotVisible(display);
}

void PlotWindow1D::activateAutoscaling(bool activate) {
	this->autoscaling = activate;
	this->plotArea->replot();
}

void PlotWindow1D::saveToDisk() {
	QString filters("Image (*.png);;Vector graphic (*.pdf);;CSV (*.csv)");
	QString defaultFilter("CSV (*.csv)");
	QString fileName = QFileDialog::getSaveFileName(this, tr("Save 1D Plot"), QDir::currentPath(), filters, &defaultFilter);
	if(fileName == ""){
		emit error(tr("Save plot to disk canceled."));
		return;
	}
	bool saved = false;
	if(defaultFilter == "Image (*.png)"){
		saved = this->plotArea->savePng(fileName);
	}else if(defaultFilter == "Vector graphic (*.pdf)"){
		saved = this->plotArea->savePdf(fileName);
	}else if(defaultFilter == "CSV (*.csv)"){
		saved = this->plotArea->saveAllCurvesToFile(fileName);
	}
	if(saved){
		emit info(tr("Plot saved to ") + fileName);
	}else{
		emit error(tr("Could not save plot to disk."));
	}
}


void PlotWindow1D::enableRawGrabbing(bool enable) {
	this->rawGrabbingAllowed = enable;
}

void PlotWindow1D::enableProcessedGrabbing(bool enable) {
	this->processedGrabbingAllowed = enable;
}

void PlotWindow1D::enableBitshift(bool enable) {
	this->bitshift = enable;
}

void PlotWindow1D::toggleDualCoordinates(bool enabled) {
	this->dataCursorEnabled = enabled;
	if (enabled){
		this->plotArea->setCursor(Qt::CrossCursor);
	} else {
		this->plotArea->unsetCursor();
		this->coordinateDisplay->setVisible(false);
	}
}

void PlotWindow1D::toggleLegend(bool enabled) {
	this->plotArea->legend->setVisible(enabled);
	this->plotArea->replot();
}

void PlotWindow1D::toggleStatsInLegend(bool enable) {
	this->showStatistics = enable;
	this->statsLabel->setVisible(enable);

	if (enable) {
		this->statsLabel->setRawStatsVisible(this->displayRaw);
		this->statsLabel->setProcessedStatsVisible(this->displayProcessed);
	}
}

void PlotWindow1D::setShowInfoInLegend(bool show) {
	this->showInfoInLegend = show;
	this->plotArea->setShowInfoInLegend(show);
}

void PlotWindow1D::enableRawPlotUpdate() {
	this->canUpdateRawPlot = true;
}

void PlotWindow1D::enableProcessedPlotUpdate() {
	this->canUpdateProcessedPlot = true;
}

void PlotWindow1D::handleCursorCoordinates(QPointF rawCoords, QPointF processedCoords, bool isOnPlotting) {
	// Only update the coordinate overlay if data cursor is enabled and mouse is on plotting area
	if (this->dataCursorEnabled && isOnPlotting) {
		QString labelText;
		bool hasPrevious = false;

		// Only show raw coordinates if the raw plot is enabled
		if (this->displayRaw) {
			labelText += QString("Raw: (%1, %2)")
				.arg(rawCoords.x(), 0, 'f', 2)
				.arg(rawCoords.y(), 0, 'f', 2);
			hasPrevious = true;
		}

		// Only show processed coordinates if the processed plot is enabled
		if (this->displayProcessed) {
			if (hasPrevious) {
				labelText += " \n";
			}
			labelText += QString("Processed: (%1, %2)")
				.arg(processedCoords.x(), 0, 'f', 2)
				.arg(processedCoords.y(), 0, 'f', 2);
		}

		// Update and position the overlay label near the mouse cursor
		QPoint mousePos = QCursor::pos();
		QPoint localPos = this->mapFromGlobal(mousePos);

		this->coordinateDisplay->setText(labelText);
		this->coordinateDisplay->adjustSize();
		this->coordinateDisplay->move(localPos + QPoint(10, -10));
		this->coordinateDisplay->setVisible(true);
	} else {
		this->coordinateDisplay->setVisible(false);
	}
}


// ---------------------- StatsLabel Implementation ----------------------
StatsLabel::StatsLabel(QWidget *parent) : QLabel(parent),
	rawStatsVisible(false), processedStatsVisible(false),
	rawMin(0), rawMax(0), rawMean(0), rawStdDev(0),
	procMin(0), procMax(0), procMean(0), procStdDev(0),
	lineNr(0), bufferNrRaw(0), bufferNrProcessed(0)
{
	//this->setFixedWidth(120); //on some systems this is enough on other it is too small
	//to ensure that the label width is large enough we estimate it using QFontMetrics and the dummy text "Raw Buffer Nr.: 0000"
	QFont font = this->font();
	QFontMetrics fm(font);
	int width = fm.horizontalAdvance("Raw Buffer Nr.: 0000");
	width += 20;
	this->setFixedWidth(width);

	this->setAlignment(Qt::AlignLeft | Qt::AlignTop);
	this->setStyleSheet("QLabel { background-color: rgba(50, 50, 50, 200); color: white; padding: 10px; border-radius: 5px; }");

	QString statsText = "<b>Statistics</b><br><br>";
	statsText += "<b>Raw Data</b><br>";
	statsText += "No raw data<br><br>";
	statsText += "<b>Processed Data</b><br>";
	statsText += "No processed data";
	this->setText(statsText);
}

void StatsLabel::updateRawStats(qreal min, qreal max, qreal mean, qreal stdDeviation, int lineNr, int bufferNr) {
	this->rawMin = min;
	this->rawMax = max;
	this->rawMean = mean;
	this->rawStdDev = stdDeviation;
	this->lineNr = lineNr;
	this->bufferNrRaw = bufferNr;

	this->refreshDisplay();
}

void StatsLabel::updateProcessedStats(qreal min, qreal max, qreal mean, qreal stdDeviation, int lineNr, int bufferNr) {
	this->procMin = min;
	this->procMax = max;
	this->procMean = mean;
	this->procStdDev = stdDeviation;
	this->lineNr = lineNr;
	this->bufferNrProcessed = bufferNr;

	this->refreshDisplay();
}

void StatsLabel::setRawStatsVisible(bool visible) {
	this->rawStatsVisible = visible;
	this->refreshDisplay();
}

void StatsLabel::setProcessedStatsVisible(bool visible) {
	this->processedStatsVisible = visible;
	this->refreshDisplay();
}

void StatsLabel::refreshDisplay() {
	QString bufferNrStringRaw;
	QString bufferNrStringProcessed;
	QString statsText = "<b>Statistics</b><br><br>";

	if (rawStatsVisible) {
		statsText += "<b>Raw</b><br>";
		statsText += QString("Min: %1<br>").arg(rawMin, 0, 'f', 2);
		statsText += QString("Max: %1<br>").arg(rawMax, 0, 'f', 2);
		statsText += QString("Mean: %1<br>").arg(rawMean, 0, 'f', 2);
		statsText += QString("Std Dev: %1<br>").arg(rawStdDev, 0, 'f', 2);
		bufferNrStringRaw = QString("Raw Buffer Nr.: %1<br>").arg(this->bufferNrRaw, 2, 10, QChar('0'));
		statsText += "<br>";
	} else {
		bufferNrStringRaw = "";
	}

	if (processedStatsVisible) {
		statsText += "<b>Processed</b><br>";
		statsText += QString("Min: %1<br>").arg(procMin, 0, 'f', 2);
		statsText += QString("Max: %1<br>").arg(procMax, 0, 'f', 2);
		statsText += QString("Mean: %1<br>").arg(procMean, 0, 'f', 2);
		statsText += QString("Std Dev: %1<br>").arg(procStdDev, 0, 'f', 2);
		bufferNrStringProcessed = QString("Proc Buffer Nr.: %1<br>").arg(this->bufferNrProcessed, 2, 10, QChar('0'));
		statsText += "<br>";
	} else {
		bufferNrStringProcessed = "";
	}

	statsText += QString("Line Nr.: %1<br>").arg(this->lineNr, 2, 10, QChar('0'));
	statsText += bufferNrStringRaw;
	statsText += bufferNrStringProcessed;

	setText(statsText);
}

void StatsLabel::enterEvent(QEvent *event) {
	emit mouseEntered();
	QLabel::enterEvent(event);
}


// ---------------------- PlotArea1D Implementation ----------------------
PlotArea1D::PlotArea1D(QWidget *parent) : QCustomPlot(parent),
	draggingLegend(false),
	showInfoInLegend(false)
{
	this->rawLineName = tr("Raw Line");
	this->processedLineName = tr("A-scan");

	this->setBackground(QColor(50, 50, 50));
	this->axisRect()->setBackground(QColor(55, 55, 55));

	this->addGraph();
	this->addGraph(this->xAxis2, this->yAxis2);
	this->processedColor = QColor(72, 99, 160);
	this->rawColor = QColor(Qt::white);
	this->plotLayout()->setMargins(QMargins(0,0,0,0));

	this->axisRect()->setMinimumMargins(QMargins(0, 0, 0, 0));
	this->axisRect()->setMargins(QMargins(0, 0, 0, 0));

	QPen rawLinePen = QPen(QColor(250, 250, 250));
	rawLinePen.setWidth(1);
	this->graph(0)->setPen(rawLinePen);
	this->graph(0)->setName(this->rawLineName);
	this->xAxis->setLabel("Sample number");
	this->yAxis->setLabel("Raw value");

	this->yAxis2->setVisible(true);
	this->xAxis2->setVisible(true);
	QPen processedLinePen = QPen(processedColor);
	processedLinePen.setWidth(1); //info: line width greater 1 slows down plotting
	this->graph(1)->setPen(processedLinePen);
	this->graph(1)->setName(this->processedLineName);
	this->xAxis2->setLabel("Sample number");
	this->yAxis2->setLabel("Processed value");

	this->setRawAxisColor(rawColor);
	this->setProcessedAxisColor(processedColor);

	this->legend->setVisible(true);
	this->legend->setBrush(QColor(80, 80, 80, 200));
	this->legend->setTextColor(Qt::white);
	this->legend->setBorderPen(QColor(180, 180, 180, 200));
	this->axisRect()->insetLayout()->setInsetPlacement(0, QCPLayoutInset::ipFree);

	connect(this, &QCustomPlot::beforeReplot, this, &PlotArea1D::preventStretching);

	//this->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectAxes | QCP::iSelectPlottables); //plot is slow if iSelectPlottable is set
	this->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectAxes);
	connect(this, &QCustomPlot::selectionChangedByUser, this, &PlotArea1D::combineSelections);
	connect(this, &QCustomPlot::mouseWheel, this, &PlotArea1D::zoomSelectedAxisWithMouseWheel);
	connect(this, &QCustomPlot::mousePress, this, &PlotArea1D::dragSelectedAxes);
}

void PlotArea1D::setRawAxisColor(QColor color) {
	this->rawColor = color;
	this->xAxis->setBasePen(QPen(color, 1));
	this->yAxis->setBasePen(QPen(color, 1));
	this->xAxis->setTickPen(QPen(color, 1));
	this->yAxis->setTickPen(QPen(color, 1));
	this->xAxis->setSubTickPen(QPen(color, 1));
	this->yAxis->setSubTickPen(QPen(color, 1));
	this->xAxis->setTickLabelColor(color);
	this->yAxis->setTickLabelColor(color);
	this->xAxis->setLabelColor(color);
	this->yAxis->setLabelColor(color);
}

void PlotArea1D::setProcessedAxisColor(QColor color) {
	this->processedColor = color;
	this->xAxis2->setBasePen(QPen(color, 1));
	this->yAxis2->setBasePen(QPen(color, 1));
	this->xAxis2->setTickPen(QPen(color, 1));
	this->yAxis2->setTickPen(QPen(color, 1));
	this->xAxis2->setSubTickPen(QPen(color, 1));
	this->yAxis2->setSubTickPen(QPen(color, 1));
	this->xAxis2->setTickLabelColor(color);
	this->yAxis2->setTickLabelColor(color);
	this->xAxis2->setLabelColor(color);
	this->yAxis2->setLabelColor(color);
}

void PlotArea1D::setRawPlotVisible(bool visible) {
	int transparency = visible ? 255 : 0;
	this->rawColor.setAlpha(transparency);
	this->setRawAxisColor(this->rawColor);
	this->graph(0)->setVisible(visible);
	visible ? this->graph(0)->addToLegend() : this->graph(0)->removeFromLegend();
}

void PlotArea1D::setProcessedPlotVisible(bool visible) {
	int transparency = visible ? 255 : 0;
	this->processedColor.setAlpha(transparency);
	this->setProcessedAxisColor(this->processedColor);
	this->graph(1)->setVisible(visible);
	visible ? this->graph(1)->addToLegend() : this->graph(1)->removeFromLegend();
}

QVariantMap PlotArea1D::getLegendSettings() const {
	QVariantMap settings;

	QRectF rect = this->axisRect()->insetLayout()->insetRect(0);
	settings.insert(PLOT1D_LEGEND_X, rect.x());
	settings.insert(PLOT1D_LEGEND_Y, rect.y());

	QCPLayoutInset::InsetPlacement placement = this->axisRect()->insetLayout()->insetPlacement(0);
	settings.insert(PLOT1D_LEGEND_PLACEMENT, static_cast<int>(placement));

	if (placement == QCPLayoutInset::ipBorderAligned) {
		int alignment = static_cast<int>(this->axisRect()->insetLayout()->insetAlignment(0));
		settings.insert(PLOT1D_LEGEND_ALIGNMENT, alignment);
	}

	return settings;
}

void PlotArea1D::applyLegendSettings(const QVariantMap& settings) {
	if (settings.contains(PLOT1D_LEGEND_X) && settings.contains(PLOT1D_LEGEND_Y)) {
		double xPos = settings.value(PLOT1D_LEGEND_X, 0.99).toDouble();
		double yPos = settings.value(PLOT1D_LEGEND_Y, 0.01).toDouble();

		//restore placement type
		QCPLayoutInset::InsetPlacement placement = static_cast<QCPLayoutInset::InsetPlacement>(
			settings.value(PLOT1D_LEGEND_PLACEMENT, QCPLayoutInset::ipFree).toInt()
		);
		this->axisRect()->insetLayout()->setInsetPlacement(0, placement);

		//restore alignment if applicable
		if (placement == QCPLayoutInset::ipBorderAligned) {
			// Convert saved integer back to Qt::Alignment
			int savedAlignment = settings.value(PLOT1D_LEGEND_ALIGNMENT,
				static_cast<int>(Qt::AlignTop | Qt::AlignRight)).toInt();
			Qt::Alignment alignment = static_cast<Qt::Alignment>(savedAlignment);
			this->axisRect()->insetLayout()->setInsetAlignment(0, alignment);
		}

		//set the legend position
		this->axisRect()->insetLayout()->setInsetRect(0, QRectF(xPos, yPos, 0, 0));
	} else {
		//default legend position
		this->axisRect()->insetLayout()->setInsetPlacement(0, QCPLayoutInset::ipBorderAligned);
		this->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignTop | Qt::AlignRight);
		this->axisRect()->insetLayout()->setInsetRect(0, QRectF(0.99, 0.01, 0, 0));
	}
}

bool PlotArea1D::saveAllCurvesToFile(QString fileName) {
	int numberOfRawSamples = this->graph(0)->data()->size();
	int numberOfProcessedSamples = this->graph(1)->data()->size();
	bool saved = false;
	QFile file(fileName);
	if (file.open(QFile::WriteOnly|QFile::Truncate)) {
		QTextStream stream(&file);
		stream << tr("Sample Number") << ";" << tr("Raw Value") << ";" << tr("Processed Value") << "\n";
		for(int i = 0; i < numberOfRawSamples; i++){
			stream << QString::number(this->graph(0)->data()->at(i)->key) << ";" << QString::number(graph(0)->data()->at(i)->value);
			if(i < numberOfProcessedSamples) {
				stream << ";" << QString::number(graph(1)->data()->at(i)->value) << "\n";
			} else {
				stream << "\n";
			}
		}
		file.close();
		saved = true;
	}
	return saved;
}

void PlotArea1D::setRawData(const QVector<qreal>& xData, const QVector<qreal>& yData) {
	this->graph(0)->setData(xData, yData, true);
}

void PlotArea1D::setProcessedData(const QVector<qreal>& xData, const QVector<qreal>& yData) {
	this->graph(1)->setData(xData, yData, true);
}

void PlotArea1D::updateInfoInRawLegend(int lineNr, int bufferNr){
	QString lineNumberStr = QString("%1").arg(lineNr, 3, 10, QChar('0'));
	QString bufferNumberStr = "   Buffer Nr.: " + QString("%1").arg(bufferNr, 2, 10, QChar('0'));
	this->graph(0)->setName(this->rawLineName + " Nr.: " + lineNumberStr + bufferNumberStr);
}

void PlotArea1D::updateInfoInProcessedLegend(int lineNr, int bufferNr){
	QString lineNumberStr = QString("%1").arg(lineNr, 3, 10, QChar('0'));
	QString bufferNumberStr = "   Buffer Nr.: " + QString("%1").arg(bufferNr, 2, 10, QChar('0'));
	this->graph(1)->setName(this->processedLineName + " Nr.: " + lineNumberStr + bufferNumberStr);
}

void PlotArea1D::setShowInfoInLegend(bool show) {
	this->showInfoInLegend = show;

	if(!show){
		this->graph(0)->setName(this->rawLineName);
		this->graph(1)->setName(this->processedLineName);
	}

	if (graph(0)->visible() || graph(1)->visible()) {
		replot();
	}
}

void PlotArea1D::contextMenuEvent(QContextMenuEvent *event) {
	//forward the context menu event to the parent widget
	if (parentWidget()) {
		QCoreApplication::sendEvent(parentWidget(), event);
	}
}

void PlotArea1D::mouseDoubleClickEvent(QMouseEvent *event) {
	Q_UNUSED(event);
	this->rescaleAxes();
	this->replot();
}

void PlotArea1D::mousePressEvent(QMouseEvent *event) {
	//check if this is a legend drag
	if (event->button() == Qt::LeftButton && this->legend->selectTest(event->pos(), false) > 0) {
		draggingLegend = true;
		this->setCursor(Qt::ClosedHandCursor);
		emit legendDragging(true);

		//if legend isn't already in free placement mode, switch to it now
		if (this->axisRect()->insetLayout()->insetPlacement(0) != QCPLayoutInset::ipFree) {
			//get current position before switching to free mode
			QPointF currentPos;
			QRectF legendRect = this->legend->outerRect();

			//convert to normalized coordinates
			currentPos.setX((legendRect.x() - this->axisRect()->left()) / (double)this->axisRect()->width());
			currentPos.setY((legendRect.y() - this->axisRect()->top()) / (double)this->axisRect()->height());

			//switch to free placement
			this->axisRect()->insetLayout()->setInsetPlacement(0, QCPLayoutInset::ipFree);
			this->axisRect()->insetLayout()->setInsetRect(0, QRectF(currentPos, QSizeF(0, 0)));
		}

		//set drag origin
		QPointF mousePoint(
			(event->pos().x() - this->axisRect()->left()) / (double)this->axisRect()->width(),
			(event->pos().y() - this->axisRect()->top()) / (double)this->axisRect()->height()
		);
		dragLegendOrigin = mousePoint - this->axisRect()->insetLayout()->insetRect(0).topLeft();

		event->accept();
	} else {
		QCustomPlot::mousePressEvent(event);
	}
}

void PlotArea1D::mouseMoveEvent(QMouseEvent* event) {
	if (draggingLegend) {
		QRectF rect = this->axisRect()->insetLayout()->insetRect(0);
		QPointF mousePoint(
			(event->pos().x() - this->axisRect()->left()) / (double)this->axisRect()->width(),
			(event->pos().y() - this->axisRect()->top()) / (double)this->axisRect()->height()
		);
		QPointF newPos = mousePoint - dragLegendOrigin;
		rect.moveTopLeft(newPos);
		this->axisRect()->insetLayout()->setInsetRect(0, rect);
		this->replot();
		event->accept();
		return;
	}

	// Calculate cursor coordinates and emit them
	QPointF rawCoords, processedCoords;
	bool isOnPlotting = this->rect().contains(event->pos()) && !(this->legend->selectTest(event->pos(), false) > 0);

	if (isOnPlotting) {
		rawCoords = QPointF(
			this->xAxis->pixelToCoord(event->pos().x()),
			this->yAxis->pixelToCoord(event->pos().y())
		);

		processedCoords = QPointF(
			this->xAxis2->pixelToCoord(event->pos().x()),
			this->yAxis2->pixelToCoord(event->pos().y())
		);

		emit cursorCoordinates(rawCoords, processedCoords, true);
	} else {
		emit cursorCoordinates(rawCoords, processedCoords, false);
	}

	// Cursor handling
	if (this->legend->selectTest(event->pos(), false) > 0) {
		this->setCursor(Qt::OpenHandCursor);
	} else {
		this->unsetCursor();
	}

	// Call the base class implementation to preserve default behavior
	QCustomPlot::mouseMoveEvent(event);
}

void PlotArea1D::mouseReleaseEvent(QMouseEvent *event) {
	if (draggingLegend) {
		draggingLegend = false;
		emit legendDragging(false);
		// Restore cursor
		if (this->legend->selectTest(event->pos(), false) > 0) {
			this->setCursor(Qt::OpenHandCursor);
		} else {
			this->unsetCursor();
		}
		event->accept();
	} else {
		QCustomPlot::mouseReleaseEvent(event);
	}
}

void PlotArea1D::zoomSelectedAxisWithMouseWheel() {
	QList<QCPAxis*> selectedAxes;
	if (this->xAxis->selectedParts().testFlag(QCPAxis::spAxis)) {
		selectedAxes.append(this->xAxis);
	}
	else if (this->yAxis->selectedParts().testFlag(QCPAxis::spAxis)) {
		selectedAxes.append(this->yAxis);
	}
	else if (this->xAxis2->selectedParts().testFlag(QCPAxis::spAxis)) {
		selectedAxes.append(this->xAxis2);
	}
	else if (this->yAxis2->selectedParts().testFlag(QCPAxis::spAxis)) {
		selectedAxes.append(this->yAxis2);
	}
	else {
		//no axis is selected --> enable zooming for all axes
		selectedAxes.append(this->xAxis);
		selectedAxes.append(this->yAxis);
		selectedAxes.append(this->xAxis2);
		selectedAxes.append(this->yAxis2);
	}

	this->axisRect()->setRangeZoomAxes(selectedAxes);
}

void PlotArea1D::dragSelectedAxes() {
	QList<QCPAxis*> selectedAxes;
	if (this->xAxis->selectedParts().testFlag(QCPAxis::spAxis)) {
		selectedAxes.append(this->xAxis);
	}
	else if (this->yAxis->selectedParts().testFlag(QCPAxis::spAxis)) {
		selectedAxes.append(this->yAxis);
	}
	else if (this->xAxis2->selectedParts().testFlag(QCPAxis::spAxis)) {
		selectedAxes.append(this->xAxis2);
	}
	else if (this->yAxis2->selectedParts().testFlag(QCPAxis::spAxis)) {
		selectedAxes.append(this->yAxis2);
	}
	else {
		//no axis is selected --> enable dragging for all axes
		selectedAxes.append(this->xAxis);
		selectedAxes.append(this->yAxis);
		selectedAxes.append(this->xAxis2);
		selectedAxes.append(this->yAxis2);
	}

	this->axisRect()->setRangeDragAxes(selectedAxes);
}

void PlotArea1D::combineSelections() {
	// axis label, axis and tick labels should act as a s single selectable item
	if (this->xAxis->selectedParts().testFlag(QCPAxis::spAxisLabel) || this->xAxis->selectedParts().testFlag(QCPAxis::spAxis) || this->xAxis->selectedParts().testFlag(QCPAxis::spTickLabels)) {
		this->xAxis->setSelectedParts(QCPAxis::spAxis | QCPAxis::spTickLabels | QCPAxis::spAxisLabel);
	}
	if (this->xAxis2->selectedParts().testFlag(QCPAxis::spAxisLabel) || this->xAxis2->selectedParts().testFlag(QCPAxis::spAxis) || this->xAxis2->selectedParts().testFlag(QCPAxis::spTickLabels)) {
		this->xAxis2->setSelectedParts(QCPAxis::spAxis | QCPAxis::spTickLabels | QCPAxis::spAxisLabel);
	}
	if (this->yAxis->selectedParts().testFlag(QCPAxis::spAxisLabel) || this->yAxis->selectedParts().testFlag(QCPAxis::spAxis) || this->yAxis->selectedParts().testFlag(QCPAxis::spTickLabels)) {
		this->yAxis->setSelectedParts(QCPAxis::spAxis | QCPAxis::spTickLabels | QCPAxis::spAxisLabel);
	}
	if (this->yAxis2->selectedParts().testFlag(QCPAxis::spAxisLabel) || this->yAxis2->selectedParts().testFlag(QCPAxis::spAxis) || this->yAxis2->selectedParts().testFlag(QCPAxis::spTickLabels)) {
		this->yAxis2->setSelectedParts(QCPAxis::spAxis | QCPAxis::spTickLabels | QCPAxis::spAxisLabel);
	}
}

void PlotArea1D::preventStretching() {
	//this is to prevent the legend from stretching if the plot is stretched.see: https://www.qcustomplot.com/index.php/support/forum/481
	//with the current QCustomPlot version, this may not be necessary anymore --> todo: check if this is necessary.
	if (this->axisRect()->insetLayout()->insetPlacement(0) == QCPLayoutInset::ipFree) {
		this->legend->setMaximumSize(this->legend->minimumOuterSizeHint());
	}
}


// ---------------------- ControlPanel1D Implementation ----------------------
ControlPanel1D::ControlPanel1D(QWidget *parent) : QWidget(parent) {
	this->panel = new QWidget(this);
	QPalette pal;
	pal.setColor(QPalette::Background, QColor(32,32,32,128));
	this->panel->setAutoFillBackground(true);
	this->panel->setPalette(pal);
	this->labelLines = new QLabel(tr("Line Nr:"), this->panel);
	this->slider = new QSlider(Qt::Horizontal, this->panel);
	this->slider->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);
	this->slider->setMinimum(0);
	this->spinBoxLine = new QSpinBox(this->panel);
	this->spinBoxLine->setMinimum(0);
	this->checkBoxRaw = new QCheckBox(tr("Raw"), this->panel);
	this->checkBoxProcessed = new QCheckBox(tr("Processed"), this->panel);
	this->checkBoxAutoscale = new QCheckBox(tr("Autoscale Axis"), this->panel);

	this->widgetLayout = new QHBoxLayout(this);
	this->widgetLayout->setContentsMargins(0, 0, 0, 0);
	this->widgetLayout->addWidget(this->panel);

	this->layout = new QGridLayout(this->panel);
	this->layout->setContentsMargins(3,0,3,0);
	this->layout->setVerticalSpacing(1);
	this->layout->addWidget(this->labelLines, 0, 0, 1, 1, Qt::AlignRight);
	this->layout->addWidget(this->slider, 0, 1, 1, 5);
	this->layout->setColumnStretch(4, 100); //set high stretch factor such that slider gets max available space
	this->layout->addWidget(this->spinBoxLine, 0, 6, 1, 1, Qt::AlignLeft);
	this->layout->addWidget(this->checkBoxRaw, 1, 0, 1, 1, Qt::AlignRight);
	this->layout->addWidget(this->checkBoxProcessed, 1, 1, 1, 1, Qt::AlignRight);
	this->layout->addWidget(this->checkBoxAutoscale, 1, 3, 1, 1, Qt::AlignRight);

	connect(this->slider, &QSlider::valueChanged, this->spinBoxLine, &QSpinBox::setValue);
	connect(this->spinBoxLine, QOverload<int>::of(&QSpinBox::valueChanged), this->slider, &QSlider::setValue);
}

ControlPanel1D::~ControlPanel1D() {
}

void ControlPanel1D::setMaxLineNr(unsigned int maxLineNr) {
	this->slider->setMaximum(maxLineNr);
	this->spinBoxLine->setMaximum(maxLineNr);
}

void ControlPanel1D::enterEvent(QEvent *event) {
	emit mouseEntered();
	QWidget::enterEvent(event);
}
