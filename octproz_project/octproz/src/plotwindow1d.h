/**
 * PlotWindow1D:
 * A 1D plotting widget for visualizing single raw spectra
 * as well as single A-scans
 *
 * Notable features:
 * - Plotting of raw and processed data with separate axes
 * - User can select axis and customize zoom
 * - Dragable Legend with optional statistical information display
 * - Data cursor for coordinate display at mouse position
 * - Export capabilities to various formats (PNG, PDF, CSV)
 * - Line selection with a slider in ControlPanel1D
 *
 * Class Structure:
 * - PlotWindow1D: Main container widget that coordinates all components
 * - PlotArea1D: The actual plotting area (based on QCustomPlot) 
 * - StatsLabel:  A widget displaying statistical information. Can be toggled through the context menu
 * - ControlPanel1D: Control panel with UI elements for adjusting plot settings (enable raw plot, enable processed plot, select line nr, ...)
 * 
 * Known Limitation:
 * With the current implementation there is no synchronization between 
 * raw and processed data plots. The graphs update independently of each other
 * which may lead to plots that show a raw spectrum and a processed A-scan that
 * do not belong to each other. This usually occurs at very high update rates.
 */


#ifndef PLOTWINDOW1D_H
#define PLOTWINDOW1D_H

#define PLOT1D_DISPLAY_RAW "plot1d_display_raw"
#define PLOT1D_DISPLAY_PROCESSED "plot1d_display_processed"
#define PLOT1D_AUTOSCALING "plot1d_autoscaling_enabled"
#define PLOT1D_BITSHIFT "plot1d_bitshift_enabled"
#define PLOT1D_LINE_NR "plot1d_line_nr"
#define PLOT1D_DATA_CURSOR "plot1d_data_cursor_enabled"
#define PLOT1D_SHOW_LEGEND "plot1d_show_legend"
#define PLOT1D_LEGEND_X "plot1d_legend_position_x"
#define PLOT1D_LEGEND_Y "plot1d_legend_position_y"
#define PLOT1D_LEGEND_PLACEMENT "plot1d_legend_placement"
#define PLOT1D_LEGEND_ALIGNMENT "plot1d_legend_alignment"
#define PLOT1D_SHOW_STATS "plot1d_show_stats"
#define PLOT1D_INFO_IN_LEGEND "plot1d_show_info_in_legend"

#define PLOT1D_XAXIS_MIN "plot1d_xaxis_min"
#define PLOT1D_XAXIS_MAX "plot1d_xaxis_max"
#define PLOT1D_YAXIS_MIN "plot1d_yaxis_min"
#define PLOT1D_YAXIS_MAX "plot1d_yaxis_max"
#define PLOT1D_XAXIS2_MIN "plot1d_xaxis2_min"
#define PLOT1D_XAXIS2_MAX "plot1d_xaxis2_max"
#define PLOT1D_YAXIS2_MIN "plot1d_yaxis2_min"
#define PLOT1D_YAXIS2_MAX "plot1d_yaxis2_max"

#include "qcustomplot.h"
#include "octproz_devkit.h"


class PlotArea1D;
class StatsLabel;
class ControlPanel1D;

class PlotWindow1D : public QWidget
{
	Q_OBJECT
public:
	explicit PlotWindow1D(QWidget *parent = nullptr);
	~PlotWindow1D();

	QString getName() const { return this->name; }
	void setName(const QString &name) { this->name = name; }
	void setSettings(QVariantMap settings);
	QVariantMap getSettings();

	QSize sizeHint() const override;
	
	void setRawPlotVisible(bool visible);
	void setProcessedPlotVisible(bool visible);

	QVector<qreal> sampleNumbers;
	QVector<qreal> sampleValues;
	QVector<qreal> sampleNumbersProcessed;
	QVector<qreal> sampleValuesProcessed;


private:
	int line;
	int linesPerBuffer;
	int currentRawBitdepth;
	bool isPlottingRaw;
	bool isPlottingProcessed;
	bool autoscaling;
	bool displayRaw;
	bool displayProcessed;
	bool bitshift;
	bool rawGrabbingAllowed;
	bool processedGrabbingAllowed;
	bool showStatistics;
	bool showInfoInLegend;
	QString name;
	
	QTimer* rawUpdateTimer;
	QTimer* processedUpdateTimer;
	bool canUpdateRawPlot;
	bool canUpdateProcessedPlot;

	PlotArea1D* plotArea;
	StatsLabel* statsLabel;
	ControlPanel1D* panel;

	QHBoxLayout* mainLayout;// Contains plotArea and statsDisplay
	QVBoxLayout* containerLayout; // Contains mainLayout and panel

	bool dataCursorEnabled;
	QLabel* coordinateDisplay;

protected:
	void contextMenuEvent(QContextMenuEvent* event) override;
	void mouseDoubleClickEvent(QMouseEvent *event) override;
	void leaveEvent(QEvent* event) override;

signals:
	void info(QString info);
	void error(QString error);

public slots:
	void plotRawData(void* buffer, unsigned bitDepth, unsigned int samplesPerLine, unsigned int linesPerFrame, unsigned int framesPerBuffer, unsigned int buffersPerVolume, unsigned int currentBufferNr);
	void plotProcessedData(void* buffer, unsigned bitDepth, unsigned int samplesPerLine, unsigned int linesPerFrame, unsigned int framesPerBuffer, unsigned int buffersPerVolume, unsigned int currentBufferNr);
	void changeLinesPerBuffer(int linesPerBuffer);
	void setLineNr(int lineNr);
	void enableDisplayRaw(bool display);
	void enableDisplayProcessed(bool display);
	void activateAutoscaling(bool activate);
	void saveToDisk();
	void enableRawGrabbing(bool enable);
	void enableProcessedGrabbing(bool enable);
	void enableBitshift(bool enable);
	void toggleDualCoordinates(bool enabled);
	void toggleLegend(bool enabled);
	void toggleStatsInLegend(bool enable);
	void setShowInfoInLegend(bool show);
	void enableRawPlotUpdate();
	void enableProcessedPlotUpdate();

	void handleCursorCoordinates(QPointF rawCoords, QPointF processedCoords, bool isOnPlotting);
};


/**
 * @brief The StatsLabel class shows data statistics in a simple label.
 */
class StatsLabel : public QLabel
{
	Q_OBJECT
public:
	explicit StatsLabel(QWidget *parent = nullptr);

	void updateRawStats(qreal min, qreal max, qreal mean, qreal stdDeviation, int lineNr, int bufferNr);
	void updateProcessedStats(qreal min, qreal max, qreal mean, qreal stdDeviation, int lineNr, int bufferNr);
	void setRawStatsVisible(bool visible);
	void setProcessedStatsVisible(bool visible);
	void refreshDisplay();

protected:
	void enterEvent(QEvent *event) override;

private:
	bool rawStatsVisible;
	bool processedStatsVisible;

	qreal rawMin;
	qreal rawMax;
	qreal rawMean;
	qreal rawStdDev;

	qreal procMin;
	qreal procMax;
	qreal procMean;
	qreal procStdDev;

	int lineNr;
	int bufferNrRaw;
	int bufferNrProcessed;

signals:
	void mouseEntered();
};

/**
 * @brief The PlotArea1D class handles the actual plotting functionality
 */
class PlotArea1D : public QCustomPlot
{
	Q_OBJECT
public:
	explicit PlotArea1D(QWidget *parent = nullptr);

	void setRawAxisColor(QColor color);
	void setProcessedAxisColor(QColor color);
	void setRawPlotVisible(bool visible);
	void setProcessedPlotVisible(bool visible);

	QVariantMap getLegendSettings() const;
	void applyLegendSettings(const QVariantMap& settings);

	bool saveAllCurvesToFile(QString fileName);

	void setRawData(const QVector<qreal>& xData, const QVector<qreal>& yData);
	void setProcessedData(const QVector<qreal>& xData, const QVector<qreal>& yData);

	QString getRawLineName() const { return rawLineName; }
	QString getProcessedLineName() const { return processedLineName; }
	void updateLineNumbersAndBufferNumbers(int lineNumber, int bufferNumber);

	void setShowInfoInLegend(bool show);
	bool isInfoInLegendEnabled() const { return this->showInfoInLegend; }
	void updateInfoInRawLegend(int lineNr, int bufferNr);
	void updateInfoInProcessedLegend(int lineNr, int bufferNr);
	

protected:
	void mousePressEvent(QMouseEvent *event) override;
	void mouseMoveEvent(QMouseEvent* event) override;
	void mouseReleaseEvent(QMouseEvent *event) override;
	void contextMenuEvent(QContextMenuEvent* event) override;
	void mouseDoubleClickEvent(QMouseEvent *event) override;

private:
	QColor rawColor;
	QColor processedColor;
	QString rawLineName;
	QString processedLineName;

	bool draggingLegend;
	QPointF dragLegendOrigin;
	
	bool showInfoInLegend;

public slots:
	void zoomSelectedAxisWithMouseWheel();
	void dragSelectedAxes();
	void combineSelections();
	void preventStretching();

signals:
	void legendDragging(bool isDragging);
	void cursorCoordinates(QPointF rawCoords, QPointF processedCoords, bool isOnPlotting);
};


class ControlPanel1D : public QWidget
{
	Q_OBJECT

public:
	ControlPanel1D(QWidget* parent);
	~ControlPanel1D();

	void setMaxLineNr(unsigned int maxLineNr);

private:
	QWidget* panel;
	QLabel* labelLines;
	QSlider* slider;
	QSpinBox* spinBoxLine;

	QCheckBox* checkBoxRaw;
	QCheckBox* checkBoxProcessed;
	QCheckBox* checkBoxAutoscale;

	QHBoxLayout* widgetLayout;
	QGridLayout* layout;

protected:
	void enterEvent(QEvent *event) override;

public slots:

signals:
	void mouseEntered();

friend class PlotWindow1D;
};

#endif // PLOTWINDOW1D_H
