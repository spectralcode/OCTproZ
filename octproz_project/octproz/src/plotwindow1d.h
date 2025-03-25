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


#include "qcustomplot.h"
#include "octproz_devkit.h"

class ControlPanel1D;
class PlotWindow1D : public QCustomPlot
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

	QVector<qreal> sampleNumbers;
	QVector<qreal> sampleValues;
	QVector<qreal> sampleNumbersProcessed;
	QVector<qreal> sampleValuesProcessed;


private:
	void setRawAxisColor(QColor color);
	void setProcessedAxisColor(QColor color);
	void setRawPlotVisible(bool visible);
	void setProcessedPlotVisible(bool visible);
	bool saveAllCurvesToFile(QString fileName);

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
	QString rawLineName;
	QString processedLineName;
	QString name;

	QColor processedColor;
	QColor rawColor;

	ControlPanel1D* panel;
	QVBoxLayout* layout;

	bool dataCursorEnabled;
	QLabel* coordinateDisplay;

	bool draggingLegend;
	QPointF dragLegendOrigin;


protected:
	void contextMenuEvent(QContextMenuEvent* event) override;
	void mouseDoubleClickEvent(QMouseEvent *event) override;
	void mouseMoveEvent(QMouseEvent* event) override;
	void leaveEvent(QEvent* event) override;

	void mousePressEvent(QMouseEvent *event) override;
	void mouseReleaseEvent(QMouseEvent *event) override;

signals:
	void info(QString info);
	void error(QString error);


public slots:
	void slot_plotRawData(void* buffer, unsigned bitDepth, unsigned int samplesPerLine, unsigned int linesPerFrame, unsigned int framesPerBuffer, unsigned int buffersPerVolume, unsigned int currentBufferNr);
	void slot_plotProcessedData(void* buffer, unsigned bitDepth, unsigned int samplesPerLine, unsigned int linesPerFrame, unsigned int framesPerBuffer, unsigned int buffersPerVolume, unsigned int currentBufferNr);
	void slot_changeLinesPerBuffer(int linesPerBuffer);
	void slot_setLine(int lineNr);
	void slot_displayRaw(bool display);
	void slot_displayProcessed(bool display);
	void slot_activateAutoscaling(bool activate);
	void slot_saveToDisk();
	void slot_enableRawGrabbing(bool enable);
	void slot_enableProcessedGrabbing(bool enable);
	void slot_enableBitshift(bool enable);
	void slot_toggleDualCoordinates(bool enabled);
	void slot_toggleLegend(bool enabled);
	void zoomSelectedAxisWithMouseWheel();
	void dragSelectedAxes();
	void combineSelections();
	void mousePressOnLegend(QMouseEvent *event);
	void mouseMoveWithLegend(QMouseEvent *event);
	void mouseReleaseFromLegend(QMouseEvent *event);
	void preventStretching();
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
