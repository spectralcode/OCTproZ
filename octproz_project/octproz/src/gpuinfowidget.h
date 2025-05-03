#ifndef GPUINFOWIDGET_H
#define GPUINFOWIDGET_H

#include <QWidget>
#include <QVBoxLayout>
#include <QTextEdit>
#include "gpuinfo.h"

class GpuInfoWidget : public QWidget
{
	Q_OBJECT

public:
	explicit GpuInfoWidget(QWidget* parent = nullptr);
	~GpuInfoWidget();

	bool checkCudaAvailability();

public slots:
	void refreshGpuInfo();

protected:
	void showEvent(QShowEvent* event) override;

private:
	GpuInfo* gpuInfo;
	QVBoxLayout* mainLayout;
	QTextEdit* gpuInfoText;

	void initGui();
	void setupConnections();

signals:
	void info(QString infoMessage);
	void error(QString errorMessage);
};

#endif // GPUINFOWIDGET_H
