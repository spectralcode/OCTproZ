/*
MIT License

Copyright (c) 2019-2021 Miroslav Zabic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef OCTSYSTEMSIMULATORPLUGIN_H
#define OCTSYSTEMSIMULATORPLUGIN_H

#define STREAM_BUFFER_SIZE 2097152

#include <QObject>
#include <QCoreApplication>
#include <QThread>
#include <QDir>
#include <QDebug>
#include "math.h"
#include "virtualoctsystemsettingsdialog.h"
#include "octproz_devkit.h"
#include <fstream>


class VirtualOCTSystem : public AcquisitionSystem
{
	Q_OBJECT
	Q_PLUGIN_METADATA(IID AcquisitionSystem_iid)
	Q_INTERFACES(AcquisitionSystem)

public:
	explicit VirtualOCTSystem();
	~VirtualOCTSystem();

	virtual void startAcquisition() override;
	virtual void stopAcquisition() override;
	virtual void settingsLoaded(QVariantMap settings) override;

private:
	FILE* file;
	VirtualOCTSystemSettingsDialog* systemDialog;
	simulatorParams currParams;
	AcquisitionBuffer* streamBuffer;

	bool init();
	void cleanup();
	bool openFileToCopyToRam();
	void acqcuisitionSimulation();
	void acqcuisitionSimulationLargeFile();
	void acquisitionSimulationWithMultiFileBuffers();

public slots:
	void slot_updateParams(simulatorParams newParams);

signals:
	void enableGui(bool enable);
};

#endif // OCTSYSTEMSIMULATORPLUGIN_H
