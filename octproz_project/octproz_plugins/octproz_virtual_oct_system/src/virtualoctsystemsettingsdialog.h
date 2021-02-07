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

#pragma once

#define SYSNAME "sys_name"
#define FILEPATH "file_path"
#define BITDEPTH "bit_depth"
#define WIDTH "width"
#define HEIGHT "height"
#define DEPTH "depth"
#define BUFFERS_PER_VOLUME "buffers_per_volume"
#define BUFFERS_FROM_FILE "buffers_from_file"
#define WAITTIME "wait_time"
#define COPY_TO_RAM "copy_file_to_ram"


#include <qstandardpaths.h>
#include <qvariant.h>
#include <QDialog>
#include <QString>
#include <QFileDialog>
#include "ui_virtualoctsystemsettingsdialog.h"

struct simulatorParams {
	QString filePath;
	int bitDepth;
	int width;
	int height;
	int depth;
	int buffersPerVolume;
	int buffersFromFile;
	int waitTimeUs;
	bool copyFileToRam;
};

class VirtualOCTSystemSettingsDialog : public QDialog
{
	Q_OBJECT

public:
	VirtualOCTSystemSettingsDialog(QWidget *parent = nullptr);
	~VirtualOCTSystemSettingsDialog();

	void setSettings(QVariantMap settings);
	void getSettings(QVariantMap* settings);


private:
	Ui::VirtualOCTSystemSettingsDialog* ui;
	simulatorParams params;

	void initGui();

public slots:
	void slot_selectFile();
	void slot_apply();
	void slot_enableGui(bool enable);
	void slot_checkWidthValue();

signals:
	void settingsUpdated(simulatorParams newParams);
};
