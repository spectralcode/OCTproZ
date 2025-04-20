/*
MIT License

Copyright (c) 2019-2022 Miroslav Zabic

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

#include "virtualoctsystemsettingsdialog.h"
#include "virtualoctsystem.h"

VirtualOCTSystemSettingsDialog::VirtualOCTSystemSettingsDialog(QWidget* parent)
	: ui(new Ui::VirtualOCTSystemSettingsDialog) //QDialog(parent)
{
	Q_UNUSED(parent);
	ui->setupUi(this);
	initGui();

	///qRegisterMetaType is needed to enabel Qt::QueuedConnection for signal slot communication with "simulatorParams"
	qRegisterMetaType<simulatorParams >("simulatorParams");
}

VirtualOCTSystemSettingsDialog::~VirtualOCTSystemSettingsDialog()
{
}

void VirtualOCTSystemSettingsDialog::setSettings(QVariantMap settings){
	this->ui->lineEdit->setText(settings.value(FILEPATH).toString());
	this->ui->spinBox_bitDepth->setValue(settings.value(BITDEPTH, 12).toInt());
	this->ui->spinBox_width->setValue(settings.value(WIDTH, 1664).toInt());
	this->ui->spinBox_height->setValue(settings.value(HEIGHT, 512).toInt());
	this->ui->spinBox_depth->setValue(settings.value(DEPTH, 16).toInt());
	this->ui->spinBox_buffersPerVolume->setValue(settings.value(BUFFERS_PER_VOLUME, 16).toInt());
	this->ui->spinBox_buffersFromFile->setValue(settings.value(BUFFERS_FROM_FILE, 16).toInt());
	this->ui->spinBox_bscanOffset->setValue(settings.value(BSCAN_OFFSET, 0).toInt());
	this->ui->spinBox_waitTime->setValue(settings.value(WAITTIME, 0).toInt());
	this->ui->checkBox_copyFileToRam->setChecked(settings.value(COPY_TO_RAM, true).toBool());
	this->ui->checkBox_sync->setChecked(settings.value(SYNC_WITH_PROCESSING, true).toBool());
	this->slot_apply();
}

void VirtualOCTSystemSettingsDialog::getSettings(QVariantMap* settings) {
	settings->insert(FILEPATH, this->ui->lineEdit->text());
	settings->insert(BITDEPTH, this->ui->spinBox_bitDepth->value());
	settings->insert(WIDTH, this->ui->spinBox_width->value());
	settings->insert(HEIGHT, this->ui->spinBox_height->value());
	settings->insert(DEPTH, this->ui->spinBox_depth->value());
	settings->insert(BUFFERS_PER_VOLUME, this->ui->spinBox_buffersPerVolume->value());
	settings->insert(BUFFERS_FROM_FILE, this->ui->spinBox_buffersFromFile->value());
	settings->insert(BSCAN_OFFSET, this->ui->spinBox_bscanOffset->value());
	settings->insert(WAITTIME, this->ui->spinBox_waitTime->value());
	settings->insert(COPY_TO_RAM, this->ui->checkBox_copyFileToRam->isChecked());
	settings->insert(SYNC_WITH_PROCESSING, this->ui->checkBox_sync->isChecked());
}

void VirtualOCTSystemSettingsDialog::initGui(){
	this->setWindowTitle(tr("Virtual OCT System Settings"));
	connect(this->ui->pushButton_selectFile, &QPushButton::clicked, this, &VirtualOCTSystemSettingsDialog::slot_selectFile);
	connect(this->ui->okButton, &QPushButton::clicked, this, &VirtualOCTSystemSettingsDialog::slot_apply);
	connect(this->ui->spinBox_width, &QSpinBox::editingFinished, this, &VirtualOCTSystemSettingsDialog::slot_checkWidthValue);
}

void VirtualOCTSystemSettingsDialog::slot_selectFile(){
	QString currentPath = this->ui->lineEdit->text();
	QString standardLocation = this->params.filePath.size() == 0 ? QStandardPaths::writableLocation(QStandardPaths::DesktopLocation) : this->params.filePath;
	QString fileName = QFileDialog::getOpenFileName(this, tr("Open Raw OCT Volume "), standardLocation, tr("Raw OCT Volume File (*.raw)"));
	if (fileName == "") {
		fileName = currentPath;
	}
	this->ui->lineEdit->setText(fileName);
}

void VirtualOCTSystemSettingsDialog::slot_apply() {
	this->params.filePath = this->ui->lineEdit->text();
	this->params.bitDepth = this->ui->spinBox_bitDepth->value();
	this->params.width = this->ui->spinBox_width->value();
	this->params.height = this->ui->spinBox_height->value();
	this->params.depth = this->ui->spinBox_depth->value();
	this->params.buffersPerVolume = this->ui->spinBox_buffersPerVolume->value();
	this->params.buffersFromFile = this->ui->spinBox_buffersFromFile->value();
	this->params.bscanOffset = this->ui->spinBox_bscanOffset->value();
	this->params.waitTimeUs = this->ui->spinBox_waitTime->value();
	this->params.copyFileToRam = this->ui->checkBox_copyFileToRam->isChecked();
	this->params.syncWithProcessing = this->ui->checkBox_sync->isChecked();
	emit settingsUpdated(this->params);
}

void VirtualOCTSystemSettingsDialog::slot_enableGui(bool enable){
	this->ui->lineEdit->setEnabled(enable);
	this->ui->pushButton_selectFile->setEnabled(enable);
	this->ui->spinBox_bitDepth->setEnabled(enable);
	this->ui->spinBox_width->setEnabled(enable);
	this->ui->spinBox_height->setEnabled(enable);
	this->ui->spinBox_depth->setEnabled(enable);
	this->ui->spinBox_buffersPerVolume->setEnabled(enable);
	this->ui->spinBox_buffersFromFile->setEnabled(enable);
	this->ui->spinBox_bscanOffset->setEnabled(enable);
	//this->ui->spinBox_waitTime->setEnabled(enable);  //waitTime does not need to be disabled. It can be safely changed during processing
	this->ui->checkBox_copyFileToRam->setEnabled(enable);
	this->ui->checkBox_sync->setEnabled(enable);
}

void VirtualOCTSystemSettingsDialog::slot_checkWidthValue(){
	int width = this->ui->spinBox_width->value();
	if(width % 2 != 0){
		this->ui->spinBox_width->setValue(width-1);
	}
}
