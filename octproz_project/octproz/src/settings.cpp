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

#include "settings.h"


//////////////////////////////////////////////////////////////////////////
//			constructor (singleton pattern!), destructor				//
//////////////////////////////////////////////////////////////////////////
Settings* Settings::settings = nullptr;

Settings::Settings() {
	this->recordSettings.insert(REC_PATH, QVariant(QString("")));
	this->recordSettings.insert(REC_MODE, QVariant((uint)RECORD_MODE::RAW));
	this->recordSettings.insert(REC_STOP, QVariant(false));
	this->recordSettings.insert(REC_META, QVariant(true));
	this->recordSettings.insert(REC_VOLUMES, QVariant(uint(2)));
	this->recordSettings.insert(REC_SKIP, QVariant(uint(0)));
	this->recordSettings.insert(REC_NAME, QVariant(QString("")));
	this->recordSettings.insert(REC_DESCRIPTION, QVariant(QString("")));

	this->processingSettings.insert(PROC_FLIP_BSCANS, QVariant(false));
	this->processingSettings.insert(PROC_BITSHIFT, QVariant(false));
	this->processingSettings.insert(PROC_MAX, QVariant(100.0));
	this->processingSettings.insert(PROC_MIN, QVariant(30.0));
	this->processingSettings.insert(PROC_LOG, QVariant(true));
	this->processingSettings.insert(PROC_COEFF, QVariant(1.0));
	this->processingSettings.insert(PROC_ADDEND, QVariant(0.0));
	this->processingSettings.insert(PROC_RESAMPLING, QVariant(false));
	this->processingSettings.insert(PROC_RESAMPLING_INTERPOLATION, QVariant(uint(0)));
	this->processingSettings.insert(PROC_RESAMPLING_C0, QVariant(0.0));
	this->processingSettings.insert(PROC_RESAMPLING_C1, QVariant(1024.0));
	this->processingSettings.insert(PROC_RESAMPLING_C2, QVariant(0.0));
	this->processingSettings.insert(PROC_RESAMPLING_C3, QVariant(0.0));
	this->processingSettings.insert(PROC_DISPERSION_COMPENSATION, QVariant(false));
	this->processingSettings.insert(PROC_DISPERSION_COMPENSATION_D0, QVariant(0.0));
	this->processingSettings.insert(PROC_DISPERSION_COMPENSATION_D1, QVariant(0.0));
	this->processingSettings.insert(PROC_DISPERSION_COMPENSATION_D2, QVariant(0.0));
	this->processingSettings.insert(PROC_DISPERSION_COMPENSATION_D3, QVariant(0.0));
	this->processingSettings.insert(PROC_WINDOWING, QVariant(false));
	this->processingSettings.insert(PROC_WINDOWING_TYPE, QVariant(uint(0)));
	this->processingSettings.insert(PROC_WINDOWING_FILL_FACTOR, QVariant(0.9));
	this->processingSettings.insert(PROC_WINDOWING_CENTER_POSITION, QVariant(0.5));
	this->processingSettings.insert(PROC_FIXED_PATTERN_REMOVAL, QVariant(false));
	this->processingSettings.insert(PROC_FIXED_PATTERN_REMOVAL_Continuously, QVariant(false));
	this->processingSettings.insert(PROC_FIXED_PATTERN_REMOVAL_BSCANS, QVariant(uint(1)));
	this->processingSettings.insert(PROC_SINUSOIDAL_SCAN_CORRECTION, QVariant(false));

	this->streamingSettings.insert(STREAM_STREAMING, QVariant(false));
	this->streamingSettings.insert(STREAM_STREAMING_SKIP, QVariant(uint(0)));

	this->mainWindowSettings.insert(MAIN_GEOMETRY, QVariant(QByteArray()));
	this->mainWindowSettings.insert(MAIN_STATE, QVariant(QByteArray()));

	//if settings file does not exists, create a settings file with the above defined values to fill the gui/octalgorithmparameters with reasonable inital values
	if(!QFileInfo::exists(SETTINGS_PATH)){
		this->storeSettings(SETTINGS_PATH);
	}
}

Settings* Settings::getInstance() {
	settings = settings != nullptr ? settings : new Settings();
	return settings;
}

Settings::~Settings() {
}


//////////////////////////////////////////////////////////////////////////
//					public methods										//
//////////////////////////////////////////////////////////////////////////
void Settings::storeSettings(QString path) {
	QSettings settings(path, QSettings::IniFormat);
	settings.setValue(TIMESTAMP, this->timestamp);
	this->storeValues(&settings, REC, &this->recordSettings);
	this->storeValues(&settings, PROC, &this->processingSettings); 
	this->storeValues(&settings, VIZ, &this->visualizationSettings); 
	this->storeValues(&settings, STREAM, &this->streamingSettings);
	this->storeValues(&settings, MAIN, &this->mainWindowSettings);
}

void Settings::loadSettings(QString path) {
	QSettings settings(path, QSettings::IniFormat);
	this->timestamp = settings.value(TIMESTAMP, "").toString();
	this->loadValues(&settings, REC, &this->recordSettings);
	this->loadValues(&settings, PROC, &this->processingSettings);
	this->loadValues(&settings, VIZ, &this->visualizationSettings);
	this->loadValues(&settings, STREAM, &this->streamingSettings);
	this->loadValues(&settings, MAIN, &this->mainWindowSettings);
}

void Settings::storeSystemSettings(QString sysName, QVariantMap settingsMap) {
	this->systemSettings = settingsMap;
	this->systemSettings.detach(); //force deep copy
	QSettings settings(SETTINGS_PATH, QSettings::IniFormat);
	this->storeValues(&settings, sysName, &this->systemSettings);
}

QVariantMap Settings::getStoredSystemSettings(QString sysName) {
	//load keys from settings file
	QSettings settings(SETTINGS_PATH, QSettings::IniFormat);
	settings.beginGroup(sysName);
	QStringList keys = settings.allKeys();
	settings.endGroup();
	for (int i = 0; i < keys.size(); i++) {
		this->systemSettings.insert(keys.at(i), 0);
	}
	//load values from settings file
	this->loadValues(&settings, sysName, &this->systemSettings);
	return this->systemSettings;
}

void Settings::copySettingsFile(QString path) {
	QString originPath = SETTINGS_PATH;
	QString destinationPath = path;
	bool success = QFile::copy(originPath, destinationPath);
	if(!success){
		emit error(tr("Could not store settings file to: ") + destinationPath);
	}
}


//////////////////////////////////////////////////////////////////////////
//					private methods										//
//////////////////////////////////////////////////////////////////////////
void Settings::storeValues(QSettings* settings, QString groupName, QVariantMap* settingsMap) {
	QMapIterator<QString, QVariant> i(*settingsMap);
	settings->beginGroup(groupName);
	while (i.hasNext()) {
		i.next();
		settings->setValue(i.key(), i.value());
	}
	settings->endGroup();
}

void Settings::loadValues(QSettings* settings, QString groupName, QVariantMap* settingsMap) {
	QMapIterator<QString, QVariant> i(*settingsMap);
	settings->beginGroup(groupName);
	while (i.hasNext()) {
		i.next();
		settingsMap->insert(i.key(), settings->value(i.key()));
	}
	settings->endGroup();
}

//////////////////////////////////////////////////////////////////////////
//					public slots										//
//////////////////////////////////////////////////////////////////////////
