/**
**  This file is part of OCTproZ.
**  OCTproZ is an open source software for processig of optical
**  coherence tomography (OCT) raw data.
**  Copyright (C) 2019-2022 Miroslav Zabic
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
**			spectralcode.de
****
**/

#include "settings.h"


Settings* Settings::settings = nullptr;

Settings::Settings() {
	//if settings file does not exists, copy default settings file with reasonable initial values
	QDir settingsDir(SETTINGS_DIR);
	if(!QFileInfo::exists(SETTINGS_PATH) || !settingsDir.exists(SETTINGS_PATH)){
		settingsDir.mkpath(SETTINGS_DIR);
		bool success = QFile::copy(":default/settings.ini", SETTINGS_PATH);
		if(!success){
			emit error(tr("Could not create settings file in: ") + SETTINGS_PATH); //todo: this signal is not received by any slot since the signal is only connected after the constructor has been executed
		}
		QFile::setPermissions(SETTINGS_PATH, QFileDevice::WriteOther);
	}
}

Settings* Settings::getInstance() {
	settings = settings != nullptr ? settings : new Settings();
	return settings;
}

Settings::~Settings() {
}

void Settings::setTimestamp(QString timestamp) {
	this->timestamp = timestamp;
	QSettings settings(SETTINGS_PATH, QSettings::IniFormat);
	settings.setValue(TIMESTAMP, this->timestamp);
}

void Settings::storeSettings(QString settingsGroupName, QVariantMap settingsMap) {
	QSettings settings(SETTINGS_PATH, QSettings::IniFormat);
	this->storeValues(&settings, settingsGroupName, settingsMap);
}

QVariantMap Settings::getStoredSettings(QString settingsGroupName) {
	QVariantMap settingsMap;
	QSettings settings(SETTINGS_PATH, QSettings::IniFormat);
	this->loadValues(&settings, settingsGroupName, &settingsMap); //todo: loadValues should return a QVariantMap instead of passing a QVariantMap by pointer
	return settingsMap;
}

void Settings::copySettingsFile(QString path) {
	QString originPath = SETTINGS_PATH;
	QString destinationPath = path;
	bool success = QFile::copy(originPath, destinationPath);
	if(!success){
		emit error(tr("Could not store settings file to: ") + destinationPath);
	}
}

void Settings::storeValues(QSettings* settings, QString groupName, QVariantMap settingsMap) {
	QMapIterator<QString, QVariant> i(settingsMap);
	settings->beginGroup(groupName);
	while (i.hasNext()) {
		i.next();
		settings->setValue(i.key(), i.value());
	}
	settings->endGroup();
}

void Settings::loadValues(QSettings* settings, QString groupName, QVariantMap* settingsMap) {
	settings->beginGroup(groupName);
	QStringList keys = settings->allKeys();
	for (int i = 0; i < keys.size(); i++) {
		settingsMap->insert(keys.at(i), settings->value(keys.at(i)));
	}
	settings->endGroup();
}
