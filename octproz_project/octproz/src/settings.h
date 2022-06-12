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

//!	Settings
/*!	The Settings class is implemented with the singleton pattern and is used to load and save all user-modifiable settings of OCTproZ and every plugin.
 * The settings are stored in an ini file. The save path is QStandardPaths::ConfigLocation (for Windows this is usually "C:/Users/<USER>/AppData/Local/<APPNAME>"
*/

#ifndef SETTINGS_H
#define SETTINGS_H

#define SETTINGS_DIR QStandardPaths::writableLocation(QStandardPaths::ConfigLocation)
#define SETTINGS_FILE_NAME "settings.ini"
#define SETTINGS_PATH  SETTINGS_DIR + "/" + SETTINGS_FILE_NAME
#define TIMESTAMP "timestamp"



#include <QStandardPaths>
#include <QSettings>
#include <QObject>
#include <QHash>
#include <QMap>
#include <QFile>
#include <QFileInfo>
#include <QDir>


class Settings : public QObject
{
	Q_OBJECT
public:
	QMap<QString, QVariant> processingSettings;
	QMap<QString, QVariant> recordSettings;
	QMap<QString, QVariant> streamingSettings;
	QMap<QString, QVariant> mainWindowSettings;
	QVariantMap systemSettings;
	QString systemName;


	/**
	* Constructor
	* @note Singleton pattern
	*
	**/
	static Settings* getInstance();


	/**
	* Destructor
	*
	**/
	~Settings();

	/**
	* Set timestamp variable which will be saved together with all other settings inside settings file. The timestamp can be used as a part of several filenames to enable easy identification of related files. 
	*
	* @param timestamp contains date and time information
	**/
	void setTimestamp(QString timestamp);

	/**
	* Set timestamp variable which will be saved together with all other settings inside settings file. The timestamp can be used as a part of several filenames to enable easy identification of related files. 
	*
	* @return timestamp that contains date time information
	**/
	QString getTimestamp() { return this->timestamp; }

	/**
	* Stores settings from arbitrary QVariantMap into the settings group defined by "sysName". This method is typically used to store settings from systems. Systems are shared libraries, so the main application can not know in advance (during compile time) which settings every system has. This method could be used to mess up previously stored settings if sysName is an already used group name.
	*
	* @param settingsGroupName is the group name that will be used in the settings file. To load the saved settings, the identical group name needs to be used.
	* @param settingsMap is a arbitrary QVariantMap that contains the settings to be saved. 
	**/
	void storeSettings(QString settingsGroupName, QVariantMap settingsMap);

	/**
	* Loads previously stored settings from settings group defined by "settingsGroupName". This method is typically used to load arbitrary system settings.
	*
	* @see storeSettings(QString sysName, QVariantMap settings)
	* @param settingsGroupName is the group name that will be used in the settings file. To load the saved settings, the identical group name needs to be used.
	* @return QVariantMap that contains previously saved settings.
	**/
	QVariantMap getStoredSettings(QString settingsGroupName);

	/**
	* Copies current settings file to path
	*
	* @param path is the file path of the settings file to be copied
	**/
	void copySettingsFile(QString path);



private:
	Settings(void);
	static Settings *settings;
	QString timestamp;

	void storeValues(QSettings* settings, QString groupName, QVariantMap settingsMap);
	void loadValues(QSettings* settings, QString groupName, QVariantMap* settingsMap);


public slots:


signals:
	void error(QString);
	void info(QString);
};

#endif // SETTINGS_H

