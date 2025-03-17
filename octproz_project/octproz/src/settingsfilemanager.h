/**
**  This file is part of OCTproZ.
**  OCTproZ is an open source software for processing of optical
**  coherence tomography (OCT) raw data.
**  Copyright (C) 2019-2025 Miroslav Zabic
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
/*!	The Settings class is used to load and save all user-modifiable settings of OCTproZ and every plugin.
 * The settings are stored in an ini file. The save path is QStandardPaths::ConfigLocation (for Windows this is usually "C:/Users/<USER>/AppData/Local/<APPNAME>"
*/

#ifndef SETTINGSFILEMANAGER_H
#define SETTINGSFILEMANAGER_H

#define SETTINGS_DIR QStandardPaths::writableLocation(QStandardPaths::ConfigLocation)
#define SETTINGS_FILE_NAME "settings.ini"
#define SETTINGS_PATH SETTINGS_DIR + "/" + SETTINGS_FILE_NAME
#define GUI_SETTINGS_FILE_NAME "gui_settings.ini"
#define GUI_SETTINGS_PATH SETTINGS_DIR + "/" + GUI_SETTINGS_FILE_NAME
#define TIMESTAMP "timestamp"

#include <QStandardPaths>
#include <QSettings>
#include <QObject>
#include <QHash>
#include <QMap>
#include <QFile>
#include <QFileInfo>
#include <QDir>

#include "settingsconstants.h"

class SettingsFileManager : public QObject
{
	Q_OBJECT
public:

	/**
	* Constructor
	* @param settingsFilePath Path to the settings file to use
	* @param createDefaultIfMissing Whether to create a default file if missing
	* @param parent Parent QObject
	**/
	explicit SettingsFileManager(const QString& settingsFilePath, QObject* parent = nullptr);

	/**
	* Default Constructor. This will use the default settings file and default settings file location.
	* @param parent Parent QObject
	**/
	explicit SettingsFileManager(QObject* parent = nullptr);


	/**
	* Destructor
	*
	**/
	~SettingsFileManager();

	/**
	* Set timestamp variable which will be saved together with all other settings inside settings file. The timestamp can be used as a part of several filenames to enable easy identification of related files. 
	*
	* @param timestamp contains date and time information
	**/
	void setTimestamp(QString timestamp);

	/**
	* Sets current system time as timestamp.
	**/
	void setCurrentTimeStamp();

	/**
	* Get timestamp variable. It can be used as a part of other filenames that are saved together with the settings file to enable easy identification of related files.
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
	* Loads previously stored settings from the specified group
	*
	* @see storeSettings(QString sysName, QVariantMap settings)
	* @param settingsGroupName is the group name that will be used in the settings file. To load the saved settings, the identical group name needs to be used.
	* @return QVariantMap that contains previously saved settings.
	**/
	QVariantMap getStoredSettings(QString settingsGroupName);

	/**
	* Copies current settings file to specified  path
	*
	* @param path is the Destination file path. The settings file will be copied to this path
	* @return bool that indicates if copy was successful. true = sucess, false = copy failed
	**/
	bool copySettingsFile(QString path);

	/**
	* Gets the settings file path
	* @return Path to the settings file
	**/
	QString getSettingsFilePath() const { return settingsFilePath; }

private:
	QString settingsFilePath;
	QString timestamp;

	void storeValues(QSettings* settings, QString groupName, QVariantMap settingsMap);
	void loadValues(QSettings* settings, QString groupName, QVariantMap* settingsMap);
	bool createSettingsDirAndEmptyFile(QString settingsFilePath);


public slots:


signals:
	void error(QString);
	void info(QString);
};

#endif // SETTINGSFILEMANAGER_H

