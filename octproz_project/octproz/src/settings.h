/**
**  This file is part of OCTproZ.
**  OCTproZ is an open source software for processig of optical
**  coherence tomography (OCT) raw data.
**  Copyright (C) 2019-2021 Miroslav Zabic
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

//!	Settings
/*!	Class for storing processing, visualization and record settings to the hard drive.
*/

#ifndef SETTINGS_H
#define SETTINGS_H

#define SETTINGS_FILE_NAME "settings.ini"
#define SETTINGS_PATH QStandardPaths::writableLocation(QStandardPaths::ConfigLocation) + "/" + SETTINGS_FILE_NAME
#define TIMESTAMP "timestamp"
#define REC "record"
#define PROC "processing"
#define VIZ "visualization"
#define STREAM "streaming"
#define MAIN "main_window_settings"
#define MAIN_GEOMETRY "main_geometry"
#define MAIN_STATE "main_state"
#define REC_PATH "path"
#define REC_MODE "record_mode"
#define REC_STOP "stop_after_record"
#define REC_META "save_meta_info"
#define REC_VOLUMES "volumes"
#define REC_SKIP "skip"
#define REC_NAME "name"
#define REC_DESCRIPTION "description"
#define PROC_FLIP_BSCANS "flip_bscans"
#define PROC_BITSHIFT "bitshift"
#define PROC_MIN "min"
#define PROC_MAX "max"
#define PROC_LOG "log"
#define PROC_COEFF "coeff"
#define PROC_ADDEND "addend"
#define PROC_RESAMPLING "resampling"
#define PROC_RESAMPLING_INTERPOLATION "resampling_interpolation"
#define PROC_RESAMPLING_C0 "resampling_c0"
#define PROC_RESAMPLING_C1 "resampling_c1"
#define PROC_RESAMPLING_C2 "resampling_c2"
#define PROC_RESAMPLING_C3 "resampling_c3"
#define PROC_DISPERSION_COMPENSATION "dispersion_compensation"
#define PROC_DISPERSION_COMPENSATION_D0 "dispersion_compensation_d0"
#define PROC_DISPERSION_COMPENSATION_D1 "dispersion_compensation_d1"
#define PROC_DISPERSION_COMPENSATION_D2 "dispersion_compensation_d2"
#define PROC_DISPERSION_COMPENSATION_D3 "dispersion_compensation_d3"
#define PROC_WINDOWING "windowing"
#define PROC_WINDOWING_TYPE "window_type"
#define PROC_WINDOWING_FILL_FACTOR "window_fill_factor"
#define PROC_WINDOWING_CENTER_POSITION "window_center_position"
#define PROC_FIXED_PATTERN_REMOVAL "fixed_pattern_removal"
#define PROC_FIXED_PATTERN_REMOVAL_Continuously "fixed_pattern_removal_Continuously"
#define PROC_FIXED_PATTERN_REMOVAL_BSCANS "fixed_pattern_removal_bscans"
#define PROC_SINUSOIDAL_SCAN_CORRECTION "sinusoidal_scan_correction"
#define STREAM_STREAMING "streaming_enabled"
#define STREAM_STREAMING_SKIP "streaming_skip"


#include <QStandardPaths>
#include <QSettings>
#include <QObject>
#include <QHash>
#include <QMap>
#include <QFile>
#include <QFileInfo>

enum SETTINGS_MODE {
	PROCESSING,
	VISUALIZATION,
	RECORD,
	MAIN_WINDOW
};

enum RECORD_MODE {
	SNAPSHOT,
	RAW,
	PROCESSED,
	ALL
};

class Settings : public QObject
{
	Q_OBJECT
public:
	QMap<QString, QVariant> processingSettings;
	QMap<QString, QVariant> visualizationSettings;
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
	* Stores settings in a file at a location definded by path
	*
	* @param path is the file path of the settings file
	**/
	void storeSettings(QString path);

	/**
	* Loads settings from a file at a location definde by path. If reading of save file fails default values will be loaded.
	*
	* @param path is the file path of the settings file
	**/
	void loadSettings(QString path);

	/**
	* Set timestamp variable which will be saved together with all other settings inside settings file. The timestamp can be used as a part of several filenames to enable easy identification of related files. 
	*
	* @param timestamp contains date and time information
	**/
	void setTimestamp(QString timestamp) { this->timestamp = timestamp; }

	/**
	* Set timestamp variable which will be saved together with all other settings inside settings file. The timestamp can be used as a part of several filenames to enable easy identification of related files. 
	*
	* @return timestamp that contains date time information
	**/
	QString getTimestamp() { return this->timestamp; }

	/**
	* Stores settings from arbitrary QVariantMap into the settings group defined by "sysName". This method is typically used to store settings from systems. Systems are shared libraries, so the main application can not know in advance (during compile time) which settings every system has. This method could be used to mess up previously stored settings if sysName is an already used group name.
	*
	* @param sysName is the group name that will be used in the settings file. To load the saved settings, the identical group name needs to be used. 
	* @param settingsMap is a arbitrary QVariantMap that contains the settings to be saved. 
	**/
	void storeSystemSettings(QString sysName, QVariantMap settingsMap);

	/**
	* Loads previously stored settings from settings group defined by "sysName". This method is typically used to load arbitrary system settings.
	*
	* @see storeSystemSettings(QString sysName, QVariantMap settings)
	* @param sysName is the group name that will be used in the settings file. To load the saved settings, the identical group name needs to be used.
	* @return QVariantMap that contains previously saved settings.
	**/
	QVariantMap getStoredSystemSettings(QString sysName);

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

	void storeValues(QSettings* settings, QString groupName, QVariantMap* settingsMap);
	void loadValues(QSettings* settings, QString groupName, QVariantMap* settingsMap);


public slots:


signals:
	void error(QString);
	void info(QString);
};

#endif // SETTINGS_H

