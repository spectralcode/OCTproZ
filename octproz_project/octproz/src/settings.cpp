#include "settings.h"
#include <QDateTime>

Settings::Settings(const QString& settingsFilePath, QObject* parent)
	: QObject(parent),
	  settingsFilePath(settingsFilePath)
{
	this->createSettingsDirAndEmptyFile(settingsFilePath);
}

Settings::Settings(QObject* parent)
	: QObject(parent),
	  settingsFilePath(SETTINGS_PATH)
{
	QDir settingsDir(SETTINGS_DIR);
	if(!QFileInfo::exists(SETTINGS_PATH)){
		settingsDir.mkpath(SETTINGS_DIR);
		bool success = QFile::copy(":default/settings.ini", SETTINGS_PATH);
		if(!success){
			emit error(tr("Could not create settings file in: ") + SETTINGS_PATH); //todo: this signal is not received by any slot since the signal is only connected after the constructor has been executed
		}
		QFile::setPermissions(SETTINGS_PATH, QFileDevice::ReadUser | QFileDevice::WriteUser | QFileDevice::ReadOther | QFileDevice::WriteOther);
	}
}

Settings::~Settings() {
}

void Settings::setTimestamp(QString timestamp) {
	this->timestamp = timestamp;
	QSettings settings(settingsFilePath, QSettings::IniFormat);
	settings.setValue(TIMESTAMP, this->timestamp);
}

void Settings::setCurrentTimeStamp() {
	QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmsszzz");
	this->setTimestamp(timestamp);
}

void Settings::storeSettings(QString settingsGroupName, QVariantMap settingsMap) {
	QSettings settings(settingsFilePath, QSettings::IniFormat);
	this->storeValues(&settings, settingsGroupName, settingsMap);
}

QVariantMap Settings::getStoredSettings(QString settingsGroupName) {
	QVariantMap settingsMap;

	// Only try to load if the file exists
	if (QFile::exists(settingsFilePath)) {
		QSettings settings(settingsFilePath, QSettings::IniFormat);
		this->loadValues(&settings, settingsGroupName, &settingsMap);
	}

	return settingsMap;
}

bool Settings::copySettingsFile(QString path) {
	QString originPath = settingsFilePath;
	QString destinationPath = path;

	// Check if the source file exists
	if (!QFile::exists(originPath)) {
		emit error(tr("Settings file does not exist: ") + originPath);
		return false;
	}

	// Remove the destination file if it exists
	if (QFile::exists(destinationPath)) {
		if (!QFile::remove(destinationPath)) {
			emit error(tr("Could not overwrite existing file: ") + destinationPath);
			return false;
		}
	}

	// Copy the settings file to the new location
	bool success = QFile::copy(originPath, destinationPath);
	if (success) {
		emit info(tr("Settings saved to: ") + destinationPath);
	}
	return success;
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

bool Settings::createSettingsDirAndEmptyFile(QString settingsFilePath) {
	QFile file(settingsFilePath);
	if(!file.exists()){
		QFileInfo fileInfo(file);
		QString dirPath = fileInfo.absolutePath(); //get file path without file name
		QDir settingsDir;
		if(!settingsDir.exists(dirPath)){
			if(!settingsDir.mkpath(dirPath)){
				return false; //failed to create the directory
			}
		}
		//ceate the file as well
		if (!file.open(QIODevice::WriteOnly)) {
			return false; //failed to create the file
		}
		file.close();
		QFile::setPermissions(settingsFilePath, QFileDevice::ReadUser | QFileDevice::WriteUser | QFileDevice::ReadOther | QFileDevice::WriteOther);
	}
	return true;
}
