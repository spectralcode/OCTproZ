#ifndef OUTPUTWINDOW_H
#define OUTPUTWINDOW_H

#include <qvariant.h>

class OutputWindow
{
public:
	OutputWindow(){}
	~OutputWindow(){}

	QString getName() {return this->name;}
	void setName(QString name) {this->name = name;}
	virtual QVariantMap getSettings() = 0;
	virtual void setSettings(QVariantMap) = 0;

protected:
	QString name;
	QVariantMap settingsMap;
};

#endif // OUTPUTWINDOW_H
