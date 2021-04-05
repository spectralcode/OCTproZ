#ifndef OUTPUTWINDOW_H
#define OUTPUTWINDOW_H

#include <qwidget.h>
#include <qvariant.h>


//class OutputWindow : public QWidget
//{
//	Q_OBJECT
//public:
//	OutputWindow(QWidget* parent = nullptr){}
//	~OutputWindow(){}

//	QString getName() {return this->name;}
//	void setName(QString name) {this->name = name;}
//	virtual QVariantMap getSettings();
//	virtual void setSettings(QVariantMap);

//protected:
//	QString name;
//	QVariantMap settingsMap;

//signals:
//	void info(QString);
//	void error(QString);

//};

//#endif // OUTPUTWINDOW_H


class OutputWindow
{
public:
	OutputWindow(){}
	~OutputWindow(){}

	QString getName() {return this->name;}
	void setName(QString name) {this->name = name;}
	virtual QVariantMap getSettings(){return QVariantMap();}
	virtual void setSettings(QVariantMap){};

protected:
	QString name;
	QVariantMap settingsMap;
};

#endif // OUTPUTWINDOW_H
