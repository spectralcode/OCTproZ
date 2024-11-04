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

#ifndef PLUGIN_H
#define PLUGIN_H

#include <qobject.h>
#include <qvariant.h>


enum PLUGIN_TYPE {
	SYSTEM,
	EXTENSION
};

class Plugin : public QObject
{
	Q_OBJECT
public:
	Plugin(){}
	~Plugin(){}

	virtual void settingsLoaded(QVariantMap settings){}

	PLUGIN_TYPE getType() { return this->type;}
	void setType(PLUGIN_TYPE type) { this->type = type;}
	QString getName() { return this->name;}
	void setName(QString name) { this->name = name;}

protected:
	PLUGIN_TYPE type;
	QString name;
	QVariantMap settingsMap;

public slots:
	void setKLinCoeffsRequestAccepted(int k0, int k1, int k2, int k3){} ///< This slot is called by OCTproZ if setKLinCoeffsRequest was accepted
	void setDispCompCoeffsRequestAccepted(int d0, int d1, int d2, int d3){} ///< This slot is called by OCTproZ if setDispCompCoeffsRequest was accepted

signals:
	void info(QString);
	void error(QString);
	void storeSettings(QString, QVariantMap);
	void setKLinCoeffsRequest(double* k0, double* k1, double* k2, double* k3); ///< This signal can be used to change the coeffs for k linearization. If parameter value is "nullptr" the respective coefficient will not be changed.
	void setDispCompCoeffsRequest(double* d0, double* d1, double* d2, double* d3); ///< This signal can be used to change the coeffs for numerical dispersion compensation. If parameter value is "nullptr" the respective coefficient will not be changed.
	void startProcessingRequest();
	void stopProcessingRequest();
	void startRecordingRequest();
	void setCustomResamplingCurveRequest(QVector<float> customCurve);
	void loadSettingsFileRequest(QString);
	void saveSettingsFileRequest(QString);
};

#endif // PLUGIN_H
