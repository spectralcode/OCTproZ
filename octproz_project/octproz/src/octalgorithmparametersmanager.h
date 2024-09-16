#ifndef OCTALGORITHMPARAMETERSMANAGER_H
#define OCTALGORITHMPARAMETERSMANAGER_H

#include <QObject>
#include <QString>
#include <QVector>
#include "octalgorithmparameters.h"

class OctAlgorithmParametersManager : public QObject
{
	Q_OBJECT
public:
	explicit OctAlgorithmParametersManager(QObject *parent = nullptr);

	
private:
	OctAlgorithmParameters* octParams;
	
	QVector<float> loadCurveFromFile(QString fileName);
	bool saveCurveToFile(QString fileName, int nSamples, float* curve);
	

public slots:
	void loadPostProcessBackgroundFromFile(QString fileName);
	void loadCustomResamplingCurveFromFile(QString fileName);
	void savePostProcessBackgroundToFile(QString fileName);
	void saveCustomResamplingCurveToFile(QString fileName);

	//todo: OctAlgorithmParametersManager" class should handle the loading and saving  of all octalgorithmparameters from and to files


signals:
	void error(QString);
	void info(QString);
	void backgroundDataUpdated();
	void resamplingCurveUpdated();
};

#endif // OCTALGORITHMPARAMETERSMANAGER_H
