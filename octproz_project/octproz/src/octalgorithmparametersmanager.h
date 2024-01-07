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
	
	QVector<float> loadCurveFromFromFile(QString fileName);
	

public slots:
	void loadPostProcessBackgroundFromFile(QString fileName);

signals:
	void error(QString);
	void info(QString);
	void backgroundDataUpdated();
	
};

#endif // OCTALGORITHMPARAMETERSMANAGER_H
