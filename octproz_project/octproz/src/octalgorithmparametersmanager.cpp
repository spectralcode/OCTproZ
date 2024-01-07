#include "octalgorithmparametersmanager.h"
#include <QFile>
#include <QTextStream>

OctAlgorithmParametersManager::OctAlgorithmParametersManager(QObject *parent) : 
	QObject(parent),
	octParams(OctAlgorithmParameters::getInstance())
{
	
}

QVector<float> OctAlgorithmParametersManager::loadCurveFromFromFile(QString fileName) {
	QVector<float> curve;
	if(fileName == ""){
		return curve;
	}
	QFile file(fileName);
	if(!file.exists()){
		return curve;
	}
	file.open(QIODevice::ReadOnly);
	QTextStream txtStream(&file);
	QString line = txtStream.readLine();
	while (!txtStream.atEnd()){
		line = txtStream.readLine();
		curve.append((line.section(";", 1, 1).toFloat()));
	}
	file.close();
	return curve;
}

void OctAlgorithmParametersManager::loadPostProcessBackgroundFromFile(QString fileName) {
	QVector<float> curve = this->loadCurveFromFromFile(fileName);
	if(curve.size() > 0){
		this->octParams->loadPostProcessingBackground(curve.data(), curve.size());
		//this->sidebar->updateBackgroundPlot();
		emit backgroundDataUpdated();
		emit info(tr("Background data for post processing loaded. File used: ") + fileName);
	}else{
		emit error(tr("Background data has a size of 0. Check if the .csv file with background data is not empty and has the right format."));
	}
}
