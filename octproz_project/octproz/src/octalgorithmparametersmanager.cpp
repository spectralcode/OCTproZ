#include "octalgorithmparametersmanager.h"
#include <QFile>
#include <QTextStream>

OctAlgorithmParametersManager::OctAlgorithmParametersManager(QObject *parent) : 
	QObject(parent),
	octParams(OctAlgorithmParameters::getInstance())
{

}

QVector<float> OctAlgorithmParametersManager::loadCurveFromFile(QString fileName) {
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

bool OctAlgorithmParametersManager::saveCurveToFile(QString fileName, int nSamples, float* curve) {
	QFile file(fileName);
	if (file.open(QFile::WriteOnly|QFile::Truncate)) {
		QTextStream stream(&file);
		stream << tr("Sample Number") << ";" << tr("Sample Value") << "\n";
		for(int i = 0; i < nSamples; i++){
			stream << QString::number(i) << ";" << curve[i] << "\n";
		}
		file.close();
		return true;
	} else {
		return false;
	}
}


void OctAlgorithmParametersManager::loadPostProcessBackgroundFromFile(QString fileName) {
	QVector<float> curve = this->loadCurveFromFile(fileName);
	if(curve.size() > 0){
		this->octParams->loadPostProcessingBackground(curve.data(), curve.size());
		//this->sidebar->updateBackgroundPlot();
		emit backgroundDataUpdated();
		emit info(tr("Background data for post processing loaded. File used: ") + fileName);
	}else{
		emit error(tr("Background data has a size of 0. Check if the .csv file with background data is not empty and has the right format."));
	}
}

void OctAlgorithmParametersManager::loadCustomResamplingCurveFromFile(QString fileName) {
	QVector<float> curve = this->loadCurveFromFile(fileName);
	if(curve.size() > 0){
		this->octParams->loadCustomResampleCurve(curve.data(), curve.size());
		this->octParams->customResampleCurveFilePath = fileName;
		emit resamplingCurveUpdated();
		emit info(tr("Resampling  curve loaded. File used: ") + fileName);
	}else{
		emit error(tr("Resampling  curve has a size of 0. Check if the .csv file with resampling curve data is not empty and has the right format."));
	}
}

void OctAlgorithmParametersManager::savePostProcessBackgroundToFile(QString fileName) {
	if((this->saveCurveToFile(fileName,octParams->postProcessBackgroundLength, octParams->postProcessBackground))){
		emit info(tr("Background data saved to: ") + fileName);
	} else {
		emit error(tr("Could not save background data to: ") + fileName);
	}
}

void OctAlgorithmParametersManager::saveCustomResamplingCurveToFile(QString fileName) {
	if((this->saveCurveToFile(fileName,octParams->customResampleCurveLength, octParams->customResampleCurve))){
		emit info(tr("Resampling curve saved to: ") + fileName);
	} else {
		emit error(tr("Could not save resamlpling curve to: ") + fileName);
	}
}
