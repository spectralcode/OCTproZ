#ifndef POLLINGNOTIFIER_H
#define POLLINGNOTIFIER_H

#include <QObject>
#include <QCoreApplication>
#include "octproz_devkit.h"
#include "octalgorithmparameters.h"

class PollingNotifier : public QObject
{
	Q_OBJECT

public:
	PollingNotifier();
	~PollingNotifier();


private:



public slots :	
	void startPolling(AcquisitionBuffer* buffer);

signals :
	void newDataAvailable(AcquisitionBuffer* buffer);

};
#endif // POLLINGNOTIFIER_H