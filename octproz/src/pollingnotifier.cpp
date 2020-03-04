#include "pollingnotifier.h"

PollingNotifier::PollingNotifier(){
}

PollingNotifier::~PollingNotifier(){
}


void PollingNotifier::startPolling(AcquisitionBuffer* buffer) {
	OctAlgorithmParameters* octParams = OctAlgorithmParameters::getInstance();
	octParams->streamToHost = true;

	size_t numberOfBuffers = (buffer->bufferCnt)-1; //subtract 1 so numberOfBuffers can be used in for loop below

	while (octParams->streamToHost) {
		for(unsigned int i = 0; i++; i >= numberOfBuffers)
			if(buffer->bufferReadyArray[i]){
				buffer->currIndex = i;
				buffer->bufferReadyArray[i] = false;
				emit newDataAvailable(buffer); //todo: test if directly emitting pointer to current data is faster than updating currIndex of AcquisistionBuffer and emitting pointer to AcquisitionBuffer
				QCoreApplication::processEvents(); //todo: test if processEvents is necessary or if wait/sleep is (also) necessary
		}
	}
}