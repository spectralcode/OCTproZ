/*
MIT License

Copyright (c) 2019-2020 Miroslav Zabic

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

#include "virtualoctsystem.h"

VirtualOCTSystem::VirtualOCTSystem() {
	this->setType((PLUGIN_TYPE)SYSTEM);
	this->systemDialog = new VirtualOCTSystemSettingsDialog();
	this->settingsDialog = static_cast<QDialog*>(this->systemDialog);
	this->name = "Virtual OCT System";
	this->file = nullptr;
    this->file2 = nullptr;
	this->multiBuffer = nullptr;

	connect(this->systemDialog, &VirtualOCTSystemSettingsDialog::settingsUpdated, this, &VirtualOCTSystem::slot_updateParams);
	connect(this, &VirtualOCTSystem::enableGui, this->systemDialog, &VirtualOCTSystemSettingsDialog::slot_enableGui);
}

VirtualOCTSystem::~VirtualOCTSystem() {
	this->cleanup();
	qDebug() << "VirtualOCTSystem destructor. Thread ID: " << QThread::currentThreadId();
}

bool VirtualOCTSystem::init() {
	//open file
	QString fileName;
        QString file2Name;
	if (this->currParams.filePath.size() < 2) {
		emit error(tr("No file selected for virtual OCT system."));
		return false;
	}else{
		fileName = this->currParams.filePath;
                file2Name = this->currParams.file2Path;
	}
	this->file = fopen(fileName.toLatin1(), "r");
	if (file == nullptr) {
		emit error(tr("Unable to open file for virtual OCT system!"));
		return false;
	}
    this->file2 = fopen(file2Name.toLatin1(), "r");
    if (file2 == nullptr) {
        emit error(tr("Unable to open file for virtual OCT system!"));
        return false;
    }

	//allocate buffer memory
	size_t bufferSize = currParams.width*currParams.height*currParams.depth*ceil((double)this->currParams.bitDepth / 8.0);
	this->buffer->allocateMemory(2, bufferSize);

	//create additional buffers if user wants to read multiple buffers per file
	if(currParams.buffersFromFile > 2){
		this->multiBuffer = new AcquisitionBuffer();
		this->multiBuffer->allocateMemory(currParams.buffersFromFile, bufferSize);
	}

	//emit info signal for OCTproZ message log console
	emit info (tr("Virtual OCT system initialized!"));
	
	//init was successful
	return true;
}


void VirtualOCTSystem::startAcquisition(){
	//init acquisition
	bool initSuccessfull = this->init();
	if(!initSuccessfull){
		emit enableGui(true);
		emit info(tr("Initialization unsuccessful. Acquisition stopped."));
		this->buffer->releaseMemory();
		this->cleanup();
		emit acquisitionStopped();
		return;
	}

	//start acquisition
	emit info("Acquisition started");
	if(currParams.buffersFromFile <= 2){
		this->acqcuisitionSimulation();
	}else{
		this->acquisitionSimulationWithMultiFileBuffers();
	}

	//acuquisition stopped
	emit enableGui(true);
	emit info("Acquisistion stopped!");
	emit acquisitionStopped();
	//wait some time before releasing buffer memory to allow extensions and 1d plot window to process last raw buffer
	QCoreApplication::processEvents();
	QThread::msleep(500);
	QCoreApplication::processEvents();
	this->buffer->releaseMemory();
	this->cleanup();
}

void VirtualOCTSystem::stopAcquisition(){
	this->acqusitionRunning = false;
	emit enableGui(true);
	qDebug() << "Plugin Thread ID stopAcq: " << QThread::currentThreadId();
}

void VirtualOCTSystem::cleanup() {
	if(this->multiBuffer != nullptr){
		delete this->multiBuffer;
		this->multiBuffer = nullptr;
	}
}

void VirtualOCTSystem::settingsLoaded(QVariantMap settings){
	this->systemDialog->setSettings(settings);
}

void VirtualOCTSystem::acqcuisitionSimulation(){
        //calculate size of buffer
        uint numberOfElements = this->currParams.depth * currParams.width * currParams.height;
        uint sizeOfElement = ceil((double)this->currParams.bitDepth / 8.0);

        //read data from file into first buffer
        void* buf = static_cast<void*>(this->buffer->bufferArray[0]);
        fread(buf, sizeOfElement, numberOfElements, this->file);

        //set position indicater associated with this->file
        if(currParams.buffersFromFile == 2){
                fseek(this->file, numberOfElements*sizeOfElement, SEEK_SET);
        }else{
                rewind(this->file);
        }

        //read data from file into second buffer
        buf = static_cast<void*>(this->buffer->bufferArray[1]);
        fread(buf, sizeOfElement, numberOfElements, this->file);

        //close file
        fclose(this->file);
        qDebug() << "file closed";

        //acquisition begins!
        emit enableGui(false);
        this->acqusitionRunning = true;
        this->buffer->currIndex = 1;
        emit acquisitionStarted(this);
        while (this->acqusitionRunning) {

                //wait untill processing thread is done with previous acquisition buffer. This is not necessary in real oct systems and may even slow down the overall processing speed. In real oct systems just check the bufferReadyArray flag of next buffer.
                while(this->buffer->bufferReadyArray[buffer->currIndex] == true && this->acqusitionRunning){
                        QThread::usleep(100);
                        QCoreApplication::processEvents();
                }

                //calculate index of next buffer
                int nextIndex = (this->buffer->currIndex+1)%2;

                //set acquisition buffer index, so that processing thread knows current buffer
                this->buffer->currIndex = nextIndex;

                //check bufferReadyArray flag to see if acquisition system is allowed to reuse this buffer
                if(buffer->bufferReadyArray[nextIndex] == false){

                        //actual data acquisition could be placed here. the content of this->buffer->bufferArray[nextIndex] could be modified here, but the acquisition buffer already contains the desired data so we just set the bufferReadyArray to true
                        //set bufferReadyArray to true to allow processing of buffer
                        this->buffer->bufferReadyArray[nextIndex] = true;

                }
                //user defined wait time
                QThread::usleep((this->currParams.waitTimeUs));
        }
}

void VirtualOCTSystem::acquisitionSimulationWithMultiFileBuffers() {
	//calculate size of buffer
    uint numberOfElements = this->currParams.depth * currParams.width * currParams.height;
    uint elementsPerBscan = this->currParams.width * currParams.height;
	uint sizeOfElement = ceil((double)this->currParams.bitDepth / 8.0);
    size_t bufferSizeInBytes = numberOfElements*sizeOfElement;
    int bscansPerBuffer = numberOfElements / elementsPerBscan;


    for(int i = 0; i < currParams.buffersFromFile; i++){
        void* buf = static_cast<void*>(this->multiBuffer->bufferArray[i]);
            for(int j = 0; j < (bscansPerBuffer / this->currParams.bscansPerComponent) * 2; j++) {
                fseek(this->file, (i * bufferSizeInBytes + j * this->currParams.bscansPerComponent * elementsPerBscan * sizeOfElement), SEEK_SET);
                fread((void*)((char*)buf + j * this->currParams.bscansPerComponent * elementsPerBscan * sizeOfElement), sizeOfElement, this->currParams.bscansPerComponent * elementsPerBscan, this->file);
                j++;
                fseek(this->file2, (i * bufferSizeInBytes + j * this->currParams.bscansPerComponent * elementsPerBscan * sizeOfElement), SEEK_SET);
                fread((void*)((char*)buf + j * this->currParams.bscansPerComponent * elementsPerBscan * sizeOfElement), sizeOfElement, this->currParams.bscansPerComponent * elementsPerBscan, this->file2);
    }
    }

    //close file
    fclose(this->file);
    fclose(this->file2);

	qDebug() << "file closed";

	//acquisition begins!
	emit enableGui(false);
	this->acqusitionRunning = true;
	this->buffer->currIndex = 0;
	int nextIndex = 1;
	int multiBufferIndex = currParams.buffersFromFile-1;
	emit acquisitionStarted(this);
	while (this->acqusitionRunning) {
		//wait untill processing thread is done with previous acquisition buffer. This is not necessary in real oct systems and may even slow down the overall processing speed. In real oct systems just check the bufferReadyArray flag of the next buffer.
		while(this->buffer->bufferReadyArray[buffer->currIndex] == true && this->acqusitionRunning){
			QThread::usleep(100);
			QCoreApplication::processEvents();
		}

		//set acquisition buffer index, so that processing thread knows current buffer
		this->buffer->currIndex = nextIndex;

		//check bufferReadyArray flag to see if acquisition system is allowed to reuse this buffer and write new data in acquisition buffer
		if(this->buffer->bufferReadyArray[nextIndex] == false){
			//get current buffer positions
			multiBufferIndex = (multiBufferIndex+1)%currParams.buffersFromFile;
			void* currAcquisitionBuf = static_cast<void*>(this->buffer->bufferArray[nextIndex]);
			void* currMultiBuf = static_cast<void*>(this->multiBuffer->bufferArray[multiBufferIndex]);

			//copy data from multiBuffer to acquisitionBuffer
			memcpy(currAcquisitionBuf, currMultiBuf, bufferSizeInBytes);

			//set bufferReadyArray to true to allow processing of buffer
			this->buffer->bufferReadyArray[nextIndex] = true;

			//calculate index of next buffer
			nextIndex = (this->buffer->currIndex+1)%2;
		}
		//user defined wait time
		QThread::usleep((this->currParams.waitTimeUs));
	}
}

void VirtualOCTSystem::slot_updateParams(simulatorParams newParams){
	this->currParams = newParams;
	AcquisitionParams params;
	params.samplesPerLine = newParams.width;
	params.ascansPerBscan = newParams.height;
	params.bscansPerBuffer = newParams.depth;
	params.buffersPerVolume = newParams.buffersPerVolume;
    params.bscansPerComponent = newParams.bscansPerComponent;
	params.bitDepth = newParams.bitDepth;
	this->params->slot_updateParams(params);

	//store settings, so settings can be reloaded into gui at next start of application
	this->systemDialog->getSettings(&this->settingsMap);
	emit storeSettings(this->name, this->settingsMap);
}
