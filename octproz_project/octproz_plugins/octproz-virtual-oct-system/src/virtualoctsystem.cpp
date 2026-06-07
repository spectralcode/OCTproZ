/*
MIT License

Copyright (c) 2019-2024 Miroslav Zabic

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
#include <cstring>


VirtualOCTSystem::VirtualOCTSystem() {
	this->setType((PLUGIN_TYPE)SYSTEM);
	this->systemDialog = new VirtualOCTSystemSettingsDialog();
	this->settingsDialog = static_cast<QDialog*>(this->systemDialog);
	this->name = "Virtual OCT System";
	this->file = nullptr;
	this->normalBuffer = this->buffer;
	this->rawOnlyBuffer = new AcquisitionBuffer();
	this->streamBuffer = nullptr;
	this->rawOnlyModeEnabled = false;
	this->isCleanupPending  = false;

	connect(this->systemDialog, &VirtualOCTSystemSettingsDialog::settingsUpdated, this, &VirtualOCTSystem::slot_updateParams);
	connect(this, &VirtualOCTSystem::enableGui, this->systemDialog, &VirtualOCTSystemSettingsDialog::slot_enableGui);
	connect(this->rawOnlyBuffer, &AcquisitionBuffer::info, this, &VirtualOCTSystem::info);
	connect(this->rawOnlyBuffer, &AcquisitionBuffer::error, this, &VirtualOCTSystem::error);

	//default values
	this->currParams.filePath = "";
	this->currParams.bitDepth = 8;
	this->currParams.width = 256;
	this->currParams.height = 256;
	this->currParams.depth = 16;
	this->currParams.buffersPerVolume = 2;
	this->currParams.buffersFromFile = 2;
	this->currParams.bscanOffset = 0;
	this->currParams.waitTimeUs = 100000;
	this->currParams.copyFileToRam = true;
	this->currParams.syncWithProcessing = true;
	this->normalParams = this->currParams;
	this->rawOnlyParams = this->acquisitionParamsFromSimulatorParams(this->currParams);
}

VirtualOCTSystem::~VirtualOCTSystem() {
	this->releaseAllBuffers();
	this->buffer = nullptr;
	qDebug() << "VirtualOCTSystem destructor. Thread ID: " << QThread::currentThreadId();
}

bool VirtualOCTSystem::init() {
	//check if user selected file can be opened
	if(!this->openFileToCopyToRam()){
		return false;
	}

	//allocate buffer memory
	size_t bufferSize = this->currentBufferSizeInBytes();
	if (!this->allocateActiveBuffer(bufferSize)) {
		return false;
	}

	//create additional buffers if user wants to read multiple buffers per file and copy entire file to ram
	if(currParams.buffersFromFile > 2 && currParams.copyFileToRam){
		this->streamBuffer = new AcquisitionBuffer();
		this->streamBuffer->allocateMemory(currParams.buffersFromFile, bufferSize);
	}

	//create small stream buffer if user wants to read multiple buffers per file and NOT copy entire file to ram
	if(currParams.buffersFromFile > 2 && !currParams.copyFileToRam){
		this->streamBuffer = new AcquisitionBuffer();
		this->streamBuffer->allocateMemory(1, STREAM_BUFFER_SIZE);
	}

	//emit info signal for OCTproZ message log console
	emit info (tr("Virtual OCT system initialized!"));
	
	//init was successful
	return true;
}


void VirtualOCTSystem::startAcquisition(){
	//check if cleanup is pending from previous acquisition
	if(this->isCleanupPending){
		this->cleanup();
	}

	this->updateCurrentAcquisitionParams();

	//init acquisition
	bool initSuccessfull = this->init();
	if(!initSuccessfull){
		emit enableGui(true);
		emit info(tr("Initialization unsuccessful. Acquisition stopped."));
		this->cleanup();
		emit acquisitionStopped();
		return;
	}
	if(this->rawOnlyModeEnabled && !this->preloadActiveBufferFromFile()){
		emit enableGui(true);
		emit info(tr("Initialization unsuccessful. Acquisition stopped."));
		this->cleanup();
		emit acquisitionStopped();
		return;
	}

	//start acquisition
	emit info("Acquisition started");
	if(currParams.buffersFromFile <= 2){
		this->acqcuisitionSimulation();
	}else if(currParams.copyFileToRam){
		this->acquisitionSimulationWithMultiFileBuffers();
	}else{
		this->acqcuisitionSimulationLargeFile();
	}

	//acuquisition stopped
	this->isCleanupPending = true;
	emit enableGui(true);
	emit info("Acquisistion stopped!");
	emit acquisitionStopped();
	//wait some time before releasing buffer memory to allow extensions and 1d plot window to process last raw buffer
	QCoreApplication::processEvents();
	QThread::msleep(500);
	QCoreApplication::processEvents();
	this->cleanup();
	this->isCleanupPending = false;
}

void VirtualOCTSystem::stopAcquisition(){
	this->acqusitionRunning = false;
	emit enableGui(true);
	qDebug() << "Plugin Thread ID stopAcq: " << QThread::currentThreadId();
}

void VirtualOCTSystem::cleanup() {
	if (this->normalBuffer != nullptr) {
		this->normalBuffer->releaseMemory();
	}
	if (this->rawOnlyBuffer != nullptr) {
		this->rawOnlyBuffer->releaseMemory();
	}

	if(this->streamBuffer != nullptr){
		delete this->streamBuffer;
		this->streamBuffer = nullptr;
	}
}

void VirtualOCTSystem::releaseAllBuffers() {
	if (this->normalBuffer != nullptr) {
		this->normalBuffer->releaseMemory();
		delete this->normalBuffer;
		this->normalBuffer = nullptr;
	}
	if (this->rawOnlyBuffer != nullptr) {
		this->rawOnlyBuffer->releaseMemory();
		delete this->rawOnlyBuffer;
		this->rawOnlyBuffer = nullptr;
	}
	if(this->streamBuffer != nullptr){
		delete this->streamBuffer;
		this->streamBuffer = nullptr;
	}
}

AcquisitionBuffer* VirtualOCTSystem::activeAcquisitionBuffer() const {
	return this->rawOnlyModeEnabled ? this->rawOnlyBuffer : this->normalBuffer;
}

bool VirtualOCTSystem::allocateActiveBuffer(size_t bufferSize) {
	this->buffer = this->activeAcquisitionBuffer();
	if (this->buffer == nullptr) {
		return false;
	}
	if (this->buffer->bufferArray.size() == 2 && this->buffer->bytesPerBuffer == bufferSize) {
		this->buffer->currIndex = -1;
		for (int i = 0; i < this->buffer->bufferReadyArray.size(); ++i) {
			this->buffer->bufferReadyArray[i] = false;
		}
		return true;
	}
	return this->buffer->allocateMemory(2, bufferSize);
}

size_t VirtualOCTSystem::currentBufferSizeInBytes() const {
	size_t numberOfElements = static_cast<size_t>(this->currParams.depth) * static_cast<size_t>(this->currParams.width) * static_cast<size_t>(this->currParams.height);
	size_t sizeOfElement = static_cast<size_t>(ceil((double)this->currParams.bitDepth / 8.0));
	return numberOfElements * sizeOfElement;
}

bool VirtualOCTSystem::preloadActiveBufferFromFile() {
	size_t bufferSize = this->currentBufferSizeInBytes();
	if (!this->allocateActiveBuffer(bufferSize)) {
		return false;
	}

	if (this->currParams.filePath.size() < 2) {
		emit error(tr("No file selected for virtual OCT system."));
		return false;
	}

	FILE* sourceFile = fopen(this->currParams.filePath.toLatin1(), "rb");
	if (sourceFile == nullptr) {
		emit error(tr("Unable to open file for virtual OCT system!"));
		return false;
	}

	size_t numberOfElements = static_cast<size_t>(this->currParams.depth) * static_cast<size_t>(this->currParams.width) * static_cast<size_t>(this->currParams.height);
	size_t sizeOfElement = static_cast<size_t>(ceil((double)this->currParams.bitDepth / 8.0));
	size_t offsetInBytes = static_cast<size_t>(this->currParams.bscanOffset) * static_cast<size_t>(this->currParams.width) * static_cast<size_t>(this->currParams.height) * sizeOfElement;

	for (int i = 0; i < this->buffer->bufferArray.size(); ++i) {
		fseek(sourceFile, static_cast<long>(offsetInBytes + static_cast<size_t>(i) * bufferSize), SEEK_SET);
		size_t readElements = fread(this->buffer->bufferArray[i], sizeOfElement, numberOfElements, sourceFile);
		if (readElements < numberOfElements) {
			size_t bytesRead = readElements * sizeOfElement;
			memset(static_cast<char*>(this->buffer->bufferArray[i]) + bytesRead, 0, bufferSize - bytesRead);
		}
		this->buffer->bufferReadyArray[i] = false;
	}

	this->buffer->currIndex = -1;
	fclose(sourceFile);
	return true;
}

bool VirtualOCTSystem::handleRawOnlyModeBuffer() {
	if(!this->rawOnlyModeEnabled){
		return false;
	}

	if(this->buffer == nullptr || this->buffer->bufferArray.size() < 2){
		QCoreApplication::processEvents();
		return true;
	}

	int nextIndex = this->buffer->currIndex < 0 ? 0 : (this->buffer->currIndex+1)%2;
	this->buffer->currIndex = nextIndex;
	if(this->buffer->bufferReadyArray.at(nextIndex) == false){
		this->buffer->bufferReadyArray[nextIndex] = true;
	}
	if(this->currParams.waitTimeUs > 0){
		QThread::usleep((this->currParams.waitTimeUs));
	}
	return true;
}

bool VirtualOCTSystem::openFileToCopyToRam() {
	QString fileName;
	if (this->currParams.filePath.size() < 2) {
		emit error(tr("No file selected for virtual OCT system."));
		return false;
	}else{
		fileName = this->currParams.filePath;
	}
	this->file = fopen(fileName.toLatin1(), "r"); //todo: consider using QFile
	if (file == nullptr) {
		emit error(tr("Unable to open file for virtual OCT system!"));
		return false;
	}
	return true;
}

void VirtualOCTSystem::settingsLoaded(QVariantMap settings){
	this->rawOnlyModeEnabled = false;
	this->rawOnlyParams.bitDepth = settings.value(RAW_ONLY_BITDEPTH, settings.value(BITDEPTH, this->currParams.bitDepth)).toUInt();
	this->rawOnlyParams.samplesPerLine = settings.value(RAW_ONLY_WIDTH, settings.value(WIDTH, this->currParams.width)).toUInt();
	this->rawOnlyParams.ascansPerBscan = settings.value(RAW_ONLY_HEIGHT, settings.value(HEIGHT, this->currParams.height)).toUInt();
	this->rawOnlyParams.bscansPerBuffer = settings.value(RAW_ONLY_DEPTH, settings.value(DEPTH, this->currParams.depth)).toUInt();
	this->rawOnlyParams.buffersPerVolume = settings.value(RAW_ONLY_BUFFERS_PER_VOLUME, settings.value(BUFFERS_PER_VOLUME, this->currParams.buffersPerVolume)).toUInt();
	this->systemDialog->setSettings(settings);
	this->updateCurrentAcquisitionParams();
}

void VirtualOCTSystem::acqcuisitionSimulation(){
	//calculate size of buffer
	uint numberOfElements = this->currParams.depth * currParams.width * currParams.height;
	uint sizeOfElement = ceil((double)this->currParams.bitDepth / 8.0);
	size_t offsetInBytes = this->currParams.bscanOffset * this->currParams.width * this->currParams.height * sizeOfElement;

	fseek(this->file, static_cast<long>(offsetInBytes), SEEK_SET);

	//read data from file into first buffer
	void* buf = static_cast<void*>(this->buffer->bufferArray[0]);
	fread(buf, sizeOfElement, numberOfElements, this->file);

	//set position indicater associated with this->file
	if(currParams.buffersFromFile == 2){
		fseek(this->file, static_cast<long>(numberOfElements*sizeOfElement + offsetInBytes), SEEK_SET);
	}else{
		fseek(this->file, static_cast<long>(offsetInBytes), SEEK_SET);
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
	bool syncEnabled = true;
	while (this->acqusitionRunning) {
		//wait until processing thread is done with copying data from previous buffer. This is not necessary in real oct systems, since they usually do not provide new data as fast as this virtual oct system. In real oct systems just check the bufferReadyArray flag of the next buffer.
		if(syncEnabled){
			while(this->buffer->currIndex >= 0 && this->buffer->bufferReadyArray.at(this->buffer->currIndex) == true && this->acqusitionRunning && syncEnabled){
				QCoreApplication::processEvents();
				syncEnabled = this->currParams.syncWithProcessing;
			}
		}

		if(this->handleRawOnlyModeBuffer()){
			continue;
		}

		//calculate index of next buffer
		int nextIndex = (this->buffer->currIndex+1)%2;

		//set acquisition buffer index, so that processing thread knows current buffer
		this->buffer->currIndex = nextIndex;

		//check bufferReadyArray flag to see if acquisition system is allowed to reuse this buffer and write new data in acquisition buffer. Once the bufferReadyArray flag is false, the acquisition system is allowed to reuse the buffer. If bufferReadyArray is true the processing thread is still copying data from the buffer.
		if(buffer->bufferReadyArray.at(nextIndex) == false){

			//actual data acquisition could be placed here. the content of this->buffer->bufferArray[nextIndex] could be modified here, but the acquisition buffer already contains the desired data so we just set the bufferReadyArray to true
			//set bufferReadyArray to true to allow processing of buffer
			this->buffer->bufferReadyArray[nextIndex] = true;

		}
		//user defined wait time
		if(this->currParams.waitTimeUs > 0){
			QThread::usleep((this->currParams.waitTimeUs));
		}
	}
}

void VirtualOCTSystem::acqcuisitionSimulationLargeFile() {
	//calculate size of buffer
	uint numberOfElements = this->currParams.depth * this->currParams.width * this->currParams.height;
	uint sizeOfElement = ceil((double)this->currParams.bitDepth / 8.0);
	size_t bufferSizeInBytes = numberOfElements*sizeOfElement;
	size_t offsetInBytes = this->currParams.bscanOffset * this->currParams.width * this->currParams.height * sizeOfElement;


	//init ifstream
	fclose(this->file); //we are going to use ifstream and do not need FILE* file, so we close it without doing anything with it
	std::ifstream bigFile(this->currParams.filePath.toLatin1(), std::ifstream::in | std::ifstream::binary);
	if(!bigFile){
		emit error(tr("could not open file"));
		return;
	}
	//set stream buffer
	bigFile.seekg(offsetInBytes);
	//bigFile.rdbuf()->pubsetbuf(static_cast<char*>(this->streamBuffer->bufferArray[0]), STREAM_BUFFER_SIZE);
	int readBuffers = 0;

	//acquisition begins!
	emit enableGui(false);
	this->acqusitionRunning = true;
	this->buffer->currIndex = 1;
	int nextIndex = 0;
	emit acquisitionStarted(this);
	bool syncEnabled = true;
	while (this->acqusitionRunning) {
		//wait until processing thread is done with copying data from previous buffer. This is not necessary in real oct systems, since they usually do not provide new data as fast as this virtual oct system. In real oct systems just check the bufferReadyArray flag of the next buffer.
		if(syncEnabled){
			while(this->buffer->currIndex >= 0 && this->buffer->bufferReadyArray.at(this->buffer->currIndex) == true && this->acqusitionRunning && syncEnabled){
				QCoreApplication::processEvents();
				syncEnabled = this->currParams.syncWithProcessing;
			}
		}

		if(this->handleRawOnlyModeBuffer()){
			continue;
		}

		//check bufferReadyArray flag to see if acquisition system is allowed to reuse this buffer and write new data in acquisition buffer. Once the bufferReadyArray flag is false, the acquisition system is allowed to reuse the buffer. If bufferReadyArray is true the processing thread is still copying data from the buffer.
		if(this->buffer->bufferReadyArray.at(nextIndex) == false){
			//get current buffer positions
			void* currAcquisitionBuf = static_cast<void*>(this->buffer->bufferArray[nextIndex]);

			//copy data from file to acquisitionBuffer
			bigFile.read(static_cast<char*>(currAcquisitionBuf), bufferSizeInBytes);

			//rewind file if necessary
			readBuffers++;
			if(readBuffers >= this->currParams.buffersFromFile){
				bigFile.seekg(offsetInBytes);
				readBuffers = 0;
			}

			//set acquisition buffer index, so that processing thread knows current buffer
			this->buffer->currIndex = nextIndex;

			//set bufferReadyArray to true to allow processing of buffer
			this->buffer->bufferReadyArray[nextIndex] = true;

			//calculate index of next buffer
			nextIndex = (this->buffer->currIndex+1)%2;
		}
		//user defined wait time
		if(this->currParams.waitTimeUs > 0){
			QThread::usleep((this->currParams.waitTimeUs));
		}
	}
}


void VirtualOCTSystem::acquisitionSimulationWithMultiFileBuffers() {
	//calculate size of buffer
	uint numberOfElements = this->currParams.depth * currParams.width * currParams.height;
	uint sizeOfElement = ceil((double)this->currParams.bitDepth / 8.0);
	size_t bufferSizeInBytes = numberOfElements*sizeOfElement;
	size_t offsetInBytes = this->currParams.bscanOffset * this->currParams.width * this->currParams.height * sizeOfElement;

	//read data from file into file buffers
	for(int i = 0; i < currParams.buffersFromFile; i++){
		void* buf = static_cast<void*>(this->streamBuffer->bufferArray[i]);
		fseek(this->file, static_cast<long>(i*numberOfElements*sizeOfElement + offsetInBytes), SEEK_SET);
		fread(buf, static_cast<size_t>(sizeOfElement), static_cast<size_t>(numberOfElements), this->file);
	}

	//close file
	fclose(this->file);
	qDebug() << "file closed";

	//acquisition begins!
	emit enableGui(false);
	this->acqusitionRunning = true;
	this->buffer->currIndex = 0;
	int nextIndex = 1;
	int streamBufferIndex = currParams.buffersFromFile-1;
	emit acquisitionStarted(this);
	bool syncEnabled = true;
	while (this->acqusitionRunning) {
		//wait until processing thread is done with copying data from previous buffer. This is not necessary in real oct systems, since they usually do not provide new data as fast as this virtual oct system. In real oct systems just check the bufferReadyArray flag of the next buffer.
		if(syncEnabled){
			while(this->buffer->currIndex >= 0 && this->buffer->bufferReadyArray.at(this->buffer->currIndex) == true && this->acqusitionRunning && syncEnabled){
				QCoreApplication::processEvents();
				syncEnabled = this->currParams.syncWithProcessing;
			}
		}

		if(this->handleRawOnlyModeBuffer()){
			continue;
		}

		//set acquisition buffer index, so that processing thread knows current buffer
		this->buffer->currIndex = nextIndex;

		//check bufferReadyArray flag to see if acquisition system is allowed to reuse this buffer and write new data in acquisition buffer. Once the bufferReadyArray flag is false, the acquisition system is allowed to reuse the buffer. If bufferReadyArray is true the processing thread is still copying data from the buffer.
		if(this->buffer->bufferReadyArray.at(nextIndex) == false){
			//get current buffer positions
			streamBufferIndex = (streamBufferIndex+1)%currParams.buffersFromFile;
			void* currAcquisitionBuf = static_cast<void*>(this->buffer->bufferArray[nextIndex]);
			void* currMultiBuf = static_cast<void*>(this->streamBuffer->bufferArray[streamBufferIndex]);

			//copy data from streamBuffer to acquisitionBuffer
			memcpy(currAcquisitionBuf, currMultiBuf, bufferSizeInBytes);

			//set bufferReadyArray to true to allow processing of buffer
			this->buffer->bufferReadyArray[nextIndex] = true;

			//calculate index of next buffer
			nextIndex = (this->buffer->currIndex+1)%2;
		}
		//user defined wait time
		if(this->currParams.waitTimeUs > 0){
			QThread::usleep((this->currParams.waitTimeUs));
		}
	}
}

void VirtualOCTSystem::slot_updateParams(simulatorParams newParams){
	this->normalParams = newParams;
	if (!this->rawOnlyModeEnabled) {
		this->updateCurrentAcquisitionParams();
	}

	//store settings, so settings can be reloaded into gui at next start of application
	this->storeCurrentSettings();
}

bool VirtualOCTSystem::supportsRawOnlyMode() const {
	return true;
}

void VirtualOCTSystem::setRawOnlyMode(bool enabled) {
	bool previousMode = this->rawOnlyModeEnabled;
	this->rawOnlyModeEnabled = enabled;
	this->updateCurrentAcquisitionParams();
	if (this->acqusitionRunning && !this->preloadActiveBufferFromFile()) {
		this->rawOnlyModeEnabled = previousMode;
		this->updateCurrentAcquisitionParams();
		emit error(tr("Failed to switch Virtual OCT raw only mode."));
	}
	this->storeCurrentSettings();
}

bool VirtualOCTSystem::isRawOnlyModeEnabled() const {
	return this->rawOnlyModeEnabled;
}

void VirtualOCTSystem::setRawOnlyModeParams(const AcquisitionParams& params) {
	AcquisitionParams previousParams = this->rawOnlyParams;
	this->rawOnlyParams = params;
	if (this->rawOnlyModeEnabled) {
		this->updateCurrentAcquisitionParams();
		if (this->acqusitionRunning && !this->preloadActiveBufferFromFile()) {
			this->rawOnlyParams = previousParams;
			this->updateCurrentAcquisitionParams();
			emit error(tr("Failed to update Virtual OCT raw only parameters."));
		}
	}
	this->storeCurrentSettings();
}

AcquisitionParams VirtualOCTSystem::getRawOnlyModeParams() const {
	return this->rawOnlyParams;
}

AcquisitionParams VirtualOCTSystem::acquisitionParamsFromSimulatorParams(const simulatorParams& params) const {
	AcquisitionParams acquisitionParams;
	acquisitionParams.samplesPerLine = params.width;
	acquisitionParams.ascansPerBscan = params.height;
	acquisitionParams.bscansPerBuffer = params.depth;
	acquisitionParams.buffersPerVolume = params.buffersPerVolume;
	acquisitionParams.bitDepth = params.bitDepth;
	return acquisitionParams;
}

simulatorParams VirtualOCTSystem::simulatorParamsFromRawOnlyParams() const {
	simulatorParams params = this->normalParams;
	params.width = static_cast<int>(this->rawOnlyParams.samplesPerLine);
	params.height = static_cast<int>(this->rawOnlyParams.ascansPerBscan);
	params.depth = static_cast<int>(this->rawOnlyParams.bscansPerBuffer);
	params.buffersPerVolume = static_cast<int>(this->rawOnlyParams.buffersPerVolume);
	params.bitDepth = static_cast<int>(this->rawOnlyParams.bitDepth);
	return params;
}

void VirtualOCTSystem::updateCurrentAcquisitionParams() {
	this->currParams = this->rawOnlyModeEnabled ? this->simulatorParamsFromRawOnlyParams() : this->normalParams;
	this->buffer = this->activeAcquisitionBuffer();
	this->params->slot_updateParams(this->acquisitionParamsFromSimulatorParams(this->normalParams));
}

void VirtualOCTSystem::storeCurrentSettings() {
	this->systemDialog->getSettings(&this->settingsMap);
	this->settingsMap.insert(RAW_ONLY_BITDEPTH, this->rawOnlyParams.bitDepth);
	this->settingsMap.insert(RAW_ONLY_WIDTH, this->rawOnlyParams.samplesPerLine);
	this->settingsMap.insert(RAW_ONLY_HEIGHT, this->rawOnlyParams.ascansPerBscan);
	this->settingsMap.insert(RAW_ONLY_DEPTH, this->rawOnlyParams.bscansPerBuffer);
	this->settingsMap.insert(RAW_ONLY_BUFFERS_PER_VOLUME, this->rawOnlyParams.buffersPerVolume);
	emit storeSettings(this->name, this->settingsMap);
}
