/*
MIT License

Copyright (c) 2019-2021 Miroslav Zabic

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

#include "acquisitionbuffer.h"


AcquisitionBuffer::AcquisitionBuffer() : QObject() {
	this->bufferCnt = 0;
	this->bytesPerBuffer = bytesPerBuffer;
	this->bufferArray.resize(bufferCnt);
	this->bufferReadyArray.resize(bufferCnt);
	this->currIndex = -1;

	for (unsigned int i = 0; i < this->bufferCnt; i++) {
		this->bufferArray[i] = nullptr;
		this->bufferReadyArray[i] = false;
	}
}

AcquisitionBuffer::~AcquisitionBuffer() {
	releaseMemory();
}

bool AcquisitionBuffer::allocateMemory(unsigned int bufferCnt, size_t bytesPerBuffer) {
	this->bufferCnt = bufferCnt;
	this->bytesPerBuffer = bytesPerBuffer;
	this->releaseMemory();
	this->bufferArray.clear();
	this->bufferArray.resize(bufferCnt);
	this->bufferReadyArray.resize(bufferCnt);
	bool success = true;

	// Allocate page aligned memory
	for (unsigned int bufferIndex = 0; (bufferIndex < this->bufferCnt) && success; bufferIndex++) {
		int err = posix_memalign((void**)&(bufferArray[bufferIndex]), 128, bytesPerBuffer);
		if (err != 0 || bufferArray[bufferIndex] == nullptr){
			emit error(tr("Buffer memory allocation error. posix_memalign() error code: ") + QString::number(err));
			success = false;
		}else{
			memset((bufferArray[bufferIndex]), 0, bytesPerBuffer);
		}
	}
	return success;
}

void AcquisitionBuffer::releaseMemory() {
	for (int i = 0; i < this->bufferArray.size(); i++) {
		if (this->bufferArray[i] != nullptr) {
			posix_memalign_free(this->bufferArray[i]);
			this->bufferArray[i] = nullptr;
			this->bufferReadyArray[i] = false;
		}
	}
}
