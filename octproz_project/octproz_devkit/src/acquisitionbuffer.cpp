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

#include "acquisitionbuffer.h"

#ifdef __aarch64__
	#include <cuda_runtime.h>
	#include <helper_cuda.h>
#endif


AcquisitionBuffer::AcquisitionBuffer() : QObject() {
	this->bufferCnt = 0;
	this->bytesPerBuffer = 0;
	this->currIndex = -1;
}

AcquisitionBuffer::~AcquisitionBuffer() {
	releaseMemory();
}

bool AcquisitionBuffer::allocateMemory(unsigned int bufferCnt, size_t bytesPerBuffer) {
	this->releaseMemory();
	this->bufferCnt = bufferCnt;
	this->bytesPerBuffer = bytesPerBuffer;
	this->bufferArray.resize(this->bufferCnt);
	this->bufferReadyArray.resize(this->bufferCnt);
	bool success = true;

	for (unsigned int bufferIndex = 0; (bufferIndex < this->bufferCnt) && success; bufferIndex++) {
		#ifdef __aarch64__
			#ifdef ENABLE_CUDA_ZERO_COPY
				cudaError_t err = cudaHostAlloc((void**)&(this->bufferArray[bufferIndex]), this->bytesPerBuffer, cudaHostAllocMapped);
			#else
				cudaError_t err = cudaHostAlloc((void**)&(this->bufferArray[bufferIndex]), this->bytesPerBuffer, cudaHostAllocPortable);
			#endif
			if (err != cudaSuccess || this->bufferArray[bufferIndex] == nullptr){
				emit error(tr("Buffer memory allocation error. cudaHostAlloc() error code: ") + QString::number(err));
				success = false;
			} else {
				cudaMemset(this->bufferArray[bufferIndex], 0, this->bytesPerBuffer);
			}
		#else
			int err = posix_memalign((void**)&(this->bufferArray[bufferIndex]), 128, this->bytesPerBuffer);
			if (err != 0 || this->bufferArray[bufferIndex] == nullptr){
				emit error(tr("Buffer memory allocation error. posix_memalign() error code: ") + QString::number(err));
				success = false;
			} else {
				memset(this->bufferArray[bufferIndex], 0, this->bytesPerBuffer);
			}
		#endif
		this->bufferReadyArray[bufferIndex] = false;
	}
	return success;
}

void AcquisitionBuffer::releaseMemory() {
	for (int i = 0; i < this->bufferArray.size(); i++) {
		if (this->bufferArray[i] != nullptr) {
			#ifdef __aarch64__
				cudaFreeHost(this->bufferArray[i]);
			#else
				posix_memalign_free(this->bufferArray[i]);
			#endif
			this->bufferArray[i] = nullptr;
			this->bufferReadyArray[i] = false;
		}
	}
	this->bufferArray.clear();
	this->bufferReadyArray.clear();
}
