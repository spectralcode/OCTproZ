/**
**  This file is part of OCTproZ.
**  OCTproZ is an open source software for processig of optical
**  coherence tomography (OCT) raw data.
**  Copyright (C) 2019-2022 Miroslav Zabic
**
**  OCTproZ is free software: you can redistribute it and/or modify
**  it under the terms of the GNU General Public License as published by
**  the Free Software Foundation, either version 3 of the License, or
**  (at your option) any later version.
**
**  This program is distributed in the hope that it will be useful,
**  but WITHOUT ANY WARRANTY; without even the implied warranty of
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
**  GNU General Public License for more details.
**
**  You should have received a copy of the GNU General Public License
**  along with this program. If not, see http://www.gnu.org/licenses/.
**
****
** Author:	Miroslav Zabic
** Contact:	zabic
**			at
**			spectralcode.de
****
**/

#include "gpu2hostnotifier.h"

Gpu2HostNotifier* Gpu2HostNotifier::gpu2hostNotifier = nullptr;

Gpu2HostNotifier::Gpu2HostNotifier(QObject *parent)
	: QObject(parent)
{
}
Gpu2HostNotifier* Gpu2HostNotifier::getInstance(QObject* parent) {
	gpu2hostNotifier = gpu2hostNotifier != nullptr ? gpu2hostNotifier : new Gpu2HostNotifier(parent);
	return gpu2hostNotifier;
}

Gpu2HostNotifier::~Gpu2HostNotifier()
{
}

void Gpu2HostNotifier::emitCurrentStreamingBuffer(void* streamingBuffer) {
	OctAlgorithmParameters* params = OctAlgorithmParameters::getInstance();
	emit newGpuDataAvailible(streamingBuffer, params->bitDepth, params->samplesPerLine / 2, params->ascansPerBscan, params->bscansPerBuffer, params->buffersPerVolume, params->currentBufferNr);
}

void CUDART_CB Gpu2HostNotifier::dh2StreamingCallback(cudaStream_t event, cudaError_t status, void *currStreamingBuffer) {
	checkCudaErrors(status);
	Gpu2HostNotifier::getInstance()->emitCurrentStreamingBuffer(currStreamingBuffer);
}

