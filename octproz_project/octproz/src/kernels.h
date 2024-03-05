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

#ifndef KERNELS_H
#define KERNELS_H


#ifdef _WIN32
	#define WINDOWS_LEAN_AND_MEAN
	//#define NOMINMAX
	#include <windows.h>
	#include "GL/glew.h"
#endif

//CUDA FFT
#include <cufft.h>
#include <cufftXt.h>

//include gl headers before cuda_gl_interop.h for aarch64 (jetson nano)
#ifdef __aarch64__
	#include <QtGui/qopengl.h>
#endif

//CUDA Runtime, Interop, and includes
#include <cuda_gl_interop.h>
#include <cuda_profiler_api.h>
#include <driver_functions.h>

//Helper functions, CUDA utilities
#include <helper_cuda.h>
#include <cuda_fp16.h>

//OCTproZ structs and classes
#include "octalgorithmparameters.h"
#include "gpu2hostnotifier.h"


//cuda_code.cu
extern "C" bool initializeCuda(void* h_buffer1, void* h_buffer2, OctAlgorithmParameters* dispParameters);
extern "C" void octCudaPipeline(void* h_inputSignal);
extern "C" void releaseBuffers();
extern "C" void destroyStreamsAndEvents();
extern "C" void cleanupCuda();
extern "C" void freeCudaMem(void** data);
extern "C" void cuda_registerStreamingBuffers(void* h_streamingBuffer1, void* h_streamingBuffer2, size_t bytesPerBuffer);
extern "C" void cuda_unregisterStreamingBuffers();
extern "C" void cuda_registerGlBufferBscan(GLuint buf);
extern "C" void cuda_registerGlBufferEnFaceView(GLuint buf);
extern "C" void cuda_registerGlBufferVolumeView(GLuint buf);

extern void* cuda_map(cudaGraphicsResource* res, cudaStream_t stream);
extern void cuda_unmap(cudaGraphicsResource* res, cudaStream_t stream);
extern cudaArray* cuda_map3dTexture(cudaGraphicsResource* res, cudaStream_t stream);

extern "C" void changeDisplayedBscanFrame(unsigned int frameNr, unsigned int displayFunctionFrames, int displayFunction); ///if framerate is low user can request another bscan to be displayed from already acquired buffer with this function
extern "C" void changeDisplayedEnFaceFrame(unsigned int frameNr, unsigned int displayFunctionFrames, int displayFunction);
extern "C" inline void updateBscanDisplayBuffer(unsigned int frameNr, unsigned int displayFunctionFrames, int displayFunction, cudaStream_t stream); ///as soon as new buffer is acquired this function is called and the display buffer gets updated
extern "C" inline void updateEnFaceDisplayBuffer(unsigned int frameNr, unsigned int displayFunctionFrames, int displayFunction, cudaStream_t stream); ///as soon as new buffer is acquired this function is called and the display buffer gets updated

#endif // KERNELS_H
