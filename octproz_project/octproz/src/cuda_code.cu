/**
**  This file is part of OCTproZ.
**  OCTproZ is an open source software for processig of optical
**  coherence tomography (OCT) raw data.
**  Copyright (C) 2019-2024 Miroslav Zabic
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

#ifndef CUDA_CODE_CU
#define CUDA_CODE_CU

#include "kernels.h"
#include <cfloat>

#define EIGHT_OVER_PI_SQUARED 0.8105694691f
#define PI_OVER_8 0.3926990817f
#define PI 3.141592654f


int blockSize;
int gridSize;
int threadsPerBlockLimit;
const int nStreams = 8;
int currStream = 0;

#if __CUDACC_VER_MAJOR__ <12
surface<void, cudaSurfaceType3D> surfaceWrite;
#endif

cudaStream_t stream[nStreams];
cudaStream_t userRequestStream;

cudaEvent_t syncEvent;

cudaGraphicsResource* cuBufHandleBscan = NULL;
cudaGraphicsResource* cuBufHandleEnFaceView = NULL;
cudaGraphicsResource* cuBufHandleVolumeView = NULL;

const int nBuffers = 1;
int currBuffer = 0;
void* d_inputBuffer[nBuffers];
void* d_outputBuffer = NULL;

void* host_buffer1 = NULL;
void* host_buffer2 = NULL;
void* host_streamingBuffer1 = NULL;
void* host_streamingBuffer2 = NULL;
void* host_floatStreamingBuffer1 = nullptr;
void* host_floatStreamingBuffer2 = nullptr;
bool floatStreamingBuffersRegistered = false;

cufftComplex* d_inputLinearized = NULL;
float* d_windowCurve= NULL;
float* d_resampleCurve = NULL;
float* d_dispersionCurve = NULL;
float* d_sinusoidalResampleCurve = NULL;
cufftComplex* d_phaseCartesian = NULL;
unsigned int bufferNumber = 0;
unsigned int bufferNumberInVolume = 0;
unsigned int streamingBufferNumber = 0;
static int floatStreamingBufferNumber = 0;

cufftComplex* d_fftBuffer = NULL;
cufftHandle d_plan = 0;
cufftComplex* d_meanALine = NULL;
float* d_postProcBackgroundLine = NULL;

bool cudaInitialized = false;
bool saveToDisk = false;

int signalLength = 0;
int ascansPerBscan = 0;
int bscansPerBuffer = 0;
int samplesPerBuffer = 0;
int samplesPerVolume = 0;
int buffersPerVolume = 0;
int bytesPerSample = 0;

float* d_processedBuffer = NULL;
float* d_sinusoidalScanTmpBuffer = NULL;
OctAlgorithmParameters* params = NULL;

unsigned int processedBuffers;
unsigned int streamedBuffers;

bool fixedPatternNoiseDetermined = false;



__global__ void inputToCufftComplex(cufftComplex* output, const void* input, const int width_out, const int width_in, const int inputBitdepth, const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(inputBitdepth <= 8){
		unsigned char* in = (unsigned char*)input;
		output[index].x = __uint2float_rd(in[index]);
	}else if(inputBitdepth > 8 && inputBitdepth <= 16){
		unsigned short* in = (unsigned short*)input;
		output[index].x = __uint2float_rd(in[index]);
	}else{
		unsigned int* in = (unsigned int*)input;
		output[index].x = __uint2float_rd(in[index]);
	}
	output[index].y = 0;
}

__global__ void inputToCufftComplex_and_bitshift(cufftComplex* output, const void* input, const int width_out, const int width_in, const int inputBitdepth, const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(inputBitdepth <= 8){
		unsigned char* in = (unsigned char*)input;
		output[index].x = __uint2float_rd(in[index] >> 4);
	}else if(inputBitdepth > 8 && inputBitdepth <= 16){
		unsigned short* in = (unsigned short*)input;
		output[index].x = __uint2float_rd(in[index] >> 4);
	}else{
		unsigned int* in = (unsigned int*)input;
		output[index].x = (in[index])/4294967296.0;
	}
	output[index].y = 0;
}

//device functions for endian byte swap //todo: check if big endian to little endian conversion may be needed and extend inputToCufftComplex kernel if necessary
inline __device__ uint32_t endianSwapUint32(uint32_t val) {
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | ((val >> 16));
}
inline __device__ int32_t endianSwapInt32(int32_t val) {
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF );
	return (val << 16) | ((val >> 16) & 0xFFFF);
}
inline __device__ uint16_t endianSwapUint16(uint16_t val) {
	return (val << 8) | (val >> 8 );
}
inline __device__ int16_t endianSwapInt16(int16_t val) {
	return (val << 8) | ((val >> 8) & 0xFF);
}

__global__ void rollingAverageBackgroundRemoval(cufftComplex* out, cufftComplex* in, const int rollingAverageWindowSize, const int width, const int height, const int samplesPerFrame, const int samples) { //width: samplesPerAscan; height: ascansPerBscan,samples: total number of samples in buffer
	extern __shared__ float s_data[];

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < samples) {
		int currentBscan = index / samplesPerFrame;
		int currentLine = (index / width) % height;
		int firstIndexOfCurrentLine = currentLine * width + (samplesPerFrame * currentBscan);
		int lastIndexOfCurrentLine = firstIndexOfCurrentLine + width - 1;

		int startIdx = max(firstIndexOfCurrentLine, index - rollingAverageWindowSize + 1);
		int endIdx = min(lastIndexOfCurrentLine, index + rollingAverageWindowSize);
		int windowSize = endIdx - startIdx + 1;

		//load data into shared memory for this line segment
		//first determine the range of data this block will process
		int blockFirstIdx = blockIdx.x * blockDim.x;
		int blockStartIdx = max(0, blockFirstIdx - rollingAverageWindowSize + 1);
		int blockEndIdx = min(samples-1, (blockFirstIdx + blockDim.x - 1) + rollingAverageWindowSize);

		//load data collaboratively (each thread loads one or more elements)
		for (int i = blockStartIdx + threadIdx.x; i <= blockEndIdx ; i += blockDim.x) {
				s_data[i - blockStartIdx] = in[i].x;
		}

		//ensure all data is loaded before proceeding
		__syncthreads();

		//calculate rolling average using shared memory
		float rollingSum = 0.0f;
		for (int i = startIdx; i <= endIdx; i++) {
			rollingSum += s_data[i - blockStartIdx];
		}

		float rollingAverage = rollingSum / windowSize;
		out[index].x = in[index].x - rollingAverage;
		out[index].y = 0;
	}
}

//todo: use/evaluate cuda texture for interpolation in klinearization kernel
__global__ void klinearization(cufftComplex* out, cufftComplex *in, const float* resampleCurve, const int width, const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int j = index%width;
	int offset = index-j;

	float x = resampleCurve[j];
	int x0 = (int)x;
	int x1 = x0 + 1;

	float f_x0 = in[offset + x0].x;
	float f_x1 = in[offset + x1].x;

	out[index].x = f_x0 + (f_x1 - f_x0) * (x - x0);
	out[index].y = 0;
}

__global__ void klinearizationQuadratic(cufftComplex* out, cufftComplex *in, const float* resampleCurve, const int width, const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int j = index%width;
	int offset = index-j;

	float x = resampleCurve[j];
	int x0 = (int)x;
	int x1 = x0 + 1;
	int x2 = x0 + 2;

	float f_x0 = in[offset + x0].x;
	float f_x1 = in[offset + x1].x;
	float f_x2 = in[offset + x2].x;
	float b0 = f_x0;
	float b1 = f_x1-f_x0;
	float b2 = ((f_x2-f_x1)-b1)/(x2-x0);

	out[index].x = b0 + b1 * (x - x0) + b2*(x-x0)*(x-x1);
	out[index].y = 0;
}

inline __device__ float cubicHermiteInterpolation(const float y0, const float y1, const float y2, const float y3, const float positionBetweenY1andY2){
	const float a = -y0 + 3.0f*(y1-y2) + y3;
	const float b = 2.0f*y0 - 5.0f*y1 + 4.0f*y2 - y3;
	const float c = -y0 + y2;

	const float pos = positionBetweenY1andY2;
	const float pos2 = pos*pos;

	return 0.5f*pos*(a * pos2 + b * pos + c) + y1;
}

__global__ void klinearizationCubic(cufftComplex* out, cufftComplex *in, const float* resampleCurve, const int width, const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int j = index%width;
	int offset = index-j;

	float nx = resampleCurve[j];
	const int n1 = (int)nx;
	int n0 = abs(n1 - 1); //using abs() to avoid negative n0 because offset can be 0 and out of bounds memory access may occur
	int n2 = n1 + 1;
	int n3 = n2 + 1; //we do not need to worry here about out of bounds memory access as the resampleCurve is restricted to values that avoid out of bound memory acces in resample kernels

	float y0 = in[offset + n0].x;
	float y1 = in[offset + n1].x;
	float y2 = in[offset + n2].x;
	float y3 = in[offset + n3].x;

	out[index].x = cubicHermiteInterpolation(y0,y1,y2,y3,nx-n1);
	out[index].y = 0;
}

inline __device__ float lanczosKernel(const float a, const float x) {
	if(x < 0.00000001f && x > -0.00000001){
		return 1.0f;
	}
	if(x >= -a || x < a){
		return (a*sinf(M_PI*x)*sinf(M_PI*x/a))/(M_PI*M_PI*x*x); //todo: optimize
	}
	return 0.0f;
}

//inline __device__ float lanczosKernel8(const float x) {
//	if(x < 0.00001f && x > -0.00001f) {
//		return 1.0f;
//	}
//	if(x >= -8.0f || x < 8.0f) {
//		return (EIGHT_OVER_PI_SQUARED*__sinf(PI*x)*__sinf(PI_OVER_8*x))/(x*x);
//	}
//	return 0.0f;
//}

inline __device__ float lanczosKernel8(const float x) {
	const float absX = fabsf(x);
	const float sincX = sinf(PI*absX)/(PI*absX);
	const float sincXOver8 = sinf(PI_OVER_8*absX)/(PI_OVER_8*absX);
	return (absX < 0.00001f) ? 1.0f :(sincX * sincXOver8);
}

__global__ void klinearizationLanczos(cufftComplex* out, cufftComplex *in, const float* resampleCurve, const int width, const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int j = index%width;
	int offset = index-j;
	offset = min(samples-9, max(offset, 8));

	float nx = resampleCurve[j];
	const int n0 = (int)nx;
	int nm7 = (n0 - 7);
	int nm6 = (n0 - 6);
	int nm5 = (n0 - 5);
	int nm4 = (n0 - 4);
	int nm3 = (n0 - 3);
	int nm2 = (n0 - 2);
	int nm1 = (n0 - 1);
	int n1 = (n0 + 1);
	int n2 = (n0 + 2);
	int n3 = (n0 + 3);
	int n4 = (n0 + 4);
	int n5 = (n0 + 5);
	int n6 = (n0 + 6);
	int n7 = (n0 + 7);
	int n8 = (n0 + 8);

	float ym7 = in[offset + nm7].x;
	float ym6 = in[offset + nm6].x;
	float ym5 = in[offset + nm5].x;
	float ym4 = in[offset + nm4].x;
	float ym3 = in[offset + nm3].x;
	float ym2 = in[offset + nm2].x;
	float ym1 = in[offset + nm1].x;
	float y0 = in[offset + n0].x;
	float y1 = in[offset + n1].x;
	float y2 = in[offset + n2].x;
	float y3 = in[offset + n3].x;
	float y4 = in[offset + n4].x;
	float y5 = in[offset + n5].x;
	float y6 = in[offset + n6].x;
	float y7 = in[offset + n7].x;
	float y8 = in[offset + n8].x;

	float sm7 = ym7 * lanczosKernel8(nx-nm7);
	float sm6 = ym6 * lanczosKernel8(nx-nm6);
	float sm5 = ym5 * lanczosKernel8(nx-nm5);
	float sm4 = ym4 * lanczosKernel8(nx-nm4);
	float sm3 = ym3 * lanczosKernel8(nx-nm3);
	float sm2 = ym2 * lanczosKernel8(nx-nm2);
	float sm1 = ym1 * lanczosKernel8(nx-nm1);
	float s0 = y0 * lanczosKernel8(nx-n0);
	float s1 = y1 * lanczosKernel8(nx-n1);
	float s2 = y2 * lanczosKernel8(nx-n2);
	float s3 = y3 * lanczosKernel8(nx-n3);
	float s4 = y4 * lanczosKernel8(nx-n4);
	float s5 = y5 * lanczosKernel8(nx-n5);
	float s6 = y6 * lanczosKernel8(nx-n6);
	float s7 = y7 * lanczosKernel8(nx-n7);
	float s8 = y8 * lanczosKernel8(nx-n8);

	out[index].x = sm7 + sm6 + sm5 + sm4 + sm3 + sm2 + sm1 + s0+ s1 + s2 + s3 +s4 +s5 +s6 +s7 + s8;
	out[index].y = 0;
}

__global__ void windowing(cufftComplex* output, cufftComplex* input, const float* window, const int lineWidth, const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < samples) {
		int line_index = index % lineWidth;
		output[index].x = input[index].x * window[line_index];
		output[index].y = 0;
	}
}

__global__ void klinearizationAndWindowing(cufftComplex* out, cufftComplex *in, const float* resampleCurve, const float* window, const int width, const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int j = index%width;
	int offset = index-j;

	float n_m = resampleCurve[j];
	int n1 = (int)n_m;
	int n2 = n1 + 1;

	float inN1 = in[offset + n1].x;
	float inN2 = in[offset + n2].x;

	out[index].x = (inN1 + (inN2 - inN1) * (n_m - n1)) * window[j];
	out[index].y = 0;
}

__global__ void klinearizationCubicAndWindowing(cufftComplex* out, cufftComplex *in, const float* resampleCurve, const float* window, const int width, const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int j = index%width;
	int offset = index-j;

	float nx = resampleCurve[j];
	const int n1 = (int)nx;
	int n0 = abs(n1 - 1);
	int n2 = n1 + 1;
	int n3 = n2 + 1;

	float y0 = in[offset + n0].x;
	float y1 = in[offset + n1].x;
	float y2 = in[offset + n2].x;
	float y3 = in[offset + n3].x;
	float pos = nx-n1;

	out[index].x = cubicHermiteInterpolation(y0,y1,y2,y3,pos) * window[j];
	out[index].y = 0;
}

__global__ void klinearizationLanczosAndWindowing(cufftComplex* out, cufftComplex *in, const float* resampleCurve, const float* window, const int width, const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int j = index%width;
	int offset = index-j;
	offset = min(samples-9, max(offset, 8));

	float nx = resampleCurve[j];
	const int n0 = (int)nx;
	int nm7 = (n0 - 7);
	int nm6 = (n0 - 6);
	int nm5 = (n0 - 5);
	int nm4 = (n0 - 4);
	int nm3 = (n0 - 3);
	int nm2 = (n0 - 2);
	int nm1 = (n0 - 1);
	int n1 = (n0 + 1);
	int n2 = (n0 + 2);
	int n3 = (n0 + 3);
	int n4 = (n0 + 4);
	int n5 = (n0 + 5);
	int n6 = (n0 + 6);
	int n7 = (n0 + 7);
	int n8 = (n0 + 8);

	float ym7 = in[offset + nm7].x;
	float ym6 = in[offset + nm6].x;
	float ym5 = in[offset + nm5].x;
	float ym4 = in[offset + nm4].x;
	float ym3 = in[offset + nm3].x;
	float ym2 = in[offset + nm2].x;
	float ym1 = in[offset + nm1].x;
	float y0 = in[offset + n0].x;
	float y1 = in[offset + n1].x;
	float y2 = in[offset + n2].x;
	float y3 = in[offset + n3].x;
	float y4 = in[offset + n4].x;
	float y5 = in[offset + n5].x;
	float y6 = in[offset + n6].x;
	float y7 = in[offset + n7].x;
	float y8 = in[offset + n8].x;

	float sm7 = ym7 * lanczosKernel8(nx-nm7);
	float sm6 = ym6 * lanczosKernel8(nx-nm6);
	float sm5 = ym5 * lanczosKernel8(nx-nm5);
	float sm4 = ym4 * lanczosKernel8(nx-nm4);
	float sm3 = ym3 * lanczosKernel8(nx-nm3);
	float sm2 = ym2 * lanczosKernel8(nx-nm2);
	float sm1 = ym1 * lanczosKernel8(nx-nm1);
	float s0 = y0 * lanczosKernel8(nx-n0);
	float s1 = y1 * lanczosKernel8(nx-n1);
	float s2 = y2 * lanczosKernel8(nx-n2);
	float s3 = y3 * lanczosKernel8(nx-n3);
	float s4 = y4 * lanczosKernel8(nx-n4);
	float s5 = y5 * lanczosKernel8(nx-n5);
	float s6 = y6 * lanczosKernel8(nx-n6);
	float s7 = y7 * lanczosKernel8(nx-n7);
	float s8 = y8 * lanczosKernel8(nx-n8);

	out[index].x = (sm7 + sm6 + sm5 + sm4 + sm3 + sm2 + sm1 + s0+ s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8) * window[j];
	out[index].y = 0;
}

__global__ void klinearizationAndWindowingAndDispersionCompensation(cufftComplex* out, cufftComplex* in, const float* resampleCurve, const float* window, const cufftComplex* phaseComplex, const int width, const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int j = index%width;
	int offset = index-j;

	float n_m = resampleCurve[j];
	int n1 = (int)n_m;
	int n2 = n1 + 1;

	float inN1 = in[offset + n1].x;
	float inN2 = in[offset + n2].x;

	float linearizedAndWindowedInX = (inN1 + (inN2 - inN1) * (n_m - n1)) * window[j];
	out[index].x = linearizedAndWindowedInX * phaseComplex[j].x;
	out[index].y = linearizedAndWindowedInX * phaseComplex[j].y;
}

__global__ void klinearizationCubicAndWindowingAndDispersionCompensation(cufftComplex* out, cufftComplex *in, const float* resampleCurve, const float* window, const cufftComplex* phaseComplex, const int width, const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int j = index%width;
	int offset = index-j;

	float nx = resampleCurve[j];
	int n1 = (int)nx;
	int n0 = abs(n1 - 1);
	int n2 = n1 + 1;
	int n3 = n2 + 1;

	float y0 = in[offset + n0].x;
	float y1 = in[offset + n1].x;
	float y2 = in[offset + n2].x;
	float y3 = in[offset + n3].x;
	float pos = nx-n1;

	float linearizedAndWindowedInX = cubicHermiteInterpolation(y0,y1,y2,y3,pos) * window[j];
	out[index].x = linearizedAndWindowedInX * phaseComplex[j].x;
	out[index].y = linearizedAndWindowedInX * phaseComplex[j].y;
}

__global__ void klinearizationLanczosAndWindowingAndDispersionCompensation(cufftComplex* out, cufftComplex *in, const float* resampleCurve, const float* window, const cufftComplex* phaseComplex, const int width, const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int j = index%width;
	int offset = index-j;
	offset = min(samples-9, max(offset, 8));

	float nx = resampleCurve[j];
	const int n0 = (int)nx;
	int nm7 = (n0 - 7);
	int nm6 = (n0 - 6);
	int nm5 = (n0 - 5);
	int nm4 = (n0 - 4);
	int nm3 = (n0 - 3);
	int nm2 = (n0 - 2);
	int nm1 = (n0 - 1);
	int n1 = (n0 + 1);
	int n2 = (n0 + 2);
	int n3 = (n0 + 3);
	int n4 = (n0 + 4);
	int n5 = (n0 + 5);
	int n6 = (n0 + 6);
	int n7 = (n0 + 7);
	int n8 = (n0 + 8);

	float ym7 = in[offset + nm7].x;
	float ym6 = in[offset + nm6].x;
	float ym5 = in[offset + nm5].x;
	float ym4 = in[offset + nm4].x;
	float ym3 = in[offset + nm3].x;
	float ym2 = in[offset + nm2].x;
	float ym1 = in[offset + nm1].x;
	float y0 = in[offset + n0].x;
	float y1 = in[offset + n1].x;
	float y2 = in[offset + n2].x;
	float y3 = in[offset + n3].x;
	float y4 = in[offset + n4].x;
	float y5 = in[offset + n5].x;
	float y6 = in[offset + n6].x;
	float y7 = in[offset + n7].x;
	float y8 = in[offset + n8].x;

	float sm7 = ym7 * lanczosKernel8(nx-nm7);
	float sm6 = ym6 * lanczosKernel8(nx-nm6);
	float sm5 = ym5 * lanczosKernel8(nx-nm5);
	float sm4 = ym4 * lanczosKernel8(nx-nm4);
	float sm3 = ym3 * lanczosKernel8(nx-nm3);
	float sm2 = ym2 * lanczosKernel8(nx-nm2);
	float sm1 = ym1 * lanczosKernel8(nx-nm1);
	float s0 = y0 * lanczosKernel8(nx-n0);
	float s1 = y1 * lanczosKernel8(nx-n1);
	float s2 = y2 * lanczosKernel8(nx-n2);
	float s3 = y3 * lanczosKernel8(nx-n3);
	float s4 = y4 * lanczosKernel8(nx-n4);
	float s5 = y5 * lanczosKernel8(nx-n5);
	float s6 = y6 * lanczosKernel8(nx-n6);
	float s7 = y7 * lanczosKernel8(nx-n7);
	float s8 = y8 * lanczosKernel8(nx-n8);

	float linearizedAndWindowedInX = (sm7 + sm6 + sm5 + sm4 + sm3 + sm2 + sm1 + s0+ s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8) * window[j];
	out[index].x = linearizedAndWindowedInX * phaseComplex[j].x;
	out[index].y = linearizedAndWindowedInX * phaseComplex[j].y;
}

__global__ void sinusoidalScanCorrection(float* out, float *in, float* sinusoidalResampleCurve, const int width, const int height, const int depth, const int samples) { //width: samplesPerAscan; height: ascansPerBscan, depth: bscansPerBuffer
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < samples-width){
		int j = index%(width); //pos within ascan
		int k = (index/width)%height; //pos within bscan
		int l = index/(width*height); //pos within buffer

		float n_sinusoidal = sinusoidalResampleCurve[k];
		float x = n_sinusoidal;
		int x0 = (int)x*width+j+l*width*height;
		int x1 = x0 + width;

		float f_x0 = in[x0];
		float f_x1 = in[x1];

		out[index] = f_x0 + (f_x1 - f_x0) * (x - (int)(x));
	}
}

__global__ void fillSinusoidalScanCorrectionCurve(float* sinusoidalResampleCurve,  const int length) {
	int index = blockIdx.x;
	if (index < length) {
		sinusoidalResampleCurve[index] = ((float)length/M_PI)*acos((float)(1.0-((2.0*(float)index)/(float)length)));
	}
}

__global__ void getMinimumVarianceMean(cufftComplex *meanLine, const cufftComplex *in, int width, int height, int segs) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= width) return;

	int segWidth = height / segs;
	int stride = width;
	float factor = 1.0f / segWidth;

	float minVariance = FLT_MAX;
	cufftComplex meanAtMinVariance = {0.0f, 0.0f};

	for (int i = 0; i < segs; i++) {
		int offset = i * segWidth * stride + index;

		float sumX = 0.0f, sumY = 0.0f;
		float sumXX = 0.0f;

		for (int j = 0; j < segWidth; j++) {
			cufftComplex val = in[offset + j * stride];
			float dx = val.x;
			float dy = val.y;
			sumX += dx;
			sumY += dy;
			sumXX += dx * dx + dy * dy;
		}

		float meanX = sumX * factor;
		float meanY = sumY * factor;
		float variance = (sumXX * factor) - (meanX * meanX + meanY * meanY);

		if (variance < minVariance) {
			minVariance = variance;
			meanAtMinVariance.x = meanX;
			meanAtMinVariance.y = meanY;
		}
	}

	meanLine[index] = meanAtMinVariance;
}

__global__ void meanALineSubtraction(cufftComplex *in_out, cufftComplex *meanLine, int width, int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < samples) {
		int meanLineIndex = index % width;
		int lineIndex = index / width;
		int volumeArrayIndex = lineIndex * width + index;
		in_out[volumeArrayIndex].x = in_out[volumeArrayIndex].x - meanLine[meanLineIndex].x;
		in_out[volumeArrayIndex].y = in_out[volumeArrayIndex].y - meanLine[meanLineIndex].y;
	}
}

__device__ cufftComplex cuMultiply(const cufftComplex& a, const cufftComplex& b) {
	cufftComplex result;
	result.x = a.x*b.x - a.y*b.y;
	result.y = a.x*b.y + a.y*b.x;
	return result;
}

__global__ void dispersionCompensation(cufftComplex* out, cufftComplex* in, const cufftComplex* phaseComplex, const int width, const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < samples) {
		int phaseIndex = index%width;
		//because in[].y is always zero we can omit full complex multiplication and just multiply in[].x
		//for full multiplication the device kernel "cuMultiply" can be used
		float inX = in[index].x;
		out[index].x = inX * phaseComplex[phaseIndex].x;
		out[index].y = inX * phaseComplex[phaseIndex].y;
	}
}

__global__ void dispersionCompensationAndWindowing(cufftComplex* out, cufftComplex* in, const cufftComplex* phaseComplex, const float* window, const int width, const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < samples) {
		int lineIndex = index%width;
		float inX = in[index].x * window[lineIndex];
		out[index].x = inX * phaseComplex[lineIndex].x;
		out[index].y = inX * phaseComplex[lineIndex].y;
	}
}

__global__ void fillDispersivePhase(cufftComplex* phaseComplex, const float* phase, const double factor, const int width, const int direction) {
	int index = blockIdx.x;
	if (index < width) {
		phaseComplex[index].x = cosf(factor*phase[index]);
		phaseComplex[index].y = sinf(factor*phase[index]) * direction;
	}
}

extern "C" void cuda_updateDispersionCurve(float* h_dispersionCurve, int size, cudaStream_t stream) {
	if (d_dispersionCurve != NULL && h_dispersionCurve != NULL)
		checkCudaErrors(cudaMemcpyAsync(d_dispersionCurve, h_dispersionCurve, size * sizeof(float), cudaMemcpyHostToDevice, stream));
}

extern "C" void cuda_updateWindowCurve(float* h_windowCurve, int size, cudaStream_t stream) {
	if (d_windowCurve != NULL && h_windowCurve != NULL)
		checkCudaErrors(cudaMemcpyAsync(d_windowCurve, h_windowCurve, size * sizeof(float), cudaMemcpyHostToDevice, stream));
}

extern "C" void cuda_updatePostProcessBackground(float* h_postProcessBackground, int size, cudaStream_t stream) {
	if (d_postProcBackgroundLine != NULL && h_postProcessBackground != NULL){
#ifdef __aarch64__
		checkCudaErrors(cudaMemcpy(d_postProcBackgroundLine, h_postProcessBackground, size * sizeof(float), cudaMemcpyHostToDevice));
		cudaStreamSynchronize(stream);
#else
		checkCudaErrors(cudaMemcpyAsync(d_postProcBackgroundLine, h_postProcessBackground, size * sizeof(float), cudaMemcpyHostToDevice, stream));
#endif

	}
}

extern "C" void cuda_copyPostProcessBackgroundToHost(float* h_postProcessBackground, int size, cudaStream_t stream) {
	if (d_postProcBackgroundLine != NULL && h_postProcessBackground != NULL) {
#ifdef __aarch64__
		checkCudaErrors(cudaMemcpy(h_postProcessBackground, d_postProcBackgroundLine, size * sizeof(float), cudaMemcpyDeviceToHost));
		cudaStreamSynchronize(stream);
		Gpu2HostNotifier::backgroundSignalCallback(h_postProcessBackground);
#else
		checkCudaErrors(cudaMemcpyAsync(h_postProcessBackground, d_postProcBackgroundLine, size * sizeof(float), cudaMemcpyDeviceToHost, stream));
		checkCudaErrors(cudaLaunchHostFunc(stream, Gpu2HostNotifier::backgroundSignalCallback, h_postProcessBackground));
#endif
	}
}

extern "C" void cuda_registerStreamingBuffers(void* h_streamingBuffer1, void* h_streamingBuffer2, size_t bytesPerBuffer) {
#ifndef __aarch64__
	checkCudaErrors(cudaHostRegister(h_streamingBuffer1, bytesPerBuffer, cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(h_streamingBuffer2, bytesPerBuffer, cudaHostRegisterPortable));
#endif
	host_streamingBuffer1 = h_streamingBuffer1;
	host_streamingBuffer2 = h_streamingBuffer2;
}

extern "C" void cuda_unregisterStreamingBuffers() {
#ifndef __aarch64__
	checkCudaErrors(cudaHostUnregister(host_streamingBuffer1));
	checkCudaErrors(cudaHostUnregister(host_streamingBuffer2));
#endif
	host_streamingBuffer1 = NULL;
	host_streamingBuffer2 = NULL;
}

extern "C" void cuda_registerFloatStreamingBuffers(void* h_floatStreamingBuffer1, void* h_floatStreamingBuffer2, size_t bytesPerBuffer) {
	host_floatStreamingBuffer1 = h_floatStreamingBuffer1;
	host_floatStreamingBuffer2 = h_floatStreamingBuffer2;
#ifndef __aarch64__
	checkCudaErrors(cudaHostRegister(host_floatStreamingBuffer1, bytesPerBuffer, cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(host_floatStreamingBuffer2, bytesPerBuffer, cudaHostRegisterPortable));
#endif
	floatStreamingBuffersRegistered = true;
}

extern "C" void cuda_unregisterFloatStreamingBuffers() {
#ifndef __aarch64__
	checkCudaErrors(cudaHostUnregister(host_floatStreamingBuffer1));
	checkCudaErrors(cudaHostUnregister(host_floatStreamingBuffer2));
#endif
	host_floatStreamingBuffer1 = nullptr;
	host_floatStreamingBuffer2 = nullptr;
	floatStreamingBuffersRegistered = false;
}


//Removes half of each processed A-scan (the mirror artefacts), logarithmizes each value of magnitude of remaining A-scan and copies it into an output array. This output array can be used to display the processed OCT data.
__global__ void postProcessTruncateLog(float *output, const cufftComplex *input, const int outputAscanLength, const int samples, const int bufferNumberInVolume, const float max, const float min, const float addend, const float coeff) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < samples / 2) {
		int lineIndex = index / outputAscanLength;
		int inputArrayIndex = lineIndex *outputAscanLength + index;

		//Note log scaling: log(sqrt(x*x+y*y)) == 0.5*log(x*x+y*y) --> the calculation in the code below is 20*log(magnitude) and not 10*log...
		//Note fft normalization://(1/(2*outputAscanLength)) is the FFT normalization factor. In addition a multiplication by 2 is performed since the acquired OCT raw signal is a real valued signal, so (1/(2*outputAscanLength)) becomes 1/outputAscanLength. (Why multiply by 2: FFT of a real-valued signal is a complex-valued signal with a symmetric spectrum, where the positive and negative frequency components are identical in magnitude. And since the signal is truncated (negative or positive frequency components are removed), doubling of the remaining components is performed here)
		//amplitude:
		float realComponent = input[inputArrayIndex].x;
		float imaginaryComponent = input[inputArrayIndex].y;
		output[index] = coeff*((((10.0f*log10f(((realComponent*realComponent) + (imaginaryComponent*imaginaryComponent))/(outputAscanLength))) - min) / (max - min)) + addend);
	}
}

//Removes half of each processed A-scan (the mirror artefacts), calculates magnitude of remaining A-scan and copies it into an output array. This output array can be used to display the processed OCT data.
__global__ void postProcessTruncateLin(float *output, const cufftComplex *input, const int outputAscanLength, const int samples, const float max, const float min, const float addend, const float coeff) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < samples / 2) {
		int lineIndex = index / outputAscanLength;
		int inputArrayIndex = lineIndex * outputAscanLength + index;

		//amplitude:
		float realComponent = input[inputArrayIndex].x;
		float imaginaryComponent = input[inputArrayIndex].y;
		output[index] = coeff * ((((sqrt((realComponent*realComponent) + (imaginaryComponent*imaginaryComponent))/(outputAscanLength)) - min) / (max - min)) + addend);
	}
}

__global__ void getPostProcessBackground(float* output, float* input, const int samplesPerAscan, const int ascansPerBuffer) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < samplesPerAscan) {
		float sum = 0;
		for (int i = 0; i < ascansPerBuffer; i++){
			sum += input[index+i*samplesPerAscan];
		}
		output[index] = sum/ascansPerBuffer;
	}
}

__global__ void postProcessBackgroundRemoval(float* data, float* background, const float backgroundWeight, const float backgroundOffset, const int samplesPerAscan, const int samplesPerBuffer) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < samplesPerBuffer) {
		data[index] = __saturatef(data[index] - (backgroundWeight * background[index%samplesPerAscan] + backgroundOffset));
	}
}

__global__ void cuda_bscanFlip_slow(float *output, float *input, int samplesPerAscan, int ascansPerBscan, int samplesInVolume) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < samplesInVolume) {
		int samplesPerBscan = ascansPerBscan * samplesPerAscan;
		int bscanIndex = index / samplesPerBscan;
		int sampleIndex = index % samplesPerBscan;
		int ascanIndex = sampleIndex / samplesPerAscan;
		int mirrorIndex = bscanIndex*samplesPerBscan+((ascansPerBscan - 1) - ascanIndex)*samplesPerAscan + (sampleIndex%samplesPerAscan);

		if (bscanIndex % 2 == 0 && ascanIndex >= ascansPerBscan/2) {
			float tmp = input[mirrorIndex];
			output[mirrorIndex] = input[index];
			output[index] = tmp;
		}
	}
}

//todo: optimize! cuda_bscanFlip should be possible with just index < samplesPerBuffer/4
__global__ void cuda_bscanFlip(float *output, float *input, const int samplesPerAscan, const int ascansPerBscan, const int samplesPerBscan, const int halfSamplesInVolume) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < halfSamplesInVolume) {
		int bscanIndex = (index / samplesPerBscan)*2; //multiplication by 2 gets us just even bscanIndex-values (0, 2, 4, 6, ...) This is necessary because we just want to flip every second Bscan.
		index = bscanIndex*samplesPerBscan + index%samplesPerBscan; //recalculation of index is necessary here to skip every second Bscan
		int sampleIndex = index % samplesPerBscan;
		int ascanIndex = sampleIndex / samplesPerAscan;
		int mirrorIndex = bscanIndex*samplesPerBscan + ((ascansPerBscan - 1) - ascanIndex)*samplesPerAscan + (sampleIndex%samplesPerAscan);

		if (ascanIndex >= ascansPerBscan / 2) {
			float tmp = input[mirrorIndex];
			output[mirrorIndex] = input[index];
			output[index] = tmp;
		}
	}
}

//todo: avoid duplicate code: updateDisplayedBscanFrame and updateDisplayedEnFaceViewFrame only differ in the way how (or in what order) processedVolume[], displayBuffer[] is accessed and what the maximum number of available frames is ("bscansPerVolume" for updateDisplayedBscanFrame and "frameWidth" for updateDisplayedEnFaceViewFrame), the rest of the code is identical --> there should be a way to avoid duplicate code
__global__ void updateDisplayedBscanFrame(float *displayBuffer, const float* processedVolume, const unsigned int bscansPerVolume, const unsigned int samplesInSingleFrame, const unsigned int frameNr, const unsigned int displayFunctionFrames, const int displayFunction) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < samplesInSingleFrame) {
		//todo: optimize averaging and MIP for very large number of displayFunctionFrames! Maybe Parallel Reduction could improve performance of averaging operation (Also maybe use enum instead of int for displayFunction to improve readability of the code)
		if(displayFunctionFrames > 1){
			switch(displayFunction){
			case 0: {//Averaging
				int frameCount = 0;
				float sum = 0;
				for (int j = 0; j < displayFunctionFrames; j++){
					int frameForAveraging = frameNr+j;
					if(frameForAveraging < bscansPerVolume){
						sum += processedVolume[frameForAveraging*samplesInSingleFrame + (samplesInSingleFrame-1) - i];
						frameCount++;
					}
				}
				displayBuffer[i] = sum/frameCount;
				break;
			}
			case 1: {//MIP
				float maxValue = 0;
				float currentValue = 0;
				if(displayFunctionFrames > 1){
					for (int j = 0; j < displayFunctionFrames; j++){
						int frameForMIP = frameNr+j;
						if(frameForMIP < bscansPerVolume){
							currentValue = processedVolume[frameForMIP*samplesInSingleFrame + (samplesInSingleFrame-1) - i];
							if(maxValue < currentValue){
								maxValue = currentValue;
							}
						}
					}
					displayBuffer[i] = maxValue;
				}
				break;
			}
			default:
				break;
			}
		}
		else {
			displayBuffer[i] = processedVolume[frameNr*samplesInSingleFrame + (samplesInSingleFrame-1) - i];
		}
	}
}

__global__ void updateDisplayedEnFaceViewFrame(float *displayBuffer, const float* processedVolume, const unsigned int frameWidth, const unsigned int samplesInSingleFrame, const unsigned int frameNr, const unsigned int displayFunctionFrames, const int displayFunction) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < samplesInSingleFrame) {
		//todo: optimize averaging and MIP for very large number of displayFunctionFrames! Maybe Parallel Reduction could improve performance of averaging operation (Also maybe use enum instead of int for displayFunction to improve readability of the code)
		if(displayFunctionFrames > 1){
			switch(displayFunction){
			case 0: {//Averaging
				int frameCount = 0;
				float sum = 0;
				for (int j = 0; j < displayFunctionFrames; j++){
					int frameForAveraging = frameNr+j;
					if(frameForAveraging < frameWidth){
						sum += processedVolume[frameForAveraging+i*frameWidth];
						frameCount++;
					}
				}
				displayBuffer[(samplesInSingleFrame-1)-i] = sum/frameCount;
				break;
			}
			case 1: {//MIP
				float maxValue = 0;
				float currentValue = 0;
				if(displayFunctionFrames > 1){
					for (int j = 0; j < displayFunctionFrames; j++){
						int frameForMIP = frameNr+j;
						if(frameForMIP < frameWidth){
						currentValue = processedVolume[frameForMIP+i*frameWidth];
							if(maxValue < currentValue){
								maxValue = currentValue;
							}
						}
					}
					displayBuffer[(samplesInSingleFrame-1)-i] = maxValue;
				}
				break;
			}
			default:
				break;
			}
		}
		else {
			displayBuffer[(samplesInSingleFrame-1)-i] = processedVolume[frameNr+i*frameWidth];
		}
	}
}

#if __CUDACC_VER_MAJOR__ >=12
__global__ void updateDisplayedVolume(cudaSurfaceObject_t surfaceWrite, const float* processedBuffer, const unsigned int samplesInBuffer, const unsigned int currBufferNr, const unsigned int bscansPerBuffer, dim3 textureDim) {
#else
__global__ void updateDisplayedVolume(const float* processedBuffer, const unsigned int samplesInBuffer, const unsigned int currBufferNr, const unsigned int bscansPerBuffer, dim3 textureDim) {
#endif
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < samplesInBuffer; i += blockDim.x * gridDim.x){
		int width = textureDim.x; //Ascans per Bscan
		//int height = textureDim.y; //Bscans per Volume
		int depth = textureDim.z; //samples in Ascan
		int samplesPerFrame = width*depth;
		int y = (i/depth)%width;
		//int z = i%depth;
		int z = (depth-1)-(i%depth); //flip back to front
		int x = i/samplesPerFrame + (currBufferNr)*bscansPerBuffer;

		unsigned char voxel = (unsigned char)(processedBuffer[i] * (255.0));
		surf3Dwrite(voxel, surfaceWrite, y * sizeof(unsigned char), x, z);
	}
}

__global__ void floatToOutput(void *output, const float *input, const int outputBitdepth, const int samplesInProcessedVolume) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(outputBitdepth <= 8){
		unsigned char* out = (unsigned char*)output;
		out[index] = (unsigned char)(__saturatef(input[index]) * (255.0)); //float input with values between 0.0 and 1.0 is converted to 8 bit (0 to 255) output
	}else if(outputBitdepth > 8 && outputBitdepth <= 10){
		unsigned short* out = (unsigned short*)output;
		out[index] = (unsigned short)(__saturatef(input[index]) * (1023.0)); //10 bit
	}else if(outputBitdepth > 10 && outputBitdepth <= 12){
		unsigned short* out = (unsigned short*)output;
		out[index] = (unsigned short)(__saturatef(input[index]) * (4095.0)); //12 bit
	}else if(outputBitdepth > 12 && outputBitdepth <= 16){
		unsigned short* out = (unsigned short*)output;
		out[index] = (unsigned short)(__saturatef(input[index]) * (65535.0)); //16 bit
	}else if(outputBitdepth > 16 && outputBitdepth <= 24){
		unsigned int* out = (unsigned int*)output;
		out[index] = (unsigned int)(__saturatef(input[index]) * (16777215.0f)); //24 bit
	}else{
		unsigned int* out = (unsigned int*)output;
		out[index] = (unsigned int)(__saturatef(input[index]) * (4294967295.0f)); //32 bit
	}
}

extern "C" void cuda_updateResampleCurve(float* h_resampleCurve, int size, cudaStream_t stream) {
	if (d_resampleCurve != NULL && h_resampleCurve != NULL && size > 0 && size <= (int)signalLength){
		checkCudaErrors(cudaMemcpyAsync(d_resampleCurve, h_resampleCurve, size * sizeof(float), cudaMemcpyHostToDevice, stream));
	}
}

bool allocateAndInitializeBuffer(void** d_buffer, size_t bufferSize) {
	size_t freeMem = 0;
	size_t totalMem = 0;
	cudaError_t status;

	//check available memory
	status = cudaMemGetInfo(&freeMem, &totalMem);
	if(status != cudaSuccess) {
		printf("Cuda: Error retrieving memory info: %s\n", cudaGetErrorString(status));
		return false;
	}

	if(freeMem >= bufferSize) {
		//allocate memory if enough memory is available
		status = cudaMalloc(d_buffer, bufferSize);
		if(status != cudaSuccess) {
			printf("cudaMalloc failed: %s\n", cudaGetErrorString(status));
			if(*d_buffer != NULL) {
				freeCudaMem(d_buffer); //cleanup on failure
				*d_buffer = NULL;
			}
			return false;
		}

		//initialize allocated memory
		status = cudaMemset(*d_buffer, 0, bufferSize);
		if(status != cudaSuccess) {
			printf("cudaMemsetAsync failed: %s\n", cudaGetErrorString(status));
			if(*d_buffer != NULL) {
				freeCudaMem(d_buffer); //cleanup on failure
				*d_buffer = NULL;
			}
			return false;
		}

		return true;
	} else {
		printf("Cuda: Not enough memory available.\n");
		return false;
	}
}

bool createStreamsAndEvents() {
	cudaError_t err;

	err = cudaStreamCreate(&userRequestStream);
	if (err != cudaSuccess) {
		printf("Cuda: Failed to create stream: %s\n", cudaGetErrorString(err));
		return false;
	}

	for (int i = 0; i < nStreams; i++) {
		err = cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
		if (err != cudaSuccess) {
			printf("cuda: Failed to create stream %d: %s\n", i, cudaGetErrorString(err));

			//cleanup already created streams before exiting
			for (int j = 0; j < i; j++) {
				cudaStreamDestroy(stream[j]);
			}
			cudaStreamDestroy(userRequestStream);
			return false;
		}
	}

	err = cudaEventCreateWithFlags(&syncEvent, cudaEventBlockingSync);
	if (err != cudaSuccess) {
		printf("Failed to create synchronization event: %s\n", cudaGetErrorString(err));

		//cleanup all streams since creating the event failed
		for (int i = 0; i < nStreams; i++) {
			cudaStreamDestroy(stream[i]);
		}
		cudaStreamDestroy(userRequestStream);
		return false;
	}

	return true;
}

int getMaxThreadsPerBlock(){
	//get current active device
	int currentDevice;
	cudaGetDevice(&currentDevice);

	//get device properties
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, currentDevice);

	return deviceProp.maxThreadsPerBlock;
}

extern "C" bool initializeCuda(void* h_buffer1, void* h_buffer2, OctAlgorithmParameters* parameters) {
	signalLength = static_cast<int>(parameters->samplesPerLine);
	ascansPerBscan = static_cast<int>(parameters->ascansPerBscan);
	bscansPerBuffer = static_cast<int>(parameters->bscansPerBuffer);
	buffersPerVolume = static_cast<int>(parameters->buffersPerVolume);
	samplesPerBuffer = signalLength*ascansPerBscan*bscansPerBuffer;
	samplesPerVolume = samplesPerBuffer * buffersPerVolume;
	host_buffer1 = h_buffer1;
	host_buffer2 = h_buffer2;
	params = parameters;
	bytesPerSample = ceil((double)(parameters->bitDepth) / 8.0);

	createStreamsAndEvents();

	bool success =
	allocateAndInitializeBuffer((void**)&d_resampleCurve, sizeof(float)*signalLength)
	&& allocateAndInitializeBuffer((void**)&d_dispersionCurve, sizeof(float)*signalLength)
	&& allocateAndInitializeBuffer((void**)&d_sinusoidalResampleCurve, sizeof(float)*ascansPerBscan)
	&& allocateAndInitializeBuffer((void**)&d_windowCurve, sizeof(float)*signalLength);

	if(!success){
		releaseBuffers();
		destroyStreamsAndEvents();
		return false;
	}

	fillSinusoidalScanCorrectionCurve<<<ascansPerBscan, 1, 0, stream[0]>>> (d_sinusoidalResampleCurve, ascansPerBscan);
	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	//allocate device memory for the raw signal. On the Jetson Nano (__aarch64__), this allocation can be skipped if the acquisition buffer is created with cudaHostAlloc with the flag cudaHostAllocMapped, this way it can be accessed by both CPU and GPU; no extra device buffer is necessary.
#if !defined(__aarch64__) || !defined(ENABLE_CUDA_ZERO_COPY)
	for (int i = 0; i < nBuffers; i++) {
		success = allocateAndInitializeBuffer((void**)&d_inputBuffer[i], bytesPerSample * samplesPerBuffer);
		if (!success) break;
	}

	if (success) {
		success = allocateAndInitializeBuffer((void**)&d_outputBuffer, bytesPerSample * samplesPerBuffer / 2);
	}

	if(!success){
		releaseBuffers();
		destroyStreamsAndEvents();
		return false;
	}
#endif

	if (success) {
		success = allocateAndInitializeBuffer((void**)&d_inputLinearized, sizeof(cufftComplex) * samplesPerBuffer)
		&& allocateAndInitializeBuffer((void**)&d_phaseCartesian, sizeof(cufftComplex) * signalLength)
		&& allocateAndInitializeBuffer((void**)&d_processedBuffer, sizeof(float) * samplesPerVolume / 2)
		&& allocateAndInitializeBuffer((void**)&d_sinusoidalScanTmpBuffer, sizeof(float) * samplesPerBuffer / 2)
		&& allocateAndInitializeBuffer((void**)&d_fftBuffer, sizeof(cufftComplex) * samplesPerBuffer)
		&& allocateAndInitializeBuffer((void**)&d_meanALine, sizeof(cufftComplex) * signalLength)
		&& allocateAndInitializeBuffer((void**)&d_postProcBackgroundLine, sizeof(float) * signalLength / 2);;
	}

	if(!success){
		releaseBuffers();
		destroyStreamsAndEvents();
		return false;
	}

	//register existing host memory for use by cuda to accelerate cudaMemcpy.
	//this is not necessary for Jetson Nano since the acquisition buffer is created with
	//cudaHostAlloc and the cudaHostAllocMapped flag, which allows for zero-copy access.
#ifndef __aarch64__
	checkCudaErrors(cudaHostRegister(host_buffer1, samplesPerBuffer * bytesPerSample, cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(host_buffer2, samplesPerBuffer * bytesPerSample, cudaHostRegisterPortable));
#endif

	//create fft plan and set stream
	checkCudaErrors(cufftPlan1d(&d_plan, signalLength, CUFFT_C2C, ascansPerBscan*bscansPerBuffer));
	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	cudaInitialized = true;
	bufferNumber = 0;
	bufferNumberInVolume = params->buffersPerVolume-1;
	streamingBufferNumber = 0;
	floatStreamingBufferNumber = 0;
	processedBuffers = 0;
	streamedBuffers = 0;
	fixedPatternNoiseDetermined = false;

	//todo: find a way to automatically determine optimal blockSize and optimal gridSize
	blockSize = 128;
	gridSize = samplesPerBuffer / blockSize;
	threadsPerBlockLimit = getMaxThreadsPerBlock();

	currStream = 0;
	currBuffer = 0;

	return success;
}

extern "C" void releaseBuffers() {
#if !defined(__aarch64__) || !defined(ENABLE_CUDA_ZERO_COPY)
		for (int i = 0; i < nBuffers; i++){
			freeCudaMem((void**)&d_inputBuffer[i]);
		}
		freeCudaMem((void**)&d_outputBuffer);
#endif
		freeCudaMem((void**)&d_windowCurve);
		freeCudaMem((void**)&d_fftBuffer);
		freeCudaMem((void**)&d_meanALine);
		freeCudaMem((void**)&d_postProcBackgroundLine);
		freeCudaMem((void**)&d_processedBuffer);
		freeCudaMem((void**)&d_sinusoidalScanTmpBuffer);
		freeCudaMem((void**)&d_inputLinearized);
		freeCudaMem((void**)&d_phaseCartesian);
		freeCudaMem((void**)&d_resampleCurve);
		freeCudaMem((void**)&d_dispersionCurve);
		freeCudaMem((void**)&d_sinusoidalResampleCurve);
}

extern "C" void destroyStreamsAndEvents() {
		checkCudaErrors(cudaStreamDestroy(userRequestStream));

		for (int i = 0; i < nStreams; i++) {
			checkCudaErrors(cudaStreamDestroy(stream[i]));
		}

		checkCudaErrors(cudaEventDestroy(syncEvent));
}

extern "C" void cleanupCuda() {
	if (cudaInitialized) {
		releaseBuffers();
		cufftDestroy(d_plan);
		destroyStreamsAndEvents();

#ifndef __aarch64__
		if (host_buffer1 != NULL) {
			cudaHostUnregister(host_buffer1);
		}
		if (host_buffer2 != NULL) {
			cudaHostUnregister(host_buffer2);
		}
#endif

		cudaInitialized = false;
		fixedPatternNoiseDetermined = false;
	}
}

extern "C" void freeCudaMem(void** data) {
	if (*data != NULL) {
		checkCudaErrors(cudaFree(*data));
		*data = NULL;
	} else {
		printf("Cuda: Failed to free memory.\n");
	}
}

extern "C" void changeDisplayedBscanFrame(unsigned int frameNr, unsigned int displayFunctionFrames, int displayFunction) {
	void* d_bscanDisplayBuffer = NULL;
	if (cuBufHandleBscan != NULL) {
		d_bscanDisplayBuffer = cuda_map(cuBufHandleBscan, userRequestStream);
	}
	//update 2D b-scan display
	int width = signalLength;
	int height = ascansPerBscan;
	unsigned int depth = static_cast<unsigned int>(bscansPerBuffer*buffersPerVolume);
	int samplesPerFrame = width * height;
	if (d_bscanDisplayBuffer != NULL) {
		frameNr = frameNr < depth ? frameNr : 0;
		updateDisplayedBscanFrame<<<gridSize/2, blockSize, 0, userRequestStream>>>((float*)d_bscanDisplayBuffer, d_processedBuffer, depth, samplesPerFrame / 2, frameNr, displayFunctionFrames, displayFunction);
	}
	if (cuBufHandleBscan != NULL) {
		cuda_unmap(cuBufHandleBscan, userRequestStream);
	}
}

//todo: simplify/refactor changeDisplayedEnfaceFrame and changedisplaydBscanFrame. avoid duplicate code
extern "C" void changeDisplayedEnFaceFrame(unsigned int frameNr, unsigned int displayFunctionFrames, int displayFunction) {
	void* d_enFaceViewDisplayBuffer = NULL;
	if (cuBufHandleEnFaceView != NULL) {
		d_enFaceViewDisplayBuffer = cuda_map(cuBufHandleEnFaceView, userRequestStream);
	}
	//update 2D en face view display
	int width = bscansPerBuffer*buffersPerVolume;
	int height = ascansPerBscan;
	int samplesPerFrame = width * height;
	int gridSizeDisplay = width;
	int blockSizeDisplay = height;
	if(height > threadsPerBlockLimit){
		blockSizeDisplay = threadsPerBlockLimit;
		gridSizeDisplay = (samplesPerFrame + blockSizeDisplay - 1)/blockSizeDisplay;
	}
	if (d_enFaceViewDisplayBuffer != NULL) {
		frameNr = frameNr < static_cast<unsigned int>(signalLength/2) ? frameNr : 0;
		updateDisplayedEnFaceViewFrame<<<gridSizeDisplay, blockSizeDisplay, 0, userRequestStream>>>((float*)d_enFaceViewDisplayBuffer, d_processedBuffer, signalLength/2, samplesPerFrame, frameNr, displayFunctionFrames, displayFunction);
	}
	if (cuBufHandleEnFaceView != NULL) {
		cuda_unmap(cuBufHandleEnFaceView, userRequestStream);
	}
}

extern "C" inline void updateBscanDisplayBuffer(unsigned int frameNr, unsigned int displayFunctionFrames, int displayFunction, cudaStream_t stream) {
	void* d_bscanDisplayBuffer = NULL;
	if (cuBufHandleBscan != NULL) {
		d_bscanDisplayBuffer = cuda_map(cuBufHandleBscan, stream);
	}
	//update 2D b-scan display
	int width = signalLength;
	int height = ascansPerBscan;
	unsigned int depth = static_cast<unsigned int>(bscansPerBuffer * buffersPerVolume);
	int samplesPerFrame = width * height;
	if (d_bscanDisplayBuffer != NULL) {
		frameNr = frameNr < depth ? frameNr : 0;
		updateDisplayedBscanFrame<<<gridSize/2, blockSize, 0, stream>>>((float*)d_bscanDisplayBuffer, d_processedBuffer, depth, samplesPerFrame / 2, frameNr, displayFunctionFrames, displayFunction);
	}
	if (cuBufHandleBscan != NULL) {
		cuda_unmap(cuBufHandleBscan, stream);
	}
}

extern "C" inline void updateEnFaceDisplayBuffer(unsigned int frameNr, unsigned int displayFunctionFrames, int displayFunction, cudaStream_t stream) {
	void* d_enFaceViewDisplayBuffer = NULL;
	if (cuBufHandleEnFaceView != NULL) {
		d_enFaceViewDisplayBuffer = cuda_map(cuBufHandleEnFaceView, stream);
	}
	//update 2D en face view display
	int width = bscansPerBuffer * buffersPerVolume;
	int height = ascansPerBscan;
	int samplesPerFrame = width * height;
	int gridSizeDisplay = width;
	int blockSizeDisplay = height;
	if(height > threadsPerBlockLimit){
		blockSizeDisplay = threadsPerBlockLimit;
		gridSizeDisplay = (samplesPerFrame + blockSizeDisplay - 1)/blockSizeDisplay;
	}
	if (d_enFaceViewDisplayBuffer != NULL) {
		frameNr = frameNr < static_cast<unsigned int>(signalLength/2) ? frameNr : 0;
		updateDisplayedEnFaceViewFrame<<<gridSizeDisplay, blockSizeDisplay, 0, stream>>>((float*)d_enFaceViewDisplayBuffer, d_processedBuffer, signalLength/2, samplesPerFrame, frameNr, displayFunctionFrames, displayFunction);
	}
	if (cuBufHandleEnFaceView != NULL) {
		cuda_unmap(cuBufHandleEnFaceView, stream);
	}
}

extern "C" inline void updateVolumeDisplayBuffer(const float* d_currBuffer, const unsigned int currentBufferNr, const unsigned int bscansPerBuffer, cudaStream_t stream) {
	//map graphics resource for access by cuda
	cudaArray* d_volumeViewDisplayBuffer = NULL;
	if (cuBufHandleVolumeView != NULL) {
		d_volumeViewDisplayBuffer = cuda_map3dTexture(cuBufHandleVolumeView, stream);
	}
	//calculate dimensions of processed volume
	unsigned int width = bscansPerBuffer * buffersPerVolume;
	unsigned int height = ascansPerBscan;
	unsigned int depth = signalLength/2;
	if (d_volumeViewDisplayBuffer != NULL) {
#if __CUDACC_VER_MAJOR__ >=12
	        cudaResourceDesc surfRes;
	        memset(&surfRes, 0, sizeof(surfRes));
	        surfRes.resType = cudaResourceTypeArray;
	        surfRes.res.array.array = d_volumeViewDisplayBuffer;
	        cudaSurfaceObject_t surfaceWrite;

	        cudaError_t error_id = cudaCreateSurfaceObject(&surfaceWrite, &surfRes);
	        if (error_id != cudaSuccess) {
	            printf("Cuda: Failed to create surface object: %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
	            return;
	        }

	        dim3 texture_dim(height, width, depth); //todo: use consistent naming of width, height, depth, x, y, z, ...
	        updateDisplayedVolume<< <gridSize/2, blockSize, 0, stream>>>(surfaceWrite, d_currBuffer, samplesPerBuffer/2, currentBufferNr, bscansPerBuffer, texture_dim);
	        cudaDestroySurfaceObject(surfaceWrite);
#else
		//bind voxel array to a writable cuda surface
		cudaError_t error_id = cudaBindSurfaceToArray(surfaceWrite, d_volumeViewDisplayBuffer);
		if (error_id != cudaSuccess) {
			printf("Cuda: Failed to bind surface to cuda array:  %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
			return;
		}

		//write to cuda surface
		dim3 texture_dim(height, width, depth); //todo: use consistent naming of width, height, depth, x, y, z, ...
		updateDisplayedVolume<< <gridSize/2, blockSize, 0, stream>>>(d_currBuffer, samplesPerBuffer/2, currentBufferNr, bscansPerBuffer, texture_dim);
#endif
    }

	//unmap the graphics resource
	if (cuBufHandleVolumeView != NULL) {
		cuda_unmap(cuBufHandleVolumeView, stream);
	}
}

inline void streamProcessedData(float* d_currProcessedBuffer, cudaStream_t stream) {
	if (streamedBuffers % (params->streamingBuffersToSkip + 1) == 0) {
		streamedBuffers = 0; //set to zero to avoid overflow
		streamingBufferNumber = (streamingBufferNumber + 1) % 2;
		void* hostDestBuffer = streamingBufferNumber == 0 ? host_streamingBuffer1 : host_streamingBuffer2;
#if defined(__aarch64__) && defined(ENABLE_CUDA_ZERO_COPY)
		checkCudaErrors(cudaHostGetDevicePointer((void**)&d_outputBuffer, (void*)hostDestBuffer, 0));
#endif
		floatToOutput<<<gridSize / 2, blockSize, 0, stream>>> (d_outputBuffer, d_currProcessedBuffer, params->bitDepth, samplesPerBuffer / 2);
#if !defined(__aarch64__) || !defined(ENABLE_CUDA_ZERO_COPY)
		checkCudaErrors(cudaMemcpyAsync(hostDestBuffer, (void*)d_outputBuffer, (samplesPerBuffer / 2) * bytesPerSample, cudaMemcpyDeviceToHost, stream));
#endif
		checkCudaErrors(cudaLaunchHostFunc(stream, Gpu2HostNotifier::dh2StreamingCallback, hostDestBuffer));
	}
	streamedBuffers++;
}

inline void streamProcessedFloatData(float* d_currProcessedBuffer, cudaStream_t stream) {
	if (!floatStreamingBuffersRegistered) return;

	floatStreamingBufferNumber = (floatStreamingBufferNumber + 1) % 2;
	void* hostDestBuffer = floatStreamingBufferNumber == 0 ? host_floatStreamingBuffer1 : host_floatStreamingBuffer2;

	#if !defined(__aarch64__) || !defined(ENABLE_CUDA_ZERO_COPY)
		size_t bufferSizeInBytes = (samplesPerBuffer / 2) * sizeof(float);
		checkCudaErrors(cudaMemcpyAsync(hostDestBuffer, d_currProcessedBuffer, bufferSizeInBytes, cudaMemcpyDeviceToHost, stream));
	#endif

	checkCudaErrors(cudaLaunchHostFunc(stream, Gpu2HostNotifier::dh2FloatStreamingCallback, hostDestBuffer));
}


extern "C" void octCudaPipeline(void* h_inputSignal) {
	//check if cuda buffers are initialized
	if (!cudaInitialized) {
		printf("Cuda: Device buffers are not initialized!\n");
		return;
	}

	currStream = (currStream+1)%nStreams;
	currBuffer = (currBuffer+1)%nBuffers;

	//copy raw oct signal from host
	if (h_inputSignal != NULL) {
#if defined(__aarch64__) && defined(ENABLE_CUDA_ZERO_COPY)
		checkCudaErrors(cudaHostGetDevicePointer((void**)&d_inputBuffer[currBuffer], (void*)h_inputSignal, 0));
#else
		checkCudaErrors(cudaMemcpyAsync(d_inputBuffer[currBuffer], h_inputSignal, samplesPerBuffer * bytesPerSample, cudaMemcpyHostToDevice, stream[currStream]));
#endif
	}

	//start processing: convert input array to cufft complex array
	if (params->bitshift) {
		inputToCufftComplex_and_bitshift<<<gridSize, blockSize, 0, stream[currStream]>>> (d_fftBuffer, d_inputBuffer[currBuffer], signalLength,  signalLength, params->bitDepth, samplesPerBuffer);
	}
	else {
		inputToCufftComplex<<<gridSize, blockSize, 0, stream[currStream]>>> (d_fftBuffer, d_inputBuffer[currBuffer], signalLength, signalLength, params->bitDepth, samplesPerBuffer);
	}

	//synchronization: block the host during cudaMemcpyAsync and inputToCufftComplex to prevent the data acquisition of the virtual OCT system from outpacing the processing, ensuring proper synchronization in the pipeline.
#if !defined(__aarch64__) || !defined(ENABLE_CUDA_ZERO_COPY)
	cudaEventRecord(syncEvent, stream[currStream]);
	cudaEventSynchronize(syncEvent);
#endif

	//rolling average background subtraction
	if (params->backgroundRemoval){
		int sharedMemSize = (blockSize + 2 * params->rollingAverageWindowSize) * sizeof(float);
		rollingAverageBackgroundRemoval<<<gridSize, blockSize, sharedMemSize, stream[currStream]>>>(d_inputLinearized, d_fftBuffer, params->rollingAverageWindowSize, signalLength, ascansPerBscan, signalLength*ascansPerBscan, samplesPerBuffer);
		cufftComplex* tmpSwapPointer = d_inputLinearized;
		d_inputLinearized = d_fftBuffer;
		d_fftBuffer = tmpSwapPointer;
	}

	//update k-linearization-, dispersion- and windowing-curves if necessary
	cufftComplex* d_fftBuffer2 = d_fftBuffer;
	if (params->resampling && params->resamplingUpdated) {
		cuda_updateResampleCurve(params->resampleCurve, params->resampleCurveLength, stream[currStream]);
		params->resamplingUpdated = false;
	}
	if (params->dispersionCompensation && params->dispersionUpdated) {
		cuda_updateDispersionCurve(params->dispersionCurve, signalLength, stream[currStream]);
		fillDispersivePhase<<<signalLength, 1, 0, stream[currStream]>>> (d_phaseCartesian, d_dispersionCurve, 1.0, signalLength, 1);
		params->dispersionUpdated = false;
	}
	if (params->windowing && params->windowUpdated) {
		cuda_updateWindowCurve(params->windowCurve, signalLength, stream[currStream]);
		params->windowUpdated = false;
	}

	//k-linearization and windowing
	if (d_inputLinearized != NULL && params->resampling && params->windowing && !params->dispersionCompensation) {
		if(params->resamplingInterpolation == OctAlgorithmParameters::INTERPOLATION::CUBIC) {
			klinearizationCubicAndWindowing<<<gridSize, blockSize, 0, stream[currStream]>>>(d_inputLinearized, d_fftBuffer, d_resampleCurve, d_windowCurve, signalLength, samplesPerBuffer);
		}
		else if(params->resamplingInterpolation == OctAlgorithmParameters::INTERPOLATION::LINEAR) {
			klinearizationAndWindowing<<<gridSize, blockSize, 0, stream[currStream]>>>(d_inputLinearized, d_fftBuffer, d_resampleCurve, d_windowCurve, signalLength, samplesPerBuffer);
		}
		else if(params->resamplingInterpolation == OctAlgorithmParameters::INTERPOLATION::LANCZOS) {
			klinearizationLanczosAndWindowing<<<gridSize, blockSize, 0, stream[currStream]>>>(d_inputLinearized, d_fftBuffer, d_resampleCurve, d_windowCurve, signalLength, samplesPerBuffer);
		}
		d_fftBuffer2 = d_inputLinearized;
	} else
		//k-linearization and windowing and dispersion compensation
	if (d_inputLinearized != NULL && params->resampling && params->windowing && params->dispersionCompensation) {
		if(params->resamplingInterpolation == OctAlgorithmParameters::INTERPOLATION::CUBIC){
			klinearizationCubicAndWindowingAndDispersionCompensation<<<gridSize, blockSize, 0, stream[currStream]>>>(d_inputLinearized, d_fftBuffer, d_resampleCurve, d_windowCurve, d_phaseCartesian, signalLength, samplesPerBuffer);
		}
		else if(params->resamplingInterpolation == OctAlgorithmParameters::INTERPOLATION::LINEAR) {
			klinearizationAndWindowingAndDispersionCompensation<<<gridSize, blockSize, 0, stream[currStream]>>>(d_inputLinearized, d_fftBuffer, d_resampleCurve, d_windowCurve, d_phaseCartesian, signalLength, samplesPerBuffer);
		}
		else if(params->resamplingInterpolation == OctAlgorithmParameters::INTERPOLATION::LANCZOS) {
			klinearizationLanczosAndWindowingAndDispersionCompensation<<<gridSize, blockSize, 0, stream[currStream]>>>(d_inputLinearized, d_fftBuffer, d_resampleCurve, d_windowCurve, d_phaseCartesian, signalLength, samplesPerBuffer);
		}
		d_fftBuffer2 = d_inputLinearized;
	} else
		//dispersion compensation and windowing
	if (!params->resampling && params->windowing && params->dispersionCompensation) {
		dispersionCompensationAndWindowing<<<gridSize, blockSize, 0, stream[currStream]>>>(d_fftBuffer2, d_fftBuffer2, d_phaseCartesian, d_windowCurve, signalLength, samplesPerBuffer);
	} else
		//just k-linearization
	if (d_inputLinearized != NULL && params->resampling && !params->windowing && !params->dispersionCompensation) {
		if(params->resamplingInterpolation == OctAlgorithmParameters::INTERPOLATION::CUBIC){
			klinearizationCubic<<<gridSize, blockSize, 0, stream[currStream]>>>(d_inputLinearized, d_fftBuffer, d_resampleCurve, signalLength, samplesPerBuffer);
		}
		else if(params->resamplingInterpolation == OctAlgorithmParameters::INTERPOLATION::LINEAR) {
			klinearization<<<gridSize, blockSize, 0, stream[currStream]>>>(d_inputLinearized, d_fftBuffer, d_resampleCurve, signalLength, samplesPerBuffer);
		}
		else if(params->resamplingInterpolation == OctAlgorithmParameters::INTERPOLATION::LANCZOS) {
			klinearizationLanczos<<<gridSize, blockSize, 0, stream[currStream]>>>(d_inputLinearized, d_fftBuffer, d_resampleCurve, signalLength, samplesPerBuffer);
		}
		d_fftBuffer2 = d_inputLinearized;
	} else
		//just windowing
	if (!params->resampling && params->windowing && !params->dispersionCompensation) {
		windowing<<<gridSize, blockSize, 0, stream[currStream]>>>(d_fftBuffer2, d_fftBuffer2, d_windowCurve, signalLength, samplesPerBuffer);
	} else
		//just dispersion compensation
	if (!params->resampling && !params->windowing && params->dispersionCompensation) {
		dispersionCompensation<<<gridSize, blockSize, 0, stream[currStream]>>> (d_fftBuffer2, d_fftBuffer2, d_phaseCartesian, signalLength, samplesPerBuffer);
	} else
		//k-linearization and dispersion compensation. nobody will use this in a serious manner, so an optimized "klinearizationAndDispersionCompensation" kernel is not necessary
	if (d_inputLinearized != NULL && params->resampling && !params->windowing && params->dispersionCompensation) {
		if(params->resamplingInterpolation == OctAlgorithmParameters::INTERPOLATION::CUBIC) {
			klinearizationCubic<<<gridSize, blockSize, 0, stream[currStream]>>>(d_inputLinearized, d_fftBuffer, d_resampleCurve, signalLength, samplesPerBuffer);
		}
		else if(params->resamplingInterpolation == OctAlgorithmParameters::INTERPOLATION::LINEAR) {
			klinearization<<<gridSize, blockSize, 0, stream[currStream]>>>(d_inputLinearized, d_fftBuffer, d_resampleCurve, signalLength, samplesPerBuffer);
		}
		else if(params->resamplingInterpolation == OctAlgorithmParameters::INTERPOLATION::LANCZOS) {
			klinearizationLanczos<<<gridSize, blockSize, 0, stream[currStream]>>>(d_inputLinearized, d_fftBuffer, d_resampleCurve, signalLength, samplesPerBuffer);
		}
		d_fftBuffer2 = d_inputLinearized;
		dispersionCompensation<<<gridSize, blockSize, 0, stream[currStream]>>> (d_fftBuffer2, d_fftBuffer2, d_phaseCartesian, signalLength, samplesPerBuffer);
	}

	//IFFT
	cufftSetStream(d_plan, stream[currStream]);
	checkCudaErrors(cufftExecC2C(d_plan, d_fftBuffer2, d_fftBuffer2, CUFFT_INVERSE));

	//Fixed-pattern noise removal
	if(params->fixedPatternNoiseRemoval){
		int width = signalLength;
		int height = params->bscansForNoiseDetermination*ascansPerBscan;//ascansPerBscan*bscansPerBuffer;
		if((!params->continuousFixedPatternNoiseDetermination && !fixedPatternNoiseDetermined) || params->continuousFixedPatternNoiseDetermination || params->redetermineFixedPatternNoise){
			getMinimumVarianceMean<<<gridSize, blockSize, 0, stream[currStream]>>>(d_meanALine, d_fftBuffer2, width, height, FIXED_PATTERN_NOISE_REMOVAL_SEGMENTS);
			fixedPatternNoiseDetermined = true;
			params->redetermineFixedPatternNoise = false;
		}
		meanALineSubtraction<<<gridSize/2, blockSize, 0, stream[currStream]>>>(d_fftBuffer2, d_meanALine, width/2, samplesPerBuffer/2); //here mean a-scan line subtraction of half volume is enough, because in the next step the volume gets truncated anyway
	}

	//get current buffer number in volume (a volume may consist of one or more buffers)
	if(buffersPerVolume > 1){
		bufferNumberInVolume = (bufferNumberInVolume+1)%buffersPerVolume;
	}

	//get current position in processed volume buffer
	float* d_currBuffer = &d_processedBuffer[(samplesPerBuffer/2)*bufferNumberInVolume];

	//postProcessTruncate contains: Mirror artefact removal, Log, Magnitude, Copy to output buffer.
	if (params->signalLogScaling) {
		postProcessTruncateLog<<<gridSize/2, blockSize, 0, stream[currStream]>>> (d_currBuffer, d_fftBuffer2, signalLength / 2, samplesPerBuffer, bufferNumberInVolume, params->signalGrayscaleMax, params->signalGrayscaleMin, params->signalAddend, params->signalMultiplicator);
	}
	else {
		postProcessTruncateLin<<<gridSize/2, blockSize, 0, stream[currStream]>>> (d_currBuffer, d_fftBuffer2, signalLength / 2, samplesPerBuffer, params->signalGrayscaleMax, params->signalGrayscaleMin, params->signalAddend, params->signalMultiplicator);
	}

	//flip every second bscan
	if (params->bscanFlip) {
		cuda_bscanFlip<<<gridSize/2, blockSize, 0, stream[currStream]>>> (d_currBuffer, d_currBuffer, signalLength / 2, ascansPerBscan, (signalLength*ascansPerBscan)/2, samplesPerBuffer/4);
	}

	//sinusoidal scan correction
	if(params->sinusoidalScanCorrection && d_sinusoidalScanTmpBuffer != NULL){
		checkCudaErrors(cudaMemcpyAsync(d_sinusoidalScanTmpBuffer, d_currBuffer, sizeof(float)*samplesPerBuffer/2, cudaMemcpyDeviceToDevice,stream[currStream]));
		sinusoidalScanCorrection<<<gridSize/2, blockSize, 0, stream[currStream]>>>(d_currBuffer, d_sinusoidalScanTmpBuffer, d_sinusoidalResampleCurve, signalLength/2, ascansPerBscan, bscansPerBuffer, samplesPerBuffer/2);
	}

	//post process background removal
	if(params->postProcessBackgroundRemoval){
		if(params->postProcessBackgroundRecordingRequested){
			getPostProcessBackground<<<gridSize/2, blockSize, 0, stream[currStream]>>>(d_postProcBackgroundLine, d_currBuffer, signalLength/2, ascansPerBscan );
			cuda_copyPostProcessBackgroundToHost(params->postProcessBackground, signalLength/2, stream[currStream]);
			params->postProcessBackgroundRecordingRequested = false;
		}
		if(params->postProcessBackgroundUpdated){
			cuda_updatePostProcessBackground(params->postProcessBackground, signalLength/2, stream[currStream]);
			params->postProcessBackgroundUpdated = false;
		}
		postProcessBackgroundRemoval<<<gridSize/2, blockSize, 0, stream[currStream]>>>(d_currBuffer, d_postProcBackgroundLine, params->postProcessBackgroundWeight, params->postProcessBackgroundOffset, signalLength/2, samplesPerBuffer/2);
	}

	//update display buffers
	if(params->bscanViewEnabled){
		updateBscanDisplayBuffer(params->frameNr, params->functionFramesBscan, params->displayFunctionBscan, stream[currStream]);
		//checkCudaErrors(cudaLaunchHostFunc(stream[currStream], Gpu2HostNotifier::bscanDisblayBufferReadySignalCallback, 0));
	}
	if(params->enFaceViewEnabled){
		updateEnFaceDisplayBuffer(params->frameNrEnFaceView, params->functionFramesEnFaceView, params->displayFunctionEnFaceView, stream[currStream]);
		//checkCudaErrors(cudaLaunchHostFunc(stream[currStream], Gpu2HostNotifier::enfaceDisplayBufferReadySignalCallback, 0));
	}
	if(params->volumeViewEnabled){
		updateVolumeDisplayBuffer(d_currBuffer, bufferNumberInVolume, bscansPerBuffer, stream[currStream]);
		//checkCudaErrors(cudaLaunchHostFunc(stream[currStream], Gpu2HostNotifier::volumeDisblayBufferReadySignalCallback, 0));
	}

#if defined(__aarch64__) && defined(ENABLE_CUDA_ZERO_COPY)
	cudaEventRecord(syncEvent, stream[currStream]);
	cudaEventSynchronize(syncEvent);
#endif

	//check errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Cuda error: %s\n", cudaGetErrorString(err));
	}

	//save processed data as 32-bit float if requested
	if (params->recParams.saveAs32bitFloat) {
		streamProcessedFloatData(d_currBuffer, stream[currStream]);
	}

	//Copy/Stream processed data to host continuously
	if (params->streamToHost && !params->streamingParamsChanged) {
		params->currentBufferNr = bufferNumberInVolume;
		streamProcessedData(d_currBuffer, stream[currStream]);
	}
}

extern "C" bool cuda_registerGlBufferBscan(GLuint buf) {
	//check if a resource is already registered, and if so, unregister it first
	if (cuBufHandleBscan != NULL) {
		cudaError_t unregisterResult = cudaGraphicsUnregisterResource(cuBufHandleBscan);
		if (unregisterResult != cudaSuccess) {
			printf("Cuda: Failed to unregister existing resource. Error: %s\n", cudaGetErrorString(unregisterResult));
		}
		cuBufHandleBscan = NULL; //set handle to NULL to ensure it no longer points to a freed resource.
	}
	//attempt to register the new buffer
	cudaError_t registerResult = cudaGraphicsGLRegisterBuffer(&cuBufHandleBscan, buf, cudaGraphicsRegisterFlagsWriteDiscard);
	if (registerResult != cudaSuccess) {
		printf("Cuda: Failed to register buffer %u. Error: %s\n", buf, cudaGetErrorString(registerResult));
		return false;
	}
	return true;
}


extern "C" bool cuda_registerGlBufferEnFaceView(GLuint buf) {
	//check if a resource is already registered, and if so, unregister it first
	if (cuBufHandleEnFaceView != NULL) {
		cudaError_t unregisterResult = cudaGraphicsUnregisterResource(cuBufHandleEnFaceView);
		if (unregisterResult != cudaSuccess) {
			printf("Cuda: Failed to unregister existing resource. Error: %s\n", cudaGetErrorString(unregisterResult));
		}
		cuBufHandleEnFaceView = NULL; //set handle to NULL to ensure it no longer points to a freed resource.
	}
	//attempt to register the new buffer
	if (cudaGraphicsGLRegisterBuffer(&cuBufHandleEnFaceView, buf, cudaGraphicsRegisterFlagsWriteDiscard) != cudaSuccess) {
		printf("Cuda: Failed to register buffer %u\n", buf);
		return false;
	}
	return true;
}
extern "C" bool cuda_registerGlBufferVolumeView(GLuint buf) {
	//check if a resource is already registered, and if so, unregister it first
	if (cuBufHandleVolumeView != NULL) {
		cudaError_t unregisterResult = cudaGraphicsUnregisterResource(cuBufHandleVolumeView);
		if (unregisterResult != cudaSuccess) {
			printf("Cuda: Failed to unregister existing resource. Error: %s\n", cudaGetErrorString(unregisterResult));
		}
		cuBufHandleVolumeView = NULL; //set handle to NULL to ensure it no longer points to a freed resource.
	}
	//attempt to register the new buffer
	cudaError_t err = cudaGraphicsGLRegisterImage(&cuBufHandleVolumeView, buf, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	if (err != cudaSuccess) {
		printf("Cuda: Failed to register buffer %u\n", buf);
		return false;
	}
	return true;
}

void* cuda_map(cudaGraphicsResource* res, cudaStream_t stream) {
	void *devPtr = 0;
	size_t size = 0;
	cudaError_t error_id = cudaGraphicsMapResources(1, &res, stream);
	if (error_id != cudaSuccess) {
		printf("Cuda: Failed to map resource:  %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		return 0;
	}
	error_id = cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, res);
	if (error_id != cudaSuccess) {
		printf("Cuda: Failed to get device pointer");
		return 0;
	}
	return devPtr;
}

cudaArray* cuda_map3dTexture(cudaGraphicsResource* res, cudaStream_t stream) {
	cudaArray* d_ArrayPtr = 0;
	cudaError_t error_id = cudaGraphicsMapResources(1, &res, stream);
	if (error_id != cudaSuccess) {
		printf("Cuda: Failed to map 3D texture resource:  %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		return 0;
	}
	error_id = cudaGraphicsSubResourceGetMappedArray(&d_ArrayPtr, res, 0, 0);
	if (error_id != cudaSuccess) {
		printf("Cuda: Failed to get device array pointer");
		return 0;
	}
	return d_ArrayPtr;
}

void cuda_unmap(cudaGraphicsResource* res, cudaStream_t stream) {
	if (cudaGraphicsUnmapResources(1, &res, stream) != cudaSuccess) {
		printf("Cuda: Failed to unmap resource");
	}
}

#endif
