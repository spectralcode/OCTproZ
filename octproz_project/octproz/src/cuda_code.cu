/**
**  This file is part of OCTproZ.
**  OCTproZ is an open source software for processig of optical
**  coherence tomography (OCT) raw data.
**  Copyright (C) 2019-2020 Miroslav Zabic
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
**			iqo.uni-hannover.de
****
**/

#ifndef CUDA_CODE_CU
#define CUDA_CODE_CU

#include "kernels.h"
#include "math.h"

int blockSize;
int gridSize;

surface<void, cudaSurfaceType3D> surfaceWrite;

cudaStream_t copyStream;
cudaStream_t copyStreamD2H;
cudaStream_t streamingStreamD2H;
cudaStream_t streamingStreamD2H2;
cudaStream_t processStream; //?
cudaStream_t displayStream; //?
cudaStream_t userRequestStream;

cudaEvent_t processEvent;

cudaGraphicsResource* cuBufHandleBscan = NULL;
cudaGraphicsResource* cuBufHandleBscan2 = NULL;
cudaGraphicsResource* cuBufHandleEnFaceView = NULL;
cudaGraphicsResource* cuBufHandleRetardance = NULL;
cudaGraphicsResource* cuBufHandleVolumeView = NULL;

void* d_inputBuffer1;
void* d_inputBuffer2;
void* d_outputBuffer;
void* d_outputBuffer2;

void* host_buffer1 = NULL;
void* host_buffer2 = NULL;
void* host_RecordBuffer = NULL;
void* host_streamingBuffer1;
void* host_streamingBuffer2;

cufftComplex* d_inputLinearized;
float* d_windowCurve= NULL;
float* d_resampleCurve = NULL;
float* d_dispersionCurve = NULL;
float* d_sinusoidalResampleCurve = NULL;
cufftComplex* d_phaseCartesian = NULL;
unsigned int bufferNumber = 0;
unsigned int bufferNumberInVolume = 0;
unsigned int streamingBufferNumber = 0;


cufftComplex* d_fftBuffer = NULL;
cufftHandle d_plan;
cufftComplex* d_meanALine = NULL;

bool cudaInitialized = false;
bool saveToDisk = false;

size_t signalLength = 0;
size_t ascansPerBscan = 0;
size_t bscansPerBuffer = 0;
size_t samplesPerBuffer = 0;
size_t samplesPerVolume = 0;
size_t buffersPerVolume = 0;
size_t bscansPerComponent = 0;
size_t samplesPerComponent = 0;
size_t bytesPerSample = 0;

float* d_processedBuffer = NULL; //?
float* d_sinusoidalScanTmpBuffer = NULL;
OctAlgorithmParameters* params = NULL;

bool firstRun = true;
unsigned int processedBuffers; //?
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

__global__ void inputToCufftComplex_and_bitshift(cufftComplex* output, const void* input, const int width_out, const int width_in, const int inputBitdepth, const int samples, const int samplesPerComponent) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(inputBitdepth <= 8){
        unsigned char* in = (unsigned char*)input;
        output[index].x = __uint2float_rd(in[index] >> 4);
    }else if(inputBitdepth > 8 && inputBitdepth <= 16){
        if((index/samplesPerComponent) % 2 == 0 ){
            unsigned short* in = (unsigned short*)input;
            output[index].x = __uint2float_rd(in[index - (index/samplesPerComponent)/2 * samplesPerComponent] >> 4);
        }else{
            unsigned short* in = (unsigned short*)input;
            output[index].x = __uint2float_rd(in[index - (index/samplesPerComponent - 1)/2 * samplesPerComponent + samples/2 - samplesPerComponent] >> 4);
        }
    }else{
        unsigned int* in = (unsigned int*)input;
        output[index].x = (in[index])/4294967296.0;
    }
    output[index].y = 0;
}

//device functions for endian byte swap //todo: check if big endian to little endian conversion may be needed and extend inputToCufftComplex kernel if necessary
__device__ inline uint32_t endianSwapUint32(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | ((val >> 16));
}
__device__ inline int32_t endianSwapInt32(int32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF );
    return (val << 16) | ((val >> 16) & 0xFFFF);
}
__device__ inline uint16_t endianSwapUint16(uint16_t val) {
    return (val << 8) | (val >> 8 );
}
__device__ inline int16_t endianSwapInt16(int16_t val) {
    return (val << 8) | ((val >> 8) & 0xFF);
}

//todo: use/evaluate cuda texture for interpolation in klinearization kernel. also try cubic spline interpolation and 3rd order hermite interpolation
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

//quadratic interpolation
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

//cubic interpolation
__global__ void klinearizationCubic(cufftComplex* out, cufftComplex *in, const float* resampleCurve, const int width, const int samples) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int j = index%width;
    int offset = index-j;

    float x = resampleCurve[j];
    int x0 = (int)x;
    int x1 = x0 + 1;
    int x2 = x0 + 2;
    int x3 = x0 + 3;

    float f_x0 = in[offset + x0].x;
    float f_x1 = in[offset + x1].x;
    float f_x2 = in[offset + x2].x;
    float f_x3 = in[offset + x2].x;
    float b0 = f_x0;
    float b1 = f_x1-f_x0;
    float b2 = ((f_x2-f_x1)-b1)/(x2-x0);
    float b3 = ((f_x3-f_x2)-b2)/(x3-x0);

    out[index].x = b0 + b1 * (x - x0) + b2*(x-x0)*(x-x1) + b3*(x-x0)*(x-x1)*(x-x2);
    out[index].y = 0;
}

//5th order interpolation
__global__ void klinearization5thOrder(cufftComplex* out, cufftComplex *in, const float* resampleCurve, const int width, const int samples) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int j = index%width;
    int offset = index-j;

    float x = resampleCurve[j];
//	int x0 = (int)x;
//	int x1 = x0 + 1;
//	int x2 = x0 + 2;
//	int x3 = x0 + 3;
//	int x4 = x0 + 4;
//	int x5 = x0 + 5;

    int x3 = (int)x;
    int x0 = x3 - 3;
    int x1 = x3 - 2;
    int x2 = x3 - 1;
    int x4 = x3 + 1;
    int x5 = x3 + 2;

    float b0 = in[offset + x0].x;
    float f_x0 = b0;
    float f_x1 = in[offset + x1].x;
    float f_x2 = in[offset + x2].x;
    float f_x3 = in[offset + x3].x;
    float f_x4 = in[offset + x4].x;
    float f_x5 = in[offset + x5].x;
    float b1 = (f_x1-f_x0)/(x1-x0);
    float b2 = (((f_x2-f_x1)/(x2-x1))-b1)/(x2-x0);
    float b3 = (((f_x3-f_x2)/(x3-x2))-b2)/(x3-x0);
    float b4 = (((f_x4-f_x3)/(x4-x3))-b3)/(x4-x0);
    float b5 = (((f_x5-f_x4)/(x5-x4))-b4)/(x5-x0);

    out[index].x = b0 + b1 * (x - x0) + b2*(x-x0)*(x-x1) + b3*(x-x0)*(x-x1)*(x-x2) + b4*(x-x0)*(x-x1)*(x-x2)*(x-x3) + b5*(x-x0)*(x-x1)*(x-x2)*(x-x3)*(x-x4);
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
    if(index < samples-4){
        int j = index%width;
        int offset = index-j;

        float x = resampleCurve[j];
        int x0 = (int)x;
        int x1 = x0 + 1;
        int x2 = x0 + 2;
        int x3 = x0 + 3;

        float f_x0 = in[offset + x0].x;
        float f_x1 = in[offset + x1].x;
        float f_x2 = in[offset + x2].x;
        float f_x3 = in[offset + x2].x;
        float b0 = f_x0;
        float b1 = f_x1-f_x0;
        float b2 = ((f_x2-f_x1)-b1)/(x2-x0);
        float b3 = ((f_x3-f_x2)-b2)/(x3-x0);

        out[index].x = (b0 + b1 * (x - x0) + b2*(x-x0)*(x-x1) + b3*(x-x0)*(x-x1)*(x-x2)) * window[j];
        out[index].y = 0;
    }
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
    if(index < samples-4){
        int j = index%width;
        int offset = index-j;

        float x = resampleCurve[j];
        int x0 = (int)x;
        int x1 = x0 + 1;
        int x2 = x0 + 2;
        int x3 = x0 + 3;

        float f_x0 = in[offset + x0].x;
        float f_x1 = in[offset + x1].x;
        float f_x2 = in[offset + x2].x;
        float f_x3 = in[offset + x2].x;
        float b0 = f_x0;
        float b1 = f_x1-f_x0;
        float b2 = ((f_x2-f_x1)-b1)/(x2-x0);
        float b3 = ((f_x3-f_x2)-b2)/(x3-x0);

        float linearizedAndWindowedInX = (b0 + b1 * (x - x0) + b2*(x-x0)*(x-x1) + b3*(x-x0)*(x-x1)*(x-x2)) * window[j];
        out[index].x = linearizedAndWindowedInX * phaseComplex[j].x;
        out[index].y = linearizedAndWindowedInX * phaseComplex[j].y;
    }
}

__global__ void sinusoidalScanCorrection(float* out, float *in, float* sinusoidalResampleCurve, const int width, const int height, const int depth, const int samples) { //width: samplesPerAscan; height: ascansPerBscan, depth: bscansPerBuffer
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < samples-width){ //
        int j = index%(width); //pos within ascan
        int k = (index/width)%height;//pos within bscan
        int l = index/(width*height);//pos within buffer

        float n_sinusoidal = sinusoidalResampleCurve[k];//((float)height/M_PI)*acos((float)(1.0-((2.0*(float)k)/(float)height)));
        //float n_sinusoidal = ((float)height/2)*(1-cos((M_PI*(float)k)/(float)height));// acos((float)(1.0-((2.0*(float)k)/(float)height)));
        //float x = sinusoidalResampleCurve[k];
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

/*	Algorithm implemented by Ben Matthias after S.Moon et al., "Reference spectrum extraction and fixed-pattern noise removal in
optical coherence tomography", Optics Express 18(23):24395-24404, 2010	*/
//todo: optimize cuda code
__global__ void getMinimumVarianceMean(cufftComplex *meanLine, cufftComplex *in, int width, int height, int segs) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < width) {
        int i, j;
        cufftComplex cur, segMean, meanAtMinVariance;
        float segVariance, minVariance;
        int segWidth = height / segs;
        int offset;
        float factor = 1.0f / (float)segWidth;

        for (i = 0; i<segs; i++) {
            segMean.x = 0.0f;
            segMean.y = 0.0f;
            segVariance = 0.0f;
            offset = i*segWidth*width + index;

            // calculate segmental mean
            for (j = 0; j < segWidth; j++) {
                cur = in[offset + j*width];
                segMean.x = segMean.x + cur.x;
                segMean.y = segMean.y + cur.y;
            }
            segMean.x = segMean.x*factor;
            segMean.y = segMean.y*factor;

            // calculate segmental variance
            for (j = 0; j<segWidth; j++) {
                cur = in[offset + j*width];
                segVariance += pow(cur.x - segMean.x, 2) + pow(cur.y - segMean.y, 2);
            }
            segVariance *= factor;

            if (i == 0) {
                minVariance = segVariance;
                meanAtMinVariance = segMean;
            }
            else { // remember segmental mean with minimum segmental variance
                if (segVariance < minVariance) {
                    minVariance = segVariance;
                    meanAtMinVariance = segMean;
                }
            }
        }

        // set segmental mean of minimum-variance segment to meanLine
        meanLine[index] = meanAtMinVariance;
    }
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

extern "C" void cuda_updateDispersionCurve(float* h_dispersionCurve, int size) {
    if (d_dispersionCurve != NULL && h_dispersionCurve != NULL)
        checkCudaErrors(cudaMemcpyAsync(d_dispersionCurve, h_dispersionCurve, size * sizeof(float), cudaMemcpyHostToDevice, processStream));
}

extern "C" void cuda_updateWindowCurve(float* h_windowCurve, int size) {
    if (d_windowCurve != NULL && h_windowCurve != NULL)
        checkCudaErrors(cudaMemcpyAsync(d_windowCurve, h_windowCurve, size * sizeof(float), cudaMemcpyHostToDevice, processStream));
}

extern "C" void cuda_registerProcessedRecordBuffer(void* h_recBuffer, size_t size) {
    checkCudaErrors(cudaHostRegister(h_recBuffer, size, cudaHostRegisterPortable));
    host_RecordBuffer = h_recBuffer;
}

extern "C" void cuda_unregisterProcessedRecordBuffer(void* h_recBuffer) {
    checkCudaErrors(cudaHostUnregister(h_recBuffer));
    host_RecordBuffer = NULL;
}

extern "C" void cuda_registerStreamingBuffers(void* h_streamingBuffer1, void* h_streamingBuffer2, size_t bytesPerBuffer) {
    checkCudaErrors(cudaHostRegister(h_streamingBuffer1, bytesPerBuffer, cudaHostRegisterPortable)); //?
    checkCudaErrors(cudaHostRegister(h_streamingBuffer2, bytesPerBuffer, cudaHostRegisterPortable)); //?
    host_streamingBuffer1 = h_streamingBuffer1; //?
    host_streamingBuffer2 = h_streamingBuffer2; //?
}

extern "C" void cuda_unregisterStreamingBuffers() {
    checkCudaErrors(cudaHostUnregister(host_streamingBuffer1)); //?
    checkCudaErrors(cudaHostUnregister(host_streamingBuffer2)); //?
    host_streamingBuffer1 = NULL; //?
    host_streamingBuffer2 = NULL; //?
}

//Removes half of each processed A-scan (the mirror artefacts), logarithmizes each value of magnitude of remaining A-scan and copies it into an output array. This output array can be used to display the processed OCT data.
__global__ void postProcessTruncateLog(float *output, const cufftComplex *input, const int outputAscanLength, const int samples, const int bufferNumberInVolume, const float max, const float min, const float addend, const float coeff) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < samples / 2) { //vielleicht hier Anhaltspunkt um PS-Komponenten zu trennen
        int lineIndex = index / outputAscanLength;
        int inputArrayIndex = lineIndex *outputAscanLength + index;

        //Note: log(sqrt(x*x+y*y)) == 0.5*log(x*x+y*y) --> the result here is 20*log(magnitude) and not 10*log...
        //amplitude:
        output[index] = coeff*((((10.0f*log10f((input[inputArrayIndex].x*input[inputArrayIndex].x) + (input[inputArrayIndex].y*input[inputArrayIndex].y))) - min) / (max - min)) + addend);
        //phase:
        //output[index] = coeff*((((10.0f*log10f(atan(input[inputArrayIndex].y/input[inputArrayIndex].x))) - min) / (max - min)) + addend);
        output[index] = __saturatef(output[index]); //Clamp values to be within the interval [+0.0, 1.0].
    }
}

//Removes half of each processed A-scan (the mirror artefacts), calculates magnitude of remaining A-scan and copies it into an output array. This output array can be used to display the processed OCT data.
__global__ void postProcessTruncateLin(float *output, const cufftComplex *input, const int outputAscanLength, const int samples, const float max, const float min, const float addend, const float coeff) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < samples / 2) {
        int lineIndex = index / outputAscanLength;
        int inputArrayIndex = lineIndex * outputAscanLength + index;

        //amplitude:
        output[index] = coeff * ((((sqrt((input[inputArrayIndex].x*input[inputArrayIndex].x) + (input[inputArrayIndex].y*input[inputArrayIndex].y))) - min) / (max - min)) + addend);
        //phase:
        //output[index] = coeff*(((((atan(input[inputArrayIndex].y/input[inputArrayIndex].x))) - min) / (max - min)) + addend);
        output[index] = __saturatef(output[index]);//Clamp values to be within the interval [+0.0, 1.0].
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

        if (bscanIndex % 2 == 0 && ascanIndex >= ascansPerBscan/2) { //?muss geändert werden
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

__global__ void updateDisplayedBscanFrame(float *displayBuffer, const float* processedVolume, const unsigned int bscansPerVolume, const unsigned int bscansPerComponent, const unsigned int samplesInSingleFrame, const unsigned int frameNr, const unsigned int displayFunctionFrames, const int displayFunction) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < samplesInSingleFrame) {
        displayBuffer[i] = processedVolume[(frameNr+bscansPerComponent*(frameNr/(bscansPerComponent)))*samplesInSingleFrame + (samplesInSingleFrame-1) - i];

        //todo: optimize averaging and MIP! Use Parallel Reduction for averaging! Use enum instead of int
        if(displayFunctionFrames > 1){
            switch(displayFunction){
            case 0: //Averaging
                int frameCount = 1;
                for (int j = 1; j <= displayFunctionFrames; j++){
                    int frameForAveraging = frameNr+j;
                    if(frameForAveraging < bscansPerVolume){
                        displayBuffer[i] += processedVolume[frameForAveraging*samplesInSingleFrame + (samplesInSingleFrame-1) - i];
                        frameCount++;
                    }
                }
                displayBuffer[i] = displayBuffer[i]/frameCount;
                break;
            case 1: //MIP
                float tmp = 0;
                if(displayFunctionFrames > 1){
                    for (int j = 1; j <= displayFunctionFrames; j++){
                        int frameForMIP = frameNr+j;
                        if(frameForMIP < bscansPerVolume){
                            if(tmp<processedVolume[frameForMIP*samplesInSingleFrame + (samplesInSingleFrame-1) - i]){
                                tmp = processedVolume[frameForMIP*samplesInSingleFrame + (samplesInSingleFrame-1) - i];
                            }
                        }
                    }
                    displayBuffer[i] = tmp;
                }
                break;
            default:
                break;
            }
        }
    }
}

__global__ void updateDisplayedBscan2Frame(float *displayBuffer2, const float* processedVolume, const unsigned int bscansPerVolume, const unsigned int bscansPerComponent, const unsigned int samplesInSingleFrame, const unsigned int frameNr, const unsigned int displayFunctionFrames, const int displayFunction) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < samplesInSingleFrame) {
        displayBuffer2[i] = processedVolume[(frameNr+bscansPerComponent*(1+frameNr/(bscansPerComponent)))*samplesInSingleFrame + (samplesInSingleFrame-1) - i];

        //todo: optimize averaging and MIP! Use Parallel Reduction for averaging! Use enum instead of int
        if(displayFunctionFrames > 1){
            switch(displayFunction){
            case 0: //Averaging
                int frameCount = 1;
                for (int j = 1; j <= displayFunctionFrames; j++){
                    int frameForAveraging = frameNr+j;
                    if(frameForAveraging < bscansPerVolume){
                        displayBuffer2[i] += processedVolume[frameForAveraging*samplesInSingleFrame + (samplesInSingleFrame-1) - i];
                        frameCount++;
                    }
                }
                displayBuffer2[i] = displayBuffer2[i]/frameCount;
                break;
            case 1: //MIP
                float tmp = 0;
                if(displayFunctionFrames > 1){
                    for (int j = 1; j <= displayFunctionFrames; j++){
                        int frameForMIP = frameNr+j;
                        if(frameForMIP < bscansPerVolume){
                            if(tmp<processedVolume[frameForMIP*samplesInSingleFrame + (samplesInSingleFrame-1) - i]){
                                tmp = processedVolume[frameForMIP*samplesInSingleFrame + (samplesInSingleFrame-1) - i];
                            }
                        }
                    }
                    displayBuffer2[i] = tmp;
                }
                break;
            default:
                break;
            }
        }
    }
}

__global__ void updateDisplayedEnFaceViewFrame(float *displayBuffer, const float* processedVolume, const unsigned int frameWidth, const unsigned int samplesInSingleFrame, const unsigned int frameNr, const unsigned int displayFunctionFrames, const int displayFunction) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < samplesInSingleFrame) {
        displayBuffer[(samplesInSingleFrame-1)-i] = processedVolume[frameNr+i*frameWidth];

        //todo: optimize averaging and MIP! Use Parallel Reduction for averaging! Use enum instead of int
        if(displayFunctionFrames > 1){
            switch(displayFunction){
            case 0: //Averaging
                int frameCount = 1;
                for (int j = 1; j <= displayFunctionFrames; j++){
                    int frameForAveraging = frameNr+j;
                    if(frameForAveraging < frameWidth){
                        displayBuffer[(samplesInSingleFrame-1)-i] += processedVolume[frameForAveraging+i*frameWidth];
                        frameCount++;
                    }
                }
                displayBuffer[(samplesInSingleFrame-1)-i] = displayBuffer[(samplesInSingleFrame-1)-i]/frameCount;
                break;
            case 1: //MIP
                float tmp = 0;
                if(displayFunctionFrames > 1){
                    for (int j = 1; j <= displayFunctionFrames; j++){
                        int frameForMIP = frameNr+j;
                        if(frameForMIP < frameWidth){
                            if(tmp<processedVolume[frameForMIP+i*frameWidth]){
                                tmp = processedVolume[frameForMIP+i*frameWidth];
                            }
                        }
                    }
                    displayBuffer[(samplesInSingleFrame-1)-i] = tmp;
                }
                break;
            default:
                break;
            }
        }
    }
}

__global__ void updateDisplayedRetardanceFrame(float *displayBuffer, const float* processedVolume, const unsigned int bscansPerVolume, const unsigned int bscansPerComponent, const unsigned int samplesInSingleFrame, const unsigned int frameNr, const unsigned int displayFunctionFrames, const int displayFunction) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < samplesInSingleFrame) {
        double component1 = processedVolume[(frameNr+bscansPerComponent*(frameNr/(bscansPerComponent)))*samplesInSingleFrame + (samplesInSingleFrame-1) - i];
        double component2 = processedVolume[(frameNr+bscansPerComponent*(1+frameNr/(bscansPerComponent)))*samplesInSingleFrame + (samplesInSingleFrame-1) - i];
        if(component1 > 0.3 || component2 > 0.3){
            double retardance = atan2(sqrt(component1), sqrt(component2));
            double norm = retardance/M_PI_2;
            if(norm < 0.25){
                displayBuffer[i*3] = 0;
                displayBuffer[i*3 + 1] = norm * 4;
                displayBuffer[i*3 + 2] = 1;
            }else if(norm >0.25 && norm < 0.5){
                displayBuffer[i*3] = (norm - 0.25) * 4;
                displayBuffer[i*3 + 1] = 1;
                displayBuffer[i*3 + 2] = 1 - (norm-0.25) * 4;
            }else if(norm >0.5 && norm < 0.75){
                displayBuffer[i*3] = 1;
                displayBuffer[i*3 + 1] = 1 - (norm-0.5) * 4 ;
                displayBuffer[i*3 + 2] = 0;
            }else if(norm >0.75 && norm < 1.0){
                displayBuffer[i*3] = 1 - (norm-0.75) * 2;
                displayBuffer[i*3 + 1] = 0 ;
                displayBuffer[i*3 + 2] = 0;
            }


        }else{
            displayBuffer[i*3] = 0;
            displayBuffer[i*3 + 1] = 0;
            displayBuffer[i*3 + 2] = 0;
}


        //todo: optimize averaging and MIP! Use Parallel Reduction for averaging! Use enum instead of int
        if(displayFunctionFrames > 1){
            switch(displayFunction){
            case 0: //Averaging
                int frameCount = 1;
                for (int j = 1; j <= displayFunctionFrames; j++){
                    int frameForAveraging = frameNr+j;
                    if(frameForAveraging < bscansPerVolume){
                        displayBuffer[i] += processedVolume[frameForAveraging*samplesInSingleFrame + (samplesInSingleFrame-1) - i];
                        frameCount++;
                    }
                }
                displayBuffer[i] = displayBuffer[i]/frameCount;
                break;
            case 1: //MIP
                float tmp = 0;
                if(displayFunctionFrames > 1){
                    for (int j = 1; j <= displayFunctionFrames; j++){
                        int frameForMIP = frameNr+j;
                        if(frameForMIP < bscansPerVolume){
                            if(tmp<processedVolume[frameForMIP*samplesInSingleFrame + (samplesInSingleFrame-1) - i]){
                                tmp = processedVolume[frameForMIP*samplesInSingleFrame + (samplesInSingleFrame-1) - i];
                            }
                        }
                    }
                    displayBuffer[i] = tmp;
                }
                break;
            default:
                break;
            }
        }
    }
}

__global__ void updateDisplayedVolume(const float* processedBuffer, const unsigned int samplesInBuffer, const unsigned int currBufferNr, const unsigned int bscansPerBuffer, dim3 textureDim) {
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
        out[index] = (unsigned char)(input[index] * (255.0)); //float input with values between 0.0 and 1.0 is converted to 8 bit (0 to 255) output
    }else if(outputBitdepth > 8 && outputBitdepth <= 10){
        unsigned short* out = (unsigned short*)output;
        out[index] = (unsigned short)(input[index] * (1023.0)); //10 bit
    }else if(outputBitdepth > 10 && outputBitdepth <= 12){
        unsigned short* out = (unsigned short*)output;
        out[index] = (unsigned short)(input[index] * (4095.0)); //12 bit
    }else if(outputBitdepth > 12 && outputBitdepth <= 16){
        unsigned short* out = (unsigned short*)output;
        out[index] = (unsigned short)(input[index] * (65535.0)); //16 bit
    }else if(outputBitdepth > 16 && outputBitdepth <= 24){
        unsigned int* out = (unsigned int*)output;
        out[index] = (unsigned int)(input[index] * (167772165.0f)); //24 bit
    }else{
        unsigned int* out = (unsigned int*)output;
        out[index] = (unsigned int)(input[index] * (4294967295.0f)); //32 bit
    }
}

extern "C" void cuda_updateResampleCurve(float* h_resampleCurve, int size) {
    if (d_resampleCurve != NULL && h_resampleCurve != NULL)
        checkCudaErrors(cudaMemcpyAsync(d_resampleCurve, h_resampleCurve, size * sizeof(float), cudaMemcpyHostToDevice, processStream));
}

extern "C" void initializeCuda(void* h_buffer1, void* h_buffer2, OctAlgorithmParameters* parameters) {
    signalLength = parameters->samplesPerLine;
    ascansPerBscan = parameters->ascansPerBscan;
    bscansPerBuffer = parameters->bscansPerBuffer;
    buffersPerVolume = parameters->buffersPerVolume;
    bscansPerComponent = parameters->bscansPerComponent;
    samplesPerBuffer = signalLength*ascansPerBscan*bscansPerBuffer;
    samplesPerVolume = samplesPerBuffer * buffersPerVolume;
    samplesPerComponent = bscansPerComponent * ascansPerBscan * signalLength;

    host_buffer1 = h_buffer1;
    host_buffer2 = h_buffer2;
    params = parameters;
    bytesPerSample = ceil((double)(parameters->bitDepth) / 8.0);

    checkCudaErrors(cudaStreamCreate(&processStream));
    checkCudaErrors(cudaStreamCreate(&copyStream));
    checkCudaErrors(cudaStreamCreate(&copyStreamD2H));
    checkCudaErrors(cudaStreamCreate(&streamingStreamD2H));
    checkCudaErrors(cudaStreamCreate(&streamingStreamD2H2));
    checkCudaErrors(cudaStreamCreate(&displayStream));
    checkCudaErrors(cudaStreamCreate(&userRequestStream));

    checkCudaErrors(cudaEventCreateWithFlags(&processEvent, cudaEventDisableTiming)); //this event is used for synchronization via cudaStreamWaitEvent() to synchronize processing and copying of processed data to ram. timing data is not necessary. Events created with this flag (cudaEventDisableTiming) specified and the cudaEventBlockingSync flag not specified will provide the best performance when used with cudaStreamWaitEvent()

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Cuda error: %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //resample curve
    checkCudaErrors(cudaMalloc((void**)&d_resampleCurve, sizeof(float)*signalLength));

    //dispersion curve
    checkCudaErrors(cudaMalloc((void**)&d_dispersionCurve, sizeof(float)*signalLength));

    //sinusoidal resample curve for sinusoidal scan correction
    checkCudaErrors(cudaMalloc((void**)&d_sinusoidalResampleCurve, sizeof(float)*ascansPerBscan));
    fillSinusoidalScanCorrectionCurve<<<ascansPerBscan, 1, 0, processStream>>> (d_sinusoidalResampleCurve, ascansPerBscan);

    //window curve
    checkCudaErrors(cudaMalloc((void**)&d_windowCurve, sizeof(float)*signalLength));

    //allocate device memory for raw signal
    checkCudaErrors(cudaMalloc((void**)&d_inputBuffer1, bytesPerSample*samplesPerBuffer));
    cudaMemset(d_inputBuffer1, 0, bytesPerSample*samplesPerBuffer);
    checkCudaErrors(cudaMalloc((void**)&d_inputBuffer2, bytesPerSample*samplesPerBuffer));
    cudaMemset(d_inputBuffer2, 0, bytesPerSample*samplesPerBuffer);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //allocate device memory for streaming processed signal
    checkCudaErrors(cudaMalloc((void**)&d_outputBuffer, bytesPerSample*samplesPerBuffer/2));
    checkCudaErrors(cudaMalloc((void**)&d_outputBuffer2, bytesPerSample*samplesPerBuffer/2));
    cudaMemset(d_outputBuffer, 0, bytesPerSample*samplesPerBuffer/2);
    cudaMemset(d_outputBuffer2, 0, bytesPerSample*samplesPerBuffer/2);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //allocate device memory for k-linearized signal
    checkCudaErrors(cudaMalloc((void**)&d_inputLinearized, sizeof(cufftComplex)*samplesPerBuffer));
    cudaMemset(d_inputLinearized, 0, sizeof(cufftComplex)*samplesPerBuffer);

    //allocate device memory for dispersion compensation phase
    checkCudaErrors(cudaMalloc((void**)&d_phaseCartesian, sizeof(cufftComplex)*signalLength));
    cudaMemset(d_phaseCartesian, 0, sizeof(cufftComplex)*signalLength);

    //allocate device memory for processed signal
    checkCudaErrors(cudaMalloc((void**)&d_processedBuffer, sizeof(float)*samplesPerVolume/2));
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //allocate device memory for temporary buffer for sinusoidal scan correction
    checkCudaErrors(cudaMalloc((void**)&d_sinusoidalScanTmpBuffer, sizeof(float)*samplesPerBuffer/2));
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //allocate device memory for fft buffer
    checkCudaErrors(cudaMalloc((void**)&d_fftBuffer, sizeof(cufftComplex)*samplesPerBuffer));
    cudaMemset(d_fftBuffer, 0, sizeof(cufftComplex)*samplesPerBuffer);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //allocate device memory for fixed noise removal mean A-scan
    checkCudaErrors(cudaMalloc((void**)&d_meanALine, sizeof(cufftComplex)*signalLength/2));
    cudaMemset(d_meanALine, 0, sizeof(cufftComplex)*signalLength/2);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //register existing host memory for use by cuda to accelerate cudaMemcpy
    checkCudaErrors(cudaHostRegister(host_buffer1, samplesPerBuffer * bytesPerSample, cudaHostRegisterPortable));
    checkCudaErrors(cudaHostRegister(host_buffer2, samplesPerBuffer * bytesPerSample, cudaHostRegisterPortable));

    //create fft plan and set stream
    cufftPlan1d(&d_plan, signalLength, CUFFT_C2C, ascansPerBscan*bscansPerBuffer);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    cufftSetStream(d_plan, processStream);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cudaInitialized = true;
    firstRun = true;
    bufferNumber = 0;
    bufferNumberInVolume = params->buffersPerVolume-1;
    streamingBufferNumber = 0;
    processedBuffers = 0;
    streamedBuffers = 0;
    fixedPatternNoiseDetermined = false;

    //todo: find a way to automatically determine optimal blockSize and optimal gridSize
    blockSize = 128;
    gridSize = samplesPerBuffer / blockSize;
}

extern "C" void cleanupCuda() {
    if (cudaInitialized) {
        freeCudaMem(d_inputBuffer1);
        freeCudaMem(d_inputBuffer2);
        freeCudaMem(d_outputBuffer);
        freeCudaMem(d_outputBuffer2);
        freeCudaMem(d_windowCurve);
        freeCudaMem(d_fftBuffer);
        freeCudaMem(d_meanALine);
        freeCudaMem(d_processedBuffer);
        freeCudaMem(d_sinusoidalScanTmpBuffer);
        freeCudaMem(d_inputLinearized);
        freeCudaMem(d_phaseCartesian);
        freeCudaMem(d_resampleCurve);
        freeCudaMem(d_dispersionCurve);
        freeCudaMem(d_sinusoidalResampleCurve);

        cufftDestroy(d_plan);

        checkCudaErrors(cudaStreamDestroy(processStream));
        checkCudaErrors(cudaStreamDestroy(copyStream));
        checkCudaErrors(cudaStreamDestroy(copyStreamD2H));
        checkCudaErrors(cudaStreamDestroy(streamingStreamD2H));
        checkCudaErrors(cudaStreamDestroy(displayStream));
        checkCudaErrors(cudaStreamDestroy(userRequestStream));

        checkCudaErrors(cudaEventDestroy(processEvent));

        if (host_buffer1 != NULL) {
            cudaHostUnregister(host_buffer1);
        }
        if (host_buffer2 != NULL) {
            cudaHostUnregister(host_buffer2);
        }

        cudaInitialized = false;
        firstRun = true;
        fixedPatternNoiseDetermined = false;
    }
}

extern "C" void freeCudaMem(void* data) {
    if (data != NULL) {
        checkCudaErrors(cudaFree(data));
        data = NULL;
    }
    else {
        printf("Cuda: Failed to free memory.");
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
    int depth = bscansPerBuffer*buffersPerVolume;
    int samplesPerFrame = width * height;
    if (d_bscanDisplayBuffer != NULL) {
        frameNr = frameNr >= 0 && frameNr < depth ? frameNr : 0;
        updateDisplayedBscanFrame<<<gridSize/2, blockSize, 0, userRequestStream>>>((float*)d_bscanDisplayBuffer, d_processedBuffer, depth, bscansPerBuffer / 2, samplesPerFrame / 2, frameNr, displayFunctionFrames, displayFunction);
    }
    if (cuBufHandleBscan != NULL) {
        cuda_unmap(cuBufHandleBscan, userRequestStream);
    }
}

extern "C" void changeDisplayedBscan2Frame(unsigned int frameNr, unsigned int displayFunctionFrames, int displayFunction){
    void* d_bscan2DisplayBuffer = NULL;
    if (cuBufHandleBscan2 != NULL) {
        d_bscan2DisplayBuffer = cuda_map(cuBufHandleBscan2, userRequestStream);
    }
    //update 2D b-scan display
    int width = signalLength;
    int height = ascansPerBscan;
    int depth = bscansPerBuffer*buffersPerVolume;
    int samplesPerFrame = width * height;
    if (d_bscan2DisplayBuffer != NULL) {
        frameNr = frameNr >= 0 && frameNr < depth ? frameNr : 0;
        updateDisplayedBscan2Frame<<<gridSize/2, blockSize, 0, userRequestStream>>>((float*)d_bscan2DisplayBuffer, d_processedBuffer, depth, bscansPerBuffer / 2, samplesPerFrame / 2, frameNr, displayFunctionFrames, displayFunction);
    }
    if (cuBufHandleBscan2 != NULL) {
       cuda_unmap(cuBufHandleBscan2, userRequestStream);
    }
}

//todo: simplify/refactor changeDisplayedEnfaceFrame and changedisplaydBscanFrame. avoid duplicate code
extern "C" void changeDisplayedEnFaceFrame(unsigned int frameNr, unsigned int displayFunctionFrames, int displayFunction) {
    void* d_enFaceViewDisplayBuffer = NULL;
    if (cuBufHandleEnFaceView != NULL) {
        d_enFaceViewDisplayBuffer = cuda_map(cuBufHandleEnFaceView, userRequestStream);
    }
    //update 2D en face view display
    unsigned int width = bscansPerBuffer*buffersPerVolume;
    unsigned int height = ascansPerBscan;
    unsigned int samplesPerFrame = width * height;
    if (d_enFaceViewDisplayBuffer != NULL) {
        frameNr = frameNr >= 0 && frameNr < signalLength/2 ? frameNr : 0;
        updateDisplayedEnFaceViewFrame<<<gridSize, blockSize, 0, userRequestStream>>>((float*)d_enFaceViewDisplayBuffer, d_processedBuffer, signalLength/2, samplesPerFrame, frameNr, displayFunctionFrames, displayFunction);
    }
    if (cuBufHandleEnFaceView != NULL) {
        cuda_unmap(cuBufHandleEnFaceView, userRequestStream);
    }
}

extern "C" void changeDisplayedRetardanceFrame(unsigned int frameNr, unsigned int displayFunctionFrames, int displayFunction) {
    void* d_retardanceDisplayBuffer = NULL;
    if (cuBufHandleRetardance != NULL) {
        d_retardanceDisplayBuffer = cuda_map(cuBufHandleRetardance, userRequestStream);
    }
    //update 2D b-scan display
    int width = signalLength;
    int height = ascansPerBscan;
    int depth = bscansPerBuffer*buffersPerVolume;
    int samplesPerFrame = width * height;
    if (d_retardanceDisplayBuffer != NULL) {
        frameNr = frameNr >= 0 && frameNr < depth ? frameNr : 0;
        updateDisplayedRetardanceFrame<<<gridSize/2, blockSize, 0, userRequestStream>>>((float*)d_retardanceDisplayBuffer, d_processedBuffer, depth, bscansPerBuffer / 2, samplesPerFrame / 2, frameNr, displayFunctionFrames, displayFunction);
    }
    if (cuBufHandleRetardance != NULL) {
       cuda_unmap(cuBufHandleRetardance, userRequestStream);
    }
}

extern "C" inline void updateBscanDisplayBuffer(unsigned int frameNr, unsigned int displayFunctionFrames, int displayFunction) {
    void* d_bscanDisplayBuffer = NULL;
    if (cuBufHandleBscan != NULL) {
        d_bscanDisplayBuffer = cuda_map(cuBufHandleBscan, displayStream);
    }
    //update 2D b-scan display
    int width = signalLength;
    int height = ascansPerBscan;
    int depth = bscansPerBuffer * buffersPerVolume;
    int samplesPerFrame = width * height;
    if (d_bscanDisplayBuffer != NULL) {
        frameNr = frameNr >= 0 && frameNr < depth ? frameNr : 0;
        updateDisplayedBscanFrame<<<gridSize/2, blockSize, 0, displayStream>>>((float*)d_bscanDisplayBuffer, d_processedBuffer, depth, bscansPerBuffer / 2, samplesPerFrame / 2, frameNr, displayFunctionFrames, displayFunction);
    }
    if (cuBufHandleBscan != NULL) {
        cuda_unmap(cuBufHandleBscan, displayStream);
    }
}

extern "C" inline void updateBscan2DisplayBuffer(unsigned int frameNr, unsigned int displayFunctionFrames, int displayFunction){
    void* d_bscan2DisplayBuffer = NULL;
    if (cuBufHandleBscan2 != NULL) {
        d_bscan2DisplayBuffer = cuda_map(cuBufHandleBscan2, displayStream);
    }
    //update 2D b-scan display
    int width = signalLength;
    int height = ascansPerBscan;
    int depth = bscansPerBuffer * buffersPerVolume;
    int samplesPerFrame = width * height;
    if (d_bscan2DisplayBuffer != NULL) {
        frameNr = frameNr >= 0 && frameNr < depth ? frameNr : 0;
        updateDisplayedBscan2Frame<<<gridSize/2, blockSize, 0, displayStream>>>((float*)d_bscan2DisplayBuffer, d_processedBuffer, depth, bscansPerBuffer / 2, samplesPerFrame / 2, frameNr, displayFunctionFrames, displayFunction);
    }
    if (cuBufHandleBscan2 != NULL) {
       cuda_unmap(cuBufHandleBscan2, displayStream);
    }
}

extern "C" inline void updateEnFaceDisplayBuffer(unsigned int frameNr, unsigned int displayFunctionFrames, int displayFunction) {
    void* d_enFaceViewDisplayBuffer = NULL;
    if (cuBufHandleEnFaceView != NULL) {
        d_enFaceViewDisplayBuffer = cuda_map(cuBufHandleEnFaceView, displayStream);
    }
    //update 2D en face view display
    unsigned int width = bscansPerBuffer * buffersPerVolume;
    unsigned int height = ascansPerBscan;
    unsigned int samplesPerFrame = width * height;
    if (d_enFaceViewDisplayBuffer != NULL) {
        frameNr = frameNr >= 0 && frameNr < signalLength/2 ? frameNr : 0;
        updateDisplayedEnFaceViewFrame<<<gridSize/2, blockSize, 0, displayStream>>>((float*)d_enFaceViewDisplayBuffer, d_processedBuffer, signalLength/2, samplesPerFrame, frameNr, displayFunctionFrames, displayFunction);
    }
    if (cuBufHandleEnFaceView != NULL) {
        cuda_unmap(cuBufHandleEnFaceView, displayStream);
    }
}

extern "C" inline void updateRetardanceDisplayBuffer(unsigned int frameNr, unsigned int displayFunctionFrames, int displayFunction) {
    void* d_retardanceDisplayBuffer = NULL;
    if (cuBufHandleRetardance != NULL) {
        d_retardanceDisplayBuffer = cuda_map(cuBufHandleRetardance, displayStream);
    }
    //update 2D b-scan display
    int width = signalLength;
    int height = ascansPerBscan;
    int depth = bscansPerBuffer * buffersPerVolume;
    int samplesPerFrame = width * height;
    if (d_retardanceDisplayBuffer != NULL) {
        frameNr = frameNr >= 0 && frameNr < depth ? frameNr : 0;
        updateDisplayedRetardanceFrame<<<gridSize/2, blockSize, 0, displayStream>>>((float*)d_retardanceDisplayBuffer, d_processedBuffer, depth, bscansPerBuffer / 2, samplesPerFrame / 2, frameNr, displayFunctionFrames, displayFunction);
    }
    if (cuBufHandleRetardance != NULL) {
        cuda_unmap(cuBufHandleRetardance, displayStream);
    }
}

extern "C" inline void updateVolumeDisplayBuffer(const float* d_currBuffer, const unsigned int currentBufferNr, const unsigned int bscansPerBuffer) {
    //map graphics resource for access by cuda
    cudaArray* d_volumeViewDisplayBuffer = NULL;
    if (cuBufHandleVolumeView != NULL) {
        d_volumeViewDisplayBuffer = cuda_map3dTexture(cuBufHandleVolumeView, displayStream);
    }
    //calculate dimensions of processed volume
    unsigned int width = bscansPerBuffer * buffersPerVolume;
    unsigned int height = ascansPerBscan;
    unsigned int depth = signalLength/2;
    if (d_volumeViewDisplayBuffer != NULL) {
        //bind voxel array to a writable cuda surface
        cudaError_t error_id = cudaBindSurfaceToArray(surfaceWrite, d_volumeViewDisplayBuffer);
        if (error_id != cudaSuccess) {
            printf("Cuda: Failed to bind surface to cuda array:  %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
            return;
        }

        //write to cuda surface
        dim3 texture_dim(height, width, depth); //todo: use consistent naming of width, height, depth, x, y, z, ...
        updateDisplayedVolume<< <gridSize/2, blockSize, 0, displayStream>>>(d_currBuffer, samplesPerBuffer/2, currentBufferNr, bscansPerBuffer, texture_dim);
    }

    //unmap the graphics resource
    if (cuBufHandleVolumeView != NULL) {
        cuda_unmap(cuBufHandleVolumeView, displayStream);
    }
}

inline void streamProcessedData(float* d_currProcessedBuffer) {
    if (streamedBuffers % (params->streamingBuffersToSkip + 1) == 0) {
        streamedBuffers = 0; //set to zero to avoid overflow
        streamingBufferNumber = (streamingBufferNumber + 1) % 2;
        void* hostDestBuffer = streamingBufferNumber == 0 ? host_streamingBuffer1 : host_streamingBuffer2;
        cudaStreamWaitEvent(streamingStreamD2H, processEvent, 0); //wait for processing to be finished before starting execution of streamingStreamD2H. This just blocks the streamingStreamD2H and not the host
        floatToOutput<<<gridSize / 2, blockSize, 0, streamingStreamD2H>>> (d_outputBuffer, d_currProcessedBuffer, params->bitDepth, samplesPerBuffer / 2);
        floatToOutput<<<gridSize / 2, blockSize, 0, streamingStreamD2H2>>> (d_outputBuffer2, d_currProcessedBuffer, params->bitDepth, samplesPerBuffer / 2);
        checkCudaErrors(cudaMemcpyAsync(hostDestBuffer, (void*)d_outputBuffer, (samplesPerBuffer / 2) * bytesPerSample, cudaMemcpyDeviceToHost, streamingStreamD2H));
        checkCudaErrors(cudaMemcpyAsync(hostDestBuffer, (void*)d_outputBuffer2, (samplesPerBuffer / 2) * bytesPerSample, cudaMemcpyDeviceToHost, streamingStreamD2H));
        checkCudaErrors(cudaStreamAddCallback(streamingStreamD2H, Gpu2HostNotifier::dh2StreamingCallback, hostDestBuffer, 0));
    }
    streamedBuffers++;
}

extern "C" void octCudaPipeline(void* h_inputSignal) {
    //check if cuda buffers are initialized
    if (!cudaInitialized) {
        printf("Cuda: Device buffers are not initialized!");
        return;
    }

    //set copy und process buffer pointers
    bufferNumber = (bufferNumber + 1) % 2;
    void* copyBuffer = NULL;
    void* procBuffer = NULL;

    if (bufferNumber == 0) {
        copyBuffer = d_inputBuffer1;
        procBuffer = d_inputBuffer2;
    }
    else {
        copyBuffer = d_inputBuffer2;
        procBuffer = d_inputBuffer1;
    }

    //copy raw oct signal from host
    if (copyBuffer != NULL && h_inputSignal != NULL) {
        checkCudaErrors(cudaMemcpyAsync(copyBuffer, h_inputSignal, samplesPerBuffer * bytesPerSample, cudaMemcpyHostToDevice, copyStream));
    }

    //do not run processing if it's the first run. In the first iteration the processing buffer is empty.
    if (firstRun) {
        firstRun = false;
        cudaStreamSynchronize(copyStream);
        return;
    }

    //start processing: convert input array to cufft complex array
    if (params->bitshift) {
        inputToCufftComplex_and_bitshift<<<gridSize, blockSize, 0, processStream>>> (d_fftBuffer, procBuffer, signalLength,  signalLength, params->bitDepth, samplesPerBuffer, samplesPerComponent);
    }
    else {
        inputToCufftComplex<<<gridSize, blockSize, 0, processStream>>> (d_fftBuffer, procBuffer, signalLength, signalLength, params->bitDepth, samplesPerBuffer);
    }

    //update k-linearization-, dispersion- and windowing-curves if necessary
    cufftComplex* d_fftBuffer2 = d_fftBuffer;
    if (params->resampling && params->resamplingUpdated) {
        cuda_updateResampleCurve(params->resampleCurve, signalLength);
        params->resamplingUpdated = false;
    }
    if (params->dispersionCompensation && params->dispersionUpdated) {
        cuda_updateDispersionCurve(params->dispersionCurve, signalLength);
        fillDispersivePhase<<<signalLength, 1, 0, processStream>>> (d_phaseCartesian, d_dispersionCurve, 1.0, signalLength, 1);
        params->dispersionUpdated = false;
    }
    if (params->windowing && params->windowUpdated) {
        cuda_updateWindowCurve(params->windowCurve, signalLength);
        params->windowUpdated = false;
    }

    //k-linearization and windowing
    if (d_inputLinearized != NULL && params->resampling && params->windowing && !params->dispersionCompensation) {
        if(params->resamplingInterpolation == INTERPOLATION::CUBIC){
            klinearizationCubicAndWindowing<<<gridSize, blockSize, 0, processStream>>>(d_inputLinearized, d_fftBuffer, d_resampleCurve, d_windowCurve, signalLength, samplesPerBuffer);
        } else {
            klinearizationAndWindowing<<<gridSize, blockSize, 0, processStream>>>(d_inputLinearized, d_fftBuffer, d_resampleCurve, d_windowCurve, signalLength, samplesPerBuffer);
        }
        d_fftBuffer2 = d_inputLinearized;
    } else
        //k-linearization and windowing and dispersion compensation
    if (d_inputLinearized != NULL && params->resampling && params->windowing && params->dispersionCompensation) {
        if(params->resamplingInterpolation == INTERPOLATION::CUBIC){
            klinearizationCubicAndWindowingAndDispersionCompensation<<<gridSize, blockSize, 0, processStream>>>(d_inputLinearized, d_fftBuffer, d_resampleCurve, d_windowCurve, d_phaseCartesian, signalLength, samplesPerBuffer);
        } else {
            klinearizationAndWindowingAndDispersionCompensation<<<gridSize, blockSize, 0, processStream>>>(d_inputLinearized, d_fftBuffer, d_resampleCurve, d_windowCurve, d_phaseCartesian, signalLength, samplesPerBuffer);
        }
        d_fftBuffer2 = d_inputLinearized;
    } else
        //dispersion compensation and windowing
    if (!params->resampling && params->windowing && params->dispersionCompensation) {
        dispersionCompensationAndWindowing<<<gridSize, blockSize, 0, processStream>>>(d_fftBuffer2, d_fftBuffer2, d_phaseCartesian, d_windowCurve, signalLength, samplesPerBuffer);
    } else
        //just k-linearization
    if (d_inputLinearized != NULL && params->resampling && !params->windowing && !params->dispersionCompensation) {
        if(params->resamplingInterpolation == INTERPOLATION::CUBIC){
            klinearizationCubic<<<gridSize, blockSize, 0, processStream>>>(d_inputLinearized, d_fftBuffer, d_resampleCurve, signalLength, samplesPerBuffer);
        } else {
            klinearization<<<gridSize, blockSize, 0, processStream>>>(d_inputLinearized, d_fftBuffer, d_resampleCurve, signalLength, samplesPerBuffer);
        }
        d_fftBuffer2 = d_inputLinearized;
    } else
        //just windowing
    if (!params->resampling && params->windowing && !params->dispersionCompensation) {
        windowing<<<gridSize, blockSize, 0, processStream>>>(d_fftBuffer2, d_fftBuffer2, d_windowCurve, signalLength, samplesPerBuffer);
    } else
        //just dispersion compensation
    if (!params->resampling && !params->windowing && params->dispersionCompensation) {
        dispersionCompensation<<<gridSize, blockSize, 0, processStream>>> (d_fftBuffer2, d_fftBuffer2, d_phaseCartesian, signalLength, samplesPerBuffer);
    } else
        //k-linearization and dispersion compensation. nobody will use this in a serious manner, so an optimized "klinearizationAndDispersionCompensation" kernel is not necessary
    if (d_inputLinearized != NULL && params->resampling && !params->windowing && params->dispersionCompensation) {
        if(params->resamplingInterpolation == INTERPOLATION::CUBIC){
            klinearizationCubic<<<gridSize, blockSize, 0, processStream>>>(d_inputLinearized, d_fftBuffer, d_resampleCurve, signalLength, samplesPerBuffer);
        } else {
            klinearization<<<gridSize, blockSize, 0, processStream>>>(d_inputLinearized, d_fftBuffer, d_resampleCurve, signalLength, samplesPerBuffer);
        }
        d_fftBuffer2 = d_inputLinearized;
        dispersionCompensation<<<gridSize, blockSize, 0, processStream>>> (d_fftBuffer2, d_fftBuffer2, d_phaseCartesian, signalLength, samplesPerBuffer);
    }

    //IFFT
    checkCudaErrors(cufftExecC2C(d_plan, d_fftBuffer2, d_fftBuffer2, CUFFT_INVERSE));

    //Fixed-pattern noise removal
    if(params->fixedPatternNoiseRemoval){
        int width = signalLength;
        int height = params->bscansForNoiseDetermination*ascansPerBscan;//ascansPerBscan*bscansPerBuffer;
        if((!params->continuousFixedPatternNoiseDetermination && !fixedPatternNoiseDetermined) || params->continuousFixedPatternNoiseDetermination || params->redetermineFixedPatternNoise){
            getMinimumVarianceMean<<<gridSize, blockSize, 0, processStream>>>(d_meanALine, d_fftBuffer2, width, height, FIXED_PATTERN_NOISE_REMOVAL_SEGMENTS);
            fixedPatternNoiseDetermined = true;
            params->redetermineFixedPatternNoise = false;
        }
        meanALineSubtraction<<<gridSize/2, blockSize, 0, processStream>>>(d_fftBuffer2, d_meanALine, width/2, samplesPerBuffer/2); //here mean a-scan line subtraction of half volume is enough, because in the next step the volume gets truncated anyway
    }

    //get current buffer number in volume (a volume may consist of one or more buffers)
    if(buffersPerVolume > 1){
        bufferNumberInVolume = (bufferNumberInVolume+1)%buffersPerVolume;
    }

    //get current position in processed volume buffer
    float* d_currBuffer = &d_processedBuffer[(samplesPerBuffer/2)*bufferNumberInVolume];

    //postProcessTruncate contains: Mirror artefact removal, Log, Magnitude, Copy to output buffer.
    if (params->signalLogScaling) {
        postProcessTruncateLog<<<gridSize/2, blockSize, 0, processStream>>> (d_currBuffer, d_fftBuffer2, signalLength / 2, samplesPerBuffer, bufferNumberInVolume, params->signalGrayscaleMax, params->signalGrayscaleMin, params->signalAddend, params->signalMultiplicator);
    }
    else {
        postProcessTruncateLin<<<gridSize/2, blockSize, 0, processStream>>> (d_currBuffer, d_fftBuffer2, signalLength / 2, samplesPerBuffer, params->signalGrayscaleMax, params->signalGrayscaleMin, params->signalAddend, params->signalMultiplicator);
    }

    //flip every second bscan
    if (params->bscanFlip) {
        cuda_bscanFlip<<<gridSize/2, blockSize, 0, processStream>>> (d_currBuffer, d_currBuffer, signalLength / 2, ascansPerBscan, (signalLength*ascansPerBscan)/2, samplesPerBuffer/4);
    }

    //sinusoidal scan correction
    if(params->sinusoidalScanCorrection && d_sinusoidalScanTmpBuffer != NULL){
        checkCudaErrors(cudaMemcpy(d_sinusoidalScanTmpBuffer, d_currBuffer, sizeof(float)*samplesPerBuffer/2, cudaMemcpyDeviceToDevice));
        sinusoidalScanCorrection<<<gridSize, blockSize, 0, processStream>>>(d_currBuffer, d_sinusoidalScanTmpBuffer, d_sinusoidalResampleCurve, signalLength/2, ascansPerBscan, bscansPerBuffer, samplesPerBuffer/2);
    }

    //capture the contents of processStream in processEvent. processEvent is used to synchronize processing and copying of processed data to ram (copying of processed data to ram takes place in the stream copyStreamD2H)
    cudaEventRecord(processEvent, processStream);

    //update display buffers
    cudaStreamWaitEvent(displayStream, processEvent, 0);
    if(params->bscanViewEnabled){
        updateBscanDisplayBuffer(params->frameNr, params->functionFramesBscan, params->displayFunctionBscan);
    }
    if(params->bscan2ViewEnabled){
        updateBscan2DisplayBuffer(params->frameNrBScan2, params->functionFramesBscan2, params->displayFunctionBscan2);
    }
    if(params->enFaceViewEnabled){
        updateEnFaceDisplayBuffer(params->frameNrEnFaceView, params->functionFramesEnFaceView, params->displayFunctionEnFaceView);
    }
    if(params->retardanceViewEnabled){
        updateRetardanceDisplayBuffer(params->frameNrRetardance, params->functionFramesRetardance, params->displayFunctionRetardance);
    }
    if(params->volumeViewEnabled){
        updateVolumeDisplayBuffer(d_currBuffer, bufferNumberInVolume, bscansPerBuffer);
    }

    //check errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Cuda error: %s\n", cudaGetErrorString(err));
    }

    //Copy/Stream processed data to Host continuously
    if (params->streamToHost && !params->recordingProcessedEnabled && !params->streamingParamsChanged) {
        params->currentBufferNr = bufferNumberInVolume;
        streamProcessedData(d_currBuffer);
    }

    //wait for copy task (host to device) to complete to avoid race condition
    cudaStreamSynchronize(copyStream);
}

extern "C" void cuda_registerGlBufferBscan(GLuint buf) {
    if (cudaGraphicsGLRegisterBuffer(&cuBufHandleBscan, buf, cudaGraphicsRegisterFlagsWriteDiscard) != cudaSuccess) {
        printf("Cuda: Failed to register buffer %u\n", buf);
    }
}
extern "C" void cuda_registerGlBufferBscan2(GLuint buf){
    if (cudaGraphicsGLRegisterBuffer(&cuBufHandleBscan2, buf, cudaGraphicsRegisterFlagsWriteDiscard) != cudaSuccess) {
        printf("Cuda: Failed to register buffer %u\n", buf);
    }
}
extern "C" void cuda_registerGlBufferEnFaceView(GLuint buf) {
    if (cudaGraphicsGLRegisterBuffer(&cuBufHandleEnFaceView, buf, cudaGraphicsRegisterFlagsWriteDiscard) != cudaSuccess) {
        printf("Cuda: Failed to register buffer %u\n", buf);
    }
}
extern "C" void cuda_registerGlBufferRetardance(GLuint buf) {
    if (cudaGraphicsGLRegisterBuffer(&cuBufHandleRetardance, buf, cudaGraphicsRegisterFlagsWriteDiscard) != cudaSuccess) {
        printf("Cuda: Failed to register buffer %u\n", buf);
    }
}
extern "C" void cuda_registerGlBufferVolumeView(GLuint buf) {
    cudaError_t err = cudaGraphicsGLRegisterImage(&cuBufHandleVolumeView, buf, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    if (err != cudaSuccess) {
        printf("Cuda: Failed to register buffer %u\n", buf);
    }
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
