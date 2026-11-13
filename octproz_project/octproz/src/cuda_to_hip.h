/**
**  CUDA-to-HIP compatibility header for OCTproZ
**  This file maps CUDA API symbols to their HIP equivalents when building
**  with HIP for AMD GPUs. On CUDA, it is a no-op.
**
**  Created by Claude for the ROCm port of OCTproZ.
**/

#ifndef CUDA_TO_HIP_H
#define CUDA_TO_HIP_H

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

// Include HIP runtime to ensure types are defined
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

// Runtime API types
#define cudaError_t                         hipError_t
#define cudaSuccess                         hipSuccess
#define cudaStream_t                        hipStream_t
#define cudaEvent_t                         hipEvent_t
#define cudaDeviceProp                      hipDeviceProp_t
// Note: cudaArray in CUDA is struct cudaArray, while hipArray_t is already a typedef for struct hipArray*
// So cudaArray* in CUDA corresponds to hipArray_t in HIP (without extra *)
// We use hipArray (the struct type) so that cudaArray* matches hipArray*
#define cudaArray                           hipArray

// Memory management
#define cudaMalloc                          hipMalloc
#define cudaFree                            hipFree
#define cudaMemcpy                          hipMemcpy
#define cudaMemcpyAsync                     hipMemcpyAsync
#define cudaMemset                          hipMemset
#define cudaMemGetInfo                      hipMemGetInfo
#define cudaMemcpyHostToDevice              hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost              hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice            hipMemcpyDeviceToDevice

// Host memory
#define cudaHostRegister                    hipHostRegister
#define cudaHostUnregister                  hipHostUnregister
#define cudaHostRegisterPortable            hipHostRegisterPortable
#define cudaHostGetDevicePointer            hipHostGetDevicePointer

// Device management
#define cudaGetDevice                       hipGetDevice
#define cudaSetDevice                       hipSetDevice
#define cudaGetDeviceCount                  hipGetDeviceCount
#define cudaGetDeviceProperties             hipGetDeviceProperties
#define cudaDeviceGetAttribute              hipDeviceGetAttribute
#define cudaDeviceSynchronize               hipDeviceSynchronize
#define cudaDevAttrClockRate                hipDeviceAttributeClockRate
#define cudaDevAttrMemoryClockRate          hipDeviceAttributeMemoryClockRate

// Streams
#define cudaStreamCreate                    hipStreamCreate
#define cudaStreamCreateWithFlags           hipStreamCreateWithFlags
#define cudaStreamDestroy                   hipStreamDestroy
#define cudaStreamNonBlocking               hipStreamNonBlocking

// Events
#define cudaEventCreate                     hipEventCreate
#define cudaEventCreateWithFlags            hipEventCreateWithFlags
#define cudaEventRecord                     hipEventRecord
#define cudaEventSynchronize                hipEventSynchronize
#define cudaEventDestroy                    hipEventDestroy
#define cudaEventBlockingSync               hipEventBlockingSync

// Error handling
#define cudaGetLastError                    hipGetLastError
#define cudaPeekAtLastError                 hipPeekAtLastError
#define cudaGetErrorString                  hipGetErrorString

// Host functions
#define cudaLaunchHostFunc                  hipLaunchHostFunc
#define cudaHostFn_t                        hipHostFn_t

// Graphics interop
// cudaGraphicsResource is the struct, cudaGraphicsResource_t is the pointer type
#define cudaGraphicsResource                hipGraphicsResource
#define cudaGraphicsResource_t              hipGraphicsResource_t
#define cudaGraphicsGLRegisterBuffer        hipGraphicsGLRegisterBuffer
#define cudaGraphicsGLRegisterImage         hipGraphicsGLRegisterImage
#define cudaGraphicsMapResources            hipGraphicsMapResources
#define cudaGraphicsUnmapResources          hipGraphicsUnmapResources
#define cudaGraphicsResourceGetMappedPointer hipGraphicsResourceGetMappedPointer
#define cudaGraphicsSubResourceGetMappedArray hipGraphicsSubResourceGetMappedArray
#define cudaGraphicsUnregisterResource      hipGraphicsUnregisterResource
#define cudaGraphicsRegisterFlagsWriteDiscard hipGraphicsRegisterFlagsWriteDiscard
#define cudaGraphicsRegisterFlagsSurfaceLoadStore hipGraphicsRegisterFlagsSurfaceLoadStore

// Surface objects
#define cudaSurfaceObject_t                 hipSurfaceObject_t
#define cudaCreateSurfaceObject             hipCreateSurfaceObject
#define cudaDestroySurfaceObject            hipDestroySurfaceObject
#define cudaResourceDesc                    hipResourceDesc
#define cudaResourceTypeArray               hipResourceTypeArray
#define cudaSurfaceType3D                   hipSurfaceType3D
#define cudaBindSurfaceToArray              hipBindSurfaceToArray

// cuFFT to hipFFT
#define cufftHandle                         hipfftHandle
#define cufftComplex                        hipfftComplex
#define cufftResult                         hipfftResult
#define cufftPlan1d                         hipfftPlan1d
#define cufftExecC2C                        hipfftExecC2C
#define cufftSetStream                      hipfftSetStream
#define cufftDestroy                        hipfftDestroy
#define CUFFT_C2C                           HIPFFT_C2C
#define CUFFT_INVERSE                       HIPFFT_BACKWARD
#define CUFFT_SUCCESS                       HIPFFT_SUCCESS

// Version macros for conditional compilation
// HIP does not have a direct equivalent to CUDART_VERSION for runtime
// Define CUDART_VERSION to indicate CUDA 13+ equivalent for HIP
// This selects the modern surface object path and CUDA 13+ attribute queries
#ifndef CUDART_VERSION
#define CUDART_VERSION 13000
#endif
#ifndef __CUDACC_VER_MAJOR__
#define __CUDACC_VER_MAJOR__ 13
#endif

// helper_cuda.h replacement: checkCudaErrors macro
// On HIP, we provide simple error checking macros for both HIP runtime and hipFFT
#include <cstdio>

#ifndef checkCudaErrors
inline void _checkHipError(hipError_t err, const char* file, int line) {
    if (err != hipSuccess) {
        fprintf(stderr, "HIP error at %s:%d: %s\n", file, line, hipGetErrorString(err));
    }
}
inline void _checkHipfftError(hipfftResult err, const char* file, int line) {
    if (err != HIPFFT_SUCCESS) {
        fprintf(stderr, "hipFFT error at %s:%d: %d\n", file, line, (int)err);
    }
}
// Use overloaded functions via a helper to dispatch based on return type
template<typename T> struct HipErrorChecker;
template<> struct HipErrorChecker<hipError_t> {
    static void check(hipError_t err, const char* file, int line) { _checkHipError(err, file, line); }
};
template<> struct HipErrorChecker<hipfftResult> {
    static void check(hipfftResult err, const char* file, int line) { _checkHipfftError(err, file, line); }
};
#define checkCudaErrors(call) do { \
    auto _err = call; \
    HipErrorChecker<decltype(_err)>::check(_err, __FILE__, __LINE__); \
} while(0)
#endif

#else // CUDA path

#include <cuda_runtime.h>
#include <cufft.h>

#endif // USE_HIP

#endif // CUDA_TO_HIP_H
