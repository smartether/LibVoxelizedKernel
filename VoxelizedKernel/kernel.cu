
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include "cuda.h"
#include "helper_cuda.h"
#include "helper_functions.h"
#include "math_functions.h"
#include "cublas.h"
#include <thread>
#include <xthreads.h>
#include "MyCudaDefine.cuh"

#define DLLEXPORT __declspec(dllexport)

//
//float DecodeFloatRGBA(float4 enc)
//{
//    float4 kDecodeDot = float4();
//    kDecodeDot.x = 1.0f;
//    kDecodeDot.y = 1 / 255.0;
//    kDecodeDot.z = 1 / 65025.0;
//    kDecodeDot.w = 1 / 16581375.0;
//
//    float depth = cublasSdot(1, (float*)&enc, 4, (float*)&kDecodeDot, 4);
//    return depth;
//}

// future
__global__ void voxelizedDiff(BYTE* g_data, unsigned int* g_dataSrc, float frontDepth, float backDepth, unsigned int size, unsigned int scaler) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int xSrc = x * scaler;
    int ySrc = y * scaler;
    int index = y * size + x;
    bool isLit = true;
    bool isShadow = true;
    int originSize = size * scaler;
    for (int v = 0; v < scaler; v++) {
        for (int u = 0; u < scaler; u++) {
            int vSrc = ySrc + v;
            int uSrc = xSrc + u;
            int indexSrc = vSrc * originSize + uSrc;
            unsigned int* src = g_dataSrc + indexSrc;
            BYTE* srcChannel = (BYTE*)src;
            float4 rgba;
            rgba.x = src[0] / 255.0f;
            rgba.y = src[1] / 255.0f;
            rgba.z = src[2] / 255.0f;
            rgba.w = src[3] / 255.0f;
            float4 kDecodeDot = float4();
            kDecodeDot.x = 1.0f;
            kDecodeDot.y = 1 / 255.0;
            kDecodeDot.z = 1 / 65025.0;
            kDecodeDot.w = 1 / 16581375.0;

            float depth = rgba.x * kDecodeDot.x + rgba.y * kDecodeDot.y + rgba.z * kDecodeDot.z + rgba.w * kDecodeDot.w; //cublasSdot(4, (float*)&rgba, 0, (float*)&kDecodeDot, 0);
            isLit &= abs((int)src - 255) < 20;
            isShadow &= (int)src < 20;
        }
    }
}

// blockSize = 64 * 64   threadDim(32 * 32,1)= (1024,1)= (32,32)
__global__ void voxelizedSample(BYTE* src, BYTE* dst, unsigned int lv4VoxelSize, unsigned int lv4PixelPerVoxel) {

    int uPixelTarget = threadIdx.x;
    int vPixelTarget = threadIdx.x;
}

const int SCALER = 2;

// g_data 2048*2048   g_dataSrc 4096 Texture2DArray subArray
__global__ void voxelizedSample_kernel(BYTE* g_data, BYTE* g_dataSrc, unsigned int size , unsigned int scaler)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int xSrc = x * scaler;
    int ySrc = y * scaler;
    int index = y * size + x;
    bool isLit = true;
    bool isShadow = true;
    
    int originSize = size * scaler;
    for (int v = 0; v < scaler; v++) {
        for (int u = 0; u < scaler; u++) {
            int vSrc = ySrc + v;
            int uSrc = xSrc + u;
            int indexSrc = vSrc * originSize + uSrc;
            BYTE src = g_dataSrc[indexSrc];
            isLit &= abs((int)src - 255) < 20;
            isShadow &= (int)src < 20;
        }
    }
    bool isIntersected = !isLit && !isShadow;
    g_data[index] = (BYTE)((isLit ? 255 : 0) + (isIntersected ? 128 : 0));
    
    
}

// Lv4 summary to lv3 support 2048 to 256
__global__ void voxelizedSample3D_kernel(BYTE* g_data, BYTE* g_dataSrc, unsigned int size, unsigned int scaler) {
    g_data[threadIdx.x] = 0;
}


#define CHECK_ERR(status)   \
    if (cudaStatus != status) {    \
        fprintf(stderr, "cudaSetDevice failed!  errorCode: %i ?", (int)status); \
        goto Error; \
    }   


static bool memPoolEnabled;
// target device memory pool
static std::vector<BYTE*>* poolTarget;

// origin device memory pool
static std::vector<BYTE*>* poolOrigin;

static int TARGET_BUFFER_POOL_SIZE = 16;
static int ORIGIN_BUFFER_POOL_SIZE = 64;

static int targetBufferInUse = 0;
static int targetBufferFree = 0;
static int originBufferInUse = 0;
static int originBufferFree = 0;
static _Mtx_t bufferMtx;

static int nThreadNum = 16;

void lockBufferMtx() {
    _Mtx_lock(bufferMtx);
}
void unlockBufferMtx() {
    _Mtx_unlock(bufferMtx);
}
bool hasTargetBuff() {
    lockBufferMtx();
    bool b = targetBufferFree > 0;
    unlockBufferMtx();
    return b;
}

bool hasOriginBuff() {
    lockBufferMtx();
    bool b = originBufferFree > 0;
    unlockBufferMtx();
    return b;
}

BYTE* getTargetBuffer(BYTE** ptr) {
    lockBufferMtx();
    if (targetBufferFree > 0) {
        *ptr = *poolTarget->begin();
        poolTarget->erase(poolTarget->begin());
        targetBufferInUse++;
        targetBufferFree--;
    }
    unlockBufferMtx();
    return *ptr;
}
void reclaimTargetBuffer(BYTE** ptr) {
    lockBufferMtx();
    poolTarget->push_back(*ptr);
    targetBufferInUse--;
    targetBufferFree++;
    unlockBufferMtx();
}
void reclaimOriginBuffer(BYTE** ptr) {
    lockBufferMtx();
    poolOrigin->push_back(*ptr);
    originBufferInUse--;
    originBufferFree++;
    unlockBufferMtx();
}
BYTE* getOriginBuffer(BYTE** ptr) {
    lockBufferMtx();
    if (originBufferFree > 0) {
        *ptr = *poolOrigin->begin();
        poolOrigin->erase(poolOrigin->begin());
        originBufferInUse++;
        originBufferFree--;
    }
    unlockBufferMtx();
    return *ptr;
}

extern "C" {
    DLLEXPORT void Init(unsigned int targetBufferPoolSize, unsigned int originBufferPoolSize,  unsigned int targetSize, unsigned int scaler = SCALER, unsigned int threadNum = 16) {
        _Mtx_init(&bufferMtx, 0);
        TARGET_BUFFER_POOL_SIZE = targetBufferPoolSize;
        ORIGIN_BUFFER_POOL_SIZE = originBufferPoolSize;
        poolTarget = new std::vector<BYTE*>();
        poolOrigin = new std::vector<BYTE*>();
        for (int i = 0; i < targetBufferPoolSize; i++) {
            BYTE* targetTex;
            cudaMalloc<BYTE>(&targetTex, targetSize * targetSize);
            poolTarget->push_back(targetTex);
            
        }
        for (int i = 0; i < originBufferPoolSize; i++) {
            BYTE* originTex;
            cudaMalloc<BYTE>(&originTex, targetSize * targetSize * scaler * scaler);
            poolOrigin->push_back(originTex);
        }
        targetBufferFree = targetBufferPoolSize;
        originBufferFree = originBufferPoolSize;
        nThreadNum = threadNum;
        memPoolEnabled = true;
    }

    DLLEXPORT void Close() {
        for (auto i = poolTarget->begin(), c = poolTarget->end(); i != c; i++) {
            auto value = *i;
            cudaFree(value);
        }
        for (auto i = poolOrigin->begin(), c = poolOrigin->end(); i != c; i++) {
            auto value = *i;
            cudaFree(value);
        }
    }

    DLLEXPORT cudaError_t Downsample(BYTE* targetTex, BYTE* originTex, unsigned int targetSize, unsigned int scaler = SCALER) {
        BYTE* dev_targetTex;
        BYTE* dev_originTex;
        
        cudaError_t cudaStatus;
        cudaStatus = cudaSetDevice(0);
        CHECK_ERR(cudaStatus);
        if (memPoolEnabled) {
            while (!hasTargetBuff())
                Sleep(30);
            getTargetBuffer(&dev_targetTex);
            while (!hasOriginBuff())
                Sleep(30);
            getOriginBuffer(&dev_originTex);
            
        }
        else {
            cudaStatus = cudaMalloc<BYTE>(&dev_targetTex, targetSize * targetSize);
            CHECK_ERR(cudaStatus);
            cudaStatus = cudaMalloc<BYTE>(&dev_originTex, targetSize * targetSize * scaler * scaler);
            CHECK_ERR(cudaStatus);
        }
        

        unsigned int threadNum = min(nThreadNum, targetSize);
        dim3 threads = dim3(threadNum, threadNum);
        dim3 blocks = dim3(targetSize / threads.x, targetSize / threads.y);

        // create cuda event handles
        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));

        cudaStream_t streamId;
        cudaStreamCreate(&streamId);

        StopWatchInterface* timer = NULL;
        sdkCreateTimer(&timer);
        sdkResetTimer(&timer);
        checkCudaErrors(cudaDeviceSynchronize());
        float gpu_time = 0.0f;
        // asynchronously issue work to the GPU (all to stream 0)
        sdkStartTimer(&timer);
        cudaEventRecord(start, streamId);

        cudaStatus = cudaMemcpyAsync(dev_originTex, originTex, targetSize * targetSize * scaler * scaler, cudaMemcpyHostToDevice, streamId);

        CHECK_ERR(cudaStatus);
        voxelizedSample_kernel << <blocks, threads, 0, streamId >> > (dev_targetTex, dev_originTex, targetSize, scaler);
        CHECK_ERR(cudaStatus);

        cudaStatus = cudaMemcpyAsync(targetTex, dev_targetTex, targetSize * targetSize, cudaMemcpyDeviceToHost, streamId);
        cudaEventRecord(stop, streamId);
        sdkStopTimer(&timer);

        //cudaStatus = cudaDeviceSynchronize();
        
        // have CPU do some work while waiting for stage 1 to finish
        unsigned long int counter = 0;

        //while (cudaEventQuery(stop) == cudaErrorNotReady)
        //{
        //    counter++;
        //}
        //cudaStatus = cudaEventSynchronize(stop);
        
        while (cudaStreamQuery(streamId) != cudaSuccess) {
            counter++;
        }
        // cudaStreamWaitEvent(streamId, stop, 0);
        // cudaStreamSynchronize(streamId);
        cudaStreamDestroy(streamId);
        checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

        // print the cpu and gpu times
        printf("time spent executing by the GPU: %.2f\n", gpu_time);
        printf("time spent by CPU in CUDA calls: %.2f\n", sdkGetTimerValue(&timer));
        printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);

        // release resources
        checkCudaErrors(cudaEventDestroy(start));
        checkCudaErrors(cudaEventDestroy(stop));

        if (memPoolEnabled) {
            reclaimTargetBuffer(&dev_targetTex);
            reclaimOriginBuffer(&dev_originTex);
        }
        else {
            cudaFree(dev_targetTex);
            cudaFree(dev_originTex);
        }
        return cudaStatus;


    Error:
        if (memPoolEnabled) {
            reclaimTargetBuffer(&dev_targetTex);
            reclaimOriginBuffer(&dev_originTex);
        }
        else {
            cudaFree(dev_targetTex);
            cudaFree(dev_originTex);
        }
        return cudaStatus;
    }

    DLLEXPORT cudaError_t StripRedundancyInfo() {
        
        return cudaSuccess;
    }

    DLLEXPORT void* AllocMem(size_t size) {
        BYTE* ptr = (BYTE*) malloc(size);
        //cudaMallocManaged<BYTE>(&ptr, size);
        return ptr;

    }

    DLLEXPORT void* ReallocMem(void* ptr, size_t size) {
        BYTE* ptr1 = (BYTE*)realloc(ptr, size);
        return ptr1;
    }

    DLLEXPORT void FreeMem(void* ptr) {
        free(ptr);
    }

    DLLEXPORT BYTE* GetSubArray(BYTE* ptr, INT64 start, INT64 length) {
        return ptr + start;
    }

}

void printDeviceInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d.\n",
            device, deviceProp.major, deviceProp.minor);
    }
}

int main1(){

    printDeviceInfo();

    int targetSize = 4096;
    BYTE* data = (BYTE*)malloc(targetSize * targetSize);
    //cudaMallocHost(&data, targetSize * targetSize);
    BYTE* originData = (BYTE*)malloc(targetSize * targetSize * 4);
    //cudaMallocHost(&originData, targetSize * targetSize * 4);
    //memset(data, (BYTE)0, targetSize * targetSize);
    int memSetValue = 0;
    BYTE oneByte[4] = { 255,255,255,255 };
    memcpy(&memSetValue, &oneByte, 4);
    memset(data, memSetValue,  1024 * 1024 / 4);
    memset(originData, memSetValue, targetSize * targetSize / 4);
    Init(8, 16, targetSize, 2);
    for (int i = 0; i < 512; i++) {
        Downsample(data, originData, targetSize, 2);
    }
    Close();
    for (int i = 0; i < 64; i++) {
        BYTE a = data[i * 64];
        printf("$$ a: %i", a);
    }
    return 0;
}
