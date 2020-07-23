
#include "common.h"

#include "MyCudaDefine.cuh"
#include "Depth2LitShadowKernel.cuh"
#include "LitShadowDistributionKernel.cuh"

static BYTE* gDev_depthTex = nullptr;

static BYTE* gDev_distributionTex = nullptr;

extern "C" {
 

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


    DLLEXPORT void BindDepthTex(BYTE* depthTex, unsigned int size, unsigned int scaler) {
        cudaSetDevice(0);
        cudaMalloc<BYTE>(&gDev_depthTex, size * size * scaler * scaler * 4);
        cudaMemcpy(gDev_depthTex, depthTex, size * size * scaler * scaler * 4, cudaMemcpyKind::cudaMemcpyHostToDevice);

    }

    DLLEXPORT void BindDistributionTex(unsigned char* distTex, unsigned int size, unsigned int scaler) {
        cudaSetDevice(0);
        cudaMalloc<unsigned char>(&gDev_distributionTex, size * size * scaler * scaler * 4);
        cudaMemcpy(gDev_distributionTex, distTex, size * size * scaler * scaler * 4, cudaMemcpyKind::cudaMemcpyHostToDevice);
    }

    DLLEXPORT void ReleaseDistributionTex() {
        cudaFree(gDev_distributionTex);
        gDev_distributionTex = nullptr;
    }

    DLLEXPORT void ReleaseDepth2LitShadowProceduce() {
        cudaFree(gDev_depthTex);
        gDev_depthTex = nullptr;
    }

    // size:texture width
    DLLEXPORT cudaError_t Depth2LitShadowMinBatch(unsigned char* g_data, unsigned int* g_dataSrc, float frontDepth, float backDepth, unsigned int size, unsigned int scaler, bool perferShadow) {
        BYTE* dev_depthTex;
        BYTE* dev_litShadowTex;
        cudaError_t cudaStatus;
        cudaStatus = cudaSetDevice(0);
        CHECK_ERR(cudaStatus);
        if (gDev_depthTex == nullptr) {
            BindDepthTex((BYTE*)g_dataSrc, size, scaler);
        }
        dev_depthTex = gDev_depthTex;
        if (memPoolEnabled) {
            //while (!hasTargetBuff())
            //    Sleep(30);
            //getTargetBuffer(&dev_depthTex);
            while (!hasOriginBuff())
                Sleep(30);
            getOriginBuffer(&dev_litShadowTex);
        }
        else {
            //cudaStatus = cudaMalloc<BYTE>(&dev_depthTex, size * size * scaler * scaler * 4);
            //CHECK_ERR(cudaStatus);
            cudaStatus = cudaMalloc<BYTE>(&dev_litShadowTex, size * size);
            CHECK_ERR(cudaStatus);
        }
        unsigned int threadNum = min(nThreadNum, size);
        dim3 threads = dim3(threadNum, threadNum);
        dim3 blocks = dim3(size / threads.x, size / threads.y);
        cudaEvent_t start, stop;
        CHECK_ERR(cudaEventCreate(&start));
        CHECK_ERR(cudaEventCreate(&stop));

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

        if(perferShadow)
            cudaMemsetAsync(dev_litShadowTex, 0, size * size, streamId);
        else
            cudaMemsetAsync(dev_litShadowTex, 255, size * size, streamId);
        //cudaStatus = cudaMemcpyAsync(dev_depthTex, g_dataSrc, size * 4, cudaMemcpyKind::cudaMemcpyHostToDevice, streamId);

        CHECK_ERR(cudaStatus);
        Depth2LitShadowKernel << <blocks, threads, 0, streamId >> > (dev_litShadowTex, (unsigned int*)dev_depthTex, frontDepth, backDepth, size, scaler, perferShadow);
        CHECK_ERR(cudaStatus);

        cudaStatus = cudaMemcpyAsync(g_data, dev_litShadowTex, size * size, cudaMemcpyKind::cudaMemcpyDeviceToHost, streamId);

        CHECK_ERR(cudaStatus);
        cudaEventRecord(stop, streamId);
        sdkStopTimer(&timer);
        while (cudaStreamQuery(streamId) != cudaSuccess) {
            
        }
        cudaStreamDestroy(streamId);
        checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

        // release resources
        checkCudaErrors(cudaEventDestroy(start));
        checkCudaErrors(cudaEventDestroy(stop));

        if (memPoolEnabled) {
            //reclaimTargetBuffer(&dev_depthTex);
            reclaimOriginBuffer(&dev_litShadowTex);
        }
        else {
            //cudaFree(dev_depthTex);
            cudaFree(dev_litShadowTex);
        }
    Error:
        if (memPoolEnabled) {
            //reclaimTargetBuffer(&dev_depthTex);
            reclaimOriginBuffer(&dev_litShadowTex);
        }
        else {
            //cudaFree(dev_depthTex);
            cudaFree(dev_litShadowTex);
        }
        return cudaError::cudaSuccess;
    }

    DLLEXPORT cudaError_t ConvertDistributionTex2LitShadowInfo(int targetVoxelSize, int originVoxelSize, int dBlockIndex, byte* targetLitShadowInfoArray,
        CompressedLitInfo* originLitShadowInfoArray, int kernelSize = 2) {
        unsigned char* dev_distributionTex;
        unsigned char* dev_litShadowTex;
        cudaError_t cudaStatus;
        cudaStatus = cudaSetDevice(0);
        CHECK_ERR(cudaStatus);
        if (gDev_distributionTex == nullptr) {
            BindDistributionTex((unsigned char*)originLitShadowInfoArray, targetVoxelSize, kernelSize);
        }
        dev_distributionTex = gDev_distributionTex;
        if (memPoolEnabled) {
            //while (!hasTargetBuff())
            //    Sleep(30);
            //getTargetBuffer(&dev_depthTex);
            while (!hasOriginBuff())
                Sleep(30);
            getOriginBuffer(&dev_litShadowTex);
        }
        else {
            //cudaStatus = cudaMalloc<BYTE>(&dev_depthTex, size * size * scaler * scaler * 4);
            //CHECK_ERR(cudaStatus);
            cudaStatus = cudaMalloc<unsigned char>(&dev_litShadowTex, targetVoxelSize * targetVoxelSize);
            CHECK_ERR(cudaStatus);
        }
        unsigned int threadNum = min(nThreadNum, targetVoxelSize);
        dim3 threads = dim3(threadNum, threadNum);
        dim3 blocks = dim3(targetVoxelSize / threads.x, targetVoxelSize / threads.y);
        cudaEvent_t start, stop;
        CHECK_ERR(cudaEventCreate(&start));
        CHECK_ERR(cudaEventCreate(&stop));

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


        CHECK_ERR(cudaStatus);
        //for (int dblockIdx = 0; dblockIdx < targetVoxelSize; dblockIdx++) {
            LitShadowDistributionKernel <<<blocks, threads, 0, streamId>>> (targetVoxelSize, originVoxelSize, dBlockIndex, dev_litShadowTex, (CompressedLitInfo*) dev_distributionTex, kernelSize);
        //}
        CHECK_ERR(cudaStatus);

        cudaStatus = cudaMemcpyAsync(targetLitShadowInfoArray, dev_litShadowTex, targetVoxelSize* targetVoxelSize, cudaMemcpyKind::cudaMemcpyDeviceToHost, streamId);

        CHECK_ERR(cudaStatus);
        cudaEventRecord(stop, streamId);
        sdkStopTimer(&timer);
        cudaError_t ev;
        while ((ev = cudaStreamQuery(streamId)) != cudaSuccess) {
            
        }
        cudaStreamDestroy(streamId);
        checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

        // release resources
        checkCudaErrors(cudaEventDestroy(start));
        checkCudaErrors(cudaEventDestroy(stop));

        if (memPoolEnabled) {
            //reclaimTargetBuffer(&dev_depthTex);
            reclaimOriginBuffer(&dev_litShadowTex);
        }
        else {
            //cudaFree(dev_depthTex);
            cudaFree(dev_litShadowTex);
        }
    Error:
        if (memPoolEnabled) {
            //reclaimTargetBuffer(&dev_depthTex);
            reclaimOriginBuffer(&dev_litShadowTex);
        }
        else {
            //cudaFree(dev_depthTex);
            cudaFree(dev_litShadowTex);
        }
        return cudaError::cudaSuccess;
    }

    DLLEXPORT cudaError_t StripRedundancyInfo() {
        
        return cudaSuccess;
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
