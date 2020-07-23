
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

//#pragma once
#ifndef __COMMON_H
#define __COMMON_H

#define DLLEXPORT __declspec(dllexport)


typedef unsigned char BYTE;

const int SCALER = 2;

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


typedef struct {
    unsigned short litEndVoxelId;
    unsigned short shadowStartVoxelId;
}CompressedLitInfo;

extern "C" {

    DLLEXPORT void Init(unsigned int targetBufferPoolSize, unsigned int originBufferPoolSize, unsigned int targetSize, unsigned int scaler = SCALER, unsigned int threadNum = 16) {
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
        delete poolTarget;
        for (auto i = poolOrigin->begin(), c = poolOrigin->end(); i != c; i++) {
            auto value = *i;
            cudaFree(value);
        }
        delete poolOrigin;
    }

}

#endif