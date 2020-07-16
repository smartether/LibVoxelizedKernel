#include "common.h"
#ifndef _MY_CUDA_DEFINE_H_
#define _MY_CUDA_DEFINE_H_
// g_data 2048*2048   g_dataSrc 4096 Texture2DArray subArray
__global__ void voxelizedSample_kernel(unsigned char* g_data, unsigned char* g_dataSrc, unsigned int size, unsigned int scaler)
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
            unsigned char src = g_dataSrc[indexSrc];
            isLit &= abs((int)src - 255) < 20;
            isShadow &= (int)src < 20;
        }
    }
    bool isIntersected = !isLit && !isShadow;
    g_data[index] = (unsigned char)((isLit ? 255 : 0) + (isIntersected ? 128 : 0));


}



#endif