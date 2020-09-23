

#include "common.h"
#include <crt/math_functions.h>

#ifndef _DEPTH_TO_LIT_SHADOW_KERNEL_H_
#define _DEPTH_TO_LIT_SHADOW_KERNEL_H_

__device__ const float4 kDecodeDot = { 1.0f, 1.0f / 255.0f, 1.0f / 65025.0f, 1.0f / 16581375.0f };

__device__  float DotAsm(float4 l, float4 r) {
	float d = 0;
	asm(".reg .f32 lenL; \n\t"
		".reg .f32 lenR; \n\t"
		".reg .f32 cosRL; \n\t"
		".reg .f32 mulD; \n\t"
		"mov.f32 lenL, 0.0; \n\t"
		"mov.f32 lenR, 0.0; \n\t"
		"mov.f32 cosRL, 0.0; \n\t"
		"mad.rn.f32 lenL, %1, %1, lenL; \n\t"
		"mad.rn.f32 lenL, %2, %2, lenL;\n\t"
		"mad.rn.f32 lenL, %3, %3, lenL;\n\t"
		"mad.rn.f32 lenL, %4, %4, lenL;\n\t"
		"mad.rn.f32 lenR, %5, %5, lenR;\n\t"
		"mad.rn.f32 lenR, %6, %6, lenR;\n\t"
		"mad.rn.f32 lenR, %7, %7, lenR;\n\t"
		"mad.rn.f32 lenR, %8, %8, lenR;\n\t"
		"mad.rn.f32 cosRL, %1, %5, cosRL;\n\t"
		"mad.rn.f32 cosRL, %2, %6, cosRL;\n\t"
		"mad.rn.f32 cosRL, %3, %7, cosRL;\n\t"
		"mad.rn.f32 cosRL, %4, %8, cosRL;\n\t"
		"sqrt.approx.f32 lenL, lenL;\n\t"
		"sqrt.approx.f32 lenR, lenR;\n\t"
		"mul.rn.f32 lenL, lenL, lenR;\n\t"
		"div.rn.f32 cosRL, cosRL, lenL;\n\t"
		"mul.rn.f32 %0, cosRL, lenL;"
		: "=f"(d) : "f"(l.x) , "f"(l.y) , "f"(l.z) , "f"(l.w) , "f"(r.x) , "f"(r.y) , "f"(r.z) , "f"(r.w));
	
	return d;
}

__device__ __forceinline__ float Dot(float4 l, float4 r) {
	return DotAsm(l, r);
}

__device__ __forceinline__ float DecodeFloatRGBA(float4 enc)
{
    float depth = Dot(enc, kDecodeDot);
    return depth;
}

// scaler = 8192 / 4096  1BYTE/Voxel  ultra: 2Bit / Voxel
__global__ void Depth2LitShadowKernel(unsigned char* g_data, unsigned int* g_dataSrc, float frontDepth, float backDepth, unsigned int size, unsigned int scaler, bool perferShadow) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * size + x;
    int xSrc = x * scaler;
    int ySrc = y * scaler;
    bool isLit = true;
    bool isShadow = true;
    bool isIntersected = true;
    int originSize = size * scaler;

    //float depthAvg = 0;
    for (int v = 0; v < scaler; v++) {
        for (int u = 0; u < scaler; u++) {
            int vSrc = ySrc + v;
            int uSrc = xSrc + u;
            int indexSrc = vSrc * originSize + uSrc;
            unsigned char* src = (unsigned char*)(g_dataSrc + indexSrc);
            // BYTE* srcChannel = (BYTE*)src;
            float4 rgba = { src[0] * 0.00392156f, src[1] * 0.00392156f, src[2] * 0.00392156f,src[3] * 0.00392156f };

            float depth = Dot(rgba, kDecodeDot);
            //depthAvg += depth;
			isShadow &= depth > frontDepth;
			isLit &= depth < backDepth;
			//isIntersected &= depth >= backDepth && depth <= frontDepth;
        }
    }
    isIntersected = !isLit & !isShadow;

	/*g_data[index] = (unsigned char)(isIntersected ? 128 : (isLit ? 255 : 0));*/
    
    if ((perferShadow && !isShadow) || (!perferShadow && !isLit))
    {
        g_data[index] = (unsigned char)(isIntersected ? 128 : (perferShadow? 255 : 0));
    }


}


// scaler = 8192 / 4096  1BYTE/Voxel  ultra: 2Bit / Voxel
// perferShadow:assume most voxel is shadow(targetTex Default state is Shadow)
__global__ void Depth2LitShadow8SlicePerBatchKernel(unsigned char* g_data, unsigned int* g_dataSrc, float frontDepth, float backDepth, unsigned int size, unsigned int scaler, bool perferShadow) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float4 kDecodeDot = float4();
	kDecodeDot.x = 1.0f;
	kDecodeDot.y = 1 / 255.0f;
	kDecodeDot.z = 1 / 65025.0f;
	kDecodeDot.w = 1 / 16581375.0f;
	int originSize = size * scaler;
	int xSrc = x * scaler;
	int ySrc = y * scaler;

	for (int d = 0; d < 8; d++) {
        int index = d * size * size + y * size + x;
		bool isLit = true;
		bool isShadow = true;
		bool isIntersected = true;
		for (int v = 0; v < scaler; v++) {
			for (int u = 0; u < scaler; u++) {
				int vSrc = ySrc + v;
				int uSrc = xSrc + u;
				int indexSrc = vSrc * originSize + uSrc;
				unsigned char* src = (unsigned char*)(g_dataSrc + indexSrc);
				// BYTE* srcChannel = (BYTE*)src;
				float4 rgba;
				rgba.x = src[0] * 0.00392156f;
				rgba.y = src[1] * 0.00392156f;
				rgba.z = src[2] * 0.00392156f;
				rgba.w = src[3] * 0.00392156f;

				float depth = 1;// Dot(rgba, kDecodeDot);

				isLit &= depth > frontDepth;
				isShadow &= depth < backDepth;
				//isIntersected &= depth >= backDepth && depth <= frontDepth;
			}
		}

		isIntersected = !isLit & !isShadow;
		if ((perferShadow && !isShadow) || (!perferShadow && !isLit))
		{
			g_data[index] = isIntersected ? 128 : 0;
		}

	}
}

#endif