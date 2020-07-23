#include "common.h"
#include <crt/math_functions.h>

#ifndef _LIT_SHADOW_DISTRIBUTION_KERNEL_H
#define _LIT_SHADOW_DISTRIBUTION_KERNEL_H



__global__ void LitShadowDistributionKernel(int targetVoxelSize, int originVoxelSize, int dBlockIndex, unsigned char * targetLitShadowInfoArray,
    CompressedLitInfo* originLitShadowInfoArray, int kernelSize = 2) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

    int dPixelBase = kernelSize * (int)dBlockIndex;
	unsigned char* lv1BlockPixels = (unsigned char*)targetLitShadowInfoArray;// +dBlockIndex * targetVoxelSize * targetVoxelSize;
    int vBlockIndex = y;
	int uBlockIndex = x;

	int uPixelBase = kernelSize * uBlockIndex;
	int vPixelBase = kernelSize * vBlockIndex;
	// voxel : 2*2*2 lv3
	bool isLayerAllLit = true;
	bool isLayerAllShadow = true;
	bool isLayerAllIntersected = true;
	for (int dPixelSub = 0, dPixelMax = kernelSize; dPixelSub < dPixelMax; dPixelSub++)
	{
		for (int vPixelSub = 0, vPixelMax = kernelSize; vPixelSub < vPixelMax; vPixelSub++)
		{
			for (int uPixelSub = 0, uPixelMax = kernelSize; uPixelSub < uPixelMax; uPixelSub++)
			{
				int dPixel = dPixelBase + dPixelSub;
				int vPixel = vPixelBase + vPixelSub;
				int uPixel = uPixelBase + uPixelSub;
				int srcIndex = vPixel * (int)originVoxelSize + uPixel;
				CompressedLitInfo originLitShadowInfo = originLitShadowInfoArray[srcIndex];
				isLayerAllLit &= originLitShadowInfo.litEndVoxelId >= dPixel;
				isLayerAllShadow &= originLitShadowInfo.shadowStartVoxelId <= dPixel;
				//isLayerAllIntersected &= originLitShadowInfo.litEndVoxelId < dPixel&&
				//	originLitShadowInfo.shadowStartVoxelId > dPixel;
			}
		}
	}
	bool isBlockIntersection = !isLayerAllLit && !isLayerAllShadow;
	float blockResult = (isLayerAllLit ? 1 : 0) + (isBlockIntersection ? 0.5f : 0);
	lv1BlockPixels[(long)vBlockIndex * (long)targetVoxelSize + (long)uBlockIndex] = (byte)lroundf(blockResult * 255);


}





















#endif
