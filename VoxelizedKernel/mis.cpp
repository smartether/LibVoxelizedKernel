
#ifndef DLLEXPORT 
#define DLLEXPORT __declspec(dllexport)
#endif

#include <stdio.h>
#include <cstdio>
#include <iostream>

extern "C" {

    DLLEXPORT void* AllocMem(size_t size) {
        char* ptr = (char*)malloc(size);
        //cudaMallocManaged<BYTE>(&ptr, size);
        return ptr;

    }

    DLLEXPORT void* ReallocMem(void* ptr, size_t size) {
        char* ptr1 = (char*)realloc(ptr, size);
        return ptr1;
    }

    DLLEXPORT void FreeMem(void* ptr) {
        free(ptr);
    }

    DLLEXPORT char* GetSubArray(char* ptr, long start, long length) {
        return ptr + start;
    }


}

