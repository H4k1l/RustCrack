// Copyright 2025 Hakil
// Licensed under the Apache License, Version 2.0

// -- code -- //
#include <sstream>
#include <iomanip>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>

extern "C" __global__ void genWord(char* chars, int charsLen, unsigned long long* range, int n, char* output){
    int i =  blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {    
        int outpOffset = 0;
        for(int prevthread = 0; prevthread < i; prevthread++) {
            int prevnumstrings = range[(prevthread*3) + 2] - range[(prevthread*3) + 1] + 1;
            int prevstringlen = range[prevthread*3];
            outpOffset += prevnumstrings * (prevstringlen + 1);
        }

        for(int j = range[(i*3)+1]; j <= range[(i*3)+2]; j++) { 
            int currentLength = range[i*3];
            int temp = j;

            for(int h = 0; h < currentLength; h++) {
                output[outpOffset + h] = chars[temp % charsLen];
                temp /= charsLen;
            }
            outpOffset += currentLength + 1;
        }
    }
}
