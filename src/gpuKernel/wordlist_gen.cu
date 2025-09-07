// Copyright 2025 Hakil
// Licensed under the Apache License, Version 2.0

// -- code -- //
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cstring>

extern "C" __global__ void gen_word(char* chars, int charsLen, unsigned long long* range, int n, char* output){
    
    int i =  blockIdx.x * blockDim.x + threadIdx.x; // calculate the ID of the single thread
    
    if (i < n) {    
        int outpOffset = 0;
        for(int prevthread = 0; prevthread < i; prevthread++) { // calculating the offset of the words with the thread ID
            int prevnumstrings = range[(prevthread*3) + 2] - range[(prevthread*3) + 1] + 1;
            int prevstringlen = range[prevthread*3];
            outpOffset += prevnumstrings * (prevstringlen + 1);
        }

        for(int j = range[(i*3)+1]; j <= range[(i*3)+2]; j++) { // iterate for every seed
            int current_length = range[i*3];
            int temp = j;

            for(int h = 0; h < current_length; h++) {// creating the word with the seed

                output[outpOffset + h] = chars[temp % charsLen];
                temp /= charsLen;
            
            }
            outpOffset += current_length + 1;
        }
    }
}
