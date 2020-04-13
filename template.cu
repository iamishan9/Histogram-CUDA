////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/* Template project which demonstrates the basics on how to setup a project
* example application.
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples


// include kernel file
// #include "kernels.cu"




// // number of ascii characters in the histogram: 70 || 96
// #define CHARCOUNT 90

// // number of elements per thread; 128
// #define THREADWORK 128

// //number of threads per block: 32
// #define BLOCKSIZE 32


// // gridsize == ??
// #define GRIDSIZE 

#define NB_THREADS 1024
#define NB_BLOCKS 10
#define TOTAL_THREAD NB_BLOCKS*NB_THREADS
#define NB_CHARS 70





using namespace std;

vector <char> readFile(string inputFile){
    char ch;    
    
    vector <char> vv;

    fstream fin(inputFile, fstream::in);

    
    while (fin >> noskipws >> ch) {
        // cout << ch << "\n";
        if(int(ch)>31 && int(ch)<127) 
            vv.push_back(ch);
    }

    return vv;
}






///////////////////////////////////////////////
////////////  GPU PROGRAM  ////////////////////
///////////////////////////////////////////////
__global__ void genHistogram(char *dict, unsigned int ){



}






///////////////////////////////////////////////
////////////  MAIN PROGRAM  ///////////////////
///////////////////////////////////////////////


int main(int argc, char **argv){
    
    // get input and output file names    
    char *inputFile = argv[1]; 
    char *outputFile = argv[2];
    
    vector<char> charDict = readFile(inputFile);

    unsigned int COUNTCHAR = charDict.size();

    // for(int i=0; i<charDict.size(); i++){
    //  cout<<charDict[i];
    // }

    // For HOST
    char *allChars;
    unsigned int *charCount; // total num of characters



    // Allocation of host copy
    allChars = (char*)malloc(COUNTCHAR*sizeof(char));
    charCount = (unsigned int*)malloc(NB_CHARS*sizeof(unsigned int));


    // For DEVICE
    char *devAllChars;
    unsigned int *devCharCount;

    // Allocation of device copy
    cudaMalloc((void**)&devAllChars, COUNTCHAR*sizeof(char));
    cudaMalloc((void**)&devCharCount, NB_CHARS*sizeof(unsigned int));


    // copy the host data to device
    cudaMemcpy(devAllChars, allChars, COUNTCHAR*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemset(devCharCount, 0, NB_CHARS*sizeof(unsigned int));

    // Call the kernel to generate the histogram 
    genHistogram <<< NB_BLOCKS, NB_THREADS >>>(devAllChars, devCharCount, COUNTCHAR);

    // copy device result back to host copy
    cudaMemcpy(charCount, devCharCount, COUNTCHAR*sizeof(unsigned int), cudaMemcpyHostToDevice);


    // write the results to output file
    writeResult(outputFile, charCount);


    // free device momory
    cudaFree(devCharCount);
    cudaFree(devAllChars); 

    // free host memory
    free(allChars);
    free(charCount);

}