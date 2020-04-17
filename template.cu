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
#define NB_CHARS 69





using namespace std;

vector <char> readFile(string inputFile){
    char ch;    
    
    vector <char> vv;

    fstream fin(inputFile, fstream::in);

    
    while (fin >> noskipws >> ch) {
        // cout << ch << "\n";
        if(int(ch)>31 && int(ch)<127 && int(ch)!= 36 && int(ch) != 42) 
            vv.push_back(ch);
    }

    // char charArr[vv.size()];
    // copy(input.begin(), input.end(), charArr);

    return vv;
}


void writeResult(char *nameOutputfile, unsigned int *charCount){
    ofstream outputFile(nameOutputfile);

    // std::ofstream outputFile(nameOutputfile); // Open output file

    if(outputFile.is_open()){ // Test if the file is opened

        // whie(i <= 69)


        for(int i = 0 ; i <= 68 ; i++){
            // cout<<" val of I is "<<i<<endl;
            if(i != 4 && i!= 10){
                if(i+32<65){
                    outputFile << char(i + 32);
                }
                else{
                    outputFile << char(i + 58);
                }

                outputFile << " : ";
                // if(i == 5 || i == 43){
                //     outputFile << charCount[i-1];
                // }
                // else{
                outputFile << charCount[i];
                // }
                outputFile << "\n";
            }
        }
    }

    outputFile.close();
}


///////////////////////////////////////////////
////////////  CPU PROGRAM  ////////////////////
///////////////////////////////////////////////

// void cpu_histogram(vector<string> dict){

//     map<char,int> char_frequency;
//     for(int i=32;i<127;i++){
 
//         if(i<65 || i>90){
//         char_frequency[char(i)]=0;
//         }
//     }

//     for(unsigned int i=0;i<dict.size();i++){
//         for(char &c: dict[i]){
//             if(isupper(c)){
//                 c=c+32;

//             }
//             map<char, int>::iterator it = char_frequency.find(c); 
//             if (it != char_frequency.end())
//                 it->second +=1;

//             }
//     }


//     ofstream myfile;
//     myfile.open("example.csv");
//     for (auto const& x : char_frequency){
//         myfile<< x.first  // string (key)
//                   << ':' 
//                   << x.second // string's value 
//                   << std::endl ;
//     }

//     myfile.close();

// }



///////////////////////////////////////////////
////////////  GPU PROGRAM  ////////////////////
///////////////////////////////////////////////
__global__ void genHistogram(char *dict, unsigned int *arrCount, unsigned int charCount ){

    const unsigned int tidb = threadIdx.x;
    const unsigned int tid = blockIdx.x * blockDim.x + tidb;

    if(tid < charCount){

        for(int i=tid; i<charCount; i += TOTAL_THREAD){
            // if(int(dict[i]) ==65 || int(dict[i]) == 97){
            //     atomicAdd(&arrCount[0], 1);
            // }


            // before uppercase characters
            if(int(dict[i]) < 65){
                atomicAdd(&arrCount[int(dict[i]) - 32], 1);
            }

            // for uppercase only
            else if(int(dict[i]) >= 65 && dict[i] <= 90 ){
                atomicAdd(&arrCount[int(dict[i]) - 32 + 6], 1);

            }

            // after uppercase characters
            else if(dict[i] > 90){
                atomicAdd(&arrCount[int(dict[i]) - 58], 1);
            }
        }
    }


}


///////////////////////////////////////////////
////////////  GPU PROGRAM USING SHARED MEM ////
///////////////////////////////////////////////
__global__ void genHistogramShared(char *dict, unsigned int *arrCount, unsigned int charCount){
    // thread id inside thread block and inside grid
    const unsigned int tidb = threadIdx.x;
    const unsigned int tid = blockIdx.x * blockDim.x + tidb;

    __shared__ unsigned int sum_by_block[NB_CHARS]; // Store the sum of the number of car by blocks

    if(tid < charCount){ // Check if the number of the threads is less than the number of elements in the dictionnary

        for(int i=tid; i<charCount; i += TOTAL_THREAD){

            // before uppercase characters
            if(int(dict[i]) < 65){
                atomicAdd(&sum_by_block[int(dict[i]) - 32], 1);
            }

            // for uppercase only
            else if(int(dict[i]) >= 65 && dict[i] <= 90 ){
                atomicAdd(&sum_by_block[int(dict[i]) - 32 + 6], 1);

            }

            // after uppercase characters
            else if(dict[i]>90){
                atomicAdd(&sum_by_block[int(dict[i]) - 58], 1);
            }
        }
                // For no usage of the shared memory
                //atomicAdd(&tab_of_occurences[blockIdx.x * NUMBER_OF_CARS + int(dictionnary[i]) - 32], 1);

        __syncthreads(); // To wait for all calculations are done

        if(tidb == 0){
            for(int i = 0 ; i < NB_CHARS ; i++){
                arrCount[blockIdx.x * NB_CHARS + i] = sum_by_block[i];
            }
        }
    }
}

__global__ void combineBlockData(unsigned int *blockCharData,unsigned int *devCharCount, unsigned int size){
    const unsigned int tidb = threadIdx.x; // Because one block
    unsigned int charCount = 0;

    for(int i = tidb ; i < size ; i+= NB_CHARS){
        charCount += blockCharData[i];
    }

    devCharCount[tidb] = charCount;
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
    cout<<"Size of countchar is "<<COUNTCHAR<<endl;

    // For HOST
    char *allChars;
    unsigned int *charCount; // total num of characters
    

    // char arrayChar[] = readFile(inputFile);
    // readF


    // Allocation of host copy
    allChars = (char*)malloc(COUNTCHAR*sizeof(char));
    charCount = (unsigned int*)malloc(NB_CHARS*sizeof(unsigned int));
    
    // char charArr[vv.size()];
    copy(charDict.begin(), charDict.end(), allChars);

    // cout<<"allchar is "<<allChars[5]<<"and ascii code is "<<int(allChars[5])<<endl;
    // cout<<"allchar is "<<allChars[6]<<"and ascii code is "<<int(allChars[6])<<endl;
    // cout<<"allchar is "<<allChars[7]<<"and ascii code is "<<int(allChars[7])<<endl;
    

    // For DEVICE
    char *devAllChars;
    unsigned int *devCharCount;
    unsigned int *devBlockCharCount;

    // Create timer     
    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Allocation of device copy
    cudaMalloc((void**)&devAllChars, COUNTCHAR*sizeof(char));
    cudaMalloc((void**)&devCharCount, NB_CHARS*sizeof(unsigned int));
    cudaMalloc((void**)&devBlockCharCount, NB_CHARS*NB_BLOCKS*sizeof(unsigned int));

    // copy the host data to device
    cudaMemcpy(devAllChars, allChars, COUNTCHAR*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemset(devCharCount, 0, NB_CHARS*sizeof(unsigned int));
    cudaMemset(devBlockCharCount, 0, NB_CHARS*NB_BLOCKS*sizeof(unsigned int));

    // cout<<"devallchar is "<<devAllChars[34]<<endl;
    // Call the kernel to generate the histogram 
    // genHistogram <<< NB_BLOCKS, NB_THREADS >>>(devAllChars, devCharCount, COUNTCHAR);
    genHistogramShared <<< NB_BLOCKS, NB_THREADS >>>(devAllChars, devBlockCharCount, COUNTCHAR);
    combineBlockData<<< 1, NB_CHARS >>>(devBlockCharCount, devCharCount, NB_CHARS*NB_BLOCKS);

    // copy device result back to host copy
    // cudaMemcpy(charCount, devCharCount, NB_CHARS*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(charCount, devCharCount, NB_CHARS*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    // End and delete the timer
    sdkStopTimer(&timer);
    printf("GPU Processing time (in ms) : %f \n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    // cout<<"the count of a is "<<charCount[0]<<endl;

    // write the results to output file
    writeResult(outputFile, charCount);


    // free device momory
    cudaFree(devCharCount);
    cudaFree(devAllChars); 

    // free host memory
    free(allChars);
    free(charCount);

}