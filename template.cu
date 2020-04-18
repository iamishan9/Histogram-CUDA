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

                        

#include "config.h"     //include config file
#include "kernels.cu"   // include kernel file


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
////////////  CPU FUNCTION TO    //////////////
///////////  GENERATE HISTOGRAM  //////////////
///////////////////////////////////////////////

void genHistogramCPU(vector<char> dict){

    vector<int> count(NB_CHARS, 0);

    for(int i=0; i<dict.size(); i++){

        if(int(dict[i]) < 65){
           count[int(dict[i]) - 32] += 1;
        }

        // for uppercase only
        else if(int(dict[i]) >= 65 && dict[i] <= 90 ){
            count[int(dict[i]) - 32 + 6] += 1;

        }

        // after uppercase characters
        else if(int(dict[i]) > 90){
            count[int(dict[i]) - 58] += 1;
        }

    }

    ofstream opfile;

    opfile.open("fromCPU.csv");
 
    for(int i = 0 ; i <= 68 ; i++){
        
        // cout<<" val of I is "<<i<<endl;
        if(i != 4 && i!= 10){
            if(i+32<65){
                opfile << char(i + 32);
            }
            else{
                opfile << char(i + 58);
            }

            opfile << " : ";

            opfile << count[i];

            opfile << "\n";
        }
    }


    opfile.close();


}


///////////////////////////////////////////////
////////////  MAIN PROGRAM  ///////////////////
///////////////////////////////////////////////
int main(int argc, char **argv){
    
    // check if arguments are valid
    if (argc != 5 || strcmp(inputPar, "-i") != 0 || strcmp(outputPar, "-o") != 0) {
        printf("COMMAND : ./template -i <inputText.txt> -o <outputHisto.csv> \n");
        return -1;
    }



    // get input and output file names    
    char *inputFile = argv[2]; 
    char *outputFile = argv[4];

    // read the input text file
    vector<char> charDict = readFile(inputFile);
    

    //                                 SECTION CPU
    ///////////////////////////////////////////////////////////////////////////////////
    
    StopWatchInterface *cpu_timer = 0;
    sdkCreateTimer(&cpu_timer);
    sdkStartTimer(&cpu_timer);
    
    // call the function
    genHistogramCPU(charDict);
    
    // end and del timer
    sdkStopTimer(&cpu_timer);
    printf("Processing time of CPU(in ms) : %f \n", sdkGetTimerValue(&cpu_timer));
    sdkDeleteTimer(&cpu_timer);


    //                                 SECTION GPU
    ///////////////////////////////////////////////////////////////////////////////////


    unsigned int COUNTCHAR = charDict.size();
    cout<<"Size of countchar is "<<COUNTCHAR<<endl;

    // For HOST
    char *allChars;             // to contain all the characters read
    unsigned int *charCount;    // total number of characters in the file
    

    // Allocation of host copy
    allChars = (char*)malloc(COUNTCHAR*sizeof(char));
    charCount = (unsigned int*)malloc(NB_CHARS*sizeof(unsigned int));
    
    // copy the characters from vector to array
    copy(charDict.begin(), charDict.end(), allChars);

 

    // allocation of device copies
    char *devAllChars;                  // device copy of all characters from file
    unsigned int *devCharCount;         // to contain the final histogram
    unsigned int *devBlockCharCount;    // for while using shared memory, this will 

    
    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // allocation of device memory
    cudaMalloc((void**)&devAllChars, COUNTCHAR*sizeof(char));
    cudaMalloc((void**)&devCharCount, NB_CHARS*sizeof(unsigned int));

    // copy the host data to device
    cudaMemcpy(devAllChars, allChars, COUNTCHAR*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemset(devCharCount, 0, NB_CHARS*sizeof(unsigned int));
    


    // Call the kernel to generate the histogram 
    genHistogram <<< NB_BLOCKS, NB_THREADS >>>(devAllChars, devCharCount, COUNTCHAR);
    


    // copy device result back to host copy
    cudaMemcpy(charCount, devCharCount, NB_CHARS*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    // end and del timer
    sdkStopTimer(&timer);
    printf("GPU Processing time (in ms)[naive method] : %f \n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);


    // write the results to output file
    writeResult(outputFile, charCount);


    // free device momory
    cudaFree(devCharCount);
    cudaFree(devAllChars); 












    StopWatchInterface *timer2 = 0;
    sdkCreateTimer(&timer2);
    sdkStartTimer(&timer2);

    // Allocation of device copy
    cudaMalloc((void**)&devAllChars, COUNTCHAR*sizeof(char));
    cudaMalloc((void**)&devCharCount, NB_CHARS*sizeof(unsigned int));
    cudaMalloc((void**)&devBlockCharCount, NB_CHARS*NB_BLOCKS*sizeof(unsigned int));

    // copy the host data to device
    cudaMemcpy(devAllChars, allChars, COUNTCHAR*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemset(devCharCount, 0, NB_CHARS*sizeof(unsigned int));
    cudaMemset(devBlockCharCount, 0, NB_CHARS*NB_BLOCKS*sizeof(unsigned int));

  
    // call the kernel to create histogram for each block
    genHistogramShared <<< NB_BLOCKS, NB_THREADS >>>(devAllChars, devBlockCharCount, COUNTCHAR);
    
    // combine the data of all blocks to produce one result containing the histogram
    combineBlockData<<< 1, NB_CHARS >>>(devBlockCharCount, devCharCount, NB_CHARS*NB_BLOCKS);

    // copy device result back to host copy
    cudaMemcpy(charCount, devCharCount, NB_CHARS*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    // end and del timer
    sdkStopTimer(&timer2);
    printf("GPU Processing time (in ms)[using shared memory] : %f \n", sdkGetTimerValue(&timer2));
    sdkDeleteTimer(&timer2);

    // cout<<"the count of a is "<<charCount[0]<<endl;

    // write the results to output file

    writeResult(outputFile, charCount);


    // free device momory
    cudaFree(devCharCount);
    cudaFree(devAllChars); 
    cudaFree(devBlockCharCount);

    //without shared memory
     // Create timer     
    // StopWatchInterface *timer = 0;


    // free host memory
    free(allChars);
    free(charCount);

}