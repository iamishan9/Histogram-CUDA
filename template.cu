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
#include <map>
#include <math.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

#define NUMBER_OF_CARS 70
# define NB_BLOCKS 10
# define NB_THREADS 1024
# define TOTAL_THREAD NB_BLOCKS*NB_THREADS

using namespace std;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

extern "C"



void cpu_histogram(vector<string> dict){
   map<char,int> char_frequency;
   for(int i=32;i<127;i++){
 
       if(i<65 || i>90){
       char_frequency[char(i)]=0;
       }
   }

   for(unsigned int i=0;i<dict.size();i++){
       for(char &c: dict[i]){
           if(isupper(c)){
               c=c+32;

           }
            map<char, int>::iterator it = char_frequency.find(c); 
            if (it != char_frequency.end())
                it->second +=1;

        }
   }


   ofstream myfile;
   myfile.open("example.csv");
   for (auto const& x : char_frequency)
{
    myfile<< x.first  // string (key)
              << ':' 
              << x.second // string's value 
              << std::endl ;
}

myfile.close();

}


////////////////////////////////////////////////
///// function returns the total no. ///////////
///// of characters in the given file //////////
////////////////////////////////////////////////
int total_characters(vector<string> dictionary){
    int nbChars=0;
    
    for(int i=0;i< dictionary.size();i++){
        for(char &c: dictionary[i]){
            nbChars+=1;
            }
    }
	return nbChars;
}



////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
//__global__ void
//testKernel(float *g_idata, float *g_odata)
//{
//    // shared memory
//    // the size is determined by the host application
//    extern  __shared__  float sdata[];
//
//    // access thread id
//    const unsigned int tid = threadIdx.x;
//    // access number of threads in this block
//    const unsigned int num_threads = blockDim.x;
//
//    // read in input data from global memory
//    sdata[tid] = g_idata[tid];
//    __syncthreads();
//
//    // perform some computations
//    sdata[tid] = (float) num_threads * sdata[tid];
//    __syncthreads();
//
//    // write data to global memory
//    g_odata[tid] = sdata[tid];
//}


////////////////////////////////////////////
// function reads file and puts the ////////
// strings in a vector of strings///////////
vector <string> read_file(string filename) {

    ifstream in(filename);
    if (!in) {
        cerr << "Cannot open" << filename << endl;
    }


    vector <string> vectorOfStrings;
    string str;
    while (getline(in, str)) {
        if (str.size() > 0) {
            vectorOfStrings.push_back(str);
        }

    }
    in.close();
    return vectorOfStrings;

}

void fillTabStrings(char *tab_dictionnary, std::vector<std::string> dictionnary){
	unsigned int number_add = 0;
	
	// Fill the tab_dictionnary
	for(int i = 0 ; i < dictionnary.size(); i++){
		for(int j = 0; j < dictionnary.at(i).length() ; j++){
			tab_dictionnary[number_add] = dictionnary.at(i).at(j);
			number_add++;
		}
	}
}



void writeInFile(char *nameOutputfile, unsigned int *tab_of_occurences){
	std::ofstream outputFile(nameOutputfile); // Open output file

	if(outputFile.is_open()){ // Test if the file is opened
		for(int i = 0 ; i <= 69 ; i++){
            if(i+32<65){
            outputFile << char(i + 32);
        }
        else{
            outputFile << char(i + 58);
        }
			outputFile << " : ";
			outputFile << tab_of_occurences[i];
			outputFile << "\n";
		}
	}

	outputFile.close();
}


__global__ void totalSum(unsigned int *tab_total_sum, unsigned int *tab_datas, unsigned int size_tab){
	const unsigned int tidb = threadIdx.x; // Because one block
	unsigned int total_sum = 0;

	// Add size(tab_datas) ; More stable in the time
	for(int i = tidb ; i < size_tab ; i+= NUMBER_OF_CARS){ // Can be not coalescent ?
		total_sum += tab_datas[i];
	}

	// More flucuation in the time
	// for(int i = tidb ; i < size_tab ; i+= NUMBER_OF_CARS*2){
	// 	total_sum += tab_datas[i];
	// }

	// for(int i = tidb + NUMBER_OF_CARS ; i < size_tab ; i+= NUMBER_OF_CARS*2){
	// 	total_sum += tab_datas[i];
	// }

	tab_total_sum[tidb] = total_sum;
}

__global__ void countNumberOfOccurencesNiave(char *dictionnary, unsigned int *tab_of_occurences, unsigned int total_car){
    const unsigned int tidb = threadIdx.x;
	const unsigned int tid = blockIdx.x * blockDim.x + tidb;

}
__global__ void countNumberOfOccurences(char *dictionnary, unsigned int *tab_of_occurences, unsigned int total_car){
	// thread id inside thread block and inside grid
	const unsigned int tidb = threadIdx.x;
	const unsigned int tid = blockIdx.x * blockDim.x + tidb;

	__shared__ unsigned int sum_by_block[NUMBER_OF_CARS]; // Store the sum of the number of car by blocks

	if(tid < total_car){ // Check if the number of the threads is less than the number of elements in the dictionnary

		for(int i = tid ; i  < total_car; i += TOTAL_THREAD){ // Coalescent method to calculate the car

            if(int(dictionnary[i]) >= 32 && int(dictionnary[i]) <= 127){ // Check if the code ascii is in the part we want
              
                if(int(dictionnary[i])<65 ){
                    atomicAdd(&sum_by_block[int(dictionnary[i]) - 32], 1);
                    
                
                }
                else if(int(dictionnary[i])>=65 && dictionnary[i]<=90 ){
                    atomicAdd(&sum_by_block[int(dictionnary[i]) - 32+6], 1);
                    
                
                }
                else if(dictionnary[i]>90){
                    atomicAdd(&sum_by_block[int(dictionnary[i]) - 58], 1);
                }
				

				// For no usage of the shared memory
				//atomicAdd(&tab_of_occurences[blockIdx.x * NUMBER_OF_CARS + int(dictionnary[i]) - 32], 1);
			}
		}

		__syncthreads(); // To wait for all calculations are done
		if(tidb == 0){
			for(int i = 0 ; i < NUMBER_OF_CARS ; i++){
				tab_of_occurences[blockIdx.x * NUMBER_OF_CARS + i] = sum_by_block[i];
			}
		}
	}
}


///////////////////////////////
// Program main //////////////
//////////////////////////////
int
main(int argc, char **argv) {

    // checking the format of the input
    char *inputPar = argv[1];
    char *outputPar = argv[3];

    //checking argument length and parameters

    // if arguments are not valid
    if (argc != 5 || strcmp(inputPar, "-i") != 0 || strcmp(outputPar, "-o") != 0) {
        printf("Usage : ./template -i <inputfilename.txt> -o <outputfilename.csv> \n");
        return -1;
    }


        // if arguments are valid
    else {

        char *inputFileName = argv[2];  // input text file name
        char *outputFileName = argv[4]; // output csv file name

        //put the strings from the dictionary in a vector of string
        vector <string> dictionary = read_file(inputFileName);

        //COMPUTING USING CPU//
        // Create timer
	    StopWatchInterface *timer = 0;
	    sdkCreateTimer(&timer);
	    sdkStartTimer(&timer);
        cpu_histogram(dictionary);
        	// End and delete the timer
	    sdkStopTimer(&timer);
	    printf("Processing time of CPU(in ms) : %f \n", sdkGetTimerValue(&timer));
        sdkDeleteTimer(&timer);
        

        //COMPUTING USING GPU//
        // Host copies
        char *tab_words; // Contains all words in one tab
        unsigned int *number_total_of_values; // Contains the total number of letter in the file 
        unsigned int *number_of_values_block; // Contains the number of words for each block
        unsigned int TOTAL_CAR = total_characters(dictionary); // Total number of car in the file


        // Allocation host copies
        tab_words = (char*)malloc(TOTAL_CAR*sizeof(char)); // Allocate a tab of char for the words
        number_total_of_values = (unsigned int*)malloc(NUMBER_OF_CARS*sizeof(unsigned int)); // Allocation of the total tab
        number_of_values_block = (unsigned int*)malloc(NUMBER_OF_CARS*NB_BLOCKS*sizeof(unsigned int)); // Allocation of the tab for blocks


        // Device copies
        char *dev_tab_words;
        unsigned int *dev_number_total_of_values;
        unsigned int *dev_number_of_values_block;


        // Fill the tab_words
        fillTabStrings(tab_words, dictionary);


        // Allocation device copies
        cudaMalloc((void**) &dev_tab_words, TOTAL_CAR*sizeof(char));
        cudaMalloc((void**) &dev_number_total_of_values, NUMBER_OF_CARS*sizeof(unsigned int));
        cudaMalloc((void**) &dev_number_of_values_block, NUMBER_OF_CARS*NB_BLOCKS*sizeof(unsigned int));


        // Copy inputs to the device
        cudaMemcpy(dev_tab_words, tab_words, TOTAL_CAR*sizeof(char), cudaMemcpyHostToDevice);
        cudaMemset(dev_number_total_of_values, 0, NUMBER_OF_CARS*sizeof(unsigned int));
        cudaMemset(dev_number_of_values_block, 0, NUMBER_OF_CARS*NB_BLOCKS*sizeof(unsigned int));


        // Methods for calculation of the number of elements
        countNumberOfOccurences<<< NB_BLOCKS, NB_THREADS >>>(dev_tab_words, dev_number_of_values_block, TOTAL_CAR); // Sum for each block
        totalSum<<< 1, NUMBER_OF_CARS >>>(dev_number_total_of_values, dev_number_of_values_block, NUMBER_OF_CARS*NB_BLOCKS); // Sum for each block


        // Copy device result back to host copy
        cudaMemcpy(number_total_of_values, dev_number_total_of_values, NUMBER_OF_CARS*sizeof(unsigned int), cudaMemcpyDeviceToHost);


        // Print the results in the output file
        writeInFile(outputFileName, number_total_of_values);


        // // End and delete the timer
        // sdkStopTimer(&timer);
        // printf("Processing time (in ms) : %f \n", sdkGetTimerValue(&timer));
        // sdkDeleteTimer(&timer);


        // Free memory of the device
        cudaFree(dev_number_total_of_values);
        cudaFree(dev_tab_words);
        cudaFree(dev_number_of_values_block);


        // Free memory of the host
        free(tab_words);
        free(number_total_of_values);
        free(number_of_values_block);


    }

}





////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
//void
//runTest(int argc, char **argv)
//{
//    bool bTestResult = true;
//
//    printf("%s Starting...\n\n", argv[0]);
//
//    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
//    int devID = findCudaDevice(argc, (const char **)argv);
//
//    StopWatchInterface *timer = 0;
//    sdkCreateTimer(&timer);
//    sdkStartTimer(&timer);
//
//    unsigned int num_threads = 32;
//    unsigned int mem_size = sizeof(float) * num_threads;
//
//    // allocate host memory
//    float *h_idata = (float *) malloc(mem_size);
//
//    // initalize the memory
//    for (unsigned int i = 0; i < num_threads; ++i)
//    {
//        h_idata[i] = (float) i;
//    }
//
//    // allocate device memory
//    float *d_idata;
//    checkCudaErrors(cudaMalloc((void **) &d_idata, mem_size));
//    // copy host memory to device
//    checkCudaErrors(cudaMemcpy(d_idata, h_idata, mem_size,
//                               cudaMemcpyHostToDevice));
//
//    // allocate device memory for result
//    float *d_odata;
//    checkCudaErrors(cudaMalloc((void **) &d_odata, mem_size));
//
//    // setup execution parameters
//    dim3  grid(1, 1, 1);
//    dim3  threads(num_threads, 1, 1);
//
//    // execute the kernel
//    testKernel<<< grid, threads, mem_size >>>(d_idata, d_odata);
//
//    // check if kernel execution generated and error
//    getLastCudaError("Kernel execution failed");
//
//    // allocate mem for the result on host side
//    float *h_odata = (float *) malloc(mem_size);
//    // copy result from device to host
//    checkCudaErrors(cudaMemcpy(h_odata, d_odata, sizeof(float) * num_threads,
//                               cudaMemcpyDeviceToHost));
//
//    sdkStopTimer(&timer);
//    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
//    sdkDeleteTimer(&timer);
//
//    // compute reference solution
//    float *reference = (float *) malloc(mem_size);
//    computeGold(reference, h_idata, num_threads);
//
//    // check result
//    if (checkCmdLineFlag(argc, (const char **) argv, "regression"))
//    {
//        // write file for regression test
//        sdkWriteFile("./data/regression.dat", h_odata, num_threads, 0.0f, false);
//    }
//    else
//    {
//        // custom output handling when no regression test running
//        // in this case check if the result is equivalent to the expected solution
//        bTestResult = compareData(reference, h_odata, num_threads, 0.0f, 0.0f);
//    }
//
//    // cleanup memory
//    free(h_idata);
//    free(h_odata);
//    free(reference);
//    checkCudaErrors(cudaFree(d_idata));
//    checkCudaErrors(cudaFree(d_odata));
//
//    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
//}
