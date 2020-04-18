

////////////////////////////////////////////////////////////////
////////////  GPU kernel to generate histogram//////////////////
////////////  without using shared memory //////////////////////
////////////////////////////////////////////////////////////////
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
            else if(int(dict[i]) > 90){
                atomicAdd(&arrCount[int(dict[i]) - 58], 1);
            }
        }
    }


}


////////////////////////////////////////////////////////////////
////////////  GPU kernel to generate histogram  ////////////////
////////////        using shared memory    /////////////////////
////////////////////////////////////////////////////////////////
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

///////////////////////////////////////////////
////  GPU KERNEL TO COMBINE ALL BLOCK DATA ////
///////////////////////////////////////////////
__global__ void combineBlockData(unsigned int *blockCharData,unsigned int *devCharCount, unsigned int size){
    const unsigned int tidb = threadIdx.x; // Because one block
    unsigned int charCount = 0;

    for(int i = tidb ; i < size ; i+= NB_CHARS){
        charCount += blockCharData[i];
    }

    devCharCount[tidb] = charCount;
}
