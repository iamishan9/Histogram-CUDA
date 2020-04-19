

////////////////////////////////////////////////////////////////
////////////  GPU kernel to generate histogram//////////////////
////////////  without using shared memory //////////////////////
////////////////////////////////////////////////////////////////
__global__ void genHistogram(char *dict, unsigned int *arrCount, unsigned int charCount ){

    const unsigned int tidb = threadIdx.x;
    const unsigned int tid = blockIdx.x * blockDim.x + tidb;

    if(tid < charCount){

        for(int i=tid; i<charCount; i += TOTAL_THREAD){
            // before uppercase characters
            if(int(dict[i]) < ASCII_DOLLAR){
                atomicAdd(&arrCount[int(dict[i]) - MIN_ASCII], 1);
            }
            else if(int(dict[i]) > ASCII_DOLLAR && int(dict[i]) < ASCII_ASTERIK){
                atomicAdd(&arrCount[int(dict[i]) - MIN_ASCII - CHAR_DIFF_BET_36_42], 1);
            }

            else if(int(dict[i])>ASCII_ASTERIK && int(dict[i]) < UPPERCASE_ALPHABET){
                atomicAdd(&arrCount[int(dict[i]) - MIN_ASCII - CHAR_DIFF_ABOVE_42], 1);
            }

            // for uppercase only
            else if(int(dict[i]) >= UPPERCASE_ALPHABET && dict[i] <= 90 ){
                atomicAdd(&arrCount[int(dict[i]) - MIN_ASCII - CHAR_DIFF_ABOVE_42 + 6], 1);

            }

            // after uppercase characters
            else if(int(dict[i]) > 90){
                atomicAdd(&arrCount[int(dict[i]) - SKIP_UPPERCASE - CHAR_DIFF_ABOVE_42], 1);
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

    __shared__ unsigned int charCountByBlock[NB_CHARS]; 

    if(tid < charCount){ 

        for(int i=tid; i<charCount; i += TOTAL_THREAD){

            // before uppercase characters
            if(int(dict[i]) < ASCII_DOLLAR){
                atomicAdd(&charCountByBlock[int(dict[i]) - MIN_ASCII], 1);
            }
            else if(int(dict[i]) > ASCII_DOLLAR && int(dict[i]) < ASCII_ASTERIK){
                atomicAdd(&charCountByBlock[int(dict[i]) - MIN_ASCII - CHAR_DIFF_BET_36_42], 1);
            }

            else if(int(dict[i])>ASCII_ASTERIK && int(dict[i]) < UPPERCASE_ALPHABET){
                atomicAdd(&charCountByBlock[int(dict[i]) - MIN_ASCII - CHAR_DIFF_ABOVE_42], 1);
            }

            // for uppercase only
            else if(int(dict[i]) >= UPPERCASE_ALPHABET && dict[i] <= 90 ){
                atomicAdd(&charCountByBlock[int(dict[i]) - MIN_ASCII - CHAR_DIFF_ABOVE_42 + 6], 1);

            }

            // after uppercase characters
            else if(int(dict[i]) > 90){
                atomicAdd(&charCountByBlock[int(dict[i]) - SKIP_UPPERCASE - CHAR_DIFF_ABOVE_42], 1);
            }
        }
        __syncthreads(); 

        if(tidb == 0){
            for(int i = 0 ; i < NB_CHARS ; i++){
                arrCount[blockIdx.x * NB_CHARS + i] = charCountByBlock[i];
            }
        }
    }
}

///////////////////////////////////////////////
////  GPU KERNEL TO COMBINE ALL BLOCK DATA ////
///////////////////////////////////////////////
__global__ void combineBlockData(unsigned int *blockCharData,unsigned int *devCharCount, unsigned int size){
    const unsigned int tidb = threadIdx.x; 
    unsigned int charCount = 0;

    for(int i = tidb ; i < size ; i+= NB_CHARS){
        charCount += blockCharData[i];
    }

    devCharCount[tidb] = charCount;
}
