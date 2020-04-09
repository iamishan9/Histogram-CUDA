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

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

extern "C"
void computeGold(int *refmat, int *idata, const unsigned int i_dim, const unsigned int j_dim);

////////////////////////////////////////////////////////////////////////////////
//! Simple transpose kernel
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void transpose_naive(int *g_idata, int *g_odata)
{


}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char **argv)
{
    bool bTestResult = true;

    printf("%s Starting...\n\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);

    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    const unsigned int I_DIM = 4096;
    const unsigned int J_DIM = 4096;
    unsigned int mem_size = sizeof(int) * I_DIM * J_DIM;

    // allocate host memory
    int *h_idata = (int *) malloc(mem_size);

    // initalize the memory
    for (unsigned int i = 0; i < I_DIM; i++)
      for (unsigned int j = 0; j < J_DIM; j++)
      {
        h_idata[i*J_DIM +j] = i;
      }

    // allocate device memory
    int *d_idata;
    checkCudaErrors(cudaMalloc((void **) &d_idata, mem_size));
    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, mem_size,
                               cudaMemcpyHostToDevice));

    // allocate device memory for result
    int *d_odata;
    checkCudaErrors(cudaMalloc((void **) &d_odata, mem_size));

    // setup execution parameters
    const unsigned int BSX = 16;
    const unsigned int BSY = 16;
    dim3  blocks(BSX, BSY, 1);
    dim3  grid(J_DIM/BSX, I_DIM/BSY, 1);

    // execute the kernel
    transpose_naive<<< grid, blocks, 0 >>>(d_idata, d_odata);

    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");

    // allocate mem for the result on host side
    int *h_odata = (int *) malloc(mem_size);
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, mem_size,
                               cudaMemcpyDeviceToHost));

    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    // compute reference solution
    int *reference = (int *) malloc(mem_size);
    computeGold(reference, h_idata, I_DIM, J_DIM);

    // check result
    bool resultOk = true;
    for (unsigned int i = 0; i < I_DIM; i++)
      for (unsigned int j = 0; j < J_DIM; j++)
      {
        if (reference[i*J_DIM +j] != h_odata[i*J_DIM +j]){
          resultOk = false;
          break;
        }
      }
    if(resultOk)
      printf("TEST PASSED\n");
    else
      printf("TEST FAILED\n");


    // cleanup memory
    free(h_idata);
    free(h_odata);
    free(reference);
    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));

    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
