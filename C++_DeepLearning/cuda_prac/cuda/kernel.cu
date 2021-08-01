﻿/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


//sample : C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.4

 // System includes
#include <stdio.h>
#include <assert.h>
#include "kernel.h"

// CUDA runtime
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

__global__ void testKernel(int val)
{
    printf("[%d, %d]:\t\tValue is:%d\n", \
        blockIdx.y * gridDim.x + blockIdx.x, \
        threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x, \
        val);
}

void tK(dim3 a,dim3 b)
{
    testKernel << <a, b >> > (10);
}
