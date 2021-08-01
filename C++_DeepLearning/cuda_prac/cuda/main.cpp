#include<stdio.h>
#include<cuda_runtime.h>

#include "kernel.h"

int main()
{

    //Kernel configuration, where a two-dimensional grid and
    //three-dimensional blocks are configured.
    dim3 dimGrid(2, 2);
    dim3 dimBlock(2, 2, 2);
    tK(dimGrid, dimBlock);
    cudaDeviceSynchronize();

    return 0;
}