#include "dropout_kernel.h"
#include <curand_kernel.h>
#include <windows.h>

#define BLOCK_SIZE 32

__device__ int WangHash(int a) {
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
}

__global__ void dropout_kernel (const float * __restrict__ src,
                                float * __restrict__ dst, float * __restrict__ dst_idx, int m, int n, float p, int seed){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if (row < m && col < n){

        // curand_init is very slow.
        // so we use the technique as bellow.
        // http://richiesams.blogspot.jp/2015/03/creating-randomness-and-acummulating.html
        // or https://devtalk.nvidia.com/default/topic/480586/curand-initialization-time/
        //generate random number
        int SEED = WangHash(seed);
        curandState_t state;
        int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
        //curand_init( (SEED << 20) + threadId, 0, 0, &state);
        curand_init( SEED + threadId, 0, 0, &state);
        float randNum = curand_uniform(&state);


        float scale = 1.0/(1.0-p);
        float flag = randNum >= p ? 1.0:0.0;
        /*
        if (randNum >= p){
            dst[row * n + col] = src[row * n + col] / (1.0-p);
            dst_idx[row * n + col] = 1.0f;
        }
        else{
            dst[row * n + col] = 0.0f;
            dst_idx[row * n + col] = 0.0f;
        }
        */
        float mask = scale * flag;
        dst_idx[row * n + col] = mask;
        dst[row * n + col] = src[row * n + col] * mask;

    }

}


//From->:https://stackoverflow.com/questions/5404277/porting-clock-gettime-to-windows
struct timespec2 { long tv_sec; long tv_nsec; };   //header part
#define exp7           10000000i64     //1E+7     //C-file part
#define exp9         1000000000i64     //1E+9
#define w2ux 116444736000000000i64     //1.jan1601 to 1.jan1970
void unix_time(struct timespec2* spec)
{
    __int64 wintime; GetSystemTimeAsFileTime((FILETIME*)&wintime);
    wintime -= w2ux;  spec->tv_sec = wintime / exp7;
    spec->tv_nsec = wintime % exp7 * 100;
}
int clock_gettime(int, timespec2* spec)
{
    static  struct timespec2 startspec; static double ticks2nano;
    static __int64 startticks, tps = 0;    __int64 tmp, curticks;
    QueryPerformanceFrequency((LARGE_INTEGER*)&tmp); //some strange system can
    if (tps != tmp) {
        tps = tmp; //init ~~ONCE         //possibly change freq ?
        QueryPerformanceCounter((LARGE_INTEGER*)&startticks);
        unix_time(&startspec); ticks2nano = (double)exp9 / tps;
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&curticks); curticks -= startticks;
    spec->tv_sec = startspec.tv_sec + (curticks / tps);
    spec->tv_nsec = startspec.tv_nsec + (double)(curticks % tps) * ticks2nano;
    if (!(spec->tv_nsec < exp9)) { spec->tv_sec++; spec->tv_nsec -= exp9; }
    return 0;
}
//<-to:https://stackoverflow.com/questions/5404277/porting-clock-gettime-to-windows

void dropout_kernel_exec(const float *src, float *dst, float *dst_idx, int m, int n, float p){
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n+block.x-1)/block.x, (m+block.y-1)/block.y);

    //int seed = time(0);
    struct timespec2 tm;
    clock_gettime(0, &tm);
    int seed = tm.tv_nsec;

    /* lunch kernel */
    dropout_kernel<<<grid, block>>>(src, dst, dst_idx, m, n, p, seed);
    cudaThreadSynchronize();
}
