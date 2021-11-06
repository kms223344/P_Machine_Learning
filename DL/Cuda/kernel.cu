#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common/cpu_bitmap.h"

#include<stdio.h>

#define dim 1000

struct cuComplex
{
	float r, i;
	__device__ cuComplex(float a, float b) : r(a), i(b) {  }
	__device__ float magnitude()
	{
		return r * r + i * i;
	}
	__device__ cuComplex operator*(const cuComplex& a)
	{
		return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
	}
	__device__ cuComplex operator+(const cuComplex& a)
	{
		return cuComplex(r + a.r, i + a.i);
	}

};

__device__ int julia(int x, int y)
{
	const float scale = 1.5;
	float jx = scale * (float)(dim / 2 - x) / (dim / 2);
	float jy = scale * (float)(dim / 2 - y) / (dim / 2);

	cuComplex c(-0.8, 0.157);
	cuComplex a(jx, jy);
	for (int i = 0; i < 200; i++)
	{
		a = a * a + c;
		if (a.magnitude() > 1000) return 0;
	}
	return 1;
}
__global__ void kernel(unsigned char* ptr)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;

	int juliavalue = julia(x, y);
	ptr[offset * 4 + 0] = 255 * juliavalue;
	ptr[offset * 4 + 1] = 0;
	ptr[offset * 4 + 2] = 0;
	ptr[offset * 4 + 3] = 255;
}
int main(void)
{
	CPUBitmap bitmap(dim, dim);
	unsigned char *dbitmap;
	cudaMalloc((void**)&dbitmap, bitmap.image_size());
	dim3 grid(dim, dim);
	kernel << <grid, 1 >> > (dbitmap);
	cudaMemcpy(bitmap.get_ptr(), dbitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
	bitmap.display_and_exit();
	cudaFree(dbitmap);

}
