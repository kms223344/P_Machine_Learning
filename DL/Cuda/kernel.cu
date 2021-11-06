#include<stdio.h>
__global__ void kernel()
{
	;
}
int main()
{
	kernel << <1, 1 >> > ();
	printf("hellworld");
	return 0;
}