#include<stdio.h>
#include<cuda.h>
__global__ void kernel(int *x,int *flag)
{
	bool leave=false;
	do{
		if(atomicCAS(flag,0,1)==0)
		{
			leave=true;
			*x=*x+1;
			*flag=0;
		}
	}while(!leave);
}
__global__ void print(int *x){
	printf("%d\n", *x);
}
__global__ void init(int *x, int *flag){
	printf("%d\n", *x);
	*x=0;
	*flag=0;
}
int main()
{
	int *x,*flag;
	cudaMalloc(&x,sizeof(int));
	cudaMalloc(&flag,sizeof(int));
	cudaMemset(x,-1,sizeof(int));
	init<<<1,1>>>(x,flag);
	int *y;
	y=(int*)malloc(sizeof(int));
	kernel<<<100,2>>>(x,flag);
	cudaDeviceSynchronize();
	print<<<1,1>>>(x);
	cudaMemcpy(y,x,sizeof(int),cudaMemcpyDeviceToHost);
	printf("Final: %d\n", *y);
}