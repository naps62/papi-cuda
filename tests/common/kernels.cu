#include <cuda.h>

#define DIVERGENCE_HERE 		\
	if(arr[id] %2 == 0)				\
		arr[id] = arr[id-1]; 	\
	else 						\
		arr[id] = arr[id+1];

__global__ void kernel_one(int *arr, int N) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= N);
	DIVERGENCE_HERE;
}

__device__ void aux(int *arr, int id, int N) {
	DIVERGENCE_HERE;
}


__global__ void kernel_two(int *arr, int N) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= N);

	aux(arr, id, N);
}
