// C standard library
#include <stdio.h>
#include <stdlib.h>

// CUDA APIs
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>

// Global constants
// Using GPU with id 0
#define GPU_ID 0 
// Number of experiments = 2^23
#define N 1024 
// Number of thread blocks in the grid
#define THREAD_BLOCKS 512 
// Number of threads in a thread block
#define THREADS 2
#define RAND_SEED 123456ULL

__global__ void experiment(int *counter, float *x_arr, float *y_arr)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Generate coordinates x, y
    float x = x_arr[index], y = y_arr[index];
    
    printf("index %d, x: %f, y: %f\n", index, x, y);

    if (x * x + y * y <= 1.0f)
    {
        atomicAdd(counter, 1);
        printf("Counter: %d\n", *counter);
    }
}

int main()
{
    // Step 1: Initialise GPU

    // Get the number of GPUs available
    int devCount;
    cudaGetDeviceCount(&devCount);

    // Check if we have enough GPUs
    if(devCount <= GPU_ID)
    {
        printf("[ERROR] Cannot initialise GPU %d.\n", GPU_ID);
        return 1;
    }

    // Tell CUDA that we want to use GPU 0
    cudaSetDevice(GPU_ID);

    // Step 2: Initialise variables on GPU memory directly
    // No need to transfer data form CPU.
    int *d_area;
    float *d_x, *d_y;
    if (cudaMalloc((void **)&d_x, N * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_y, N * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_area, sizeof(int)) != cudaSuccess)
    {
        printf("Failed to allocate GPU memory!\n");
        return 2;
    }


    // Step 3: Calculate PI

    // Initialise random number generator
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, RAND_SEED);
    
    // Generate random numbers
    curandGenerateUniform(gen, d_x, N);
    curandGenerateUniform(gen, d_y, N);
    
    // Launch kernel
    printf("Launch\n");
    experiment<<<THREAD_BLOCKS, THREADS>>>(d_area, d_x, d_y);

    // Step 4: Print results
    int *h_area = (int *)malloc(sizeof(int));
    cudaMemcpy(h_area, d_area, sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nPi:\t%f\n", (4.0 * (*h_area)) / (float)N);
    
    return 0;
}
