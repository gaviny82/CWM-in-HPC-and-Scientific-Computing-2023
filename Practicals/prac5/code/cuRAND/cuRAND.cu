// random number generation
#include <cuda.h>
#include <curand.h>
#include <stdio.h>
#include <math.h>

#define NUM_ELS 100

float arr_mean(float *arr, int n);
float standard_deviation(float *arr, int n, float mean);

int main(void)
{
    // Declare variable
    curandGenerator_t gen;
    // Create random number generator
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    // Set the generator options
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
 
    // Allocate memory on GPU
    float *d_nums;
    size_t arr_size = sizeof(float) * NUM_ELS;
    cudaMalloc((void **)&d_nums, arr_size);

    // Generate the randoms
    curandGenerateNormal(gen, d_nums, NUM_ELS, 0.0f, 1.0f);

    // Copy to CPU and free GPU memory
    float h_nums[NUM_ELS];
    cudaMemcpy(h_nums, d_nums, arr_size, cudaMemcpyDeviceToHost);
    cudaFree(d_nums);

    // Print random numbers generated
    for (int i = 0; i < NUM_ELS; i++)
    {
        printf("Number %d: %f\n", i, h_nums[i]);
    }

    // Calculate mean and sd
    float mean = arr_mean(h_nums, NUM_ELS);
    float sd = standard_deviation(h_nums, NUM_ELS, mean);
    printf("Mean: %.4e\n", mean);
    printf("SD: %.4e\n", sd);

    return 0;
}

float arr_mean(float *arr, int n)
{
    float sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += arr[i];
    }
    return sum / n;
}

float standard_deviation(float *arr, int n, float mean)
{
    float square_sum = 0;
    for (int i = 0; i < n; i++)
    {
        float diff = arr[i] - mean;
        square_sum += diff * diff;
    }
    return sqrt(square_sum / n);
}