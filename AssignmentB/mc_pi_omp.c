#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define PI_ACCURATE 3.141592653589794

int main()
{
    int N = 65536;
    int area = 0;

    double time_start = omp_get_wtime();

    #pragma omp parallel default(none) shared(area, N)
    {
        #pragma omp for reduction(+:area)
        for (int i = 0; i < N; i++)
        {
            float x = ((float)rand()) / RAND_MAX; // Random number in [0, 1]
            float y = ((float)rand()) / RAND_MAX; // Random number in [0, 1]
            if (x * x + y * y <= 1.0f)
                area++;
        }
    }
    
    double time_end = omp_get_wtime();

    float pi = (4.0 * area) / (float)N;
    float error = pi - PI_ACCURATE;

    printf("Pi:\t%f\n", pi);
    printf("Error = %.2f%%\n", error * 100);
    printf("Time elapsed:\t%f\n", time_end - time_start);

    return 0;
}
