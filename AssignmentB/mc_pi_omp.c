#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main()
{
    int N = 8388608;
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

    printf("Pi:\t%f\n", (4.0 * area) / (float)N);
    printf("Time elapsed:\t%f\n", time_end - time_start);

    return 0;
}
