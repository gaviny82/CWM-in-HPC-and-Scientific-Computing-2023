#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main()
{
    int N = 100000000;
    int area = 0;

    #pragma omp parallel default(none) private(x, y) shared(area, N)
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
    
    printf("\nPi:\t%f\n", (4.0 * area) / (float)N);
    return (0);
}
