#include <math.h>

#define PI 3.1415926

float area_of_circle(float radius)
{
    float area = PI * radius * radius;
    return area;
}

float area_of_rect(float length, float width)
{
    return length * width;
}

float area_of_ring(float inner_radius, float outer_radius)
{
    return area_of_circle(outer_radius) - area_of_circle(inner_radius);
}