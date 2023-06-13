#include <stdio.h>
#include "shapes.h"

float parse_rect_area(char *str)
{
    char c;
    float length, width;
    int result = sscanf(str, "%c %f %f\n", &c, &length, &width);

    if (result != 3)
    {
        printf("Error in parsing rectangle dimensions.\n");
        return -1.0f;
    }

    return area_of_rect(length, width);
}

float parse_circle_area(char *str)
{
    char c;
    float radius;
    int result = sscanf(str, "%c %f\n", &c, &radius);

    if (result != 2)
    {
        printf("Error in parsing circle dimensions.\n");
        return -1.0f;
    }

    return area_of_circle(radius);
}

float parse_ring_area(char *str)
{
    char c;
    float r_in, r_out;
    int result = sscanf(str, "%c %f %f\n", &c, &r_in, &r_out);

    if (result != 3)
    {
        printf("Error in parsing ring dimensions.\n");
        return -1.0f;
    }

    return area_of_ring(r_in, r_out);
}

float parse_and_get_area(char *str)
{
    char shape = str[0];
    switch (shape)
    {
        case 'c': // Circle
            return parse_circle_area(str);

        case 'r': // Ring
            return parse_ring_area(str);

        case 'R': // Rectangle
            return parse_rect_area(str);

        default:
            return -1.0f;
    }
}