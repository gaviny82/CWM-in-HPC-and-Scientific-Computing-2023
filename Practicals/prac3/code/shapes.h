#ifndef SHAPES_H
#define SHAPES_H

/// @brief Calculate the area of a circle
/// @param radius Radius of the circle
/// @return Area of the circle
float area_of_circle(float radius);

/// @brief Calculate the area of a rectangle
/// @param length Length of the rectangle
/// @param width Width of the rectangle
/// @return Area of the rectangle
float area_of_rect(float length, float width);

/// @brief Calculate the area of a ring
/// @param inner_radius Inner radius of the ring
/// @param outer_radius Outer radius of the ring
/// @return Area of the ring
float area_of_ring(float inner_radius, float outer_radius);

#endif