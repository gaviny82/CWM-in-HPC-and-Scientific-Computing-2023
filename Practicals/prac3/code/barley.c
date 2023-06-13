/**************************************************
 *                                                *
 * First attempt at a code to calcule lost barley *
 * by A. Farmer                                   *
 * 18/05/18                                       *
 *                                                *
 **************************************************/

// Include any headers from the C standard library here
#include <stdio.h>
#include "shapes.h"

// Now I start my code with main()
int main(void)
{
    // In here I need to delare my variables
    float loss_in_kg = 0.0f;
    float total_area = 0.0f;

    // Next I need to get input from the user.
    // I'll do this by using a printf() to ask the user to input the radii.
    printf("Input the radii in radii.txt\n");
    printf("Press [Enter] to continue when radii.txt is saved.\n");
    getchar();

    // Open file radii.txt
    FILE *radii_file = fopen("radii.txt", "r");
    if (radii_file == NULL)
    {
        printf("Unable to open radii.txt\n");
        return 1;
    }

    // Now I need to loop through the radii caluclating the area for each
    float radius;
    while (fscanf(radii_file, "%f\n", &radius) == 1)
    {
        // Next I'll sum up all of the individual areas
        printf("%f\n", radius);
        total_area += area_of_circle(radius);
    }


    /******************************************************************
     *                                                                *
     * Now I know the total area I can use the following information: *
     *                                                                *
     * One square meter of crop produces about 135 grams of barley    *
     *                                                                *
     * One kg of barley sells for about 10 pence                      *
     *                                                                *
     ******************************************************************/

    // Using the above I'll work out how much barley has been lost.
    loss_in_kg = total_area * 0.135;

    // Finally I'll use a printf() to print this to the screen.
    printf("Total area lossed in m^2 is:\t%f\n", total_area);
    printf("Total loss in kg is:\t\t%f\n", loss_in_kg);

    return 0;
}
