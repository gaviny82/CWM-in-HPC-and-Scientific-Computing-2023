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
#include "parse.h"

// Now I start my code with main()
int main(void)
{
    // In here I need to delare my variables
    float loss_in_kg = 0.0f;
    float loss_in_pence = 0.0f;
    float total_area = 0.0f;

    // Next I need to get input from the user.
    // I'll do this by using a printf() to ask the user to input the radii.
    printf("Input the shapes and their dimensions in dimensions.txt\n");
    printf("Press [Enter] to continue when dimensions.txt is saved.\n");
    getchar();

    // Open file dimensions.txt
    FILE *dim_file = fopen("dimensions.txt", "r");
    if (dim_file == NULL)
    {
        printf("Unable to open the dimensions file.\n");
        return 1;
    }

    // Now I need to loop through the lines caluclating the area for each shape
    char buffer[50];
    while (!feof(dim_file))
    {
        // Read a line
        char *line = fgets(buffer, 50, dim_file);
        if (line == NULL)
        {
            printf("Error in reading the dimensions.\n");
            return 2;
        }

        // Parse a line and calculate the area
        float area = parse_and_get_area(line);
        if (area == -1.0f)
        {
            printf("Unsupported shape %c\n", line[0]);
            return 3;
        }
        printf("Area %c: %f\n", line[0], area);

        // Next I'll sum up all of the individual areas
        total_area += area;
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
    loss_in_pence = loss_in_kg * 10;

    // Finally I'll use a printf() to print this to the screen.
    printf("Total area lossed in m^2 is:\t%f\n", total_area);
    printf("Total loss in kg is:\t\t%f\n", loss_in_kg);
    printf("Total loss in pence is:\t\t%f\n", loss_in_pence);

    // Ask the user to input the dimension of the farm
    float length, width;
    printf("\n");
    printf("Enter the dimensions of your farm:\n");
    printf("Length: ");
    scanf("%f", &length);
    printf("Width: ");
    scanf("%f", &width);
    float area_of_farm = area_of_rect(length, width);
    float percentage_loss = total_area / area_of_farm;
    printf("Percentage lost: %.2f%%\n", percentage_loss);

    return 0;
}
