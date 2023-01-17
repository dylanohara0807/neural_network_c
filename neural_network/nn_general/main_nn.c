//
// File: name_here.c
//
// Description:
//
// @author: Dylan O'Hara
//
// // // // // // // // // // // // // // // // // // // // // // //

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#include "nn_model.h"

double *** gen_ex(int length) {

    srand(31);
    double ***ret = malloc(16);
    double **inputs = malloc(8 * length);
    double **outputs = malloc(8 * length);

    for (int i = 0; i < length; i++) {

        inputs[i] = malloc(8 * 4); outputs[i] = malloc(8);
        inputs[i][0] = (rand() % 200) - 100;
        inputs[i][1] = (rand() % 200) - 100;
        inputs[i][2] = (rand() % 200) - 100;
        inputs[i][3] = 0;
        if (inputs[i][0] * inputs[i][2] > inputs[i][1] * inputs[i][1] || inputs[i][2] + inputs[i][1] > 0)
            outputs[i][0] = 1;
        else
            outputs[i][0] = 0;

    }

    ret[0] = inputs; ret[1] = outputs;
    return ret;

}

/// main()
///
/// @param agrc  int, number of arguments (including program call)
/// @param argv  array, the arguments as C strings
/// @return      EXIT_SUCCESS or EXIT_FAILURE

int main(int argc, char *argv[]) {

    FILE* fp = fopen("../../training_data/titanic_train.csv", "r");
    char buffer[1024];
    int row = 0; int column = 0;
    int length = 891;
    double **output = malloc(8 * length); double **input = malloc(8 * length);

    for (int i = 0; i < length; i++) {

        output[i] = malloc(8);
        input[i] = malloc(8 * 3);

    }

    fgets(buffer, 1024, fp);
    while (fgets(buffer, 1024, fp) != NULL) {

        column = 0;
        char* value = strtok(buffer, ",");
        value = strtok(NULL, ",");
 
        while (value) {

            if (column == 0) output[row][0] = strtod(value, NULL);
            else if (column == 1) input[row][0] = strtod(value, NULL);
            else if (column == 3) {
                value = strtok(NULL, ",");
                if (strcmp(value, "male") == 0) input[row][1] = -10;
                else input[row][1] = 10;
            }
            else if (column == 4) input[row][2] = strtod(value, NULL) / 10;
            else;
            value = strtok(NULL, ",");
            column++;

        }
 
       row++;

    }
 
    fclose(fp);

    NModel test = nmodel_new(16, 3);
    nmodel_insert(test, 8, 0);
    nmodel_insert(test, 8, 0);
    nmodel_insert(test, 1, 1);

    printf("\n");
    printf("Cost Function: %f\n\n", nmodel_costfunc(test, output, input, length));
    printf("----------------------------------\n");
    for (int i = 0; i < 30000; i++) nmodel_bp(test, output, input, length, 3);
    printf("----------------------------------\n\n");
    nmodel_print(test);
    printf("\nCost Function: %f\n", nmodel_costfunc(test, output, input, length));

    for (int i = 0; i < length; i++) {

        free(output[i]);
        free(input[i]);

    } free(output); free(input);

    fp = fopen("../../training_data/titanic_test.csv", "r");
    row = 0; column = 0;
    length = 418;
    output = malloc(8 * length); input = malloc(8 * length);

    for (int i = 0; i < length; i++) {

        output[i] = malloc(8);
        input[i] = malloc(8 * 3);

    }

    fgets(buffer, 1024, fp);
    while (fgets(buffer, 1024, fp) != NULL) {

        column = 0;
        char* value = strtok(buffer, ",");
        value = strtok(NULL, ",");
 
        while (value) {

            if (column == 0) input[row][0] = strtod(value, NULL);
            else if (column == 2) {
                value = strtok(NULL, ",");
                if (strcmp(value, "male") == 0) input[row][1] = -10;
                else input[row][1] = 10;
            }
            else if (column == 3) input[row][2] = strtod(value, NULL) / 10;
            else;
            value = strtok(NULL, ",");
            column++;

        }
 
       row++;

    }
 
    fclose(fp);

    for (int i = 0; i < length; i++) 
        if (nmodel_fp(test, input[i])[3][0] >= .5) output[i][0] = 1;
        else output[i][0] = 0;
    fp = fopen("../../training_data/titanic_results.csv", "w");
    fprintf(fp, "PassengerId,Survived\n"); int id = 891;
    for (int i = 0; i < length; i++) fprintf(fp, "%d,%d\n", id+=1, (int) output[i][0]);


}