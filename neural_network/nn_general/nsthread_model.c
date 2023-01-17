//
// File: nn_model.c
//
// Description:
//
// @author: Dylan O'Hara
//
// // // // // // // // // // // // // // // // // // // // // // //

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nn_layer.h"

struct neural_network_model {
    
    NLayer *layer_list;    ///< array of layers
    int size;    ///< number of layers
    //double **activations;    ///< nn weight activations (not including inputs)

};

#include "nn_model.h"

NModel nmodel_new(int size, int ninput) {

    NModel temp = malloc(sizeof(struct neural_network_model));
    temp->layer_list = malloc(8);
    temp->layer_list[0] = nlayer_new(size, ninput + 1);
    temp->size = 1;
    return temp;

}

void nmodel_insert(NModel tb, int size, int end) {

    tb->layer_list = realloc(tb->layer_list, (tb->size + 1) * 8);
    tb->layer_list[tb->size] = nlayer_new(size, nlayer_size(tb->layer_list[tb->size - 1]) + 1);
    tb->size += 1;

}

double ** nmodel_fp(NModel tb, double inputs[]) {

    double **activations = malloc(8 * tb->size);

    activations[0] = nlayer_fp(tb->layer_list[0], inputs, 0);
    for (int i = 1; i < tb->size; i++)
        activations[i] = nlayer_fp(tb->layer_list[i], activations[i - 1], i == tb->size - 1 ? 1 : 0);
    return activations;

}

void nmodel_gradientcheck(NModel tb, double **reals, double **inputs, int length, int ninputs) {
    // init holders
    double ***zeros = malloc(8 * tb->size);
    for (int i = 0; i < tb->size; i++) zeros[i] = nlayer_nodetheta(tb->layer_list[i], 1);
    double ***der = malloc(8 * tb->size);
    for (int i = 0; i < tb->size; i++) der[i] = nlayer_nodetheta(tb->layer_list[i], 0);
    // compute approx derivatives
    for (int i = 0; i < tb->size; i++) {
        
        for (int j = 0; j < nlayer_size(tb->layer_list[tb->size - 1]); j++) {
            
            for (int k = 0; k < (i == 0 ? ninputs : nlayer_size(tb->layer_list[i - 1])); k++) {
                
                // J(theta + e)
                zeros[i][j][k] = -(1 / (1000000));
                nlayer_updatetheta(tb->layer_list[i], zeros[i]);
                double top = nmodel_costfunc(tb, reals, inputs, length);
                zeros[i][j][k] = 0;
                nlayer_updatetheta(tb->layer_list[i], zeros[i]);
                // J(theta - e);
                zeros[i][j][k] = (1 / (1000000));
                nlayer_updatetheta(tb->layer_list[i], zeros[i]);
                double bottom = nmodel_costfunc(tb, reals, inputs, length);
                zeros[i][j][k] = 0;
                nlayer_updatetheta(tb->layer_list[i], zeros[i]);
                // compute
                der[i][j][k] = (top + bottom) / (2000000);

            }

        }

    }
    // print approx derivatives
    for (int layer = 0; layer < tb->size; layer++) {

        for (int node = 0; node < nlayer_size(tb->layer_list[layer]); node++) {

            for (int theta = 0; theta < (layer == 0 ? ninputs : nlayer_size(tb->layer_list[layer - 1])); theta++) {

                printf("|%f|", der[layer][node][theta]); // TODO plus regularization

            }

            printf("---");

        }

        printf("\n");

    }
    // free holders
    for (int layer = 0; layer < tb->size; layer++) {

        for (int node = 0; node < nlayer_size(tb->layer_list[layer]); node++) {

            free(zeros[layer][node]); 
            free(der[layer][node]);

        }

        free(zeros[layer]);
        free(der[layer]);

    }
    free(zeros);
    free(der);
    // end
}

void nmodel_bp(NModel tb, double **reals, double **inputs, int length, int ninputs) {
    static int times = 0;
    // set up error and derivative holders
    double ***delta = malloc(8 * tb->size);
    for (int i = 0; i < tb->size; i++) delta[i] = nlayer_nodetheta(tb->layer_list[i], 1);
    double **errors = malloc(8 * tb->size);
    for (int i = 0; i < tb->size; i++) errors[i] = malloc(8 * nlayer_size(tb->layer_list[i]));
    // backpropagation 
    double l_rate = .001;
    for (int i = 0; i < length; i++) {

        double **activations = nmodel_fp(tb, inputs[i]);
        for (int j = 0; j < nlayer_size(tb->layer_list[tb->size - 1]); j++) 
            errors[tb->size - 1][j] = pow(activations[tb->size - 1][j] - reals[i][j], 2);

        for (int k = tb->size - 2; k >= 0; k--) { 

            double *t = nlayer_bp(tb->layer_list[k + 1], errors[k + 1], nlayer_size(tb->layer_list[k]));
            for (int l = 0; l < nlayer_size(tb->layer_list[k]); l++) 
                t[l] = t[l] * activations[k][l] * (1 - activations[k][l]);
            free(errors[k]);
            errors[k] = t;

        }

        for (int layer = 0; layer < tb->size; layer++) {

            for (int node = 0; node < nlayer_size(tb->layer_list[layer]); node++) {

                if (layer == 0)
                    for (int theta = 0; theta < ninputs + 1; theta++) {
                        double v = theta < ninputs ? inputs[i][theta] : 1;
                        delta[layer][node][theta] += tanh(v * errors[layer][node] / 1) * l_rate;
                    }
                else
                    for (int theta = 0; theta < nlayer_size(tb->layer_list[layer - 1]) + 1; theta++)
                        delta[layer][node][theta] += tanh(activations[layer - 1][theta] * errors[layer][node] / 1) * l_rate;

            }

        }

        for (int m = 0; m < tb->size; m++) free(activations[m]);
        free(activations);

    }
    // update partial derivatives (length and regularization) with alpha 
    for (int layer = 0; layer < tb->size; layer++) {
        for (int node = 0; node < nlayer_size(tb->layer_list[layer]); node++)
            for (int theta = 0; theta < (layer == 0 ? ninputs + 1 : nlayer_size(tb->layer_list[layer - 1]) + 1); theta++)
                delta[layer][node][theta] /= (length); // TODO plus regularization
    }
    // print delta (partial derivatives)
    if (times % 99 == 1000) { int spot = 1;
    for (int layer = 0; layer < tb->size; layer++) {

        for (int node = 0; node < nlayer_size(tb->layer_list[layer]); node++) {

            if (layer == tb->size - 1) spot = 0;
            for (int theta = 0; theta < (layer == 0 ? ninputs + 1 : nlayer_size(tb->layer_list[layer - 1]) + spot); theta++) {

                printf("|%f|", delta[layer][node][theta]); // TODO plus regularization

            }

            printf("---");

        }

        printf("\n");

    } }
    times++;
    // update neural network theta values and free holders
    for (int i = 0; i < tb->size; i++) { 
        nlayer_updatetheta(tb->layer_list[i], delta[i]); 
        for (int j = 0; j < nlayer_size(tb->layer_list[i]); j++) free(delta[i][j]);
        free(delta[i]); 
        free(errors[i]);
    }
    free(delta);
    free(errors);
    // end
}

double nmodel_costfunc(NModel tb, double **reals, double **inputs, double length) {
    static int rep = 0;

    double ret = 0; double **activations; int tex = 0; 

    for (int i = 0; i < length; i++) {

        for (int k = 0; k < nlayer_size(tb->layer_list[tb->size - 1]); k++) {

            activations = nmodel_fp(tb, inputs[i]);
            //ret += (reals[i][k] * log(t[k])) + ((1 - reals[i][k]) * log(1 - t[k])); // Binary Cross Entropy
            //if (reals[i][k] < .5 && t[k] < .5) tex++;
            //if (reals[i][k] > .5 && t[k] > .5) tex++;
            ret += pow(reals[i][k] - activations[tb->size - 1][k], 2); // Mean Squared Error
            if (rep % 10000 == 0) {printf("\n%f;", reals[i][k]); printf("%f\n\n", activations[tb->size - 1][k]);}
            for (int m = 0; m < tb->size; m++) free(activations[m]);
            free(activations);

        } rep++;

    }

    ret = (1 / length) * ret;
    //double isum = 0;
    //for (int l = 0; l < tb->size - 1; l++) isum += nlayer_costfunc(tb->layer_list[l]);
    //ret += (1 / (2 * length)) * isum; regularized value (theta^2)
    return ret;

}

void nmodel_delete(NModel tb) {

    for (int i = 0; i < tb->size; i++) nlayer_delete(tb->layer_list[i]);
    free(tb->layer_list);
    free(tb);

}

void nmodel_fileprint(NModel tb, char* filename, int ninputs, double cfunc) {

    FILE* fp = fopen(filename, "w");
    double ***delta = malloc(8 * tb->size);
    for (int i = 0; i < tb->size; i++) delta[i] = nlayer_nodetheta(tb->layer_list[i], 0);

    for (int layer = 0; layer < tb->size; layer++)
        for (int node = 0; node < nlayer_size(tb->layer_list[layer]); node++)
            for (int theta = 0; theta < (layer == 0 ? ninputs + 1 : nlayer_size(tb->layer_list[layer - 1]) + 1); theta++)
                fprintf(fp, "%f,", delta[layer][node][theta]);
    fprintf(fp, "Cost Function = %f\n", cfunc);
    for (int i = 0; i < tb->size; i++) { 
        for (int j = 0; j < nlayer_size(tb->layer_list[i]); j++) free(delta[i][j]);
        free(delta[i]); 
    }
    free(delta);
    fclose(fp);

}

void nmodel_print(NModel tb) {

    printf("Neural Network Model:\n");
    for (int i = 0; i < tb->size; i++) {

        nlayer_print(tb->layer_list[i]);
        printf("\n---\n");

    }

}

