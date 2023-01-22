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
#include <pthread.h>
#include "nn_layer.h"

struct neural_network_model {
    
    NLayer *layer_list;    ///< array of layers
    int size;    ///< number of layers

};

#include "nn_model.h"

struct Grad_S {

    NModel tb;    ///< model to update
    double **reals;    ///< target values
    double **inputs;    ///< input values
    int length;    ///< number of examples
    int ninputs;    ///< number of inputs
    double ***delta;    ///< accumulated gradients

}; typedef struct Grad_S *GS;

///

NModel nmodel_new(int size, int ninput) {

    NModel temp = malloc(sizeof(struct neural_network_model));
    temp->layer_list = malloc(8);
    temp->layer_list[0] = nlayer_new(size, ninput + 1);
    temp->size = 1;
    return temp;

}

///

void nmodel_insert(NModel tb, int size) {

    tb->layer_list = realloc(tb->layer_list, (tb->size + 1) * 8);
    tb->layer_list[tb->size] = nlayer_new(size, nlayer_size(tb->layer_list[tb->size - 1]) + 1);
    tb->size += 1;

}

///

double ** nmodel_fp(NModel tb, double inputs[]) {

    double **activations = malloc(8 * tb->size);

    activations[0] = nlayer_fp(tb->layer_list[0], inputs, 0);
    for (int i = 1; i < tb->size; i++)
        activations[i] = nlayer_fp(tb->layer_list[i], activations[i - 1], i == tb->size - 1 ? 1 : 0);
    return activations;

}

// Counter for input location in multi-threaded training
static int data_loc = 0;
/// Thread for locking/unlocking
static pthread_mutex_t muthread = PTHREAD_MUTEX_INITIALIZER;

/// @brief Performs gradient descent using multiple threads. Utilizes struct
/// to pass information to function, including the NModel, length of examples
/// inputs, real outputs, and number of inputs. Performs full batch gradient descent.
///
/// @param arg (*void) *Grad_S, struct containing information for gradient descent
/// @return NULL, when finished

static void * thread_gradient(void *arg) {

    GS data = arg;
    NModel tb = data->tb; int length = data->length; double **reals = data->reals;
    double **inputs = data->inputs; int ninputs = data->ninputs;
    double **errors = malloc(8 * tb->size);
    for (int i = 0; i < tb->size; i++) errors[i] = malloc(8 * nlayer_size(tb->layer_list[i]));
    // backpropagation
    double l_rate = 1;
    while (data_loc < length) {

        pthread_mutex_lock(&muthread);
        int i = data_loc;
        if (i >= length) {pthread_mutex_unlock(&muthread); break;}
        data_loc++;
        pthread_mutex_unlock(&muthread);

        double **activations = nmodel_fp(tb, inputs[i]);
        for (int j = 0; j < nlayer_size(tb->layer_list[tb->size - 1]); j++) 
            errors[tb->size - 1][j] = activations[tb->size - 1][j] - reals[i][j];

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
                        data->delta[layer][node][theta] += (v * errors[layer][node]) * l_rate;
                    }
                else
                    for (int theta = 0; theta < nlayer_size(tb->layer_list[layer - 1]) + 1; theta++)
                        data->delta[layer][node][theta] += (activations[layer - 1][theta] * errors[layer][node]) * l_rate;

            }

        }

        for (int m = 0; m < tb->size; m++) free(activations[m]);
        free(activations);

    }

    for (int i = 0; i < tb->size; i++) free(errors[i]);
    free(errors);
    return NULL;

}

///

void nmodel_bp(NModel tb, double **reals, double **inputs, int length, int ninputs) {

    static int times = 0;
    data_loc = 0;
    // set up error and derivative holders
    double ***delta = malloc(8 * tb->size);
    for (int i = 0; i < tb->size; i++) delta[i] = nlayer_nodetheta(tb->layer_list[i], 1);
    // init struct for args and create/run threads
    GS arg = malloc(sizeof(struct Grad_S)); int tc = 100;
    arg->tb = tb; arg->reals = reals; arg->inputs = inputs; 
    arg->length = length; arg->ninputs = ninputs; arg->delta = delta;
    pthread_t thread_list[tc];
    for (int i = 0; i < tc; i++) {

        pthread_t temp;
        pthread_create(&temp, NULL, thread_gradient, (void *) arg);
        thread_list[i] = temp;

    }
    // join threads
    for (int i = 0; i < tc; i++) pthread_join(thread_list[i], NULL);
    free(arg);
    // update partial derivatives (length and regularization) with alpha 
    for (int layer = 0; layer < tb->size; layer++) {
        for (int node = 0; node < nlayer_size(tb->layer_list[layer]); node++)
            for (int theta = 0; theta < (layer == 0 ? ninputs + 1 : nlayer_size(tb->layer_list[layer - 1]) + 1); theta++)
                delta[layer][node][theta] /= (length); // TODO plus regularization
    }
    // print delta (partial derivatives)
    if (times % 2 == 3) { int spot = 1; nmodel_print(tb);
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
    }
    free(delta);
    // end
}

///

double nmodel_costfunc(NModel tb, double **reals, double **inputs, double length) {
    static int rep = 0;

    double ret = 0; double **activations; int tex = 0; 

    for (int i = 0; i < length; i++) {

        for (int k = 0; k < nlayer_size(tb->layer_list[tb->size - 1]); k++) {

            //if (rep % 100 == 0) {

                activations = nmodel_fp(tb, inputs[i]);
                ret += fabs((reals[i][k] * log(activations[tb->size - 1][k])) + ((1 - reals[i][k]) * log(1 - activations[tb->size - 1][k]))); // Binary Cross Entropy
                //if (reals[i][k] < .5 && activations[tb->size - 1][k] < .5) tex++;
                //if (reals[i][k] > .5 && activations[tb->size - 1][k] > .5) tex++;
                //ret += pow(log(reals[i][k]) - log(activations[tb->size - 1][k]), 2); // Mean Squared Error

                //if (rep % 100 == 0) {printf("\n%f | ", reals[i][k]); printf("%f\n", activations[tb->size - 1][k]);}
                for (int m = 0; m < tb->size; m++) free(activations[m]);
                free(activations);

            //}

        } rep++;

    }

    ret = (1 / (length)) * ret;
    //double isum = 0;
    //for (int l = 0; l < tb->size - 1; l++) isum += nlayer_costfunc(tb->layer_list[l]);
    //ret += (1 / (2 * length)) * isum; regularized value (theta^2)
    return ret;

}

///

void nmodel_predict(NModel tb, double inputs[]) {

    double **act = nmodel_fp(tb, inputs);
    for (int layer = 0; layer < tb->size; layer++) {
        for (int node = 0; node < nlayer_size(tb->layer_list[layer]); node++) 
            printf("|%f|", act[layer][node]);
        printf("\n");
    }
    for (int m = 0; m < tb->size; m++) free(act[m]);
        free(act);

}

///

void nmodel_delete(NModel tb) {

    for (int i = 0; i < tb->size; i++) nlayer_delete(tb->layer_list[i]);
    free(tb->layer_list);
    free(tb);

}

///

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

///

void nmodel_print(NModel tb) {

    printf("Neural Network Model:\n");
    for (int i = 0; i < tb->size; i++) {

        nlayer_print(tb->layer_list[i]);
        printf("\n---\n");

    }

}

