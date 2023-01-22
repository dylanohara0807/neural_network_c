//
// File: nn_layer.c
//
// Description:
//
// @author: Dylan O'Hara
//
// // // // // // // // // // // // // // // // // // // // // // //

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nn_node.h"

struct neural_network_layer {

    NNode *node_list;    ///< array of nodes
    int size;    ///< number of nodes

};

#include "nn_layer.h"

///

NLayer nlayer_new(int size_curr, int size_prev) {

    NLayer temp = malloc(sizeof(struct neural_network_layer));
    temp->node_list = malloc(8 * size_curr);
    for (int i = 0; i < size_curr; i++) temp->node_list[i] = nnode_new(size_prev);
    temp->size = size_curr;
    return temp;

}

///

double * nlayer_fp(NLayer tb, double inputs[], int end) {

    double *n;
    if (end != 1 ) n = malloc(8 * tb->size + 1); 
    else n = malloc(8 * tb->size);
    double t = 0;

    for (int i = 0; i < tb->size; i++) {
        
        t = nnode_fp(tb->node_list[i], inputs);
        //printf("|||%f|||", t);
        if (end == 1)
            //n[i] = t; // Linear
            n[i] = (1 / (1 + exp(-t))); // Sigmoid
            //n[i] = t; // ReLu
            //n[i] = 10*tanh(t/10); // Tanh (Stretched X and Y)
            //n[i] = fmin(fmax(t, -10), 10); // Clipped Output
        else
            //n[i] = .5 * t; // Linear
            n[i] = (1 / (1 + exp(-t))); // Sigmoid
            //n[i] = t > 0 ? t : 0; // ReLu
            //n[i] = t > 0 ? sqrt(t) / 1 : -sqrt(-t) / 1;
            

    }
    //printf("---\n");
    if (end != 1) n[tb->size] = 1;
    return n;

}

///

double * nlayer_bp(NLayer tb, double errors[], int prev_size) {

    double *sum = malloc(8 * prev_size);
    for (int i = 0; i < prev_size; i++) { 

        sum[i] = 0;
        for (int j = 0; j < tb->size; j++)
            sum[i] += nnode_theta(tb->node_list[j], i) * errors[j];

    }

    return sum;

}

///

double nlayer_costfunc(NLayer tb) {

    double ret = 0;
    for (int i = 0; i < tb->size; i++) ret += nnode_costfunc(tb->node_list[i]);
    return ret;

}

///

void nlayer_updatetheta(NLayer tb, double **theta) {

    for (int i = 0; i < tb->size; i++) nnode_updatetheta(tb->node_list[i], theta[i]);

}

///

double ** nlayer_nodetheta(NLayer tb, int initd) {

    double **ret = malloc(8 * tb->size);

    for (int i = 0; i < tb->size; i++) {

        ret[i] = malloc(8 * nnode_size(tb->node_list[i]));
        for (int j = 0; j < nnode_size(tb->node_list[i]); j++)
            ret[i][j] = initd == 1 ? 0 : nnode_theta(tb->node_list[i], j);
            
    }

    return ret;

}

///

void nlayer_delete(NLayer tb) {
    
    for (int i = 0; i < tb->size; i++) nnode_delete(tb->node_list[i]);
    free(tb->node_list);
    free(tb);

}

///

int nlayer_size(NLayer tb) {

    return tb->size;

}

///

void nlayer_print(NLayer tb) {

    for (int i = 0; i < tb->size; i++) {

        nnode_print(tb->node_list[i]);
        if (i == tb->size - 1) printf("\n"); else printf(" || ");

    }

}
