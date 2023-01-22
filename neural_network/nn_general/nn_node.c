//
// File: nn_node.c
//
// Description:
//
// @author: Dylan O'Hara
//
// // // // // // // // // // // // // // // // // // // // // // //

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

struct neural_network_node {

    double *theta;    ///< array of input weights
    int size;    ///< number of inputs

};

#include "nn_node.h"

///

NNode nnode_new(int size) {

    // randomization for value and theta
    NNode temp = malloc(sizeof(struct neural_network_node));
    temp->size = size;
    temp->theta = malloc(8 * size);
    for (int i = 0; i < size; i++) {
        temp->theta[i] = i % 2 == 0 ? ((rand() % 5) + 1) : -((rand() % 5) + 1);
        temp->theta[i] /= 1;
    }
    return temp;

}

///

double nnode_fp(NNode tb, double inputs[]) {

    double sum = 0;
    for (int i = 0; i < tb->size - 1; i++) sum += tb->theta[i] * inputs[i];
    sum += tb->theta[tb->size - 1] * 1; 
    return sum;

}

///

double nnode_costfunc(NNode tb) {

    double ret = 0;
    for (int i = 0; i < tb->size; i++) ret += tb->theta[i] * tb->theta[i];
    return ret;

}

///

void nnode_updatetheta(NNode tb, double *theta) {

    for (int i = 0; i < tb->size; i++) tb->theta[i] -= theta[i];

}

///

double nnode_theta(NNode tb, int pos) {

    return tb->theta[pos];

}

///

void nnode_delete(NNode tb) {
    
    free(tb->theta);
    free(tb);

}

///

int nnode_size(NNode tb) {

    return tb->size;

}

///

void nnode_print(NNode tb) {

    printf("{");
    for (int i = 0; i < tb->size; i++) {

        printf("%f", tb->theta[i]);
        if (i == tb->size - 1) printf("}"); else printf(" ");

    }

}
