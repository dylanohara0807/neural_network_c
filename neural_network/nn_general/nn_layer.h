#ifndef _NLAYER_H_
#define _NLAYER_H_

typedef struct neural_network_layer *NLayer;

NLayer nlayer_new(int size_curr, int size_prev);

double * nlayer_fp(NLayer tb, double inputs[], int end);

double * nlayer_bp(NLayer tb, double errors[], int prev_size);

double nlayer_costfunc(NLayer tb);

void nlayer_updatetheta(NLayer tb, double **theta);

double ** nlayer_nodetheta(NLayer tb, int initd);

void nlayer_delete(NLayer tb);

int nlayer_size(NLayer tb);

void nlayer_print(NLayer tb);

#endif