#ifndef _NNODE_H_
#define _NNODE_H_

typedef struct neural_network_node *NNode;

NNode nnode_new(int size);

double nnode_fp(NNode tb, double inputs[]);

double nnode_costfunc(NNode tb);

void nnode_updatetheta(NNode tb, double *theta);

double nnode_theta(NNode tb, int pos);

void nnode_delete(NNode tb);

int nnode_size(NNode tb);

void nnode_print(NNode tb);

#endif