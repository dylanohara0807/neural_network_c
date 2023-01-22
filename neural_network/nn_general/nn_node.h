#ifndef _NNODE_H_
#define _NNODE_H_

typedef struct neural_network_node *NNode;

/// @brief create a new node with weights of size
///
/// @param size int, number of weights
/// @return NNode, new node

NNode nnode_new(int size);

/// @brief compute foward propagation for a single node
///
/// @pre inputs.length = size - 1 (no input for bias unit)
/// @param tb NNode, to fp on
/// @param inputs double[], inputs for the node
/// @return double, activation for this node

double nnode_fp(NNode tb, double inputs[]);

/// @brief compute cost function (theta regularization term)
///
/// @param tb NNode, to compute from
/// @return double, theta[i]^2 for all (0<i<size)

double nnode_costfunc(NNode tb);

/// @brief update each theta value with partial derivatives
///
/// @param tb NNode, to update
/// @param theta double[], partial derivatives

void nnode_updatetheta(NNode tb, double *theta);

/// @brief return weight value for node in previous layer (pos)
///
/// @param tb NNode, to get weight from
/// @param pos int, location of node from prev layer
/// @return double, weight value

double nnode_theta(NNode tb, int pos);

/// @brief free malloc'd space for tb
///
/// @param tb NNode, to free

void nnode_delete(NNode tb);

/// @brief NNode size
///
/// @param tb NNode, to get from
/// @return int, node size

int nnode_size(NNode tb);

/// @brief print weight values of tb
///
/// @param tb NNode, to print

void nnode_print(NNode tb);

#endif