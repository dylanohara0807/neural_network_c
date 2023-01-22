#ifndef _NLAYER_H_
#define _NLAYER_H_

typedef struct neural_network_layer *NLayer;

/// @brief create a new layer with size_curr nodes, each with
/// size_prev weights (including bias)
///
/// @param size_curr int, nodes in new layer
/// @param size_prev int, nodes in previous layer + 1
/// @return NLayer, the new layer

NLayer nlayer_new(int size_curr, int size_prev);

/// @brief perform forward propagation on one layer
///
/// @param tb NLayer, layer to fp on
/// @param inputs double[], input values
/// @param end bool, whether or not this is the output layer
/// @return double[], activations of the layer

double * nlayer_fp(NLayer tb, double inputs[], int end);

/// @brief perform backpropagation on the layer
///
/// @param tb NLayer, the layer to bp on
/// @param errors double[], the errors of the layer
/// @param prev_size int, number of nodes in the previous layer,
/// needed for number of weights in each node in tb
/// @return double[], errors for the previous layer

double * nlayer_bp(NLayer tb, double errors[], int prev_size);

/// @brief perform the theta regularization portion of cost function
///
/// @param tb NLayer, to compute
/// @return double, regularization term of tb

double nlayer_costfunc(NLayer tb);

/// @brief update the weights of layer
///
/// @param tb NLayer, to update
/// @param theta double[][], partial derivates to update with

void nlayer_updatetheta(NLayer tb, double **theta);

/// @brief returns malloc'd 2d array of the current weights in tb
///
/// @param tb NLayer, to return weights of
/// @param initd bool, 1 inits to zero, returns an empty 2d array,
/// 0 inits using actul weight values
/// @return double[][], 2d array of layer weights

double ** nlayer_nodetheta(NLayer tb, int initd);

/// @brief free spaces malloc'd for tb
///
/// @param tb NLayer, to free

void nlayer_delete(NLayer tb);

/// @brief return number of ndoes in tb
///
/// @param tb NLayer, to return from
/// @return int, number of nodes in tb (not including bias)

int nlayer_size(NLayer tb);

/// @brief print formatted weight values of tb
///
/// @param tb NLayer, to print from

void nlayer_print(NLayer tb);

#endif