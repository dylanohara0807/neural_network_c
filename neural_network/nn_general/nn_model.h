#ifndef _NMODEL_H_
#define _NMODEL_H_

typedef struct neural_network_model *NModel;

/// @brief create a new neural network model with a size and number of inputs,
/// and automatically adds a bias unit to each layer.
///
/// @param size int, number of nodes
/// @param ninput int, number of inputs
/// @return NModel, the created neural model

NModel nmodel_new(int size, int ninput);

/// @brief insert a layer of specified size to the end of the given model
///
/// @param tb NModel, model to add layer to
/// @param size int, size of new layer

void nmodel_insert(NModel tb, int size);

/// @brief forward propagation function, returns a malloc'd 2d array of
/// the model activation values, including the output layer at [size - 1]
///
/// @pre inputs.length must be equal to size of first NModel layer
/// @param tb NModel, model to perform propagation on
/// @param inputs double[], inputs
/// @return double[][], malloc'd 2d array of activations

double ** nmodel_fp(NModel tb, double inputs[]);

/// @brief setup partial derivative accumalator, create threads, and calls 
/// thread_gradient with the created Grad_S struct for passing information.
///
/// @pre inputs[0].length = ninputs
/// @param tb NModel, model to backpropagate on
/// @param reals double[][], 2d array of real output values
/// @param inputs double[][], 2d array of input vales
/// @param length int, number of training examples
/// @param ninputs int, number of inputs

void nmodel_bp(NModel tb, double **reals, double **inputs, int length, int ninputs);

/// @brief calculate the cost function (binary cross entropy) for a
/// sigmoid activation output (not including regularlization terms)
///
/// @pre inputs[0].length = ninputs
/// @param tb NModel, model to backpropagate on
/// @param reals double[][], 2d array of real output values
/// @param inputs double[][], 2d array of input vales
/// @param length int, number of training examples
/// @return double, the cost function

double nmodel_costfunc(NModel tb, double **reals, double **inputs, double length);

/// @brief propogate on a give input and print the model activations
///
/// @pre inputs.length = number of inputs in tb
/// @param tb NModel, model to predict with
/// @param inputs double[], input values

void nmodel_predict(NModel tb, double inputs[]);

/// @brief free space malloc'd for NModel
///
/// @param tb NModel, model to free

void nmodel_delete(NModel tb);

/// @brief print the weights of NModel to a file
///
/// @pre ninputs = number of inputs in first layer of model
/// @param tb NModel, model to save weights of
/// @param filename char[], name of file
/// @param ninputs int, number of inputs
/// @param cfunc double, cost function associated with NModel

void nmodel_fileprint(NModel tb, char* filename, int ninputs, double cfunc);

void nmodel_print(NModel tb);

#endif