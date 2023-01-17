#ifndef _NMODEL_H_
#define _NMODEL_H_

typedef struct neural_network_model *NModel;

NModel nmodel_new(int size, int ninput);

void nmodel_insert(NModel tb, int size, int end);

double ** nmodel_fp(NModel tb, double inputs[]);

void nmodel_gradientcheck(NModel tb,  double **reals, double **inputs, int length, int ninputs);

void nmodel_bp(NModel tb, double **reals, double **inputs, int length, int ninputs);

double nmodel_costfunc(NModel tb, double **reals, double **inputs, double length);

void nmodel_predict(NModel tb, double inputs[]);

void nmodel_delete(NModel tb);

void nmodel_fileprint(NModel tb, char* filename, int ninputs, double cfunc);

void nmodel_print(NModel tb);

#endif