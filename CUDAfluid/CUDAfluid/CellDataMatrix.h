#ifndef CELLDATA_MATRIX_H
#define CELLDATA_MATRIX_H

#include "CudaHeaders.h"
#include "Constants.h"

class CellDataMatrix
{
public:
	float *diag;
	float *plusI;
	float *plusJ;
	float *plusK;

	void Init();
};

#endif