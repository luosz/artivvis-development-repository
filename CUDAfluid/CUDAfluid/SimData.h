#ifndef SIMDATA_H
#define SIMDATA_H

#include "GLM.h"
#include "CudaHeaders.h"
#include "Constants.h"

class SimData
{
public:
	int *gridRes;
	int *numCells;

	int *xFaceRes;
	int *yFaceRes;
	int *zFaceRes;

	int *numFaces;

	float *dt;
	float *dx;

	float *xVelocities;
	float *yVelocities;
	float *zVelocities;

	float *pressures;
	float *densities;
	float *temperatures;

	void Init();
};

#endif