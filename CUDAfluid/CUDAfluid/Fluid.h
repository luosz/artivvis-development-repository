#ifndef  FLUID_H
#define FLUID_H

#include "CudaHeaders.h"
#include "GLM.h"
#include "SimData.h"
#include "CellDataMatrix.h"
#include <vector>
#include "thrust\reduce.h"
#include "thrust\device_ptr.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust\inner_product.h>
#include <thrust\execution_policy.h>

class Fluid
{
public:
	SimData simData;
	SimData tempData;
	CellDataMatrix AMatrix;

	std::vector<float> hostDensities;
	std::vector<float> hostTemperatures;

	float *vortX, *vortY, *vortZ, *vortMag;

	void Init();
	void Update();

	void AdvectVelocity();
	void AddExternalForces();
	void Project();
	void AdvectTemperature();
	void AdvectDensity();

	void CalculateBuoyancy();
	void CalculateVorticity();
	void ConjugateGradient(CellDataMatrix &A, float *pressure, thrust::device_vector<float> &divergence, int maxIterations, float tolerance);
	void ApplyPreconditioner(CellDataMatrix &A, thrust::device_vector<float> &vector, thrust::device_vector<float> &results);
};



#endif // ! FLUID_H
