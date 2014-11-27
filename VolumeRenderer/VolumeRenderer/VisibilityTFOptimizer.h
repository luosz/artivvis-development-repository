#ifndef VISIBILITY_OPTIMIZER
#define VISIBILITY_OPTIMIZER

#include "VolumeDataset.h"
#include "TransferFunction.h"
#include "VisibilityHistogram.h"

class VisibilityTFOptimizer
{
public:
	std::vector<float> Es, Ev, Ec, energyFunc;
	int iterations;

	void Init();
	void Optimize(VolumeDataset &volume, VisibilityHistogram &histogram, TransferFunction &transFunction);
	void DrawEnergy(ShaderManager shaderManager, Camera &camera);
};


#endif