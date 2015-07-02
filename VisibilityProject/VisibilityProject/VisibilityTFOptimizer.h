#ifndef VISIBILITY_OPTIMIZER
#define VISIBILITY_OPTIMIZER

#include "VisibilityHistogram.h"
#include "TransferFunction.h"
#include "GLM.h"

class VisibilityTFOptimizer
{
public:
	VisibilityTFOptimizer(VolumeDataset *volume_, VisibilityHistogram *visibilityHistogram_, TransferFunction *transferFunction_);
	void Optimize();
	void Draw(ShaderManager &shaderManager, Camera &camera);


	std::vector<float> Es, Ev, Ec, energyFunc;
	int iterations;
	int numBins;

	VolumeDataset *volume;
	TransferFunction *transferFunction;
	VisibilityHistogram *visibilityHistogram;	
};


#endif