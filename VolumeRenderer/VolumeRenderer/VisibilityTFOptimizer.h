#ifndef VISIBILITY_OPTIMIZER
#define VISIBILITY_OPTIMIZER

#include "VolumeDataset.h"
#include "TransferFunction.h"
#include "VisibilityHistogram.h"

class VisibilityTFOptimizer
{
public:
	std::vector<float> Es, Ev, Ec, energyFunc;

	void Init();
	void Optimize(VolumeDataset &volume, VisibilityHistogram &histogram, TransferFunction &transFunction);
};


#endif