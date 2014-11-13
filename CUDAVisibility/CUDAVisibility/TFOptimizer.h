#ifndef TF_OPTIMIZER
#define TF_OPTIMIZER

#include "GLM.h"
#include <vector>
#include "VolumeDataset.h"
#include "VisibilityHist.h"

class TFOptimizer
{
public:
	virtual void Optimize(float targetIntensity, VisibilityHistogram histogram) { }
};


#endif