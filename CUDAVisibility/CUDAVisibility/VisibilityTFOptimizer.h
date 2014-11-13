#ifndef VISIBILITY_OPTIMIZER
#define VISIBILITY_OPTIMIZER

#include "TFOptimizer.h"

class VisibilityTFOptimizer  :  public TFOptimizer
{
public:
	VolumeDataset *volume;
	glm::vec4 *colors;
	float *intensities;
	int numIntensities;
	int numVoxels;

	VisibilityTFOptimizer(VolumeDataset &volume_, int numIntensities_, glm::vec4 *colors_, float *intensities_);

	void Optimize(float targetIntensity, VisibilityHistogram histogram);

};

#endif