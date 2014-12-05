#ifndef INTENSITY_OPTIMIZER
#define INTENSITY_OPTIMIZER

#include "GLM.h"
#include <vector>
#include "VolumeDataset.h"

class IntensityTFOptimizer
{
public:
	std::vector<int> frequencies;
	std::vector<float> weights;
	int numVoxels;
	int numIterations;

	VolumeDataset *volume;
	glm::vec4 *colors;
	float *intensities;
	int numIntensities;

	IntensityTFOptimizer(VolumeDataset &volume_, int &numIntensities_, glm::vec4 *colors_, float *intensities_);

	virtual void Optimize(float targetIntensity);

	float GetWeightedAreaEntropy(int index);
	float GetWeightedEntropyOpacityByID(float intensity, int index);
	void CalculateFrequencies();
	float GetOpacityByInterp(float intensity, int index);
	float GetWeightByInterp(float intensity, int index);
	float GetWeight(int index);
	float GetWeightedNeighbourArea(int index);
};

#endif