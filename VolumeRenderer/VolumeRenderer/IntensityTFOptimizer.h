#ifndef INTENSITY_OPTIMIZER
#define INTENSITY_OPTIMIZER

#include "GLM.h"
#include <vector>
#include "TFOptimizer.h"

class IntensityTFOptimizer     :     public TFOptimizer
{
public:
	IntensityTFOptimizer(VolumeDataset *volume_, TransferFunction *transferFunction);
	void Optimize();


	std::vector<int> frequencies;
	std::vector<float> weights;
	int numVoxels;
	int numIterations;

	VolumeDataset *volume;
	TransferFunction *transferFunction;

	glm::vec4 *colors;
	float *intensities;
	float targetIntensity;
	int numIntensities;

	void OptimizeForIntensity();
	float GetWeightedAreaEntropy(int index);
	float GetWeightedEntropyOpacityByID(float intensity, int index);
	void CalculateFrequencies();
	float GetOpacityByInterp(float intensity, int index);
	float GetWeightByInterp(float intensity, int index);
	float GetWeight(int index);
	float GetWeightedNeighbourArea(int index);
};

#endif