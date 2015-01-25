#ifndef IntensityTFOptimizerV2_h
#define IntensityTFOptimizerV2_h

#include "TFOptimizer.h"

class IntensityTFOptimizerV2     :     public TFOptimizer
{
public:
	IntensityTFOptimizerV2(VolumeDataset *volume_, TransferFunction *transferFunction_, VisibilityHistogram *visibilityHistogram_);
	void Optimize();


	std::vector<int> frequencies;
	std::vector<float> weights;
	int numVoxels;
	int numIterations;

	VolumeDataset *volume;
	TransferFunction *transferFunction;
	VisibilityHistogram *visibilityHistogram;

	glm::vec4 *colors;
	float *intensities;
	float targetIntensity;
	int numIntensities;
	
	void OptimizeForIntensity();

	float GetEntropyOpacityByID(float intensity, int index);
	float GetVisibilityOpacityByID(float intensity, int index);
	float GetAreaEntropy(int index);
	float GetAreaVisibility(int index);
	void BalanceEdges();
	
	void BalanceVisibilityOnce();
	void BalanceVisibility();

	void CalculateFrequencies();
	float GetOpacityByInterp(float intensity, int index);
	float GetWeightedEntropyOpacityByID(float intensity, int index);
	float GetWeightByInterp(float intensity, int index);
	float GetWeight(int index);
	float GetWeightedAreaEntropy(int index);

	void BeginOfOptimization()
	{
		memcpy(&transferFunction->colors[0], &transferFunction->origColors[0], transferFunction->numIntensities * sizeof(glm::vec4));
		numIntensities = transferFunction->numIntensities;
		colors = &transferFunction->colors[0];
		intensities = &transferFunction->intensities[0];
		targetIntensity = transferFunction->targetIntensity;
	}

	void BeginOfOptimizationNoCopy()
	{
		numIntensities = transferFunction->numIntensities;
		colors = &transferFunction->colors[0];
		intensities = &transferFunction->intensities[0];
		targetIntensity = transferFunction->targetIntensity;
	}

	void EndOfOptimization()
	{
		transferFunction->LoadLookup(transferFunction->currentColorTable);
	}
};

#endif // IntensityTFOptimizerV2_h
