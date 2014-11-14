#ifndef TRANSFER_FUNCTION_H
#define TRANSFER_FUNCTION_H

#include "tinyxml2.h"
#include <vector>
#include "GLM.h"
#include <iostream>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "VolumeDataset.h"

#define tfEPSILON glm::epsilon<float>()

class TransferFunction
{
public:
	std::vector<glm::vec4> origColors;
	std::vector<glm::vec4> colors;
	std::vector<float> intensities;
	std::vector<int> frequencies;
	std::vector<float> weights;
	
	int numIntensities;
	GLuint tfTexture;
	float targetIntensity;
	int numVoxels;
	int numIterations;
	bool optimizeIntensity;

	VolumeDataset *volume;

	std::vector<glm::vec4> colorTable;

	void Init(const char *filename, VolumeDataset &volume_);
	void Update();

	void LoadXML(const char *filename);
	void LoadLookup();
	void IntensityOptimize();
	float GetWeightedAreaEntropy(int index);
	float GetWeightedEntropyOpacityByID(float intensity, int index);
	void CalculateFrequencies();
	float GetOpacityByInterp(float intensity, int index);
	float GetWeightByInterp(float intensity, int index);
	float GetWeight(int index);
	float GetWeightedNeighbourArea(int index);


	glm::vec4 LERPColor(glm::vec4 firstColor, glm::vec4 secondColor, float firstIntensity, float secondIntensity, float currentIntensity);
};

#endif