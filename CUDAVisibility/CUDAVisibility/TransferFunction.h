#ifndef TRANSFER_FUNCTION_H
#define TRANSFER_FUNCTION_H

#include "tinyxml2.h"
#include <vector>
#include "GLM.h"
#include <iostream>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "VolumeDataset.h"
#include "IntensityTFOptimizer.h"
#include "VisibilityTFOptimizer.h"
#include "VisibilityHist.h"

class TransferFunction
{
public:
	std::vector<glm::vec4> colors;
	std::vector<float> intensities;
	int numIntensities;
	GLuint tfTexture;

	float targetIntensity;
	TFOptimizer *optimizer;

	std::vector<glm::vec4> colorTable;

	void Init(const char *filename, VolumeDataset &volume_);
	void LoadXML(const char *filename);
	void LoadLookup();

	void Optimize(VisibilityHistogram &histogram);

	glm::vec4 LERPColor(glm::vec4 firstColor, glm::vec4 secondColor, float firstIntensity, float secondIntensity, float currentIntensity);
};

#endif