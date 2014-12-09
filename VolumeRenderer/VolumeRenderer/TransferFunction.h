#ifndef TRANSFER_FUNCTION_H
#define TRANSFER_FUNCTION_H

#include "tinyxml2.h"
#include <vector>
#include "GLM.h"
#include <iostream>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "VolumeDataset.h"
#include "VisibilityHistogram.h"
#include "IntensityTFOptimizer.h"
#include "IntensityTFOptimizerV2.h"
#include "../TransferFunctionEditor/AbstractGraphicsView.h"

#define tfEPSILON glm::epsilon<float>()

class TransferFunction
{
public:
	std::vector<glm::vec4> origColors;
	std::vector<glm::vec4> colors;
	std::vector<float> intensities;
	
	int numIntensities;
	GLuint tfTexture;
	float targetIntensity;
	bool optimizeIntensity;

	IntensityTFOptimizer *intensityOptimizer;
	IntensityTFOptimizerV2 *intensityOptimizerV2;
	AbstractGraphicsView *tfView;
	AbstractGraphicsView *visibilityView;

	std::vector<glm::vec4> origColorTable;
	std::vector<glm::vec4> currentColorTable;

	void Init(const char *filename, VolumeDataset &volume_);
	void Update();

	void LoadXML(const char *filename);
	void LoadLookup(std::vector<glm::vec4> &colorTable);
	void CopyToTex(std::vector<glm::vec4> &data);

	glm::vec4 LERPColor(glm::vec4 firstColor, glm::vec4 secondColor, float firstIntensity, float secondIntensity, float currentIntensity);
};

#endif