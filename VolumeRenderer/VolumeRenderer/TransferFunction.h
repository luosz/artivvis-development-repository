#ifndef TRANSFER_FUNCTION_H
#define TRANSFER_FUNCTION_H

#include "tinyxml2.h"
#include <vector>
#include "GLM.h"
#include <iostream>
#include <GL/glew.h>
#include <GL/freeglut.h>


class TransferFunction
{
public:
	std::vector<glm::vec2> divisions;
	std::vector<glm::vec4> colors;
	std::vector<float> intensities;
	int numIntensities;
	GLuint tfTexture;

	std::vector<glm::vec4> colorTable;

	void Init(const char *filename);
	void LoadXML(const char *filename);
	void LoadLookup();

	glm::vec4 LERPColor(glm::vec4 firstColor, glm::vec4 secondColor, float firstIntensity, float secondIntensity, float currentIntensity);
};

#endif