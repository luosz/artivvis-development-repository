#ifndef TRANSFER_FUNCTION_H
#define TRANSFER_FUNCTION_H

#include "tinyxml2.h"
#include <vector>
#include "GLM.h"
#include <iostream>

class TransferFunction
{
public:
	std::vector<glm::vec2> divisions;
	std::vector<glm::vec3> colours;
	std::vector<float> intensities;
	std::vector<float> opacities;

	void LoadXML(const char *filename);
};

#endif