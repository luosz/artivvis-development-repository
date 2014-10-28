#ifndef RAYCASTER_H
#define RAYCASTER_H

#include "ShaderManager.h"
#include "Camera.h"
#include "VolumeDataset.h"

class Raycaster
{
public:
	int maxRaySteps;
	float rayStepSize;
	float gradientStepSize;

	float minRange;
	float cutOff;
	float maxRange;

	std::vector<glm::vec2> opacityDivisions;
	std::vector<float> opacities;

	int numXPixels;
	int numYPixels;

	virtual void Init(int screenWidth, int screenHeight, VolumeDataset &volume) = 0;
	virtual void Raycast(VolumeDataset &volume, GLuint shaderProgramID, Camera &camera) = 0;
};

#endif