#ifndef RAYCASTER_H
#define RAYCASTER_H

#include "ShaderManager.h"
#include "Camera.h"
#include "VolumeDataset.h"
#include "TransferFunction.h"

class Raycaster
{
public:
	int maxRaySteps;
	float rayStepSize;
	float gradientStepSize;

	float minRange;
	float cutOff;
	float maxRange;

	float clipPlaneDistance;
	glm::vec3 clipPlaneNormal;

	std::vector<glm::vec2> opacityDivisions;
	std::vector<float> opacities;

	int numXPixels;
	int numYPixels;

	virtual void Raycast(VolumeDataset &volume, TransferFunction &transferFunction, GLuint shaderProgramID, Camera &camera) = 0;
};

#endif