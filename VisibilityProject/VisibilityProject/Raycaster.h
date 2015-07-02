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

	int numXPixels;
	int numYPixels;

	glm::vec3 lightPosition;

	Raycaster() { }
	Raycaster(int screenWidth, int screenHeight, VolumeDataset &volume);
	void Raycast(VolumeDataset &volume, TransferFunction &transferFunction, GLuint shaderProgramID, Camera &camera);
};

#endif