#ifndef RAYCASTER_H
#define RAYCASTER_H

#include <GL/glew.h>
#include <GL/freeglut.h>
#include "Camera.h"
#include "GLM.h"
#include <vector>
#include "VolumeDataset.h"
#include "TransferFunction.h"

class Raycaster
{
	public:
	int maxRaySteps;
	float rayStepSize;
	float gradientStepSize;
	glm::vec3 lightPosition;

	Raycaster(int screenWidth, int screenHeight, VolumeDataset &volume);
	void Raycast(TransferFunction &transferFunction, GLuint shaderProgramID, Camera &camera, GLuint tex3D);
};


#endif