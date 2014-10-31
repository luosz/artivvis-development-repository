#ifndef GPU_RAYCASTER_H
#define GPU_RAYCASTER_H

#include <GL/glew.h>
#include <GL/freeglut.h>
#include "Camera.h"
#include "GLM.h"
#include <vector>
#include "VolumeDataset.h"
#include "TransferFunction.h"


class GPURaycaster
{
public:
	GLuint texID, tex2ID;

	int maxRaySteps;
	float rayStepSize;
	float gradientStepSize;

	glm::vec3 lightPosition;

	void Init(int screenWidth, int screenHeight);
	void Raycast(VolumeDataset &volume, TransferFunction &transferFunction, GLuint shaderProgramID, Camera &camera);
};

#endif

