#ifndef VISIBILITY_RAYCASTER_H
#define VISIBILITY_RAYCASTER_H

#include <GL/glew.h>
#include <GL/freeglut.h>
#include "Camera.h"
#include "GLM.h"
#include <vector>
#include "VolumeDataset.h"
#include "XToon.h"
#include "Raycaster.h"
#include "GPUContours.h"

class VisibilityRaycaster		:		public Raycaster
{
public:
	glm::vec3 lightPosition;

	GLuint histTexture;


	void Init(int screenWidth, int screenHeight, VolumeDataset &volume);
	void Raycast(VolumeDataset &volume, TransferFunction &transferFunction, GLuint shaderProgramID, Camera &camera);
};

#endif

