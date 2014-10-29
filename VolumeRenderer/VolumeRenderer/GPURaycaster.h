#ifndef GPU_RAYCASTER_H
#define GPU_RAYCASTER_H

#include <GL/glew.h>
#include <GL/freeglut.h>
#include "Camera.h"
#include "GLM.h"
#include <vector>
#include "VolumeDataset.h"
#include "XToon.h"
#include "Raycaster.h"
#include "GPUContours.h"

class GPURaycaster		:		public Raycaster
{
public:
	std::vector<GLuint> textures;
	XToon xToon;

	float contourThreshold;
	glm::vec3 lightPosition;


	void Init(int screenWidth, int screenHeight, VolumeDataset &volume);
	void GenerateTextures(VolumeDataset &volume);
	void Raycast(VolumeDataset &volume, TransferFunction &transferFunction, GLuint shaderProgramID, Camera &camera);
};

#endif

