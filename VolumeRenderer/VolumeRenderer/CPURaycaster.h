#ifndef CPU_RAYCASTER_H
#define CPU_RAYCASTER_H

#include <GL/glew.h>
#include <GL/freeglut.h>
#include "Camera.h"
#include "GLM.h"
#include <vector>
#include "VolumeDataset.h"
#include "XToon.h"
#include "Raycaster.h"
#include "GPUContours.h"

class CPURaycaster		:		public Raycaster
{
public:
	GLuint texture;

	glm::vec3 lightPosition;

	void Init(int screenWidth, int screenHeight, VolumeDataset &volume);
	void Raycast(VolumeDataset &volume, TransferFunction &transferFunction, GLuint shaderProgramID, Camera &camera);

	void GenerateTexture(int screenWidth, int screenHeight);

	glm::vec3 CalculateNormal(int x, int y, int z, VolumeDataset &volume);
	glm::vec4 CalculateLighting(glm::vec4 color, glm::vec3 N, glm::vec3 rayPosition);
	
	float ByteToFloat(VolumeDataset &volume, int index);

};

#endif

