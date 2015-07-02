#ifndef OPENGL_RENDERER_H
#define OPENGL_RENDERER_H

#include "VolumeDataset.h"
#include "ShaderManager.h"
#include "GPURaycaster.h"
#include "TransferFunction.h"
#include "ClipPlane.h"

class OpenGLRenderer
{
public:
	GPURaycaster *raycaster;
	TransferFunction transferFunction;

	float tfBandWidth;
	float tfBandPos;

	GLuint currTexture3D;
	int textureSize;

	OpenGLRenderer(int screenWidth, int screenHeight, VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera);
	void UpdateTexture(int currentTimestep, VolumeDataset &volume);
	GLuint GenerateTexture(VolumeDataset &volume);
	void Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera, ClipPlane &clipPlane, bool focused, bool removed, float sphereRadius, glm::vec3 &spherePoint);
};

#endif