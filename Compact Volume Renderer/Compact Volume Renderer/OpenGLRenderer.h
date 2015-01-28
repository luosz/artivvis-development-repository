#ifndef OPENGL_RENDERER_H
#define OPENGL_RENDERER_H

#include "VolumeDataset.h"
#include "ShaderManager.h"
#include "GPURaycaster.h"
#include "TransferFunction.h"

class OpenGLRenderer
{
public:
	GPURaycaster *raycaster;
	TransferFunction transferFunction;
	GLuint currTexture3D;
	int textureSize;

	OpenGLRenderer(int screenWidth, int screenHeight, VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera);
	void UpdateTexture(int currentTimestep, VolumeDataset &volume);
	GLuint GenerateTexture(VolumeDataset &volume);
	void Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera);
};

#endif