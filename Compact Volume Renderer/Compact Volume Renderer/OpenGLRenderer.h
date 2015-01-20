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

	OpenGLRenderer(int screenWidth, int screenHeight, VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera);
	void Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera);
};

#endif