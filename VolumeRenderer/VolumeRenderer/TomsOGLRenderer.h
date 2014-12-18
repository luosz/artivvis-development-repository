#ifndef TOMS_OGL_RENDERER_H
#define TOMS_OGL_RENDERER_H

#include "OpenGLRenderer.h"

class TomsOGLRenderer     :     public OpenGLRenderer
{
public:
	TomsOGLRenderer(int screenWidth, int screenHeight, VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera);
	void Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera);
};

#endif