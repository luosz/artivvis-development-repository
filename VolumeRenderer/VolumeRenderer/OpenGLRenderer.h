#ifndef OPENGL_RENDERER_H
#define OPENGL_RENDERER_H

#include "Renderer.h"

class OpenGLRenderer		:		public Renderer
{
public:

	void Init(int screenWidth, int screenHeight, VolumeDataset &volume);
	void Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera);

	
};

#endif