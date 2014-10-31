#ifndef VISIBILITY_RENDERER_H
#define VISIBILITY_RENDERER_H

#include "Renderer.h"

class VisibilityRenderer		:		public Renderer
{
public:

	void Init(int screenWidth, int screenHeight, VolumeDataset &volume);
	void Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera);

	
};

#endif