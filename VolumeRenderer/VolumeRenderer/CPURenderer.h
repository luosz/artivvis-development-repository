#ifndef CPU_RENDERER_H
#define CPU_RENDERER_H

#include "Renderer.h"

class CPURenderer		:		public Renderer
{
public:

	void Init(int screenWidth, int screenHeight, VolumeDataset &volume);
	void Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera);
};

#endif