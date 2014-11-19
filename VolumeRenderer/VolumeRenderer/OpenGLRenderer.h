#ifndef OPENGL_RENDERER_H
#define OPENGL_RENDERER_H

#include "Renderer.h"
#include "VisibilityHistogram.h"
#include "RegionVisibilityOptimizer.h"

class OpenGLRenderer		:		public Renderer
{
public:

	VisibilityHistogram visibilityHistogram;
	RegionVisibilityOptimizer regionOptimizer;

	void Init(int screenWidth, int screenHeight, VolumeDataset &volume);
	void Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera);
};

#endif