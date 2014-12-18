#ifndef OPENGL_RENDERER_H
#define OPENGL_RENDERER_H

#include "Renderer.h"
#include "VisibilityHistogram.h"
#include "RegionVisibilityOptimizer.h"
#include "VisibilityTFOptimizer.h"
#include "IntensityTFOptimizer.h"
#include "IntensityTFOptimizerV2.h"

class OpenGLRenderer		:		public Renderer
{
public:
	OpenGLRenderer(int screenWidth, int screenHeight, VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera);
	virtual void Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera);
};

#endif