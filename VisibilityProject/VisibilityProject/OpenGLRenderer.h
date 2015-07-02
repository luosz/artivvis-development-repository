#ifndef OPENGL_RENDERER_H
#define OPENGL_RENDERER_H

#include "VisibilityHistogram.h"
#include "VisibilityTFOptimizer.h"
#include "VolumeDataset.h"
#include "TransferFunction.h"
#include "Raycaster.h"

class OpenGLRenderer
{
public:
	TransferFunction transferFunction;
	VisibilityHistogram visibilityHistogram;
	Raycaster *raycaster;
	VisibilityTFOptimizer *optimizer;

	OpenGLRenderer(int screenWidth, int screenHeight, VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera);
	void Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera);
};

#endif