#ifndef JOES_OGL_RENDERER_H
#define JOES_OGL_RENDERER_H

#include <memory>
#include "OpenGLRenderer.h"

class JoesOGLRenderer : public OpenGLRenderer
{
public:
	//std::shared_ptr<IntensityTFOptimizerV2> intensityOptimizerV2;

	JoesOGLRenderer(int screenWidth, int screenHeight, VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera);
	void Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera);
};

#endif
