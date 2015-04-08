#include "OpenGLRenderer.h"

OpenGLRenderer::OpenGLRenderer(int screenWidth, int screenHeight, VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera)
{
	volume.InitTexture();
	raycaster = new GPURaycaster(screenWidth, screenHeight, volume);
	transferFunction.Init(" ", volume);

	_optimizer = new VisibilityTFOptimizer(&volume, &visibilityHistogram, &transferFunction);
	_intensityTFOptimizerV2 = new IntensityTFOptimizerV2(&volume, &transferFunction, &visibilityHistogram);
//	_optimizer = new RegionVisibilityOptimizer(&volume, &transferFunction, raycaster, &shaderManager, &camera);
//	_optimizer = new IntensityTFOptimizer(&volume, &transferFunction);
}


void OpenGLRenderer::Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera)
{
	GLuint shaderProgramID = shaderManager.UseShader(shaderManager.currentShader);
	raycaster->Raycast(volume, transferFunction, shaderProgramID, camera);

	if (optimizer())
		optimizer()->Draw(shaderManager, camera);
}
