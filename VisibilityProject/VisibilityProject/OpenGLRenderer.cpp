#include "OpenGLRenderer.h"

OpenGLRenderer::OpenGLRenderer(int screenWidth, int screenHeight, VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera)
{
	volume.InitTexture();
	raycaster = new Raycaster(screenWidth, screenHeight, volume);
	transferFunction.Init(" ", volume);
	visibilityHistogram.Init(screenWidth, screenHeight);
	optimizer = new VisibilityTFOptimizer(&volume, &visibilityHistogram, &transferFunction);
}


void OpenGLRenderer::Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera)
{
	visibilityHistogram.CalculateHistogram(volume, transferFunction.tfTexture, shaderManager, camera);
	optimizer->Optimize();

	GLuint shaderProgramID = shaderManager.UseShader(shaderManager.currentShader);
	raycaster->Raycast(volume, transferFunction, shaderProgramID, camera);

	visibilityHistogram.DrawHistogram(shaderManager, camera);
	optimizer->Draw(shaderManager, camera);
}