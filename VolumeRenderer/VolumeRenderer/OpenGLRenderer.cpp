#include "OpenGLRenderer.h"

OpenGLRenderer::OpenGLRenderer(int screenWidth, int screenHeight, VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera)
{
	volume.InitTexture();
	raycaster = new GPURaycaster(screenWidth, screenHeight, volume);
	transferFunction.Init(" ", volume);
}


void OpenGLRenderer::Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera)
{
	GLuint shaderProgramID = shaderManager.UseShader(shaderManager.currentShader);
	raycaster->Raycast(volume, transferFunction, shaderProgramID, camera);
}
