#include "OpenGLRenderer.h"

void OpenGLRenderer::Init(int screenWidth, int screenHeight, VolumeDataset &volume)
{
	raycaster = new GPURaycaster();
	raycaster->Init(screenWidth, screenHeight, volume);

	transferFunction.Init(" ");
}


void OpenGLRenderer::Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera)
{
	GLuint shaderProgramID = shaderManager.UseShader(shaderManager.currentShader);
	raycaster->Raycast(volume, transferFunction, shaderProgramID, camera);
	
}
