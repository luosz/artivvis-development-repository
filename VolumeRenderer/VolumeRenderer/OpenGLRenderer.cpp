#include "OpenGLRenderer.h"

void OpenGLRenderer::Init(int screenWidth, int screenHeight, VolumeDataset &volume)
{
	raycaster = new GPURaycaster();
	contourDrawer = new GPUContours();

	raycaster->Init(screenWidth, screenHeight, volume);
	contourDrawer->Init(screenWidth, screenHeight, volume);

	transferFunction.LoadXML(" ");
}


void OpenGLRenderer::Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera)
{
	GLuint shaderProgramID = shaderManager.UseShader(shaderManager.currentShader);
	shaderProgramID = shaderManager.UseShader(SmokeShader);
	raycaster->Raycast(volume, shaderProgramID, camera);

//	contourDrawer->DrawContours(volume, camera, shaderManager, *raycaster);

	
}
