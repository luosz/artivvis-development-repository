#include "ServerRenderer.h"



void VolumeRenderer::Init(int screenWidth, int screenHeight, VolumeDataset &volume_)
{
	glEnable(GL_DEPTH_TEST);

	camera.Init(screenWidth, screenHeight);
	ShaderManager::Init();
	volume = &volume_;

	raycaster = new Raycaster(screenWidth, screenHeight, volume_);

	transferFunction.Init(" ", volume_);
}




void VolumeRenderer::Update()
{	
	camera.Update();

	GLuint shaderProgramID = ShaderManager::UseShader(TFShader);
	raycaster->Raycast(transferFunction, shaderProgramID, camera, volume->currTexture3D);
}


