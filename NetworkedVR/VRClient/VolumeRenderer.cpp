#include "VolumeRenderer.h"



void VolumeRenderer::Init(int screenWidth, int screenHeight)
{
	glEnable(GL_DEPTH_TEST);

	camera.Init(screenWidth, screenHeight);
	ShaderManager::Init();

	

	raycaster = new Raycaster(screenWidth, screenHeight, volume);

	transferFunction.Init(" ", volume);
}




void VolumeRenderer::Update()
{
	
	camera.Update();

	if (volume.timesteps > 0)
	{
		volume.CopyToTexture();

		GLuint shaderProgramID = ShaderManager::UseShader(TFShader);
		raycaster->Raycast(transferFunction, shaderProgramID, camera, volume.currTexture3D);
	}
}


