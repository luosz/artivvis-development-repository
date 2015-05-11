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
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	camera.Update();

	GLuint shaderProgramID = ShaderManager::UseShader(TFShader);
	raycaster->Raycast(transferFunction, shaderProgramID, camera, volume->currTexture3D);
	


	glutSwapBuffers();
}


