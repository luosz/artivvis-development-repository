#include "VolumeRenderer.h"

GLuint GenerateTexture2D(int xPixels, int yPixels)
{
	GLuint tex;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, xPixels, yPixels, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);

	return tex;
}


void VolumeRenderer::Init(int screenWidth, int screenHeight)
{
	glEnable(GL_DEPTH_TEST);

	camera.Init(screenWidth, screenHeight);
	ShaderManager::Init();
	volume.Init();


	tempCoherence = new TempCoherence(screenWidth, screenHeight, volume);

	raycaster = new Raycaster(screenWidth, screenHeight, volume);

	transferFunction.Init(" ", volume);
}




void VolumeRenderer::Update()
{
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	camera.Update();
	volume.Update();

	GLuint shaderProgramID = ShaderManager::UseShader(TFShader);
	raycaster->Raycast(transferFunction, shaderProgramID, camera, volume.currTexture3D);
	


	glutSwapBuffers();
}


