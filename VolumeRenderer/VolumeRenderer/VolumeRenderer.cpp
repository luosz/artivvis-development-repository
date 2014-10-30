#include "VolumeRenderer.h"

void VolumeRenderer::Init(int screenWidth, int screenHeight)
{
	glEnable(GL_DEPTH_TEST);

	camera.Init(screenWidth, screenHeight);
	shaderManager.Init();
	volume.Init();

	renderer = new OpenGLRenderer();

	renderer->Init(screenWidth, screenHeight, volume);
}


void VolumeRenderer::Update()
{
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	

	camera.Update();
	volume.Update();

	renderer->Draw(volume, shaderManager, camera);


	glutSwapBuffers();
}