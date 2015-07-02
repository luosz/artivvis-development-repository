#include "VolumeRenderer.h"

void VolumeRenderer::Init(int screenWidth, int screenHeight)
{
	glEnable(GL_DEPTH_TEST);

	camera.Init(screenWidth, screenHeight);
	shaderManager.Init();
	volume.Init();

	renderer = new OpenGLRenderer(screenWidth, screenHeight, volume, shaderManager, camera);
}


void VolumeRenderer::Update()
{
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	

	camera.Update();
	volume.Update();

	renderer->Draw(volume, shaderManager, camera);

	glutSwapBuffers();
}

