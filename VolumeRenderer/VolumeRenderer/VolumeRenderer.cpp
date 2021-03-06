#include "VolumeRenderer.h"

void VolumeRenderer::Init(int screenWidth, int screenHeight)
{
	glEnable(GL_DEPTH_TEST);

	camera.Init(screenWidth, screenHeight);
	shaderManager.Init();
	volume.Init();

	renderer = new TomsOGLRenderer(screenWidth, screenHeight, volume, shaderManager, camera);
//	renderer = new JoesOGLRenderer(screenWidth, screenHeight, volume, shaderManager, camera);


	grabRegion = false;
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



void VolumeRenderer::OptimizeForSelectedRegion(int mouseX, int mouseY, int screenWidth, int screenHeight)
{
	float avgIntensity = regionGrabber.Grab(mouseX, mouseY, screenWidth, screenHeight, camera, renderer->raycaster->clipPlaneNormal, renderer->raycaster->clipPlaneDistance, volume);

	std::cout << "avgIntensity=" << avgIntensity << std::endl;

	if (avgIntensity == -1.0f)
		return;

	renderer->transferFunction.targetIntensity = avgIntensity;
}