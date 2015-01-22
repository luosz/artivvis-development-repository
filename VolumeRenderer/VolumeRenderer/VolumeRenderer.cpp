#include "VolumeRenderer.h"

void VolumeRenderer::Init(int screenWidth, int screenHeight)
{
	glEnable(GL_DEPTH_TEST);

	camera.Init(screenWidth, screenHeight);
	shaderManager.Init();
	volume.Init();

#ifndef USE_JoesOGLRenderer
	std::cout << "Tom's OpenGL renderer is used." << std::endl;
	renderer = new TomsOGLRenderer(screenWidth, screenHeight, volume, shaderManager, camera);
#else
	std::cout << "Joe's OpenGL renderer is used." << std::endl;
	renderer = new JoesOGLRenderer(screenWidth, screenHeight, volume, shaderManager, camera);
#endif

	grabRegion = false;
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



void VolumeRenderer::OptimizeForSelectedRegion(int mouseX, int mouseY, int screenWidth, int screenHeight)
{
	float avgIntensity = regionGrabber.Grab(mouseX, mouseY, screenWidth, screenHeight, camera, renderer->raycaster->clipPlaneNormal, renderer->raycaster->clipPlaneDistance, volume);

	if (avgIntensity == -1.0f)
		return;

	renderer->transferFunction.targetIntensity = avgIntensity;
}