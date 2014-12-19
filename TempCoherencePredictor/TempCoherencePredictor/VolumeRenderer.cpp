#include "VolumeRenderer.h"

void VolumeRenderer::Init(int screenWidth, int screenHeight)
{
	glEnable(GL_DEPTH_TEST);

	camera.Init(screenWidth, screenHeight);
	shaderManager.Init();
	volume.Init();

	raycaster = new BlockRaycaster(screenWidth, screenHeight, volume);

	transferFunction.Init(" ", volume);
}


void VolumeRenderer::Update()
{
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	

	camera.Update();
	volume.Update();
	raycaster->TemporalCoherence(volume);

	GLuint shaderProgramID = shaderManager.UseShader(TFShader);
	raycaster->Raycast(volume, transferFunction, shaderProgramID, camera);

	glutSwapBuffers();
}

