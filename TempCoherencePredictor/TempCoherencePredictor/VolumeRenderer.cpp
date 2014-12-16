#include "VolumeRenderer.h"

void VolumeRenderer::Init(int screenWidth, int screenHeight)
{
//	glEnable(GL_DEPTH_TEST);

	camera.Init(screenWidth, screenHeight);
	shaderManager.Init();
	volume.Init();

	volume.currTexture3D = volume.GenerateTexture();

	if (volume.timesteps > 1)
	{
		volume.voxelReader.CopyFileToBuffer(volume.memblock3D, 1);
		volume.nextTexture3D = volume.GenerateTexture();
	}

	raycaster = new BlockRaycaster(screenWidth, screenHeight, volume);

	transferFunction.Init(" ", volume);
}


void VolumeRenderer::Update()
{
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	

	camera.Update();
	volume.Update();

	GLuint shaderProgramID = shaderManager.UseShader(BlockShader);
	raycaster->Raycast(volume, transferFunction, shaderProgramID, camera);

	glutSwapBuffers();
}

