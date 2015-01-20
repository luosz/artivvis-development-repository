#include "VolumeRenderer.h"

void VolumeRenderer::Init(int screenWidth, int screenHeight)
{
	glEnable(GL_DEPTH_TEST);

	camera.Init(screenWidth, screenHeight);
	shaderManager.Init();
	volume.Init();


	tempCoherence = new TempCoherence(volume);
	bruteForce = new BruteForce(volume);

	raycaster = new Raycaster(screenWidth, screenHeight, volume);

	transferFunction.Init(" ", volume);

	oldTime = clock();
}


void VolumeRenderer::Update()
{
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	

	camera.Update();
	
	if (volume.timesteps > 1)
	{
		clock_t currentTime = clock();
		float time = (currentTime - oldTime) / (float) CLOCKS_PER_SEC;

		if (time > volume.timePerFrame)
		{
			if (currentTimestep < volume.timesteps - 2)
				currentTimestep++;
			else
				currentTimestep = 0;

			oldTime = currentTime;

//			tex3D = tempCoherence->TemporalCoherence(volume, currentTimestep);
			tex3D = bruteForce->BruteForceCopy(volume, currentTimestep);
		}
	}

	GLuint shaderProgramID = shaderManager.UseShader(TFShader);
	raycaster->Raycast(volume, transferFunction, shaderProgramID, camera, tex3D);

	glutSwapBuffers();
}

