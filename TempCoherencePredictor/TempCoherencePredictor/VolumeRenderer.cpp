#include "VolumeRenderer.h"

void VolumeRenderer::Init(int screenWidth, int screenHeight)
{
	glEnable(GL_DEPTH_TEST);

	camera.Init(screenWidth, screenHeight);
	shaderManager.Init();
	volume.Init();


	tempCoherence = new TempCoherence(screenWidth, screenHeight, volume);
	bruteForce = new BruteForce(volume);

	raycaster = new Raycaster(screenWidth, screenHeight, volume);

	transferFunction.Init(" ", volume);

	tester.Init(screenWidth, screenHeight, volume);


	writeToFile = true;
	if (writeToFile)
		fileWriter.Init();


	oldTime = clock();
}


void VolumeRenderer::Update()
{
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
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

			interpTex3D = tempCoherence->TemporalCoherence(volume, currentTimestep, transferFunction, shaderManager, camera);
			bruteTex3D = bruteForce->BruteForceCopy(volume, currentTimestep);

			tester.Test(volume, transferFunction, shaderManager, camera, *raycaster, bruteTex3D, interpTex3D, currentTimestep);
			
			if (writeToFile)
				fileWriter.Write(currentTimestep, *tempCoherence, tester);
		}

		
	}

	GLuint shaderProgramID = shaderManager.UseShader(TFShader);
	raycaster->Raycast(transferFunction, shaderProgramID, camera, interpTex3D);

	glutSwapBuffers();
}


