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

	tester.Init(screenWidth, screenHeight);
	errorEvaluator.Init(screenWidth, screenHeight);

	targetFileName = "Error Metrics - SimilarFunc = maxDiff, Epsilon = 10, Camera = front.txt";
	std::remove(targetFileName.c_str());

	ofstream outStream(targetFileName);
	if (outStream.is_open())
	{
		outStream << "Time \t\tCopy \tExtrap \tMSE \t\t\tMAE \t\t\tPSN" << std::endl;
		outStream.close();
	}


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

			interpTex3D = tempCoherence->TemporalCoherence(volume, currentTimestep);
			bruteTex3D = bruteForce->BruteForceCopy(volume, currentTimestep);

			tester.Test(transferFunction, shaderManager, camera, *raycaster, bruteTex3D, interpTex3D);
			WriteToFile();
		}

		
	}

	GLuint shaderProgramID = shaderManager.UseShader(TFShader);
	raycaster->Raycast(transferFunction, shaderProgramID, camera, interpTex3D);

	

	glutSwapBuffers();
}


void VolumeRenderer::WriteToFile()
{
	ofstream outStream(targetFileName, std::ios::app);

	if (outStream.is_open())
	{
		outStream << currentTimestep << "\t\t" << tempCoherence->numBlocksCopied << "\t" << tempCoherence->numBlocksExtrapolated << std::fixed << std::setprecision(6) << "\t\t" << tester.meanSqrError << "\t\t" << tester.meanAvgErr << "\t\t" << tester.peakSigToNoise << std::endl;
		outStream.close();
	}
	if (currentTimestep == 598)
		getchar();
}