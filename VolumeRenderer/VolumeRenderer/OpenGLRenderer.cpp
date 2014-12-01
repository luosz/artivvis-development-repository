#include "OpenGLRenderer.h"

void OpenGLRenderer::Init(int screenWidth, int screenHeight, VolumeDataset &volume)
{
	volume.currTexture3D = volume.GenerateTexture();

	if (volume.timesteps > 1)
	{
		volume.voxelReader.CopyFileToBuffer(volume.memblock3D, 1);
		volume.nextTexture3D = volume.GenerateTexture();
	}

	raycaster = new GPURaycaster();
	raycaster->Init(screenWidth, screenHeight, volume);

	transferFunction.Init(" ", volume);

	visibilityHistogram.Init(screenWidth, screenHeight);

	visibilityOptimizer.Init();

//	regionOptimizer.Init(transferFunction);
}


void OpenGLRenderer::Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera)
{

	visibilityHistogram.CalculateHistogram(volume, transferFunction.tfTexture, shaderManager, camera);

	transferFunction.Update();

	visibilityOptimizer.Optimize(volume, visibilityHistogram, transferFunction);

	

//	regionOptimizer.CalculateVisibility(shaderManager, camera, volume, transferFunction, raycaster);


	GLuint shaderProgramID = shaderManager.UseShader(shaderManager.currentShader);
	raycaster->Raycast(volume, transferFunction, shaderProgramID, camera);

	visibilityOptimizer.DrawEnergy(shaderManager, camera);
	visibilityHistogram.DrawHistogram(shaderManager, camera);
//	regionOptimizer.DrawHistogram(shaderManager, camera);
}

