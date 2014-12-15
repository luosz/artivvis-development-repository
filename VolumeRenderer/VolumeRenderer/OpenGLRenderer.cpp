#include "OpenGLRenderer.h"

OpenGLRenderer::OpenGLRenderer(int screenWidth, int screenHeight, VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera)
{
	volume.currTexture3D = volume.GenerateTexture();

	if (volume.timesteps > 1)
	{
		volume.voxelReader.CopyFileToBuffer(volume.memblock3D, 1);
		volume.nextTexture3D = volume.GenerateTexture();
	}

	raycaster = new GPURaycaster(screenWidth, screenHeight, volume);
	visibilityHistogram.Init(screenWidth, screenHeight);
	transferFunction.Init(" ", volume);

	optimizer = new VisibilityTFOptimizer(&volume, &visibilityHistogram, &transferFunction);
//	optimizer = new RegionVisibilityOptimizer(&volume, &transferFunction, raycaster, &shaderManager, &camera);
//	optimizer = new IntensityTFOptimizer(&volume, &transferFunction);
//	optimizer = new IntensityTFOptimizerV2(&volume, &transferFunction, &visibilityHistogram);
}


void OpenGLRenderer::Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera)
{
	GLuint shaderProgramID = shaderManager.UseShader(shaderManager.currentShader);
	raycaster->Raycast(volume, transferFunction, shaderProgramID, camera);

	if (optimizer)
		optimizer->Draw(shaderManager, camera);

	visibilityHistogram.DrawHistogram(shaderManager, camera);
}

