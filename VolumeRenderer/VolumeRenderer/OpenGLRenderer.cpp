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

	transferFunction.intensityOptimizerV2->SetVisibilityHistogram(visibilityHistogram);

//	regionOptimizer.Init(transferFunction);
}


void OpenGLRenderer::Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera)
{

	visibilityHistogram.CalculateHistogram(volume, transferFunction.tfTexture, shaderManager, camera);

	transferFunction.Update();

	if (!transferFunction.tfView || transferFunction.tfView->isMaOptimizerEnable())
	{
		visibilityOptimizer.Optimize(volume, visibilityHistogram, transferFunction, shaderManager, camera);
	}

//	regionOptimizer.CalculateVisibility(shaderManager, camera, volume, transferFunction, raycaster);

	// intensity and visibility optimization
	if (transferFunction.tfView && transferFunction.tfView->isLuoOptimizerEnable())
	{
		transferFunction.tfView->updateTransferFunctionFromView();
		transferFunction.intensityOptimizerV2->BalanceVisibilityOnce();
		transferFunction.LoadLookup(transferFunction.currentColorTable);
		transferFunction.tfView->updateViewFromTransferFunction();
	}

	if (transferFunction.visibilityView)
	{
		transferFunction.visibilityView->setVisibilityHistogram(visibilityHistogram.visibilities, visibilityHistogram.numVis);
		transferFunction.visibilityView->draw();
	}

	GLuint shaderProgramID = shaderManager.UseShader(shaderManager.currentShader);
	raycaster->Raycast(volume, transferFunction, shaderProgramID, camera);

	visibilityOptimizer.DrawEnergy(shaderManager, camera);
	visibilityHistogram.DrawHistogram(shaderManager, camera);
//	regionOptimizer.DrawHistogram(shaderManager, camera);
}

