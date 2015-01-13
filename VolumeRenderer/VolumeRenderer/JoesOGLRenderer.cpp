#include "JoesOGLRenderer.h"

JoesOGLRenderer::JoesOGLRenderer(int screenWidth, int screenHeight, VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera)   :   OpenGLRenderer(screenWidth, screenHeight, volume, shaderManager, camera)
{

}


void JoesOGLRenderer::Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera)
{
	visibilityHistogram.CalculateHistogram(volume, transferFunction.tfTexture, shaderManager, camera);

	/*
	if (!transferFunction.tfView || transferFunction.tfView->isMaOptimizerEnable())
	{
//		visibilityTFOptimizer.Optimize();		
	}


// intensity and visibility optimization
	if (transferFunction.tfView && transferFunction.tfView->isLuoOptimizerEnable())
	{
		transferFunction.tfView->updateTransferFunctionFromView();
		transferFunction.intensityOptimizerV2->BalanceVisibilityOnce();
		transferFunction.LoadLookup(transferFunction.currentColorTable);
		transferFunction.tfView->updateViewFromTransferFunction();
	}
*/

	if (transferFunction.visibilityView)
	{
		transferFunction.visibilityView->setVisibilityHistogram(visibilityHistogram.visibilities, visibilityHistogram.numVis);
		transferFunction.visibilityView->draw();
	}

	OpenGLRenderer::Draw(volume, shaderManager, camera);
}