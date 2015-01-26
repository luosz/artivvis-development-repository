#include "JoesOGLRenderer.h"
#include "../TransferFunctionEditor/import_transfer_function_editor.h"

JoesOGLRenderer::JoesOGLRenderer(int screenWidth, int screenHeight, VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera) : OpenGLRenderer(screenWidth, screenHeight, volume, shaderManager, camera)
{
	//visibilityTFOptimizer = std::make_shared<VisibilityTFOptimizer>(&volume, &visibilityHistogram, &transferFunction);
	//intensityOptimizerV2 = std::make_shared<IntensityTFOptimizerV2>(&volume, &transferFunction, &visibilityHistogram);
}


void JoesOGLRenderer::Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera)
{
	visibilityHistogram.CalculateHistogram(volume, transferFunction.tfTexture, shaderManager, camera);

	// if transfer function view is not NULL
	if (transferFunction.tfView)
	{
		// Ma's optimization
		if (transferFunction.tfView->isMaOptimizerEnable())
		{
			optimizer()->Optimize();
		}

		// Joe's intensity and visibility optimization
		if (transferFunction.tfView->isLuoOptimizerEnable())
		{
			transferFunction.tfView->updateTransferFunctionFromView();
			//auto p = dynamic_cast<TransferFunctionView *>(transferFunction.tfView);
			//if (p)
			//{
			//	auto optimizer2 = p->optimizer();
			//	optimizer2->BalanceVisibilityOnce();
			//}
			intensityTFOptimizerV2()->BalanceVisibilityOnce();
			transferFunction.tfView->updateViewFromTransferFunction();
		}
	}

	if (transferFunction.visibilityView)
	{
		transferFunction.visibilityView->setVisibilityHistogram(visibilityHistogram.visibilities, visibilityHistogram.numVis);
		transferFunction.visibilityView->draw();
	}

	OpenGLRenderer::Draw(volume, shaderManager, camera);
}
